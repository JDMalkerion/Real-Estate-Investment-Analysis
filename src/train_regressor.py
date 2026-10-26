"""
Model training pipeline for 5-Year Price Regressor.
Trains Baseline, Ridge, Random Forest, and XGBoost with MLflow tracking.
"""

import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple

# Enable local file store for MLflow
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from xgboost import XGBRegressor

from src.config import (
    BEST_REGRESSOR_PATH,
    CLEAN_DATA_PATH,
    EXPERIMENT_REGRESSOR,
    FEATURES_REGRESSOR_PATH,
    MLRUNS_DIR,
    MODELS_DIR,
    PREPROCESSOR_REGRESSOR_PATH,
    PROJECTION_YEARS,
    RANDOM_SEED,
    REGRESSOR_FEATURE_COLUMNS,
)
from src.evaluate import (
    calculate_regression_metrics,
    plot_feature_importance,
    plot_residuals,
)


def build_regressor_preprocessor() -> Tuple[ColumnTransformer, List[str], List[str], List[str]]:
    """Build ColumnTransformer for regression features."""
    low_card_cats = [
        "State", "City", "Property_Type", "Furnished_Status",
        "Public_Transport_Accessibility", "Parking_Space", "Security",
        "Facing", "Owner_Type", "Availability_Status", "age_bucket"
    ]
    high_card_cats = ["Locality"]
    num_cols = [
        "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Year_Built",
        "Floor_No", "Total_Floors", "Age_of_Property", "Nearby_Schools",
        "Nearby_Hospitals", "infrastructure_score", "amenity_count",
        "security_feature_count", "floor_ratio", "school_density_score"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "low_card",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                low_card_cats,
            ),
            (
                "high_card",
                TargetEncoder(
                    cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
                    smooth="auto",
                ),
                high_card_cats,
            ),
            (
                "num",
                StandardScaler(),
                num_cols,
            ),
        ],
        remainder="drop",
    )
    return preprocessor, low_card_cats, high_card_cats, num_cols


def get_feature_names(
    preprocessor: ColumnTransformer,
    low_card_cats: List[str],
    high_card_cats: List[str],
    num_cols: List[str]
) -> List[str]:
    """Extract transformed feature names for importance plotting."""
    feature_names: List[str] = []
    ohe = preprocessor.named_transformers_["low_card"]
    feature_names.extend(ohe.get_feature_names_out(low_card_cats).tolist())
    feature_names.extend(high_card_cats)
    feature_names.extend(num_cols)
    return feature_names


def train_regressors() -> Dict[str, Dict[str, float]]:
    """Train all regression models and track with MLflow."""
    print("=" * 60)
    print("STAGE 4: TRAINING 5-YEAR PRICE REGRESSORS")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLRUNS_DIR)
    mlflow.set_experiment(EXPERIMENT_REGRESSOR)

    print(f"Loading data from {CLEAN_DATA_PATH}...")
    df = pd.read_csv(CLEAN_DATA_PATH)

    # Strictly isolate features to prevent data leakage
    X = df[REGRESSOR_FEATURE_COLUMNS].copy()
    y = df["Future_Price_5Y"].values
    growth_rates = df["estimated_growth_rate"].values

    print(f"Features: {len(REGRESSOR_FEATURE_COLUMNS)} columns | Target: Future_Price_5Y")
    print(f"Train/Test split (80/20, seed={RANDOM_SEED})...")
    (
        X_train,
        X_test,
        y_train,
        y_test,
        growth_train,
        growth_test,
    ) = train_test_split(
        X, y, growth_rates, test_size=0.20, random_state=RANDOM_SEED
    )

    # Compute Naive Average Growth Baseline on Train Only
    mean_train_growth = float(np.mean(growth_train))
    baseline_factor = (1.0 + mean_train_growth) ** PROJECTION_YEARS
    y_pred_baseline = X_test["Price_in_Lakhs"].values * baseline_factor

    baseline_metrics = calculate_regression_metrics(y_test, y_pred_baseline)
    print(
        f"\n[Baseline Model: Price_in_Lakhs * (1 + {mean_train_growth:.3%})^5]:"
        f"\n  RMSE: {baseline_metrics['rmse']:.3f} Lakhs | MAE: {baseline_metrics['mae']:.3f} Lakhs | R2: {baseline_metrics['r2']:.4f}"
    )

    all_metrics: Dict[str, Dict[str, float]] = {"Baseline_Average_Growth": baseline_metrics}

    scratch_dir = Path("scratch/regressor_artifacts")
    scratch_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="Baseline_Average_Growth"):
        mlflow.log_params({
            "model_type": "Heuristic Baseline",
            "mean_train_growth": mean_train_growth,
            "projection_years": PROJECTION_YEARS
        })
        mlflow.log_metrics(baseline_metrics)
        plot_residuals(
            y_test,
            y_pred_baseline,
            title="Baseline (Average Growth) Residuals",
            output_path=scratch_dir / "Baseline_residuals.png"
        )
        mlflow.log_artifact(str(scratch_dir / "Baseline_residuals.png"), artifact_path="plots")

    print("\nFitting preprocessing pipeline on training data only...")
    preprocessor, low_cats, high_cats, num_cols = build_regressor_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train, y_train)
    X_test_proc = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor, low_cats, high_cats, num_cols)
    print(f"Transformed feature dimension: {X_train_proc.shape[1]}")

    models = {
        "Ridge_Regression": (
            Ridge(alpha=1.0, random_state=RANDOM_SEED),
            {"alpha": 1.0, "random_state": RANDOM_SEED},
        ),
        "Random_Forest_Regressor": (
            RandomForestRegressor(
                n_estimators=50, max_depth=12, n_jobs=-1, random_state=RANDOM_SEED
            ),
            {"n_estimators": 50, "max_depth": 12, "random_state": RANDOM_SEED},
        ),
        "XGBoost_Regressor": (
            XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                n_jobs=-1,
                random_state=RANDOM_SEED,
            ),
            {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": RANDOM_SEED,
            },
        ),
    }

    best_model_name: str = ""
    best_rmse: float = float("inf")
    best_model = None

    for name, (model, params) in models.items():
        print(f"\n--- Training {name} ---")
        with mlflow.start_run(run_name=name):
            mlflow.log_params(params)
            model.fit(X_train_proc, y_train)

            y_pred = model.predict(X_test_proc)
            metrics = calculate_regression_metrics(y_test, y_pred)
            all_metrics[name] = metrics
            mlflow.log_metrics(metrics)

            print(
                f"{name} Metrics -> RMSE: {metrics['rmse']:.3f} Lakhs, "
                f"MAE: {metrics['mae']:.3f} Lakhs, R2: {metrics['r2']:.4f}"
            )

            # Generate and log Residual Plot
            res_path = scratch_dir / f"{name}_residuals.png"
            plot_residuals(
                y_test,
                y_pred,
                title=f"{name} Residuals (RMSE: {metrics['rmse']:.2f}L)",
                output_path=res_path,
            )
            mlflow.log_artifact(str(res_path), artifact_path="plots")

            # Generate and log Feature Importance
            if hasattr(model, "feature_importances_"):
                fi = model.feature_importances_
            elif hasattr(model, "coef_"):
                fi = np.abs(model.coef_)
            else:
                fi = np.ones(len(feature_names))

            fi_path = scratch_dir / f"{name}_feature_importance.png"
            plot_feature_importance(
                feature_names,
                fi,
                title=f"{name} Top Feature Importances",
                output_path=fi_path,
            )
            mlflow.log_artifact(str(fi_path), artifact_path="plots")

            # Log model
            if isinstance(model, XGBRegressor):
                mlflow.xgboost.log_model(model, name="model")
            else:
                mlflow.sklearn.log_model(model, name="model", serialization_format="pickle")

            # Check for best model by RMSE
            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_model_name = name
                best_model = model

    print(f"\n✔ Best Regressor: {best_model_name} with RMSE = {best_rmse:.3f} Lakhs")

    # Save artifacts for deployment & Streamlit app
    joblib.dump(best_model, BEST_REGRESSOR_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_REGRESSOR_PATH)
    joblib.dump(REGRESSOR_FEATURE_COLUMNS, FEATURES_REGRESSOR_PATH)
    print(f"Saved best regressor and preprocessor to {MODELS_DIR}")

    return all_metrics


def main() -> None:
    """Execute regressor training pipeline."""
    train_regressors()


if __name__ == "__main__":
    main()
