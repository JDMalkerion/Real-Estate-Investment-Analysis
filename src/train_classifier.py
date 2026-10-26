"""
Model training pipeline for Investment Classifier.
Trains Logistic Regression, Random Forest, and XGBoost with MLflow tracking.
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from xgboost import XGBClassifier

from src.config import (
    BEST_CLASSIFIER_PATH,
    CLASSIFIER_FEATURE_COLUMNS,
    CLEAN_DATA_PATH,
    EXPERIMENT_CLASSIFIER,
    FEATURES_CLASSIFIER_PATH,
    MLRUNS_DIR,
    MODELS_DIR,
    PREPROCESSOR_CLASSIFIER_PATH,
    RANDOM_SEED,
)
from src.evaluate import (
    calculate_classification_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
)


def build_preprocessor() -> Tuple[ColumnTransformer, List[str], List[str], List[str]]:
    """Build ColumnTransformer for classification features."""
    low_card_cats = [
        "State", "City", "Property_Type", "Parking_Space",
        "Security", "Facing", "Owner_Type", "Availability_Status"
    ]
    high_card_cats = ["Locality"]
    num_cols = [
        "BHK", "Size_in_SqFt", "Price_in_Lakhs",
        "Floor_No", "Total_Floors", "Age_of_Property",
        "Nearby_Schools", "Nearby_Hospitals", "amenity_count", "floor_ratio"
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


def train_classifiers() -> Dict[str, Dict[str, float]]:
    """Train all classification models and track with MLflow."""
    print("=" * 60)
    print("STAGE 4: TRAINING INVESTMENT CLASSIFIERS")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLRUNS_DIR)
    mlflow.set_experiment(EXPERIMENT_CLASSIFIER)

    print(f"Loading data from {CLEAN_DATA_PATH}...")
    df = pd.read_csv(CLEAN_DATA_PATH)

    # Strictly isolate features to prevent data leakage
    X = df[CLASSIFIER_FEATURE_COLUMNS].copy()
    y = df["Good_Investment"].values

    print(f"Features: {len(CLASSIFIER_FEATURE_COLUMNS)} columns | Target: Good_Investment")
    print(f"Train/Test split (80/20, stratified, seed={RANDOM_SEED})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    print("Fitting preprocessing pipeline on training data only...")
    preprocessor, low_cats, high_cats, num_cols = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train, y_train)
    X_test_proc = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor, low_cats, high_cats, num_cols)
    print(f"Transformed feature dimension: {X_train_proc.shape[1]}")

    models = {
        "Logistic_Regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
            {"max_iter": 1000, "solver": "lbfgs", "random_state": RANDOM_SEED},
        ),
        "Random_Forest": (
            RandomForestClassifier(
                n_estimators=50, max_depth=12, n_jobs=-1, random_state=RANDOM_SEED
            ),
            {"n_estimators": 50, "max_depth": 12, "random_state": RANDOM_SEED},
        ),
        "XGBoost": (
            XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                n_jobs=-1,
                random_state=RANDOM_SEED,
                eval_metric="logloss",
            ),
            {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": RANDOM_SEED,
            },
        ),
    }

    all_metrics: Dict[str, Dict[str, float]] = {}
    best_model_name: str = ""
    best_f1: float = -1.0
    best_model = None

    scratch_dir = Path("scratch/classifier_artifacts")
    scratch_dir.mkdir(parents=True, exist_ok=True)

    for name, (model, params) in models.items():
        print(f"\n--- Training {name} ---")
        with mlflow.start_run(run_name=name):
            mlflow.log_params(params)
            model.fit(X_train_proc, y_train)

            y_pred = model.predict(X_test_proc)
            y_prob = model.predict_proba(X_test_proc)[:, 1] if hasattr(model, "predict_proba") else None

            metrics = calculate_classification_metrics(y_test, y_pred, y_prob)
            all_metrics[name] = metrics
            mlflow.log_metrics(metrics)

            print(
                f"{name} Metrics -> Acc: {metrics['accuracy']:.4f}, "
                f"Prec: {metrics['precision']:.4f}, Rec: {metrics['recall']:.4f}, "
                f"F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}"
            )

            # Leakage safeguard alert
            if any(val > 0.98 for val in metrics.values()):
                print(f"⚠️ SUSPECTED LEAKAGE WARNING: {name} scored > 0.98 on a held-out test metric!")

            # Generate and log Confusion Matrix
            cm_path = scratch_dir / f"{name}_confusion_matrix.png"
            plot_confusion_matrix(
                y_test,
                y_pred,
                title=f"{name} Confusion Matrix (F1: {metrics['f1']:.3f})",
                output_path=cm_path,
            )
            mlflow.log_artifact(str(cm_path), artifact_path="plots")

            # Generate and log Feature Importance
            if hasattr(model, "feature_importances_"):
                fi = model.feature_importances_
            elif hasattr(model, "coef_"):
                fi = np.abs(model.coef_[0])
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
            if isinstance(model, XGBClassifier):
                mlflow.xgboost.log_model(model, name="model")
            else:
                mlflow.sklearn.log_model(model, name="model", serialization_format="pickle")

            # Check for best model by F1
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_model_name = name
                best_model = model

    print(f"\n✔ Best Classifier: {best_model_name} with F1 = {best_f1:.4f}")

    # Save artifacts for deployment & Streamlit app
    joblib.dump(best_model, BEST_CLASSIFIER_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_CLASSIFIER_PATH)
    joblib.dump(CLASSIFIER_FEATURE_COLUMNS, FEATURES_CLASSIFIER_PATH)
    print(f"Saved best classifier and preprocessor to {MODELS_DIR}")

    return all_metrics


def main() -> None:
    """Execute classifier training pipeline."""
    train_classifiers()


if __name__ == "__main__":
    main()
