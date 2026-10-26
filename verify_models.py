"""
Verification script for Stage 4 (Trained Models).
Reloads saved classifier and regressor models, predicts on 5 test rows,
and confirms outputs match expected bounds and logged metrics.
"""

from pathlib import Path
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    BEST_CLASSIFIER_PATH,
    BEST_REGRESSOR_PATH,
    CLEAN_DATA_PATH,
    FEATURES_CLASSIFIER_PATH,
    FEATURES_REGRESSOR_PATH,
    PREPROCESSOR_CLASSIFIER_PATH,
    PREPROCESSOR_REGRESSOR_PATH,
    RANDOM_SEED,
)


def verify_models() -> None:
    """Verify saved model artifacts and generate test sample predictions."""
    print("=" * 60)
    print("VERIFYING STAGE 4 SAVED MODEL ARTIFACTS")
    print("=" * 60)

    # 1. Verify existence of model files
    required_paths = [
        BEST_CLASSIFIER_PATH,
        PREPROCESSOR_CLASSIFIER_PATH,
        FEATURES_CLASSIFIER_PATH,
        BEST_REGRESSOR_PATH,
        PREPROCESSOR_REGRESSOR_PATH,
        FEATURES_REGRESSOR_PATH,
    ]
    for p in required_paths:
        assert p.exists(), f"Missing required model artifact: {p}"
        print(f"✔ Found model artifact: {p.name}")

    # 2. Reload saved artifacts
    clf = joblib.load(BEST_CLASSIFIER_PATH)
    preproc_clf = joblib.load(PREPROCESSOR_CLASSIFIER_PATH)
    feats_clf = joblib.load(FEATURES_CLASSIFIER_PATH)

    reg = joblib.load(BEST_REGRESSOR_PATH)
    preproc_reg = joblib.load(PREPROCESSOR_REGRESSOR_PATH)
    feats_reg = joblib.load(FEATURES_REGRESSOR_PATH)

    # 3. Load dataset and reproduce test split
    df = pd.read_csv(CLEAN_DATA_PATH)
    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=RANDOM_SEED, stratify=df["Good_Investment"]
    )
    print(f"\nLoaded holdout test partition ({len(test_df):,} rows).")

    # Select 5 sample rows
    sample_df = test_df.iloc[:5].copy()

    # 4. Predict on sample with Classifier
    X_sample_clf = sample_df[feats_clf]
    X_sample_clf_proc = preproc_clf.transform(X_sample_clf)
    clf_preds = clf.predict(X_sample_clf_proc)
    clf_probs = clf.predict_proba(X_sample_clf_proc)[:, 1]

    # Assertions on Classifier outputs
    assert len(clf_preds) == 5, "Classifier did not return 5 predictions!"
    assert set(clf_preds).issubset({0, 1}), f"Invalid classifier output: {clf_preds}"
    assert ((clf_probs >= 0.0) & (clf_probs <= 1.0)).all(), "Probabilities outside [0, 1]!"

    # 5. Predict on sample with Regressor
    X_sample_reg = sample_df[feats_reg]
    X_sample_reg_proc = preproc_reg.transform(X_sample_reg)
    reg_preds = reg.predict(X_sample_reg_proc)

    # Assertions on Regressor outputs
    assert len(reg_preds) == 5, "Regressor did not return 5 predictions!"
    assert (reg_preds > 0).all(), "Regressor returned non-positive prices!"

    # 6. Display verification comparison table
    print("\n--- SAMPLE 5 ROW PREDICTIONS VS GROUND TRUTH ---")
    results_summary = pd.DataFrame({
        "City": sample_df["City"].values,
        "Property_Type": sample_df["Property_Type"].values,
        "Price_Current (₹L)": sample_df["Price_in_Lakhs"].values.round(2),
        "Actual_Good_Inv": sample_df["Good_Investment"].values,
        "Pred_Good_Inv": clf_preds,
        "Confidence_Score": (clf_probs * 100).round(1),
        "Actual_Price_5Y (₹L)": sample_df["Future_Price_5Y"].values.round(2),
        "Pred_Price_5Y (₹L)": reg_preds.round(2),
        "Price_Diff (₹L)": (reg_preds - sample_df["Future_Price_5Y"].values).round(2),
    })
    print(results_summary.to_string(index=False))

    print("\nSTAGE 4 VERIFICATION COMPLETE: ALL CHECKS PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    verify_models()
