"""
Verification script for Stage 3 (Target Generation).
Asserts:
- Future_Price_5Y >= Price_in_Lakhs for all rows
- Good_Investment label is strictly binary (0 or 1)
- Class balance is within valid boundaries (no more extreme than 80/20)
- Prints class balance and growth rate distributions
"""

import sys
import pandas as pd

from src.config import CLEAN_DATA_PATH


def verify_targets() -> None:
    """Run verification checks on generated targets."""
    print("=" * 60)
    print("VERIFYING STAGE 3 TARGETS OUTPUT")
    print("=" * 60)

    if not CLEAN_DATA_PATH.exists():
        print(f"FAILED: Data file not found at {CLEAN_DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(CLEAN_DATA_PATH)
    print(f"Loaded {len(df):,} rows from {CLEAN_DATA_PATH}")

    # Check 1: Future_Price_5Y >= Price_in_Lakhs for all rows
    assert "Future_Price_5Y" in df.columns, "Future_Price_5Y column missing!"
    assert (df["Future_Price_5Y"] >= df["Price_in_Lakhs"]).all(), (
        "Found rows where Future_Price_5Y < Price_in_Lakhs!"
    )
    print(f"✔ Check 1 Passed: Future_Price_5Y >= Price_in_Lakhs for all {len(df):,} rows.")

    # Check 2: Good_Investment is binary
    assert "Good_Investment" in df.columns, "Good_Investment column missing!"
    unique_labels = set(df["Good_Investment"].unique())
    assert unique_labels == {0, 1}, f"Unexpected labels in Good_Investment: {unique_labels}"
    print(f"✔ Check 2 Passed: Good_Investment is strictly binary {unique_labels}.")

    # Check 3: Class balance boundaries (not more extreme than 80/20)
    counts = df["Good_Investment"].value_counts().to_dict()
    proportions = df["Good_Investment"].value_counts(normalize=True).to_dict()
    print("\nClass Balance:")
    print(f"- Class 0 (Not Recommended): {counts[0]:,} ({proportions[0]:.2%})")
    print(f"- Class 1 (Good Investment): {counts[1]:,} ({proportions[1]:.2%})")

    assert proportions[1] >= 0.20 and proportions[1] <= 0.80, (
        f"Class balance {proportions[1]:.2%} is more extreme than 80/20!"
    )
    print("✔ Check 3 Passed: Class balance is healthy and within 80/20 limits.")

    # Growth rate stats
    growth = df["estimated_growth_rate"]
    print(f"\nGrowth Rate Distribution: min={growth.min():.2%}, median={growth.median():.2%}, max={growth.max():.2%}")
    print("\nSTAGE 3 VERIFICATION COMPLETE: ALL CHECKS PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    verify_targets()
