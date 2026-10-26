"""
Verification script for Stage 1 (Preprocessing and Feature Engineering).
Asserts:
- No nulls remain
- No duplicate rows or IDs
- Derived columns consistent (Floor_No <= Total_Floors, Age matches year built, Price matches Size * Price_per_SqFt)
"""

import sys
import numpy as np
import pandas as pd

from src.config import CLEAN_DATA_PATH, INR_PER_LAKH, REFERENCE_YEAR


def verify_preprocessing() -> None:
    """Run verification checks on the cleaned dataset."""
    print("=" * 60)
    print("VERIFYING STAGE 1 PREPROCESSING OUTPUT")
    print("=" * 60)

    if not CLEAN_DATA_PATH.exists():
        print(f"FAILED: Cleaned data file does not exist at {CLEAN_DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(CLEAN_DATA_PATH)
    print(f"Loaded {len(df):,} rows from {CLEAN_DATA_PATH}")

    # Check 1: No nulls remain
    null_count = int(df.isnull().sum().sum())
    assert null_count == 0, f"Found {null_count} null values in cleaned data!"
    print(f"✔ Check 1 Passed: 0 nulls remain across all columns.")

    # Check 2: No duplicates
    dup_rows = int(df.duplicated().sum())
    dup_ids = int(df["ID"].duplicated().sum())
    assert dup_rows == 0, f"Found {dup_rows} duplicate rows!"
    assert dup_ids == 0, f"Found {dup_ids} duplicate IDs!"
    print(f"✔ Check 2 Passed: 0 duplicate rows and 0 duplicate IDs.")

    # Check 3: Derived floor consistency
    inconsistent_floors = (df["Floor_No"] > df["Total_Floors"]).sum()
    assert inconsistent_floors == 0, f"Found {inconsistent_floors} rows with Floor_No > Total_Floors!"
    assert ((df["floor_ratio"] >= 0.0) & (df["floor_ratio"] <= 1.0)).all(), "Invalid floor_ratio values outside [0, 1]!"
    print(f"✔ Check 3 Passed: Floor_No <= Total_Floors for all {len(df):,} rows (floor_ratio in [0, 1]).")

    # Check 4: Age consistency
    expected_age = REFERENCE_YEAR - df["Year_Built"]
    age_mismatch = (df["Age_of_Property"] != expected_age).sum()
    assert age_mismatch == 0, f"Found {age_mismatch} age mismatches!"
    print(f"✔ Check 4 Passed: Age_of_Property strictly consistent with {REFERENCE_YEAR} - Year_Built.")

    # Check 5: Price per SqFt consistency with Price and Size
    expected_price = (df["Price_per_SqFt"] * df["Size_in_SqFt"]) / INR_PER_LAKH
    max_price_diff = np.abs(df["Price_in_Lakhs"] - expected_price).max()
    assert max_price_diff < 1e-3, f"Price inconsistency detected! Max diff = {max_price_diff}"
    print(f"✔ Check 5 Passed: Price_in_Lakhs matches (Price_per_SqFt * Size_in_SqFt / 100,000) (max diff: {max_price_diff:.6f}).")

    # Check 6: Infrastructure and engineered features
    assert ((df["infrastructure_score"] >= 0.0) & (df["infrastructure_score"] <= 1.0)).all(), "Infrastructure score outside [0, 1]!"
    assert ((df["amenity_count"] >= 0) & (df["amenity_count"] <= 5)).all(), "Amenity count out of range!"
    assert set(df["security_feature_count"].unique()).issubset({0, 1}), "Security feature count not binary!"
    print(f"✔ Check 6 Passed: Engineered features (infrastructure_score, amenity_count, security_feature_count) properly bounded.")

    print("\nSTAGE 1 VERIFICATION COMPLETE: ALL CHECKS PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    verify_preprocessing()
