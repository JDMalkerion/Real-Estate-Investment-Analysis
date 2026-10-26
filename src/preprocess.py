from pathlib import Path
import sys
from typing import Any, Dict, Tuple

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import (
    AGE_BINS,
    AGE_LABELS,
    CLEAN_DATA_PATH,
    INFRASTRUCTURE_WEIGHTS,
    INR_PER_LAKH,
    IQR_FACTOR,
    PUBLIC_TRANSPORT_MAP,
    RAW_DATA_PATH,
    REFERENCE_YEAR,
)


def audit_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Audit raw dataset and print data quality report.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset loaded from CSV.

    Returns
    -------
    Dict[str, Any]
        Audit summary metrics.
    """
    print("=" * 60)
    print("STAGE 1: RAW DATASET AUDIT REPORT")
    print("=" * 60)

    shape = df.shape
    print(f"Shape: {shape[0]:,} rows, {shape[1]} columns")

    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())
    print(f"Total nulls: {total_nulls}")

    duplicate_rows = int(df.duplicated().sum())
    duplicate_ids = int(df["ID"].duplicated().sum())
    print(f"Duplicate rows: {duplicate_rows}")
    print(f"Duplicate IDs: {duplicate_ids}")

    # Price per SqFt check
    # 1 Lakh = 100,000 INR
    # Recomputed price per sqft in Lakhs: Price_in_Lakhs / Size_in_SqFt
    # Recomputed price per sqft in INR: Price_in_Lakhs * 100,000 / Size_in_SqFt
    calc_price_per_sqft_lakhs = df["Price_in_Lakhs"] / df["Size_in_SqFt"]
    calc_price_per_sqft_inr = (df["Price_in_Lakhs"] * INR_PER_LAKH) / df["Size_in_SqFt"]
    diff_lakhs = np.abs(df["Price_per_SqFt"] - calc_price_per_sqft_lakhs)
    max_diff_lakhs = float(diff_lakhs.max())
    mean_diff_lakhs = float(diff_lakhs.mean())

    print(
        f"\n[Price_per_SqFt Unit & Consistency]:"
        f"\n- Raw Price_per_SqFt unit is Lakhs per SqFt (stored to 2 decimals)."
        f"\n- Max difference between raw column and calculated Lakhs/SqFt: {max_diff_lakhs:.6f}"
        f"\n- Mean rounding discrepancy: {mean_diff_lakhs:.6f} Lakhs/SqFt"
        f"\n- Standard unit adopted for clean dataset: INR per SqFt (Price_in_Lakhs * 100,000 / Size_in_SqFt)"
    )

    # Age of property check
    calc_age = REFERENCE_YEAR - df["Year_Built"]
    age_matches = bool((df["Age_of_Property"] == calc_age).all())
    print(
        f"\n[Age_of_Property Consistency]:"
        f"\n- Baseline reference year: {REFERENCE_YEAR}"
        f"\n- Age_of_Property == ({REFERENCE_YEAR} - Year_Built) for all rows: {age_matches}"
    )

    # Floor_No vs Total_Floors check
    floor_inconsistent_mask = df["Floor_No"] > df["Total_Floors"]
    floor_inconsistent_count = int(floor_inconsistent_mask.sum())
    print(
        f"\n[Floor Inconsistency]:"
        f"\n- Rows where Floor_No > Total_Floors: {floor_inconsistent_count:,} ({floor_inconsistent_count / shape[0]:.2%})"
    )

    # Category checks
    security_unique = sorted(df["Security"].dropna().unique().tolist())
    amenity_items = set()
    for item_str in df["Amenities"].dropna():
        for token in str(item_str).split(","):
            amenity_items.add(token.strip())

    print(f"\n[Categorical Integrity]:")
    print(f"- Security unique values: {security_unique}")
    print(f"- Amenities distinct parsed items: {sorted(list(amenity_items))}")

    audit_summary = {
        "shape": shape,
        "total_nulls": total_nulls,
        "duplicate_rows": duplicate_rows,
        "duplicate_ids": duplicate_ids,
        "floor_inconsistent_count": floor_inconsistent_count,
        "age_matches_reference_2025": age_matches,
        "security_unique": security_unique,
        "amenity_items": sorted(list(amenity_items)),
    }
    return audit_summary


def clean_and_engineer(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Clean dataset and perform feature engineering.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, int]]
        Cleaned dataframe and dictionary of logged counts.
    """
    logs: Dict[str, int] = {
        "dropped_duplicates": 0,
        "numeric_imputed": 0,
        "categorical_imputed": 0,
        "floor_inconsistencies_fixed": 0,
        "price_per_sqft_capped": 0,
        "size_capped": 0,
    }

    clean_df = df.copy()

    # 1. Deduplication
    init_count = len(clean_df)
    clean_df = clean_df.drop_duplicates(subset=["ID"]).drop_duplicates()
    logs["dropped_duplicates"] = init_count - len(clean_df)

    # 2. Imputation (median for numeric, mode for categorical)
    numeric_cols = clean_df.select_dtypes(include=[np.number]).columns
    categorical_cols = clean_df.select_dtypes(exclude=[np.number]).columns

    for col in numeric_cols:
        null_count = int(clean_df[col].isnull().sum())
        if null_count > 0:
            median_val = clean_df[col].median()
            clean_df[col] = clean_df[col].fillna(median_val)
            logs["numeric_imputed"] += null_count

    for col in categorical_cols:
        null_count = int(clean_df[col].isnull().sum())
        if null_count > 0:
            mode_val = clean_df[col].mode()[0]
            clean_df[col] = clean_df[col].fillna(mode_val)
            logs["categorical_imputed"] += null_count

    # 3. Fix inconsistent derived/logical columns
    # Floor_No vs Total_Floors: A property floor cannot exceed the total building floors.
    # Set Total_Floors = max(Total_Floors, Floor_No)
    floor_mask = clean_df["Floor_No"] > clean_df["Total_Floors"]
    logs["floor_inconsistencies_fixed"] = int(floor_mask.sum())
    clean_df["Total_Floors"] = np.maximum(clean_df["Total_Floors"], clean_df["Floor_No"])

    # Recompute Age_of_Property strictly as REFERENCE_YEAR - Year_Built
    clean_df["Age_of_Property"] = REFERENCE_YEAR - clean_df["Year_Built"]

    # 4. Feature engineering: Price per SqFt in INR
    clean_df["Price_per_SqFt"] = (clean_df["Price_in_Lakhs"] * INR_PER_LAKH) / clean_df["Size_in_SqFt"]

    # 5. Outlier detection and capping via IQR
    # Price per sqft capping
    q1_p = clean_df["Price_per_SqFt"].quantile(0.25)
    q3_p = clean_df["Price_per_SqFt"].quantile(0.75)
    iqr_p = q3_p - q1_p
    lower_p = max(0.0, q1_p - IQR_FACTOR * iqr_p)
    upper_p = q3_p + IQR_FACTOR * iqr_p
    price_outliers = (clean_df["Price_per_SqFt"] < lower_p) | (clean_df["Price_per_SqFt"] > upper_p)
    logs["price_per_sqft_capped"] = int(price_outliers.sum())
    clean_df["Price_per_SqFt"] = clean_df["Price_per_SqFt"].clip(lower=lower_p, upper=upper_p)

    # Size capping
    q1_s = clean_df["Size_in_SqFt"].quantile(0.25)
    q3_s = clean_df["Size_in_SqFt"].quantile(0.75)
    iqr_s = q3_s - q1_s
    lower_s = max(100.0, q1_s - IQR_FACTOR * iqr_s)
    upper_s = q3_s + IQR_FACTOR * iqr_s
    size_outliers = (clean_df["Size_in_SqFt"] < lower_s) | (clean_df["Size_in_SqFt"] > upper_s)
    logs["size_capped"] = int(size_outliers.sum())
    clean_df["Size_in_SqFt"] = clean_df["Size_in_SqFt"].clip(lower=lower_s, upper=upper_s)

    # Keep Price_in_Lakhs mathematically synchronized with capped Price_per_SqFt and Size
    clean_df["Price_in_Lakhs"] = (clean_df["Price_per_SqFt"] * clean_df["Size_in_SqFt"]) / INR_PER_LAKH

    # 6. Feature Engineering: Infrastructure Score
    min_schools, max_schools = clean_df["Nearby_Schools"].min(), clean_df["Nearby_Schools"].max()
    min_hosp, max_hosp = clean_df["Nearby_Hospitals"].min(), clean_df["Nearby_Hospitals"].max()

    norm_schools = (clean_df["Nearby_Schools"] - min_schools) / max(1.0, (max_schools - min_schools))
    norm_hospitals = (clean_df["Nearby_Hospitals"] - min_hosp) / max(1.0, (max_hosp - min_hosp))
    norm_transport = clean_df["Public_Transport_Accessibility"].map(PUBLIC_TRANSPORT_MAP).astype(float)

    clean_df["infrastructure_score"] = (
        norm_schools * INFRASTRUCTURE_WEIGHTS["schools"]
        + norm_hospitals * INFRASTRUCTURE_WEIGHTS["hospitals"]
        + norm_transport * INFRASTRUCTURE_WEIGHTS["transport"]
    )

    # 7. Additional Feature Engineering
    # School density score (normalized school availability index)
    clean_df["school_density_score"] = (clean_df["Nearby_Schools"] - min_schools) / max(1.0, (max_schools - min_schools))

    # Amenity count
    clean_df["amenity_count"] = clean_df["Amenities"].apply(
        lambda val: len([x.strip() for x in str(val).split(",") if x.strip()]) if pd.notnull(val) else 0
    )

    # Security feature count (binary indicator)
    clean_df["security_feature_count"] = (clean_df["Security"] == "Yes").astype(int)

    # Floor ratio (Floor_No / Total_Floors)
    clean_df["floor_ratio"] = clean_df["Floor_No"] / clean_df["Total_Floors"]

    # Age bucket
    clean_df["age_bucket"] = pd.cut(
        clean_df["Age_of_Property"],
        bins=AGE_BINS,
        labels=AGE_LABELS
    ).astype(str)

    print("\n" + "=" * 60)
    print("STAGE 1: CLEANING AND ENGINEERING LOGS")
    print("=" * 60)
    for key, val in logs.items():
        print(f"- {key}: {val:,}")

    return clean_df, logs


def main() -> None:
    """Execute Stage 1 preprocessing pipeline."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data file not found at: {RAW_DATA_PATH}")

    print(f"Loading raw dataset from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)
    audit_dataset(df)
    clean_df, logs = clean_and_engineer(df)

    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"\nSuccessfully cleaned dataset saved to: {CLEAN_DATA_PATH} (Shape: {clean_df.shape})")


if __name__ == "__main__":
    main()
