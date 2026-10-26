"""
Target generation module for the Real Estate Investment Advisor.

Constructs:
1. Annual growth rate r per property based on economic rules
2. Future_Price_5Y = Price_in_Lakhs * (1 + r) ** 5
3. Good_Investment = binary label based on multi-factor investment criteria
"""

from pathlib import Path
import sys
from typing import Dict, Tuple

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import (
    BASE_GROWTH_RATE,
    CLEAN_DATA_PATH,
    GOOD_INVESTMENT_EXCLUDE_STATUS,
    GOOD_INVESTMENT_INFRA_PERCENTILE,
    GOOD_INVESTMENT_MIN_ANNUAL_GROWTH,
    GOOD_INVESTMENT_PRICE_PERCENTILE,
    GROWTH_AGE_ADJUSTMENTS,
    GROWTH_INFRASTRUCTURE_MAX_BONUS,
    GROWTH_PROPERTY_TYPE_ADJUSTMENTS,
    GROWTH_TIER_ADJUSTMENTS,
    MAX_GROWTH_RATE,
    MIN_GROWTH_RATE,
    PROJECTION_YEARS,
    TIER_1_CITIES,
    TIER_2_CITIES,
)


def compute_city_tier(city: str) -> str:
    """Classify city into Tier 1, Tier 2, or Tier 3."""
    if city in TIER_1_CITIES:
        return "Tier 1"
    elif city in TIER_2_CITIES:
        return "Tier 2"
    return "Tier 3"


def compute_growth_rate(df: pd.DataFrame) -> pd.Series:
    """
    Compute estimated annual appreciation rate r per property.

    r = BASE_RATE
        + city_tier_adjustment
        + property_type_adjustment
        + infrastructure_score_adjustment
        + age_adjustment
    Clipped to [MIN_GROWTH_RATE, MAX_GROWTH_RATE] (3% to 12%).
    """
    city_tier = df["City"].apply(compute_city_tier)
    tier_adj = city_tier.map(GROWTH_TIER_ADJUSTMENTS).astype(float)
    prop_adj = df["Property_Type"].map(GROWTH_PROPERTY_TYPE_ADJUSTMENTS).astype(float)
    infra_adj = (df["infrastructure_score"] * GROWTH_INFRASTRUCTURE_MAX_BONUS).astype(float)
    age_adj = df["age_bucket"].map(GROWTH_AGE_ADJUSTMENTS).astype(float)

    r = BASE_GROWTH_RATE + tier_adj + prop_adj + infra_adj + age_adj
    r_clipped = r.clip(lower=MIN_GROWTH_RATE, upper=MAX_GROWTH_RATE)
    return r_clipped


def compute_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Compute Future_Price_5Y and Good_Investment labels.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float]]
        DataFrame with targets and summary metrics.
    """
    df_out = df.copy()

    # Add City_Tier column for downstream reporting
    df_out["City_Tier"] = df_out["City"].apply(compute_city_tier)

    # 1. Growth rate r
    df_out["estimated_growth_rate"] = compute_growth_rate(df_out)

    # 2. 5-Year Future Price
    df_out["Future_Price_5Y"] = df_out["Price_in_Lakhs"] * (
        (1.0 + df_out["estimated_growth_rate"]) ** PROJECTION_YEARS
    )

    # 3. Good Investment Rule Criteria:
    # Criterion A: Price per sqft at or below the city's 55th percentile threshold
    # Criterion B: Infrastructure score in the top 50%
    # Criterion C: Availability is not "Sold"
    # Criterion D: Annual growth rate r >= 7.0%
    city_price_thresh = df_out.groupby("City")["Price_per_SqFt"].transform(
        lambda s: s.quantile(GOOD_INVESTMENT_PRICE_PERCENTILE)
    )
    infra_thresh = float(df_out["infrastructure_score"].quantile(GOOD_INVESTMENT_INFRA_PERCENTILE))

    cond_price = df_out["Price_per_SqFt"] <= city_price_thresh
    cond_infra = df_out["infrastructure_score"] >= infra_thresh
    cond_avail = ~df_out["Availability_Status"].isin(GOOD_INVESTMENT_EXCLUDE_STATUS)
    cond_growth = df_out["estimated_growth_rate"] >= GOOD_INVESTMENT_MIN_ANNUAL_GROWTH

    good_investment = (cond_price & cond_infra & cond_avail & cond_growth).astype(int)
    df_out["Good_Investment"] = good_investment

    balance = df_out["Good_Investment"].value_counts(normalize=True).to_dict()
    counts = df_out["Good_Investment"].value_counts().to_dict()

    print("=" * 60)
    print("STAGE 3: TARGET GENERATION SUMMARY")
    print("=" * 60)
    print(f"Annual Growth Rate (r): min={df_out['estimated_growth_rate'].min():.3%}, "
          f"mean={df_out['estimated_growth_rate'].mean():.3%}, "
          f"max={df_out['estimated_growth_rate'].max():.3%}")
    print(f"Class 0 (Not Recommended): {counts.get(0, 0):,} ({balance.get(0, 0.0):.2%})")
    print(f"Class 1 (Good Investment): {counts.get(1, 0):,} ({balance.get(1, 0.0):.2%})")

    # Guard check for 80/20 threshold
    if balance.get(1, 0.0) < 0.20 or balance.get(0, 0.0) < 0.20:
        print("WARNING: Class balance is more extreme than 80/20! Threshold adjustment recommended.")
    else:
        print("✔ Class balance is within healthy 80/20 boundaries.")

    metrics = {
        "mean_growth_rate": float(df_out["estimated_growth_rate"].mean()),
        "class_0_ratio": float(balance.get(0, 0.0)),
        "class_1_ratio": float(balance.get(1, 0.0)),
        "class_0_count": int(counts.get(0, 0)),
        "class_1_count": int(counts.get(1, 0)),
    }
    return df_out, metrics


def main() -> None:
    """Load cleaned data, generate targets, and save back to clean data file."""
    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned data file not found at: {CLEAN_DATA_PATH}")

    print(f"Loading cleaned dataset from {CLEAN_DATA_PATH}...")
    df = pd.read_csv(CLEAN_DATA_PATH)
    df_with_targets, _ = compute_targets(df)

    df_with_targets.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"Targets saved to {CLEAN_DATA_PATH} (Shape: {df_with_targets.shape})")


if __name__ == "__main__":
    main()
