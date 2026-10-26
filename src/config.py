"""
Configuration module for the Real Estate Investment Advisor project.

Contains all paths, tunable constants, growth rate rules, city tiers,
feature column definitions, and random seeds.
"""

from pathlib import Path
from typing import Dict, List, Set

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"
REPORTS_DIR = BASE_DIR / "reports"
EDA_REPORTS_DIR = REPORTS_DIR / "eda"
MODELS_DIR = BASE_DIR / "models"
MLRUNS_DIR = str(BASE_DIR / "mlruns")

# File paths
RAW_DATA_PATH = DATA_DIR / "india_housing_prices.csv"
CLEAN_DATA_PATH = DATA_DIR / "housing_clean.csv"
EDA_FINDINGS_PATH = EDA_REPORTS_DIR / "findings.md"
MODEL_RESULTS_PATH = REPORTS_DIR / "model_results.md"

# Model artifact paths
BEST_CLASSIFIER_PATH = MODELS_DIR / "best_classifier.joblib"
BEST_REGRESSOR_PATH = MODELS_DIR / "best_regressor.joblib"
PREPROCESSOR_CLASSIFIER_PATH = MODELS_DIR / "preprocessor_classifier.joblib"
PREPROCESSOR_REGRESSOR_PATH = MODELS_DIR / "preprocessor_regressor.joblib"
FEATURES_CLASSIFIER_PATH = MODELS_DIR / "features_classifier.joblib"
FEATURES_REGRESSOR_PATH = MODELS_DIR / "features_regressor.joblib"

# Reproducibility
RANDOM_SEED: int = 42

# MLflow experiments
EXPERIMENT_CLASSIFIER: str = "investment_classifier"
EXPERIMENT_REGRESSOR: str = "price_regressor"

# Preprocessing & Units
REFERENCE_YEAR: int = 2025
INR_PER_LAKH: float = 100000.0
IQR_FACTOR: float = 1.5

# Feature Engineering Definitions
AGE_BINS: List[float] = [-1.0, 5.0, 10.0, 20.0, 100.0]
AGE_LABELS: List[str] = ["0-5 Years", "6-10 Years", "11-20 Years", "20+ Years"]

PUBLIC_TRANSPORT_MAP: Dict[str, float] = {
    "Low": 0.0,
    "Medium": 0.5,
    "High": 1.0
}

INFRASTRUCTURE_WEIGHTS: Dict[str, float] = {
    "schools": 1.0 / 3.0,
    "hospitals": 1.0 / 3.0,
    "transport": 1.0 / 3.0
}

# City Tiers
TIER_1_CITIES: Set[str] = {
    "Bangalore", "Chennai", "Gurgaon", "Hyderabad",
    "Kolkata", "Mumbai", "New Delhi", "Noida", "Pune", "Ahmedabad"
}

TIER_2_CITIES: Set[str] = {
    "Bhopal", "Bhubaneswar", "Coimbatore", "Dehradun", "Faridabad",
    "Indore", "Jaipur", "Kochi", "Lucknow", "Ludhiana", "Mysore",
    "Nagpur", "Patna", "Surat", "Trivandrum", "Vishakhapatnam"
}

# Growth Rate Rules (Annual r per property)
# Base rate + city tier adjustment + property type adjustment + infrastructure bonus + age adjustment
BASE_GROWTH_RATE: float = 0.06

GROWTH_TIER_ADJUSTMENTS: Dict[str, float] = {
    "Tier 1": 0.02,
    "Tier 2": 0.01,
    "Tier 3": 0.00
}

GROWTH_PROPERTY_TYPE_ADJUSTMENTS: Dict[str, float] = {
    "Apartment": 0.010,
    "Independent House": 0.015,
    "Villa": 0.005
}

GROWTH_INFRASTRUCTURE_MAX_BONUS: float = 0.02

GROWTH_AGE_ADJUSTMENTS: Dict[str, float] = {
    "0-5 Years": 0.015,
    "6-10 Years": 0.005,
    "11-20 Years": -0.005,
    "20+ Years": -0.015
}

MIN_GROWTH_RATE: float = 0.03
MAX_GROWTH_RATE: float = 0.12
PROJECTION_YEARS: int = 5

# Good Investment Rule Thresholds
# Initial naive thresholds (<=50th price, >=60th infra) led to 81.1% / 18.9% (exceeded 80/20 limit).
# Adjusted thresholds to maintain healthy ~75/25 balance:
GOOD_INVESTMENT_PRICE_PERCENTILE: float = 0.55  # At or below 55th percentile of city price/sqft
GOOD_INVESTMENT_INFRA_PERCENTILE: float = 0.50  # Top 50% infrastructure score
GOOD_INVESTMENT_MIN_ANNUAL_GROWTH: float = 0.070  # Min estimated growth rate of 7.0%
GOOD_INVESTMENT_EXCLUDE_STATUS: List[str] = ["Sold"]

# Data Column Specifications
ALL_RAW_COLUMNS: List[str] = [
    "ID", "State", "City", "Locality", "Property_Type", "BHK",
    "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Year_Built",
    "Furnished_Status", "Floor_No", "Total_Floors", "Age_of_Property",
    "Nearby_Schools", "Nearby_Hospitals", "Public_Transport_Accessibility",
    "Parking_Space", "Security", "Amenities", "Facing", "Owner_Type",
    "Availability_Status"
]

# Classification Leakage Isolation:
# Exclude direct label formula inputs and growth rate rules:
# - estimated_growth_rate (direct driver of future price & label)
# - Price_per_SqFt (used directly in city price percentile threshold)
# - infrastructure_score (used directly in infra threshold)
# - City_Tier (used directly in growth rate calculation)
# We keep remaining independent property attributes.
CLASSIFIER_FEATURE_COLUMNS: List[str] = [
    "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt",
    "Price_in_Lakhs", "Year_Built", "Furnished_Status", "Floor_No", "Total_Floors",
    "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals",
    "Public_Transport_Accessibility", "Parking_Space", "Security",
    "Facing", "Owner_Type", "Availability_Status", "amenity_count",
    "security_feature_count", "floor_ratio", "school_density_score", "age_bucket"
]

# Regression Leakage Isolation:
# Exclude columns derived directly from Price_in_Lakhs or the target formula:
# - Price_per_SqFt, estimated_growth_rate, City_Tier
# We keep Price_in_Lakhs with a mandatory baseline comparison (Price * (1 + avg_growth)^5)
# to evaluate if the model genuinely learns property-specific growth dynamics.
REGRESSOR_FEATURE_COLUMNS: List[str] = [
    "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt",
    "Price_in_Lakhs", "Year_Built", "Furnished_Status", "Floor_No", "Total_Floors",
    "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals",
    "Public_Transport_Accessibility", "Parking_Space", "Security",
    "Facing", "Owner_Type", "Availability_Status", "amenity_count",
    "security_feature_count", "floor_ratio", "school_density_score",
    "infrastructure_score", "age_bucket"
]
