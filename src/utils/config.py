"""Project Configuration and Constants for Ginger Yield Prediction System.

Module: Research Methods and Scientific Writing (IT41012)
Project: An Explainable AI-Based Dynamic In-Season Ginger Yield Prediction System
         for Sri Lankan Farmers.
"""

from pathlib import Path

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_STATIC_BASELINE_DIR = MODELS_DIR
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_PLOTS_DIR = RESULTS_DIR / "plots"

# File Paths - Datasets
RAW_DATA_PATH = DATA_RAW_DIR / "ginger_dataset.csv"
PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / "ginger_processed.csv"

# File Paths - Phase 2 Artifacts
FEATURE_SUMMARY_PATH = RESULTS_DIR / "feature_engineering_summary.txt"
FEATURE_METADATA_PATH = RESULTS_DIR / "feature_metadata.csv"

# File Paths - Phase 3 Static Baseline Model Artifacts
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
RIDGE_MODEL_PATH = MODELS_DIR / "ridge_model.joblib"
RF_MODEL_PATH = MODELS_DIR / "random_forest_model.joblib"
XGB_MODEL_PATH = MODELS_DIR / "xgboost_model.joblib"
LGBM_MODEL_PATH = MODELS_DIR / "lightgbm_model.joblib"
CATBOOST_MODEL_PATH = MODELS_DIR / "catboost_model.joblib"

# File Paths - Results & Reports
CV_RESULTS_PATH = RESULTS_DIR / "cross_validation_results.csv"
FINAL_TEST_RESULTS_PATH = RESULTS_DIR / "final_test_results.csv"
MODEL_COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"
HYPERPARAMETER_RESULTS_PATH = RESULTS_DIR / "hyperparameter_results.csv"
TEST_PREDICTIONS_PATH = RESULTS_DIR / "test_predictions.csv"
MODEL_FEATURES_PATH = RESULTS_DIR / "model_features.csv"
PHASE3_REPORT_PATH = RESULTS_DIR / "phase3_modeling_report.txt"

# Target Variable (Ground Truth)
TARGET_COLUMN = "Actual_Final_Yield_t_ha"

# Data Leakage Columns - DO NOT use as model training features
LEAKAGE_COLUMNS = ["Predicted_Final_Yield_t_ha"]

# Identification / Grouping Columns - Excluded from model predictive features
ID_COLUMNS = ["Farm_ID"]

# Temporal Columns
RAW_TEMPORAL_COLUMNS = ["Planting_Date"]
TEMPORAL_COLUMNS = RAW_TEMPORAL_COLUMNS

# Nominal Categorical Columns (Targeted for One-Hot Encoding)
NOMINAL_CATEGORICAL_COLUMNS = [
    "District",
    "Seed_Variety",
    "Soil_Type",
]

# All Raw Categorical Columns
CATEGORICAL_COLUMNS = [
    "District",
    "Growth_Stage",
    "Seed_Variety",
    "Soil_Type",
    "Pest_Severity",
    "Disease_Severity",
]

# Canonical Sri Lankan Ginger Agro-Ecological Domain Values
DISTRICTS = [
    "Badulla",
    "Galle",
    "Kandy",
    "Kegalle",
    "Kurunegala",
    "Matale",
    "Matara",
    "Monaragala",
    "Nuwara Eliya",
    "Ratnapura",
]

# Canonical Geographical Coordinates (Centroids) for Sri Lankan Districts
DISTRICT_COORDINATES = {
    "Badulla": (6.9895, 81.0557),
    "Galle": (6.0535, 80.2210),
    "Kandy": (7.2906, 80.6337),
    "Kegalle": (7.2513, 80.3464),
    "Kurunegala": (7.4863, 80.3623),
    "Matale": (7.4675, 80.6234),
    "Matara": (5.9549, 80.5550),
    "Monaragala": (6.8728, 81.3507),
    "Nuwara Eliya": (6.9497, 80.7891),
    "Ratnapura": (6.6828, 80.4029),
}

SEED_VARIETIES = [
    "Chinese",
    "Local",
    "Nadun",
]

SOIL_TYPES = [
    "Clay Loam",
    "Loam",
    "Sandy Loam",
]

GROWTH_STAGES = [
    "Sprouting",
    "Early Vegetative",
    "Vegetative",
    "Rhizome Initiation",
    "Rhizome Development",
    "Maturity",
]

SEVERITY_LEVELS = [
    "Low",
    "Medium",
    "High",
]

# Ordinal Categorical Columns and Their Defensible Numerical Mappings
ORDINAL_MAPPINGS = {
    "Growth_Stage": {
        "Sprouting": 0,
        "Early Vegetative": 1,
        "Vegetative": 2,
        "Rhizome Initiation": 3,
        "Rhizome Development": 4,
        "Maturity": 5,
    },
    "Pest_Severity": {
        "Low": 0,
        "Medium": 1,
        "High": 2,
    },
    "Disease_Severity": {
        "Low": 0,
        "Medium": 1,
        "High": 2,
    },
}

# Original Numerical Features in Raw Dataset
RAW_NUMERICAL_COLUMNS = [
    "Days_After_Planting",
    "Land_Size_Acres",
    "Soil_pH",
    "Soil_Moisture_pct",
    "Weekly_Rainfall_mm",
    "Avg_Temperature_C",
    "Relative_Humidity_pct",
    "Solar_Radiation_MJ_m2_day",
    "Irrigation_mm_week",
    "Fertilizer_kg_acre",
    "Plant_Height_cm",
    "Leaf_Greenness_Index",
]
NUMERICAL_COLUMNS = RAW_NUMERICAL_COLUMNS

# Engineered In-Season Features
ENGINEERED_FEATURE_COLUMNS = [
    "Total_Water_Input_mm",
    "Plant_Height_per_DAP",
    "Canopy_Greenness_Volume",
    "Biotic_Stress_Index",
    "Planting_Month",
]

# Feature Groups for Domain-Specific Analysis
ENVIRONMENTAL_FEATURES = [
    "Weekly_Rainfall_mm",
    "Avg_Temperature_C",
    "Relative_Humidity_pct",
    "Solar_Radiation_MJ_m2_day",
    "Soil_Moisture_pct",
    "Soil_pH",
]

FARM_MANAGEMENT_FEATURES = [
    "Land_Size_Acres",
    "Irrigation_mm_week",
    "Fertilizer_kg_acre",
]

CROP_HEALTH_FEATURES = [
    "Plant_Height_cm",
    "Leaf_Greenness_Index",
    "Pest_Severity",
    "Disease_Severity",
]

# Data Source Taxonomy (Zero Physical Hardware Requirement)
DATA_SOURCE_FARMER = "FARMER_REPORTED"
DATA_SOURCE_API = "API_DERIVED"
DATA_SOURCE_HISTORICAL = "HISTORICAL_DATASET"
DATA_SOURCE_SYNTHETIC = "SYNTHETIC_SIMULATED"

# Agronomic & Unit Conversion Constants
SQM_PER_ACRE = 4046.86  # 1 acre = 4046.86 m²
MM_PER_LITER_PER_SQM = 1.0  # 1 mm depth = 1 L/m²

# Validation & Modeling Settings
TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5

# Plausible Range Validation Boundaries for Agronomic Integrity
VALIDATION_RANGES = {
    "Soil_pH": (4.0, 9.0),
    "Soil_Moisture_pct": (0.0, 100.0),
    "Weekly_Rainfall_mm": (0.0, 500.0),
    "Avg_Temperature_C": (10.0, 45.0),
    "Relative_Humidity_pct": (0.0, 100.0),
    "Solar_Radiation_MJ_m2_day": (0.0, 40.0),
    "Irrigation_mm_week": (0.0, 200.0),
    "Fertilizer_kg_acre": (0.0, 300.0),
    "Plant_Height_cm": (0.0, 250.0),
    "Leaf_Greenness_Index": (0.0, 100.0),
    "Days_After_Planting": (1, 365),
    "Actual_Final_Yield_t_ha": (0.0, 50.0),
}
