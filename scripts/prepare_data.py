"""Data Preparation Script for Phase 2.

Executes end-to-end data validation, feature engineering, categorical encoding,
and dataset serialization. Produces:
- data/processed/ginger_processed.csv
- results/feature_engineering_summary.txt
- results/feature_metadata.csv

Usage:
    python scripts/prepare_data.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.data_loader import load_raw_data
from src.data.feature_engineering import get_feature_metadata_records
from src.data.preprocessing import (
    get_model_feature_columns,
    preprocess_dataset,
    validate_raw_schema,
)
from src.utils.config import (
    FEATURE_METADATA_PATH,
    FEATURE_SUMMARY_PATH,
    ID_COLUMNS,
    LEAKAGE_COLUMNS,
    NOMINAL_CATEGORICAL_COLUMNS,
    ORDINAL_MAPPINGS,
    PROCESSED_DATA_PATH,
    RAW_NUMERICAL_COLUMNS,
    RAW_TEMPORAL_COLUMNS,
    RESULTS_DIR,
    TARGET_COLUMN,
)


def export_feature_summary(
    raw_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    model_features: list,
    output_path: Path,
):
    """Generate detailed textual feature engineering report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PHASE 2: FEATURE ENGINEERING & DATA PREPROCESSING SUMMARY\n")
        f.write("Research Module: IT41012 | Ginger Yield Prediction Research\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. ORIGINAL RAW COLUMNS (Total: 22)\n")
        for col in raw_df.columns:
            f.write(f"   - {col}\n")
        f.write("\n")

        f.write("2. TARGET VARIABLE\n")
        f.write(f"   - Target Column: {TARGET_COLUMN}\n")
        f.write("   - Role: Ground truth dependent variable (Final harvested yield in t/ha).\n")
        f.write("   - In-Season Status: Post-harvest ground truth (Never used as feature).\n\n")

        f.write("3. LEAKAGE RISK COLUMNS (ISOLATED FROM MODEL INPUTS)\n")
        for col in LEAKAGE_COLUMNS:
            f.write(f"   - {col}\n")
            f.write("     Reason: Pre-existing prediction field strongly correlated (r=0.988) with target.\n")
            f.write("     Action: Retained in processed CSV for benchmark comparisons only; excluded from model X.\n")
        f.write("\n")

        f.write("4. IDENTIFIER / GROUPING COLUMNS\n")
        for col in ID_COLUMNS:
            f.write(f"   - {col}\n")
            f.write("     Reason: Administrative identifier; excluded from model inputs to prevent ID memorization.\n")
            f.write("     Action: Retained in processed CSV for grouping and dynamic trajectory evaluation.\n")
        f.write("\n")

        f.write("5. CATEGORICAL ENCODING\n")
        f.write("   A. Nominal Features (One-Hot Encoded):\n")
        for col in NOMINAL_CATEGORICAL_COLUMNS:
            unique_cats = sorted(raw_df[col].unique())
            f.write(f"      * {col} ({len(unique_cats)} categories): {unique_cats}\n")
            for cat in unique_cats:
                f.write(f"        -> {col}_{cat}\n")
        f.write("\n   B. Ordinal Features (Biologically / Logically Ranked):\n")
        for col, mapping in ORDINAL_MAPPINGS.items():
            f.write(f"      * {col} -> {col}_Ordinal: {mapping}\n")
        f.write("\n")

        f.write("6. ENGINEERED IN-SEASON FEATURES & AGRONOMIC JUSTIFICATIONS\n")
        f.write("   - Total_Water_Input_mm:\n")
        f.write("     * Formula: Weekly_Rainfall_mm + Irrigation_mm_week\n")
        f.write("     * Agronomic Rationale: Net combined water supplied via natural rainfall and supplemental irrigation.\n")
        f.write("     * In-Season Availability: Available at observation date.\n\n")

        f.write("   - Plant_Height_per_DAP:\n")
        f.write("     * Formula: Plant_Height_cm / Days_After_Planting\n")
        f.write("     * Agronomic Rationale: Daily vertical elongation velocity (cm/day), capturing early vegetative vigor.\n")
        f.write("     * In-Season Availability: Available at observation date.\n\n")

        f.write("   - Canopy_Greenness_Volume:\n")
        f.write("     * Formula: Plant_Height_cm * Leaf_Greenness_Index\n")
        f.write("     * Agronomic Rationale: Composite proxy for active photosynthetic canopy biomass and chlorophyll content.\n")
        f.write("     * In-Season Availability: Available at observation date.\n\n")

        f.write("   - Biotic_Stress_Index:\n")
        f.write("     * Formula: Pest_Severity_Ordinal + Disease_Severity_Ordinal (Range: 0 to 4)\n")
        f.write("     * Agronomic Rationale: Quantifies cumulative biotic stress pressure from insect pests and pathogens.\n")
        f.write("     * In-Season Availability: Available at observation date.\n\n")

        f.write("   - Planting_Month:\n")
        f.write("     * Formula: Extraction of calendar month from Planting_Date\n")
        f.write("     * Agronomic Rationale: Captures planting timing relative to Sri Lankan monsoonal seasons (Yala / Maha).\n")
        f.write("     * In-Season Availability: Known at planting.\n\n")

        f.write("7. FINAL MODEL PREDICTOR MATRIX SPECIFICATION\n")
        f.write(f"   - Total Features in Processed File: {len(processed_df.columns)}\n")
        f.write(f"   - Features Used for Machine Learning (X): {len(model_features)}\n")
        f.write("   - Feature List (X):\n")
        for i, feat in enumerate(model_features, 1):
            f.write(f"     {i:2d}. {feat}\n")
        f.write("\n")

        f.write("8. RESEARCH ASSUMPTIONS & LIMITATIONS\n")
        f.write("   - Assumption 1: Weekly rainfall and irrigation are assumed to be recorded accurately for the 7 days prior to observation.\n")
        f.write("   - Assumption 2: Leaf greenness index acts as a direct monotonic proxy for SPAD / chlorophyll density.\n")
        f.write("   - Limitation 1: Cross-sectional nature of Farm_ID precludes within-farm temporal autoregressive lag features.\n")
        f.write("   - Limitation 2: Weather observations represent weekly snapshots rather than continuous cumulative degree-day series.\n\n")

        f.write("=" * 80 + "\n")
        f.write("END OF FEATURE ENGINEERING SUMMARY\n")
        f.write("=" * 80 + "\n")


def export_feature_metadata(output_path: Path):
    """Generate structured CSV metadata dictionary."""
    records = get_feature_metadata_records()
    df_meta = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_meta.to_csv(output_path, index=False)


def main():
    print("=" * 80)
    print("  GINGER YIELD PREDICTION RESEARCH - DATA PREPARATION (PHASE 2)")
    print("  Module: Research Methods and Scientific Writing (IT41012)")
    print("=" * 80)

    # 1. Load Raw Data
    print("\n[1/5] Loading raw dataset...")
    df_raw = load_raw_data()
    print(f"      Loaded: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")

    # 2. Validate Raw Schema
    print("\n[2/5] Validating raw schema and domain constraints...")
    val_report = validate_raw_schema(df_raw)
    if val_report["is_valid"]:
        print("      Validation PASSED (0 missing values, 0 infinite values, all ranges valid)")
    else:
        print(f"      Validation FAILED: {val_report}")
        sys.exit(1)

    # 3. Apply Preprocessing & Feature Engineering
    print("\n[3/5] Applying feature engineering and categorical encoding...")
    df_processed = preprocess_dataset(df_raw)
    model_features = get_model_feature_columns(df_processed)

    print(f"      Raw columns:       {len(df_raw.columns)}")
    print(f"      Processed columns: {len(df_processed.columns)}")
    print(f"      Model predictors:  {len(model_features)} features (strictly in-season & leakage-free)")

    # 4. Save Processed Dataset
    print("\n[4/5] Saving processed dataset...")
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"      Saved to: {PROCESSED_DATA_PATH.resolve()} ({PROCESSED_DATA_PATH.stat().st_size / (1024*1024):.2f} MB)")

    # 5. Save Feature Metadata & Summary
    print("\n[5/5] Exporting feature metadata and documentation...")
    export_feature_summary(df_raw, df_processed, model_features, FEATURE_SUMMARY_PATH)
    export_feature_metadata(FEATURE_METADATA_PATH)
    print(f"      - Summary text: {FEATURE_SUMMARY_PATH.resolve()}")
    print(f"      - Metadata CSV: {FEATURE_METADATA_PATH.resolve()}")

    print("\n" + "=" * 80)
    print("  PHASE 2 COMPLETED SUCCESSFULLY!")
    print(f"  Processed Dataset: {df_processed.shape[0]:,} rows x {df_processed.shape[1]} columns")
    print(f"  Model Feature Count (X): {len(model_features)} predictors")
    print("=" * 80)


if __name__ == "__main__":
    main()
