"""Standalone Data Inspection Script.

Executable entry point to load, inspect, and evaluate the raw ginger dataset,
print formatted summaries to the terminal, and export artifacts into results/.

Usage:
    python scripts/inspect_data.py
"""

import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_inspection import (
    generate_all_inspection_artifacts,
    inspect_categorical_distributions,
    inspect_crop_health,
    inspect_cultivation_info,
    inspect_data_quality,
    inspect_dataset_structure,
    inspect_descriptive_statistics,
    inspect_dynamic_feasibility,
    inspect_environmental_variables,
    inspect_existing_predictions,
    inspect_farm_management,
    inspect_target_variable,
)
from src.data.data_loader import load_raw_data
from src.utils.config import RESULTS_DIR


def main():
    print("=" * 80)
    print("  GINGER YIELD PREDICTION RESEARCH - DATA INSPECTION (PHASE 1)")
    print("  Module: Research Methods and Scientific Writing (IT41012)")
    print("=" * 80)

    # 1. Load raw data
    print("\n[1/6] Loading raw dataset...")
    try:
        df = load_raw_data()
        print(f"      Successfully loaded dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
    except Exception as e:
        print(f"      ERROR loading dataset: {e}")
        sys.exit(1)

    # 2. Structure & Quality
    print("\n[2/6] Inspecting dataset structure & data quality...")
    struct = inspect_dataset_structure(df)
    quality = inspect_data_quality(df)
    print(f"      - Dimensions: {struct['n_rows']:,} rows, {struct['n_columns']} columns")
    print(f"      - Duplicate Rows: {quality['duplicate_rows_count']}")
    print(f"      - Total Missing Cells: {quality['total_missing_cells']}")
    print("      - Column Data Types Summary:")
    type_counts = pd.Series(struct['dtypes']).value_counts()
    for dtype, count in type_counts.items():
        print(f"        * {dtype}: {count} columns")

    # 3. Domain Variables Inspection
    print("\n[3/6] Inspecting domain variables (Cultivation, Environmental, Management, Health)...")
    cult = inspect_cultivation_info(df)
    if "district_distribution" in cult:
        dist_str = ", ".join([f"{k}: {v}" for k, v in cult["district_distribution"].items()])
        print(f"      - Districts ({len(cult['district_distribution'])}): {dist_str}")
    if "variety_distribution" in cult:
        var_str = ", ".join([f"{k}: {v}" for k, v in cult["variety_distribution"].items()])
        print(f"      - Seed Varieties: {var_str}")
    if "growth_stage_distribution" in cult:
        gs_str = ", ".join([f"{k}: {v}" for k, v in cult["growth_stage_distribution"].items()])
        print(f"      - Growth Stages: {gs_str}")
    if "dap_range" in cult:
        dap = cult["dap_range"]
        print(f"      - Days After Planting (DAP) Range: {dap['min']} to {dap['max']} (mean={dap['mean']:.1f})")

    # 4. Target Variable Analysis
    print("\n[4/6] Inspecting target variable (Actual_Final_Yield_t_ha)...")
    target = inspect_target_variable(df)
    print(f"      - Range: [{target['min']:.2f}, {target['max']:.2f}] t/ha")
    print(f"      - Mean: {target['mean']:.2f} t/ha | Median: {target['median']:.2f} t/ha | Std: {target['std']:.2f}")
    print(f"      - IQR: [{target['q25']:.2f}, {target['q75']:.2f}] (IQR={target['iqr']:.2f})")
    print(f"      - Skewness: {target['skewness']:.3f} | Kurtosis: {target['kurtosis']:.3f}")

    # 5. Existing Predictions & Data Leakage Review
    print("\n[5/6] Inspecting existing prediction field (Predicted_Final_Yield_t_ha)...")
    pred = inspect_existing_predictions(df)
    print("      * WARNING: This field is an external/pre-existing estimate.")
    print("      * STRICT RULE: Must NOT be used as an input feature for ML models.")
    if "correlation_with_actual" in pred:
        print(f"      - Correlation with Actual Yield: {pred['correlation_with_actual']:.4f}")
        print(f"      - MAE vs Actual: {pred['mae_vs_actual']:.2f} t/ha | RMSE: {pred['rmse_vs_actual']:.2f} t/ha")
        print(f"      - Mean Error Bias: {pred['mean_error_bias']:.2f} t/ha")

    # 6. Dynamic Feasibility & Artifact Export
    print("\n[6/6] Evaluating dynamic in-season feasibility & exporting artifacts...")
    feasibility = inspect_dynamic_feasibility(df)
    print(f"      - Explicit Farm_ID Present: {feasibility.get('has_farm_id')}")
    print(f"      - Unique Farms: {feasibility.get('unique_farms')}")
    print(f"      - Observations per Farm: {feasibility.get('min_observations_per_farm')} to {feasibility.get('max_observations_per_farm')} (mean={feasibility.get('mean_observations_per_farm')})")
    print(f"      - Farms with Varying Districts: {feasibility.get('pct_farms_with_multiple_districts')}%")
    print(f"      - Farms with Varying Yield: {feasibility.get('pct_farms_with_varying_final_yield')}%")
    print("      - Research Finding: Synthetic observations sampled cross-sectionally.")

    artifacts = generate_all_inspection_artifacts(df, RESULTS_DIR)
    print("\n" + "=" * 80)
    print("  INSPECTION COMPLETED SUCCESSFULLY!")
    print(f"  Artifacts saved to: {RESULTS_DIR.resolve()}")
    for name, path in artifacts.items():
        print(f"   * {name:30s} -> {path.name}")
    print("=" * 80)


if __name__ == "__main__":
    main()
