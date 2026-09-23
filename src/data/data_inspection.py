"""Comprehensive Data Inspection Module for Ginger Yield Prediction Research.

Performs structured, non-destructive exploratory data analysis covering:
1. Dataset structure and data types
2. Data quality (missing values, duplicates, unexpected values)
3. Domain-specific features (environmental, management, crop health, cultivation)
4. Target variable statistical profile (Actual_Final_Yield_t_ha)
5. Existing baseline prediction profile and data leakage risk assessment
6. Dynamic in-season feasibility and longitudinal consistency analysis
7. Result artifact serialization to results/ directory.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.config import (
    CATEGORICAL_COLUMNS,
    CROP_HEALTH_FEATURES,
    ENVIRONMENTAL_FEATURES,
    FARM_MANAGEMENT_FEATURES,
    ID_COLUMNS,
    LEAKAGE_COLUMNS,
    NUMERICAL_COLUMNS,
    RESULTS_DIR,
    TARGET_COLUMN,
    TEMPORAL_COLUMNS,
)


def inspect_dataset_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """Inspect high-level dataset dimensions, columns, and data types."""
    return {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3),
    }


def inspect_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Inspect missing values, duplicates, and unique cardinality."""
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    missing_df = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_count.values,
        "missing_percentage": missing_pct.round(4).values,
    })

    duplicate_rows_count = int(df.duplicated().sum())

    unique_counts = {col: int(df[col].nunique()) for col in df.columns}

    return {
        "missing_df": missing_df,
        "total_missing_cells": int(missing_count.sum()),
        "duplicate_rows_count": duplicate_rows_count,
        "unique_counts": unique_counts,
    }


def inspect_descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate detailed descriptive statistics for numerical columns."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stats = df[num_cols].describe().T

    # Enrich with additional statistical moments
    stats["median"] = df[num_cols].median()
    stats["skewness"] = df[num_cols].skew().round(4)
    stats["kurtosis"] = df[num_cols].kurt().round(4)
    stats["iqr"] = stats["75%"] - stats["25%"]

    # Reorder columns logically
    ordered_cols = [
        "count", "mean", "std", "min", "25%", "median", "75%", "max", "iqr", "skewness", "kurtosis"
    ]
    return stats[[col for col in ordered_cols if col in stats.columns]].round(4)


def inspect_categorical_distributions(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Calculate frequency distributions and percentages for categorical variables."""
    cat_summaries = {}
    cat_cols = [col for col in CATEGORICAL_COLUMNS if col in df.columns]

    for col in cat_cols:
        counts = df[col].value_counts(dropna=False)
        pcts = (df[col].value_counts(dropna=False, normalize=True) * 100).round(2)
        cat_summaries[col] = pd.DataFrame({
            "category": counts.index.astype(str),
            "count": counts.values,
            "percentage": pcts.values,
        })
    return cat_summaries


def inspect_cultivation_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Inspect ginger cultivation metadata (district, variety, growth stage, DAP, planting date)."""
    info = {}
    if "District" in df.columns:
        info["district_distribution"] = df["District"].value_counts().to_dict()
    if "Seed_Variety" in df.columns:
        info["variety_distribution"] = df["Seed_Variety"].value_counts().to_dict()
    if "Growth_Stage" in df.columns:
        info["growth_stage_distribution"] = df["Growth_Stage"].value_counts().to_dict()
    if "Days_After_Planting" in df.columns:
        info["dap_range"] = {
            "min": int(df["Days_After_Planting"].min()),
            "max": int(df["Days_After_Planting"].max()),
            "mean": float(df["Days_After_Planting"].mean()),
            "median": float(df["Days_After_Planting"].median()),
        }
    if "Planting_Date" in df.columns:
        info["planting_date_range"] = {
            "min": str(df["Planting_Date"].min()),
            "max": str(df["Planting_Date"].max()),
            "unique_dates": int(df["Planting_Date"].nunique()),
        }
    return info


def inspect_environmental_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Inspect key environmental variables (Rainfall, Temp, Humidity, Solar, Soil Moisture, pH)."""
    env_cols = [col for col in ENVIRONMENTAL_FEATURES if col in df.columns]
    return df[env_cols].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]].round(3)


def inspect_farm_management(df: pd.DataFrame) -> pd.DataFrame:
    """Inspect farm management variables (Land Size, Irrigation, Fertilizer)."""
    mgmt_cols = [col for col in FARM_MANAGEMENT_FEATURES if col in df.columns]
    return df[mgmt_cols].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]].round(3)


def inspect_crop_health(df: pd.DataFrame) -> Dict[str, Any]:
    """Inspect crop health indicators (Pest/Disease severity, Plant height, Leaf greenness)."""
    health_info = {}
    if "Pest_Severity" in df.columns:
        health_info["pest_severity"] = df["Pest_Severity"].value_counts().to_dict()
    if "Disease_Severity" in df.columns:
        health_info["disease_severity"] = df["Disease_Severity"].value_counts().to_dict()

    num_health = [col for col in ["Plant_Height_cm", "Leaf_Greenness_Index"] if col in df.columns]
    if num_health:
        health_info["numerical_stats"] = (
            df[num_health].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]].round(3)
        )
    return health_info


def inspect_target_variable(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze ground truth target variable: Actual_Final_Yield_t_ha."""
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    target = df[TARGET_COLUMN]
    q25 = target.quantile(0.25)
    q75 = target.quantile(0.75)
    iqr = q75 - q25

    return {
        "target_column": TARGET_COLUMN,
        "count": int(target.count()),
        "missing_count": int(target.isnull().sum()),
        "min": round(float(target.min()), 4),
        "max": round(float(target.max()), 4),
        "mean": round(float(target.mean()), 4),
        "median": round(float(target.median()), 4),
        "std": round(float(target.std()), 4),
        "variance": round(float(target.var()), 4),
        "skewness": round(float(target.skew()), 4),
        "kurtosis": round(float(target.kurt()), 4),
        "q25": round(float(q25), 4),
        "q75": round(float(q75), 4),
        "iqr": round(float(iqr), 4),
    }


def inspect_existing_predictions(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze the existing Predicted_Final_Yield_t_ha column and evaluate leakage risk.

    NOTE: This column MUST NOT be used as a training feature in any ML models.
    """
    col = "Predicted_Final_Yield_t_ha"
    if col not in df.columns:
        return {"status": "Not Present in Dataset"}

    pred_series = df[col]
    actual_series = df[TARGET_COLUMN] if TARGET_COLUMN in df.columns else None

    result = {
        "column_name": col,
        "role": "EXTERNAL_BENCHMARK_ONLY (DO NOT USE AS ML INPUT FEATURE)",
        "missing_count": int(pred_series.isnull().sum()),
        "min": round(float(pred_series.min()), 4),
        "max": round(float(pred_series.max()), 4),
        "mean": round(float(pred_series.mean()), 4),
        "median": round(float(pred_series.median()), 4),
        "std": round(float(pred_series.std()), 4),
    }

    if actual_series is not None:
        diff = pred_series - actual_series
        abs_diff = (pred_series - actual_series).abs()
        result.update({
            "correlation_with_actual": round(float(pred_series.corr(actual_series)), 4),
            "mae_vs_actual": round(float(abs_diff.mean()), 4),
            "rmse_vs_actual": round(float(np.sqrt((diff ** 2).mean())), 4),
            "mean_error_bias": round(float(diff.mean()), 4),
            "max_underprediction": round(float(diff.min()), 4),
            "max_overprediction": round(float(diff.max()), 4),
        })

    return result


def inspect_dynamic_feasibility(df: pd.DataFrame) -> Dict[str, Any]:
    """Rigorous scientific feasibility check for dynamic in-season prediction.

    Checks:
    - Farm_ID presence and unique count
    - Observations per Farm_ID
    - Longitudinal consistency of static properties (District, Land_Size_Acres, Planting_Date)
    - Consistency of Actual_Final_Yield_t_ha across same Farm_ID
    - Distribution of Days_After_Planting and Growth_Stage per Farm_ID
    - Feasibility of true panel tracking vs synthetic cross-sectional sampling
    """
    has_farm_id = "Farm_ID" in df.columns
    has_dap = "Days_After_Planting" in df.columns
    has_growth_stage = "Growth_Stage" in df.columns

    feasibility: Dict[str, Any] = {
        "has_farm_id": has_farm_id,
        "has_observation_id": "Observation_ID" in df.columns,
        "has_days_after_planting": has_dap,
        "has_growth_stage": has_growth_stage,
    }

    if not has_farm_id:
        feasibility["finding"] = "No Farm_ID found. Data is cross-sectional."
        feasibility["dynamic_suitability"] = "LOW (Requires grouping or stage-based slicing)."
        return feasibility

    n_farms = int(df["Farm_ID"].nunique())
    obs_per_farm = df["Farm_ID"].value_counts()
    min_obs = int(obs_per_farm.min())
    max_obs = int(obs_per_farm.max())
    mean_obs = round(float(obs_per_farm.mean()), 2)

    feasibility.update({
        "unique_farms": n_farms,
        "min_observations_per_farm": min_obs,
        "max_observations_per_farm": max_obs,
        "mean_observations_per_farm": mean_obs,
    })

    # Check longitudinal consistency across Farm_IDs
    # 1. Does District stay constant for the same Farm_ID?
    if "District" in df.columns:
        districts_per_farm = df.groupby("Farm_ID")["District"].nunique()
        farms_with_multiple_districts = int((districts_per_farm > 1).sum())
    else:
        farms_with_multiple_districts = 0

    # 2. Does Actual_Final_Yield_t_ha stay constant for the same Farm_ID?
    if TARGET_COLUMN in df.columns:
        yields_per_farm = df.groupby("Farm_ID")[TARGET_COLUMN].nunique()
        farms_with_varying_final_yield = int((yields_per_farm > 1).sum())
    else:
        farms_with_varying_final_yield = 0

    # 3. Does Planting_Date stay constant for the same Farm_ID?
    if "Planting_Date" in df.columns:
        planting_dates_per_farm = df.groupby("Farm_ID")["Planting_Date"].nunique()
        farms_with_multiple_dates = int((planting_dates_per_farm > 1).sum())
    else:
        farms_with_multiple_dates = 0

    # 4. Growth stages per farm
    stages_per_farm = df.groupby("Farm_ID")["Growth_Stage"].nunique() if has_growth_stage else pd.Series()

    feasibility.update({
        "farms_with_multiple_districts": farms_with_multiple_districts,
        "pct_farms_with_multiple_districts": round((farms_with_multiple_districts / n_farms) * 100, 2),
        "farms_with_varying_final_yield": farms_with_varying_final_yield,
        "pct_farms_with_varying_final_yield": round((farms_with_varying_final_yield / n_farms) * 100, 2),
        "farms_with_multiple_planting_dates": farms_with_multiple_dates,
        "pct_farms_with_multiple_planting_dates": round((farms_with_multiple_dates / n_farms) * 100, 2),
        "avg_growth_stages_per_farm": round(float(stages_per_farm.mean()), 2) if not stages_per_farm.empty else 0,
    })

    # Critical research interpretation
    if farms_with_multiple_districts > 0 or farms_with_varying_final_yield > 0:
        feasibility["critical_limitation"] = (
            "CRITICAL RESEARCH FINDING: In the current dataset, records sharing the same 'Farm_ID' "
            "exhibit varying Districts, Planting Dates, and Final Actual Yield values. "
            "This confirms that 'Farm_ID' was randomly assigned or sampled per observation rather than "
            "tracking a true physical farm over a single cultivation cycle. "
            "Consequently, dynamic prediction must be framed as growth-stage/DAP-indexed observation slices "
            "rather than true farm-level longitudinal panel tracking."
        )
        feasibility["recommended_dynamic_approach"] = (
            "Model dynamic in-season predictions using stratified growth-stage cohorts "
            "(e.g., Emergence, Vegetative, Rhizome Initiation, Rhizome Enlargement, Maturity) "
            "where cumulative feature observations up to stage 't' predict the final yield."
        )
    else:
        feasibility["critical_limitation"] = "Data exhibits strict longitudinal consistency."
        feasibility["recommended_dynamic_approach"] = "Farm-level sequential panel modeling."

    return feasibility


def generate_all_inspection_artifacts(
    df: pd.DataFrame, output_dir: Optional[Path] = None
) -> Dict[str, Path]:
    """Execute all inspection routines and serialize structured outputs into results/."""
    out_dir = Path(output_dir) if output_dir is not None else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    artifacts: Dict[str, Path] = {}

    # 1. Structure
    struct = inspect_dataset_structure(df)

    # 2. Quality & Missing values CSV
    quality = inspect_data_quality(df)
    missing_csv_path = out_dir / "missing_values.csv"
    quality["missing_df"].to_csv(missing_csv_path, index=False)
    artifacts["missing_values"] = missing_csv_path

    # 3. Descriptive statistics CSV
    desc_stats = inspect_descriptive_statistics(df)
    desc_csv_path = out_dir / "descriptive_statistics.csv"
    desc_stats.to_csv(desc_csv_path)
    artifacts["descriptive_statistics"] = desc_csv_path

    # 4. Categorical summary CSV
    cat_summaries = inspect_categorical_distributions(df)
    cat_records = []
    for col_name, summary_df in cat_summaries.items():
        summary_copy = summary_df.copy()
        summary_copy["variable"] = col_name
        cat_records.append(summary_copy[["variable", "category", "count", "percentage"]])
    if cat_records:
        cat_df = pd.concat(cat_records, ignore_index=True)
        cat_csv_path = out_dir / "categorical_summary.csv"
        cat_df.to_csv(cat_csv_path, index=False)
        artifacts["categorical_summary"] = cat_csv_path

    # 5. Target summary CSV
    target_info = inspect_target_variable(df)
    target_df = pd.DataFrame([target_info])
    target_csv_path = out_dir / "target_summary.csv"
    target_df.to_csv(target_csv_path, index=False)
    artifacts["target_summary"] = target_csv_path

    # 6. Existing predictions & leakage assessment
    pred_info = inspect_existing_predictions(df)

    # 7. Dynamic feasibility analysis
    feasibility = inspect_dynamic_feasibility(df)
    feasibility_path = out_dir / "dynamic_feasibility_report.txt"
    with open(feasibility_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("DYNAMIC IN-SEASON PREDICTION FEASIBILITY & LIMITATION REPORT\n")
        f.write("Research Module: IT41012 | Ginger Yield Prediction System\n")
        f.write("=" * 80 + "\n\n")
        for k, v in feasibility.items():
            f.write(f"[{k}]\n{v}\n\n")
    artifacts["dynamic_feasibility_report"] = feasibility_path

    # 8. Comprehensive Dataset Summary Text Report
    summary_txt_path = out_dir / "dataset_summary.txt"
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("RESEARCH DATASET INSPECTION SUMMARY REPORT\n")
        f.write("Project: Explainable AI-Based Dynamic In-Season Ginger Yield Prediction\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. DATASET DIMENSIONS & STRUCTURE\n")
        f.write(f"   - Total Observations (Rows): {struct['n_rows']:,}\n")
        f.write(f"   - Total Attributes (Columns): {struct['n_columns']}\n")
        f.write(f"   - Memory Footprint: {struct['memory_usage_mb']} MB\n")
        f.write(f"   - Duplicate Rows: {quality['duplicate_rows_count']}\n")
        f.write(f"   - Total Missing Cells: {quality['total_missing_cells']}\n\n")

        f.write("2. COLUMN ROLES AND DATA TYPES\n")
        for col, dtype in struct["dtypes"].items():
            role_tag = ""
            if col == TARGET_COLUMN:
                role_tag = " [TARGET VARIABLE]"
            elif col in LEAKAGE_COLUMNS:
                role_tag = " [LEAKAGE RISK - BENCHMARK ONLY]"
            elif col in ID_COLUMNS:
                role_tag = " [IDENTIFIER]"
            elif col in CATEGORICAL_COLUMNS:
                role_tag = " [CATEGORICAL FEATURE]"
            elif col in NUMERICAL_COLUMNS:
                role_tag = " [NUMERICAL FEATURE]"
            f.write(f"   - {col:30s} : {dtype:10s} {role_tag}\n")
        f.write("\n")

        f.write("3. TARGET VARIABLE SUMMARY (Actual_Final_Yield_t_ha)\n")
        for k, v in target_info.items():
            f.write(f"   - {k:25s}: {v}\n")
        f.write("\n")

        f.write("4. EXISTING PREDICTION FIELD PROFILE (Predicted_Final_Yield_t_ha)\n")
        f.write("   WARNING: Isolated from model training to prevent target leakage.\n")
        for k, v in pred_info.items():
            f.write(f"   - {k:30s}: {v}\n")
        f.write("\n")

        f.write("5. DYNAMIC FEASIBILITY FINDINGS & LIMITATIONS\n")
        f.write(f"   - Farm IDs Present: {feasibility.get('has_farm_id')}\n")
        f.write(f"   - Unique Farm IDs: {feasibility.get('unique_farms')}\n")
        f.write(f"   - Observations per Farm: min={feasibility.get('min_observations_per_farm')}, "
                f"mean={feasibility.get('mean_observations_per_farm')}, "
                f"max={feasibility.get('max_observations_per_farm')}\n")
        f.write(f"   - Farms with Varying Districts: {feasibility.get('farms_with_multiple_districts')} "
                f"({feasibility.get('pct_farms_with_multiple_districts')}%)\n")
        f.write(f"   - Farms with Varying Final Yield: {feasibility.get('farms_with_varying_final_yield')} "
                f"({feasibility.get('pct_farms_with_varying_final_yield')}%)\n")
        f.write(f"   - Research Limitation Note:\n     {feasibility.get('critical_limitation')}\n")
        f.write(f"   - Recommended Modeling Strategy:\n     {feasibility.get('recommended_dynamic_approach')}\n\n")

        f.write("=" * 80 + "\n")
        f.write("END OF INSPECTION REPORT\n")
        f.write("=" * 80 + "\n")

    artifacts["dataset_summary"] = summary_txt_path
    return artifacts
