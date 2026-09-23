"""Data Preprocessing Module for Ginger Yield Prediction Research.

Provides pure, reproducible functions to:
- Validate raw dataset schemas and physical/agronomic range constraints
- Apply One-Hot Encoding to nominal variables (District, Seed_Variety, Soil_Type)
- Apply Ordinal Encoding to ordered variables (Growth_Stage, Pest_Severity, Disease_Severity)
- Integrate engineered in-season features
- Segregate predictive features from target, identifier, and leakage fields.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data.feature_engineering import engineer_all_features
from src.utils.config import (
    ID_COLUMNS,
    LEAKAGE_COLUMNS,
    NOMINAL_CATEGORICAL_COLUMNS,
    ORDINAL_MAPPINGS,
    RAW_TEMPORAL_COLUMNS,
    TARGET_COLUMN,
    VALIDATION_RANGES,
)


def validate_raw_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform rigorous sanity and domain-range validation on the raw dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Raw ginger dataset.

    Returns
    -------
    Dict[str, Any]
        Dictionary summarizing validation results, detected anomalies, and checks.

    Raises
    ------
    ValueError
        If critical required columns are missing, dataset is empty, or impossible
        values are detected.
    """
    if df.empty:
        raise ValueError("Cannot preprocess an empty dataframe.")

    # 1. Check critical columns presence
    required_cols = [
        "Farm_ID",
        "District",
        "Planting_Date",
        "Days_After_Planting",
        "Growth_Stage",
        "Land_Size_Acres",
        "Seed_Variety",
        "Soil_Type",
        "Soil_pH",
        "Soil_Moisture_pct",
        "Weekly_Rainfall_mm",
        "Avg_Temperature_C",
        "Relative_Humidity_pct",
        "Solar_Radiation_MJ_m2_day",
        "Irrigation_mm_week",
        "Fertilizer_kg_acre",
        "Pest_Severity",
        "Disease_Severity",
        "Plant_Height_cm",
        "Leaf_Greenness_Index",
        "Predicted_Final_Yield_t_ha",
        "Actual_Final_Yield_t_ha",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Raw dataset is missing required columns: {missing_cols}")

    # 2. Check for missing values
    missing_counts = df.isnull().sum()
    total_missing = int(missing_counts.sum())

    # 3. Check for infinite values in numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(df[num_cols]).sum().sum()
    if inf_counts > 0:
        raise ValueError(f"Dataset contains {inf_counts} infinite values.")

    # 4. Check domain range boundaries
    range_violations = {}
    for col, (min_val, max_val) in VALIDATION_RANGES.items():
        if col in df.columns:
            out_of_bounds = df[(df[col] < min_val) | (df[col] > max_val)]
            if not out_of_bounds.empty:
                range_violations[col] = {
                    "violation_count": len(out_of_bounds),
                    "allowed_range": (min_val, max_val),
                    "observed_range": (float(df[col].min()), float(df[col].max())),
                }

    # 5. Check categorical validity
    cat_violations = {}
    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df.columns:
            invalid_cats = set(df[col].dropna().unique()) - set(mapping.keys())
            if invalid_cats:
                cat_violations[col] = list(invalid_cats)

    return {
        "is_valid": total_missing == 0 and len(range_violations) == 0 and len(cat_violations) == 0,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "total_missing_cells": total_missing,
        "range_violations": range_violations,
        "categorical_violations": cat_violations,
    }


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode nominal variables via One-Hot and ordinal variables via Ordinal mapping.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with raw categorical columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with encoded columns added.
    """
    df_out = df.copy()

    # 1. Ordinal Encoding
    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df_out.columns:
            ord_col_name = f"{col}_Ordinal"
            df_out[ord_col_name] = df_out[col].astype(str).map(mapping).fillna(0).astype(int)

    # 2. One-Hot Encoding for Nominal Columns
    for col in NOMINAL_CATEGORICAL_COLUMNS:
        if col in df_out.columns:
            # Sort categories for deterministic column ordering
            categories = sorted(df_out[col].dropna().unique())
            for cat in categories:
                dummy_col_name = f"{col}_{cat}"
                df_out[dummy_col_name] = (df_out[col] == cat).astype(int)

    return df_out


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Execute complete validation, feature engineering, and encoding pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw ginger dataset.

    Returns
    -------
    pd.DataFrame
        Processed research dataset ready for Phase 3 modeling.
    """
    # 1. Validate schema and ranges
    val_report = validate_raw_schema(df)
    if not val_report["is_valid"]:
        raise ValueError(
            f"Dataset failed validation: {val_report['range_violations']} "
            f"{val_report['categorical_violations']}"
        )

    # 2. Apply Feature Engineering
    df_processed = engineer_all_features(df)

    # 3. Apply Categorical Encoding
    df_processed = encode_categorical_features(df_processed)

    return df_processed


def get_model_feature_columns(df: pd.DataFrame) -> List[str]:
    """Retrieve the exact list of valid in-season predictor columns for ML models.

    Strictly excludes:
    - Target: Actual_Final_Yield_t_ha
    - Leakage: Predicted_Final_Yield_t_ha
    - Identifiers: Farm_ID
    - Raw unencoded string columns: District, Seed_Variety, Soil_Type, Growth_Stage,
      Pest_Severity, Disease_Severity, Planting_Date

    Parameters
    ----------
    df : pd.DataFrame
        Processed dataframe.

    Returns
    -------
    List[str]
        Clean list of predictive feature column names.
    """
    excluded_columns = set(
        [TARGET_COLUMN]
        + LEAKAGE_COLUMNS
        + ID_COLUMNS
        + RAW_TEMPORAL_COLUMNS
        + NOMINAL_CATEGORICAL_COLUMNS
        + list(ORDINAL_MAPPINGS.keys())
    )

    return [c for c in df.columns if c not in excluded_columns]


def get_feature_matrix_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract model training matrix X and target vector y.

    Parameters
    ----------
    df : pd.DataFrame
        Processed dataframe.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        X: Feature matrix with in-season predictors.
        y: Target series (Actual_Final_Yield_t_ha).
    """
    feature_cols = get_model_feature_columns(df)
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y
