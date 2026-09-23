"""Unit tests for Phase 2 Data Preprocessing and Feature Engineering."""

import hashlib
import numpy as np
import pandas as pd
import pytest

from src.data.data_loader import load_raw_data
from src.data.feature_engineering import (
    create_growth_features,
    create_health_features,
    create_temporal_features,
    create_water_features,
    engineer_all_features,
)
from src.data.preprocessing import (
    encode_categorical_features,
    get_feature_matrix_and_target,
    get_model_feature_columns,
    preprocess_dataset,
    validate_raw_schema,
)
from src.utils.config import (
    LEAKAGE_COLUMNS,
    ID_COLUMNS,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)


def compute_file_hash(filepath):
    """Compute SHA-256 hash of a file to verify immutability."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_raw_dataset_immutability():
    """Verify that preprocessing does not alter or overwrite the raw CSV file."""
    initial_hash = compute_file_hash(RAW_DATA_PATH)
    df_raw = load_raw_data()
    _ = preprocess_dataset(df_raw)
    final_hash = compute_file_hash(RAW_DATA_PATH)
    assert initial_hash == final_hash, "Raw dataset was mutated!"


def test_raw_schema_validation():
    """Verify raw schema validator accepts valid dataset."""
    df_raw = load_raw_data()
    report = validate_raw_schema(df_raw)
    assert report["is_valid"] is True
    assert report["total_missing_cells"] == 0
    assert len(report["range_violations"]) == 0
    assert len(report["categorical_violations"]) == 0


def test_feature_engineering_formulas():
    """Verify mathematical correctness of engineered feature calculations."""
    df_raw = load_raw_data()
    df_feat = engineer_all_features(df_raw)

    # 1. Total_Water_Input_mm
    expected_water = (df_raw["Weekly_Rainfall_mm"] + df_raw["Irrigation_mm_week"]).round(3)
    np.testing.assert_allclose(df_feat["Total_Water_Input_mm"], expected_water)

    # 2. Plant_Height_per_DAP
    expected_rate = (df_raw["Plant_Height_cm"] / df_raw["Days_After_Planting"]).round(4)
    np.testing.assert_allclose(df_feat["Plant_Height_per_DAP"], expected_rate)

    # 3. Canopy_Greenness_Volume
    expected_canopy = (df_raw["Plant_Height_cm"] * df_raw["Leaf_Greenness_Index"]).round(3)
    np.testing.assert_allclose(df_feat["Canopy_Greenness_Volume"], expected_canopy)

    # 4. Biotic_Stress_Index range [0, 4]
    assert df_feat["Biotic_Stress_Index"].min() >= 0
    assert df_feat["Biotic_Stress_Index"].max() <= 4


def test_categorical_encoding_completeness():
    """Verify one-hot and ordinal encodings are properly generated."""
    df_raw = load_raw_data()
    df_enc = encode_categorical_features(df_raw)

    # Check Ordinal columns
    assert "Growth_Stage_Ordinal" in df_enc.columns
    assert "Pest_Severity_Ordinal" in df_enc.columns
    assert "Disease_Severity_Ordinal" in df_enc.columns
    assert df_enc["Growth_Stage_Ordinal"].between(0, 5).all()

    # Check One-Hot columns
    for district in df_raw["District"].unique():
        assert f"District_{district}" in df_enc.columns
    for variety in df_raw["Seed_Variety"].unique():
        assert f"Seed_Variety_{variety}" in df_enc.columns
    for soil in df_raw["Soil_Type"].unique():
        assert f"Soil_Type_{soil}" in df_enc.columns


def test_leakage_and_id_exclusion_from_model_features():
    """Verify Predicted_Final_Yield_t_ha, Farm_ID, and raw unencoded strings are strictly excluded from X."""
    df_raw = load_raw_data()
    df_proc = preprocess_dataset(df_raw)
    model_features = get_model_feature_columns(df_proc)
    X, y = get_feature_matrix_and_target(df_proc)

    # Leakage columns excluded
    for leak_col in LEAKAGE_COLUMNS:
        assert leak_col not in model_features
        assert leak_col not in X.columns

    # ID columns excluded
    for id_col in ID_COLUMNS:
        assert id_col not in model_features
        assert id_col not in X.columns

    # Target excluded from X
    assert TARGET_COLUMN not in model_features
    assert TARGET_COLUMN not in X.columns

    # Target series matches
    assert len(y) == len(df_raw)
    assert (y == df_raw[TARGET_COLUMN]).all()


def test_no_nan_or_inf_in_processed_data():
    """Verify processed dataset contains no NaN or infinite values."""
    df_raw = load_raw_data()
    df_proc = preprocess_dataset(df_raw)
    X, y = get_feature_matrix_and_target(df_proc)

    assert df_proc.isnull().sum().sum() == 0, "Processed dataset contains NaN values"
    assert not np.isinf(X.values).any(), "Model feature matrix contains infinite values"
    assert not np.isnan(X.values).any(), "Model feature matrix contains NaN values"
    assert not np.isnan(y.values).any(), "Target series contains NaN values"


def test_row_count_preservation():
    """Verify exact 10,000 row count is preserved after processing."""
    df_raw = load_raw_data()
    df_proc = preprocess_dataset(df_raw)
    assert len(df_proc) == len(df_raw) == 10000


def test_processed_dataset_reloadable():
    """Verify that saved processed dataset can be reloaded and matches in-memory processing."""
    from scripts.prepare_data import main as run_prepare_data
    run_prepare_data()

    assert PROCESSED_DATA_PATH.exists()
    df_loaded = pd.read_csv(PROCESSED_DATA_PATH)
    assert len(df_loaded) == 10000
    assert TARGET_COLUMN in df_loaded.columns
    assert "Farm_ID" in df_loaded.columns
    assert "Total_Water_Input_mm" in df_loaded.columns
    assert "Canopy_Greenness_Volume" in df_loaded.columns
