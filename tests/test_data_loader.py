"""Unit tests for Phase 1 data loading and inspection functionality."""

import pytest
import pandas as pd

from src.data.data_loader import load_raw_data
from src.data.data_inspection import (
    inspect_dataset_structure,
    inspect_data_quality,
    inspect_target_variable,
    inspect_dynamic_feasibility,
)
from src.utils.config import TARGET_COLUMN, RAW_DATA_PATH


def test_raw_data_file_exists():
    """Verify that the raw dataset file exists."""
    assert RAW_DATA_PATH.exists(), f"Raw dataset not found at {RAW_DATA_PATH}"


def test_load_raw_data_returns_dataframe():
    """Verify data loader loads expected shape and dataframe type."""
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10000
    assert len(df.columns) == 22
    assert TARGET_COLUMN in df.columns


def test_data_quality_no_missing_values():
    """Verify data quality checks return zero missing values."""
    df = load_raw_data()
    quality = inspect_data_quality(df)
    assert quality["total_missing_cells"] == 0
    assert quality["duplicate_rows_count"] == 0


def test_target_variable_distribution():
    """Verify target variable summary statistics."""
    df = load_raw_data()
    target_stats = inspect_target_variable(df)
    assert target_stats["min"] == 4.0
    assert target_stats["max"] == 21.9
    assert 13.0 < target_stats["mean"] < 14.0


def test_dynamic_feasibility_detection():
    """Verify dynamic feasibility analysis correctly flags cross-sectional properties."""
    df = load_raw_data()
    feasibility = inspect_dynamic_feasibility(df)
    assert feasibility["has_farm_id"] is True
    assert feasibility["unique_farms"] == 1000
    assert feasibility["pct_farms_with_varying_final_yield"] > 90.0
