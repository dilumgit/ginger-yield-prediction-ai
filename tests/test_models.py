"""Unit tests for Phase 3 Model Development, Training, Prediction, and Evaluation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupShuffleSplit

from src.data.preprocessing import get_feature_matrix_and_target, get_model_feature_columns
from src.models.evaluation import (
    compute_mape,
    compute_regression_metrics,
    compute_residuals,
)
from src.models.predictor import GingerYieldPredictor
from src.models.train_models import build_candidate_models, train_and_save_model
from src.utils.config import (
    BEST_MODEL_PATH,
    ID_COLUMNS,
    LEAKAGE_COLUMNS,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)


def test_processed_dataset_exists():
    """Verify that the Phase 2 processed dataset is available."""
    assert PROCESSED_DATA_PATH.exists(), f"Missing processed dataset at {PROCESSED_DATA_PATH}"


def test_target_leakage_and_id_exclusion():
    """Verify that Target, Leakage, and ID columns are strictly excluded from feature matrix X."""
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X, y = get_feature_matrix_and_target(df)
    feature_names = get_model_feature_columns(df)

    assert TARGET_COLUMN not in X.columns
    assert TARGET_COLUMN not in feature_names

    for leak in LEAKAGE_COLUMNS:
        assert leak not in X.columns
        assert leak not in feature_names

    for id_col in ID_COLUMNS:
        assert id_col not in X.columns
        assert id_col not in feature_names

    assert len(X.columns) == len(feature_names) == 36
    assert len(y) == len(df) == 10000


def test_train_test_group_isolation():
    """Verify zero overlap between training and testing Farm_IDs."""
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X, y = get_feature_matrix_and_target(df)
    groups = df["Farm_ID"]

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    train_farms = set(groups.iloc[train_idx])
    test_farms = set(groups.iloc[test_idx])

    assert len(train_farms.intersection(test_farms)) == 0, "Farm_ID overlap detected between train and test!"
    assert len(train_farms) == 800
    assert len(test_farms) == 200


def test_regression_metrics_calculation():
    """Verify accuracy and numerical sanity of metric calculations."""
    y_true = np.array([10.0, 15.0, 20.0, 25.0])
    y_pred = np.array([11.0, 14.0, 22.0, 23.0])

    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["MAE"] == pytest.approx(1.5, 0.01)
    assert metrics["RMSE"] == pytest.approx(1.5811, 0.01)
    assert metrics["R2"] > 0.85
    assert metrics["MAPE"] > 0.0

    residuals, abs_residuals = compute_residuals(y_true, y_pred)
    np.testing.assert_allclose(residuals, [1.0, -1.0, 2.0, -2.0])
    np.testing.assert_allclose(abs_residuals, [1.0, 1.0, 2.0, 2.0])


def test_safe_mape_with_zeros():
    """Verify that MAPE handles near-zero and zero values safely without crashing."""
    y_true = np.array([0.0, 10.0, 20.0])
    y_pred = np.array([1.0, 11.0, 19.0])
    mape = compute_mape(y_true, y_pred)
    assert not np.isnan(mape)
    assert not np.isinf(mape)


def test_candidate_models_instantiation():
    """Verify all 6 candidate regression models instantiate with proper random state."""
    models = build_candidate_models(random_state=42)
    assert "Dummy (Mean)" in models
    assert "Ridge" in models
    assert "Random Forest" in models
    assert "XGBoost" in models
    assert "LightGBM" in models
    assert "CatBoost" in models


def test_predictor_and_model_artifact_loading():
    """Verify model training, saving, and GingerYieldPredictor inference."""
    if not BEST_MODEL_PATH.exists():
        from scripts.train import main as run_training
        run_training()

    assert BEST_MODEL_PATH.exists(), f"Best model not saved at {BEST_MODEL_PATH}"

    predictor = GingerYieldPredictor(BEST_MODEL_PATH)
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X, y = get_feature_matrix_and_target(df)

    # Test single row prediction
    single_pred = predictor.predict(X.iloc[0:1])
    assert isinstance(single_pred, np.ndarray)
    assert len(single_pred) == 1
    assert not np.isnan(single_pred[0])
    assert not np.isinf(single_pred[0])
    assert single_pred[0] > 0.0

    # Test batch predictions
    batch_preds = predictor.predict(X.iloc[:50])
    assert len(batch_preds) == 50
    assert not np.isnan(batch_preds).any()
    assert not np.isinf(batch_preds).any()
