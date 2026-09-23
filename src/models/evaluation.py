"""Model Evaluation Module for Ginger Yield Prediction Research.

Implements pure, standardized regression metrics and group-aware cross-validation
evaluation functions for crop yield forecasting.

Metrics:
- Mean Absolute Error (MAE, t/ha)
- Root Mean Squared Error (RMSE, t/ha)
- Coefficient of Determination (R²)
- Mean Absolute Percentage Error (MAPE, %)
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
    """Calculate Mean Absolute Percentage Error (MAPE) safely.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth actual values.
    y_pred : np.ndarray
        Predicted values.
    epsilon : float, default=1e-6
        Small positive constant to prevent division by zero.

    Returns
    -------
    float
        MAPE in percentage (%).
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    # Avoid zero division
    denominator = np.where(np.abs(y_true_arr) < epsilon, epsilon, np.abs(y_true_arr))
    mape = np.mean(np.abs((y_true_arr - y_pred_arr) / denominator)) * 100.0
    return float(mape)


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute comprehensive regression evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth target values (Actual_Final_Yield_t_ha).
    y_pred : array-like
        Predicted values.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys: 'MAE', 'RMSE', 'R2', 'MAPE'.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(root_mean_squared_error(y_t, y_p))
    r2 = float(r2_score(y_t, y_p))
    mape = float(compute_mape(y_t, y_p))

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "MAPE": round(mape, 4),
    }


def compute_residuals(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate signed error residuals and absolute errors.

    Parameters
    ----------
    y_true : array-like
        Ground truth actual values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (residuals, abs_residuals)
        residuals = y_pred - y_true
        abs_residuals = |y_pred - y_true|
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    residuals = y_p - y_t
    abs_residuals = np.abs(residuals)
    return residuals, abs_residuals


def evaluate_group_cross_validation(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
    model_name: str = "Model",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute rigorous group-aware cross validation and collect fold-by-fold metrics.

    Parameters
    ----------
    model : Estimator
        Scikit-learn compatible regression estimator or pipeline.
    X : pd.DataFrame
        Training feature matrix.
    y : pd.Series
        Training target vector.
    groups : pd.Series
        Group identifier (Farm_ID).
    n_splits : int, default=5
        Number of cross-validation folds.
    model_name : str
        Name identifier for reporting.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        1. DataFrame of fold-level metrics (columns: Model, Fold, MAE, RMSE, R2, MAPE)
        2. Dictionary summarizing mean and standard deviation for each metric.
    """
    gkf = GroupKFold(n_splits=n_splits)
    fold_records = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Clone and fit model on fold training split
        from sklearn.base import clone
        fold_model = clone(model)
        fold_model.fit(X_tr, y_tr)

        y_pred = fold_model.predict(X_val)
        metrics = compute_regression_metrics(y_val, y_pred)

        fold_records.append({
            "Model": model_name,
            "Fold": fold_idx,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"],
            "MAPE": metrics["MAPE"],
            "Train_Samples": len(X_tr),
            "Val_Samples": len(X_val),
            "Val_Farms": int(groups.iloc[val_idx].nunique()),
        })

    fold_df = pd.DataFrame(fold_records)

    summary = {
        "Model": model_name,
        "CV_MAE_mean": round(float(fold_df["MAE"].mean()), 4),
        "CV_MAE_std": round(float(fold_df["MAE"].std()), 4),
        "CV_RMSE_mean": round(float(fold_df["RMSE"].mean()), 4),
        "CV_RMSE_std": round(float(fold_df["RMSE"].std()), 4),
        "CV_R2_mean": round(float(fold_df["R2"].mean()), 4),
        "CV_R2_std": round(float(fold_df["R2"].std()), 4),
        "CV_MAPE_mean": round(float(fold_df["MAPE"].mean()), 4),
        "CV_MAPE_std": round(float(fold_df["MAPE"].std()), 4),
    }

    return fold_df, summary
