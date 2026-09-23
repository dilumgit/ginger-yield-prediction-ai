"""Model Inference and Prediction Module for Ginger Yield Prediction Research.

Provides a clean, reusable predictor class to load serialized model artifacts,
validate input schemas, and produce yield forecasts (t/ha).
Supports direct dynamic farm state prediction and prediction history tracking.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from src.models.evaluation import compute_regression_metrics, compute_residuals
from src.utils.config import BEST_MODEL_PATH


class GingerYieldPredictor:
    """Predictor class for loading trained models and performing inference."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None, model_version: str = "Phase3_Static_Baseline"):
        """Initialize predictor by loading trained model from disk.

        Parameters
        ----------
        model_path : str or Path, optional
            Path to serialized joblib model. If None, defaults to `BEST_MODEL_PATH`.
        model_version : str, default="Phase3_Static_Baseline"
            Version tag identifier for prediction history provenance.
        """
        self.model_path = Path(model_path) if model_path is not None else BEST_MODEL_PATH
        self.model_version = model_version
        self.model = self._load_model()
        self.feature_names: Optional[List[str]] = self._extract_feature_names()

    def _load_model(self) -> Any:
        """Load joblib artifact from disk with error validation."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at '{self.model_path.resolve()}'.\n"
                f"Please run 'python scripts/train.py' first to train and save models."
            )
        try:
            return joblib.load(self.model_path)
        except Exception as e:
            raise ValueError(
                f"Failed to deserialize model artifact from '{self.model_path.resolve()}': {e}"
            ) from e

    def _extract_feature_names(self) -> Optional[List[str]]:
        """Extract expected input feature names from fitted model if available."""
        if hasattr(self.model, "feature_names_in_"):
            return list(self.model.feature_names_in_)
        elif hasattr(self.model, "feature_names_"):
            return list(self.model.feature_names_)
        elif hasattr(self.model, "named_steps") and hasattr(self.model.named_steps.get("regressor", None), "feature_names_in_"):
            return list(self.model.named_steps["regressor"].feature_names_in_)
        return None

    def predict(
        self, X: Union[pd.DataFrame, pd.Series, Dict[str, Any]]
    ) -> np.ndarray:
        """Generate yield predictions in metric tons per hectare (t/ha).

        Parameters
        ----------
        X : pd.DataFrame, pd.Series, or Dict[str, Any]
            Feature inputs containing all required model predictors.

        Returns
        -------
        np.ndarray
            Predicted final ginger yield in t/ha.
        """
        # Convert Series or Dict to single-row DataFrame
        if isinstance(X, dict):
            X_df = pd.DataFrame([X])
        elif isinstance(X, pd.Series):
            X_df = pd.DataFrame([X])
        elif isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            raise TypeError(f"Unsupported input type for X: {type(X)}")

        # Align columns if feature names are known
        if self.feature_names is not None:
            missing_cols = [c for c in self.feature_names if c not in X_df.columns]
            if missing_cols:
                raise ValueError(
                    f"Input is missing required model features: {missing_cols}"
                )
            X_aligned = X_df[self.feature_names]
        else:
            from src.data.preprocessing import get_model_feature_columns
            feat_cols = get_model_feature_columns(X_df)
            X_aligned = X_df[feat_cols]

        predictions = self.model.predict(X_aligned)
        return np.asarray(predictions, dtype=float)

    def predict_farm_state(
        self,
        farm_state_mgr: Any,
        as_of_date: Optional[str] = None,
        history_tracker: Optional[Any] = None,
    ) -> float:
        """Predict yield dynamically from a FarmStateManager instance.

        Parameters
        ----------
        farm_state_mgr : FarmStateManager
            Farm state manager holding observations and irrigation log.
        as_of_date : str, optional
            Cutoff date string (YYYY-MM-DD). If None, uses latest observation.
        history_tracker : PredictionHistory, optional
            If provided, automatically logs the prediction and computes the delta.

        Returns
        -------
        float
            Predicted final yield in t/ha.
        """
        state_dict = farm_state_mgr.reconstruct_current_state(as_of_date=as_of_date)
        pred_arr = self.predict(state_dict)
        pred_val = float(round(pred_arr[0], 4))

        if history_tracker is not None:
            history_tracker.record_prediction(
                farm_id=farm_state_mgr.profile.farm_id,
                days_after_planting=state_dict.get("Days_After_Planting", 0),
                growth_stage=state_dict.get("Growth_Stage", "Unknown"),
                predicted_yield_t_ha=pred_val,
                model_version=self.model_version,
                state_snapshot=state_dict,
            )

        return pred_val

    def evaluate_test_data(
        self, X: pd.DataFrame, y_true: pd.Series, farm_ids: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Generate test predictions with residuals and performance metrics.

        Parameters
        ----------
        X : pd.DataFrame
            Predictor feature matrix.
        y_true : pd.Series
            Actual ground truth target series (Actual_Final_Yield_t_ha).
        farm_ids : pd.Series, optional
            Farm_ID series for identification and traceability.

        Returns
        -------
        Tuple[pd.DataFrame, Dict[str, float]]
            1. Predictions DataFrame with actual, predicted, and residual columns.
            2. Dictionary of regression performance metrics (MAE, RMSE, R2, MAPE).
        """
        y_pred = self.predict(X)
        metrics = compute_regression_metrics(y_true, y_pred)
        residuals, abs_residuals = compute_residuals(y_true, y_pred)

        results_df = pd.DataFrame({
            "Actual_Final_Yield_t_ha": np.round(y_true.values, 4),
            "Predicted_Yield_t_ha": np.round(y_pred, 4),
            "Prediction_Error_t_ha": np.round(residuals, 4),
            "Absolute_Error_t_ha": np.round(abs_residuals, 4),
        }, index=X.index)

        if farm_ids is not None:
            results_df.insert(0, "Farm_ID", farm_ids.values)

        return results_df, metrics
