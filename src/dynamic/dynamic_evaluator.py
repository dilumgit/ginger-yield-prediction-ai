"""Dynamic Model Evaluation & Stage-Checkpoint Performance Module (Phase 4).

Evaluates in-season prediction accuracy across chronological checkpoints
(Days After Planting / Growth Stages) on the hold-out test partition to quantify
prediction error reduction as in-season information accumulates.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.data.preprocessing import get_feature_matrix_and_target
from src.models.evaluation import compute_regression_metrics
from src.models.predictor import GingerYieldPredictor
from src.utils.config import (
    BEST_MODEL_PATH,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    RESULTS_DIR,
    TEST_SIZE,
)


class DynamicEvaluator:
    """Evaluates yield forecasting accuracy across in-season developmental horizons."""

    CHECKPOINTS = [
        {"name": "Stage 1: Sprouting (DAP <= 45)", "dap_min": 1, "dap_max": 45},
        {"name": "Stage 2: Early Vegetative (DAP 46-75)", "dap_min": 46, "dap_max": 75},
        {"name": "Stage 3: Vegetative (DAP 76-120)", "dap_min": 76, "dap_max": 120},
        {"name": "Stage 4: Rhizome Initiation (DAP 121-180)", "dap_min": 121, "dap_max": 180},
        {"name": "Stage 5: Rhizome Dev & Maturity (DAP > 180)", "dap_min": 181, "dap_max": 365},
    ]

    def __init__(self, model_path: Optional[str] = None):
        self.predictor = GingerYieldPredictor(model_path or BEST_MODEL_PATH)

    def evaluate_test_checkpoints(
        self, df_processed: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Evaluate holdout test partition across in-season growth checkpoints.

        Parameters
        ----------
        df_processed : pd.DataFrame, optional
            Processed dataframe. If None, loaded from `PROCESSED_DATA_PATH`.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            1. checkpoint_summary_df: Summary metrics per stage checkpoint.
            2. test_detailed_df: Detailed predictions per test observation.
        """
        if df_processed is None:
            df_processed = pd.read_csv(PROCESSED_DATA_PATH)

        X, y = get_feature_matrix_and_target(df_processed)
        groups = df_processed["Farm_ID"]

        # Re-partition exact hold-out test set
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        _, test_idx = next(gss.split(X, y, groups=groups))

        df_test = df_processed.iloc[test_idx].copy()
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        # Generate model predictions on test set
        y_preds = self.predictor.predict(X_test)
        df_test["Predicted_Yield_t_ha"] = np.round(y_preds, 4)
        df_test["Prediction_Error_t_ha"] = np.round(y_preds - y_test.values, 4)
        df_test["Absolute_Error_t_ha"] = np.round(np.abs(y_preds - y_test.values), 4)

        records = []
        for cp in self.CHECKPOINTS:
            mask = (df_test["Days_After_Planting"] >= cp["dap_min"]) & (df_test["Days_After_Planting"] <= cp["dap_max"])
            sub_test = df_test[mask]

            if len(sub_test) > 0:
                metrics = compute_regression_metrics(
                    y_true=sub_test["Actual_Final_Yield_t_ha"].values,
                    y_pred=sub_test["Predicted_Yield_t_ha"].values,
                )
                records.append({
                    "Checkpoint_Name": cp["name"],
                    "DAP_Min": cp["dap_min"],
                    "DAP_Max": cp["dap_max"],
                    "Sample_Count": len(sub_test),
                    "Unique_Farms": int(sub_test["Farm_ID"].nunique()),
                    "MAE_t_ha": metrics["MAE"],
                    "RMSE_t_ha": metrics["RMSE"],
                    "R2_Score": metrics["R2"],
                    "MAPE_pct": metrics["MAPE"],
                    "Mean_Abs_Error_kg_ha": round(metrics["MAE"] * 1000.0, 1),
                })

        summary_df = pd.DataFrame(records)
        return summary_df, df_test
