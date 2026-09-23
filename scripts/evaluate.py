"""Standalone Model Evaluation Script (Phase 3).

Evaluates the saved best model artifact (models/best_model.joblib)
on the hold-out test set and prints a performance breakdown.

Usage:
    python scripts/evaluate.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.data.preprocessing import get_feature_matrix_and_target
from src.models.predictor import GingerYieldPredictor
from src.utils.config import BEST_MODEL_PATH, PROCESSED_DATA_PATH, RANDOM_STATE, TEST_SIZE


def main():
    print("=" * 80)
    print("  GINGER YIELD PREDICTION RESEARCH - MODEL EVALUATION (PHASE 3)")
    print("  Module: Research Methods and Scientific Writing (IT41012)")
    print("=" * 80)

    # 1. Load Processed Dataset
    print("\n[1/3] Loading processed dataset...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X, y = get_feature_matrix_and_target(df)
    groups = df["Farm_ID"]

    # 2. Partition identical hold-out test set
    print("\n[2/3] Extracting untouched hold-out test partition...")
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    _, test_idx = next(gss.split(X, y, groups=groups))
    X_test, y_test, groups_test = X.iloc[test_idx], y.iloc[test_idx], groups.iloc[test_idx]

    print(f"      Hold-out test samples: {len(X_test):,} across {groups_test.nunique()} unique farms")

    # 3. Load Best Model and Evaluate
    print("\n[3/3] Evaluating serialized best model artifact...")
    predictor = GingerYieldPredictor(BEST_MODEL_PATH)
    pred_df, metrics = predictor.evaluate_test_data(X_test, y_test, farm_ids=groups_test)

    print("\n" + "=" * 80)
    print("  FINAL EVALUATION PERFORMANCE ON UNTOUCHED TEST SET")
    print("=" * 80)
    print(f"  - Mean Absolute Error (MAE):     {metrics['MAE']:.4f} t/ha")
    print(f"  - Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f} t/ha")
    print(f"  - Coefficient of Determination:   {metrics['R2']:.4f}")
    print(f"  - Mean Absolute Percentage Error: {metrics['MAPE']:.2f}%")
    print(f"  - Average Actual Harvest Yield:   {y_test.mean():.2f} t/ha")
    print(f"  - Average Predicted Yield:        {pred_df['Predicted_Yield_t_ha'].mean():.2f} t/ha")
    print(f"  - Mean Residual Bias:             {pred_df['Prediction_Error_t_ha'].mean():.4f} t/ha")
    print("=" * 80)


if __name__ == "__main__":
    main()
