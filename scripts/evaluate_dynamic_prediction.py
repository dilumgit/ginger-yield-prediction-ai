"""Dynamic In-Season Model Evaluation Script (Phase 4).

Evaluates prediction accuracy across developmental growth stage checkpoints on the
untouched hold-out test set to measure empirical performance gains from in-season updating.

Usage:
    python scripts/evaluate_dynamic_prediction.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.dynamic.dynamic_evaluator import DynamicEvaluator
from src.utils.config import RESULTS_DIR, RESULTS_PLOTS_DIR


def generate_improvement_plot(summary_df: pd.DataFrame, output_path: Path):
    """Generate plot showing error reduction across progressive developmental checkpoints."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    plt.figure(figsize=(9, 5))
    x_labels = [f"Stage {i+1}\n(DAP {row['DAP_Min']}-{row['DAP_Max']})" for i, row in summary_df.iterrows()]

    plt.plot(x_labels, summary_df["MAE_t_ha"], marker="o", markersize=8, color="#1f77b4", linewidth=2.2, label="MAE (t/ha)")
    plt.plot(x_labels, summary_df["RMSE_t_ha"], marker="s", markersize=8, color="#d62728", linewidth=2.2, label="RMSE (t/ha)")

    plt.xlabel("Developmental Growth Checkpoint (Days After Planting)", fontsize=11, fontweight="bold")
    plt.ylabel("Prediction Error (t/ha)", fontsize=11, fontweight="bold")
    plt.title("Empirical Prediction Error Progression Across In-Season Checkpoints\n(Untouched Hold-Out Test Set: n=1,977 across 200 Farms)", fontsize=12, pad=12)

    for i, row in summary_df.iterrows():
        plt.annotate(f"MAE: {row['MAE_t_ha']:.3f}\n(R²: {row['R2_Score']:.3f})",
                     (x_labels[i], row["MAE_t_ha"]),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha="center",
                     fontsize=8.5,
                     fontweight="bold")

    plt.ylim(0, max(summary_df["RMSE_t_ha"]) + 0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    print("=" * 80)
    print("  PHASE 4: DYNAMIC IN-SEASON STAGE CHECKPOINT EVALUATION")
    print("  Module: Research Methods and Scientific Writing (IT41012)")
    print("=" * 80)

    # 1. Run Dynamic Evaluator on Test Partition
    print("\n[1/3] Benchmarking predictive accuracy across in-season growth checkpoints...")
    evaluator = DynamicEvaluator()
    summary_df, detailed_df = evaluator.evaluate_test_checkpoints()

    # 2. Save Results CSV
    out_csv = RESULTS_DIR / "dynamic_evaluation.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"\n[2/3] Exported dynamic evaluation metrics to: {out_csv.resolve()}")

    # 3. Generate Plot
    dynamic_plots_dir = RESULTS_PLOTS_DIR / "dynamic"
    plot_path = dynamic_plots_dir / "dynamic_prediction_improvement.png"
    generate_improvement_plot(summary_df, plot_path)
    print(f"\n[3/3] Generated dynamic error improvement plot: {plot_path.resolve()}")

    # 4. Display Checkpoint Performance Table
    print("\n" + "=" * 80)
    print("  DYNAMIC IN-SEASON CHECKPOINT PERFORMANCE (HOLDOUT TEST SET)")
    print("=" * 80)
    print(summary_df[["Checkpoint_Name", "Sample_Count", "Unique_Farms", "MAE_t_ha", "RMSE_t_ha", "R2_Score", "MAPE_pct"]].to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
