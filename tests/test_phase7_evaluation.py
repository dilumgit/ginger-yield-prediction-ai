"""Unit & Integration Tests for Phase 7 Scientific Evaluation.

Verifies:
1. Static vs dynamic evaluation output schemas and metric validity.
2. In-season growth stage checkpoint evaluation metrics (5 developmental horizons).
3. Dynamic update error trajectory analysis and paired statistical calculations.
4. Irrigation information value sensitivity analysis.
5. TreeSHAP additive reconstruction property on hold-out test set (<1e-4).
6. Formal 6-point temporal leakage and target isolation audit compliance (100%).
7. Farm-level generalization error distribution calculation (200 distinct test farms).
8. Robustness simulation scenarios under sparse and missing updates.
9. Research acceptance matrix integrity (15 criteria).
10. End-to-end execution of scripts/run_phase7_evaluation.py.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.evaluation.phase7_evaluator import Phase7ResearchEvaluator


@pytest.fixture(scope="module")
def evaluator():
    """Module-level fixture providing an instantiated Phase7ResearchEvaluator."""
    return Phase7ResearchEvaluator()


def test_1_static_vs_dynamic_evaluation(evaluator):
    """Verify static vs dynamic comparison dataframe structure and metrics."""
    df = evaluator.evaluate_static_vs_dynamic()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 7  # 3 baselines + 5 checkpoints
    assert "MAE_t_ha" in df.columns
    assert "RMSE_t_ha" in df.columns
    assert "R2_Score" in df.columns
    assert "MAE_kg_ha" in df.columns
    assert all(df["MAE_t_ha"] > 0)
    assert all(df["MAE_kg_ha"] > 0)


def test_2_stage_checkpoints_evaluation(evaluator):
    """Verify in-season growth stage checkpoint evaluation over 5 horizons."""
    df_cp = evaluator.evaluate_stage_checkpoints()
    assert isinstance(df_cp, pd.DataFrame)
    assert len(df_cp) == 5
    assert "Stage_Name" in df_cp.columns
    assert "N_Observations" in df_cp.columns
    assert "N_Farms" in df_cp.columns
    assert df_cp["N_Observations"].sum() == len(evaluator.df_test)
    assert all(df_cp["N_Farms"] > 0)
    assert all(df_cp["R2_Score"] > 0.65)
    assert all(df_cp["MAE_t_ha"] < 0.75)


def test_3_dynamic_update_value(evaluator):
    """Verify dynamic update trajectory evaluation and paired statistical tests."""
    df_updates, stats_dict = evaluator.evaluate_dynamic_update_value()
    assert isinstance(df_updates, pd.DataFrame)
    assert len(df_updates) > 0
    assert "Delta_t_ha" in df_updates.columns
    assert "Error_Change_t_ha" in df_updates.columns
    assert "paired_t_pvalue" in stats_dict
    assert "wilcoxon_pvalue" in stats_dict


def test_4_irrigation_information_value(evaluator):
    """Verify irrigation information value sensitivity calculation."""
    df_irr = evaluator.evaluate_irrigation_information_value()
    assert isinstance(df_irr, pd.DataFrame)
    assert len(df_irr) > 0
    assert "Forecast_Without_Irrigation_t_ha" in df_irr.columns
    assert "Forecast_With_Irrigation_t_ha" in df_irr.columns
    assert "Forecast_Delta_t_ha" in df_irr.columns


def test_5_explainability_validation(evaluator):
    """Verify TreeSHAP additive decomposition on hold-out test samples."""
    df_imp, shap_summary = evaluator.validate_explainability()
    assert isinstance(df_imp, pd.DataFrame)
    assert len(df_imp) == 36
    assert shap_summary["additive_identity_verified"] is True
    assert shap_summary["max_additive_reconstruction_error_t_ha"] < 1e-4


def test_6_temporal_leakage_audit(evaluator):
    """Verify 100% compliance across all 6 audit points of temporal leakage guard."""
    df_leakage = evaluator.audit_temporal_leakage()
    assert isinstance(df_leakage, pd.DataFrame)
    assert len(df_leakage) == 6
    assert all(df_leakage["Compliant"] == True), "Temporal or target leakage detected!"


def test_7_farm_level_generalization(evaluator):
    """Verify farm-level error distribution across 200 distinct test farms."""
    df_farms, farm_summary = evaluator.evaluate_farm_level_generalization()
    assert isinstance(df_farms, pd.DataFrame)
    assert farm_summary["n_test_farms"] == 200
    assert farm_summary["mean_farm_mae_t_ha"] > 0.0
    assert farm_summary["median_farm_mae_t_ha"] > 0.0


def test_8_robustness_scenarios(evaluator):
    """Verify robustness to missing update frequencies without failure."""
    df_robust = evaluator.evaluate_robustness_scenarios()
    assert isinstance(df_robust, pd.DataFrame)
    assert len(df_robust) == 5
    assert all(df_robust["State_Reconstruction_Success"] == True)
    assert all(df_robust["No_Data_Fabrication_Confirmed"] == True)


def test_9_acceptance_matrix(evaluator):
    """Verify 15-point research acceptance matrix completeness."""
    df_matrix = evaluator.generate_acceptance_matrix()
    assert isinstance(df_matrix, pd.DataFrame)
    assert len(df_matrix) == 15
    assert all(df_matrix["Validation_Status"] == "VERIFIED / ACCEPTED")


def test_10_phase7_master_script_execution():
    """Verify scripts/run_phase7_evaluation.py runs and generates all deliverables."""
    from scripts.run_phase7_evaluation import main as run_p7
    run_p7()

    p7_dir = Path("results/phase7")
    plots_dir = Path("results/plots/phase7")

    assert (p7_dir / "final_evaluation_report.txt").exists()
    assert (p7_dir / "static_vs_dynamic_comparison.csv").exists()
    assert (p7_dir / "checkpoint_evaluation.csv").exists()
    assert (p7_dir / "dynamic_update_analysis.csv").exists()
    assert (p7_dir / "irrigation_information_analysis.csv").exists()
    assert (p7_dir / "explainability_validation.csv").exists()
    assert (p7_dir / "leakage_audit.csv").exists()
    assert (p7_dir / "robustness_evaluation.csv").exists()
    assert (p7_dir / "farm_level_evaluation.csv").exists()
    assert (p7_dir / "research_acceptance_matrix.csv").exists()

    assert (plots_dir / "static_vs_dynamic_performance.png").exists()
    assert (plots_dir / "checkpoint_mae_rmse.png").exists()
    assert (plots_dir / "prediction_error_over_dap.png").exists()
    assert (plots_dir / "dynamic_update_error_change.png").exists()
    assert (plots_dir / "farm_level_error_distribution.png").exists()
    assert (plots_dir / "explainability_validation.png").exists()
    assert (plots_dir / "prediction_accuracy_by_growth_stage.png").exists()
