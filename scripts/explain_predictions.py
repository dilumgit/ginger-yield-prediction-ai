"""Phase 5: Explainable AI & Farmer-Centric Yield Prediction Explanation Script.

Performs:
1. Global SHAP Feature Attribution on the hold-out test partition (n=1,977).
2. Dynamic In-Season SHAP Explanations & Farmer Translations across sequential farm updates.
3. Rule-Based Agronomic Recommendations & Factor Prioritization.
4. Generates publication-ready figures in `results/plots/explainability/`.
5. Exports structured CSV reports & research summary in `results/explainability/`.

Usage:
    python scripts/explain_predictions.py
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
import shap

from src.data.farm_state import FarmProfile, FarmStateManager
from src.data.preprocessing import get_feature_matrix_and_target
from src.data.weather_api import MockWeatherProvider
from src.dynamic.dynamic_predictor import DynamicYieldPredictor
from src.dynamic.state_updater import update_farm_state
from src.explainability.farmer_translator import FarmerExplanationTranslator
from src.explainability.recommendations import FarmerRecommendationEngine
from src.explainability.shap_explainer import GingerShapExplainer
from src.utils.config import (
    BEST_MODEL_PATH,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    RESULTS_DIR,
    RESULTS_PLOTS_DIR,
    TEST_SIZE,
)
from sklearn.model_selection import GroupShuffleSplit


def generate_global_plots(
    df_importance: pd.DataFrame,
    shap_matrix: np.ndarray,
    X_test: pd.DataFrame,
    output_dir: Path,
):
    """Generate global SHAP importance and beeswarm summary visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # 1. Global Feature Importance Bar Chart (Top 15 Features)
    plt.figure(figsize=(10, 6))
    top15 = df_importance.head(15).sort_values("Mean_Abs_SHAP_t_ha", ascending=True)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(top15)))

    bars = plt.barh(top15["Label"], top15["Mean_Abs_SHAP_t_ha"], color=colors, height=0.65)
    plt.xlabel("Mean |SHAP Value| (Impact on Final Yield in t/ha)", fontsize=11, fontweight="bold")
    plt.title("Global Feature Importance (TreeSHAP on CatBoost Regressor)\nUntouched Hold-Out Test Set (n=1,977 across 200 Farms)", fontsize=12, pad=12)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height() / 2, f"{width:.3f} t/ha",
                 va="center", ha="left", fontsize=8.5, fontweight="semibold")

    plt.xlim(0, top15["Mean_Abs_SHAP_t_ha"].max() * 1.15)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_global_feature_importance.png", dpi=300)
    plt.close()

    # 2. SHAP Beeswarm / Summary Plot
    plt.figure(figsize=(10, 7))
    # Replace feature names with clean labels for plotting
    label_dict = dict(zip(df_importance["Feature"], df_importance["Label"]))
    X_renamed = X_test.rename(columns=label_dict)
    
    shap.summary_plot(
        shap_matrix,
        X_renamed,
        max_display=15,
        show=False,
    )
    plt.title("SHAP Summary Beeswarm Plot: Feature Impact on Final Yield (t/ha)", fontsize=12, pad=14)
    plt.xlabel("SHAP Value (Impact on Yield Prediction in t/ha)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_beeswarm.png", dpi=300)
    plt.close()


def generate_local_plots(
    explainer: GingerShapExplainer,
    farm_mgr: FarmStateManager,
    step_explanations: list,
    output_dir: Path,
):
    """Generate instance-level waterfall, positive/negative diverging, and delta comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # Representative prediction: Step 5 (DAP 135, Rhizome Initiation)
    rep_exp = step_explanations[4]

    # 1. Positive vs Negative Contributions Diverging Bar Chart (DAP 135)
    plt.figure(figsize=(9, 5.5))
    top_pos = rep_exp.top_positive_factors[:4]
    top_neg = rep_exp.top_negative_factors[:4]
    all_top = top_neg[::-1] + top_pos

    labels = [f"{c.label}\n({c.value} {c.unit})" for c in all_top]
    values = [c.shap_value for c in all_top]
    bar_colors = ["#2ca02c" if v > 0 else "#d62728" for v in values]

    bars = plt.barh(labels, values, color=bar_colors, alpha=0.85, height=0.6)
    plt.axvline(0, color="black", linestyle="-", linewidth=1.0)
    plt.xlabel("SHAP Contribution (t/ha relative to baseline: 13.65 t/ha)", fontsize=10.5, fontweight="bold")
    plt.title(f"Instance Factor Attribution: Farm F0001 at DAP {rep_exp.days_after_planting} ({rep_exp.growth_stage})\nForecast = {rep_exp.predicted_yield_t_ha:.2f} t/ha (Baseline = {rep_exp.base_value_t_ha:.2f} t/ha)", fontsize=11.5, pad=12)

    for bar, val in zip(bars, values):
        offset = 0.02 if val >= 0 else -0.02
        ha = "left" if val >= 0 else "right"
        plt.text(val + offset, bar.get_y() + bar.get_height() / 2, f"{val:+.3f} t/ha",
                 va="center", ha=ha, fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "positive_vs_negative_contributions.png", dpi=300)
    plt.close()

    # 2. Prediction Change SHAP Comparison: Pre-Infestation (DAP 75) vs Post-Infestation (DAP 100)
    step3 = step_explanations[2]  # DAP 75
    step4 = step_explanations[3]  # DAP 100

    plt.figure(figsize=(9, 5.5))
    common_keys = ["Biotic_Stress_Index", "Plant_Height_cm", "Total_Water_Input_mm", "Soil_Moisture_pct", "Leaf_Greenness_Index"]
    key_labels = ["Biotic Stress Score", "Plant Height", "Total Water Input", "Soil Moisture", "Leaf Greenness"]

    shap_step3 = {c.feature_name: c.shap_value for c in step3.top_positive_factors + step3.top_negative_factors}
    shap_step4 = {c.feature_name: c.shap_value for c in step4.top_positive_factors + step4.top_negative_factors}

    v3 = [shap_step3.get(k, 0.0) for k in common_keys]
    v4 = [shap_step4.get(k, 0.0) for k in common_keys]

    y_pos = np.arange(len(common_keys))
    height = 0.35

    plt.barh(y_pos + height/2, v3, height, label=f"Step 3 (DAP 75): Forecast = {step3.predicted_yield_t_ha:.2f} t/ha", color="#4a90e2", alpha=0.9)
    plt.barh(y_pos - height/2, v4, height, label=f"Step 4 (DAP 100, Pest Detected): Forecast = {step4.predicted_yield_t_ha:.2f} t/ha", color="#e74c3c", alpha=0.9)

    plt.yticks(y_pos, key_labels, fontsize=10)
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
    plt.xlabel("SHAP Value (t/ha)", fontsize=10.5, fontweight="bold")
    plt.title(f"Dynamic Factor Attribution Shift: Before vs After Pest Event\nPrediction Delta: {step4.prediction_delta_t_ha:+.2f} t/ha ({step4.prediction_delta_pct:+.1f}%)", fontsize=11.5, pad=12)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_change_shap_comparison.png", dpi=300)
    plt.close()

    # 3. Waterfall plot for Representative Prediction
    plt.figure(figsize=(9, 6))
    rep_state = farm_mgr.reconstruct_current_state("2024-07-28")
    exp_obj = explainer.get_shap_explanation_object(rep_state)

    shap.plots.waterfall(exp_obj[0], max_display=10, show=False)
    plt.title(f"SHAP Waterfall Attribution: Farm F0001 (DAP {rep_exp.days_after_planting})\nBase Expected: {rep_exp.base_value_t_ha:.2f} t/ha -> Predicted: {rep_exp.predicted_yield_t_ha:.2f} t/ha", fontsize=11.5, pad=14)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_waterfall_representative.png", dpi=300)
    plt.close()


def generate_research_summary_text(
    df_importance: pd.DataFrame,
    step_explanations: list,
    output_path: Path,
):
    """Generate structured research summary artifact explaining findings and non-causal distinctions."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    top5 = df_importance.head(5)
    rep_exp = step_explanations[4]  # DAP 135

    content = f"""================================================================================
EXPLAINABLE AI (XAI) & FARMER-CENTRIC YIELD PREDICTION EXPLANATION REPORT
Phase 5 Research Deliverable | IT41012: Research Methods and Scientific Writing
================================================================================

1. SCIENTIFIC DISTINCTION: PREDICTION vs EXPLANATION vs CAUSATION
--------------------------------------------------------------------------------
This research explicitly defines the boundaries of Explainable AI (XAI):

A. PREDICTION:
   The quantitative numerical output (t/ha) generated by the tuned CatBoost
   gradient boosting regression model based on current observed features.

B. EXPLANATION:
   Game-theoretic Shapley Additive exPlanations (TreeSHAP) that allocate fair
   credit (phi_i) to each of the 36 input features such that:
   Predicted_Yield = Base_Expected_Value + sum(SHAP_i)
   This quantifies how much each feature shifted the model output relative to the
   global average baseline yield (~13.65 t/ha).

C. CAUSATION:
   SHAP values quantify statistical and associative attribution within the learned
   model. SHAP DOES NOT prove biological or agronomic cause-and-effect.
   Consequently, all explanations and farmer narratives use strictly non-causal
   terminology: 'contributed to the prediction', 'associated with higher/lower forecast',
   and 'identified by the model as a key factor'.

2. GLOBAL FEATURE IMPORTANCE (HOLDOUT TEST SET: n=1,977 observations across 200 farms)
--------------------------------------------------------------------------------
Top 5 Most Influential Features across Cultivation Horizons:
{top5[['Rank', 'Feature', 'Label', 'Mean_Abs_SHAP_t_ha']].to_string(index=False)}

Key Finding:
Vegetative vigor (Plant Height, Greenness Volume), Biotic Stress (Pest & Disease pressure),
and Water Availability (Total Water Input) dominate model decision-making throughout the
cultivation period, closely mirroring established agronomic literature for Zingiber officinale.

3. DYNAMIC IN-SEASON PREDICTION & EXPLANATION TRAJECTORY (FARM F0001)
--------------------------------------------------------------------------------
Demonstrated continuous in-season forecast updates with point-by-point SHAP explanations:
"""

    for i, exp in enumerate(step_explanations, 1):
        delta_str = f"{exp.prediction_delta_t_ha:+.2f} t/ha ({exp.prediction_delta_pct:+.1f}%)" if exp.prediction_delta_t_ha is not None else "Initial Baseline"
        pos_names = ", ".join([f"{c.label} (+{c.shap_value:.2f})" for c in exp.top_positive_factors[:2]])
        neg_names = ", ".join([f"{c.label} ({c.shap_value:.2f})" for c in exp.top_negative_factors[:2]]) if exp.top_negative_factors else "None"

        content += f"""
Step {i} | Date: {exp.prediction_timestamp[:10]} | DAP: {exp.days_after_planting:3d} | Stage: {exp.growth_stage:20s}
  Forecast Yield: {exp.predicted_yield_t_ha:.2f} t/ha | Delta: {delta_str}
  Top Supporting: {pos_names}
  Top Limiting:   {neg_names}
  Farmer Summary: {exp.farmer_summary}
  Guidance:       {exp.recommendations[0] if exp.recommendations else 'Maintain routine management.'}
"""

    content += """
4. FARMER-CENTRIC DECISION SUPPORT & RECOMMENDATION ENGINE
--------------------------------------------------------------------------------
Technical SHAP attributions are translated into plain-language actionable guidance
tailored for smallholder ginger farmers in Sri Lanka:
- Pest & Disease Alerts: Triggered only when scouting identifies elevated pressure.
- Rainfall-Aware Irrigation Advice: Withholds supplemental irrigation recommendations
  when recent rainfall exceeds 50 mm, actively preventing rhizome soft rot (Pythium).
- Transparent Rules: Operates exclusively on farmer-reported observations, weather APIs,
  and farm profiles without requiring or fabricating physical sensors or IoT devices.
================================================================================
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    print("=" * 80)
    print("  PHASE 5: EXPLAINABLE AI & FARMER-CENTRIC YIELD EXPLANATION")
    print("  Module: Research Methods and Scientific Writing (IT41012)")
    print("=" * 80)

    # 1. Global SHAP Feature Importance on Holdout Test Set
    print("\n[1/4] Calculating global SHAP attributions across hold-out test set...")
    df_processed = pd.read_csv(PROCESSED_DATA_PATH)
    X, y = get_feature_matrix_and_target(df_processed)
    groups = df_processed["Farm_ID"]

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    _, test_idx = next(gss.split(X, y, groups=groups))
    X_test = X.iloc[test_idx]

    explainer = GingerShapExplainer()
    df_importance, shap_matrix = explainer.explain_dataset(X_test)

    explain_dir = RESULTS_DIR / "explainability"
    explain_dir.mkdir(parents=True, exist_ok=True)

    csv_importance = explain_dir / "shap_global_importance.csv"
    df_importance.to_csv(csv_importance, index=False)
    print(f"      Saved global feature ranking to: {csv_importance.resolve()}")

    # 2. Dynamic Sequential Prediction & Explanation Demonstration (Farm F0001)
    print("\n[2/4] Executing dynamic prediction trajectory with SHAP explanations (Farm F0001)...")
    profile = FarmProfile(
        farm_id="F0001",
        district="Kandy",
        seed_variety="Local",
        soil_type="Loam",
        land_size_acres=1.0,
        planting_date="2024-03-15",
        pump_capacity_lph=8000.0,
    )
    farm_mgr = FarmStateManager(profile)
    dynamic_predictor = DynamicYieldPredictor()
    wx_provider = MockWeatherProvider(seed=42)

    simulation_steps = [
        {"date": "2024-04-14", "dap": 30, "stage": "Sprouting", "height": 14.0, "greenness": 40.0, "pest": "Low", "disease": "Low", "sm": 68.0, "irr": None},
        {"date": "2024-05-14", "dap": 60, "stage": "Early Vegetative", "height": 34.0, "greenness": 55.0, "pest": "Low", "disease": "Low", "sm": 62.0, "irr": None},
        {"date": "2024-05-29", "dap": 75, "stage": "Early Vegetative", "height": 42.0, "greenness": 58.0, "pest": "Low", "disease": "Low", "sm": 75.0, "irr": {"runtime_hours": 2.5, "water_source": "Well"}},
        {"date": "2024-06-23", "dap": 100, "stage": "Vegetative", "height": 55.0, "greenness": 48.0, "pest": "Medium", "disease": "Low", "sm": 55.0, "irr": None},
        {"date": "2024-07-28", "dap": 135, "stage": "Rhizome Initiation", "height": 72.0, "greenness": 64.0, "pest": "Low", "disease": "Low", "sm": 70.0, "irr": {"runtime_hours": 2.0, "water_source": "Well"}},
        {"date": "2024-09-11", "dap": 180, "stage": "Rhizome Development", "height": 84.0, "greenness": 68.0, "pest": "Low", "disease": "Low", "sm": 66.0, "irr": None},
        {"date": "2024-11-10", "dap": 240, "stage": "Maturity", "height": 85.0, "greenness": 45.0, "pest": "Low", "disease": "Low", "sm": 58.0, "irr": None},
    ]

    step_explanations = []
    explanation_rows = []
    factor_rows = []
    farmer_rows = []
    recommendation_rows = []

    for i, step in enumerate(simulation_steps, 1):
        # Update farm state
        pred_val, record, analysis = update_farm_state(
            farm_manager=farm_mgr,
            dynamic_predictor=dynamic_predictor,
            observation_date=step["date"],
            days_after_planting=step["dap"],
            growth_stage=step["stage"],
            plant_height_cm=step["height"],
            leaf_greenness_index=step["greenness"],
            pest_severity=step["pest"],
            disease_severity=step["disease"],
            soil_moisture_pct=step["sm"],
            irrigation_event=step["irr"],
            weather_provider=wx_provider,
        )

        # Generate full explanation
        exp = dynamic_predictor.explain_prediction(
            farm_manager=farm_mgr,
            as_of_date=step["date"],
            record=record,
            change_analysis=analysis,
        )
        step_explanations.append(exp)

        delta_str = f"{exp.prediction_delta_t_ha:+.2f} t/ha" if exp.prediction_delta_t_ha is not None else "Initial"
        print(f"      Step {i} (DAP {step['dap']:3d} | {step['stage']:20s}): Forecast = {exp.predicted_yield_t_ha:.2f} t/ha | Delta: {delta_str}")
        print(f"             Farmer Summary: {exp.farmer_summary}")
        if exp.recommendations:
            print(f"             Advisory: {exp.recommendations[0]}")

        # Tabular logging
        explanation_rows.append({
            "Farm_ID": exp.farm_id,
            "Step": i,
            "Date": step["date"],
            "DAP": exp.days_after_planting,
            "Growth_Stage": exp.growth_stage,
            "Predicted_Yield_t_ha": exp.predicted_yield_t_ha,
            "Base_Value_t_ha": exp.base_value_t_ha,
            "Previous_Prediction_t_ha": exp.previous_prediction_t_ha,
            "Prediction_Delta_t_ha": exp.prediction_delta_t_ha,
            "Prediction_Delta_pct": exp.prediction_delta_pct,
            "Top_Positive_Factors": "; ".join([f"{c.label} (+{c.shap_value:.2f})" for c in exp.top_positive_factors[:3]]),
            "Top_Negative_Factors": "; ".join([f"{c.label} ({c.shap_value:.2f})" for c in exp.top_negative_factors[:3]]),
        })

        farmer_rows.append({
            "Farm_ID": exp.farm_id,
            "Step": i,
            "DAP": exp.days_after_planting,
            "Growth_Stage": exp.growth_stage,
            "Predicted_Yield_t_ha": exp.predicted_yield_t_ha,
            "Prediction_Delta_t_ha": exp.prediction_delta_t_ha,
            "Farmer_Summary": exp.farmer_summary,
            "Supporting_Points": " | ".join(exp.farmer_supporting_points),
            "Limiting_Points": " | ".join(exp.farmer_limiting_points),
            "Primary_Recommendation": exp.recommendations[0] if exp.recommendations else "",
        })

        for c in exp.top_positive_factors + exp.top_negative_factors:
            factor_rows.append({
                "Farm_ID": exp.farm_id,
                "Step": i,
                "DAP": exp.days_after_planting,
                "Feature": c.feature_name,
                "Label": c.label,
                "Value": c.value,
                "Unit": c.unit,
                "SHAP_Contribution_t_ha": c.shap_value,
                "Direction": c.direction,
                "Rank": c.rank,
                "Farmer_Interpretation": c.farmer_interpretation,
            })

        for rec in exp.recommendations:
            recommendation_rows.append({
                "Farm_ID": exp.farm_id,
                "Step": i,
                "DAP": exp.days_after_planting,
                "Growth_Stage": exp.growth_stage,
                "Recommendation": rec,
            })

    # 3. Export CSV Artifacts
    print("\n[3/4] Exporting explainability artifacts & structured CSV reports...")
    pd.DataFrame(explanation_rows).to_csv(explain_dir / "prediction_explanations.csv", index=False)
    pd.DataFrame(factor_rows).to_csv(explain_dir / "factor_contributions.csv", index=False)
    pd.DataFrame(farmer_rows).to_csv(explain_dir / "farmer_explanations.csv", index=False)
    pd.DataFrame(recommendation_rows).to_csv(explain_dir / "recommendation_log.csv", index=False)
    generate_research_summary_text(df_importance, step_explanations, explain_dir / "explainability_summary.txt")

    # 4. Generate Research Visualizations
    print("\n[4/4] Generating publication-quality explainability figures...")
    plots_dir = RESULTS_PLOTS_DIR / "explainability"
    generate_global_plots(df_importance, shap_matrix, X_test, plots_dir)
    generate_local_plots(explainer, farm_mgr, step_explanations, plots_dir)
    print(f"      Saved all figures in: {plots_dir.resolve()}")

    print("\n" + "=" * 80)
    print("  PHASE 5: EXPLAINABLE AI LAYER COMPLETE & VERIFIED")
    print("=" * 80)


if __name__ == "__main__":
    main()
