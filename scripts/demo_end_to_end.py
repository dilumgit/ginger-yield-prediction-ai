"""Phase 6: End-to-End Dynamic Prototype & Integration Demo Script.

Demonstrates the complete integrated workflow:
1. Farm Profile Initialization & Storage Persistence.
2. Sequential In-Season Farmer Updates & Irrigation Logging without Historical Re-entry.
3. Current State Reconstruction & Temporal Leakage Enforcement.
4. Dynamic CatBoost Yield Forecasting with Delta Tracking.
5. TreeSHAP Attribution & Plain-Language Farmer Translations.
6. Rule-Based Agronomic Recommendations.
7. Verification of Farm State Persistence and Recovery.
8. Generation of Phase 6 Visualizations and Research Reports.

Usage:
    python scripts/demo_end_to_end.py
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

from src.data.farm_state import CropObservation, FarmProfile, FarmStateManager
from src.data.irrigation import IrrigationEvent
from src.data.weather_api import MockWeatherProvider
from src.dynamic.dynamic_predictor import DynamicYieldPredictor
from src.dynamic.state_updater import update_farm_state
from src.storage.farm_repository import FarmRepository
from src.utils.config import RESULTS_DIR


def generate_phase6_plots(df_demo: pd.DataFrame, df_shap_evol: pd.DataFrame, output_dir: Path):
    """Generate Phase 6 integration research figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # 1. End-to-End Prediction Trajectory
    plt.figure(figsize=(9, 5.5))
    plt.plot(
        df_demo["DAP"],
        df_demo["Predicted_Yield_t_ha"],
        marker="o",
        markersize=8,
        color="#1f77b4",
        linewidth=2.2,
        label="Dynamic Model Forecast (t/ha)",
    )
    plt.axhline(14.20, color="#d62728", linestyle="--", linewidth=1.6, label="Ground Truth Final Yield (14.20 t/ha)")

    for _, row in df_demo.iterrows():
        delta_str = f" ({row['Delta_t_ha']:+.2f})" if pd.notna(row['Delta_t_ha']) else ""
        plt.annotate(
            f"{row['Predicted_Yield_t_ha']:.2f} t/ha{delta_str}\n[{row['Growth_Stage']}]",
            (row["DAP"], row["Predicted_Yield_t_ha"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            fontweight="semibold",
        )

    plt.xlabel("Days After Planting (DAP)", fontsize=11, fontweight="bold")
    plt.ylabel("Predicted Final Yield (t/ha)", fontsize=11, fontweight="bold")
    plt.title("End-to-End Dynamic Prediction Trajectory (Farm F0001)\nContinuous Updates Without Historical Re-entry", fontsize=12, pad=12)
    plt.ylim(10.5, 18.0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "end_to_end_prediction_trajectory.png", dpi=300)
    plt.close()

    # 2. Key SHAP Attributions Evolution across Stages
    if not df_shap_evol.empty:
        plt.figure(figsize=(10, 5.5))
        key_features = ["Biotic_Stress_Index", "Fertilizer_kg_acre", "Plant_Height_per_DAP", "Total_Water_Input_mm"]
        labels = ["Biotic Stress Score", "Fertilizer Dosage", "Growth Velocity", "Total Water Input"]
        colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]

        for feat, label, col in zip(key_features, labels, colors):
            sub = df_shap_evol[df_shap_evol["Feature"] == feat]
            if not sub.empty:
                plt.plot(sub["DAP"], sub["SHAP_Contribution_t_ha"], marker="s", markersize=6, linewidth=1.8, label=label, color=col)

        plt.axhline(0, color="black", linestyle="-", linewidth=0.8)
        plt.xlabel("Days After Planting (DAP)", fontsize=11, fontweight="bold")
        plt.ylabel("SHAP Contribution (t/ha relative to baseline: 13.65 t/ha)", fontsize=11, fontweight="bold")
        plt.title("Longitudinal SHAP Feature Contribution Evolution (Farm F0001)", fontsize=12, pad=12)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(output_dir / "end_to_end_shap_evolution.png", dpi=300)
        plt.close()


def generate_integration_report(df_demo: pd.DataFrame, output_path: Path):
    """Generate structured Phase 6 Integration Report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = f"""================================================================================
PHASE 6 SYSTEM INTEGRATION & END-TO-END DYNAMIC PROTOTYPE REPORT
Research Project: An Explainable AI-Based Dynamic In-Season Ginger Yield Prediction
Module: Research Methods and Scientific Writing (IT41012)
================================================================================

1. SYSTEM ARCHITECTURE & INTEGRATION SUMMARY
--------------------------------------------------------------------------------
The Phase 6 prototype successfully unifies all Phase 1-5 components into an
interactive, smallholder-accessible research system:

  [Farmer Input Interface (Streamlit UI)]
                   │
  [FarmStateManager & FarmRepository (JSON Persistence)]
                   │
  [Strict Temporal Filtering (timestamp <= t)]
                   │
  [In-Season Feature Engineering Pipeline]
                   │
  [Trained Phase 3 CatBoost Regressor]
                   │
  [Dynamic Yield Prediction (t/ha) & PredictionHistory Audit]
                   │
  [TreeSHAP Explainer (Base Value + phi_i)]
                   │
  [Farmer Explanation Translator & Rule-Based Recommendation Engine]
                   │
  [Interactive Farmer Dashboard & Trajectory Visualizer]

2. END-TO-END DEMONSTRATION TRAJECTORY (FARM F0001)
--------------------------------------------------------------------------------
Demonstrated multi-step in-season forecast evolution without requiring historical re-entry:

{df_demo[['Step', 'Date', 'DAP', 'Growth_Stage', 'Predicted_Yield_t_ha', 'Delta_t_ha', 'Delta_pct', 'Cumulative_Irrigation_Hours']].to_string(index=False)}

3. KEY SCIENTIFIC & TECHNICAL VERIFICATIONS
--------------------------------------------------------------------------------
A. ZERO PHYSICAL HARDWARE:
   Operates strictly on farmer-reported observations, farm profiles, and weather
   abstractions (Open-Meteo & Mock provider). Zero claim of physical IoT sensors.

B. TEMPORAL LEAKAGE PROTECTION:
   Reconstructs feature states using strictly data with timestamp <= t. Future
   crop measurements, future rainfall, and future irrigation are excluded.

C. PREDICTION ≠ EXPLANATION ≠ CAUSATION:
   TreeSHAP explains model mathematical attribution (phi_i) relative to the global
   baseline (13.65 t/ha). All translations use non-causal agronomic phrasing.

D. FARMER-FRIENDLY IRRIGATION:
   Farmers enter pump runtime (hours/minutes). The system converts to depth (mm)
   if pump capacity is known, or preserves runtime directly as an observable feature.

E. DATA PERSISTENCE:
   Farm states, observations, irrigation logs, and prediction histories persist
   seamlessly via FarmRepository across app restarts.
================================================================================
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    print("=" * 80)
    print("  PHASE 6: SYSTEM INTEGRATION & END-TO-END DYNAMIC PROTOTYPE DEMO")
    print("  Module: Research Methods and Scientific Writing (IT41012)")
    print("=" * 80)

    # 1. Initialize Repository & Demonstration Farm
    repo = FarmRepository()
    profile = FarmProfile(
        farm_id="F0001",
        district="Kandy",
        seed_variety="Local",
        soil_type="Loam",
        land_size_acres=1.0,
        planting_date="2024-03-15",
        water_source="Well",
        irrigation_method="Sprinkler",
        pump_capacity_lph=8000.0,
    )
    farm_mgr = FarmStateManager(profile)
    dynamic_predictor = DynamicYieldPredictor()
    wx_provider = MockWeatherProvider(seed=42)

    print(f"\n[1/4] Initialized Farm State Manager & Persistence Repository for {profile.farm_id}...")

    # 2. Sequential In-Season Cultivation Simulation Steps
    simulation_steps = [
        {"desc": "Initial Emergence Scouting", "date": "2024-04-14", "dap": 30, "stage": "Sprouting", "h": 14.0, "g": 40.0, "p": "Low", "d": "Low", "sm": 68.0, "fert": 120.0, "irr": None},
        {"desc": "Early Vegetative Elongation", "date": "2024-05-14", "dap": 60, "stage": "Early Vegetative", "h": 34.0, "g": 55.0, "p": "Low", "d": "Low", "sm": 62.0, "fert": 120.0, "irr": None},
        {"desc": "Farmer Logs Supplemental Irrigation", "date": "2024-05-29", "dap": 75, "stage": "Early Vegetative", "h": 42.0, "g": 58.0, "p": "Low", "d": "Low", "sm": 75.0, "fert": 120.0, "irr": {"runtime_hours": 2.5, "water_source": "Well", "irrigation_method": "Sprinkler"}},
        {"desc": "Vegetative Scouting (Pest Detected)", "date": "2024-06-23", "dap": 100, "stage": "Vegetative", "h": 55.0, "g": 48.0, "p": "Medium", "d": "Low", "sm": 55.0, "fert": 120.0, "irr": None},
        {"desc": "Rhizome Initiation & Recovery", "date": "2024-07-28", "dap": 135, "stage": "Rhizome Initiation", "h": 72.0, "g": 64.0, "p": "Low", "d": "Low", "sm": 70.0, "fert": 120.0, "irr": {"runtime_hours": 2.0, "water_source": "Well", "irrigation_method": "Sprinkler"}},
        {"desc": "Rhizome Development Vigor Check", "date": "2024-09-11", "dap": 180, "stage": "Rhizome Development", "h": 84.0, "g": 68.0, "p": "Low", "d": "Low", "sm": 66.0, "fert": 120.0, "irr": None},
        {"desc": "Maturity & Pre-Harvest Assessment", "date": "2024-11-10", "dap": 240, "stage": "Maturity", "h": 85.0, "g": 45.0, "p": "Low", "d": "Low", "sm": 58.0, "fert": 120.0, "irr": None},
    ]

    print("\n[2/4] Executing dynamic update pipeline across in-season developmental steps...")
    demo_rows = []
    shap_rows = []

    for i, step in enumerate(simulation_steps, 1):
        pred_val, rec, analysis = update_farm_state(
            farm_manager=farm_mgr,
            dynamic_predictor=dynamic_predictor,
            observation_date=step["date"],
            days_after_planting=step["dap"],
            growth_stage=step["stage"],
            plant_height_cm=step["h"],
            leaf_greenness_index=step["g"],
            pest_severity=step["p"],
            disease_severity=step["d"],
            soil_moisture_pct=step["sm"],
            fertilizer_kg_acre=step["fert"],
            irrigation_event=step["irr"],
            weather_provider=wx_provider,
        )

        # Generate full explanation
        exp = dynamic_predictor.explain_prediction(
            farm_manager=farm_mgr,
            as_of_date=step["date"],
            record=rec,
            change_analysis=analysis,
        )

        delta_str = f"{exp.prediction_delta_t_ha:+.2f} t/ha ({exp.prediction_delta_pct:+.1f}%)" if exp.prediction_delta_t_ha is not None else "Initial Baseline"
        print(f"      Step {i} (DAP {step['dap']:3d} | {step['stage']:20s}): Forecast = {exp.predicted_yield_t_ha:.2f} t/ha | Delta: {delta_str}")
        print(f"             Farmer Summary: {exp.farmer_summary}")
        if exp.recommendations:
            print(f"             Advisory: {exp.recommendations[0]}")

        demo_rows.append({
            "Farm_ID": profile.farm_id,
            "Step": i,
            "Date": step["date"],
            "DAP": step["dap"],
            "Growth_Stage": step["stage"],
            "Description": step["desc"],
            "Predicted_Yield_t_ha": exp.predicted_yield_t_ha,
            "Base_Value_t_ha": exp.base_value_t_ha,
            "Delta_t_ha": exp.prediction_delta_t_ha,
            "Delta_pct": exp.prediction_delta_pct,
            "Cumulative_Irrigation_Hours": farm_mgr.irrigation_log.get_cumulative_runtime_hours(step["date"]),
            "Top_Supporting": exp.top_positive_factors[0].label if exp.top_positive_factors else "",
            "Top_Limiting": exp.top_negative_factors[0].label if exp.top_negative_factors else "",
            "Farmer_Summary": exp.farmer_summary,
            "Primary_Recommendation": exp.recommendations[0] if exp.recommendations else "",
        })

        for c in exp.top_positive_factors + exp.top_negative_factors:
            shap_rows.append({
                "Farm_ID": profile.farm_id,
                "Step": i,
                "DAP": step["dap"],
                "Feature": c.feature_name,
                "Label": c.label,
                "SHAP_Contribution_t_ha": c.shap_value,
                "Direction": c.direction,
            })

    # 3. Test Persistence & Reloading
    print("\n[3/4] Testing persistent storage and state reconstruction from repository...")
    saved_path = repo.save_farm_state(farm_mgr, dynamic_predictor)
    print(f"      Persisted farm state to: {saved_path.resolve()}")

    reloaded = repo.load_farm_state("F0001")
    assert reloaded is not None, "Failed to reload persisted farm state."
    reloaded_mgr, reloaded_pred = reloaded
    assert len(reloaded_mgr.crop_observations) == 7, "Mismatch in reloaded crop observation count."
    assert len(reloaded_mgr.irrigation_log) == 2, "Mismatch in reloaded irrigation events count."
    assert len(reloaded_pred.history.get_farm_history("F0001")) == 7, "Mismatch in prediction history count."
    print("      Verification successful: Reconstructed 7 crop observations, 2 irrigation events, and 7 prediction records.")

    # 4. Save Results and Figures
    print("\n[4/4] Exporting Phase 6 artifacts and research figures...")
    phase6_dir = RESULTS_DIR / "phase6"
    phase6_dir.mkdir(parents=True, exist_ok=True)

    df_demo = pd.DataFrame(demo_rows)
    df_shap_evol = pd.DataFrame(shap_rows)

    csv_demo = phase6_dir / "end_to_end_demo.csv"
    csv_history = phase6_dir / "prediction_history.csv"
    df_demo.to_csv(csv_demo, index=False)
    dynamic_predictor.history.to_dataframe().to_csv(csv_history, index=False)

    generate_integration_report(df_demo, phase6_dir / "phase6_integration_report.txt")
    generate_integration_report(df_demo, phase6_dir / "integration_summary.txt")

    plots_dir = phase6_dir / "plots"
    generate_phase6_plots(df_demo, df_shap_evol, plots_dir)

    print(f"      Saved demo records: {csv_demo.resolve()}")
    print(f"      Saved prediction history: {csv_history.resolve()}")
    print(f"      Saved research figures: {plots_dir.resolve()}")

    print("\n" + "=" * 80)
    print("  PHASE 6: SYSTEM INTEGRATION & END-TO-END DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
