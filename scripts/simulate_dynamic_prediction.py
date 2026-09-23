"""Dynamic In-Season Prediction Simulation Script (Phase 4).

Simulates a continuous in-season cultivation journey for a ginger farm in Sri Lanka.
Demonstrates how the system ingests new observations, updates farm state, recalculates
engineered features, triggers updated predictions, and calculates prediction deltas
without requiring the farmer to re-enter historical records.

Usage:
    python scripts/simulate_dynamic_prediction.py
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

from src.data.farm_state import FarmProfile, FarmStateManager
from src.data.weather_api import MockWeatherProvider
from src.dynamic.dynamic_predictor import DynamicYieldPredictor
from src.dynamic.state_updater import update_farm_state
from src.utils.config import RESULTS_DIR, RESULTS_PLOTS_DIR


def generate_simulation_plots(df_demo: pd.DataFrame, output_dir: Path, ground_truth_yield: float = 14.50):
    """Generate research visualization figures for dynamic prediction trajectory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font="sans-serif")

    # 1. Prediction vs Days After Planting (DAP) Trajectory
    plt.figure(figsize=(8, 5))
    plt.plot(
        df_demo["DAP"],
        df_demo["Current_Prediction_t_ha"],
        marker="o",
        markersize=8,
        color="#1f77b4",
        linewidth=2.2,
        label="Dynamic Model Forecast (t/ha)",
    )
    plt.axhline(
        ground_truth_yield,
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label=f"Ground Truth Final Yield ({ground_truth_yield:.2f} t/ha)",
    )
    plt.xlabel("Days After Planting (DAP)", fontsize=11, fontweight="bold")
    plt.ylabel("Predicted Final Yield (t/ha)", fontsize=11, fontweight="bold")
    plt.title("Continuous In-Season Final Yield Prediction Trajectory (Farm F0001)", fontsize=12, pad=12)

    for _, row in df_demo.iterrows():
        plt.annotate(
            f"{row['Current_Prediction_t_ha']:.2f}\n({row['Growth_Stage']})",
            (row["DAP"], row["Current_Prediction_t_ha"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            fontweight="semibold",
        )

    plt.ylim(min(df_demo["Current_Prediction_t_ha"].min(), ground_truth_yield) - 1.5,
             max(df_demo["Current_Prediction_t_ha"].max(), ground_truth_yield) + 2.0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_vs_dap.png", dpi=300)
    plt.close()

    # 2. Step-by-Step Prediction Trajectory with Deltas
    plt.figure(figsize=(9, 5))
    steps = [f"Step {i+1}\n(DAP {row['DAP']})" for i, row in df_demo.iterrows()]
    colors = ["#2ca02c" if (d is not None and d >= 0) else "#d62728" for d in df_demo["Prediction_Delta_t_ha"]]
    bars = plt.bar(steps, df_demo["Current_Prediction_t_ha"], color="#4a90e2", alpha=0.85, width=0.55)

    for i, bar in enumerate(bars):
        row = df_demo.iloc[i]
        delta_str = f"Delta: {row['Prediction_Delta_t_ha']:+.2f} t/ha" if pd.notna(row['Prediction_Delta_t_ha']) else "Initial"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{row['Current_Prediction_t_ha']:.2f} t/ha\n{delta_str}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    plt.ylabel("Predicted Final Harvest Yield (t/ha)", fontsize=11, fontweight="bold")
    plt.title("Step-by-Step Prediction Evolution Across Farmer Update Events", fontsize=12, pad=12)
    plt.ylim(0, max(df_demo["Current_Prediction_t_ha"]) + 3.0)
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_trajectory.png", dpi=300)
    plt.close()

    # 3. Absolute Prediction Error vs DAP
    plt.figure(figsize=(8, 5))
    abs_errors = np.abs(df_demo["Current_Prediction_t_ha"] - ground_truth_yield)
    plt.plot(
        df_demo["DAP"],
        abs_errors,
        marker="s",
        markersize=8,
        color="#ff7f0e",
        linewidth=2.2,
    )
    plt.xlabel("Days After Planting (DAP)", fontsize=11, fontweight="bold")
    plt.ylabel("Absolute Forecast Error |y - ŷ| (t/ha)", fontsize=11, fontweight="bold")
    plt.title("Prediction Error Convergence as In-Season Observations Accumulate", fontsize=12, pad=12)

    for dap, err in zip(df_demo["DAP"], abs_errors):
        plt.annotate(
            f"{err:.2f} t/ha",
            (dap, err),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8.5,
        )

    plt.ylim(0, max(abs_errors) + 0.8)
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_error_vs_dap.png", dpi=300)
    plt.close()


def main():
    print("=" * 80)
    print("  PHASE 4: DYNAMIC IN-SEASON FARM STATE & CONTINUOUS PREDICTION DEMO")
    print("  Module: Research Methods and Scientific Writing (IT41012)")
    print("=" * 80)

    # 1. Initialize Farm Profile (Zero Hardware; Farmer-Reported Static Profile)
    profile = FarmProfile(
        farm_id="F0001",
        district="Kandy",
        seed_variety="Local",
        soil_type="Loam",
        land_size_acres=1.0,
        planting_date="2024-03-15",
        pump_capacity_lph=8000.0,  # 8000 L/h pump capacity
    )
    farm_mgr = FarmStateManager(profile)
    dynamic_predictor = DynamicYieldPredictor()
    weather_provider = MockWeatherProvider(seed=42)

    print(f"\n[1/3] Initialized Farm State Tracker:")
    print(f"      Farm_ID: {profile.farm_id} | District: {profile.district} | Variety: {profile.seed_variety} | Soil: {profile.soil_type}")
    print(f"      Land Size: {profile.land_size_acres} Acres | Planting Date: {profile.planting_date}")

    # 2. Simulate In-Season Update Sequence
    simulation_steps = [
        {
            "event_desc": "Initial Emergence Scouting (Sprouting)",
            "date": "2024-04-14",
            "dap": 30,
            "stage": "Sprouting",
            "height": 14.0,
            "greenness": 40.0,
            "pest": "Low",
            "disease": "Low",
            "sm": 68.0,
            "irrigation": None,
        },
        {
            "event_desc": "Early Vegetative Measurement",
            "date": "2024-05-14",
            "dap": 60,
            "stage": "Early Vegetative",
            "height": 34.0,
            "greenness": 55.0,
            "pest": "Low",
            "disease": "Low",
            "sm": 62.0,
            "irrigation": None,
        },
        {
            "event_desc": "Farmer Logs Supplemental Irrigation",
            "date": "2024-05-29",
            "dap": 75,
            "stage": "Early Vegetative",
            "height": 42.0,
            "greenness": 58.0,
            "pest": "Low",
            "disease": "Low",
            "sm": 75.0,
            "irrigation": {"runtime_hours": 2.5, "water_source": "Well", "irrigation_method": "Sprinkler"},
        },
        {
            "event_desc": "Vegetative Scouting (Pest Infestation Detected)",
            "date": "2024-06-23",
            "dap": 100,
            "stage": "Vegetative",
            "height": 55.0,
            "greenness": 48.0,
            "pest": "Medium",
            "disease": "Low",
            "sm": 55.0,
            "irrigation": None,
        },
        {
            "event_desc": "Rhizome Initiation & Recovery Post-Treatment",
            "date": "2024-07-28",
            "dap": 135,
            "stage": "Rhizome Initiation",
            "height": 72.0,
            "greenness": 64.0,
            "pest": "Low",
            "disease": "Low",
            "sm": 70.0,
            "irrigation": {"runtime_hours": 2.0, "water_source": "Well", "irrigation_method": "Sprinkler"},
        },
        {
            "event_desc": "Rhizome Development Vigor Check",
            "date": "2024-09-11",
            "dap": 180,
            "stage": "Rhizome Development",
            "height": 84.0,
            "greenness": 68.0,
            "pest": "Low",
            "disease": "Low",
            "sm": 66.0,
            "irrigation": None,
        },
        {
            "event_desc": "Maturity & Pre-Harvest Assessment",
            "date": "2024-11-10",
            "dap": 240,
            "stage": "Maturity",
            "height": 85.0,
            "greenness": 45.0,
            "pest": "Low",
            "disease": "Low",
            "sm": 58.0,
            "irrigation": None,
        },
    ]

    print("\n[2/3] Simulating continuous in-season updates without historical re-entry...")
    demo_records = []

    for i, step in enumerate(simulation_steps, 1):
        prev_pred = demo_records[-1]["Current_Prediction_t_ha"] if demo_records else None

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
            irrigation_event=step["irrigation"],
            weather_provider=weather_provider,
        )

        delta = record.prediction_delta_t_ha
        delta_pct = analysis["prediction_delta_pct"] if analysis else None

        demo_records.append({
            "Farm_ID": profile.farm_id,
            "Update_Step": i,
            "Observation_Date": step["date"],
            "DAP": step["dap"],
            "Growth_Stage": step["stage"],
            "Event_Description": step["event_desc"],
            "Previous_Prediction_t_ha": prev_pred,
            "Current_Prediction_t_ha": pred_val,
            "Prediction_Delta_t_ha": delta,
            "Prediction_Delta_pct": delta_pct,
            "Cumulative_Irrigation_Hours": farm_mgr.irrigation_log.get_cumulative_runtime_hours(step["date"]),
            "Recent_Irrigation_Hours": farm_mgr.irrigation_log.get_recent_runtime_hours(step["date"]),
        })

        delta_str = f"{delta:+.2f} t/ha ({delta_pct:+.1f}%)" if delta is not None else "Initial"
        print(f"      Step {i} (DAP {step['dap']:3d} | {step['stage']:20s}): Forecast = {pred_val:.2f} t/ha | Delta: {delta_str}")
        if analysis and analysis["modified_variables"]:
            top_var = analysis["modified_variables"][0]
            print(f"             Associated shift factor: {top_var['label']} ({top_var['description']})")

    df_demo = pd.DataFrame(demo_records)

    # 3. Save Demo CSV and Generate Visualizations
    out_csv = RESULTS_DIR / "dynamic_prediction_demo.csv"
    df_demo.to_csv(out_csv, index=False)
    print(f"\n[3/3] Exported dynamic demonstration records to: {out_csv.resolve()}")

    dynamic_plots_dir = RESULTS_PLOTS_DIR / "dynamic"
    generate_simulation_plots(df_demo, dynamic_plots_dir, ground_truth_yield=14.20)
    print(f"      Generated research plots in: {dynamic_plots_dir.resolve()}")

    print("\n" + "=" * 80)
    print("  DYNAMIC PREDICTION DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
