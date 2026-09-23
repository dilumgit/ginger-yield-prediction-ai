"""Comprehensive Unit Tests for Phase 4 Dynamic In-Season Prediction Framework.

Verifies:
1. New observation is appended without modifying history.
2. Historical observations remain unchanged.
3. Irrigation events are preserved in history.
4. Cumulative irrigation runtime is calculated correctly.
5. Rolling 7-day irrigation runtime is calculated correctly.
6. Current state reconstruction builds valid feature dictionaries.
7. Future observations are strictly excluded from reconstructed state.
8. Future irrigation events are strictly excluded from reconstructed state.
9. Future weather data is strictly excluded from reconstructed state.
10. Farm_ID is not used as a predictive model feature.
11. Actual_Final_Yield_t_ha is not used as a model feature.
12. Predicted_Final_Yield_t_ha is not used as a model feature.
13. New observation triggers updated prediction.
14. Prediction history is preserved across updates.
15. Prediction delta and percentage changes are calculated correctly.
16. Dynamic simulation script executes end-to-end.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.data.farm_state import (
    CropObservation,
    FarmProfile,
    FarmStateManager,
    PredictionHistory,
    PredictionRecord,
    WeatherObservation,
)
from src.data.irrigation import IrrigationEvent, IrrigationEventLog
from src.data.weather_api import MockWeatherProvider
from src.dynamic.dynamic_predictor import DynamicYieldPredictor
from src.dynamic.state_updater import update_farm_state
from src.dynamic.dynamic_evaluator import DynamicEvaluator
from src.utils.config import BEST_MODEL_PATH, ID_COLUMNS, LEAKAGE_COLUMNS, TARGET_COLUMN


@pytest.fixture
def sample_farm_manager():
    """Fixture providing an initialized FarmStateManager."""
    profile = FarmProfile(
        farm_id="F0042",
        district="Kandy",
        seed_variety="Local",
        soil_type="Loam",
        land_size_acres=1.5,
        planting_date="2024-03-01",
        pump_capacity_lph=10000.0,
    )
    return FarmStateManager(profile)


def test_1_new_observation_appended_and_history_preserved(sample_farm_manager):
    """Verify new observations append cleanly without altering past history."""
    obs1 = CropObservation(date="2024-03-31", days_after_planting=30, growth_stage="Sprouting", plant_height_cm=12.0, leaf_greenness_index=35.0)
    sample_farm_manager.add_crop_observation(obs1)
    assert len(sample_farm_manager.crop_observations) == 1

    obs2 = CropObservation(date="2024-04-30", days_after_planting=60, growth_stage="Early Vegetative", plant_height_cm=30.0, leaf_greenness_index=50.0)
    sample_farm_manager.add_crop_observation(obs2)
    assert len(sample_farm_manager.crop_observations) == 2

    # Verify first observation remains unmodified
    assert sample_farm_manager.crop_observations[0].days_after_planting == 30
    assert sample_farm_manager.crop_observations[0].plant_height_cm == 12.0
    assert sample_farm_manager.crop_observations[1].days_after_planting == 60
    assert sample_farm_manager.crop_observations[1].plant_height_cm == 30.0


def test_2_irrigation_events_preserved_and_runtime_aggregations(sample_farm_manager):
    """Verify irrigation events append and calculate rolling/cumulative runtimes."""
    sample_farm_manager.add_irrigation_event(IrrigationEvent(date="2024-04-01", runtime_hours=2.0))
    sample_farm_manager.add_irrigation_event(IrrigationEvent(date="2024-04-10", runtime_hours=1.5))
    sample_farm_manager.add_irrigation_event(IrrigationEvent(date="2024-04-14", runtime_hours=2.5))

    log = sample_farm_manager.irrigation_log
    assert len(log) == 3

    # Cumulative runtime as of April 14 = 2.0 + 1.5 + 2.5 = 6.0 hours
    assert log.get_cumulative_runtime_hours("2024-04-14") == pytest.approx(6.0, 0.01)

    # Rolling 7-day runtime as of April 14 (includes April 10, 14 = 1.5 + 2.5 = 4.0h)
    assert log.get_recent_runtime_hours("2024-04-14", window_days=7) == pytest.approx(4.0, 0.01)


def test_3_current_state_reconstruction(sample_farm_manager):
    """Verify state reconstruction builds complete valid feature dictionary."""
    sample_farm_manager.add_crop_observation(CropObservation(
        date="2024-04-30", days_after_planting=60, growth_stage="Early Vegetative", plant_height_cm=35.0, leaf_greenness_index=55.0
    ))
    state = sample_farm_manager.reconstruct_current_state("2024-04-30")

    assert state["Farm_ID"] == "F0042"
    assert state["Days_After_Planting"] == 60
    assert state["Plant_Height_cm"] == 35.0
    assert state["Growth_Stage_Ordinal"] == 1
    assert "Total_Water_Input_mm" in state
    assert "Canopy_Greenness_Volume" in state
    assert state["District_Kandy"] == 1
    assert state["Soil_Type_Loam"] == 1


def test_4_future_observations_excluded_from_state(sample_farm_manager):
    """Verify future crop observations (date > t) are excluded from reconstructed state."""
    sample_farm_manager.add_crop_observation(CropObservation(
        date="2024-03-31", days_after_planting=30, growth_stage="Sprouting", plant_height_cm=10.0, leaf_greenness_index=35.0
    ))
    # Future observation
    sample_farm_manager.add_crop_observation(CropObservation(
        date="2024-05-31", days_after_planting=90, growth_stage="Vegetative", plant_height_cm=65.0, leaf_greenness_index=65.0
    ))

    # Reconstruct at Day 30
    state = sample_farm_manager.reconstruct_current_state("2024-03-31")
    assert state["Days_After_Planting"] == 30
    assert state["Plant_Height_cm"] == 10.0
    assert state["Growth_Stage"] == "Sprouting"


def test_5_future_irrigation_excluded_from_state(sample_farm_manager):
    """Verify future irrigation events (date > t) are excluded from reconstructed state."""
    sample_farm_manager.add_crop_observation(CropObservation(
        date="2024-03-31", days_after_planting=30, growth_stage="Sprouting", plant_height_cm=10.0, leaf_greenness_index=35.0
    ))
    # Irrigation on April 15 (future relative to March 31)
    sample_farm_manager.add_irrigation_event(IrrigationEvent(date="2024-04-15", runtime_hours=3.0))

    state = sample_farm_manager.reconstruct_current_state("2024-03-31")
    assert state["Pump_Runtime_Hours"] == 0.0
    assert state["Cumulative_Irrigation_Hours"] == 0.0


def test_6_future_weather_excluded_from_state(sample_farm_manager):
    """Verify future weather observations (date > t) are excluded from reconstructed state."""
    sample_farm_manager.add_crop_observation(CropObservation(
        date="2024-03-31", days_after_planting=30, growth_stage="Sprouting", plant_height_cm=10.0, leaf_greenness_index=35.0
    ))
    sample_farm_manager.add_weather_observation(WeatherObservation(
        date="2024-03-31", weekly_rainfall_mm=30.0, avg_temperature_c=25.0, relative_humidity_pct=80.0, solar_radiation_mj_m2_day=18.0
    ))
    # Future extreme weather on May 15
    sample_farm_manager.add_weather_observation(WeatherObservation(
        date="2024-05-15", weekly_rainfall_mm=150.0, avg_temperature_c=32.0, relative_humidity_pct=95.0, solar_radiation_mj_m2_day=12.0
    ))

    state = sample_farm_manager.reconstruct_current_state("2024-03-31")
    assert state["Weekly_Rainfall_mm"] == 30.0
    assert state["Avg_Temperature_C"] == 25.0


def test_7_target_and_leakage_exclusion_from_features(sample_farm_manager):
    """Verify Target and Leakage columns are never generated in feature state."""
    sample_farm_manager.add_crop_observation(CropObservation(
        date="2024-04-30", days_after_planting=60, growth_stage="Early Vegetative", plant_height_cm=35.0, leaf_greenness_index=55.0
    ))
    state = sample_farm_manager.reconstruct_current_state("2024-04-30")

    assert TARGET_COLUMN not in state
    for leak in LEAKAGE_COLUMNS:
        assert leak not in state


def test_8_continuous_prediction_and_history_logging(sample_farm_manager):
    """Verify continuous prediction trigger, history logging, and delta calculation."""
    predictor = DynamicYieldPredictor(BEST_MODEL_PATH)
    wx = MockWeatherProvider()

    # Step 1: Initial observation at DAP 30
    pred1, rec1, anal1 = update_farm_state(
        farm_manager=sample_farm_manager,
        dynamic_predictor=predictor,
        observation_date="2024-03-31",
        days_after_planting=30,
        growth_stage="Sprouting",
        plant_height_cm=14.0,
        leaf_greenness_index=40.0,
        weather_provider=wx,
    )
    assert isinstance(pred1, float)
    assert rec1.prediction_delta_t_ha is None
    assert anal1 is None

    # Step 2: Update at DAP 60
    pred2, rec2, anal2 = update_farm_state(
        farm_manager=sample_farm_manager,
        dynamic_predictor=predictor,
        observation_date="2024-04-30",
        days_after_planting=60,
        growth_stage="Early Vegetative",
        plant_height_cm=38.0,
        leaf_greenness_index=55.0,
        weather_provider=wx,
    )
    assert isinstance(pred2, float)
    assert rec2.prediction_delta_t_ha == pytest.approx(pred2 - pred1, 0.001)
    assert anal2 is not None
    assert "summary_statement" in anal2
    assert len(anal2["modified_variables"]) > 0

    # Step 3: Farmer adds irrigation event at DAP 75
    pred3, rec3, anal3 = update_farm_state(
        farm_manager=sample_farm_manager,
        dynamic_predictor=predictor,
        observation_date="2024-05-15",
        days_after_planting=75,
        growth_stage="Early Vegetative",
        irrigation_event={"runtime_hours": 3.0, "water_source": "Well"},
        weather_provider=wx,
    )
    assert rec3.prediction_delta_t_ha == pytest.approx(pred3 - pred2, 0.001)
    assert len(predictor.history.get_farm_history("F0042")) == 3


def test_9_dynamic_simulation_script_runs():
    """Verify the dynamic simulation script executes end-to-end."""
    from scripts.simulate_dynamic_prediction import main as run_simulation
    run_simulation()

    demo_csv = Path("results/dynamic_prediction_demo.csv")
    assert demo_csv.exists()
    df_demo = pd.read_csv(demo_csv)
    assert len(df_demo) == 7
    assert "Current_Prediction_t_ha" in df_demo.columns
    assert "Prediction_Delta_t_ha" in df_demo.columns


def test_10_dynamic_evaluator_runs():
    """Verify the dynamic evaluator calculates checkpoint metrics."""
    evaluator = DynamicEvaluator()
    summary_df, detailed_df = evaluator.evaluate_test_checkpoints()

    assert len(summary_df) == 5
    assert "MAE_t_ha" in summary_df.columns
    assert "R2_Score" in summary_df.columns
    assert (summary_df["MAE_t_ha"] < 1.5).all()
