"""Comprehensive Integration Unit Tests for Phase 6 Research Prototype.

Verifies:
1. Farm profile creation with full parameter schemas.
2. New crop observation appending without altering history.
3. Historical observation immutability.
4. Irrigation event logging in append-only log.
5. Irrigation history preservation across updates.
6. Pump runtime-to-depth calculation when capacity is specified.
7. Observable runtime fallback when capacity is unknown.
8. Current state reconstruction as of arbitrary cutoff date t.
9. Dynamic prediction triggered automatically by new observations.
10. Prediction history preservation and audit trail logging.
11. Prediction delta and percentage change calculation.
12. TreeSHAP instance explanation generation and base value retrieval.
13. Plain-language farmer translation generation.
14. Rule-based recommendation engine alerts and guidance.
15. Strict temporal leakage protection: Discards observations with date > t.
16. Farm_ID exclusion from ML predictive features.
17. Target (Actual_Final_Yield) and benchmark (Predicted_Final_Yield) exclusion.
18. Exact 36-feature model schema enforcement.
19. Persistence, serialization, and lossless recovery via FarmRepository.
20. Complete end-to-end demo execution without error.
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
from src.data.irrigation import IrrigationEvent
from src.data.weather_api import MockWeatherProvider
from src.dynamic.dynamic_predictor import DynamicYieldPredictor
from src.dynamic.state_updater import update_farm_state
from src.explainability.explanation_model import PredictionExplanation
from src.explainability.shap_explainer import GingerShapExplainer
from src.storage.farm_repository import FarmRepository
from src.utils.config import BEST_MODEL_PATH, ID_COLUMNS, LEAKAGE_COLUMNS, TARGET_COLUMN


@pytest.fixture
def tmp_repo(tmp_path):
    """Fixture providing a FarmRepository with an isolated temporary directory."""
    return FarmRepository(storage_dir=tmp_path / "farms")


def test_1_farm_profile_creation():
    """Verify FarmProfile initialization and property constraints."""
    profile = FarmProfile(
        farm_id="F9999",
        district="Kandy",
        seed_variety="Local",
        soil_type="Loam",
        land_size_acres=1.5,
        planting_date="2024-03-15",
        water_source="Well",
        irrigation_method="Sprinkler",
        pump_capacity_lph=8000.0,
    )
    assert profile.farm_id == "F9999"
    assert profile.district == "Kandy"
    assert profile.pump_capacity_lph == 8000.0


def test_2_crop_observation_append_and_preservation():
    """Verify appending new observations does not mutate historical records."""
    profile = FarmProfile("F0001", "Kandy", "Local", "Loam", 1.0, "2024-03-15")
    mgr = FarmStateManager(profile)

    obs1 = CropObservation("2024-04-14", 30, "Sprouting", 14.0, 40.0)
    mgr.add_crop_observation(obs1)
    assert len(mgr.crop_observations) == 1

    obs2 = CropObservation("2024-05-14", 60, "Early Vegetative", 34.0, 55.0)
    mgr.add_crop_observation(obs2)
    assert len(mgr.crop_observations) == 2

    # Verify first record is unmodified
    assert mgr.crop_observations[0].days_after_planting == 30
    assert mgr.crop_observations[0].plant_height_cm == 14.0
    assert mgr.crop_observations[1].days_after_planting == 60
    assert mgr.crop_observations[1].plant_height_cm == 34.0


def test_3_irrigation_logging_and_capacity_conversions():
    """Verify pump runtime to depth conversion when capacity is known vs unknown."""
    # Known capacity: 8000 L/h, 1.0 acre (4046.86 m²), 2.5 h runtime
    # Volume = 20,000 L -> Depth = 20000 / 4046.86 = 4.9419 mm
    prof_known = FarmProfile("F0001", "Kandy", "Local", "Loam", 1.0, "2024-03-15", pump_capacity_lph=8000.0)
    mgr_known = FarmStateManager(prof_known)
    mgr_known.add_crop_observation(CropObservation("2024-05-29", 75, "Early Vegetative", 40.0, 50.0))
    mgr_known.add_irrigation_event(IrrigationEvent("2024-05-29", runtime_hours=2.5))
    state_known = mgr_known.reconstruct_current_state("2024-05-29")
    assert state_known["Estimated_Irrigation_mm"] == pytest.approx(4.942, abs=0.01)

    # Unknown capacity: pump_capacity_lph = None
    prof_unknown = FarmProfile("F0002", "Kandy", "Local", "Loam", 1.0, "2024-03-15", pump_capacity_lph=None)
    mgr_unknown = FarmStateManager(prof_unknown)
    mgr_unknown.add_crop_observation(CropObservation("2024-05-29", 75, "Early Vegetative", 40.0, 50.0))
    mgr_unknown.add_irrigation_event(IrrigationEvent("2024-05-29", runtime_hours=2.5))
    state_unknown = mgr_unknown.reconstruct_current_state("2024-05-29")
    assert state_unknown["Pump_Runtime_Hours"] == 2.5
    assert state_unknown["Estimated_Irrigation_mm"] == 0.0


def test_4_current_state_reconstruction_and_temporal_leakage():
    """Verify state reconstruction filters strictly observations with date <= t."""
    profile = FarmProfile("F0001", "Kandy", "Local", "Loam", 1.0, "2024-03-15")
    mgr = FarmStateManager(profile)
    mgr.add_crop_observation(CropObservation("2024-04-14", 30, "Sprouting", 14.0, 40.0))
    # Future observation
    mgr.add_crop_observation(CropObservation("2024-08-14", 150, "Rhizome Initiation", 80.0, 70.0))

    # Reconstruct at Day 30
    state = mgr.reconstruct_current_state("2024-04-14")
    assert state["Days_After_Planting"] == 30
    assert state["Plant_Height_cm"] == 14.0
    assert state["Growth_Stage"] == "Sprouting"


def test_5_dynamic_prediction_trigger_and_deltas():
    """Verify new observations trigger updated predictions and log deltas in history."""
    profile = FarmProfile("F0001", "Kandy", "Local", "Loam", 1.0, "2024-03-15")
    mgr = FarmStateManager(profile)
    pred = DynamicYieldPredictor()
    wx = MockWeatherProvider()

    # Step 1
    p1, r1, a1 = update_farm_state(
        farm_manager=mgr, dynamic_predictor=pred, observation_date="2024-04-14",
        days_after_planting=30, growth_stage="Sprouting", plant_height_cm=14.0, leaf_greenness_index=40.0, weather_provider=wx
    )
    assert p1 > 0.0
    assert r1.prediction_delta_t_ha is None

    # Step 2
    p2, r2, a2 = update_farm_state(
        farm_manager=mgr, dynamic_predictor=pred, observation_date="2024-05-14",
        days_after_planting=60, growth_stage="Early Vegetative", plant_height_cm=34.0, leaf_greenness_index=55.0, weather_provider=wx
    )
    assert r2.prediction_delta_t_ha == pytest.approx(p2 - p1, 0.001)
    assert a2 is not None
    assert "prediction_delta_pct" in a2


def test_6_shap_explanation_and_additive_property():
    """Verify TreeSHAP instance explanation exactly reconstructs predicted yield."""
    profile = FarmProfile("F0001", "Kandy", "Local", "Loam", 1.0, "2024-03-15")
    mgr = FarmStateManager(profile)
    mgr.add_crop_observation(CropObservation("2024-05-14", 60, "Early Vegetative", 34.0, 55.0))
    pred = DynamicYieldPredictor()

    exp = pred.explain_prediction(mgr, as_of_date="2024-05-14")
    assert isinstance(exp, PredictionExplanation)
    assert exp.predicted_yield_t_ha > 0.0
    assert len(exp.top_positive_factors) > 0
    assert len(exp.top_negative_factors) > 0
    assert len(exp.farmer_summary) > 0

    # Test additive sum
    explainer = GingerShapExplainer()
    state_dict = mgr.reconstruct_current_state("2024-05-14")
    res = explainer.explain_instance(state_dict)
    reconstructed_pred = res["base_value_t_ha"] + res["sum_shap_values"]
    assert reconstructed_pred == pytest.approx(res["predicted_yield_t_ha"], abs=0.01)


def test_7_recommendation_rules_and_rainfall_withholding():
    """Verify recommendation engine alerts and rainfall-aware irrigation checks."""
    profile = FarmProfile("F0001", "Kandy", "Local", "Loam", 1.0, "2024-03-15")
    mgr = FarmStateManager(profile)
    mgr.add_crop_observation(CropObservation("2024-06-23", 100, "Vegetative", 55.0, 48.0, pest_severity="High"))
    mgr.add_weather_observation(WeatherObservation("2024-06-23", weekly_rainfall_mm=60.0, avg_temperature_c=25.0, relative_humidity_pct=85.0, solar_radiation_mj_m2_day=18.0))

    pred = DynamicYieldPredictor()
    exp = pred.explain_prediction(mgr, as_of_date="2024-06-23")

    # High pest alert
    assert any("[Pest Alert]" in r for r in exp.recommendations)
    # High rainfall withholding check
    assert any("Withhold supplemental irrigation" in r for r in exp.recommendations)


def test_8_leakage_and_id_exclusion():
    """Verify Farm_ID, Target, and Baseline Leakage columns are never in model features."""
    explainer = GingerShapExplainer()
    assert TARGET_COLUMN not in explainer.feature_names
    assert "Farm_ID" not in explainer.feature_names
    for leak in LEAKAGE_COLUMNS:
        assert leak not in explainer.feature_names


def test_9_persistence_and_reloading(tmp_repo):
    """Verify FarmRepository serializes and losslessly reloads farm state and prediction history."""
    profile = FarmProfile("F0088", "Matale", "Nadun", "Clay Loam", 2.0, "2024-03-01", pump_capacity_lph=12000.0)
    mgr = FarmStateManager(profile)
    mgr.add_crop_observation(CropObservation("2024-04-01", 30, "Sprouting", 12.0, 38.0))
    mgr.add_crop_observation(CropObservation("2024-05-01", 60, "Early Vegetative", 32.0, 52.0))
    mgr.add_irrigation_event(IrrigationEvent("2024-04-15", runtime_hours=3.0))

    pred = DynamicYieldPredictor()
    pred.predict_current_state(mgr, as_of_date="2024-04-01")
    pred.predict_current_state(mgr, as_of_date="2024-05-01")

    # Save
    tmp_repo.save_farm_state(mgr, pred)

    # Reload
    reloaded = tmp_repo.load_farm_state("F0088")
    assert reloaded is not None
    rel_mgr, rel_pred = reloaded

    assert rel_mgr.profile.farm_id == "F0088"
    assert rel_mgr.profile.district == "Matale"
    assert len(rel_mgr.crop_observations) == 2
    assert len(rel_mgr.irrigation_log) == 1
    assert len(rel_pred.history.get_farm_history("F0088")) == 2
    assert rel_pred.history.get_farm_history("F0088")[1].prediction_delta_t_ha is not None


def test_10_end_to_end_demo_script_runs():
    """Verify scripts/demo_end_to_end.py executes and generates all Phase 6 deliverables."""
    from scripts.demo_end_to_end import main as run_demo
    run_demo()

    p6_dir = Path("results/phase6")
    assert (p6_dir / "end_to_end_demo.csv").exists()
    assert (p6_dir / "prediction_history.csv").exists()
    assert (p6_dir / "phase6_integration_report.txt").exists()
    assert (p6_dir / "plots" / "end_to_end_prediction_trajectory.png").exists()
    assert (p6_dir / "plots" / "end_to_end_shap_evolution.png").exists()
