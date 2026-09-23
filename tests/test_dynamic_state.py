"""Unit tests for Dynamic Farm State Management, Farmer-Friendly Irrigation, and Prediction History Tracking."""

import pytest
import numpy as np
import pandas as pd

from src.data.irrigation import (
    IrrigationEvent,
    IrrigationEventLog,
    runtime_to_irrigation_depth_mm,
)
from src.data.farm_state import (
    CropObservation,
    FarmProfile,
    FarmStateManager,
    PredictionHistory,
    PredictionRecord,
    WeatherObservation,
    calculate_dap,
    suggest_growth_stage,
)
from src.data.weather_api import (
    AbstractWeatherProvider,
    MockWeatherProvider,
    OpenMeteoWeatherProvider,
    WeatherAPIError,
)
from src.dynamic.dynamic_predictor import DynamicYieldPredictor
from src.dynamic.state_updater import update_farm_state
from src.models.predictor import GingerYieldPredictor
from src.utils.config import (
    BEST_MODEL_PATH,
    DATA_SOURCE_API,
    DATA_SOURCE_FARMER,
    DATA_SOURCE_SYNTHETIC,
    DISTRICT_COORDINATES,
    DISTRICTS,
    ID_COLUMNS,
    LEAKAGE_COLUMNS,
    TARGET_COLUMN,
)


def test_runtime_to_irrigation_depth_conversion():
    """Verify mathematical accuracy of pump runtime to mm conversion."""
    # 2.0 hours runtime, 10,000 L/hr pump capacity, 0.5 acres land size
    # Volume = 20,000 L
    # Area = 0.5 * 4046.86 = 2023.43 m²
    # Depth = 20000 / 2023.43 ≈ 9.8842 mm
    depth_mm = runtime_to_irrigation_depth_mm(
        runtime_hours=2.0,
        pump_capacity_lph=10000.0,
        land_size_acres=0.5,
    )
    assert depth_mm is not None
    assert depth_mm == pytest.approx(9.8842, 0.001)

    # When pump capacity is unknown (None), function returns None (runtime fallback)
    depth_none = runtime_to_irrigation_depth_mm(
        runtime_hours=2.0,
        pump_capacity_lph=None,
        land_size_acres=0.5,
    )
    assert depth_none is None


def test_irrigation_input_validation():
    """Verify validation boundaries for farmer-reported irrigation events."""
    # Valid event
    ev = IrrigationEvent(date="2024-05-15", runtime_hours=1.5, water_source="Well")
    assert ev.runtime_hours == 1.5

    # Negative runtime should fail
    with pytest.raises(ValueError, match="negative"):
        IrrigationEvent(date="2024-05-15", runtime_hours=-1.0)

    # Runtime > 24 hours in a single day should fail
    with pytest.raises(ValueError, match="exceeds 24 hours"):
        IrrigationEvent(date="2024-05-15", runtime_hours=26.0)

    # Invalid date format should fail
    with pytest.raises(ValueError, match="Invalid date format"):
        IrrigationEvent(date="invalid-date", runtime_hours=1.0)


def test_irrigation_event_log_preservation():
    """Verify that multiple irrigation events are preserved in history rather than overwritten."""
    log = IrrigationEventLog()
    log.add_event(IrrigationEvent(date="2024-05-10", runtime_hours=1.0))
    log.add_event(IrrigationEvent(date="2024-05-13", runtime_hours=1.5))
    log.add_event(IrrigationEvent(date="2024-05-17", runtime_hours=0.5))
    log.add_event(IrrigationEvent(date="2024-05-20", runtime_hours=2.0))

    assert len(log) == 4

    # Cumulative runtime as of May 20
    assert log.get_cumulative_runtime_hours("2024-05-20") == pytest.approx(5.0, 0.01)

    # Rolling 7-day runtime as of May 20 (includes May 13, 17, 20 = 1.5 + 0.5 + 2.0 = 4.0h)
    recent_7d = log.get_recent_runtime_hours("2024-05-20", window_days=7)
    assert recent_7d == pytest.approx(4.0, 0.01)


def test_farm_state_reconstruction_and_temporal_leakage_guard():
    """Verify farm state construction at time t strictly isolates future records (t_obs <= t)."""
    profile = FarmProfile(
        farm_id="F0999",
        district="Kandy",
        seed_variety="Local",
        soil_type="Loam",
        land_size_acres=1.0,
        planting_date="2024-03-01",
        pump_capacity_lph=8000.0,
    )
    mgr = FarmStateManager(profile)

    # Observation 1: Day 30 (Sprouting)
    mgr.add_crop_observation(CropObservation(
        date="2024-03-31",
        days_after_planting=30,
        growth_stage="Sprouting",
        plant_height_cm=15.0,
        leaf_greenness_index=40.0,
    ))

    # Observation 2: Day 90 (Vegetative) - In the future relative to Day 30
    mgr.add_crop_observation(CropObservation(
        date="2024-05-30",
        days_after_planting=90,
        growth_stage="Vegetative",
        plant_height_cm=45.0,
        leaf_greenness_index=65.0,
    ))

    # Irrigation event on Day 45 (after March 31, before May 30)
    mgr.add_irrigation_event(IrrigationEvent(date="2024-04-15", runtime_hours=2.0))

    # Reconstruct state as of Day 30 (2024-03-31)
    state_day30 = mgr.reconstruct_current_state(as_of_date="2024-03-31")

    # State at Day 30 MUST NOT contain Day 90 observations or April 15 irrigation
    assert state_day30["Days_After_Planting"] == 30
    assert state_day30["Growth_Stage"] == "Sprouting"
    assert state_day30["Plant_Height_cm"] == 15.0
    assert state_day30["Pump_Runtime_Hours"] == 0.0  # April 15 irrigation event excluded

    # Reconstruct state as of Day 90 (2024-05-30)
    state_day90 = mgr.reconstruct_current_state(as_of_date="2024-05-30")
    assert state_day90["Days_After_Planting"] == 90
    assert state_day90["Growth_Stage"] == "Vegetative"
    assert state_day90["Plant_Height_cm"] == 45.0
    assert state_day90["Cumulative_Irrigation_Hours"] == 2.0


def test_prediction_history_logging_and_deltas():
    """Verify prediction history logging, delta computation, and audit trail."""
    history = PredictionHistory()

    r1 = history.record_prediction(
        farm_id="F0100",
        days_after_planting=60,
        growth_stage="Early Vegetative",
        predicted_yield_t_ha=14.2,
        model_version="CatBoost_Baseline_v1",
        state_snapshot={"Plant_Height_cm": 25.0},
    )
    assert r1.prediction_delta_t_ha is None
    assert r1.predicted_yield_t_ha == 14.2

    # Farmer adds irrigation -> yield updates to 14.5 (+0.3 t/ha)
    r2 = history.record_prediction(
        farm_id="F0100",
        days_after_planting=65,
        growth_stage="Early Vegetative",
        predicted_yield_t_ha=14.5,
        model_version="CatBoost_Baseline_v1",
        state_snapshot={"Plant_Height_cm": 28.0, "Pump_Runtime_Hours": 2.0},
    )
    assert r2.prediction_delta_t_ha == pytest.approx(0.3, 0.001)

    # Later pest attack -> yield updates to 13.8 (-0.7 t/ha)
    r3 = history.record_prediction(
        farm_id="F0100",
        days_after_planting=80,
        growth_stage="Vegetative",
        predicted_yield_t_ha=13.8,
        model_version="CatBoost_Baseline_v1",
        state_snapshot={"Biotic_Stress_Index": 2},
    )
    assert r3.prediction_delta_t_ha == pytest.approx(-0.7, 0.001)

    # Verify history dataframe conversion
    df_hist = history.to_dataframe()
    assert len(df_hist) == 3
    assert "prediction_delta_t_ha" in df_hist.columns


def test_dynamic_predictor_state_inference():
    """Verify GingerYieldPredictor can predict directly from FarmStateManager."""
    predictor = GingerYieldPredictor(BEST_MODEL_PATH)
    history = PredictionHistory()

    profile = FarmProfile(
        farm_id="F0555",
        district="Kandy",
        seed_variety="Local",
        soil_type="Loam",
        land_size_acres=1.0,
        planting_date="2024-04-01",
    )
    mgr = FarmStateManager(profile)
    mgr.add_crop_observation(CropObservation(
        date="2024-06-01",
        days_after_planting=61,
        growth_stage="Early Vegetative",
        plant_height_cm=28.0,
        leaf_greenness_index=55.0,
        pest_severity="Low",
        disease_severity="Low",
    ))
    mgr.add_irrigation_event(IrrigationEvent(date="2024-05-28", runtime_hours=1.5))

    pred = predictor.predict_farm_state(mgr, history_tracker=history)
    assert isinstance(pred, float)
    assert pred > 5.0  # Realistic ginger yield in t/ha
    assert len(history.get_farm_history("F0555")) == 1


def test_mock_weather_provider():
    """Verify mock weather provider operates deterministically with correct source tagging."""
    provider = MockWeatherProvider(seed=42)
    weather = provider.get_weather_for_date("Kandy", "2024-05-15")

    assert weather["District"] == "Kandy"
    assert weather["Weekly_Rainfall_mm"] >= 0.0
    assert weather["Avg_Temperature_C"] > 15.0
    assert weather["Data_Source"] == DATA_SOURCE_SYNTHETIC


def test_calculate_dap_exact_and_types():
    """Verify calculate_dap computes accurate elapsed days across date types."""
    # Test case from requirements: Planting 2024-03-15, Observation 2024-05-29 -> 75 days
    dap = calculate_dap("2024-03-15", "2024-05-29")
    assert dap == 75

    # Same day observation -> 0 days
    assert calculate_dap("2024-03-15", "2024-03-15") == 0

    # Date object inputs
    from datetime import date, datetime
    d_plant = date(2024, 3, 15)
    d_obs = date(2024, 5, 29)
    assert calculate_dap(d_plant, d_obs) == 75

    # Datetime object inputs
    dt_plant = datetime(2024, 3, 15, 8, 30)
    dt_obs = datetime(2024, 5, 29, 14, 0)
    assert calculate_dap(dt_plant, dt_obs) == 75


def test_calculate_dap_rejects_earlier_observation_date():
    """Verify calculate_dap raises ValueError when observation date precedes planting date."""
    with pytest.raises(ValueError, match="cannot be earlier than Planting Date"):
        calculate_dap("2024-03-15", "2024-03-14")

    with pytest.raises(ValueError, match="cannot be earlier than Planting Date"):
        calculate_dap("2024-03-15", "2023-12-01")


def test_crop_observation_negative_dap_rejection():
    """Verify CropObservation dataclass validates non-negative DAP constraint."""
    # Non-negative DAP is valid
    obs_valid = CropObservation("2024-03-15", 0, "Sprouting", 10.0, 30.0)
    assert obs_valid.days_after_planting == 0

    # Negative DAP should be rejected
    with pytest.raises(ValueError, match="must be non-negative"):
        CropObservation("2024-03-14", -1, "Sprouting", 10.0, 30.0)


def test_suggest_growth_stage_boundaries():
    """Verify suggest_growth_stage returns sensible defaults matching canonical agro-ecological ranges."""
    # Sprouting: DAP 0 to 30
    assert suggest_growth_stage(0) == "Sprouting"
    assert suggest_growth_stage(15) == "Sprouting"
    assert suggest_growth_stage(30) == "Sprouting"

    # Early Vegetative: DAP 31 to 90
    assert suggest_growth_stage(31) == "Early Vegetative"
    assert suggest_growth_stage(75) == "Early Vegetative"
    assert suggest_growth_stage(90) == "Early Vegetative"

    # Vegetative: DAP 91 to 150
    assert suggest_growth_stage(91) == "Vegetative"
    assert suggest_growth_stage(120) == "Vegetative"
    assert suggest_growth_stage(150) == "Vegetative"

    # Rhizome Initiation: DAP 151 to 210
    assert suggest_growth_stage(151) == "Rhizome Initiation"
    assert suggest_growth_stage(180) == "Rhizome Initiation"
    assert suggest_growth_stage(210) == "Rhizome Initiation"

    # Rhizome Development: DAP 211 to 260
    assert suggest_growth_stage(211) == "Rhizome Development"
    assert suggest_growth_stage(240) == "Rhizome Development"
    assert suggest_growth_stage(260) == "Rhizome Development"

    # Maturity: DAP > 260
    assert suggest_growth_stage(261) == "Maturity"
    assert suggest_growth_stage(300) == "Maturity"


def test_update_farm_state_automatic_dap_and_pipeline():
    """Verify update_farm_state computes DAP automatically when None and propagates to predictions."""
    profile = FarmProfile(
        farm_id="F0123",
        district="Kandy",
        seed_variety="Local",
        soil_type="Loam",
        land_size_acres=1.0,
        planting_date="2024-03-15",
    )
    mgr = FarmStateManager(profile)
    pred = DynamicYieldPredictor()

    # Ingest observation without manual DAP or Growth Stage
    pred_val, rec, analysis = update_farm_state(
        farm_manager=mgr,
        dynamic_predictor=pred,
        observation_date="2024-05-29",
        days_after_planting=None,  # Farmer does NOT enter DAP
        growth_stage=None,         # System suggests stage
        plant_height_cm=42.0,
        leaf_greenness_index=58.0,
    )

    # Verify auto-calculated DAP and suggested stage
    assert rec.days_after_planting == 75
    assert rec.growth_stage == "Early Vegetative"
    assert len(mgr.crop_observations) == 1
    assert mgr.crop_observations[0].days_after_planting == 75
    assert mgr.crop_observations[0].growth_stage == "Early Vegetative"

    # Verify prediction output and record
    assert isinstance(pred_val, float)
    assert pred_val > 5.0
    assert rec.predicted_yield_t_ha == pytest.approx(pred_val, 0.001)

    # Ingest second observation with auto DAP
    pred_val2, rec2, analysis2 = update_farm_state(
        farm_manager=mgr,
        dynamic_predictor=pred,
        observation_date="2024-06-23",
        days_after_planting=None,
        plant_height_cm=55.0,
        leaf_greenness_index=60.0,
    )
    # 2024-03-15 to 2024-06-23 is 100 days
    assert rec2.days_after_planting == 100
    assert rec2.growth_stage == "Vegetative"
    assert len(mgr.crop_observations) == 2
    # Ensure first record was preserved unchanged
    assert mgr.crop_observations[0].days_after_planting == 75
    assert mgr.crop_observations[0].plant_height_cm == 42.0


def test_update_farm_state_rejects_earlier_observation_date():
    """Verify update_farm_state raises ValueError for observation date earlier than planting date."""
    profile = FarmProfile(
        farm_id="F0124",
        district="Kandy",
        seed_variety="Local",
        soil_type="Loam",
        land_size_acres=1.0,
        planting_date="2024-03-15",
    )
    mgr = FarmStateManager(profile)
    pred = DynamicYieldPredictor()

    with pytest.raises(ValueError, match="cannot be earlier than Planting Date"):
        update_farm_state(
            farm_manager=mgr,
            dynamic_predictor=pred,
            observation_date="2024-03-10",
            plant_height_cm=10.0,
        )


def test_district_coordinates_resolution():
    """Verify all 10 canonical Sri Lankan districts map to valid coordinates."""
    provider = OpenMeteoWeatherProvider()
    for district in DISTRICTS:
        assert district in DISTRICT_COORDINATES
        lat, lon = provider._get_district_coordinates(district)
        assert 5.0 <= lat <= 10.0  # Sri Lankan latitude bounds
        assert 79.0 <= lon <= 82.0  # Sri Lankan longitude bounds

    # Case-insensitive resolution
    assert provider._get_district_coordinates("kandy") == DISTRICT_COORDINATES["Kandy"]

    # Unknown district raises WeatherAPIError
    with pytest.raises(WeatherAPIError, match="Unknown or unsupported"):
        provider._get_district_coordinates("UnknownDistrict123")


def test_open_meteo_historical_rainfall_and_temporal_bounds():
    """Verify OpenMeteoWeatherProvider retrieves previous 7-day rainfall and excludes future dates."""
    provider = OpenMeteoWeatherProvider()
    obs_date = "2024-05-29"

    weather = provider.get_weather_for_date("Kandy", obs_date)

    assert weather["District"] == "Kandy"
    assert weather["Date"] == obs_date
    assert weather["Window_End"] == obs_date
    assert weather["Window_Start"] == "2024-05-23"
    assert weather["Data_Source"] == DATA_SOURCE_API
    assert weather["Source"] == "Open-Meteo"
    assert isinstance(weather["Weekly_Rainfall_mm"], float)
    assert weather["Weekly_Rainfall_mm"] >= 0.0

    # 7 daily precipitation values
    daily_precip = weather.get("Daily_Precipitation", [])
    assert len(daily_precip) == 7
    assert weather["Weekly_Rainfall_mm"] == pytest.approx(sum(daily_precip), 0.01)

    # Strictly verify zero future temporal leakage: all retrieved dates <= 2024-05-29
    daily_dates = weather.get("Daily_Dates", [])
    assert len(daily_dates) == 7
    from datetime import datetime
    cutoff = datetime.strptime(obs_date, "%Y-%m-%d").date()
    for d in daily_dates:
        dt = datetime.strptime(d, "%Y-%m-%d").date()
        assert dt <= cutoff, f"Temporal leakage: date {d} is greater than observation date {obs_date}"


def test_weather_provider_caching():
    """Verify weather provider caches responses in-memory to prevent repeated network calls."""
    provider = OpenMeteoWeatherProvider()
    obs_date = "2024-05-29"

    # First call - populates cache
    res1 = provider.get_weather_for_date("Kandy", obs_date)
    cache_key = ("Kandy", "2024-05-23", "2024-05-29")
    assert cache_key in provider._cache

    # Second call - retrieves from cache
    res2 = provider.get_weather_for_date("Kandy", obs_date)
    assert res1 is res2  # Identical cached object


def test_weather_api_error_handling():
    """Verify structured error handling for invalid parameters and endpoints."""
    # Bad endpoint should raise WeatherAPIError gracefully
    provider = OpenMeteoWeatherProvider(archive_url="https://invalid-non-existent-domain.xyz/api")
    with pytest.raises(WeatherAPIError):
        provider.get_weather_for_date("Kandy", "2024-05-29")

    # Invalid district
    normal_provider = OpenMeteoWeatherProvider()
    with pytest.raises(WeatherAPIError, match="Unknown or unsupported"):
        normal_provider.get_weather_for_date("Atlantis", "2024-05-29")


def test_update_farm_state_with_weather_pipeline():
    """Verify update_farm_state ingests weather from weather_provider into state and model prediction."""
    profile = FarmProfile(
        farm_id="F0125",
        district="Kandy",
        seed_variety="Local",
        soil_type="Loam",
        land_size_acres=1.0,
        planting_date="2024-03-15",
    )
    mgr = FarmStateManager(profile)
    pred = DynamicYieldPredictor()
    wx_provider = OpenMeteoWeatherProvider()

    pred_val, rec, analysis = update_farm_state(
        farm_manager=mgr,
        dynamic_predictor=pred,
        observation_date="2024-05-29",
        plant_height_cm=42.0,
        leaf_greenness_index=58.0,
        weather_provider=wx_provider,
    )

    assert len(mgr.weather_observations) == 1
    wx_obs = mgr.weather_observations[0]
    assert wx_obs.date == "2024-05-29"
    assert wx_obs.data_source == DATA_SOURCE_API
    assert wx_obs.weekly_rainfall_mm >= 0.0

    state = mgr.reconstruct_current_state("2024-05-29")
    assert state["Weekly_Rainfall_mm"] == wx_obs.weekly_rainfall_mm
    assert state["Total_Water_Input_mm"] == state["Weekly_Rainfall_mm"] + state["Irrigation_mm_week"]
    assert isinstance(pred_val, float)


def test_streamlit_ui_no_manual_rainfall_input():
    """Verify Streamlit Update Crop Observation interface contains no manual rainfall input."""
    from streamlit.testing.v1 import AppTest
    from datetime import date

    at = AppTest.from_file("src/ui/app.py", default_timeout=30)
    at.run()
    at.sidebar.radio[0].set_value("📝 Update Crop Observation").run()

    # Verify no editable number input for rainfall exists
    number_input_labels = [n.label for n in at.get("number_input")]
    for lbl in number_input_labels:
        assert "rainfall" not in lbl.lower(), f"Found manual rainfall number input: {lbl}"

    # Verify read-only rainfall text display exists
    text_inputs = [t for t in at.get("text_input") if "Rainfall" in t.label]
    assert len(text_inputs) == 1, "Expected read-only rainfall display widget"
    assert text_inputs[0].disabled is True, "Rainfall display must be read-only (disabled)"


