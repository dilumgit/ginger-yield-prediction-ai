"""Unit Tests for Streamlit UI Module (Phase 6).

Verifies:
1. UI helper functions and repository initialization.
2. Form actions: Crop observation update and dynamic prediction trigger.
3. Form actions: Irrigation event logging and state updates.
4. Form actions: New FarmProfile creation and persistence.
5. SHAP factor attribution plotting logic.
"""

from datetime import date
from pathlib import Path
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.farm_state import (
    CropObservation,
    FarmProfile,
    FarmStateManager,
    calculate_dap,
    suggest_growth_stage,
)
from src.data.irrigation import IrrigationEvent
from src.data.weather_api import MockWeatherProvider
from src.dynamic.dynamic_predictor import DynamicYieldPredictor
from src.dynamic.state_updater import update_farm_state
from src.storage.farm_repository import FarmRepository
from src.ui.app import get_repository


def test_ui_get_repository(tmp_path, monkeypatch):
    """Verify get_repository initializes repository and seeds demo farms."""
    monkeypatch.setattr(
        "src.ui.app.FarmRepository",
        lambda *args, **kwargs: FarmRepository(storage_dir=tmp_path / "farms"),
    )
    repo = get_repository()
    assert isinstance(repo, FarmRepository)
    farms = repo.list_farms()
    assert len(farms) > 0
    assert "F0001" in farms


def test_ui_crop_observation_submission():
    """Verify crop observation update workflow executed in UI."""
    profile = FarmProfile("F_TEST_UI", "Kandy", "Local", "Loam", 1.5, "2024-03-15")
    mgr = FarmStateManager(profile)
    pred = DynamicYieldPredictor()
    wx = MockWeatherProvider(seed=42)

    dap = calculate_dap("2024-03-15", "2024-04-14")
    stage = suggest_growth_stage(dap)

    pred_val, rec, analysis = update_farm_state(
        farm_manager=mgr,
        dynamic_predictor=pred,
        observation_date="2024-04-14",
        days_after_planting=dap,
        growth_stage=stage,
        plant_height_cm=14.0,
        leaf_greenness_index=40.0,
        pest_severity="None",
        disease_severity="None",
        soil_moisture_pct=65.0,
        fertilizer_kg_acre=120.0,
        weather_provider=wx,
    )

    assert pred_val > 0.0
    assert len(mgr.crop_observations) == 1
    assert mgr.crop_observations[0].plant_height_cm == 14.0


def test_ui_irrigation_event_logging(tmp_path):
    """Verify irrigation event logging workflow executed in UI."""
    repo = FarmRepository(storage_dir=tmp_path / "farms")
    profile = FarmProfile("F_TEST_IRR", "Kandy", "Local", "Loam", 1.0, "2024-03-15", pump_capacity_lph=8000.0)
    mgr = FarmStateManager(profile)
    mgr.add_crop_observation(CropObservation("2024-04-14", 30, "Sprouting", 14.0, 40.0))
    pred = DynamicYieldPredictor()

    # Log irrigation event
    ev = IrrigationEvent(
        date="2024-04-15",
        runtime_hours=2.5,
        water_source="Well",
        irrigation_method="Sprinkler",
    )
    mgr.add_irrigation_event(ev)
    pred_val, rec, analysis = pred.predict_current_state(mgr, as_of_date="2024-04-15")
    repo.save_farm_state(mgr, pred)

    assert len(mgr.irrigation_log) == 1
    assert pred_val > 0.0

    # Reload from repo
    loaded = repo.load_farm_state("F_TEST_IRR")
    assert loaded is not None
    loaded_mgr, loaded_pred = loaded
    assert len(loaded_mgr.irrigation_log) == 1


def test_ui_farm_profile_creation(tmp_path):
    """Verify new farm profile creation flow in UI."""
    repo = FarmRepository(storage_dir=tmp_path / "farms")
    new_profile = FarmProfile(
        farm_id="F9999",
        district="Kurunegala",
        seed_variety="Chinese",
        soil_type="Sandy Loam",
        land_size_acres=2.5,
        planting_date="2024-04-01",
        water_source="Rainwater Tank",
        irrigation_method="Drip",
        pump_capacity_lph=10000.0,
    )
    new_mgr = FarmStateManager(new_profile)
    new_pred = DynamicYieldPredictor()
    repo.save_farm_state(new_mgr, new_pred)

    assert "F9999" in repo.list_farms()
    loaded = repo.load_farm_state("F9999")
    assert loaded is not None
    assert loaded[0].profile.district == "Kurunegala"


def test_irrigation_event_provenance_field():
    """Verify IrrigationEvent data_source defaults to FARMER_REPORTED and accepts custom provenance."""
    from src.utils.config import DATA_SOURCE_FARMER, DATA_SOURCE_SYNTHETIC
    import pandas as pd

    # Default creation
    ev_default = IrrigationEvent(date="2024-05-01", runtime_hours=2.0)
    assert hasattr(ev_default, "data_source")
    assert ev_default.data_source == DATA_SOURCE_FARMER

    # Explicit provenance
    ev_custom = IrrigationEvent(date="2024-05-01", runtime_hours=2.0, data_source=DATA_SOURCE_SYNTHETIC)
    assert ev_custom.data_source == DATA_SOURCE_SYNTHETIC


def test_ui_irrigation_history_table_generation_with_data_source():
    """Verify the Streamlit UI historical irrigation table construction does not raise AttributeError."""
    import pandas as pd
    from src.data.irrigation import IrrigationEventLog

    log = IrrigationEventLog()
    log.add_event(IrrigationEvent(date="2024-05-01", runtime_hours=2.0, water_source="Well", irrigation_method="Sprinkler"))
    log.add_event(IrrigationEvent(date="2024-05-15", runtime_hours=3.5, water_source="Stream", irrigation_method="Drip"))

    events = log.events
    assert len(events) == 2

    # Exactly mirror the UI construction in app.py View 3
    df_irr = pd.DataFrame([
        {
            "Date": ev.date,
            "Runtime (Hours)": ev.runtime_hours,
            "Water Source": ev.water_source,
            "Method": ev.irrigation_method,
            "Data Source": ev.data_source,
        }
        for ev in events
    ])

    assert len(df_irr) == 2
    assert "Data Source" in df_irr.columns
    assert list(df_irr["Data Source"]) == ["FARMER_REPORTED", "FARMER_REPORTED"]


def test_irrigation_backward_compatibility_without_data_source_in_json(tmp_path):
    """Verify legacy JSON files without data_source key in irrigation_events load losslessly."""
    import json
    from src.utils.config import DATA_SOURCE_FARMER

    repo = FarmRepository(storage_dir=tmp_path / "farms")
    legacy_payload = {
        "farm_id": "F_LEGACY",
        "profile": {
            "farm_id": "F_LEGACY",
            "district": "Kandy",
            "seed_variety": "Local",
            "soil_type": "Loam",
            "land_size_acres": 1.0,
            "planting_date": "2024-03-15",
            "pump_capacity_lph": 8000.0,
            "water_source": "Well",
            "irrigation_method": "Sprinkler",
        },
        "crop_observations": [],
        "irrigation_events": [
            {
                "date": "2024-05-29",
                "runtime_hours": 2.5,
                "water_source": "Well",
                "irrigation_method": "Sprinkler",
                "notes": None,
            }
        ],
        "weather_observations": [],
        "prediction_history": [],
    }

    legacy_file = tmp_path / "farms" / "F_LEGACY.json"
    with open(legacy_file, "w", encoding="utf-8") as f:
        json.dump(legacy_payload, f)

    loaded = repo.load_farm_state("F_LEGACY")
    assert loaded is not None
    mgr, pred = loaded
    assert len(mgr.irrigation_log) == 1
    reconstructed_event = mgr.irrigation_log.events[0]
    assert hasattr(reconstructed_event, "data_source")
    assert reconstructed_event.data_source == DATA_SOURCE_FARMER
    assert reconstructed_event.runtime_hours == 2.5


def test_apptest_newly_created_farm_with_zero_observations(tmp_path):
    """Verify Streamlit app renders a newly created farm without observations across views with 0 exceptions."""
    from streamlit.testing.v1 import AppTest

    repo = FarmRepository()
    test_id = "F_TEST_EMPTY_UI"
    prof = FarmProfile(farm_id=test_id, district="Kandy", seed_variety="Local", soil_type="Loam", land_size_acres=1.0, planting_date="2024-03-15")
    mgr = FarmStateManager(prof)
    pred = DynamicYieldPredictor()
    repo.save_farm_state(mgr, pred)

    try:
        app_path = Path(__file__).resolve().parents[1] / "src" / "ui" / "app.py"
        at = AppTest.from_file(str(app_path), default_timeout=30)
        at.run()
        if test_id in at.sidebar.selectbox[0].options:
            at.sidebar.selectbox[0].set_value(test_id).run()

        # View 1: Farm Dashboard (empty observations)
        assert len(at.exception) == 0

        # View 4: Prediction History
        at.sidebar.radio[0].set_value("📈 Prediction History & Trajectory").run()
        assert len(at.exception) == 0

        # View 5: Explain Prediction (SHAP XAI)
        at.sidebar.radio[0].set_value("🔍 Explain Prediction (SHAP XAI)").run()
        assert len(at.exception) == 0

        # View 3: Log Irrigation Event
        at.sidebar.radio[0].set_value("💧 Log Irrigation Event").run()
        assert len(at.exception) == 0

        # View 6: Farm Profile Management
        at.sidebar.radio[0].set_value("⚙️ Farm Profile Management").run()
        assert len(at.exception) == 0

    finally:
        repo.delete_farm(test_id, predictor=pred)


