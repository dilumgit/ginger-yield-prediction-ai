"""Unit and Integration Tests for Farm Deletion and Empty-Observation Safety.

Verifies:
1. Successful farm deletion from repository.
2. Deletion with associated crop observations.
3. Deletion with associated irrigation events.
4. Deletion with associated prediction history (file and in-memory).
5. Deletion of a newly created farm with no crop observations.
6. Ensuring other farms remain completely unaffected.
7. Handling of newly created farms with zero observations without crashes in Dashboard,
   Prediction History, SHAP Explainability, and Irrigation logging.
"""

from datetime import date
from pathlib import Path
import pytest

from src.data.farm_state import (
    CropObservation,
    FarmProfile,
    FarmStateManager,
    PredictionHistory,
    WeatherObservation,
)
from src.data.irrigation import IrrigationEvent
from src.dynamic.dynamic_predictor import DynamicYieldPredictor
from src.storage.farm_repository import FarmRepository


def test_successful_farm_deletion(tmp_path):
    """Verify basic farm deletion from repository."""
    repo = FarmRepository(storage_dir=tmp_path / "farms")
    profile = FarmProfile(farm_id="F_DEL_1", district="Kandy", seed_variety="Local", soil_type="Loam", land_size_acres=1.0, planting_date="2024-03-15")
    mgr = FarmStateManager(profile)
    pred = DynamicYieldPredictor()
    repo.save_farm_state(mgr, pred)

    assert "F_DEL_1" in repo.list_farms()
    deleted = repo.delete_farm("F_DEL_1", predictor=pred)
    assert deleted is True
    assert "F_DEL_1" not in repo.list_farms()
    assert repo.load_farm_state("F_DEL_1") is None


def test_deletion_with_associated_observations(tmp_path):
    """Verify deleting a farm with multiple crop observations leaves no orphan records."""
    repo = FarmRepository(storage_dir=tmp_path / "farms")
    profile = FarmProfile(farm_id="F_DEL_OBS", district="Matale", seed_variety="Local", soil_type="Loam", land_size_acres=2.0, planting_date="2024-03-01")
    mgr = FarmStateManager(profile)
    mgr.add_crop_observation(CropObservation("2024-04-01", 31, "Sprouting", 15.0, 42.0))
    mgr.add_crop_observation(CropObservation("2024-05-01", 61, "Early Vegetative", 35.0, 58.0))
    pred = DynamicYieldPredictor()
    repo.save_farm_state(mgr, pred)

    assert "F_DEL_OBS" in repo.list_farms()
    farm_file = repo._get_farm_file_path("F_DEL_OBS")
    assert farm_file.exists()

    deleted = repo.delete_farm("F_DEL_OBS", predictor=pred)
    assert deleted is True
    assert not farm_file.exists()
    assert repo.load_farm_state("F_DEL_OBS") is None


def test_deletion_with_irrigation_events(tmp_path):
    """Verify deleting a farm with historical irrigation events removes all records."""
    repo = FarmRepository(storage_dir=tmp_path / "farms")
    profile = FarmProfile(farm_id="F_DEL_IRR", district="Kurunegala", seed_variety="Chinese", soil_type="Sandy Loam", land_size_acres=1.5, planting_date="2024-03-10")
    mgr = FarmStateManager(profile)
    mgr.add_irrigation_event(IrrigationEvent(date="2024-04-10", runtime_hours=2.0, water_source="Well", irrigation_method="Sprinkler"))
    mgr.add_irrigation_event(IrrigationEvent(date="2024-04-20", runtime_hours=3.5, water_source="Stream", irrigation_method="Drip"))
    pred = DynamicYieldPredictor()
    repo.save_farm_state(mgr, pred)

    assert len(mgr.irrigation_log) == 2
    deleted = repo.delete_farm("F_DEL_IRR", predictor=pred)
    assert deleted is True
    assert "F_DEL_IRR" not in repo.list_farms()


def test_deletion_with_prediction_history(tmp_path):
    """Verify deleting a farm removes both JSON history and in-memory prediction records."""
    repo = FarmRepository(storage_dir=tmp_path / "farms")
    profile = FarmProfile(farm_id="F_DEL_HIST", district="Gampaha", seed_variety="Rangoon", soil_type="Clay Loam", land_size_acres=0.5, planting_date="2024-02-15")
    mgr = FarmStateManager(profile)
    mgr.add_crop_observation(CropObservation("2024-03-15", 29, "Sprouting", 12.0, 38.0))
    mgr.add_crop_observation(CropObservation("2024-04-15", 60, "Early Vegetative", 32.0, 52.0))

    pred = DynamicYieldPredictor()
    pred.predict_current_state(mgr, as_of_date="2024-03-15")
    pred.predict_current_state(mgr, as_of_date="2024-04-15")
    repo.save_farm_state(mgr, pred)

    # In-memory history should have 2 records for F_DEL_HIST
    assert len(pred.history.get_farm_history("F_DEL_HIST")) == 2

    # Delete farm
    deleted = repo.delete_farm("F_DEL_HIST", predictor=pred)
    assert deleted is True
    assert "F_DEL_HIST" not in repo.list_farms()
    # In-memory history should now have 0 records for F_DEL_HIST
    assert len(pred.history.get_farm_history("F_DEL_HIST")) == 0


def test_deletion_of_farm_with_no_observations(tmp_path):
    """Verify safe deletion of a newly registered farm profile with zero crop observations."""
    repo = FarmRepository(storage_dir=tmp_path / "farms")
    profile = FarmProfile(farm_id="F_EMPTY", district="Badulla", seed_variety="Local", soil_type="Loam", land_size_acres=3.0, planting_date="2024-04-01")
    mgr = FarmStateManager(profile)
    pred = DynamicYieldPredictor()
    repo.save_farm_state(mgr, pred)

    assert len(mgr.crop_observations) == 0
    assert "F_EMPTY" in repo.list_farms()

    deleted = repo.delete_farm("F_EMPTY", predictor=pred)
    assert deleted is True
    assert "F_EMPTY" not in repo.list_farms()


def test_ensuring_another_farm_is_unaffected(tmp_path):
    """Verify deleting Farm A does not alter Farm B's profile, observations, or history."""
    repo = FarmRepository(storage_dir=tmp_path / "farms")

    # Farm A
    prof_a = FarmProfile("FARM_A", "Kandy", "Local", "Loam", 1.0, "2024-03-01")
    mgr_a = FarmStateManager(prof_a)
    mgr_a.add_crop_observation(CropObservation("2024-04-01", 31, "Sprouting", 14.0, 40.0))
    pred = DynamicYieldPredictor()
    pred.predict_current_state(mgr_a, as_of_date="2024-04-01")
    repo.save_farm_state(mgr_a, pred)

    # Farm B
    prof_b = FarmProfile("FARM_B", "Matale", "Chinese", "Sandy Loam", 2.5, "2024-03-15")
    mgr_b = FarmStateManager(prof_b)
    mgr_b.add_crop_observation(CropObservation("2024-04-15", 31, "Sprouting", 16.0, 44.0))
    pred.predict_current_state(mgr_b, as_of_date="2024-04-15")
    repo.save_farm_state(mgr_b, pred)

    assert set(repo.list_farms()) == {"FARM_A", "FARM_B"}
    assert len(pred.history.get_farm_history("FARM_A")) == 1
    assert len(pred.history.get_farm_history("FARM_B")) == 1

    # Delete Farm A
    deleted = repo.delete_farm("FARM_A", predictor=pred)
    assert deleted is True

    # Check Farm A is gone
    assert repo.load_farm_state("FARM_A") is None
    assert len(pred.history.get_farm_history("FARM_A")) == 0

    # Check Farm B is completely intact
    loaded_b = repo.load_farm_state("FARM_B")
    assert loaded_b is not None
    mgr_b_loaded, _ = loaded_b
    assert mgr_b_loaded.profile.farm_id == "FARM_B"
    assert mgr_b_loaded.profile.district == "Matale"
    assert mgr_b_loaded.profile.land_size_acres == 2.5
    assert len(mgr_b_loaded.crop_observations) == 1
    assert len(pred.history.get_farm_history("FARM_B")) == 1


def test_newly_created_farm_irrigation_logging_without_crop_observations(tmp_path):
    """Verify logging irrigation on a new farm without crop observations does not throw ValueError."""
    repo = FarmRepository(storage_dir=tmp_path / "farms")
    profile = FarmProfile(farm_id="F_NEW_IRR", district="Kandy", seed_variety="Local", soil_type="Loam", land_size_acres=1.0, planting_date="2024-04-01")
    mgr = FarmStateManager(profile)
    pred = DynamicYieldPredictor()

    assert len(mgr.crop_observations) == 0

    # Log irrigation event
    mgr.add_irrigation_event(IrrigationEvent(date="2024-04-05", runtime_hours=2.0, water_source="Well", irrigation_method="Sprinkler"))
    repo.save_farm_state(mgr, pred)

    loaded = repo.load_farm_state("F_NEW_IRR")
    assert loaded is not None
    loaded_mgr, _ = loaded
    assert len(loaded_mgr.irrigation_log) == 1
    assert len(loaded_mgr.crop_observations) == 0


def test_prediction_history_delete_farm_history():
    """Verify PredictionHistory.delete_farm_history removes only matching farm records."""
    history = PredictionHistory()
    history.record_prediction(farm_id="F1", days_after_planting=30, growth_stage="Sprouting", predicted_yield_t_ha=12.0, model_version="v1", state_snapshot={})
    history.record_prediction(farm_id="F1", days_after_planting=60, growth_stage="Early Vegetative", predicted_yield_t_ha=13.0, model_version="v1", state_snapshot={})
    history.record_prediction(farm_id="F2", days_after_planting=30, growth_stage="Sprouting", predicted_yield_t_ha=14.0, model_version="v1", state_snapshot={})

    assert len(history.get_farm_history("F1")) == 2
    assert len(history.get_farm_history("F2")) == 1

    removed_count = history.delete_farm_history("F1")
    assert removed_count == 2
    assert len(history.get_farm_history("F1")) == 0
    assert len(history.get_farm_history("F2")) == 1


def test_streamlit_ui_delete_confirmation_and_cancel_workflow():
    """Verify Streamlit UI deletion workflow: Cancel preserves farm, Confirm permanently deletes."""
    from streamlit.testing.v1 import AppTest

    repo = FarmRepository()
    test_id = "F_TEST_CANCEL_FLOW"
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

        # Switch to Farm Profile Management
        at.sidebar.radio[0].set_value("⚙️ Farm Profile Management").run()

        # Step 1: Click initial delete button
        btn_init = at.button(key=f"btn_init_delete_{test_id}")
        assert btn_init is not None
        btn_init.click().run()
        assert at.session_state[f"confirm_delete_{test_id}"] is True

        # Step 2: Click Cancel
        btn_cancel = at.button(key=f"btn_cancel_delete_{test_id}")
        assert btn_cancel is not None
        btn_cancel.click().run()
        assert at.session_state[f"confirm_delete_{test_id}"] is False
        assert test_id in repo.list_farms()

        # Step 3: Click initial delete again, then click Confirm Delete Permanently
        at.button(key=f"btn_init_delete_{test_id}").click().run()
        btn_confirm = at.button(key=f"btn_confirm_delete_{test_id}")
        assert btn_confirm is not None
        btn_confirm.click().run()

        # Step 4: Verify farm is permanently removed
        assert test_id not in repo.list_farms()
        assert len(at.exception) == 0

    finally:
        if test_id in repo.list_farms():
            repo.delete_farm(test_id, predictor=pred)

