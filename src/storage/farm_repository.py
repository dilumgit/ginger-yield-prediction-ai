"""Farm State Persistence and Repository Layer (Phase 6).

Provides lightweight JSON-based persistent storage for farm profiles,
historical crop observations, irrigation event logs, weather records,
and prediction histories across application sessions.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.data.farm_state import (
    CropObservation,
    FarmProfile,
    FarmStateManager,
    PredictionHistory,
    PredictionRecord,
    WeatherObservation,
)
from src.data.irrigation import IrrigationEvent
from src.dynamic.dynamic_predictor import DynamicYieldPredictor
from src.utils.config import DATA_DIR


class FarmRepository:
    """Persistent storage manager for longitudinal farm states."""

    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize repository with target storage directory.

        Parameters
        ----------
        storage_dir : Path, optional
            Directory to store farm state JSON files. Defaults to `data/farms/`.
        """
        self.storage_dir = Path(storage_dir) if storage_dir else DATA_DIR / "farms"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_farm_file_path(self, farm_id: str) -> Path:
        """Get sanitized filesystem path for a farm ID."""
        safe_id = "".join(c for c in farm_id if c.isalnum() or c in ("-", "_"))
        return self.storage_dir / f"{safe_id}.json"

    def save_farm_state(
        self,
        farm_manager: FarmStateManager,
        predictor: Optional[DynamicYieldPredictor] = None,
    ) -> Path:
        """Serialize and persist complete farm lifecycle state to JSON.

        Parameters
        ----------
        farm_manager : FarmStateManager
            The state manager holding profile and observations.
        predictor : DynamicYieldPredictor, optional
            The predictor holding historical prediction logs.

        Returns
        -------
        Path
            Path to the saved JSON file.
        """
        farm_id = farm_manager.profile.farm_id
        file_path = self._get_farm_file_path(farm_id)

        # 1. Profile
        profile_dict = asdict(farm_manager.profile)

        # 2. Crop Observations
        crop_obs_list = [asdict(obs) for obs in farm_manager.crop_observations]

        # 3. Irrigation Events
        irr_events_list = [asdict(ev) for ev in farm_manager.irrigation_log.events]

        # 4. Weather Observations
        wx_obs_list = [asdict(wx) for wx in farm_manager.weather_observations]

        # 5. Prediction History
        pred_history_list = []
        if predictor and predictor.history:
            for rec in predictor.history.get_farm_history(farm_id):
                pred_history_list.append(asdict(rec))

        payload = {
            "farm_id": farm_id,
            "profile": profile_dict,
            "crop_observations": crop_obs_list,
            "irrigation_events": irr_events_list,
            "weather_observations": wx_obs_list,
            "prediction_history": pred_history_list,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        return file_path

    def load_farm_state(
        self,
        farm_id: str,
        predictor: Optional[DynamicYieldPredictor] = None,
    ) -> Optional[Tuple[FarmStateManager, DynamicYieldPredictor]]:
        """Load and reconstruct complete farm state and prediction history from storage.

        Parameters
        ----------
        farm_id : str
            Farm identifier to load.
        predictor : DynamicYieldPredictor, optional
            Existing predictor instance to populate history into. If None, instantiates a new one.

        Returns
        -------
        Optional[Tuple[FarmStateManager, DynamicYieldPredictor]]
            Reconstructed (FarmStateManager, DynamicYieldPredictor) or None if farm not found.
        """
        file_path = self._get_farm_file_path(farm_id)
        if not file_path.exists():
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 1. Reconstruct Profile
        profile = FarmProfile(**data["profile"])
        farm_mgr = FarmStateManager(profile)

        # 2. Reconstruct Crop Observations
        for obs_d in data.get("crop_observations", []):
            farm_mgr.add_crop_observation(CropObservation(**obs_d))

        # 3. Reconstruct Irrigation Events
        for irr_d in data.get("irrigation_events", []):
            farm_mgr.add_irrigation_event(IrrigationEvent(**irr_d))

        # 4. Reconstruct Weather Observations
        for wx_d in data.get("weather_observations", []):
            farm_mgr.add_weather_observation(WeatherObservation(**wx_d))

        # 5. Reconstruct Predictor & Prediction History
        if predictor is None:
            predictor = DynamicYieldPredictor()

        for pred_d in data.get("prediction_history", []):
            # Avoid duplicate insertion if already in history
            existing = predictor.history.get_farm_history(farm_id)
            if not any(e.days_after_planting == pred_d["days_after_planting"] and e.predicted_yield_t_ha == pred_d["predicted_yield_t_ha"] for e in existing):
                rec = PredictionRecord(**pred_d)
                predictor.history._history.append(rec)

        return farm_mgr, predictor

    def list_farms(self) -> List[str]:
        """List all saved farm IDs in the repository."""
        return [f.stem for f in self.storage_dir.glob("*.json")]

    def delete_farm(
        self,
        farm_id: str,
        predictor: Optional[DynamicYieldPredictor] = None,
    ) -> bool:
        """Delete a farm record and all associated observations, irrigation events, and history.

        Parameters
        ----------
        farm_id : str
            Farm identifier to delete.
        predictor : DynamicYieldPredictor, optional
            Predictor instance to clean up in-memory prediction history for this farm.

        Returns
        -------
        bool
            True if the farm was successfully found and deleted, False otherwise.
        """
        if predictor is not None and predictor.history is not None:
            predictor.history.delete_farm_history(farm_id)

        file_path = self._get_farm_file_path(farm_id)
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def seed_demo_farms(self) -> List[str]:
        """Seed initial demonstration farms for interactive exploration if repository is empty."""
        existing = self.list_farms()
        if existing:
            return existing

        # Seed Farm F0001 (Kandy, Local Variety, Loam)
        prof1 = FarmProfile(
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
        mgr1 = FarmStateManager(prof1)
        mgr1.add_crop_observation(CropObservation(
            date="2024-04-14", days_after_planting=30, growth_stage="Sprouting", plant_height_cm=14.0, leaf_greenness_index=40.0,
            pest_severity="Low", disease_severity="Low", soil_moisture_pct=68.0, soil_ph=6.2, fertilizer_kg_acre=120.0
        ))
        mgr1.add_crop_observation(CropObservation(
            date="2024-05-14", days_after_planting=60, growth_stage="Early Vegetative", plant_height_cm=34.0, leaf_greenness_index=55.0,
            pest_severity="Low", disease_severity="Low", soil_moisture_pct=62.0, soil_ph=6.2, fertilizer_kg_acre=120.0
        ))
        pred1 = DynamicYieldPredictor()
        pred1.predict_current_state(mgr1, as_of_date="2024-04-14")
        pred1.predict_current_state(mgr1, as_of_date="2024-05-14")
        self.save_farm_state(mgr1, pred1)

        # Seed Farm F0042 (Kurunegala, Chinese Variety, Sandy Loam)
        prof2 = FarmProfile(
            farm_id="F0042",
            district="Kurunegala",
            seed_variety="Chinese",
            soil_type="Sandy Loam",
            land_size_acres=1.5,
            planting_date="2024-03-01",
            water_source="Stream",
            irrigation_method="Drip",
            pump_capacity_lph=10000.0,
        )
        mgr2 = FarmStateManager(prof2)
        mgr2.add_crop_observation(CropObservation(
            date="2024-04-15", days_after_planting=45, growth_stage="Sprouting", plant_height_cm=16.0, leaf_greenness_index=45.0,
            pest_severity="Low", disease_severity="Low", soil_moisture_pct=55.0, soil_ph=5.8, fertilizer_kg_acre=140.0
        ))
        pred2 = DynamicYieldPredictor()
        pred2.predict_current_state(mgr2, as_of_date="2024-04-15")
        self.save_farm_state(mgr2, pred2)

        return self.list_farms()
