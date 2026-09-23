"""Farm State Management and Dynamic In-Season Prediction History Tracker.

This module formalizes the dynamic research concept:
- A farmer can provide new/current farm observations at any time during cultivation.
- The system stores historical observations, incorporates newly provided information,
  updates the current farm state, and triggers an updated final yield forecast.
- Complete prediction history is recorded along with explanation metadata and prediction deltas.
- ZERO future-data leakage is enforced at state reconstruction time.
"""

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.data.irrigation import IrrigationEvent, IrrigationEventLog
from src.utils.config import (
    DATA_SOURCE_API,
    DATA_SOURCE_FARMER,
    DATA_SOURCE_HISTORICAL,
    DISTRICTS,
    GROWTH_STAGES,
    NOMINAL_CATEGORICAL_COLUMNS,
    ORDINAL_MAPPINGS,
    SEED_VARIETIES,
    SOIL_TYPES,
)


def calculate_dap(
    planting_date: Union[str, date, datetime],
    observation_date: Union[str, date, datetime],
) -> int:
    """Calculate Days After Planting (DAP) from planting date and observation date.

    DAP = observation_date - planting_date (in elapsed calendar days).

    Parameters
    ----------
    planting_date : str, date, or datetime
        Date the crop was planted (YYYY-MM-DD or date object).
    observation_date : str, date, or datetime
        Date the in-season observation occurred (YYYY-MM-DD or date object).

    Returns
    -------
    int
        Elapsed days after planting.

    Raises
    ------
    ValueError
        If observation_date is earlier than planting_date (resulting in negative DAP).
    TypeError
        If input dates cannot be parsed.
    """
    if isinstance(planting_date, str):
        p_dt = datetime.strptime(str(planting_date)[:10], "%Y-%m-%d").date()
    elif isinstance(planting_date, datetime):
        p_dt = planting_date.date()
    elif isinstance(planting_date, date):
        p_dt = planting_date
    else:
        raise TypeError(f"Expected str, date, or datetime for planting_date, got {type(planting_date)}")

    if isinstance(observation_date, str):
        o_dt = datetime.strptime(str(observation_date)[:10], "%Y-%m-%d").date()
    elif isinstance(observation_date, datetime):
        o_dt = observation_date.date()
    elif isinstance(observation_date, date):
        o_dt = observation_date
    else:
        raise TypeError(f"Expected str, date, or datetime for observation_date, got {type(observation_date)}")

    dap = (o_dt - p_dt).days
    if dap < 0:
        raise ValueError(
            f"Observation Date ({o_dt}) cannot be earlier than Planting Date ({p_dt}). "
            f"Calculated DAP ({dap}) cannot be negative."
        )
    return dap


def suggest_growth_stage(dap: int) -> str:
    """Provide a sensible default/suggested growth stage based on calculated DAP.

    Agronomic Stage Boundaries (Sri Lankan ginger cultivation):
    - DAP 0 - 30: Sprouting
    - DAP 31 - 90: Early Vegetative
    - DAP 91 - 150: Vegetative
    - DAP 151 - 210: Rhizome Initiation
    - DAP 211 - 260: Rhizome Development
    - DAP > 260: Maturity

    Parameters
    ----------
    dap : int
        Elapsed days after planting.

    Returns
    -------
    str
        Suggested growth stage conforming to GROWTH_STAGES.
    """
    if dap <= 30:
        return "Sprouting"
    elif dap <= 90:
        return "Early Vegetative"
    elif dap <= 150:
        return "Vegetative"
    elif dap <= 210:
        return "Rhizome Initiation"
    elif dap <= 260:
        return "Rhizome Development"
    else:
        return "Maturity"


@dataclass
class FarmProfile:
    """Static metadata profile for a ginger farm."""

    farm_id: str
    district: str
    seed_variety: str
    soil_type: str
    land_size_acres: float
    planting_date: str
    pump_capacity_lph: Optional[float] = None  # Optional pump capacity (L/hour)
    water_source: str = "Well"
    irrigation_method: str = "Sprinkler"


@dataclass
class CropObservation:
    """In-season observation recorded by farmer or historical dataset."""

    date: str  # YYYY-MM-DD
    days_after_planting: int
    growth_stage: str
    plant_height_cm: float
    leaf_greenness_index: float
    pest_severity: str = "Low"
    disease_severity: str = "Low"
    soil_moisture_pct: float = 65.0
    soil_ph: float = 6.2
    fertilizer_kg_acre: float = 120.0
    data_source: str = DATA_SOURCE_FARMER

    def __post_init__(self):
        if self.days_after_planting < 0:
            raise ValueError(
                f"Days after planting (DAP) must be non-negative, got {self.days_after_planting}."
            )


@dataclass
class WeatherObservation:
    """Environmental weather observation from API or station."""

    date: str  # YYYY-MM-DD
    weekly_rainfall_mm: float
    avg_temperature_c: float
    relative_humidity_pct: float
    solar_radiation_mj_m2_day: float
    data_source: str = DATA_SOURCE_API


@dataclass
class PredictionRecord:
    """Historical snapshot of a yield prediction."""

    farm_id: str
    prediction_timestamp: str
    days_after_planting: int
    growth_stage: str
    predicted_yield_t_ha: float
    model_version: str
    state_snapshot: Dict[str, Any]
    prediction_delta_t_ha: Optional[float] = None


class PredictionHistory:
    """Store and audit trail for progressive in-season predictions."""

    def __init__(self):
        self._history: List[PredictionRecord] = []

    def record_prediction(
        self,
        farm_id: str,
        days_after_planting: int,
        growth_stage: str,
        predicted_yield_t_ha: float,
        model_version: str,
        state_snapshot: Dict[str, Any],
        timestamp: Optional[str] = None,
    ) -> PredictionRecord:
        """Log a new prediction and automatically calculate delta relative to previous forecast."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Find latest previous prediction for this farm
        previous_preds = [p for p in self._history if p.farm_id == farm_id]
        delta = None
        if previous_preds:
            last_pred = previous_preds[-1].predicted_yield_t_ha
            delta = round(predicted_yield_t_ha - last_pred, 4)

        record = PredictionRecord(
            farm_id=farm_id,
            prediction_timestamp=timestamp,
            days_after_planting=days_after_planting,
            growth_stage=growth_stage,
            predicted_yield_t_ha=round(predicted_yield_t_ha, 4),
            model_version=model_version,
            state_snapshot=state_snapshot,
            prediction_delta_t_ha=delta,
        )
        self._history.append(record)
        return record

    def get_farm_history(self, farm_id: str) -> List[PredictionRecord]:
        """Retrieve complete chronological prediction history for a farm."""
        return [p for p in self._history if p.farm_id == farm_id]

    def delete_farm_history(self, farm_id: str) -> int:
        """Remove all prediction records for a specific farm ID from history.

        Parameters
        ----------
        farm_id : str
            Identifier of the farm whose prediction history is to be deleted.

        Returns
        -------
        int
            Number of prediction records removed.
        """
        initial_len = len(self._history)
        self._history = [p for p in self._history if p.farm_id != farm_id]
        return initial_len - len(self._history)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert history into a tabular pandas DataFrame."""
        if not self._history:
            return pd.DataFrame(columns=[
                "farm_id", "prediction_timestamp", "days_after_planting",
                "growth_stage", "predicted_yield_t_ha", "model_version", "prediction_delta_t_ha"
            ])
        rows = []
        for r in self._history:
            d = asdict(r)
            d.pop("state_snapshot", None)
            rows.append(d)
        return pd.DataFrame(rows)


class FarmStateManager:
    """Manages the evolving state of a farm and constructs feature matrices on demand."""

    def __init__(self, profile: FarmProfile):
        self.profile = profile
        self.crop_observations: List[CropObservation] = []
        self.irrigation_log = IrrigationEventLog()
        self.weather_observations: List[WeatherObservation] = []

    def add_crop_observation(self, observation: Union[CropObservation, Dict[str, Any]]):
        """Append a new crop measurement."""
        if isinstance(observation, dict):
            obs = CropObservation(**observation)
        elif isinstance(observation, CropObservation):
            obs = observation
        else:
            raise TypeError(f"Expected CropObservation or dict, got {type(observation)}")
        self.crop_observations.append(obs)
        self.crop_observations.sort(key=lambda x: str(x.date)[:10])

    def add_irrigation_event(self, event: Union[IrrigationEvent, Dict[str, Any]]):
        """Append an irrigation event."""
        self.irrigation_log.add_event(event)

    def add_weather_observation(self, observation: Union[WeatherObservation, Dict[str, Any]]):
        """Append weather observation."""
        if isinstance(observation, dict):
            obs = WeatherObservation(**observation)
        elif isinstance(observation, WeatherObservation):
            obs = observation
        else:
            raise TypeError(f"Expected WeatherObservation or dict, got {type(observation)}")
        self.weather_observations.append(obs)
        self.weather_observations.sort(key=lambda x: str(x.date)[:10])

    def reconstruct_current_state(
        self, as_of_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Reconstruct current farm feature state at `as_of_date` without future leakage.

        Parameters
        ----------
        as_of_date : str, optional
            Cutoff date (YYYY-MM-DD). If None, uses date of most recent observation.

        Returns
        -------
        Dict[str, Any]
            Complete feature dictionary ready for preprocessing and model inference.
        """
        if not self.crop_observations:
            raise ValueError(f"No crop observations recorded for Farm_ID '{self.profile.farm_id}'.")

        # Determine cutoff date
        if as_of_date is None:
            as_of_date = str(self.crop_observations[-1].date)[:10]

        cutoff_dt = datetime.strptime(str(as_of_date)[:10], "%Y-%m-%d")

        # 1. Filter observations up to cutoff (Zero Future Data Leakage Guard)
        valid_crops = [
            c for c in self.crop_observations
            if datetime.strptime(str(c.date)[:10], "%Y-%m-%d") <= cutoff_dt
        ]
        if not valid_crops:
            raise ValueError(
                f"No crop observations exist on or before cutoff date '{as_of_date}' "
                f"for Farm_ID '{self.profile.farm_id}'."
            )

        latest_crop = valid_crops[-1]

        # 2. Filter weather up to cutoff
        valid_weather = [
            w for w in self.weather_observations
            if datetime.strptime(str(w.date)[:10], "%Y-%m-%d") <= cutoff_dt
        ]
        if valid_weather:
            latest_weather = valid_weather[-1]
            weekly_rainfall = latest_weather.weekly_rainfall_mm
            avg_temp = latest_weather.avg_temperature_c
            rel_humidity = latest_weather.relative_humidity_pct
            solar_rad = latest_weather.solar_radiation_mj_m2_day
        else:
            # Fallback to standard district defaults if weather not logged
            weekly_rainfall = 40.0
            avg_temp = 26.0
            rel_humidity = 80.0
            solar_rad = 18.0

        # 3. Aggregate irrigation up to cutoff
        recent_runtime = self.irrigation_log.get_recent_runtime_hours(as_of_date, window_days=7)
        cum_runtime = self.irrigation_log.get_cumulative_runtime_hours(as_of_date)

        # Convert to mm if pump capacity is known, else use estimated proxy
        if self.profile.pump_capacity_lph is not None:
            irrigation_mm = self.irrigation_log.get_estimated_recent_depth_mm(
                as_of_date,
                land_size_acres=self.profile.land_size_acres,
                pump_capacity_lph=self.profile.pump_capacity_lph,
                window_days=7,
            ) or 0.0
            estimated_depth = irrigation_mm
        else:
            # When capacity is unknown, runtime hours are preserved directly as observable feature
            estimated_depth = 0.0
            irrigation_mm = round(recent_runtime * 5.0, 2)  # Representative ~5mm/hr fallback for model input

        # 4. Compute In-Season Engineered Features
        dap = max(1, latest_crop.days_after_planting)
        height = latest_crop.plant_height_cm
        greenness = latest_crop.leaf_greenness_index

        growth_ord = ORDINAL_MAPPINGS["Growth_Stage"].get(latest_crop.growth_stage, 2)
        pest_ord = ORDINAL_MAPPINGS["Pest_Severity"].get(latest_crop.pest_severity, 0)
        dis_ord = ORDINAL_MAPPINGS["Disease_Severity"].get(latest_crop.disease_severity, 0)

        total_water = weekly_rainfall + irrigation_mm
        height_per_dap = height / dap
        canopy_volume = height * greenness
        biotic_stress = pest_ord + dis_ord
        planting_month = int(str(self.profile.planting_date)[5:7]) if len(str(self.profile.planting_date)) >= 7 else 5

        # 5. Assemble Full Raw-Compatible State Dictionary
        state = {
            "Farm_ID": self.profile.farm_id,
            "District": self.profile.district,
            "Seed_Variety": self.profile.seed_variety,
            "Soil_Type": self.profile.soil_type,
            "Land_Size_Acres": self.profile.land_size_acres,
            "Planting_Date": self.profile.planting_date,
            "Days_After_Planting": dap,
            "Growth_Stage": latest_crop.growth_stage,
            "Growth_Stage_Ordinal": growth_ord,
            "Plant_Height_cm": height,
            "Leaf_Greenness_Index": greenness,
            "Pest_Severity": latest_crop.pest_severity,
            "Pest_Severity_Ordinal": pest_ord,
            "Disease_Severity": latest_crop.disease_severity,
            "Disease_Severity_Ordinal": dis_ord,
            "Soil_pH": latest_crop.soil_ph,
            "Soil_Moisture_pct": latest_crop.soil_moisture_pct,
            "Fertilizer_kg_acre": latest_crop.fertilizer_kg_acre,
            "Weekly_Rainfall_mm": weekly_rainfall,
            "Avg_Temperature_C": avg_temp,
            "Relative_Humidity_pct": rel_humidity,
            "Solar_Radiation_MJ_m2_day": solar_rad,
            "Irrigation_mm_week": irrigation_mm,
            # Dynamic Irrigation additions
            "Estimated_Irrigation_mm": estimated_depth,
            "Pump_Runtime_Hours": recent_runtime,
            "Cumulative_Irrigation_Hours": cum_runtime,
            # In-Season Engineered Features
            "Total_Water_Input_mm": total_water,
            "Plant_Height_per_DAP": round(height_per_dap, 4),
            "Canopy_Greenness_Volume": round(canopy_volume, 4),
            "Biotic_Stress_Index": biotic_stress,
            "Planting_Month": planting_month,
        }

        # One-Hot Dummies matching Phase 2 schema
        for d in DISTRICTS:
            state[f"District_{d}"] = 1 if self.profile.district == d else 0

        for v in SEED_VARIETIES:
            state[f"Seed_Variety_{v}"] = 1 if self.profile.seed_variety == v else 0

        for s in SOIL_TYPES:
            state[f"Soil_Type_{s}"] = 1 if self.profile.soil_type == s else 0

        return state
