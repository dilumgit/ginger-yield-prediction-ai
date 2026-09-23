"""Continuous Farmer Data Update Interface (Phase 4).

Provides a clean programmatic workflow for ingesting new farmer observations,
logging irrigation events, updating weather information, and triggering yield forecasts
without requiring the farmer to re-enter historical observations.
"""

from typing import Any, Dict, Optional, Tuple, Union

from src.data.farm_state import (
    CropObservation,
    FarmStateManager,
    PredictionRecord,
    WeatherObservation,
    calculate_dap,
    suggest_growth_stage,
)
from src.data.irrigation import IrrigationEvent
from src.data.weather_api import AbstractWeatherProvider, MockWeatherProvider
from src.dynamic.dynamic_predictor import DynamicYieldPredictor


def update_farm_state(
    farm_manager: FarmStateManager,
    dynamic_predictor: DynamicYieldPredictor,
    observation_date: str,
    days_after_planting: Optional[int] = None,
    growth_stage: Optional[str] = None,
    plant_height_cm: Optional[float] = None,
    leaf_greenness_index: Optional[float] = None,
    pest_severity: Optional[str] = None,
    disease_severity: Optional[str] = None,
    soil_moisture_pct: Optional[float] = None,
    soil_ph: Optional[float] = None,
    fertilizer_kg_acre: Optional[float] = None,
    irrigation_event: Optional[Union[Dict[str, Any], IrrigationEvent]] = None,
    weather_provider: Optional[AbstractWeatherProvider] = None,
    weather_observation: Optional[Union[Dict[str, Any], WeatherObservation]] = None,
) -> Tuple[float, PredictionRecord, Optional[Dict[str, Any]]]:
    """Incorporate newly observed farm information and immediately generate updated prediction.

    Parameters
    ----------
    farm_manager : FarmStateManager
        The state manager tracking the farm's lifecycle.
    dynamic_predictor : DynamicYieldPredictor
        The dynamic prediction engine.
    observation_date : str
        Date of current observation (YYYY-MM-DD).
    days_after_planting : int, optional
        Current crop age in days elapsed since planting. If None, calculated automatically
        from (observation_date - farm_manager.profile.planting_date).
    growth_stage : str, optional
        Current biological growth stage. If None, suggested automatically from calculated DAP.
    plant_height_cm : float, optional
        Newly measured plant shoot height (cm). If None, carries forward latest recorded height.
    leaf_greenness_index : float, optional
        Newly measured SPAD greenness index. If None, carries forward latest recorded value.
    pest_severity : str, optional
        Scouted pest severity ('Low', 'Medium', 'High'). Defaults to latest or 'Low'.
    disease_severity : str, optional
        Scouted disease severity ('Low', 'Medium', 'High'). Defaults to latest or 'Low'.
    soil_moisture_pct : float, optional
        Soil moisture percentage. Defaults to latest or 65.0%.
    soil_ph : float, optional
        Soil pH. Defaults to latest or 6.2.
    fertilizer_kg_acre : float, optional
        Fertilizer applied (kg/acre). Defaults to latest or 120.0.
    irrigation_event : Dict or IrrigationEvent, optional
        Farmer-reported irrigation event (runtime_hours, water_source, method).
    weather_provider : AbstractWeatherProvider, optional
        Weather provider interface to automatically fetch weather for current date and district.
    weather_observation : Dict or WeatherObservation, optional
        Explicitly provided weather parameters for current date.

    Returns
    -------
    Tuple[float, PredictionRecord, Optional[Dict[str, Any]]]
        (predicted_yield_t_ha, prediction_record, factor_change_analysis)
    """
    # 1. Append new irrigation event if farmer reported one
    if irrigation_event is not None:
        if isinstance(irrigation_event, dict):
            # Ensure date matches observation date if not specified
            if "date" not in irrigation_event:
                irrigation_event["date"] = observation_date
            farm_manager.add_irrigation_event(irrigation_event)
        else:
            farm_manager.add_irrigation_event(irrigation_event)

    # 2. Ingest weather data for current date
    if weather_observation is not None:
        farm_manager.add_weather_observation(weather_observation)
    elif weather_provider is not None:
        wx_dict = weather_provider.get_weather_for_date(
            district=farm_manager.profile.district,
            date=observation_date,
        )
        farm_manager.add_weather_observation(WeatherObservation(
            date=observation_date,
            weekly_rainfall_mm=wx_dict.get("Weekly_Rainfall_mm", 40.0),
            avg_temperature_c=wx_dict.get("Avg_Temperature_C", 26.0),
            relative_humidity_pct=wx_dict.get("Relative_Humidity_pct", 80.0),
            solar_radiation_mj_m2_day=wx_dict.get("Solar_Radiation_MJ_m2_day", 18.5),
            data_source=wx_dict.get("Data_Source", "API_DERIVED"),
        ))
    else:
        # Fallback to Mock provider if neither provided
        default_mock = MockWeatherProvider()
        wx_dict = default_mock.get_weather_for_date(farm_manager.profile.district, observation_date)
        farm_manager.add_weather_observation(WeatherObservation(
            date=observation_date,
            weekly_rainfall_mm=wx_dict["Weekly_Rainfall_mm"],
            avg_temperature_c=wx_dict["Avg_Temperature_C"],
            relative_humidity_pct=wx_dict["Relative_Humidity_pct"],
            solar_radiation_mj_m2_day=wx_dict["Solar_Radiation_MJ_m2_day"],
            data_source="SYNTHETIC_SIMULATED",
        ))

    # 3. Resolve and validate DAP and Growth Stage
    if days_after_planting is None:
        eff_dap = calculate_dap(
            planting_date=farm_manager.profile.planting_date,
            observation_date=observation_date,
        )
    else:
        if farm_manager.profile.planting_date:
            # Enforce temporal validity (cannot be earlier than planting date)
            calculate_dap(farm_manager.profile.planting_date, observation_date)
        if days_after_planting < 0:
            raise ValueError(f"Days after planting cannot be negative, got {days_after_planting}")
        eff_dap = int(days_after_planting)

    if growth_stage is None:
        eff_stage = suggest_growth_stage(eff_dap)
    else:
        eff_stage = growth_stage

    # 4. Carry forward unsupplied parameters from latest previous observation
    latest_prev = farm_manager.crop_observations[-1] if farm_manager.crop_observations else None

    eff_height = plant_height_cm if plant_height_cm is not None else (latest_prev.plant_height_cm if latest_prev else 20.0)
    eff_green = leaf_greenness_index if leaf_greenness_index is not None else (latest_prev.leaf_greenness_index if latest_prev else 50.0)
    eff_pest = pest_severity if pest_severity is not None else (latest_prev.pest_severity if latest_prev else "Low")
    eff_dis = disease_severity if disease_severity is not None else (latest_prev.disease_severity if latest_prev else "Low")
    eff_sm = soil_moisture_pct if soil_moisture_pct is not None else (latest_prev.soil_moisture_pct if latest_prev else 65.0)
    eff_ph = soil_ph if soil_ph is not None else (latest_prev.soil_ph if latest_prev else 6.2)
    eff_fert = fertilizer_kg_acre if fertilizer_kg_acre is not None else (latest_prev.fertilizer_kg_acre if latest_prev else 120.0)

    # 5. Append new crop observation
    new_obs = CropObservation(
        date=observation_date,
        days_after_planting=eff_dap,
        growth_stage=eff_stage,
        plant_height_cm=eff_height,
        leaf_greenness_index=eff_green,
        pest_severity=eff_pest,
        disease_severity=eff_dis,
        soil_moisture_pct=eff_sm,
        soil_ph=eff_ph,
        fertilizer_kg_acre=eff_fert,
        data_source="FARMER_REPORTED",
    )
    farm_manager.add_crop_observation(new_obs)

    # 6. Trigger updated prediction
    pred_val, record, analysis = dynamic_predictor.predict_current_state(
        farm_manager=farm_manager,
        as_of_date=observation_date,
    )

    return pred_val, record, analysis
