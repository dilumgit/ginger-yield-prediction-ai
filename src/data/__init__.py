"""Data loading, inspection, preprocessing, feature engineering, irrigation, and dynamic farm state management."""

from src.data.data_loader import load_raw_data
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
from src.data.feature_engineering import (
    create_growth_features,
    create_health_features,
    create_temporal_features,
    create_water_features,
    engineer_all_features,
    get_feature_metadata_records,
)
from src.data.irrigation import (
    IrrigationEvent,
    IrrigationEventLog,
    runtime_to_irrigation_depth_mm,
)
from src.data.preprocessing import (
    encode_categorical_features,
    get_feature_matrix_and_target,
    get_model_feature_columns,
    preprocess_dataset,
    validate_raw_schema,
)
from src.data.weather_api import (
    AbstractWeatherProvider,
    MockWeatherProvider,
    OpenMeteoWeatherProvider,
    WeatherAPIError,
)

__all__ = [
    "load_raw_data",
    "create_water_features",
    "create_growth_features",
    "create_health_features",
    "create_temporal_features",
    "engineer_all_features",
    "get_feature_metadata_records",
    "validate_raw_schema",
    "encode_categorical_features",
    "preprocess_dataset",
    "get_model_feature_columns",
    "get_feature_matrix_and_target",
    "IrrigationEvent",
    "IrrigationEventLog",
    "runtime_to_irrigation_depth_mm",
    "FarmProfile",
    "CropObservation",
    "WeatherObservation",
    "FarmStateManager",
    "PredictionRecord",
    "PredictionHistory",
    "calculate_dap",
    "suggest_growth_stage",
    "AbstractWeatherProvider",
    "MockWeatherProvider",
    "OpenMeteoWeatherProvider",
    "WeatherAPIError",
]
