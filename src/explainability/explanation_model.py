"""Explanation Data Models for Explainable AI & Farmer Decision Support.

Defines structured containers for feature-level SHAP attributions,
farmer-friendly translations, and actionable agricultural recommendations.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class FeatureContribution:
    """Individual feature contribution derived from SHAP attribution."""

    feature_name: str
    label: str
    value: Any
    unit: str
    shap_value: float
    abs_shap_value: float
    direction: str  # "positive" or "negative"
    rank: int
    farmer_interpretation: str


@dataclass
class PredictionExplanation:
    """Comprehensive explanation object integrating technical and farmer-facing insights."""

    farm_id: str
    prediction_timestamp: str
    days_after_planting: int
    growth_stage: str
    predicted_yield_t_ha: float
    base_value_t_ha: float
    previous_prediction_t_ha: Optional[float] = None
    prediction_delta_t_ha: Optional[float] = None
    prediction_delta_pct: Optional[float] = None
    top_positive_factors: List[FeatureContribution] = field(default_factory=list)
    top_negative_factors: List[FeatureContribution] = field(default_factory=list)
    changed_factors: List[Dict[str, Any]] = field(default_factory=list)
    farmer_summary: str = ""
    farmer_supporting_points: List[str] = field(default_factory=list)
    farmer_limiting_points: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    factor_priorities: Dict[str, List[str]] = field(default_factory=dict)
    model_version: str = "CatBoost_Phase3_Baseline"
    explanation_method: str = "TreeSHAP"
    data_sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert full explanation object to dictionary."""
        return asdict(self)

    def to_farmer_summary_dict(self) -> Dict[str, Any]:
        """Convert to simplified farmer-facing summary dictionary."""
        return {
            "Farm_ID": self.farm_id,
            "DAP": self.days_after_planting,
            "Growth_Stage": self.growth_stage,
            "Predicted_Yield_t_ha": self.predicted_yield_t_ha,
            "Prediction_Change_t_ha": self.prediction_delta_t_ha,
            "Farmer_Summary": self.farmer_summary,
            "Supporting_Factors": self.farmer_supporting_points,
            "Limiting_Factors": self.farmer_limiting_points,
            "Recommendations": self.recommendations,
        }
