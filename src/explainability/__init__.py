"""Explainable AI (XAI) package using TreeSHAP, farmer translation, and recommendation engines."""

from src.explainability.explanation_model import (
    FeatureContribution,
    PredictionExplanation,
)
from src.explainability.farmer_translator import FarmerExplanationTranslator
from src.explainability.recommendations import FarmerRecommendationEngine
from src.explainability.shap_explainer import GingerShapExplainer

__all__ = [
    "FeatureContribution",
    "PredictionExplanation",
    "FarmerExplanationTranslator",
    "FarmerRecommendationEngine",
    "GingerShapExplainer",
]
