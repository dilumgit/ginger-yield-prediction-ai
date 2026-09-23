"""Farmer-Friendly Explanation Translation Layer (Phase 5).

Translates technical game-theoretic SHAP attributions into clear, intuitive,
and agronomically sound plain-language explanations for Sri Lankan ginger farmers
without presenting raw math or making unsupported causal claims.
"""

from typing import Any, Dict, List, Optional, Tuple

from src.explainability.explanation_model import FeatureContribution


class FarmerExplanationTranslator:
    """Translates technical SHAP explanations into farmer-accessible narratives."""

    # Agronomic narrative templates for individual features
    NARRATIVE_TEMPLATES = {
        "Plant_Height_cm": {
            "pos": "Current plant height ({val} cm) indicates healthy vegetative shoot development, supporting the yield prediction.",
            "neg": "Measured plant height ({val} cm) is somewhat lower than average for this growth stage, pulling the yield forecast lower.",
        },
        "Plant_Height_per_DAP": {
            "pos": "Daily growth rate ({val:.2f} cm/day) is vigorous, indicating strong early vegetative establishment.",
            "neg": "Daily vertical growth velocity ({val:.2f} cm/day) is slower than optimal, which the model associates with lower final yield.",
        },
        "Leaf_Greenness_Index": {
            "pos": "Foliage greenness ({val:.1f} SPAD) reflects healthy chlorophyll and active nitrogen assimilation.",
            "neg": "Foliage greenness ({val:.1f} SPAD) is paler than optimal, reflecting reduced photosynthetic potential.",
        },
        "Canopy_Greenness_Volume": {
            "pos": "Combined shoot height and green canopy volume are strong, supporting active carbohydrate synthesis for rhizome growth.",
            "neg": "Overall active canopy volume is below potential, reducing the predicted yield.",
        },
        "Biotic_Stress_Index": {
            "pos": "Low incidence of pests and diseases is protecting developing rhizomes from yield loss.",
            "neg": "Current pest and disease pressure (Score: {val}/4) is exerting stress on the crop, lowering the forecast.",
        },
        "Pest_Severity_Ordinal": {
            "pos": "Minimal pest damage is allowing healthy foliage and shoot growth.",
            "neg": "Elevated pest presence is associated with a downward adjustment in the yield forecast.",
        },
        "Disease_Severity_Ordinal": {
            "pos": "Absence of active fungal rhizome rot is favorable for high harvest yields.",
            "neg": "Signs of disease (rhizome rot/wilt) are identified by the model as a major limiting factor.",
        },
        "Total_Water_Input_mm": {
            "pos": "Combined rainfall and supplemental irrigation ({val:.1f} mm) provide adequate root-zone moisture.",
            "neg": "Total water input ({val:.1f} mm) is below optimal requirements, contributing negatively to the forecast.",
        },
        "Soil_Moisture_pct": {
            "pos": "Soil moisture level ({val:.1f}%) is within the favorable range for root development.",
            "neg": "Current soil moisture ({val:.1f}%) is outside optimal range, which the model associates with reduced yield.",
        },
        "Soil_pH": {
            "pos": "Soil pH ({val:.2f}) is well-suited for ginger nutrient uptake (optimal: 5.5-6.5).",
            "neg": "Soil pH ({val:.2f}) is suboptimal, potentially impeding nutrient absorption.",
        },
        "Fertilizer_kg_acre": {
            "pos": "Nutrient application ({val} kg/acre) is adequately supporting crop growth requirements.",
            "neg": "Fertilizer dosage ({val} kg/acre) differs from optimal nutrient demand.",
        },
        "Weekly_Rainfall_mm": {
            "pos": "Weekly rainfall ({val:.1f} mm) has provided beneficial moisture.",
            "neg": "Recent rainfall ({val:.1f} mm) was limited or excessively wet.",
        },
        "Irrigation_mm_week": {
            "pos": "Supplemental irrigation applied ({val:.1f} mm) is effectively supporting crop water needs.",
            "neg": "Lack of supplemental irrigation is identified as a constraint.",
        },
        "Avg_Temperature_C": {
            "pos": "Ambient temperature ({val:.1f}°C) is favorable for ginger vegetative growth.",
            "neg": "Ambient temperature ({val:.1f}°C) is cooler/warmer than optimal.",
        },
        "Solar_Radiation_MJ_m2_day": {
            "pos": "Adequate sunlight ({val:.1f} MJ/m²/day) is driving robust photosynthesis.",
            "neg": "Lower solar radiation ({val:.1f} MJ/m²/day) is limiting photosynthetic accumulation.",
        },
    }

    def translate_contribution(self, contrib: FeatureContribution) -> str:
        """Translate a single feature contribution into a farmer-friendly sentence."""
        templates = self.NARRATIVE_TEMPLATES.get(contrib.feature_name)
        if templates:
            tmpl = templates["pos"] if contrib.direction == "positive" else templates["neg"]
            try:
                return tmpl.format(val=contrib.value)
            except Exception:
                pass

        # Fallback template
        sign = "+" if contrib.direction == "positive" else "-"
        action = "supporting" if contrib.direction == "positive" else "lowering"
        return f"{contrib.label} ({contrib.value} {contrib.unit}) is {action} the prediction ({sign}{contrib.abs_shap_value:.2f} t/ha)."

    def generate_farmer_explanation(
        self,
        predicted_yield_t_ha: float,
        growth_stage: str,
        days_after_planting: int,
        top_positive: List[FeatureContribution],
        top_negative: List[FeatureContribution],
        previous_prediction_t_ha: Optional[float] = None,
        prediction_delta_t_ha: Optional[float] = None,
    ) -> Tuple[str, List[str], List[str]]:
        """Generate comprehensive farmer-facing summary, supporting points, and limiting points.

        Returns
        -------
        Tuple[str, List[str], List[str]]
            (farmer_summary, supporting_points, limiting_points)
        """
        # 1. Headline summary
        if previous_prediction_t_ha is not None and prediction_delta_t_ha is not None:
            if prediction_delta_t_ha > 0:
                trend = f"increased by {abs(prediction_delta_t_ha):.2f} t/ha (from {previous_prediction_t_ha:.2f} to {predicted_yield_t_ha:.2f} t/ha)"
            elif prediction_delta_t_ha < 0:
                trend = f"decreased by {abs(prediction_delta_t_ha):.2f} t/ha (from {previous_prediction_t_ha:.2f} to {predicted_yield_t_ha:.2f} t/ha)"
            else:
                trend = f"remained steady at {predicted_yield_t_ha:.2f} t/ha"
            summary = (
                f"At Day {days_after_planting} ({growth_stage}), your predicted final harvest yield is "
                f"{predicted_yield_t_ha:.2f} t/ha. Based on your latest farm update, the forecast has {trend}."
            )
        else:
            summary = (
                f"At Day {days_after_planting} ({growth_stage}), your current predicted final harvest yield is "
                f"{predicted_yield_t_ha:.2f} metric tons per hectare (t/ha)."
            )

        # 2. Supporting points (Top positive contributors)
        supporting = [self.translate_contribution(c) for c in top_positive[:3]]
        if not supporting:
            supporting = ["Current baseline growing conditions are supporting the harvest forecast."]

        # 3. Limiting points (Top negative contributors)
        limiting = [self.translate_contribution(c) for c in top_negative[:3]]
        if not limiting:
            limiting = ["No major crop stress factors are currently pulling the forecast down."]

        return summary, supporting, limiting
