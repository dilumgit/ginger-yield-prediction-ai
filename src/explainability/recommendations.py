"""Rule-Based Farmer Advisory & Recommendation Engine (Phase 5).

Generates transparent, scientifically sound, and actionable agricultural guidance
tailored to the farmer's current crop state, pest/disease scouting, and recent
rainfall/irrigation history without fabricating unavailable hardware sensor measurements.
"""

from typing import Any, Dict, List, Optional, Tuple

from src.explainability.explanation_model import FeatureContribution


class FarmerRecommendationEngine:
    """Rule-based decision support system generating transparent farmer recommendations."""

    def prioritize_factors(
        self,
        top_positive: List[FeatureContribution],
        top_negative: List[FeatureContribution],
        changed_factors: List[Dict[str, Any]],
        state_snapshot: Dict[str, Any],
    ) -> Dict[str, List[str]]:
        """Categorize factors into Supporting, Attention, Changed, and Unchanged groups."""
        supporting = [c.label for c in top_positive[:4]]
        attention = [c.label for c in top_negative[:4]]
        changed = [cf["label"] for cf in changed_factors]

        # Identify key unchanged baseline factors
        all_important = supporting + attention
        unchanged = [f for f in all_important if f not in changed]

        return {
            "Supporting_Factors": supporting,
            "Attention_Factors": attention,
            "Recent_Changes": changed,
            "Unchanged_Baselines": unchanged,
        }

    def generate_recommendations(
        self,
        state_snapshot: Dict[str, Any],
        top_negative: List[FeatureContribution],
    ) -> List[str]:
        """Generate targeted agronomic recommendations based on verified current state.

        Parameters
        ----------
        state_snapshot : Dict[str, Any]
            The current reconstructed state dictionary.
        top_negative : List[FeatureContribution]
            The top negative SHAP contributors.

        Returns
        -------
        List[str]
            List of prioritized, practical recommendation strings.
        """
        recommendations = []

        # 1. Biotic Stress: Pest Management
        pest_sev = str(state_snapshot.get("Pest_Severity", "Low")).capitalize()
        if pest_sev in ["Medium", "High"]:
            recommendations.append(
                f"[Pest Alert]: Pest pressure is currently {pest_sev}. Inspect shoot collars for shoot borer "
                f"damage and foliage for caterpillars; apply approved Integrated Pest Management (IPM) practices."
            )

        # 2. Biotic Stress: Disease Management (Rhizome Rot / Soft Rot)
        dis_sev = str(state_snapshot.get("Disease_Severity", "Low")).capitalize()
        if dis_sev in ["Medium", "High"]:
            recommendations.append(
                f"[Disease Alert]: Disease symptoms (rhizome rot/wilt) are rated as {dis_sev}. Immediately clear "
                f"drainage ditches to prevent stagnant water around raised beds and consult local agricultural guidance."
            )

        # 3. Water Management (Rainfall-Aware Irrigation Advice)
        rainfall_mm = float(state_snapshot.get("Weekly_Rainfall_mm", 40.0))
        soil_moisture = float(state_snapshot.get("Soil_Moisture_pct", 65.0))
        recent_runtime = float(state_snapshot.get("Pump_Runtime_Hours", 0.0))

        if rainfall_mm >= 50.0:
            recommendations.append(
                f"[Water Management]: Recent rainfall was substantial ({rainfall_mm:.1f} mm). Withhold supplemental "
                f"irrigation to prevent bed saturation and rhizome fungal infections."
            )
        elif soil_moisture < 50.0 and recent_runtime == 0.0:
            recommendations.append(
                f"[Irrigation Advice]: Soil moisture is relatively low ({soil_moisture:.1f}%) and no recent irrigation "
                f"was recorded. Consider applying supplemental irrigation (e.g. 1.5-2.0 hours pump runtime) according to normal practice."
            )
        elif soil_moisture > 85.0 and rainfall_mm < 30.0:
            recommendations.append(
                f"[Moisture Balance]: Soil moisture is very high ({soil_moisture:.1f}%). Ensure drainage furrows allow free runoff."
            )

        # 4. Soil Chemistry: pH Balance
        soil_ph = float(state_snapshot.get("Soil_pH", 6.2))
        if soil_ph < 5.2:
            recommendations.append(
                f"[Soil Management]: Soil pH ({soil_ph:.2f}) is moderately acidic. Consider incorporating agricultural "
                f"dolomite or lime during bed preparation for subsequent crop cycles."
            )
        elif soil_ph > 6.8:
            recommendations.append(
                f"[Soil Management]: Soil pH ({soil_ph:.2f}) is higher than optimal for ginger (5.5-6.5). Incorporating "
                f"organic compost will help buffer soil reaction."
            )

        # 5. Crop Vigor & Leaf Greenness
        greenness = float(state_snapshot.get("Leaf_Greenness_Index", 55.0))
        growth_stage = str(state_snapshot.get("Growth_Stage", "Vegetative"))
        if greenness < 42.0 and growth_stage in ["Early Vegetative", "Vegetative"]:
            recommendations.append(
                f"[Crop Nutrition]: Foliage greenness ({greenness:.1f} SPAD) is paler than typical for {growth_stage}. "
                f"Check for signs of nitrogen deficiency or root-zone water stress."
            )

        # Default recommendation if all parameters are in optimal range
        if not recommendations:
            recommendations.append(
                "[Good Practice]: Crop parameters and growing conditions are currently in a favorable range. "
                "Continue routine field scouting and maintain standard crop management."
            )

        return recommendations
