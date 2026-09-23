"""Comprehensive Unit Tests for Phase 5 Explainable AI (XAI) & Farmer Explanations.

Verifies:
1. GingerShapExplainer loads model and extracts expected base value.
2. SHAP values match exact 36-feature model schema in identical order.
3. Positive and negative contributions are separated and sorted accurately.
4. Instance-level explanation generates valid FeatureContribution objects.
5. Global explanation computes mean |SHAP| and ranks across test dataset.
6. FarmerExplanationTranslator generates plain-language, non-causal narratives.
7. FarmerRecommendationEngine produces targeted alerts (pest, disease, waterlogging).
8. Rainfall-aware irrigation rule withholds irrigation when rainfall >= 50 mm.
9. Factor prioritization correctly groups Supporting, Attention, Changed, Unchanged.
10. PredictionExplanation data model converts cleanly to dict and farmer summary.
11. DynamicYieldPredictor.explain_prediction integrates prediction, delta, and SHAP.
12. Temporal boundary enforcement: Future observations strictly excluded from explanations.
13. Target and Leakage columns (Actual_Final_Yield, Predicted_Final_Yield, Farm_ID) excluded.
14. Unknown pump capacity fallback preserved in state and explanations.
15. Weather provenance tags (API_DERIVED, SYNTHETIC_SIMULATED) are preserved in explanation metadata.
16. scripts/explain_predictions.py executes end-to-end and creates all artifacts.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from src.data.farm_state import (
    CropObservation,
    FarmProfile,
    FarmStateManager,
    PredictionRecord,
    WeatherObservation,
)
from src.data.irrigation import IrrigationEvent
from src.data.weather_api import MockWeatherProvider
from src.dynamic.dynamic_predictor import DynamicYieldPredictor
from src.dynamic.state_updater import update_farm_state
from src.explainability.explanation_model import FeatureContribution, PredictionExplanation
from src.explainability.farmer_translator import FarmerExplanationTranslator
from src.explainability.recommendations import FarmerRecommendationEngine
from src.explainability.shap_explainer import GingerShapExplainer
from src.utils.config import BEST_MODEL_PATH, ID_COLUMNS, LEAKAGE_COLUMNS, TARGET_COLUMN


@pytest.fixture
def initialized_farm_manager():
    """Fixture providing an initialized FarmStateManager."""
    profile = FarmProfile(
        farm_id="F0099",
        district="Kandy",
        seed_variety="Local",
        soil_type="Loam",
        land_size_acres=1.2,
        planting_date="2024-03-01",
        pump_capacity_lph=8000.0,
    )
    mgr = FarmStateManager(profile)
    mgr.add_crop_observation(CropObservation(
        date="2024-04-15",
        days_after_planting=45,
        growth_stage="Sprouting",
        plant_height_cm=18.0,
        leaf_greenness_index=45.0,
        pest_severity="Low",
        disease_severity="Low",
        soil_moisture_pct=65.0,
        soil_ph=6.2,
        fertilizer_kg_acre=120.0,
    ))
    return mgr


def test_1_shap_explainer_initialization_and_base_value():
    """Verify GingerShapExplainer loads correctly and extracts expected base value."""
    explainer = GingerShapExplainer()
    assert explainer.model is not None
    assert len(explainer.feature_names) == 36
    assert isinstance(explainer.base_value, float)
    assert 12.0 < explainer.base_value < 15.0  # Training target mean is ~13.65 t/ha


def test_2_shap_feature_alignment_and_dimension(initialized_farm_manager):
    """Verify SHAP values match exact 36 model features in precise order."""
    explainer = GingerShapExplainer()
    state_dict = initialized_farm_manager.reconstruct_current_state("2024-04-15")
    res = explainer.explain_instance(state_dict)

    assert "shap_array" in res
    assert len(res["shap_array"]) == 36
    assert len(res["all_contributions"]) == 36
    assert res["predicted_yield_t_ha"] > 0.0

    # Additive Property: Base_Value + sum(SHAP) approx Predicted_Yield
    calculated_pred = res["base_value_t_ha"] + res["sum_shap_values"]
    assert calculated_pred == pytest.approx(res["predicted_yield_t_ha"], abs=0.01)


def test_3_positive_and_negative_contribution_separation(initialized_farm_manager):
    """Verify positive and negative contributors are segregated and sorted correctly."""
    explainer = GingerShapExplainer()
    state_dict = initialized_farm_manager.reconstruct_current_state("2024-04-15")
    res = explainer.explain_instance(state_dict, top_k=5)

    assert len(res["top_positive"]) <= 5
    assert len(res["top_negative"]) <= 5

    for c in res["top_positive"]:
        assert c.shap_value >= 0
        assert c.direction == "positive"

    for c in res["top_negative"]:
        assert c.shap_value <= 0
        assert c.direction == "negative"


def test_4_global_dataset_explanation():
    """Verify explain_dataset produces ranked feature importance table."""
    explainer = GingerShapExplainer()
    sample_df = pd.DataFrame([{
        feat: 1.0 if "District" in feat or "Soil_Type" in feat or "Seed" in feat else 50.0
        for feat in explainer.feature_names
    }])
    df_imp, shap_mat = explainer.explain_dataset(sample_df)

    assert len(df_imp) == 36
    assert "Mean_Abs_SHAP_t_ha" in df_imp.columns
    assert "Rank" in df_imp.columns
    assert df_imp["Rank"].iloc[0] == 1
    assert df_imp["Mean_Abs_SHAP_t_ha"].is_monotonic_decreasing


def test_5_farmer_translation_layer():
    """Verify FarmerExplanationTranslator generates clear plain-language descriptions."""
    translator = FarmerExplanationTranslator()
    contrib_pos = FeatureContribution(
        feature_name="Plant_Height_cm",
        label="Plant Height",
        value=75.0,
        unit="cm",
        shap_value=0.45,
        abs_shap_value=0.45,
        direction="positive",
        rank=1,
        farmer_interpretation="",
    )
    contrib_neg = FeatureContribution(
        feature_name="Biotic_Stress_Index",
        label="Biotic Stress Index",
        value=2.0,
        unit="score (0-4)",
        shap_value=-0.65,
        abs_shap_value=0.65,
        direction="negative",
        rank=2,
        farmer_interpretation="",
    )

    text_pos = translator.translate_contribution(contrib_pos)
    text_neg = translator.translate_contribution(contrib_neg)

    assert "supporting" in text_pos or "healthy" in text_pos
    assert "stress" in text_neg or "lowering" in text_neg

    summary, supp, lim = translator.generate_farmer_explanation(
        predicted_yield_t_ha=14.5,
        growth_stage="Vegetative",
        days_after_planting=90,
        top_positive=[contrib_pos],
        top_negative=[contrib_neg],
        previous_prediction_t_ha=13.5,
        prediction_delta_t_ha=1.0,
    )
    assert "increased by 1.00 t/ha" in summary
    assert len(supp) >= 1
    assert len(lim) >= 1


def test_6_recommendation_engine_biotic_alerts():
    """Verify recommendation engine produces pest/disease alerts on elevated severity."""
    engine = FarmerRecommendationEngine()
    state_pest = {"Pest_Severity": "High", "Disease_Severity": "Low", "Weekly_Rainfall_mm": 35.0, "Soil_Moisture_pct": 65.0, "Soil_pH": 6.2}
    recs = engine.generate_recommendations(state_pest, [])
    assert any("[Pest Alert]" in r for r in recs)

    state_disease = {"Pest_Severity": "Low", "Disease_Severity": "Medium", "Weekly_Rainfall_mm": 35.0, "Soil_Moisture_pct": 65.0, "Soil_pH": 6.2}
    recs = engine.generate_recommendations(state_disease, [])
    assert any("[Disease Alert]" in r for r in recs)


def test_7_recommendation_engine_rainfall_aware_irrigation():
    """Verify recommendation engine withholds irrigation when rainfall is high (>= 50 mm)."""
    engine = FarmerRecommendationEngine()
    state_rain = {"Pest_Severity": "Low", "Disease_Severity": "Low", "Weekly_Rainfall_mm": 65.0, "Soil_Moisture_pct": 75.0, "Soil_pH": 6.2}
    recs = engine.generate_recommendations(state_rain, [])
    assert any("Withhold supplemental irrigation" in r for r in recs)


def test_8_recommendation_engine_low_moisture_irrigation_advice():
    """Verify recommendation engine advises irrigation when soil moisture is dry and runtime is 0."""
    engine = FarmerRecommendationEngine()
    state_dry = {"Pest_Severity": "Low", "Disease_Severity": "Low", "Weekly_Rainfall_mm": 10.0, "Soil_Moisture_pct": 42.0, "Pump_Runtime_Hours": 0.0, "Soil_pH": 6.2}
    recs = engine.generate_recommendations(state_dry, [])
    assert any("[Irrigation Advice]" in r for r in recs)


def test_9_factor_prioritization(initialized_farm_manager):
    """Verify factor prioritization groups factors into Supporting, Attention, Changed, Unchanged."""
    engine = FarmerRecommendationEngine()
    c_pos = FeatureContribution("Plant_Height_cm", "Plant Height", 70.0, "cm", 0.3, 0.3, "positive", 1, "")
    c_neg = FeatureContribution("Biotic_Stress_Index", "Biotic Stress", 2.0, "", -0.4, 0.4, "negative", 2, "")
    changed = [{"variable": "Plant_Height_cm", "label": "Plant Height"}]

    priorities = engine.prioritize_factors([c_pos], [c_neg], changed, {})
    assert "Plant Height" in priorities["Supporting_Factors"]
    assert "Biotic Stress" in priorities["Attention_Factors"]
    assert "Plant Height" in priorities["Recent_Changes"]
    assert "Biotic Stress" in priorities["Unchanged_Baselines"]


def test_10_dynamic_predictor_explain_prediction(initialized_farm_manager):
    """Verify DynamicYieldPredictor.explain_prediction returns complete PredictionExplanation."""
    predictor = DynamicYieldPredictor()
    exp = predictor.explain_prediction(initialized_farm_manager, as_of_date="2024-04-15")

    assert isinstance(exp, PredictionExplanation)
    assert exp.farm_id == "F0099"
    assert exp.days_after_planting == 45
    assert len(exp.top_positive_factors) > 0
    assert len(exp.top_negative_factors) > 0
    assert len(exp.farmer_summary) > 0
    assert len(exp.recommendations) > 0
    assert exp.explanation_method == "TreeSHAP"

    summary_dict = exp.to_farmer_summary_dict()
    assert summary_dict["Farm_ID"] == "F0099"
    assert "Farmer_Summary" in summary_dict


def test_11_temporal_leakage_exclusion_in_explanations(initialized_farm_manager):
    """Verify future observations cannot enter the SHAP explanation."""
    # Future observation at Day 120
    initialized_farm_manager.add_crop_observation(CropObservation(
        date="2024-07-01",
        days_after_planting=120,
        growth_stage="Rhizome Initiation",
        plant_height_cm=80.0,
        leaf_greenness_index=70.0,
    ))

    # Explain at Day 45
    predictor = DynamicYieldPredictor()
    exp = predictor.explain_prediction(initialized_farm_manager, as_of_date="2024-04-15")

    assert exp.days_after_planting == 45
    assert exp.growth_stage == "Sprouting"
    for c in exp.top_positive_factors + exp.top_negative_factors:
        if c.feature_name == "Plant_Height_cm":
            assert c.value == 18.0  # Not the future 80.0 cm


def test_12_leakage_and_target_columns_prohibited():
    """Verify Target and Leakage columns are never used as model features."""
    explainer = GingerShapExplainer()
    assert TARGET_COLUMN not in explainer.feature_names
    for leak in LEAKAGE_COLUMNS:
        assert leak not in explainer.feature_names
    assert "Farm_ID" not in explainer.feature_names


def test_13_unknown_pump_capacity_fallback_in_state():
    """Verify unknown pump capacity fallback is supported without error in explanations."""
    profile_no_cap = FarmProfile(
        farm_id="F0100",
        district="Matale",
        seed_variety="Nadun",
        soil_type="Clay Loam",
        land_size_acres=0.8,
        planting_date="2024-03-01",
        pump_capacity_lph=None,  # Unknown capacity
    )
    mgr = FarmStateManager(profile_no_cap)
    mgr.add_crop_observation(CropObservation(
        date="2024-04-01",
        days_after_planting=30,
        growth_stage="Sprouting",
        plant_height_cm=14.0,
        leaf_greenness_index=40.0,
    ))
    mgr.add_irrigation_event(IrrigationEvent(date="2024-04-01", runtime_hours=2.0))

    predictor = DynamicYieldPredictor()
    exp = predictor.explain_prediction(mgr, as_of_date="2024-04-01")
    assert exp.predicted_yield_t_ha > 0.0


def test_14_weather_provenance_preserved(initialized_farm_manager):
    """Verify data source provenance tags are captured in explanation metadata."""
    initialized_farm_manager.add_weather_observation(WeatherObservation(
        date="2024-04-15",
        weekly_rainfall_mm=40.0,
        avg_temperature_c=26.0,
        relative_humidity_pct=80.0,
        solar_radiation_mj_m2_day=18.0,
        data_source="API_DERIVED",
    ))
    predictor = DynamicYieldPredictor()
    exp = predictor.explain_prediction(initialized_farm_manager, as_of_date="2024-04-15")
    assert "API_DERIVED" in exp.data_sources


def test_15_explain_predictions_script_runs_end_to_end():
    """Verify scripts/explain_predictions.py executes and produces all artifacts."""
    from scripts.explain_predictions import main as run_explain_script
    run_explain_script()

    exp_dir = Path("results/explainability")
    assert (exp_dir / "shap_global_importance.csv").exists()
    assert (exp_dir / "prediction_explanations.csv").exists()
    assert (exp_dir / "farmer_explanations.csv").exists()
    assert (exp_dir / "recommendation_log.csv").exists()
    assert (exp_dir / "explainability_summary.txt").exists()

    plots_dir = Path("results/plots/explainability")
    assert (plots_dir / "shap_global_feature_importance.png").exists()
    assert (plots_dir / "shap_summary_beeswarm.png").exists()
    assert (plots_dir / "shap_waterfall_representative.png").exists()
    assert (plots_dir / "positive_vs_negative_contributions.png").exists()
    assert (plots_dir / "prediction_change_shap_comparison.png").exists()
