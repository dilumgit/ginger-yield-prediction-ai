"""Streamlit User Interface & Interactive Prototype (Phase 6).

End-to-End dynamic in-season ginger yield prediction prototype with
continuous state management, farmer-friendly irrigation logging,
TreeSHAP explainability, and rule-based decision support.

Usage:
    streamlit run src/ui/app.py
"""

import sys
from datetime import date, datetime
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.data.farm_state import (
    CropObservation,
    FarmProfile,
    FarmStateManager,
    WeatherObservation,
    calculate_dap,
    suggest_growth_stage,
)
from src.data.irrigation import IrrigationEvent
from src.data.weather_api import (
    AbstractWeatherProvider,
    MockWeatherProvider,
    OpenMeteoWeatherProvider,
    WeatherAPIError,
)
from src.dynamic.dynamic_predictor import DynamicYieldPredictor
from src.dynamic.state_updater import update_farm_state
from src.explainability.farmer_translator import FarmerExplanationTranslator
from src.explainability.recommendations import FarmerRecommendationEngine
from src.explainability.shap_explainer import GingerShapExplainer
from src.storage.farm_repository import FarmRepository
from src.utils.config import (
    DISTRICTS,
    GROWTH_STAGES,
    SEED_VARIETIES,
    SEVERITY_LEVELS,
    SOIL_TYPES,
)
from src.utils.units import format_harvest_kg, total_yield_kg, yield_delta_kg


# Page Configuration
st.set_page_config(
    page_title="Ginger Yield AI | Sri Lanka",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_repository() -> FarmRepository:
    """Initialize repository and seed demo farms if needed."""
    repo = FarmRepository()
    repo.seed_demo_farms()
    return repo


def main():
    repo = get_repository()

    # 1. Sidebar: Farm Selection & Navigation
    st.sidebar.title("🌱 Ginger Yield AI")
    st.sidebar.markdown(
        "**Explainable Dynamic Yield Prediction**  \n"
        "*Sri Lankan Smallholder Research Prototype (IT41012)*"
    )
    st.sidebar.markdown("---")

    farm_ids = repo.list_farms()

    # Track active farm in session state
    if "active_farm_id" in st.session_state and st.session_state["active_farm_id"] in farm_ids:
        active_idx = farm_ids.index(st.session_state["active_farm_id"])
    else:
        active_idx = 0
        if farm_ids:
            st.session_state["active_farm_id"] = farm_ids[0]
        else:
            st.session_state.pop("active_farm_id", None)

    if farm_ids:
        active_farm_id = st.sidebar.selectbox("🏠 Select Active Farm", options=farm_ids, index=active_idx, key="active_farm_select")
        st.session_state["active_farm_id"] = active_farm_id

        # Load active farm state
        loaded = repo.load_farm_state(active_farm_id)
        if loaded is None:
            st.sidebar.error(f"Failed to load farm {active_farm_id}.")
            farm_mgr = None
            predictor = None
            profile = None
        else:
            farm_mgr, predictor = loaded
            profile = farm_mgr.profile
            # Display profile snippet in sidebar
            st.sidebar.info(
                f"**District:** {profile.district}  \n"
                f"**Land Size:** {profile.land_size_acres} Acres  \n"
                f"**Variety:** {profile.seed_variety} | **Soil:** {profile.soil_type}  \n"
                f"**Planting Date:** {profile.planting_date}  \n"
                f"**Pump Capacity:** {f'{profile.pump_capacity_lph:,.0f} L/h' if profile.pump_capacity_lph else 'Unknown (Runtime Fallback)'}"
            )
    else:
        active_farm_id = None
        farm_mgr = None
        predictor = None
        profile = None
        st.sidebar.warning("⚠️ No farms available in repository.")

    st.sidebar.markdown("---")
    menu = st.sidebar.radio(
        "Navigation",
        [
            "📊 Farm Dashboard",
            "📝 Update Crop Observation",
            "💧 Log Irrigation Event",
            "📈 Prediction History & Trajectory",
            "🔍 Explain Prediction (SHAP XAI)",
            "⚙️ Farm Profile Management",
            "ℹ️ Research & Data Provenance",
        ],
    )

    st.sidebar.markdown("---")
    wx_source_option = st.sidebar.selectbox(
        "🌦️ Weather Provider",
        ["Open-Meteo (Live API)", "Mock Simulator (Offline Research)"],
        index=0,
        help="External weather service used for automatic precipitation and climate data ingestion.",
    )
    if "Open-Meteo" in wx_source_option:
        active_weather_provider = OpenMeteoWeatherProvider()
    else:
        active_weather_provider = MockWeatherProvider(seed=42)

    st.sidebar.caption("Zero Hardware Requirement | TreeSHAP Explainability | Continuous Updating")

    # Handle global empty state when no farms exist
    if not farm_ids or farm_mgr is None:
        if menu == "⚙️ Farm Profile Management":
            pass  # Allowed to create new farm
        else:
            st.title("🌱 Ginger Yield AI — Sri Lanka")
            st.warning(
                "⚠️ **No farm profiles are currently available.**  \n\n"
                "Please go to **'⚙️ Farm Profile Management'** in the navigation menu to register a new farm profile."
            )
            if st.button("🌱 Load Demonstration Farms", type="primary"):
                repo.seed_demo_farms()
                st.rerun()
            return

    # Generate or retrieve latest prediction & explanation
    latest_obs = farm_mgr.crop_observations[-1] if (farm_mgr and farm_mgr.crop_observations) else None
    latest_date = latest_obs.date if latest_obs else (profile.planting_date if profile else date.today().strftime("%Y-%m-%d"))

    # -------------------------------------------------------------
    # VIEW 1: DASHBOARD
    # -------------------------------------------------------------
    if menu == "📊 Farm Dashboard":
        st.title(f"📊 Farm Dashboard — {profile.farm_id}")
        st.caption(f"Location: {profile.district} District | Variety: {profile.seed_variety} | Cultivation Area: {profile.land_size_acres} Acres")

        if not farm_mgr.crop_observations:
            st.info(
                "ℹ️ **No crop observations are available yet.**  \n"
                "Please add the first crop observation in **'📝 Update Crop Observation'** to generate a yield prediction."
            )
            return

        # Reconstruct current state and explanation
        exp = predictor.explain_prediction(farm_mgr, as_of_date=latest_date)
        state = farm_mgr.reconstruct_current_state(as_of_date=latest_date)

        # KPI Metrics Row
        total_harvest_kg = total_yield_kg(exp.predicted_yield_t_ha, profile.land_size_acres)
        delta_kg = yield_delta_kg(exp.prediction_delta_t_ha, profile.land_size_acres)
        delta_str = f"{delta_kg:+,.0f} kg ({exp.prediction_delta_pct:+.1f}%)" if delta_kg is not None else None

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="Expected Final Harvest",
                value=f"{total_harvest_kg:,.0f} kg",
                delta=delta_str,
                help=f"Total expected harvest for {profile.land_size_acres:.2f} acres. Based on model estimate: {exp.predicted_yield_t_ha:.2f} t/ha",
            )
            st.caption(f"For **{profile.land_size_acres:.2f} acres**  \n*Model estimate: `{exp.predicted_yield_t_ha:.2f} t/ha`*")
        with col2:
            st.metric(
                label="Crop Age & Stage",
                value=f"Day {exp.days_after_planting}",
                delta=exp.growth_stage,
                delta_color="off",
            )
        with col3:
            st.metric(
                label="Plant Height",
                value=f"{state.get('Plant_Height_cm', 0.0):.1f} cm",
                delta=f"{state.get('Plant_Height_per_DAP', 0.0):.2f} cm/day",
                delta_color="off",
            )
        with col4:
            recent_irr = state.get("Pump_Runtime_Hours", 0.0)
            st.metric(
                label="Recent 7-Day Irrigation",
                value=f"{recent_irr:.1f} Hours",
                delta=f"Total: {state.get('Cumulative_Irrigation_Hours', 0.0):.1f} h",
                delta_color="off",
            )

        st.markdown("---")

        # Two-Column Layout: Farmer Narrative & Recommendations + Trajectory
        c_left, c_right = st.columns([1.1, 0.9])

        with c_left:
            st.subheader("🌾 Farmer Plain-Language Summary")
            st.info(
                f"🌾 **Expected Total Harvest:** **{total_harvest_kg:,.0f} kg** (for {profile.land_size_acres:.2f} acres)  \n"
                f"*(Research model estimate: {exp.predicted_yield_t_ha:.2f} t/ha)*\n\n"
                f"{exp.farmer_summary}"
            )

            st.markdown("#### 💡 Actionable Agricultural Recommendations")
            for rec in exp.recommendations:
                if "[Pest Alert]" in rec or "[Disease Alert]" in rec:
                    st.error(rec)
                elif "[Water Management]" in rec:
                    st.warning(rec)
                elif "[Irrigation Advice]" in rec:
                    st.info(rec)
                else:
                    st.success(rec)

            st.markdown("#### 🔍 Key Model Influences (SHAP Attributions)")
            cp1, cp2 = st.columns(2)
            with cp1:
                st.markdown("**Supporting Factors (Positive):**")
                for c in exp.top_positive_factors[:3]:
                    st.markdown(f"- ✅ **{c.label}** (`+{c.shap_value:.2f} t/ha`)")
            with cp2:
                st.markdown("**Limiting Factors (Negative):**")
                for c in exp.top_negative_factors[:3]:
                    st.markdown(f"- ⚠️ **{c.label}** (`{c.shap_value:.2f} t/ha`)")

        with c_right:
            st.subheader("📈 Prediction Evolution Trajectory")
            history_recs = predictor.history.get_farm_history(profile.farm_id)
            if history_recs:
                df_traj = pd.DataFrame([
                    {
                        "DAP": r.days_after_planting,
                        "Expected_Harvest_kg": round(total_yield_kg(r.predicted_yield_t_ha, profile.land_size_acres)),
                        "Forecast_Yield_t_ha": r.predicted_yield_t_ha,
                        "Stage": r.growth_stage,
                        "Delta_kg": round(yield_delta_kg(r.prediction_delta_t_ha, profile.land_size_acres)) if r.prediction_delta_t_ha is not None else None,
                        "Delta_t_ha": r.prediction_delta_t_ha,
                    }
                    for r in history_recs
                ])
                st.line_chart(df_traj.set_index("DAP")["Expected_Harvest_kg"])
                st.dataframe(
                    df_traj[["DAP", "Stage", "Expected_Harvest_kg", "Forecast_Yield_t_ha", "Delta_kg"]].rename(
                        columns={
                            "Expected_Harvest_kg": "Expected Harvest (kg)",
                            "Forecast_Yield_t_ha": "Model Est (t/ha)",
                            "Delta_kg": "Δ Harvest (kg)",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

    # -------------------------------------------------------------
    # VIEW 2: UPDATE CROP OBSERVATION
    # -------------------------------------------------------------
    elif menu == "📝 Update Crop Observation":
        st.title(f"📝 Update In-Season Crop Observation — {profile.farm_id}")
        st.markdown(
            "Enter newly scouted crop measurements. "
            "**You do NOT need to re-enter historical observations or manually calculate DAP.** "
            "The system preserves all previous records and automatically calculates Days After Planting (DAP)."
        )

        try:
            p_date = datetime.strptime(str(profile.planting_date)[:10], "%Y-%m-%d").date()
        except Exception:
            p_date = date.today()

        # Sensible initial default observation date
        latest_obs = farm_mgr.crop_observations[-1] if farm_mgr.crop_observations else None
        if latest_obs:
            try:
                latest_obs_dt = datetime.strptime(str(latest_obs.date)[:10], "%Y-%m-%d").date()
                default_obs = max(p_date, latest_obs_dt)
            except Exception:
                default_obs = p_date
        else:
            default_obs = p_date

        c1, c2 = st.columns(2)
        with c1:
            obs_date = st.date_input(
                "Observation Date",
                value=default_obs,
                min_value=p_date,
                help=f"Date of field observation. Cannot precede registered planting date ({profile.planting_date}).",
                key="obs_date_input",
            )
            if isinstance(obs_date, (tuple, list)):
                obs_date = obs_date[0] if obs_date else default_obs

            # Validate observation date vs planting date
            if obs_date < p_date:
                st.error(
                    f"❌ Invalid Date: Observation Date ({obs_date}) cannot be earlier than Planting Date ({p_date})."
                )
                calculated_dap = 0
            else:
                calculated_dap = calculate_dap(p_date, obs_date)

            st.text_input(
                "Days After Planting (DAP)",
                value=f"{calculated_dap} days",
                disabled=True,
                help=f"Automatically calculated: {obs_date} minus planting date ({p_date}) = {calculated_dap} days.",
            )

            suggested_stage = suggest_growth_stage(calculated_dap)
            default_stage_idx = GROWTH_STAGES.index(suggested_stage) if suggested_stage in GROWTH_STAGES else 0

            growth_stage = st.selectbox(
                "Growth Stage",
                options=GROWTH_STAGES,
                index=default_stage_idx,
                help=f"Suggested default for DAP {calculated_dap} is '{suggested_stage}'. Confirm or adjust based on actual field observation.",
                key="growth_stage_select",
            )
            plant_height = st.number_input(
                "Plant Height (cm)", min_value=1.0, max_value=200.0, value=42.0, step=1.0, key="plant_height_input"
            )
            leaf_greenness = st.number_input(
                "Leaf Greenness Index (SPAD / 0-100)", min_value=1.0, max_value=100.0, value=58.0, step=1.0, key="greenness_input"
            )

        # Automatic Weather Retrieval for Previous 7 Calendar Days ending on obs_date
        wx_error_msg = None
        wx_data = None
        if obs_date >= p_date:
            try:
                wx_data = active_weather_provider.get_weather_for_date(
                    district=profile.district,
                    date=str(obs_date),
                )
            except WeatherAPIError as we:
                wx_error_msg = str(we)
            except Exception as e:
                wx_error_msg = f"Unable to retrieve weather data: {str(e)}"

        with c2:
            pest_sev = st.selectbox("Pest Severity", options=SEVERITY_LEVELS, index=0, key="pest_input")
            disease_sev = st.selectbox("Disease Severity", options=SEVERITY_LEVELS, index=0, key="disease_input")
            soil_moisture = st.number_input(
                "Soil Moisture (%) [Optional/Estimated]", min_value=10.0, max_value=100.0, value=65.0, step=1.0, key="moisture_input"
            )
            fertilizer_kg = st.number_input(
                "Fertilizer Applied to Date (kg/acre)", min_value=0.0, max_value=500.0, value=120.0, step=5.0, key="fertilizer_input"
            )

            # Display Automatically Retrieved 7-Day Rainfall (Read-Only)
            if wx_data is not None:
                rain_val = wx_data.get("Weekly_Rainfall_mm", 0.0)
                wx_src = wx_data.get("Source", "Open-Meteo")
                wx_prov = wx_data.get("Data_Source", "API_DERIVED")
                w_start = wx_data.get("Window_Start", "")
                w_end = wx_data.get("Window_End", str(obs_date))

                st.text_input(
                    "🌧️ Weekly Rainfall (Previous 7-Day Precipitation)",
                    value=f"{rain_val:.2f} mm",
                    disabled=True,
                    help=f"7-day cumulative rainfall ({w_start} to {w_end}) automatically retrieved from {wx_src} ({wx_prov}). Strictly excludes future dates.",
                )
                st.caption(f"Source: **{wx_src}** | Provenance: `{wx_prov}` | Window: {w_start} to {w_end}")
            else:
                st.text_input(
                    "🌧️ Weekly Rainfall (Previous 7-Day Precipitation)",
                    value="Unavailable / Error",
                    disabled=True,
                    help="Weather data could not be retrieved from the provider.",
                )
                if wx_error_msg:
                    st.error(f"⚠️ Weather API Error: {wx_error_msg}")
                    st.info("💡 Switch to 'Mock Simulator' in the sidebar if working offline or in test environments.")

        submitted = st.button(
            "🌱 Submit Observation & Update Forecast",
            type="primary",
            use_container_width=True,
            key="submit_crop_obs_btn",
        )

        if submitted:
            if obs_date < p_date:
                st.error(
                    f"❌ Cannot submit observation: Observation Date ({obs_date}) is earlier than Planting Date ({p_date})."
                )
            elif wx_data is None:
                st.error(
                    f"❌ Cannot submit observation without valid weather data: {wx_error_msg or 'Weather data unavailable'}. "
                    f"Please check your internet connection or switch to Mock Simulator in the sidebar."
                )
            else:
                pred_val, rec, analysis = update_farm_state(
                    farm_manager=farm_mgr,
                    dynamic_predictor=predictor,
                    observation_date=str(obs_date),
                    days_after_planting=int(calculated_dap),
                    growth_stage=growth_stage,
                    plant_height_cm=float(plant_height),
                    leaf_greenness_index=float(leaf_greenness),
                    pest_severity=pest_sev,
                    disease_severity=disease_sev,
                    soil_moisture_pct=float(soil_moisture),
                    fertilizer_kg_acre=float(fertilizer_kg),
                    weather_provider=active_weather_provider,
                )
                repo.save_farm_state(farm_mgr, predictor)

                total_kg = total_yield_kg(pred_val, profile.land_size_acres)
                st.success(
                    f"✅ Farm state updated! **New Expected Harvest: {total_kg:,.0f} kg** (for {profile.land_size_acres:.2f} acres)  \n"
                    f"*Model estimate: **{pred_val:.2f} t/ha** | DAP: **{calculated_dap} days** | Stage: **{growth_stage}** | "
                    f"7-Day Rain: **{wx_data['Weekly_Rainfall_mm']:.2f} mm** via {wx_data.get('Source', 'API')}*"
                )
                if rec.prediction_delta_t_ha is not None:
                    delta_kg = yield_delta_kg(rec.prediction_delta_t_ha, profile.land_size_acres)
                    st.info(
                        f"📊 **Harvest Shift:** **{delta_kg:+,.0f} kg** ({analysis['prediction_delta_pct']:+.1f}%) | "
                        f"*Model shift: `{rec.prediction_delta_t_ha:+.2f} t/ha`*"
                    )

    # -------------------------------------------------------------
    # VIEW 3: LOG IRRIGATION EVENT
    # -------------------------------------------------------------
    elif menu == "💧 Log Irrigation Event":
        st.title(f"💧 Log Farmer-Friendly Irrigation Event — {profile.farm_id}")
        st.markdown(
            "Record supplemental irrigation using **pump operating runtime**. "
            "No volumetric liter measurements required."
        )

        with st.form("irrigation_form"):
            c1, c2 = st.columns(2)
            with c1:
                irr_date = st.date_input("Irrigation Event Date", value=date.today())
                irrigated_yes = st.radio("Did you irrigate the field?", ["Yes", "No"], index=0)
                hours = st.number_input("Pump Runtime (Hours)", min_value=0, max_value=24, value=2, step=1)
                minutes = st.number_input("Pump Runtime (Minutes)", min_value=0, max_value=59, value=30, step=5)

            with c2:
                water_src = st.selectbox("Water Source", ["Well", "Stream", "Canal", "Rainwater Tank"], index=0)
                irr_method = st.selectbox("Irrigation Method", ["Sprinkler", "Drip", "Flood / Furrow", "Manual Hose"], index=0)

                total_runtime_h = hours + (minutes / 60.0)
                st.markdown(f"**Total Operating Duration:** `{total_runtime_h:.2f} Hours`")

                if profile.pump_capacity_lph:
                    est_depth = (total_runtime_h * profile.pump_capacity_lph) / (profile.land_size_acres * 4046.86)
                    st.success(f"💧 Calculated Depth Equivalent: **{est_depth:.2f} mm** (Pump: {profile.pump_capacity_lph:,.0f} L/h)")
                else:
                    st.info("ℹ️ Pump capacity not registered. Preserving runtime hours directly as observable feature.")

            log_submitted = st.form_submit_button("💧 Log Irrigation & Refresh State", use_container_width=True)

            if log_submitted:
                if irrigated_yes == "Yes" and total_runtime_h > 0:
                    farm_mgr.add_irrigation_event(IrrigationEvent(
                        date=str(irr_date),
                        runtime_hours=float(total_runtime_h),
                        water_source=water_src,
                        irrigation_method=irr_method,
                    ))
                    if farm_mgr.crop_observations:
                        try:
                            pred_val, rec, analysis = predictor.predict_current_state(farm_mgr, as_of_date=str(irr_date))
                            repo.save_farm_state(farm_mgr, predictor)
                            total_kg = total_yield_kg(pred_val, profile.land_size_acres)
                            st.success(
                                f"✅ Irrigation logged! Cumulative Runtime: **{farm_mgr.irrigation_log.get_cumulative_runtime_hours(str(irr_date)):.1f} Hours** | "
                                f"**Expected Harvest: {total_kg:,.0f} kg** (*Model estimate: `{pred_val:.2f} t/ha` for {profile.land_size_acres:.2f} acres*)"
                            )
                        except Exception:
                            repo.save_farm_state(farm_mgr, predictor)
                            st.success(f"✅ Irrigation logged! Cumulative Runtime: **{farm_mgr.irrigation_log.get_cumulative_runtime_hours(str(irr_date)):.1f} Hours**")
                    else:
                        repo.save_farm_state(farm_mgr, predictor)
                        st.success(
                            f"✅ Irrigation logged! Cumulative Runtime: **{farm_mgr.irrigation_log.get_cumulative_runtime_hours(str(irr_date)):.1f} Hours**  \n"
                            f"*Note: No crop observations are available yet. Add observations in 'Update Crop Observation' to generate a yield prediction.*"
                        )
                else:
                    st.warning("Zero runtime logged. No event recorded.")

        # Event History Table
        st.subheader("📋 Historical Irrigation Event Log")
        events = farm_mgr.irrigation_log.events
        if events:
            df_irr = pd.DataFrame([
                {
                    "Date": ev.date,
                    "Runtime (Hours)": ev.runtime_hours,
                    "Water Source": ev.water_source,
                    "Method": ev.irrigation_method,
                    "Data Source": ev.data_source,
                }
                for ev in events
            ])
            st.dataframe(df_irr, use_container_width=True, hide_index=True)
        else:
            st.info("No historical irrigation events recorded yet.")

    # -------------------------------------------------------------
    # VIEW 4: PREDICTION HISTORY & TRAJECTORY
    # -------------------------------------------------------------
    elif menu == "📈 Prediction History & Trajectory":
        st.title(f"📈 Prediction History & Evolution Trajectory — {profile.farm_id}")
        st.markdown("Chronological record of all dynamic yield forecasts generated across the cultivation period.")

        history_recs = predictor.history.get_farm_history(profile.farm_id)
        if not history_recs:
            st.info(
                "ℹ️ **No predictions recorded yet.**  \n"
                "Please add crop observations in **'📝 Update Crop Observation'** to generate yield predictions."
            )
            return

        df_hist = pd.DataFrame([
            {
                "Timestamp": r.prediction_timestamp,
                "DAP": r.days_after_planting,
                "Growth Stage": r.growth_stage,
                "Expected Harvest (kg)": round(total_yield_kg(r.predicted_yield_t_ha, profile.land_size_acres)),
                "Forecast (t/ha)": r.predicted_yield_t_ha,
                "Delta (kg)": round(yield_delta_kg(r.prediction_delta_t_ha, profile.land_size_acres)) if r.prediction_delta_t_ha is not None else 0.0,
                "Delta (t/ha)": r.prediction_delta_t_ha if r.prediction_delta_t_ha is not None else 0.0,
                "Model Version": r.model_version,
            }
            for r in history_recs
        ])

        c1, c2 = st.columns([1.2, 0.8])
        with c1:
            st.subheader("Expected Harvest Trajectory Curve (kg)")
            st.line_chart(df_hist.set_index("DAP")["Expected Harvest (kg)"])
        with c2:
            st.subheader("Harvest Deltas (Δ kg)")
            st.bar_chart(df_hist.set_index("DAP")["Delta (kg)"])

        st.subheader("📋 Complete Audit Trail")
        st.dataframe(df_hist, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------
    # VIEW 5: EXPLAIN PREDICTION (SHAP XAI)
    # -------------------------------------------------------------
    elif menu == "🔍 Explain Prediction (SHAP XAI)":
        st.title(f"🔍 Explainable AI (SHAP) Factor Attribution — {profile.farm_id}")
        st.markdown(
            r"Game-theoretic feature attribution using **TreeSHAP** on the trained CatBoost model. "
            r"Decomposes predicted yield relative to expected baseline ($\phi_0 = 13.65\text{ t/ha}$)."
        )

        if not farm_mgr.crop_observations:
            st.info(
                "ℹ️ **No crop observations are available yet.**  \n"
                "Please add the first crop observation in **'📝 Update Crop Observation'** to generate a SHAP explanation."
            )
            return

        exp = predictor.explain_prediction(farm_mgr, as_of_date=latest_date)
        total_kg = total_yield_kg(exp.predicted_yield_t_ha, profile.land_size_acres)
        base_kg = total_yield_kg(exp.base_value_t_ha, profile.land_size_acres)

        st.markdown(
            f"### Expected Final Harvest: **{total_kg:,.0f} kg** (for {profile.land_size_acres:.2f} acres)  \n"
            f"*Model Prediction: **{exp.predicted_yield_t_ha:.2f} t/ha** | Research Baseline ($\\phi_0$): {exp.base_value_t_ha:.2f} t/ha ({base_kg:,.0f} kg)*"
        )
        st.info(exp.farmer_summary)

        # Plot Diverging Horizontal Bar Chart
        st.subheader("Top Positive vs Negative Contributing Factors")
        top_pos = exp.top_positive_factors[:5]
        top_neg = exp.top_negative_factors[:5]
        all_top = top_neg[::-1] + top_pos

        labels = [f"{c.label} ({c.value} {c.unit})" for c in all_top]
        values = [c.shap_value for c in all_top]
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in values]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(labels, values, color=colors, alpha=0.85, height=0.55)
        ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
        ax.set_xlabel("SHAP Value (t/ha relative to baseline: 13.65 t/ha)", fontweight="bold")
        ax.set_title(f"Local Feature Attribution (DAP {exp.days_after_planting}: {exp.growth_stage})", pad=10)
        for i, val in enumerate(values):
            offset = 0.02 if val >= 0 else -0.02
            ha = "left" if val >= 0 else "right"
            ax.text(val + offset, i, f"{val:+.2f} t/ha", va="center", ha=ha, fontsize=8.5, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Detailed Attribution Table
        st.subheader("Detailed Feature Attribution Table")
        df_exp = pd.DataFrame([
            {
                "Rank": c.rank,
                "Feature": c.label,
                "Observed Value": f"{c.value} {c.unit}",
                "SHAP Impact (t/ha)": f"{c.shap_value:+.3f}",
                "Direction": "Supporting (Positive)" if c.direction == "positive" else "Limiting (Negative)",
                "Farmer Interpretation": c.farmer_interpretation,
            }
            for c in exp.top_positive_factors + exp.top_negative_factors
        ])
        st.dataframe(df_exp, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------
    # VIEW 6: FARM PROFILE MANAGEMENT
    # -------------------------------------------------------------
    elif menu == "⚙️ Farm Profile Management":
        st.title("⚙️ Farm Profile Management")
        st.markdown("Create a new farm profile or manage the active farm.")

        # Section 1: Create New Farm Profile
        st.subheader("🌱 Register New Farm Profile")
        with st.form("new_farm_form"):
            c1, c2 = st.columns(2)
            with c1:
                next_num = len(farm_ids) + 1 if farm_ids else 1
                new_id = st.text_input("Farm ID", value=f"F{next_num:04d}")
                dist = st.selectbox("District", DISTRICTS, index=0)
                variety = st.selectbox("Seed Variety", SEED_VARIETIES, index=0)
                soil = st.selectbox("Soil Type", SOIL_TYPES, index=0)
                acres = st.number_input("Land Size (Acres)", min_value=0.1, max_value=50.0, value=1.0, step=0.1)

            with c2:
                p_date = st.date_input("Planting Date", value=date.today())
                w_src = st.selectbox("Water Source", ["Well", "Stream", "Canal", "Rainwater Tank"], index=0)
                i_meth = st.selectbox("Irrigation Method", ["Sprinkler", "Drip", "Flood / Furrow", "Manual Hose"], index=0)
                p_cap = st.number_input("Pump Capacity (L/h) [Optional, 0 if unknown]", min_value=0.0, max_value=100000.0, value=8000.0, step=500.0)

            create_sub = st.form_submit_button("🌱 Save New Farm Profile", use_container_width=True)

            if create_sub:
                clean_id = new_id.strip()
                if not clean_id:
                    st.error("❌ Farm ID cannot be empty.")
                else:
                    new_profile = FarmProfile(
                        farm_id=clean_id,
                        district=dist,
                        seed_variety=variety,
                        soil_type=soil,
                        land_size_acres=float(acres),
                        planting_date=str(p_date),
                        water_source=w_src,
                        irrigation_method=i_meth,
                        pump_capacity_lph=float(p_cap) if p_cap > 0 else None,
                    )
                    new_mgr = FarmStateManager(new_profile)
                    new_pred = DynamicYieldPredictor()
                    repo.save_farm_state(new_mgr, new_pred)
                    st.session_state["active_farm_id"] = clean_id
                    st.success(f"✅ Created Farm Profile: **{clean_id}**! Automatically switched to this farm.")
                    st.rerun()

        # Section 2: Delete Active Farm Profile
        if profile is not None:
            st.markdown("---")
            st.subheader("🗑️ Delete Active Farm Profile")
            st.markdown(
                f"Permanently delete active farm **{profile.farm_id}** "
                f"({profile.district} District | {profile.land_size_acres} Acres | Variety: {profile.seed_variety})."
            )

            confirm_key = f"confirm_delete_{profile.farm_id}"
            if confirm_key not in st.session_state:
                st.session_state[confirm_key] = False

            if not st.session_state[confirm_key]:
                if st.button(
                    "🗑️ Delete Farm Profile",
                    type="secondary",
                    use_container_width=True,
                    key=f"btn_init_delete_{profile.farm_id}",
                ):
                    st.session_state[confirm_key] = True
                    st.rerun()
            else:
                st.warning(
                    f"⚠️ **Are you sure you want to delete Farm '{profile.farm_id}'?**  \n\n"
                    f"Deleting this farm will permanently remove the farm profile (**{profile.farm_id}**) "
                    f"and all associated crop observations, irrigation event logs, and historical prediction records.  \n"
                    f"**This action cannot be undone.**"
                )
                col_cancel, col_confirm = st.columns(2)
                with col_cancel:
                    if st.button("❌ Cancel", use_container_width=True, key=f"btn_cancel_delete_{profile.farm_id}"):
                        st.session_state[confirm_key] = False
                        st.rerun()
                with col_confirm:
                    if st.button("🗑️ Delete Permanently", type="primary", use_container_width=True, key=f"btn_confirm_delete_{profile.farm_id}"):
                        target_id = profile.farm_id
                        st.session_state[confirm_key] = False
                        deleted = repo.delete_farm(target_id, predictor=predictor)
                        if deleted:
                            remaining = repo.list_farms()
                            if remaining:
                                st.session_state["active_farm_id"] = remaining[0]
                            else:
                                st.session_state.pop("active_farm_id", None)
                            st.success(f"✅ Farm profile **{target_id}** and all associated records have been permanently deleted.")
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to delete farm '{target_id}'.")

    # -------------------------------------------------------------
    # VIEW 7: RESEARCH & DATA PROVENANCE
    # -------------------------------------------------------------
    elif menu == "ℹ️ Research & Data Provenance":
        st.title("ℹ️ Research & Data Provenance")
        st.markdown(
            r"""
            ### Research Methods and Scientific Writing (IT41012)
            **Project**: *An Explainable AI-Based Dynamic In-Season Ginger Yield Prediction System for Sri Lankan Farmers*

            ---

            #### 1. Core Principles & Non-Causal Distinction
            - **Prediction**: Quantitative harvest forecast ($\hat{y}$ in $t/ha$) from tuned CatBoost regressor.
            - **Explanation**: Game-theoretic Shapley value attribution ($\hat{y} = \phi_0 + \sum \phi_i$) quantifying feature impacts.
            - **Causation**: SHAP values explain **model behavior**, not biological cause-and-effect.

            #### 2. Zero Physical Hardware Declaration
            This research prototype is designed for smallholder feasibility and **strictly operates without IoT sensors, cameras, drones, or physical weather stations**.

            #### 3. Data Source Provenance Taxonomy
            | Source Classification | Description |
            | :--- | :--- |
            | `FARMER_REPORTED` | Field measurements observed directly by the farmer (shoot height, greenness, pest scouting, pump runtime). |
            | `API_DERIVED` | Meteorological data retrieved via external weather APIs (e.g. Open-Meteo). |
            | `HISTORICAL_DATASET` | Verified historical records and baseline survey data. |
            | `SYNTHETIC_SIMULATED` | Controlled simulation data generated for offline experimentation and testing. |

            #### 4. Temporal Leakage Safeguards
            For any forecast generated at date $t$, the system programmatically filters all inputs to ensure that **only records with timestamp $\le t$ are used**. Future observations, future rainfall, and future irrigation events are strictly inaccessible.
            """
        )


if __name__ == "__main__":
    main()
