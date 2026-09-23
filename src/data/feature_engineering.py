"""Feature Engineering Module for Ginger Yield Prediction Research.

Implements pure, non-destructive, and scientifically defensible agronomic
feature transformations based strictly on in-season observable data.

Engineered Features:
1. Total_Water_Input_mm: Cumulative weekly water from rainfall and irrigation.
2. Plant_Height_per_DAP: Mean daily vertical growth rate (cm/day).
3. Canopy_Greenness_Volume: Proxy for active photosynthetic canopy biomass.
4. Biotic_Stress_Index: Combined severity index from pests and diseases (0-4).
5. Planting_Month: Seasonal temporal feature extracted from Planting_Date.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.utils.config import ORDINAL_MAPPINGS


def create_water_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create water availability features from rainfall and irrigation.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing Weekly_Rainfall_mm and Irrigation_mm_week.

    Returns
    -------
    pd.DataFrame
        Copy of dataframe with Total_Water_Input_mm added.
    """
    df_out = df.copy()
    if "Weekly_Rainfall_mm" in df_out.columns and "Irrigation_mm_week" in df_out.columns:
        df_out["Total_Water_Input_mm"] = (
            df_out["Weekly_Rainfall_mm"] + df_out["Irrigation_mm_week"]
        ).round(3)
    return df_out


def create_growth_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create growth dynamic features including daily vertical growth rate.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing Plant_Height_cm and Days_After_Planting.

    Returns
    -------
    pd.DataFrame
        Copy of dataframe with Plant_Height_per_DAP added.
    """
    df_out = df.copy()
    if "Plant_Height_cm" in df_out.columns and "Days_After_Planting" in df_out.columns:
        dap = df_out["Days_After_Planting"]
        height = df_out["Plant_Height_cm"]
        # Safe division avoiding divide-by-zero
        rate = np.where(dap > 0, height / dap, 0.0)
        df_out["Plant_Height_per_DAP"] = pd.Series(rate, index=df_out.index).round(4)
    return df_out


def create_health_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create crop health and biotic stress indices.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing Plant_Height_cm, Leaf_Greenness_Index,
        Pest_Severity, and Disease_Severity.

    Returns
    -------
    pd.DataFrame
        Copy of dataframe with Canopy_Greenness_Volume and Biotic_Stress_Index added.
    """
    df_out = df.copy()

    # Canopy Greenness Volume (Photosynthetic Biomass Proxy)
    if "Plant_Height_cm" in df_out.columns and "Leaf_Greenness_Index" in df_out.columns:
        df_out["Canopy_Greenness_Volume"] = (
            df_out["Plant_Height_cm"] * df_out["Leaf_Greenness_Index"]
        ).round(3)

    # Biotic Stress Index (Combined Pest & Disease Severity 0 to 4)
    if "Pest_Severity" in df_out.columns and "Disease_Severity" in df_out.columns:
        pest_map = ORDINAL_MAPPINGS["Pest_Severity"]
        disease_map = ORDINAL_MAPPINGS["Disease_Severity"]

        # Map to ordinal integers
        if pd.api.types.is_numeric_dtype(df_out["Pest_Severity"]):
            pest_ord = df_out["Pest_Severity"]
        else:
            pest_ord = df_out["Pest_Severity"].astype(str).map(pest_map).fillna(0)

        if pd.api.types.is_numeric_dtype(df_out["Disease_Severity"]):
            disease_ord = df_out["Disease_Severity"]
        else:
            disease_ord = df_out["Disease_Severity"].astype(str).map(disease_map).fillna(0)

        df_out["Biotic_Stress_Index"] = (pest_ord + disease_ord).astype(int)

    return df_out


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract research-justified temporal features from Planting_Date.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing Planting_Date.

    Returns
    -------
    pd.DataFrame
        Copy of dataframe with Planting_Month extracted as an integer.
    """
    df_out = df.copy()
    if "Planting_Date" in df_out.columns:
        dates = pd.to_datetime(df_out["Planting_Date"], errors="coerce")
        df_out["Planting_Month"] = dates.dt.month.fillna(0).astype(int)
    return df_out


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Execute complete feature engineering pipeline sequentially.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or validated dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe enriched with all defensible engineered features.
    """
    df_feat = df.copy()
    df_feat = create_water_features(df_feat)
    df_feat = create_growth_features(df_feat)
    df_feat = create_health_features(df_feat)
    df_feat = create_temporal_features(df_feat)
    return df_feat


def get_feature_metadata_records() -> List[Dict[str, Any]]:
    """Return comprehensive metadata for all raw, encoded, and engineered features."""
    return [
        # Identifier
        {
            "Feature": "Farm_ID",
            "Original_or_Engineered": "Original",
            "Data_Type": "String",
            "Source_Features": "Raw Dataset",
            "Description": "Unique farm identification code (e.g. F0001 to F1000).",
            "In_Season_Available": "Yes",
            "Used_For_Model": "No",
            "Reason": "Identifier / grouping column. Excluded to prevent spurious farm-id memorization.",
        },
        # Raw Temporal
        {
            "Feature": "Planting_Date",
            "Original_or_Engineered": "Original",
            "Data_Type": "Date String",
            "Source_Features": "Raw Dataset",
            "Description": "Rhizome planting date (YYYY-MM-DD).",
            "In_Season_Available": "Yes",
            "Used_For_Model": "No",
            "Reason": "Raw date string retained for traceability; decomposed into Planting_Month.",
        },
        # Raw Numerical
        {
            "Feature": "Days_After_Planting",
            "Original_or_Engineered": "Original",
            "Data_Type": "Integer",
            "Source_Features": "Raw Dataset",
            "Description": "Observation time point measured in days elapsed since planting.",
            "In_Season_Available": "Yes",
            "Used_For_Model": "Yes",
            "Reason": "Primary temporal index of crop phenological maturity.",
        },
        {
            "Feature": "Land_Size_Acres",
            "Original_or_Engineered": "Original",
            "Data_Type": "Float",
            "Source_Features": "Raw Dataset",
            "Description": "Farm plot cultivation size in acres.",
            "In_Season_Available": "Yes",
            "Used_For_Model": "Yes",
            "Reason": "Farm scale covariate impacting management capacity.",
        },
        {
            "Feature": "Soil_pH",
            "Original_or_Engineered": "Original",
            "Data_Type": "Float",
            "Source_Features": "Raw Dataset",
            "Description": "Soil acidity/alkalinity level (optimal for ginger: 5.5 - 6.5).",
            "In_Season_Available": "Yes",
            "Used_For_Model": "Yes",
            "Reason": "Critical soil chemistry parameter governing nutrient availability.",
        },
        {
            "Feature": "Soil_Moisture_pct",
            "Original_or_Engineered": "Original",
            "Data_Type": "Float",
            "Source_Features": "Raw Dataset",
            "Description": "Volumetric soil moisture percentage.",
            "In_Season_Available": "Yes",
            "Used_For_Model": "Yes",
            "Reason": "Direct root-zone water availability indicator.",
        },
        {
            "Feature": "Weekly_Rainfall_mm",
            "Original_or_Engineered": "Original",
            "Data_Type": "Float",
            "Source_Features": "Raw Dataset",
            "Description": "Cumulative rainfall recorded in observation week (mm).",
            "In_Season_Available": "Yes",
            "Used_For_Model": "Yes",
            "Reason": "Primary meteorological water source.",
        },
        {
            "Feature": "Avg_Temperature_C",
            "Original_or_Engineered": "Original",
            "Data_Type": "Float",
            "Source_Features": "Raw Dataset",
            "Description": "Mean ambient air temperature (°C).",
            "In_Season_Available": "Yes",
            "Used_For_Model": "Yes",
            "Reason": "Thermal driving force for vegetative growth and evapotranspiration.",
        },
        {
            "Feature": "Relative_Humidity_pct",
            "Original_or_Engineered": "Original",
            "Data_Type": "Float",
            "Source_Features": "Raw Dataset",
            "Description": "Mean relative atmospheric humidity (%).",
            "In_Season_Available": "Yes",
            "Used_For_Model": "Yes",
            "Reason": "Atmospheric moisture indicator affecting transpirational demand and fungal risk.",
        },
        {
            "Feature": "Solar_Radiation_MJ_m2_day",
            "Original_or_Engineered": "Original",
            "Data_Type": "Float",
            "Source_Features": "Raw Dataset",
            "Description": "Daily incident solar radiation flux (MJ/m²/day).",
            "In_Season_Available": "Yes",
            "Used_For_Model": "Yes",
            "Reason": "Photosynthetically active radiation driver for biomass accumulation.",
        },
        {
            "Feature": "Irrigation_mm_week",
            "Original_or_Engineered": "Original",
            "Data_Type": "Float",
            "Source_Features": "Raw Dataset",
            "Description": "Supplemental irrigation applied per week (mm).",
            "In_Season_Available": "Yes",
            "Used_For_Model": "Yes",
            "Reason": "Farmer-controlled water input compensating for rainfall deficit.",
        },
        {
            "Feature": "Fertilizer_kg_acre",
            "Original_or_Engineered": "Original",
            "Data_Type": "Float",
            "Source_Features": "Raw Dataset",
            "Description": "Fertilizer dosage applied per acre (kg/acre).",
            "In_Season_Available": "Yes",
            "Used_For_Model": "Yes",
            "Reason": "Nutrient input driving shoot and rhizome biomass synthesis.",
        },
        {
            "Feature": "Plant_Height_cm",
            "Original_or_Engineered": "Original",
            "Data_Type": "Float",
            "Source_Features": "Raw Dataset",
            "Description": "Measured shoot height (cm).",
            "In_Season_Available": "Yes",
            "Used_For_Model": "Yes",
            "Reason": "Direct morphological indicator of vegetative vigor.",
        },
        {
            "Feature": "Leaf_Greenness_Index",
            "Original_or_Engineered": "Original",
            "Data_Type": "Float",
            "Source_Features": "Raw Dataset",
            "Description": "SPAD / chlorophyll proxy index of foliage.",
            "In_Season_Available": "Yes",
            "Used_For_Model": "Yes",
            "Reason": "Direct proxy for leaf nitrogen content and photosynthetic capacity.",
        },
        # Ordinal Encoded Features
        {
            "Feature": "Growth_Stage_Ordinal",
            "Original_or_Engineered": "Engineered",
            "Data_Type": "Integer (0-5)",
            "Source_Features": "Growth_Stage",
            "Description": "Ordinal biological progression: Sprouting (0) -> Early Veg (1) -> Veg (2) -> Rhizome Init (3) -> Rhizome Dev (4) -> Maturity (5).",
            "In_Season_Available": "Yes",
            "Used_For_Model": "Yes",
            "Reason": "Preserves sequential biological phenology across the crop lifecycle.",
        },
        {
            "Feature": "Pest_Severity_Ordinal",
            "Original_or_Engineered": "Engineered",
            "Data_Type": "Integer (0-2)",
            "Source_Features": "Pest_Severity",
            "Description": "Ordinal pest severity score: Low (0), Medium (1), High (2).",
            "In_Season_Available": "Yes",
            "Used_For_Model": "Yes",
            "Reason": "Quantifies progressive biotic damage from insect infestation.",
        },
        {
            "Feature": "Disease_Severity_Ordinal",
            "Original_or_Engineered": "Engineered",
            "Data_Type": "Integer (0-2)",
            "Source_Features": "Disease_Severity",
            "Description": "Ordinal disease severity score: Low (0), Medium (1), High (2).",
            "In_Season_Available": "Yes",
            "Used_For_Model": "Yes",
            "Reason": "Quantifies progressive fungal/bacterial rhizome rot pressure.",
        },
        # Nominal One-Hot Encoded Features
        {
            "Feature": "District_[DistrictName]",
            "Original_or_Engineered": "Engineered",
            "Data_Type": "Binary (0/1)",
            "Source_Features": "District",
            "Description": "10 One-Hot encoded binary indicators for Sri Lankan ginger growing districts.",
            "In_Season_Available": "Yes",
            "Used_For_Model": "Yes",
            "Reason": "Captures agro-ecological zone specific baseline conditions without imposing artificial ordering.",
        },
        {
            "Feature": "Seed_Variety_[VarietyName]",
            "Original_or_Engineered": "Engineered",
            "Data_Type": "Binary (0/1)",
            "Source_Features": "Seed_Variety",
            "Description": "3 One-Hot encoded binary indicators (Chinese, Local, Nadun).",
            "In_Season_Available": "Yes",
            "Used_For_Model": "Yes",
            "Reason": "Encodes cultivar genetic yield potentials without imposing artificial ordering.",
        },
        {
            "Feature": "Soil_Type_[SoilType]",
            "Original_or_Engineered": "Engineered",
            "Data_Type": "Binary (0/1)",
            "Source_Features": "Soil_Type",
            "Description": "3 One-Hot encoded binary indicators (Clay Loam, Loam, Sandy Loam).",
            "In_Season_Available": "Yes",
            "Used_For_Model": "Yes",
            "Reason": "Encodes soil physical drainage and aeration characteristics.",
        },
        # Engineered In-Season Features
        {
            "Feature": "Total_Water_Input_mm",
            "Original_or_Engineered": "Engineered",
            "Data_Type": "Float",
            "Source_Features": "Weekly_Rainfall_mm, Irrigation_mm_week",
            "Description": "Combined weekly water supply from rainfall and supplemental irrigation: Weekly_Rainfall_mm + Irrigation_mm_week.",
            "In_Season_Available": "Yes",
            "Used_For_Model": "Yes",
            "Reason": "Represents net water available to crop root zone during observation week.",
        },
        {
            "Feature": "Plant_Height_per_DAP",
            "Original_or_Engineered": "Engineered",
            "Data_Type": "Float",
            "Source_Features": "Plant_Height_cm, Days_After_Planting",
            "Description": "Mean daily vertical growth velocity: Plant_Height_cm / Days_After_Planting (cm/day).",
            "In_Season_Available": "Yes",
            "Used_For_Model": "Yes",
            "Reason": "Captures vegetative shoot vigor and early establishment rate.",
        },
        {
            "Feature": "Canopy_Greenness_Volume",
            "Original_or_Engineered": "Engineered",
            "Data_Type": "Float",
            "Source_Features": "Plant_Height_cm, Leaf_Greenness_Index",
            "Description": "Photosynthetic biomass proxy: Plant_Height_cm * Leaf_Greenness_Index.",
            "In_Season_Available": "Yes",
            "Used_For_Model": "Yes",
            "Reason": "Strong proxy for cumulative active photosynthetic leaf volume and chlorophyll concentration.",
        },
        {
            "Feature": "Biotic_Stress_Index",
            "Original_or_Engineered": "Engineered",
            "Data_Type": "Integer (0-4)",
            "Source_Features": "Pest_Severity, Disease_Severity",
            "Description": "Combined biotic stress severity score: Pest_Severity_Ordinal + Disease_Severity_Ordinal.",
            "In_Season_Available": "Yes",
            "Used_For_Model": "Yes",
            "Reason": "Captures cumulative compounding crop stress from multiple biological threats.",
        },
        {
            "Feature": "Planting_Month",
            "Original_or_Engineered": "Engineered",
            "Data_Type": "Integer (1-12)",
            "Source_Features": "Planting_Date",
            "Description": "Calendar month of planting extracted from Planting_Date.",
            "In_Season_Available": "Yes",
            "Used_For_Model": "Yes",
            "Reason": "Captures planting timing relative to Sri Lankan bimodal monsoonal patterns (Yala / Maha).",
        },
        # Benchmark / Leakage
        {
            "Feature": "Predicted_Final_Yield_t_ha",
            "Original_or_Engineered": "Original",
            "Data_Type": "Float",
            "Source_Features": "Raw Dataset",
            "Description": "Pre-existing external baseline yield prediction.",
            "In_Season_Available": "Yes",
            "Used_For_Model": "No",
            "Reason": "LEAKAGE RISK: Exhibits 0.988 correlation with target. Retained in processed data only as external benchmark.",
        },
        # Target
        {
            "Feature": "Actual_Final_Yield_t_ha",
            "Original_or_Engineered": "Original",
            "Data_Type": "Float",
            "Source_Features": "Raw Dataset",
            "Description": "Ground truth final harvested ginger yield in metric tons per hectare (t/ha).",
            "In_Season_Available": "No (End of season)",
            "Used_For_Model": "Target (y)",
            "Reason": "Primary prediction target for all supervised machine learning models.",
        },
    ]
