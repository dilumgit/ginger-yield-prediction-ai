"""SHAP Explainability Module for Ginger Yield Prediction Research.

Implements game-theoretic feature attribution using TreeSHAP on the trained
CatBoost regression model to explain both global feature importance and
instance-level in-season yield forecasts.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import catboost
import joblib
import numpy as np
import pandas as pd
import shap

from src.data.preprocessing import get_model_feature_columns
from src.explainability.explanation_model import FeatureContribution
from src.models.predictor import GingerYieldPredictor
from src.utils.config import BEST_MODEL_PATH


class GingerShapExplainer:
    """SHAP explanation engine for CatBoost ginger yield models."""

    # Human-readable labels and units for the 36 features
    FEATURE_METADATA_MAP = {
        "Days_After_Planting": ("Days After Planting", "days"),
        "Land_Size_Acres": ("Cultivated Land Size", "acres"),
        "Soil_pH": ("Soil pH Level", "pH"),
        "Soil_Moisture_pct": ("Soil Moisture", "%"),
        "Weekly_Rainfall_mm": ("Weekly Rainfall", "mm"),
        "Avg_Temperature_C": ("Average Temperature", "°C"),
        "Relative_Humidity_pct": ("Relative Humidity", "%"),
        "Solar_Radiation_MJ_m2_day": ("Solar Radiation", "MJ/m²/day"),
        "Irrigation_mm_week": ("Weekly Irrigation Applied", "mm"),
        "Fertilizer_kg_acre": ("Fertilizer Applied", "kg/acre"),
        "Plant_Height_cm": ("Plant Height", "cm"),
        "Leaf_Greenness_Index": ("Leaf Greenness Index", "SPAD"),
        "Growth_Stage_Ordinal": ("Growth Stage Level", "stage (0-5)"),
        "Pest_Severity_Ordinal": ("Pest Severity Level", "score (0-2)"),
        "Disease_Severity_Ordinal": ("Disease Severity Level", "score (0-2)"),
        "Total_Water_Input_mm": ("Total Water Input", "mm"),
        "Plant_Height_per_DAP": ("Daily Growth Velocity", "cm/day"),
        "Canopy_Greenness_Volume": ("Canopy Greenness Volume", "index"),
        "Biotic_Stress_Index": ("Biotic Stress Index", "score (0-4)"),
        "Planting_Month": ("Planting Month", "month (1-12)"),
        "District_Badulla": ("District: Badulla", "binary"),
        "District_Galle": ("District: Galle", "binary"),
        "District_Kandy": ("District: Kandy", "binary"),
        "District_Kegalle": ("District: Kegalle", "binary"),
        "District_Kurunegala": ("District: Kurunegala", "binary"),
        "District_Matale": ("District: Matale", "binary"),
        "District_Matara": ("District: Matara", "binary"),
        "District_Monaragala": ("District: Monaragala", "binary"),
        "District_Nuwara Eliya": ("District: Nuwara Eliya", "binary"),
        "District_Ratnapura": ("District: Ratnapura", "binary"),
        "Seed_Variety_Chinese": ("Seed Variety: Chinese", "binary"),
        "Seed_Variety_Local": ("Seed Variety: Local", "binary"),
        "Seed_Variety_Nadun": ("Seed Variety: Nadun", "binary"),
        "Soil_Type_Clay Loam": ("Soil Type: Clay Loam", "binary"),
        "Soil_Type_Loam": ("Soil Type: Loam", "binary"),
        "Soil_Type_Sandy Loam": ("Soil Type: Sandy Loam", "binary"),
    }

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """Initialize SHAP explainer with trained CatBoost model.

        Parameters
        ----------
        model_path : str or Path, optional
            Path to serialized joblib model. Defaults to `BEST_MODEL_PATH`.
        """
        self.model_path = Path(model_path) if model_path is not None else BEST_MODEL_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model artifact not found at '{self.model_path.resolve()}'."
            )

        self.model = joblib.load(self.model_path)
        self.feature_names: List[str] = self._extract_feature_names()

        # Compute standard expected base value (global average yield in training set)
        dummy_row = pd.DataFrame([{feat: 0.0 for feat in self.feature_names}])
        pool = catboost.Pool(dummy_row)
        raw_dummy = self.model.get_feature_importance(pool, type="ShapValues")
        self.base_value: float = float(raw_dummy[0, -1])

    def _extract_feature_names(self) -> List[str]:
        """Extract exact expected 36 feature names from model in precise order."""
        if hasattr(self.model, "feature_names_in_"):
            return list(self.model.feature_names_in_)
        elif hasattr(self.model, "feature_names_"):
            return list(self.model.feature_names_)
        elif hasattr(self.model, "named_steps") and hasattr(self.model.named_steps.get("regressor", None), "feature_names_in_"):
            return list(self.model.named_steps["regressor"].feature_names_in_)
        raise ValueError("Could not extract feature names from fitted model.")

    def _align_features(self, X: Union[pd.DataFrame, pd.Series, Dict[str, Any]]) -> pd.DataFrame:
        """Align input features with exact 36-feature schema expected by CatBoost."""
        if isinstance(X, dict):
            X_df = pd.DataFrame([X])
        elif isinstance(X, pd.Series):
            X_df = pd.DataFrame([X])
        elif isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            raise TypeError(f"Unsupported input type for X: {type(X)}")

        missing_cols = [c for c in self.feature_names if c not in X_df.columns]
        if missing_cols:
            raise ValueError(f"Input is missing required model features: {missing_cols}")

        return X_df[self.feature_names]

    def explain_instance(
        self,
        X: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Generate instance-level SHAP explanation for a single prediction point.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series, or Dict
            Feature inputs.
        top_k : int, default=5
            Number of top positive and negative contributors to extract.

        Returns
        -------
        Dict[str, Any]
            Detailed technical SHAP explanation dictionary.
        """
        X_aligned = self._align_features(X)
        pool = catboost.Pool(X_aligned)
        raw_shap = self.model.get_feature_importance(pool, type="ShapValues")

        shap_vals = raw_shap[0, :-1]
        base_val = float(raw_shap[0, -1])
        pred_val = float(self.model.predict(X_aligned)[0])

        contributions: List[FeatureContribution] = []
        for rank_idx, (feat_name, val, shap_val) in enumerate(
            zip(self.feature_names, X_aligned.iloc[0].values, shap_vals)
        ):
            label, unit = self.FEATURE_METADATA_MAP.get(feat_name, (feat_name, ""))
            direction = "positive" if shap_val >= 0 else "negative"

            # Plain language interpretation
            if shap_val >= 0:
                interp = f"{label} ({val} {unit}) contributed +{abs(shap_val):.3f} t/ha towards a higher yield."
            else:
                interp = f"{label} ({val} {unit}) contributed -{abs(shap_val):.3f} t/ha towards a lower yield."

            contributions.append(
                FeatureContribution(
                    feature_name=feat_name,
                    label=label,
                    value=val,
                    unit=unit,
                    shap_value=round(float(shap_val), 4),
                    abs_shap_value=round(float(abs(shap_val)), 4),
                    direction=direction,
                    rank=0,  # Will be populated after sorting
                    farmer_interpretation=interp,
                )
            )

        # Sort all contributions by absolute impact
        contributions.sort(key=lambda x: x.abs_shap_value, reverse=True)
        for i, c in enumerate(contributions, 1):
            c.rank = i

        # Separate positive and negative contributors
        positive_contribs = sorted(
            [c for c in contributions if c.shap_value > 0],
            key=lambda x: x.shap_value,
            reverse=True,
        )
        negative_contribs = sorted(
            [c for c in contributions if c.shap_value < 0],
            key=lambda x: x.shap_value,  # Most negative first
        )

        return {
            "predicted_yield_t_ha": round(pred_val, 4),
            "base_value_t_ha": round(base_val, 4),
            "sum_shap_values": round(float(np.sum(shap_vals)), 4),
            "all_contributions": contributions,
            "top_positive": positive_contribs[:top_k],
            "top_negative": negative_contribs[:top_k],
            "shap_array": shap_vals,
            "feature_names": self.feature_names,
            "feature_values": X_aligned.iloc[0].to_dict(),
        }

    def explain_dataset(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """Compute global feature importance across a dataset using mean absolute SHAP values.

        Parameters
        ----------
        X : pd.DataFrame
            Feature dataset containing model features.

        Returns
        -------
        Tuple[pd.DataFrame, np.ndarray]
            1. Ranked global feature importance DataFrame.
            2. 2D numpy array of SHAP values.
        """
        X_aligned = self._align_features(X)
        pool = catboost.Pool(X_aligned)
        raw_shap = self.model.get_feature_importance(pool, type="ShapValues")
        shap_matrix = raw_shap[:, :-1]
        mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)

        records = []
        for feat_name, importance in zip(self.feature_names, mean_abs_shap):
            label, unit = self.FEATURE_METADATA_MAP.get(feat_name, (feat_name, ""))
            records.append({
                "Feature": feat_name,
                "Label": label,
                "Unit": unit,
                "Mean_Abs_SHAP_t_ha": round(float(importance), 4),
            })

        df_importance = pd.DataFrame(records).sort_values("Mean_Abs_SHAP_t_ha", ascending=False).reset_index(drop=True)
        df_importance["Rank"] = df_importance.index + 1

        return df_importance, shap_matrix

    def get_shap_explanation_object(
        self, X: Union[pd.DataFrame, pd.Series, Dict[str, Any]]
    ) -> shap.Explanation:
        """Create standard shap.Explanation object for shap visualization functions."""
        X_aligned = self._align_features(X)
        pool = catboost.Pool(X_aligned)
        raw_shap = self.model.get_feature_importance(pool, type="ShapValues")

        shap_vals = raw_shap[:, :-1]
        base_vals = raw_shap[:, -1]

        return shap.Explanation(
            values=shap_vals,
            base_values=base_vals,
            data=X_aligned.values,
            feature_names=[self.FEATURE_METADATA_MAP.get(f, (f, ""))[0] for f in self.feature_names],
        )
