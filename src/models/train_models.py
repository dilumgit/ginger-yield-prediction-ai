"""Model Training & Hyperparameter Optimization Module for Ginger Yield Prediction.

Implements candidate model definitions, group-aware cross-validation,
randomized hyperparameter search, and final model training & serialization.

Candidate Models:
1. DummyRegressor (Naive mean baseline)
2. Ridge Regression (Standardized linear baseline)
3. RandomForestRegressor (Bagging ensemble)
4. XGBRegressor (Gradient boosting)
5. LightGBMRegressor (Leaf-wise gradient boosting)
6. CatBoostRegressor (Oblivious tree gradient boosting)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.models.evaluation import evaluate_group_cross_validation
from src.utils.config import MODELS_DIR, RANDOM_STATE


def build_candidate_models(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """Instantiate baseline and candidate regression models with standardized seeds.

    Parameters
    ----------
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping model names to estimator / pipeline objects.
    """
    models = {
        "Dummy (Mean)": DummyRegressor(strategy="mean"),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0, random_state=random_state)),
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            num_leaves=31,
            learning_rate=0.08,
            subsample=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=250,
            depth=6,
            learning_rate=0.08,
            random_state=random_state,
            verbose=0,
        ),
    }
    return models


def get_tuning_search_spaces() -> Dict[str, Dict[str, Any]]:
    """Define targeted hyperparameter distributions for strong candidate models."""
    return {
        "CatBoost": {
            "iterations": [150, 250, 350],
            "depth": [4, 6, 8],
            "learning_rate": [0.03, 0.06, 0.1],
            "l2_leaf_reg": [1, 3, 5],
        },
        "LightGBM": {
            "n_estimators": [100, 150, 200],
            "max_depth": [4, 6, 8],
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.03, 0.06, 0.1],
            "subsample": [0.7, 0.85, 1.0],
        },
        "XGBoost": {
            "n_estimators": [100, 150, 200],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.03, 0.06, 0.1],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
        },
        "Random Forest": {
            "n_estimators": [100, 150, 200],
            "max_depth": [8, 12, 16],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
    }


def tune_candidate_model(
    model_name: str,
    estimator: Any,
    param_dist: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    n_iter: int = 10,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> Tuple[Any, Dict[str, Any], float]:
    """Execute group-aware randomized hyperparameter search.

    Parameters
    ----------
    model_name : str
        Name of model.
    estimator : Any
        Base estimator.
    param_dist : Dict[str, Any]
        Hyperparameter search space.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target series.
    groups_train : pd.Series
        Farm_ID grouping series.
    n_iter : int, default=10
        Number of parameter settings sampled.
    n_splits : int, default=5
        Number of GroupKFold splits.
    random_state : int, default=42
        Random seed.

    Returns
    -------
    Tuple[Any, Dict[str, Any], float]
        (best_estimator, best_params, best_cv_mae)
    """
    gkf = GroupKFold(n_splits=n_splits)

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=list(gkf.split(X_train, y_train, groups=groups_train)),
        random_state=random_state,
        n_jobs=-1 if model_name != "CatBoost" else 1,
        refit=True,
    )

    search.fit(X_train, y_train)
    best_estimator = search.best_estimator_
    best_params = search.best_params_
    best_mae = -float(search.best_score_)

    return best_estimator, best_params, best_mae


def train_and_save_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    save_path: Path,
) -> Any:
    """Fit model on full training set and serialize to disk.

    Parameters
    ----------
    model : Any
        Estimator / pipeline to fit.
    X_train : pd.DataFrame
        Full training features.
    y_train : pd.Series
        Full training target.
    save_path : Path
        Destination path for joblib serialization.

    Returns
    -------
    Any
        Fitted model object.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    return model
