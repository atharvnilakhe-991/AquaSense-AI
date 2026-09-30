"""
AquaSense AI - ML Model Factory
Provides configurations for Random Forest, XGBoost, and LightGBM models.
Parameters match the project's verified spatial generalization benchmark.
"""

from typing import Any, Dict
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from ml.config import RANDOM_STATE


def get_random_forest_regressor(random_state: int = RANDOM_STATE) -> RandomForestRegressor:
    """
    Returns configured Random Forest Regressor for groundwater prediction.
    """
    return RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
    )


def get_xgboost_regressor(random_state: int = RANDOM_STATE) -> XGBRegressor:
    """
    Returns configured XGBoost Regressor for groundwater prediction.
    """
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=random_state,
        n_jobs=-1,
    )


def get_lightgbm_regressor(random_state: int = RANDOM_STATE) -> LGBMRegressor:
    """
    Returns configured LightGBM Regressor for groundwater prediction.
    """
    return LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def get_benchmark_models(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """
    Returns a dictionary of all 3 candidate benchmark models.
    """
    return {
        "LightGBM": get_lightgbm_regressor(random_state=random_state),
        "XGBoost": get_xgboost_regressor(random_state=random_state),
        "Random Forest": get_random_forest_regressor(random_state=random_state),
    }


def get_model(name: str, random_state: int = RANDOM_STATE) -> Any:
    """
    Factory function returning a model by name.
    """
    models = get_benchmark_models(random_state=random_state)
    if name not in models:
        raise ValueError(
            f"Unknown model name '{name}'. Choose from: {list(models.keys())}"
        )
    return models[name]
