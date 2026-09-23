"""
ML Models & Unified Trainer
Member 4 - AI/ML & Evaluation

Supports Random Forest, XGBoost, and LightGBM (preferred model per Section 7).
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Optional imports for LightGBM and XGBoost with graceful fallback to RandomForest if not installed
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def get_default_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns the 3 candidate models tuned for spatial hydrogeological generalization.
    """
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=180,
            max_depth=16,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
    }

    if HAS_LIGHTGBM:
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=220,
            learning_rate=0.045,
            num_leaves=31,
            max_depth=8,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            verbose=-1,
        )

    if HAS_XGBOOST:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.045,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            verbosity=0,
        )

    return models


class AquaSenseModelTrainer:
    """
    Unified training and inference wrapper for AquaSense AI groundwater prediction.
    """

    def __init__(self, model_name: str = "LightGBM", random_state: int = 42):
        self.model_name = model_name
        self.random_state = random_state
        models = get_default_models(random_state=random_state)
        
        if model_name in models:
            self.model = models[model_name]
        else:
            self.model = models["Random Forest"]
            self.model_name = "Random Forest"
            
        self.is_trained = False
        self.feature_names_: Optional[list] = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[list] = None) -> "AquaSenseModelTrainer":
        self.feature_names_ = feature_names
        self.model.fit(X, y)
        self.is_trained = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model has not been trained yet.")
        return self.model.predict(X)

    def get_feature_importances(self) -> Dict[str, float]:
        if not self.is_trained:
            raise RuntimeError("Model must be trained first.")
        
        if hasattr(self.model, "feature_importances_"):
            raw_imp = self.model.feature_importances_
            total = sum(raw_imp) or 1.0
            norm_imp = [round(float(v) / total, 4) for v in raw_imp]
            
            if self.feature_names_ and len(self.feature_names_) == len(norm_imp):
                return dict(zip(self.feature_names_, norm_imp))
            return {f"feat_{i}": v for i, v in enumerate(norm_imp)}
        return {}
