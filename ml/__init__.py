"""
AquaSense AI - Machine Learning, XAI, Recharge & Decision Support
Member 4 / Group Leader: ML models, Grouped Cross-Validation, SHAP XAI, Recharge & Risk.
"""

from .dataset import load_or_generate_phelps_dataset, FEATURE_COLUMNS, TARGET_COLUMN
from .models import AquaSenseModelTrainer, get_default_models
from .evaluate import evaluate_unseen_wells_group_kfold, evaluate_temporal_holdout
from .recharge import estimate_recharge_potential, classify_recharge_tier
from .risk import calculate_depletion_velocity, assess_well_risk
from .explainability import ShapExplainerService

__all__ = [
    "load_or_generate_phelps_dataset",
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "AquaSenseModelTrainer",
    "get_default_models",
    "evaluate_unseen_wells_group_kfold",
    "evaluate_temporal_holdout",
    "estimate_recharge_potential",
    "classify_recharge_tier",
    "calculate_depletion_velocity",
    "assess_well_risk",
    "ShapExplainerService",
]
