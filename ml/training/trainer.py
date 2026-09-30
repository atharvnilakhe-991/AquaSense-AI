"""
AquaSense AI - Training Pipeline
Handles model fitting and training execution across configured models.
"""

from typing import Any, Dict
import pandas as pd
from ml.models.factory import get_benchmark_models
from ml.config import RANDOM_STATE


def train_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    Fits a single regressor on training features and target.
    """
    model.fit(X_train, y_train)
    return model


def train_all_benchmark_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """
    Trains all 3 candidate models on the provided training set.

    Returns:
        Dictionary mapping model names to fitted model instances.
    """
    models = get_benchmark_models(random_state=random_state)
    trained_models = {}

    for name, model in models.items():
        print(f"Training {name} on {len(X_train)} samples...")
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models
