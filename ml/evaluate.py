"""
Rigorous Unseen-Well & Temporal Validation Suite
Member 4 - AI/ML & Evaluation

Validates spatial generalization across unseen wells (GroupKFold) and temporal causality.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from .models import get_default_models


def evaluate_unseen_wells_group_kfold(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "observed_level",
    group_col: str = "well_id",
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Performs GroupKFold cross-validation grouped by well_id.
    
    Guarantees 0% well overlap between train and test sets, simulating true unseen-well prediction.
    """
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df[group_col].values
    
    gkf = GroupKFold(n_splits=n_splits)
    models = get_default_models(random_state=random_state)
    
    results = {}
    for model_name, model in models.items():
        r2_scores = []
        mae_scores = []
        rmse_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train fold model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            r2_scores.append(r2_score(y_test, y_pred))
            mae_scores.append(mean_absolute_error(y_test, y_pred))
            rmse_scores.append(root_mean_squared_error(y_test, y_pred))
            
        results[model_name] = {
            "R2_mean": round(float(np.mean(r2_scores)), 4),
            "R2_std": round(float(np.std(r2_scores)), 4),
            "MAE_mean": round(float(np.mean(mae_scores)), 2),
            "MAE_std": round(float(np.std(mae_scores)), 2),
            "RMSE_mean": round(float(np.mean(rmse_scores)), 2),
            "RMSE_std": round(float(np.std(rmse_scores)), 2),
            "n_folds": n_splits,
            "test_type": "GroupKFold_Unseen_Wells"
        }
        
    return results


def evaluate_temporal_holdout(
    df: pd.DataFrame,
    feature_cols: List[str],
    split_year: int = 2019,
    target_col: str = "observed_level",
    time_col: str = "year",
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates temporal generalization: Train on historical data (<= split_year),
    test on future periods (> split_year) to test chronological causality.
    """
    train_mask = df[time_col] <= split_year
    test_mask = df[time_col] > split_year
    
    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, target_col].values
    X_test = df.loc[test_mask, feature_cols].values
    y_test = df.loc[test_mask, target_col].values
    
    models = get_default_models(random_state=random_state)
    results = {}
    
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[model_name] = {
            "R2": round(float(r2_score(y_test, y_pred)), 4),
            "MAE": round(float(mean_absolute_error(y_test, y_pred)), 2),
            "RMSE": round(float(root_mean_squared_error(y_test, y_pred)), 2),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "split_year": split_year,
            "test_type": "Temporal_Causal_Holdout"
        }
        
    return results
