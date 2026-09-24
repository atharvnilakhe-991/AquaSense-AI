"""
AquaSense AI - Evaluation Metrics & Comparison
Computes standard regression metrics (R², MAE, RMSE, Pearson r) and formats comparison tables.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Computes standard regression evaluation metrics:
    - R² (Coefficient of Determination)
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - Pearson r (Correlation between true and predicted)

    Returns:
        Dictionary with metric names and values.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    r2 = float(r2_score(y_t, y_p))
    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))

    # Pearson r
    if np.std(y_t) > 0 and np.std(y_p) > 0:
        pearson_r = float(np.corrcoef(y_t, y_p)[0, 1])
    else:
        pearson_r = 0.0

    return {
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "Pearson_r": pearson_r,
    }


def evaluate_model_predictions(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Generates predictions from a trained model and computes evaluation metrics.

    Returns:
        Tuple of (predictions_array, metrics_dict).
    """
    y_pred = model.predict(X_test)
    metrics = calculate_regression_metrics(y_test.values, y_pred)
    return y_pred, metrics


def create_model_comparison_table(
    results: List[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Formats model evaluation records into a structured summary DataFrame.
    """
    df = pd.DataFrame(results)
    if "Test_R2" in df.columns:
        df = df.sort_values(by="Test_R2", ascending=False).reset_index(drop=True)
    return df
