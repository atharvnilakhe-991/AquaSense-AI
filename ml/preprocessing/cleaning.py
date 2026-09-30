"""
AquaSense AI - Preprocessing & Cleaning
Handles missing-value management, type normalization, and integrity verification.
"""

from typing import List, Optional, Tuple
import pandas as pd
from ml.config import PRIMARY_FEATURES, TARGET_COLUMN


def clean_dataset(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    target_column: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, int]:
    """
    Cleans dataset by dropping rows with missing values in the feature or target columns.

    Parameters:
        df: Input DataFrame.
        feature_columns: List of feature columns to validate (defaults to PRIMARY_FEATURES).
        target_column: Target column name.

    Returns:
        Tuple of (cleaned_df, num_dropped_rows).
    """
    features = feature_columns or PRIMARY_FEATURES
    subset = features + [target_column]
    
    initial_count = len(df)
    cleaned = df.dropna(subset=subset).copy()
    dropped = initial_count - len(cleaned)

    return cleaned, dropped


def validate_data_integrity(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    target_column: str = TARGET_COLUMN,
) -> dict:
    """
    Performs data validation audit including missingness, value ranges, and target stats.

    Returns:
        Dictionary summarizing validation metrics.
    """
    features = feature_columns or PRIMARY_FEATURES
    metrics = {
        "total_rows": len(df),
        "target_nulls": int(df[target_column].isna().sum()),
        "target_min": float(df[target_column].min()),
        "target_max": float(df[target_column].max()),
        "target_mean": float(df[target_column].mean()),
        "target_std": float(df[target_column].std()),
        "feature_nulls": {col: int(df[col].isna().sum()) for col in features},
    }
    return metrics
