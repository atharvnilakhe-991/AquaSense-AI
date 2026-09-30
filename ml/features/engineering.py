"""
AquaSense AI - Feature Engineering & Observation-History Handling
Extracts environmental predictors and constructs leakage-safe irregular monitoring features.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from ml.config import (
    PRIMARY_FEATURES,
    TARGET_COLUMN,
    WELL_ID_COLUMN,
    DATE_COLUMN,
)


def get_primary_feature_matrix(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    target: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts the feature matrix X and target vector y.

    Parameters:
        df: Input DataFrame.
        features: List of feature columns (defaults to PRIMARY_FEATURES).
        target: Target column name (defaults to TARGET_COLUMN).

    Returns:
        Tuple of (X, y).
    """
    cols = features or PRIMARY_FEATURES
    X = df[cols].copy()
    y = df[target].copy()
    return X, y


def engineer_temporal_observation_features(
    df: pd.DataFrame,
    well_column: str = WELL_ID_COLUMN,
    date_column: str = DATE_COLUMN,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """
    Builds temporal observation-history features for irregular monitoring
    under strict temporal leakage guarantees:
    - ALL features use strictly prior observations (DateMsr_prev < DateMsr_t).
    - Current date is NEVER included.
    - Captures previous water level, days since previous measurement, and rolling mean/trend.

    Parameters:
        df: Input DataFrame.
        well_column: Well identifier column.
        date_column: Date string column.
        target_column: Groundwater level column.

    Returns:
        pd.DataFrame with added observation-history features.
    """
    data = df.copy()
    data[date_column] = pd.to_datetime(data[date_column])
    data = data.sort_values(by=[well_column, date_column]).reset_index(drop=True)

    # 1. Unique prior observation features per well
    data["Previous_WatLevel"] = (
        data.groupby(well_column)[target_column].shift(1)
    )
    data["Previous2_WatLevel"] = (
        data.groupby(well_column)[target_column].shift(2)
    )

    # 2. Time elapsed since previous measurement
    data["Days_Since_Previous"] = (
        data.groupby(well_column)[date_column].diff().dt.days
    )
    data["Years_Since_Previous"] = data["Days_Since_Previous"] / 365.25

    # 3. Observation counts and rolling mean
    data["Previous_Observation_Count"] = (
        data.groupby(well_column).cumcount()
    )
    data["Rolling_Mean_3"] = (
        data.groupby(well_column)[target_column]
        .shift(1)
        .rolling(window=3, min_periods=1)
        .mean()
    )

    # 4. Gaps and long gap indicators
    data["Long_Gap_Flag"] = (data["Days_Since_Previous"] > 365).astype(int)

    return data
