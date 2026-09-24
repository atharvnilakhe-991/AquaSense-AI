"""
AquaSense AI - Dataset Loading & Splitting
Implements strict leakage-safe loading and group-based unseen-well splitting.
"""

from pathlib import Path
from typing import List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.config import (
    LEAKAGE_SAFE_DATASET,
    PRIMARY_FEATURES,
    TARGET_COLUMN,
    WELL_ID_COLUMN,
    YEAR_COLUMN,
    RANDOM_STATE,
    TEST_WELL_RATIO,
    LEAKAGE_COLUMNS,
)


def load_groundwater_dataset(
    filepath: Optional[Path] = None,
    enforce_leakage_check: bool = True,
) -> pd.DataFrame:
    """
    Loads the groundwater prediction dataset and validates data integrity.

    Parameters:
        filepath: Optional path to dataset CSV. Defaults to LEAKAGE_SAFE_DATASET.
        enforce_leakage_check: If True, asserts no target-derived TIFF columns are loaded.

    Returns:
        pd.DataFrame containing groundwater observations and features.
    """
    path = filepath or LEAKAGE_SAFE_DATASET
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Groundwater dataset not found at: {path}. "
            f"Please verify project paths."
        )

    df = pd.read_csv(path)

    # Verify required base columns exist
    required_cols = PRIMARY_FEATURES + [TARGET_COLUMN, WELL_ID_COLUMN]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Enforce strict leakage prevention
    if enforce_leakage_check:
        found_leakage = [c for c in LEAKAGE_COLUMNS if c in df.columns]
        if found_leakage:
            raise ValueError(
                f"Target leakage violation! Found target-derived columns: {found_leakage}. "
                f"These must NOT be used as ML predictors."
            )

    return df


def split_by_unseen_wells(
    df: pd.DataFrame,
    well_column: str = WELL_ID_COLUMN,
    test_size: float = TEST_WELL_RATIO,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Splits dataset into training and testing sets based on UNIQUE WELL IDENTIFIERS.
    This guarantees that wells in the test set are completely UNSEEN during training,
    evaluating true spatial hydrogeological generalization.

    Parameters:
        df: Input DataFrame.
        well_column: Column name identifying wells (default: CSD_ID).
        test_size: Proportion of unique wells to hold out (default: 0.20).
        random_state: Random state for reproducibility (default: 42).

    Returns:
        Tuple of (train_df, test_df, train_well_ids, test_well_ids).
    """
    unique_wells = df[well_column].drop_duplicates().values
    train_wells, test_wells = train_test_split(
        unique_wells,
        test_size=test_size,
        random_state=random_state,
    )

    # Verify zero overlap between train and test wells
    overlap = set(train_wells).intersection(set(test_wells))
    if len(overlap) > 0:
        raise RuntimeError(
            f"Data integrity error: {len(overlap)} wells overlap between train and test!"
        )

    train_df = df[df[well_column].isin(train_wells)].copy()
    test_df = df[df[well_column].isin(test_wells)].copy()

    return train_df, test_df, train_wells, test_wells


def split_by_temporal_cutoff(
    df: pd.DataFrame,
    cutoff_year: int = 2020,
    year_column: str = YEAR_COLUMN,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset temporally by cutoff year to evaluate future prediction capability.

    Parameters:
        df: Input DataFrame.
        cutoff_year: Year dividing train and test sets (default: 2020).
        year_column: Column name indicating observation year.

    Returns:
        Tuple of (train_df, test_df).
    """
    if year_column not in df.columns:
        raise ValueError(f"Year column '{year_column}' not found in dataset.")

    train_df = df[df[year_column] < cutoff_year].copy()
    test_df = df[df[year_column] >= cutoff_year].copy()

    return train_df, test_df
