"""
AquaSense AI - AMDFE Feature Assembler
Constructs clean predictor matrices and targets for all 5 AMDFE Ablation configurations (B0 - B4).
"""

from typing import List, Tuple, Dict
import pandas as pd

from ml.amdfe.config import (
    ABLATION_CONFIGS,
    MODALITY_A_FEATURES,
    MODALITY_B_FEATURES,
    MODALITY_C_FEATURES,
    TARGET_COLUMN,
    LEAKAGE_COLUMNS,
)


def get_feature_columns_for_configuration(
    df: pd.DataFrame,
    config_key: str = "B4",
) -> List[str]:
    """
    Returns the list of valid predictor column names for a given AMDFE ablation configuration.
    """
    if config_key not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown AMDFE configuration '{config_key}'. Choose from: {list(ABLATION_CONFIGS.keys())}")

    cfg = ABLATION_CONFIGS[config_key]
    fusion_type = cfg["fusion_type"]
    modalities = cfg["modalities"]

    feature_cols = []

    if config_key == "B0":
        # Groundwater only (Modality A)
        feature_cols = [c for c in MODALITY_A_FEATURES if c in df.columns]
        # Include missingness indicators
        feature_cols += [f"Missing_{c}" for c in MODALITY_A_FEATURES if f"Missing_{c}" in df.columns]

    elif config_key == "B1":
        # Groundwater + Weather (Modalities A + B)
        raw_cols = MODALITY_A_FEATURES + MODALITY_B_FEATURES
        feature_cols = [c for c in raw_cols if c in df.columns]
        feature_cols += [f"Missing_{c}" for c in raw_cols if f"Missing_{c}" in df.columns]
        if "Weather_Available_At_Prediction" in df.columns:
            feature_cols.append("Weather_Available_At_Prediction")

    elif config_key == "B2":
        # Fixed-weight multimodal fusion (Fused_ features across A, B, C)
        all_raw = MODALITY_A_FEATURES + MODALITY_B_FEATURES + MODALITY_C_FEATURES
        feature_cols = [f"Fused_{c}" for c in all_raw if f"Fused_{c}" in df.columns]
        feature_cols += [f"Missing_{c}" for c in all_raw if f"Missing_{c}" in df.columns]

    elif config_key == "B3":
        # Global reliability-weighted fusion
        all_raw = MODALITY_A_FEATURES + MODALITY_B_FEATURES + MODALITY_C_FEATURES
        feature_cols = [f"Fused_{c}" for c in all_raw if f"Fused_{c}" in df.columns]
        feature_cols += [f"Missing_{c}" for c in all_raw if f"Missing_{c}" in df.columns]
        # Modality global weights
        feature_cols += [f"w_{m}" for m in ["modality_a", "modality_b", "modality_c"] if f"w_{m}" in df.columns]

    elif config_key == "B4":
        # Full Context-Adaptive AMDFE
        all_raw = MODALITY_A_FEATURES + MODALITY_B_FEATURES + MODALITY_C_FEATURES
        feature_cols = [f"Fused_{c}" for c in all_raw if f"Fused_{c}" in df.columns]
        feature_cols += [f"Missing_{c}" for c in all_raw if f"Missing_{c}" in df.columns]
        # Dynamic weights and availability/reliability indicators
        feature_cols += [f"w_{m}" for m in ["modality_a", "modality_b", "modality_c"] if f"w_{m}" in df.columns]
        feature_cols += [f"R_{m}" for m in ["modality_a", "modality_b", "modality_c"] if f"R_{m}" in df.columns]
        feature_cols += [f"A_{m}" for m in ["modality_a", "modality_b", "modality_c"] if f"A_{m}" in df.columns]
        if "Fallback_Flag" in df.columns:
            feature_cols.append("Fallback_Flag")
        if "Weather_Available_At_Prediction" in df.columns:
            feature_cols.append("Weather_Available_At_Prediction")

    # Enforce strict leakage purge
    feature_cols = [c for c in feature_cols if c not in LEAKAGE_COLUMNS and c != TARGET_COLUMN]

    # De-duplicate while preserving order
    seen = set()
    unique_feature_cols = [c for c in feature_cols if not (c in seen or seen.add(c))]
    return unique_feature_cols


def extract_feature_matrix(
    df: pd.DataFrame,
    config_key: str = "B4",
    target_col: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts predictor matrix X and target y for the chosen configuration.
    """
    feature_cols = get_feature_columns_for_configuration(df, config_key=config_key)
    
    # Assert target is not in feature columns
    if target_col in feature_cols:
        raise ValueError(f"CRITICAL LEAKAGE: Target '{target_col}' found in predictor columns!")

    for leak in LEAKAGE_COLUMNS:
        if leak in feature_cols:
            raise ValueError(f"CRITICAL LEAKAGE: Target-derived column '{leak}' found in predictors!")

    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y
