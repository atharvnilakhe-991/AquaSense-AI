"""
AquaSense AI - AMDFE Version 2 Adaptive Weighting & Two-Stage Fusion Engine
Implements two-stage mathematical fusion: G_{i,m} = R_m^alpha * A_{i,m}^beta,
dynamic normalization w_{i,m} = G_{i,m} / sum(G_{i,k}), standardized block scaling,
and feature matrix assemblers for ablations A0 through A6.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml.amdfe.config import (
    MODALITY_A_FEATURES,
    MODALITY_B_FEATURES,
    MODALITY_C_FEATURES,
    TARGET_COLUMN,
    LEAKAGE_COLUMNS,
    ABLATION_CONFIGS_V2,
    ABLATION_CONFIGS_V21,
)


def compute_v2_adaptive_weights(
    reliability_scores: Dict[str, float],
    observability_df: pd.DataFrame,
    active_modalities: List[str] = ["modality_a", "modality_b", "modality_c"],
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Computes normalized adaptive weights w_{i,m} = G_{i,m} / sum_{k} G_{i,k}
    where G_{i,m} = (R_m)^alpha * (A_{i,m})^beta.
    """
    weights_df = pd.DataFrame(index=observability_df.index)
    g_cols = {}

    for mod in ["modality_a", "modality_b", "modality_c", "modality_d"]:
        if mod not in active_modalities or f"A_{mod}" not in observability_df.columns:
            weights_df[f"w_{mod}"] = 0.0
            weights_df[f"G_{mod}"] = 0.0
            weights_df[f"R_{mod}"] = 0.0
            continue

        r_val = max(1e-4, reliability_scores.get(mod, 0.0))
        a_val = observability_df[f"A_{mod}"].clip(lower=1e-4, upper=1.0)
        
        # Two-stage formulation
        g_val = (r_val ** alpha) * (a_val ** beta)
        weights_df[f"G_{mod}"] = g_val
        weights_df[f"R_{mod}"] = r_val
        g_cols[mod] = g_val

    # Sum across active modalities
    active_g_matrix = np.column_stack([g_cols[m] for m in active_modalities if m in g_cols])
    sum_g = active_g_matrix.sum(axis=1)
    fallback_flag = (sum_g <= 1e-5).astype(int)

    num_active = len(active_modalities)
    for mod in active_modalities:
        if mod in g_cols:
            w_vals = np.where(
                sum_g > 1e-5,
                g_cols[mod] / sum_g,
                1.0 / num_active
            )
            weights_df[f"w_{mod}"] = w_vals

    weights_df["Fallback_Flag"] = fallback_flag
    return weights_df, fallback_flag


def fuse_v2_modality_blocks(
    df_std: pd.DataFrame,
    weights_df: pd.DataFrame,
    reliability_scores: Dict[str, float],
    active_modalities: List[str] = ["modality_a", "modality_b", "modality_c"],
    fusion_type: str = "adaptive_amdfe",
) -> pd.DataFrame:
    """
    Applies the specified fusion transformation across standardized modality blocks.
    """
    fused_df = df_std.copy()
    mod_feature_map = {
        "modality_a": MODALITY_A_FEATURES,
        "modality_b": MODALITY_B_FEATURES,
        "modality_c": MODALITY_C_FEATURES,
    }
    num_active = len(active_modalities)

    # Calculate global weights for global reliability baseline (A4)
    active_r = [reliability_scores.get(m, 0.0) for m in active_modalities]
    sum_r = sum(active_r) if sum(active_r) > 0 else 1.0
    global_weights = {m: reliability_scores.get(m, 0.0) / sum_r for m in active_modalities}

    for mod in active_modalities:
        cols = mod_feature_map.get(mod, [])
        valid_std_cols = [f"Std_{c}" for c in cols if f"Std_{c}" in fused_df.columns]

        if fusion_type in ["single", "concat", "metadata_only", "metadata_only_all"]:
            # Unweighted standardized features (C0, C1, A0, A1, A2, A3)
            for std_col in valid_std_cols:
                raw_name = std_col.replace("Std_", "")
                fused_df[f"Fused_{raw_name}"] = fused_df[std_col]

        elif fusion_type == "global_reliability":
            # Globally scaled features w_m * Z(X) (A4)
            gw = global_weights.get(mod, 1.0 / num_active)
            for std_col in valid_std_cols:
                raw_name = std_col.replace("Std_", "")
                fused_df[f"Fused_{raw_name}"] = gw * fused_df[std_col]

        elif fusion_type in ["adaptive_amdfe", "adaptive_amdfe_plus_meta"]:
            # Sample-level adaptive gated features w_{i,m} * Z(X) (C2, A5, A6)
            w_series = weights_df[f"w_{mod}"] if f"w_{mod}" in weights_df.columns else 0.0
            for std_col in valid_std_cols:
                raw_name = std_col.replace("Std_", "")
                fused_df[f"Fused_{raw_name}"] = w_series * fused_df[std_col]

    return fused_df


def get_v2_feature_columns_for_configuration(
    df: pd.DataFrame,
    config_key: str = "A5",
) -> Tuple[List[str], Dict[str, int]]:
    """
    Returns the exact feature columns and explicit feature counts for configurations A0 - A6 and C0 - C2.
    Guarantees strict feature-count fairness and zero leakage.
    """
    if config_key in ABLATION_CONFIGS_V21:
        cfg = ABLATION_CONFIGS_V21[config_key]
    elif config_key in ABLATION_CONFIGS_V2:
        cfg = ABLATION_CONFIGS_V2[config_key]
    else:
        raise ValueError(f"Unknown AMDFE configuration '{config_key}'.")

    modalities = cfg["modalities"]

    base_cols = []
    meta_cols = []
    rel_cols = []
    weight_cols = []

    # 1. Base Modality Features (Standardized or Fused)
    for mod in modalities:
        if mod == "modality_a":
            base_cols += [f"Fused_{c}" for c in MODALITY_A_FEATURES if f"Fused_{c}" in df.columns]
        elif mod == "modality_b":
            base_cols += [f"Fused_{c}" for c in MODALITY_B_FEATURES if f"Fused_{c}" in df.columns]
        elif mod == "modality_c":
            base_cols += [f"Fused_{c}" for c in MODALITY_C_FEATURES if f"Fused_{c}" in df.columns]

    # 2. Metadata Augmentation (A3, A4, A5, A6, C1, C2)
    if cfg.get("has_reliability_meta", False):
        # Static source reliability features
        rel_cols += [f"R_{m}" for m in modalities if f"R_{m}" in df.columns]

    if cfg.get("has_observability_meta", False):
        # Per-sample observability features
        meta_cols += [f"A_{m}" for m in modalities if f"A_{m}" in df.columns]

    if cfg.get("has_reliability_weights", False):
        # Dynamic weights
        weight_cols += [f"w_{m}" for m in modalities if f"w_{m}" in df.columns]

    if config_key in ["A6"]:
        # Robustness metadata: missingness indicators, gap flags, outlier flags
        all_raw = MODALITY_A_FEATURES + MODALITY_B_FEATURES + MODALITY_C_FEATURES
        meta_cols += [f"Missing_{c}" for c in all_raw if f"Missing_{c}" in df.columns]
        if "Fallback_Flag" in df.columns:
            meta_cols.append("Fallback_Flag")
        if "Weather_Available_At_Prediction" in df.columns:
            meta_cols.append("Weather_Available_At_Prediction")
        if "Outlier_Flag_Previous_WatLevel" in df.columns:
            meta_cols.append("Outlier_Flag_Previous_WatLevel")
        if "Robust_Z_Previous_WatLevel" in df.columns:
            meta_cols.append("Robust_Z_Previous_WatLevel")

    all_features = base_cols + rel_cols + meta_cols + weight_cols

    # Leakage purges
    all_features = [c for c in all_features if c not in LEAKAGE_COLUMNS and c != TARGET_COLUMN]
    
    # De-duplicate while preserving order
    seen = set()
    unique_features = [c for c in all_features if not (c in seen or seen.add(c))]

    counts = {
        "base_feature_count": len(base_cols),
        "reliability_feature_count": len(rel_cols),
        "metadata_feature_count": len(meta_cols),
        "adaptive_weight_feature_count": len(weight_cols),
        "total_feature_count": len(unique_features),
    }

    return unique_features, counts


def extract_v2_feature_matrix(
    df: pd.DataFrame,
    config_key: str = "A5",
    target_col: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, int]]:
    """
    Extracts predictor matrix X, target vector y, and feature count audit dictionary.
    """
    feature_cols, counts = get_v2_feature_columns_for_configuration(df, config_key=config_key)

    if target_col in feature_cols:
        raise ValueError(f"CRITICAL LEAKAGE: Target '{target_col}' found in predictor columns!")

    for leak in LEAKAGE_COLUMNS:
        if leak in feature_cols:
            raise ValueError(f"CRITICAL LEAKAGE: Target-derived column '{leak}' found in predictors!")

    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y, counts
