"""
AquaSense AI - AMDFE Adaptive Weight Assignment & Multimodal Fusion
Calculates dynamic sample weights w_{i,m}, standardizes modality blocks using
training statistics, and creates weighted fused representations X'_{i,m}.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml.amdfe.config import (
    MODALITY_A_FEATURES,
    MODALITY_B_FEATURES,
    MODALITY_C_FEATURES,
)
from ml.amdfe.reliability import (
    compute_global_reliability_scores,
    compute_context_aware_reliability,
)


class ModalityBlockScaler:
    """
    Fits and applies StandardScaler on feature subsets using training partition statistics.
    """

    def __init__(self):
        self.scalers: Dict[str, StandardScaler] = {}
        self.is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame, modality_features: Dict[str, List[str]]) -> "ModalityBlockScaler":
        for mod, cols in modality_features.items():
            valid_cols = [c for c in cols if c in train_df.columns]
            if valid_cols:
                scaler = StandardScaler()
                scaler.fit(train_df[valid_cols])
                self.scalers[mod] = scaler
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, modality_features: Dict[str, List[str]]) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("ModalityBlockScaler must be fitted on train before transform!")

        data = df.copy()
        for mod, cols in modality_features.items():
            valid_cols = [c for c in cols if c in data.columns]
            if valid_cols and mod in self.scalers:
                scaled_mat = self.scalers[mod].transform(data[valid_cols])
                for idx, col in enumerate(valid_cols):
                    data[f"Std_{col}"] = scaled_mat[:, idx]
        return data


def compute_adaptive_weights(
    reliability_df: pd.DataFrame,
    active_modalities: List[str] = ["modality_a", "modality_b", "modality_c"],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Computes normalized adaptive weights per row:
        w_{i,m} = R_{i,m} / sum_{k in active} R_{i,k}

    If sum(R_{i,k}) == 0, applies fallback uniform weighting across active modalities and sets Fallback_Flag = 1.
    """
    weights_df = pd.DataFrame(index=reliability_df.index)
    r_cols = [f"R_{mod}" for mod in active_modalities if f"R_{mod}" in reliability_df.columns]

    sum_r = reliability_df[r_cols].sum(axis=1)
    fallback_flag = (sum_r <= 1e-6).astype(int)

    for mod in ["modality_a", "modality_b", "modality_c", "modality_d"]:
        if mod not in active_modalities or f"R_{mod}" not in reliability_df.columns:
            weights_df[f"w_{mod}"] = 0.0
            continue

        r_mod = reliability_df[f"R_{mod}"]
        # Compute normalized weight with fallback handling
        w_val = np.where(
            sum_r > 1e-6,
            r_mod / sum_r,
            1.0 / len(active_modalities)
        )
        weights_df[f"w_{mod}"] = w_val

    weights_df["Fallback_Flag"] = fallback_flag
    return weights_df, fallback_flag


def fuse_modality_blocks(
    df_std: pd.DataFrame,
    weights_df: pd.DataFrame,
    active_modalities: List[str] = ["modality_a", "modality_b", "modality_c"],
    fusion_mode: str = "adaptive_amdfe",
) -> pd.DataFrame:
    """
    Creates fused features X'_{i,m} by applying modality weights to standardized feature blocks.

    Supported fusion modes:
    - 'single': Raw/standardized single modality.
    - 'concat': Concatenation of standardized features without weighting.
    - 'fixed_weight': Equal fixed-weight multiplier (e.g. 1/3 each).
    - 'global_reliability': Fixed global dataset reliability weight.
    - 'adaptive_amdfe': Row-level dynamic weight w_{i,m} * Std_Feature.
    """
    fused_df = df_std.copy()

    mod_feature_map = {
        "modality_a": MODALITY_A_FEATURES,
        "modality_b": MODALITY_B_FEATURES,
        "modality_c": MODALITY_C_FEATURES,
    }

    num_active = len(active_modalities)

    for mod in active_modalities:
        cols = mod_feature_map.get(mod, [])
        valid_std_cols = [f"Std_{c}" for c in cols if f"Std_{c}" in fused_df.columns]

        if fusion_mode == "fixed_weight":
            fixed_w = 1.0 / num_active if num_active > 0 else 1.0
            for std_col in valid_std_cols:
                raw_name = std_col.replace("Std_", "")
                fused_df[f"Fused_{raw_name}"] = fixed_w * fused_df[std_col]

        elif fusion_mode == "global_reliability":
            # Use mean of modality weight across samples as global weight
            glob_w = float(weights_df[f"w_{mod}"].mean()) if f"w_{mod}" in weights_df.columns else 1.0 / num_active
            for std_col in valid_std_cols:
                raw_name = std_col.replace("Std_", "")
                fused_df[f"Fused_{raw_name}"] = glob_w * fused_df[std_col]

        elif fusion_mode == "adaptive_amdfe":
            w_series = weights_df[f"w_{mod}"] if f"w_{mod}" in weights_df.columns else 0.0
            for std_col in valid_std_cols:
                raw_name = std_col.replace("Std_", "")
                fused_df[f"Fused_{raw_name}"] = w_series * fused_df[std_col]

        else:  # 'single' or 'concat'
            for std_col in valid_std_cols:
                raw_name = std_col.replace("Std_", "")
                fused_df[f"Fused_{raw_name}"] = fused_df[std_col]

    return fused_df
