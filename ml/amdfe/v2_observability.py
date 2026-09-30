"""
AquaSense AI - AMDFE Version 2 Context Observability Engine
Computes sample-level point-in-time observation state A_{i,m} for every sample i.
Uses strictly prior temporal information. Decouples local observation availability
from global intrinsic source reliability.
"""

from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd

from ml.amdfe.config import (
    AMDFE_V2_OBSERVABILITY_DIR,
    MODALITY_A_FEATURES,
    MODALITY_B_FEATURES,
    MODALITY_C_FEATURES,
    WELL_ID_COLUMN,
    DATE_COLUMN,
    YEAR_COLUMN,
)


class ContextObservabilityEngine:
    """
    Computes sample-level observability factors A_{i,m} and fits cohort thresholds on TRAIN only.
    """

    def __init__(
        self,
        gap_decay_scale: float = 365.25,
        count_saturation: float = 3.0,
        w_base: float = 0.20,
        w_gap: float = 0.50,
        w_count: float = 0.30,
        cold_base: float = 0.10,
    ):
        self.gap_decay_scale = gap_decay_scale
        self.count_saturation = count_saturation
        self.w_base = w_base
        self.w_gap = w_gap
        self.w_count = w_count
        self.cold_base = cold_base
        self.density_tertiles: Tuple[float, float] = (1.0, 5.0)
        self.gap_tertiles: Tuple[float, float] = (180.0, 365.0)
        self.is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame) -> "ContextObservabilityEngine":
        """
        Learns empirical quantile boundaries for density and gap cohorts on TRAIN ONLY.
        """
        if "Previous_Observation_Count" in train_df.columns:
            counts = train_df["Previous_Observation_Count"].dropna()
            if len(counts) > 10:
                q33 = float(counts.quantile(0.33))
                q66 = float(counts.quantile(0.66))
                self.density_tertiles = (q33, q66)

        if "Days_Since_Previous" in train_df.columns:
            gaps = train_df["Days_Since_Previous"].dropna()
            if len(gaps) > 10:
                g33 = float(gaps.quantile(0.33))
                g66 = float(gaps.quantile(0.66))
                self.gap_tertiles = (g33, g66)

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes A_{i,m} and cohort classifications for any partition using frozen train thresholds.
        """
        obs_df = pd.DataFrame(index=df.index)

        # 1. Modality A Observability (Groundwater State)
        has_prior = (df["Previous_Observation_Count"] > 0).astype(float) if "Previous_Observation_Count" in df.columns else (df["Previous_WatLevel"].notna()).astype(float)
        obs_count = df["Previous_Observation_Count"].fillna(0).values if "Previous_Observation_Count" in df.columns else np.zeros(len(df))
        days_since = df["Days_Since_Previous"].fillna(3650.0).values if "Days_Since_Previous" in df.columns else np.full(len(df), 3650.0)

        # Count saturation factor
        sat = max(0.1, self.count_saturation)
        f_count = np.clip(obs_count / sat, 0.0, 1.0)
        # Gap decay factor
        scale = max(1.0, self.gap_decay_scale)
        f_gap = np.exp(-np.clip(days_since, 0.0, 3650.0) / scale)

        # Cold start vs Warm start
        a_gw = np.where(
            has_prior > 0,
            self.w_base + self.w_gap * f_gap + self.w_count * f_count,
            self.cold_base
        )
        obs_df["A_modality_a"] = np.clip(a_gw, 0.05, 1.0)

        # 2. Modality B Observability (Weather)
        if "Weather_Available_At_Prediction" in df.columns:
            obs_df["A_modality_b"] = df["Weather_Available_At_Prediction"].astype(float)
        else:
            weather_cols = [c for c in MODALITY_B_FEATURES if c in df.columns]
            obs_df["A_modality_b"] = (df[weather_cols].notna().all(axis=1)).astype(float) if weather_cols else 0.0

        # 3. Modality C Observability (Spatial)
        spatial_cols = [c for c in MODALITY_C_FEATURES if c in df.columns]
        obs_df["A_modality_c"] = (df[spatial_cols].notna().all(axis=1)).astype(float) if spatial_cols else 1.0

        # 4. Modality D Observability (Unavailable)
        obs_df["A_modality_d"] = 0.0

        # 5. Add Explicit Cohort Identifiers for Sub-population Analyses
        # Start Type: Cold-Start vs Warm-Start
        obs_df["Start_Type"] = np.where(obs_count == 0, "Cold-Start", "Warm-Start")

        # Density Cohort: Low / Medium / High
        d_low, d_high = self.density_tertiles
        obs_df["Density_Cohort"] = np.where(
            obs_count <= d_low, "Low_Density",
            np.where(obs_count <= d_high, "Medium_Density", "High_Density")
        )

        # Gap Cohort: Short / Moderate / Long
        g_low, g_high = self.gap_tertiles
        obs_df["Gap_Cohort"] = np.where(
            obs_count == 0, "No_Prior_Record",
            np.where(days_since <= g_low, "Short_Gap",
            np.where(days_since <= g_high, "Moderate_Gap", "Long_Gap"))
        )

        return obs_df


def compute_context_observability(
    df: pd.DataFrame,
    engine: Optional[ContextObservabilityEngine] = None,
) -> pd.DataFrame:
    """
    Convenience function returning observability dataframe.
    """
    if engine is None:
        engine = ContextObservabilityEngine().fit(df)
    return engine.transform(df)
