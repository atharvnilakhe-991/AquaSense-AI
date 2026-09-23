"""
Adaptive Multimodal Data Fusion Engine (AMDFE)
Member 2 - AMDFE

Coordinates data quality filtering, causal missing data imputation,
and dynamic reliability-weighted multimodal fusion.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from .quality import assess_data_quality, calculate_temporal_irregularity
from .reliability import attach_reliability_metrics


class AdaptiveMultimodalDataFusionEngine:
    """
    AMDFE: Fuses heterogeneous environmental data (NASA POWER, elevation) with sparse
    and irregular groundwater time series without target leakage.
    """

    def __init__(self, causal_window_years: int = 5):
        self.causal_window_years = causal_window_years
        self.fitted_ = False
        self.regional_weather_priors_: Dict[str, float] = {}

    def fit_priors(self, df: pd.DataFrame) -> "AdaptiveMultimodalDataFusionEngine":
        """
        Computes regional baseline priors from historical weather data to enable
        unseen-well imputation without accessing target groundwater levels.
        """
        weather_cols = [
            "annual_precipitation_mm",
            "annual_temperature_c",
            "annual_relative_humidity_pct",
            "annual_wind_speed_ms",
            "annual_solar_radiation_mj",
        ]
        for col in weather_cols:
            if col in df.columns:
                self.regional_weather_priors_[col] = float(df[col].median())
        self.fitted_ = True
        return self

    def process_and_fuse(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        End-to-end AMDFE processing:
        1. Run physical plausibility quality checks.
        2. Calculate observation frequency and temporal gaps strictly causally.
        3. Impute any missing environmental predictors using spatial-temporal causal smoothing.
        4. Calculate adaptive reliability weights per modality.
        5. Form fused multimodal feature vectors.
        """
        # Step 1: Quality assessment
        clean_df, quality_report = assess_data_quality(df)
        
        # Step 2: Causal temporal irregularity metrics
        irregular_df = calculate_temporal_irregularity(clean_df, well_col="well_id", time_col="year")
        
        # Step 3: Leakage-safe missing data recovery for weather
        fused_df = irregular_df.copy()
        for col, prior_val in self.regional_weather_priors_.items():
            if col in fused_df.columns:
                fused_df[col] = fused_df[col].fillna(prior_val)
        
        # Step 4: Attach AMDFE adaptive reliability weights
        fused_df = attach_reliability_metrics(fused_df)
        
        # Step 5: Engineer interaction features between reliability and modalities
        # (Allows tree-based models to attend to weather more when well reliability is low)
        if "annual_precipitation_mm" in fused_df.columns and "composite_reliability" in fused_df.columns:
            fused_df["precip_x_reliability"] = (
                fused_df["annual_precipitation_mm"] * fused_df["composite_reliability"]
            )
            fused_df["temp_x_gap"] = (
                fused_df.get("annual_temperature_c", 11.0) * fused_df["temporal_gap_years"]
            )

        return fused_df
