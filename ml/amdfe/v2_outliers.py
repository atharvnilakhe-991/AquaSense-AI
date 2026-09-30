"""
AquaSense AI - AMDFE Version 2 Robust Outlier Detection & Anomaly Indicator Engine
Computes robust outlier indicators on historical prior features using train-fitted statistics (Median + MAD).
Strictly preserves all observations in the dataset to avoid deleting real physical extremes.
Never uses current target value to construct predictor-side outlier indicators.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


class RobustOutlierDetector:
    """
    Fits median and MAD on training prior groundwater features and computes robust anomaly indicators.
    """

    def __init__(self, threshold: float = 3.5):
        self.threshold = threshold
        self.feature_medians: Dict[str, float] = {}
        self.feature_mads: Dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame, feature_cols: list = ["Previous_WatLevel"]) -> "RobustOutlierDetector":
        """
        Fits robust statistics strictly on the training partition.
        """
        for col in feature_cols:
            if col in train_df.columns:
                vals = train_df[col].dropna()
                if len(vals) > 0:
                    med = float(vals.median())
                    mad = float((vals - med).abs().median())
                    mad_normal = mad * 1.4826  # Normal-consistent MAD scale
                    self.feature_medians[col] = med
                    self.feature_mads[col] = max(mad_normal, 1e-6)

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, feature_cols: list = ["Previous_WatLevel"]) -> pd.DataFrame:
        """
        Generates robust z-scores and outlier flags using frozen training median/MAD.
        """
        if not self.is_fitted:
            raise RuntimeError("RobustOutlierDetector must be fitted on train before transform!")

        out_df = pd.DataFrame(index=df.index)

        for col in feature_cols:
            if col in df.columns and col in self.feature_medians:
                med = self.feature_medians[col]
                mad_scale = self.feature_mads[col]
                raw_vals = df[col]

                # Compute robust z-score (using only prior feature)
                rob_z = (raw_vals - med) / mad_scale
                out_flag = (rob_z.abs() > self.threshold).astype(int)

                out_df[f"Robust_Z_{col}"] = rob_z.fillna(0.0)
                out_df[f"Outlier_Flag_{col}"] = out_flag.fillna(0)

        out_df["Outlier_Method"] = "train_fitted_median_mad"
        return out_df
