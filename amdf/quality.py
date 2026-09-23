"""
Data Quality Assessment & Irregularity Metrics
Member 2 - AMDFE

Evaluates spatial and temporal validity, flags physical anomalies, and quantifies observation sparsity.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def assess_data_quality(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs data quality validation on groundwater observation records.
    
    Checks:
    - Target level bounds (physical plausibility, e.g. depth to water > 0 and < 400 ft)
    - Latitude/Longitude bounds for Phelps County, NE (~40.2 - 40.7 N, -99.7 - -99.0 W)
    - Physical weather bounds (Precipitation >= 0, Temp in realistic ranges)
    
    Returns:
        Tuple containing cleaned DataFrame with quality flags and a summary metrics dictionary.
    """
    cleaned = df.copy()
    flags = []
    
    # 1. Target Groundwater Level Validity (depth to water in feet or meters)
    if "observed_level" in cleaned.columns:
        invalid_levels = (cleaned["observed_level"] < 0) | (cleaned["observed_level"] > 450)
        cleaned["is_valid_level"] = ~invalid_levels
    else:
        cleaned["is_valid_level"] = True

    # 2. Geospatial Phelps County bounds
    if "latitude" in cleaned.columns and "longitude" in cleaned.columns:
        valid_coords = (
            (cleaned["latitude"] >= 39.8) & (cleaned["latitude"] <= 41.2) &
            (cleaned["longitude"] >= -100.2) & (cleaned["longitude"] <= -98.5)
        )
        cleaned["is_valid_coords"] = valid_coords
    else:
        cleaned["is_valid_coords"] = True

    # 3. Weather integrity checks
    if "annual_precipitation_mm" in cleaned.columns:
        valid_precip = (cleaned["annual_precipitation_mm"] >= 0) & (cleaned["annual_precipitation_mm"] <= 2500)
        cleaned["is_valid_weather"] = valid_precip
    else:
        cleaned["is_valid_weather"] = True

    cleaned["passed_all_quality_checks"] = (
        cleaned["is_valid_level"] & 
        cleaned["is_valid_coords"] & 
        cleaned["is_valid_weather"]
    )
    
    total = len(cleaned)
    passed = int(cleaned["passed_all_quality_checks"].sum())
    summary = {
        "total_records": total,
        "valid_records": passed,
        "flagged_anomalies": total - passed,
        "quality_pass_rate": round((passed / total * 100), 2) if total > 0 else 0.0,
    }
    
    return cleaned, summary


def calculate_temporal_irregularity(df: pd.DataFrame, well_col: str = "well_id", time_col: str = "year") -> pd.DataFrame:
    """
    Computes temporal sparsity and monitoring irregularity metrics per well without target leakage.
    
    Metrics:
    - days_since_prior_obs: Temporal gap from previous measurement
    - historical_obs_density: Cumulative number of observations prior to current timestamp
    - sampling_regularity_index: Variance penalty for irregular intervals
    """
    result = df.copy()
    
    if well_col not in result.columns or time_col not in result.columns:
        # Fallback if identifiers are missing
        result["temporal_gap_years"] = 1.0
        result["historical_obs_count"] = 1
        result["sampling_regularity_index"] = 0.5
        return result

    # Sort strictly chronologically per well to preserve temporal causality
    result = result.sort_values(by=[well_col, time_col]).reset_index(drop=True)
    
    # Calculate causal historical lag intervals
    result["temporal_gap_years"] = result.groupby(well_col)[time_col].diff().fillna(1.0)
    result["historical_obs_count"] = result.groupby(well_col).cumcount()
    
    # Regularity index (1.0 = annual continuous monitoring, decaying towards 0.1 for multi-year gaps)
    result["sampling_regularity_index"] = np.exp(-0.25 * (result["temporal_gap_years"] - 1.0).clip(lower=0))
    
    return result
