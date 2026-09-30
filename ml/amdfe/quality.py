"""
AquaSense AI - AMDFE Data Quality Assessment
Computes objective data quality dimensions (Completeness, Temporal Coverage,
Spatial Consistency, Observation Density, Integrity) for each modality block.
Distinguishes source-level quality from sample-level availability.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from ml.amdfe.config import (
    AMDFE_QUALITY_DIR,
    MODALITY_A_FEATURES,
    MODALITY_B_FEATURES,
    MODALITY_C_FEATURES,
    WELL_ID_COLUMN,
    DATE_COLUMN,
    YEAR_COLUMN,
)


def compute_modality_completeness(df: pd.DataFrame, feature_cols: list) -> Tuple[float, pd.Series]:
    """
    Computes overall feature completeness (1 - missing_ratio) and per-sample completeness.
    """
    if not feature_cols:
        return 0.0, pd.Series(0.0, index=df.index)
    
    missing_ratios = df[feature_cols].isnull().mean(axis=1)
    sample_completeness = 1.0 - missing_ratios
    global_completeness = float(sample_completeness.mean())
    return global_completeness, sample_completeness


def assess_groundwater_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Quality assessment for Modality A (Groundwater State / History).
    """
    total_samples = len(df)
    unique_wells = df[WELL_ID_COLUMN].nunique()
    
    # 1. Missingness across history features
    feature_nulls = {col: int(df[col].isnull().sum()) for col in MODALITY_A_FEATURES if col in df.columns}
    mean_null_ratio = float(np.mean([nulls / total_samples for nulls in feature_nulls.values()]))
    completeness = 1.0 - mean_null_ratio

    # 2. Temporal coverage & continuity
    has_prior = (df["Previous_Observation_Count"] > 0).astype(int)
    prior_ratio = float(has_prior.mean())  # warm-start ratio
    long_gap_ratio = float(df["Long_Gap_Flag"].mean()) if "Long_Gap_Flag" in df.columns else 0.0
    temporal_consistency = float(np.clip(prior_ratio * (1.0 - 0.5 * long_gap_ratio), 0.05, 1.0))

    # 3. Observation density
    avg_obs_density = float(df["Observation_Density"].mean()) if "Observation_Density" in df.columns else 1.0
    normalized_density = float(np.clip(avg_obs_density / 3.0, 0.1, 1.0))

    # 4. Outlier detection (IQR method on key lag features)
    if "Previous_WatLevel" in df.columns and df["Previous_WatLevel"].notna().sum() > 10:
        q25 = df["Previous_WatLevel"].quantile(0.25)
        q75 = df["Previous_WatLevel"].quantile(0.75)
        iqr = q75 - q25
        outliers = ((df["Previous_WatLevel"] < (q25 - 1.5 * iqr)) | 
                    (df["Previous_WatLevel"] > (q75 + 1.5 * iqr))).sum()
        outlier_ratio = float(outliers / total_samples)
    else:
        outlier_ratio = 0.0

    return {
        "Modality": "Modality A (Groundwater State)",
        "Completeness_C": completeness,
        "Temporal_Consistency_T": temporal_consistency,
        "Spatial_Coverage_S": 1.0,  # Defined at well locations
        "Observation_Density_O": normalized_density,
        "Measurement_Quality_M": "unavailable",
        "Outlier_Ratio": outlier_ratio,
        "Total_Observations": total_samples,
        "Unique_Wells": unique_wells,
        "Warm_Start_Fraction": prior_ratio,
        "Long_Gap_Fraction": long_gap_ratio,
    }


def assess_weather_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Quality assessment for Modality B (Meteorological Covariates).
    """
    total_samples = len(df)
    
    # 1. Completeness of lagged weather features
    weather_nulls = {col: int(df[col].isnull().sum()) for col in MODALITY_B_FEATURES if col in df.columns}
    mean_null_ratio = float(np.mean([nulls / total_samples for nulls in weather_nulls.values()]))
    completeness = 1.0 - mean_null_ratio

    # 2. Temporal coverage (fraction of observation years where Y-1 was available)
    avail_ratio = float(df["Weather_Available_At_Prediction"].mean()) if "Weather_Available_At_Prediction" in df.columns else completeness
    temporal_consistency = avail_ratio

    # 3. Weather dataset consistency across 2000-2024
    spatial_coverage = 1.0  # Regional aggregate covers study basin

    return {
        "Modality": "Modality B (Weather)",
        "Completeness_C": completeness,
        "Temporal_Consistency_T": temporal_consistency,
        "Spatial_Coverage_S": spatial_coverage,
        "Observation_Density_O": 1.0,  # Annual continuous series
        "Measurement_Quality_M": "unavailable",
        "Outlier_Ratio": 0.0,
        "Total_Observations": total_samples,
        "Available_Fraction": avail_ratio,
        "Missing_Prior_Year_Observations": int((1.0 - avail_ratio) * total_samples),
    }


def assess_spatial_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Quality assessment for Modality C (Spatial & Topography).
    """
    total_samples = len(df)
    
    # Check coordinates and surface elevation
    coords_valid = ((df["LatDD"].notna()) & (df["LongDD"].notna()) & (df["Surf_Elev"].notna())).mean()
    completeness = float(coords_valid)
    
    # Coordinates in valid bounding range
    lat_in_bounds = ((df["LatDD"] >= 20.0) & (df["LatDD"] <= 40.0)).mean()
    long_in_bounds = ((df["LongDD"] >= -125.0) & (df["LongDD"] <= -70.0)).mean()
    spatial_consistency = float((lat_in_bounds + long_in_bounds) / 2.0)

    return {
        "Modality": "Modality C (Spatial/Topography)",
        "Completeness_C": completeness,
        "Temporal_Consistency_T": 1.0,  # Static coordinates are temporally invariant
        "Spatial_Coverage_S": spatial_consistency,
        "Observation_Density_O": 1.0,
        "Measurement_Quality_M": "unavailable",
        "Outlier_Ratio": 0.0,
        "Total_Observations": total_samples,
        "Valid_Coordinate_Fraction": completeness,
    }


def run_data_quality_assessment(df: pd.DataFrame, save_reports: bool = True) -> pd.DataFrame:
    """
    Runs comprehensive quality assessment across all modalities and writes paper-ready reports.
    """
    q_gw = assess_groundwater_quality(df)
    q_weather = assess_weather_quality(df)
    q_spatial = assess_spatial_quality(df)

    summary_records = [
        {
            "Modality": q_gw["Modality"],
            "Completeness_C": q_gw["Completeness_C"],
            "Temporal_Consistency_T": q_gw["Temporal_Consistency_T"],
            "Spatial_Coverage_S": q_gw["Spatial_Coverage_S"],
            "Observation_Density_O": q_gw["Observation_Density_O"],
            "Measurement_Quality_M": q_gw["Measurement_Quality_M"],
            "Outlier_Ratio": q_gw["Outlier_Ratio"],
        },
        {
            "Modality": q_weather["Modality"],
            "Completeness_C": q_weather["Completeness_C"],
            "Temporal_Consistency_T": q_weather["Temporal_Consistency_T"],
            "Spatial_Coverage_S": q_weather["Spatial_Coverage_S"],
            "Observation_Density_O": q_weather["Observation_Density_O"],
            "Measurement_Quality_M": q_weather["Measurement_Quality_M"],
            "Outlier_Ratio": q_weather["Outlier_Ratio"],
        },
        {
            "Modality": q_spatial["Modality"],
            "Completeness_C": q_spatial["Completeness_C"],
            "Temporal_Consistency_T": q_spatial["Temporal_Consistency_T"],
            "Spatial_Coverage_S": q_spatial["Spatial_Coverage_S"],
            "Observation_Density_O": q_spatial["Observation_Density_O"],
            "Measurement_Quality_M": q_spatial["Measurement_Quality_M"],
            "Outlier_Ratio": q_spatial["Outlier_Ratio"],
        },
        {
            "Modality": "Modality D (Earth Observation / SoilGrids)",
            "Completeness_C": 0.0,
            "Temporal_Consistency_T": 0.0,
            "Spatial_Coverage_S": 0.0,
            "Observation_Density_O": 0.0,
            "Measurement_Quality_M": "unavailable",
            "Outlier_Ratio": 0.0,
        },
    ]

    summary_df = pd.DataFrame(summary_records)

    if save_reports:
        AMDFE_QUALITY_DIR.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(AMDFE_QUALITY_DIR / "modality_quality_summary.csv", index=False)
        pd.DataFrame([q_gw]).to_csv(AMDFE_QUALITY_DIR / "groundwater_quality.csv", index=False)
        pd.DataFrame([q_weather]).to_csv(AMDFE_QUALITY_DIR / "weather_quality.csv", index=False)
        pd.DataFrame([q_spatial]).to_csv(AMDFE_QUALITY_DIR / "spatial_quality.csv", index=False)

        # Write quality methodology documentation
        methodology_text = (
            "# AMDFE Data Quality Assessment Methodology\n\n"
            "## Quality Dimensions\n"
            "- **Completeness ($C$)**: Ratio of non-missing values across feature blocks ($C \\in [0, 1]$).\n"
            "- **Temporal Consistency ($T$)**: Evaluates observation continuity and temporal precedence adherence.\n"
            "- **Spatial Coverage ($S$)**: Valid spatial coordinate bounds and domain representation.\n"
            "- **Observation Density ($O$)**: Normalized observation frequency per well.\n"
            "- **Measurement Quality ($M$)**: Sensor instrumentation precision metadata. Recorded as `unavailable` because no explicit sensor error bars exist in the raw dataset.\n\n"
            "## Source vs. Sample-Level Differentiation\n"
            "Global modality quality measures source-level trustworthiness, while sample-level availability indicators $A_{i,m}$ quantify per-row data completeness for adaptive fusion.\n"
        )
        with open(AMDFE_QUALITY_DIR / "quality_methodology.md", "w", encoding="utf-8") as f:
            f.write(methodology_text)

        print(f"[AMDFE Quality] Saved data quality reports to: {AMDFE_QUALITY_DIR}")

    return summary_df
