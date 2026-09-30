"""
AquaSense AI - AMDFE Ingestion & Temporal Leakage-Safe Mapping
Loads multimodal source datasets, purges target-derived rasters, and applies
strict causal/temporal precedence mapping for meteorological covariates.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from ml.amdfe.config import (
    INTEGRATED_DATASET_PATH,
    ANNUAL_WEATHER_PATH,
    IRREGULAR_FEATURES_PATH,
    MODALITY_A_FEATURES,
    MODALITY_B_FEATURES,
    MODALITY_C_FEATURES,
    TARGET_COLUMN,
    WELL_ID_COLUMN,
    DATE_COLUMN,
    YEAR_COLUMN,
    LEAKAGE_COLUMNS,
    AMDFE_VALIDATION_DIR,
)


def load_raw_multimodal_data(
    features_path: Optional[Path] = None,
    weather_path: Optional[Path] = None,
    enforce_strict_leakage_check: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads groundwater observations with irregular features and standalone annual weather tables.
    Guarantees no target-derived rasters enter the processing stream.
    """
    f_path = features_path or IRREGULAR_FEATURES_PATH
    w_path = weather_path or ANNUAL_WEATHER_PATH

    if not Path(f_path).exists():
        raise FileNotFoundError(f"Source irregular features dataset not found at: {f_path}")
    if not Path(w_path).exists():
        raise FileNotFoundError(f"Source annual weather dataset not found at: {w_path}")

    df_gw = pd.read_csv(f_path)
    df_weather = pd.read_csv(w_path)

    if enforce_strict_leakage_check:
        for col in LEAKAGE_COLUMNS:
            if col in df_gw.columns:
                df_gw = df_gw.drop(columns=[col])

    return df_gw, df_weather


def apply_leakage_safe_weather_policy(
    df_gw: pd.DataFrame,
    df_weather: pd.DataFrame,
    save_audit: bool = True,
) -> pd.DataFrame:
    """
    Maps each groundwater observation to the latest COMPLETED annual weather period.

    Policy:
        For observation at date D in calendar year Y:
        Completed weather year = Y - 1.
        
        If completed weather year exists in df_weather, its values are assigned.
        If completed weather year is unavailable (e.g. Y=2000 requiring 1999 weather),
        weather values remain NaN, with Weather_Available_At_Prediction = 0.

    Audit columns:
        - Weather_Year_Used: The calendar year of weather features attached.
        - Weather_Available_At_Prediction: Binary flag (1 if available, 0 if missing).
        - Weather_Leakage_Check: Binary flag (1 if Year_Used < YearMsr, confirming zero lookahead).
    """
    df = df_gw.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    
    # Identify observation year
    if YEAR_COLUMN not in df.columns:
        df[YEAR_COLUMN] = df[DATE_COLUMN].dt.year
    else:
        df[YEAR_COLUMN] = df[YEAR_COLUMN].astype(int)

    # Determine safe prior completed weather year (Y - 1)
    df["Weather_Year_Used"] = df[YEAR_COLUMN] - 1

    # Drop any pre-existing same-year weather columns from df to prevent accidental retention
    for col in MODALITY_B_FEATURES:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Merge lagged annual weather
    weather_safe = df_weather.copy()
    weather_safe = weather_safe.rename(columns={YEAR_COLUMN: "Weather_Year_Used"})
    
    merged = pd.merge(
        df,
        weather_safe[["Weather_Year_Used"] + MODALITY_B_FEATURES],
        on="Weather_Year_Used",
        how="left",
    )

    # Compute availability and leakage audit flags
    merged["Weather_Available_At_Prediction"] = (
        merged[MODALITY_B_FEATURES[0]].notna().astype(int)
    )
    merged["Weather_Leakage_Check"] = (
        (merged["Weather_Year_Used"] < merged[YEAR_COLUMN]) & 
        (merged["DateMsr"] > pd.to_datetime(merged["Weather_Year_Used"].astype(str) + "-12-31"))
    ).astype(int)

    if save_audit:
        AMDFE_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        audit_df = merged[[
            WELL_ID_COLUMN,
            DATE_COLUMN,
            YEAR_COLUMN,
            "Weather_Year_Used",
            "Weather_Available_At_Prediction",
            "Weather_Leakage_Check",
        ] + MODALITY_B_FEATURES].copy()
        
        audit_path = AMDFE_VALIDATION_DIR / "weather_temporal_leakage_audit.csv"
        audit_df.to_csv(audit_path, index=False)
        print(f"[AMDFE Ingestion] Saved weather temporal leakage audit to: {audit_path}")

    return merged


def ingest_amdfe_dataset(
    enforce_strict_leakage_check: bool = True,
    save_audit: bool = True,
) -> pd.DataFrame:
    """
    Main ingestion interface returning the clean, leakage-safe multimodal dataset.
    """
    df_gw, df_weather = load_raw_multimodal_data(
        enforce_strict_leakage_check=enforce_strict_leakage_check
    )
    dataset = apply_leakage_safe_weather_policy(
        df_gw=df_gw,
        df_weather=df_weather,
        save_audit=save_audit,
    )
    return dataset
