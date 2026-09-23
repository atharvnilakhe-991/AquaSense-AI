"""
Unit Tests for Member 2: AMDFE
"""

import numpy as np
import pandas as pd
import pytest

from amdf.quality import assess_data_quality, calculate_temporal_irregularity
from amdf.reliability import estimate_source_reliability, compute_modality_reliability_vector
from amdf.fusion import AdaptiveMultimodalDataFusionEngine


def test_assess_data_quality():
    sample_data = pd.DataFrame({
        "well_id": ["W1", "W2", "W3"],
        "observed_level": [120.5, -5.0, 160.0],  # -5.0 is invalid
        "latitude": [40.5, 40.4, 55.0],  # 55.0 is outside Phelps Co.
        "longitude": [-99.4, -99.3, -99.5],
        "annual_precipitation_mm": [620.0, 700.0, -10.0]  # -10 is invalid
    })
    
    cleaned, summary = assess_data_quality(sample_data)
    assert summary["total_records"] == 3
    assert summary["valid_records"] == 1
    assert summary["flagged_anomalies"] == 2
    assert cleaned.loc[0, "passed_all_quality_checks"] == True
    assert cleaned.loc[1, "passed_all_quality_checks"] == False


def test_estimate_source_reliability_bounds():
    # Fresh frequent well
    rel_high = estimate_source_reliability(temporal_gap_years=1.0, historical_count=15)
    # Stale sparse well
    rel_low = estimate_source_reliability(temporal_gap_years=6.0, historical_count=1)
    
    assert 0.0 <= rel_high <= 1.0
    assert 0.0 <= rel_low <= 1.0
    assert rel_high > rel_low


def test_temporal_irregularity_causality():
    # Verify that lags only use prior years (causal ordering)
    df = pd.DataFrame({
        "well_id": ["W1", "W1", "W1"],
        "year": [2010, 2012, 2015],
        "observed_level": [110.0, 112.0, 115.0]
    })
    
    res = calculate_temporal_irregularity(df)
    assert list(res["temporal_gap_years"]) == [1.0, 2.0, 3.0]
    assert list(res["historical_obs_count"]) == [0, 1, 2]


def test_amdfe_fusion_engine():
    df = pd.DataFrame({
        "well_id": ["W1", "W1"],
        "year": [2020, 2021],
        "latitude": [40.45, 40.45],
        "longitude": [-99.40, -99.40],
        "surface_elevation_m": [715.0, 715.0],
        "observed_level": [120.0, 121.5],
        "annual_precipitation_mm": [600.0, np.nan],  # test imputation
        "annual_temperature_c": [11.0, 11.2],
        "annual_relative_humidity_pct": [68.0, 69.0],
        "annual_wind_speed_ms": [4.5, 4.6],
        "annual_solar_radiation_mj": [16.0, 16.2]
    })
    
    engine = AdaptiveMultimodalDataFusionEngine()
    engine.fit_priors(df)
    fused = engine.process_and_fuse(df)
    
    assert not fused["annual_precipitation_mm"].isna().any()
    assert "composite_reliability" in fused.columns
    assert "precip_x_reliability" in fused.columns
