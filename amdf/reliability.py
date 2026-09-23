"""
Source & Modality Reliability Estimation
Member 2 - AMDFE

Calculates observation-aware confidence weights by combining temporal frequency,
sensor consistency, and weather modality completeness.
"""

from typing import Dict, List
import numpy as np
import pandas as pd


def estimate_source_reliability(
    temporal_gap_years: float,
    historical_count: int,
    weather_completeness: float = 1.0,
    elevation_confidence: float = 0.95
) -> float:
    """
    Computes a composite reliability score in [0.0, 1.0] for an individual observation.
    
    Factors:
    - recency_score: Exponential decay based on years since last reading (1.0 = current, 0.2 = >5yr gap)
    - frequency_score: Logarithmic ramp based on historical count (saturates around 10+ readings)
    - environmental_score: Completeness of NASA POWER weather and DEM elevation
    """
    # 1. Recency / Temporal continuity weight (decay rate lambda = 0.18)
    recency_score = float(np.exp(-0.18 * max(0.0, temporal_gap_years - 1.0)))
    
    # 2. Historical baseline richness
    frequency_score = float(min(1.0, 0.3 + 0.7 * (np.log1p(historical_count) / np.log1p(15))))
    
    # 3. Environmental data reliability
    environmental_score = float(0.6 * weather_completeness + 0.4 * elevation_confidence)
    
    # Composite weighted average
    composite_reliability = (
        0.45 * recency_score +
        0.35 * frequency_score +
        0.20 * environmental_score
    )
    
    return float(np.clip(composite_reliability, 0.05, 0.99))


def compute_modality_reliability_vector(row: pd.Series) -> Dict[str, float]:
    """
    Determines modality contribution weights (Groundwater History vs Meteorology vs Topography)
    based on the well's local observation sparsity.
    
    When groundwater history is sparse or stale:
      -> Groundwater modality weight decreases.
      -> Meteorology & topographic modality weights dynamically increase to compensate.
    """
    temporal_gap = row.get("temporal_gap_years", 1.0)
    history_count = row.get("historical_obs_count", 0)
    
    well_reliability = estimate_source_reliability(temporal_gap, history_count)
    
    # Adaptive weight allocation:
    # If well is highly reliable (frequent readings): Well History gets up to 55%
    # If well is sparse (high gap): Weather and Topography take over up to 75%
    gw_weight = 0.20 + 0.35 * well_reliability
    weather_weight = 0.50 - 0.20 * well_reliability
    topo_weight = 1.0 - (gw_weight + weather_weight)
    
    return {
        "groundwater_history_weight": round(gw_weight, 4),
        "meteorological_weight": round(weather_weight, 4),
        "topographic_weight": round(topo_weight, 4),
        "composite_reliability": round(well_reliability, 4),
    }


def attach_reliability_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies reliability scoring row-by-row across the dataset in a vectorized/leakage-safe manner.
    """
    out = df.copy()
    if "temporal_gap_years" not in out.columns:
        out["temporal_gap_years"] = 1.0
    if "historical_obs_count" not in out.columns:
        out["historical_obs_count"] = 1
        
    weights = out.apply(compute_modality_reliability_vector, axis=1)
    weight_df = pd.DataFrame(list(weights))
    
    for col in weight_df.columns:
        out[col] = weight_df[col]
        
    return out
