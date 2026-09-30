"""
AquaSense AI - AMDFE Reliability Engine
Calculates global source reliability R_{global,m}, sample-level availability A_{i,m},
and dynamic context-aware reliability scores R_{i,m}.
Performs transparent mathematical sensitivity analysis across aggregation formulations.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.stats import gmean, hmean

from ml.amdfe.config import (
    AMDFE_RELIABILITY_DIR,
    MODALITY_A_FEATURES,
    MODALITY_B_FEATURES,
    MODALITY_C_FEATURES,
)
from ml.amdfe.quality import (
    assess_groundwater_quality,
    assess_weather_quality,
    assess_spatial_quality,
)


def compute_global_reliability_scores(
    df: pd.DataFrame,
    aggregation: str = "geometric",
) -> Dict[str, float]:
    """
    Computes global source reliability R_{global,m} from available quality dimensions
    (C = Completeness, T = Temporal Consistency, S = Spatial Coverage, O = Observation Density).

    Measurement quality M is explicitly excluded from calculation because it is unavailable.

    Parameters:
        df: Training partition DataFrame.
        aggregation: Aggregation method ('geometric', 'arithmetic', 'harmonic').

    Returns:
        Dictionary mapping modality names to global reliability scores in (0, 1].
    """
    q_gw = assess_groundwater_quality(df)
    q_weather = assess_weather_quality(df)
    q_spatial = assess_spatial_quality(df)

    dims = {
        "modality_a": [q_gw["Completeness_C"], q_gw["Temporal_Consistency_T"], q_gw["Spatial_Coverage_S"], q_gw["Observation_Density_O"]],
        "modality_b": [q_weather["Completeness_C"], q_weather["Temporal_Consistency_T"], q_weather["Spatial_Coverage_S"], q_weather["Observation_Density_O"]],
        "modality_c": [q_spatial["Completeness_C"], q_spatial["Temporal_Consistency_T"], q_spatial["Spatial_Coverage_S"], q_spatial["Observation_Density_O"]],
        "modality_d": [0.0, 0.0, 0.0, 0.0],
    }

    scores = {}
    for mod, vals in dims.items():
        if mod == "modality_d" or sum(vals) == 0:
            scores[mod] = 0.0
            continue

        clean_vals = np.clip(np.array(vals), 1e-4, 1.0)
        if aggregation == "geometric":
            score = float(gmean(clean_vals))
        elif aggregation == "arithmetic":
            score = float(np.mean(clean_vals))
        elif aggregation == "harmonic":
            score = float(hmean(clean_vals))
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        scores[mod] = float(np.clip(score, 0.0, 1.0))

    return scores


def evaluate_reliability_aggregation_sensitivity(
    df: pd.DataFrame,
    save_results: bool = True,
) -> pd.DataFrame:
    """
    Performs sensitivity analysis comparing geometric, arithmetic, and harmonic formulations.
    """
    aggregations = ["arithmetic", "geometric", "harmonic"]
    rows = []

    for agg in aggregations:
        scores = compute_global_reliability_scores(df, aggregation=agg)
        total = sum(scores.values())
        weights = {f"w_{k}": scores[k] / total if total > 0 else 0.0 for k in scores}

        row = {
            "Aggregation_Method": agg,
            "R_Modality_A": scores["modality_a"],
            "R_Modality_B": scores["modality_b"],
            "R_Modality_C": scores["modality_c"],
            "R_Modality_D": scores["modality_d"],
            "Weight_Modality_A": weights["w_modality_a"],
            "Weight_Modality_B": weights["w_modality_b"],
            "Weight_Modality_C": weights["w_modality_c"],
            "Weight_Modality_D": weights["w_modality_d"],
        }
        rows.append(row)

    sensitivity_df = pd.DataFrame(rows)

    if save_results:
        AMDFE_RELIABILITY_DIR.mkdir(parents=True, exist_ok=True)
        out_path = AMDFE_RELIABILITY_DIR / "reliability_aggregation_sensitivity.csv"
        sensitivity_df.to_csv(out_path, index=False)
        print(f"[AMDFE Research] Saved reliability aggregation sensitivity to: {out_path}")

    return sensitivity_df


def compute_sample_availability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes sample-level availability factors A_{i,m} for every observation row i.

    Rules:
    - Modality A (Groundwater):
        Evaluated from presence of prior observations, gap duration, and historical stats.
        Warm-start with small gap: A_{i,A} ≈ 1.0.
        Cold-start (first observation of well): A_{i,A} = 0.25 (baseline prior state info).
        Long gaps (> 365 days): degraded by gap duration factor exp(-Days/730).
    - Modality B (Weather):
        A_{i,B} = 1.0 if completed prior-year weather exists and is complete; 0.0 otherwise.
    - Modality C (Spatial):
        A_{i,C} = 1.0 if Lat, Long, Surf_Elev are non-null; 0.0 otherwise.
    - Modality D (Earth Observation):
        A_{i,D} = 0.0 (unavailable).
    """
    n_samples = len(df)
    avail_df = pd.DataFrame(index=df.index)

    # 1. Modality A Availability
    has_prior = (df["Previous_Observation_Count"] > 0).astype(float) if "Previous_Observation_Count" in df.columns else (df["Previous_WatLevel"].notna()).astype(float)
    
    # Gap decay factor for temporal continuity
    days_since = df["Days_Since_Previous"].fillna(365.0) if "Days_Since_Previous" in df.columns else pd.Series(365.0, index=df.index)
    gap_factor = np.exp(-np.clip(days_since, 0.0, 3650.0) / 730.0)
    
    # Prior count saturation
    obs_count = df["Previous_Observation_Count"].fillna(0) if "Previous_Observation_Count" in df.columns else pd.Series(0, index=df.index)
    count_factor = np.clip(obs_count / 5.0, 0.2, 1.0)
    
    # Warm start combines gap decay & count history; cold start retains minimum baseline availability
    avail_a = np.where(
        has_prior > 0,
        0.4 + 0.4 * gap_factor + 0.2 * count_factor,
        0.25
    )
    avail_df["A_modality_a"] = np.clip(avail_a, 0.1, 1.0)

    # 2. Modality B Availability
    if "Weather_Available_At_Prediction" in df.columns:
        avail_df["A_modality_b"] = df["Weather_Available_At_Prediction"].astype(float)
    else:
        weather_cols = [c for c in MODALITY_B_FEATURES if c in df.columns]
        avail_df["A_modality_b"] = (df[weather_cols].notna().all(axis=1)).astype(float) if weather_cols else 0.0

    # 3. Modality C Availability
    spatial_cols = [c for c in MODALITY_C_FEATURES if c in df.columns]
    avail_df["A_modality_c"] = (df[spatial_cols].notna().all(axis=1)).astype(float) if spatial_cols else 1.0

    # 4. Modality D Availability (Unavailable)
    avail_df["A_modality_d"] = 0.0

    return avail_df


def compute_context_aware_reliability(
    df: pd.DataFrame,
    global_reliability: Dict[str, float],
) -> pd.DataFrame:
    """
    Computes row-level context-aware reliability:
        R_{i,m} = R_{global,m} * A_{i,m}
    """
    avail_df = compute_sample_availability(df)
    rel_df = pd.DataFrame(index=df.index)

    for mod in ["modality_a", "modality_b", "modality_c", "modality_d"]:
        r_glob = global_reliability.get(mod, 0.0)
        rel_df[f"R_{mod}"] = r_glob * avail_df[f"A_{mod}"]
        rel_df[f"A_{mod}"] = avail_df[f"A_{mod}"]

    return rel_df
