"""
Hyperlocal Recharge Potential Assessment Engine
Member 4 - AI/ML, Recharge & Risk

Calculates hydrological recharge score [0.0 - 100.0] and categorizes into Low, Moderate, High.
"""

from typing import Dict, Any, Tuple
import numpy as np


def classify_recharge_tier(score: float) -> str:
    """Classifies numerical recharge score into standard tier."""
    if score >= 65.0:
        return "High"
    elif score >= 40.0:
        return "Moderate"
    else:
        return "Low"


def estimate_recharge_potential(
    annual_precip_mm: float,
    surface_elevation_m: float,
    relative_humidity_pct: float,
    baseline_precip_mean: float = 620.0
) -> Dict[str, Any]:
    """
    Estimates hyperlocal groundwater recharge potential based on hydro-meteorological
    precipitation surplus, moisture conditions, and topographic slope proxies.
    
    Returns:
        Dict with recharge_score, recharge_potential tier, and contributing factors.
    """
    # 1. Precipitation Surplus Factor (dominant driver of infiltration)
    precip_ratio = annual_precip_mm / baseline_precip_mean
    precip_component = float(np.clip((precip_ratio - 0.5) * 60.0, 5.0, 70.0))
    
    # 2. Moisture / Evapotranspiration proxy (higher humidity reduces surface evaporative loss)
    humidity_component = float(np.clip((relative_humidity_pct - 45.0) * 0.5, 0.0, 20.0))
    
    # 3. Topographic drainage proxy (moderate elevations in Phelps County facilitate valley recharge)
    topo_norm = abs(surface_elevation_m - 715.0) / 50.0
    topo_component = float(np.clip(15.0 - topo_norm * 4.0, 2.0, 15.0))
    
    raw_score = precip_component + humidity_component + topo_component
    final_score = round(float(np.clip(raw_score, 5.0, 98.0)), 1)
    tier = classify_recharge_tier(final_score)
    
    return {
        "recharge_score": final_score,
        "recharge_potential": tier,
        "precip_contribution": round(precip_component, 1),
        "humidity_contribution": round(humidity_component, 1),
        "topography_contribution": round(topo_component, 1),
    }
