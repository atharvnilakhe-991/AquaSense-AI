"""
Groundwater Depletion Risk & Monitoring Priority Engine
Member 4 - AI/ML, Recharge & Risk

Calculates annual depletion velocity, risk level, monitoring priority, and uncertainty warning.
"""

from typing import Dict, Any, List, Optional
import numpy as np


def calculate_depletion_velocity(observed_history: List[float], years: List[int]) -> float:
    """
    Calculates linear trend of groundwater depth (positive value = water table dropping / deeper).
    Units: ft/year or m/year.
    """
    if len(observed_history) < 2:
        return 0.0
    
    # Simple linear slope
    x = np.array(years, dtype=float)
    y = np.array(observed_history, dtype=float)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    denominator = np.sum((x - x_mean) ** 2)
    if denominator == 0:
        return 0.0
    slope = np.sum((x - x_mean) * (y - y_mean)) / denominator
    return round(float(slope), 2)


def assess_well_risk(
    predicted_level: float,
    baseline_level: float,
    depletion_velocity: float,
    reliability_score: float,
    recharge_potential: str,
    threshold_critical_ft: float = 160.0
) -> Dict[str, Any]:
    """
    Assigns comprehensive risk tier and monitoring priority for decision support.
    
    Risk Tiers:
    - Critical: Depletion velocity > 1.2 ft/yr OR depth > threshold_critical_ft
    - Moderate: Moderate drawdown or Low recharge with positive drawdown velocity
    - Low: Stable or recovering water table
    """
    # 1. Base risk assessment
    is_deep = predicted_level >= threshold_critical_ft
    is_rapid_drawdown = depletion_velocity >= 1.0
    
    if (is_deep and is_rapid_drawdown) or (depletion_velocity >= 1.8):
        risk_level = "Critical"
    elif is_deep or is_rapid_drawdown or (recharge_potential == "Low" and depletion_velocity > 0.3):
        risk_level = "Moderate"
    else:
        risk_level = "Low"
        
    # 2. Monitoring Priority Score [0.0 - 100.0]
    # Wells with HIGH risk but LOW reliability need urgent field inspection
    monitoring_priority = (
        (1.0 - reliability_score) * 45.0 +  # Sparsity penalty (needs visit)
        (min(predicted_level, 200.0) / 200.0) * 30.0 +  # Depth severity
        (max(0.0, depletion_velocity) / 2.0) * 25.0  # Velocity severity
    )
    monitoring_priority = round(float(np.clip(monitoring_priority, 5.0, 99.0)), 1)
    
    # 3. High Uncertainty Flag (e.g. sparse well with large predicted change)
    high_uncertainty_warning = (reliability_score < 0.35) or (abs(predicted_level - baseline_level) > 28.0)
    
    return {
        "risk_level": risk_level,
        "depletion_velocity_ft_per_year": round(depletion_velocity, 2),
        "monitoring_priority_score": monitoring_priority,
        "high_uncertainty_warning": high_uncertainty_warning,
        "confidence_score": round(float(np.clip(reliability_score * 0.9 + 0.1, 0.40, 0.98)), 2),
    }
