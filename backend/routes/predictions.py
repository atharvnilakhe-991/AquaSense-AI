"""
Predictions & Summary API Endpoints
Member 3 - Backend & Database
"""

from typing import List, Dict, Any
import numpy as np
from fastapi import APIRouter
from backend.database import db
from backend.schemas import PredictionRequest, WellPredictionItem, CountySummaryKPIs
from ml.dataset import FEATURE_COLUMNS
from ml.recharge import estimate_recharge_potential
from ml.risk import assess_well_risk
from amdf.reliability import estimate_source_reliability

router = APIRouter(prefix="/api/v1/predictions", tags=["Predictions"])


@router.post("", response_model=WellPredictionItem)
def predict_hyperlocal_level(req: PredictionRequest):
    """
    Predicts groundwater depth at any custom coordinate in Phelps County
    using AMDFE adaptive fusion and the trained LightGBM model.
    """
    if not db.is_initialized:
        db.initialize()
        
    gap_years = max(1.0, req.days_since_prior_obs / 365.25)
    rel_score = estimate_source_reliability(temporal_gap_years=gap_years, historical_count=5)
    
    # Feature vector matching FEATURE_COLUMNS:
    # latitude, longitude, surface_elevation_m, annual_temperature_c, annual_precipitation_mm,
    # annual_relative_humidity_pct, annual_wind_speed_ms, annual_solar_radiation_mj,
    # temporal_gap_years, sampling_regularity_index, composite_reliability
    reg_index = float(np.exp(-0.25 * max(0.0, gap_years - 1.0)))
    
    feature_vals = [
        req.latitude,
        req.longitude,
        req.surface_elevation_m,
        req.annual_temperature_c,
        req.annual_precipitation_mm,
        req.annual_relative_humidity_pct,
        req.annual_wind_speed_ms,
        req.annual_solar_radiation_mj,
        gap_years,
        reg_index,
        rel_score
    ]
    
    X = np.array(feature_vals).reshape(1, -1)
    pred_level = float(db.trainer.predict(X)[0])
    
    # Recharge
    recharge_info = estimate_recharge_potential(
        annual_precip_mm=req.annual_precipitation_mm,
        surface_elevation_m=req.surface_elevation_m,
        relative_humidity_pct=req.annual_relative_humidity_pct
    )
    
    # Risk
    depletion_vel = (pred_level - (req.prior_observed_level or pred_level)) / gap_years
    risk_info = assess_well_risk(
        predicted_level=pred_level,
        baseline_level=req.prior_observed_level or pred_level,
        depletion_velocity=depletion_vel,
        reliability_score=rel_score,
        recharge_potential=recharge_info["recharge_potential"]
    )
    
    return WellPredictionItem(
        well_id="CUSTOM-QUERY-POINT",
        latitude=req.latitude,
        longitude=req.longitude,
        observed_level=req.prior_observed_level,
        predicted_level=round(pred_level, 2),
        risk_level=risk_info["risk_level"],
        recharge_potential=recharge_info["recharge_potential"],
        confidence=risk_info["confidence_score"],
        surface_elevation_m=req.surface_elevation_m,
        year=2024,
        depletion_velocity_ft_yr=risk_info["depletion_velocity_ft_per_year"],
        monitoring_priority_score=risk_info["monitoring_priority_score"],
        high_uncertainty_warning=risk_info["high_uncertainty_warning"],
        source_reliability_score=round(rel_score, 3)
    )


@router.get("/summary", response_model=CountySummaryKPIs)
def get_county_summary_kpis():
    """
    Returns aggregated county-wide intelligence KPIs for Phelps County.
    """
    wells = db.get_all_wells()
    
    total = len(wells)
    critical_count = sum(1 for w in wells if w["risk_level"] == "Critical")
    moderate_count = sum(1 for w in wells if w["risk_level"] == "Moderate")
    low_count = sum(1 for w in wells if w["risk_level"] == "Low")
    high_recharge = sum(1 for w in wells if w["recharge_potential"] == "High")
    high_uncertainty = sum(1 for w in wells if w.get("high_uncertainty_warning", False))
    
    mean_depth = float(np.mean([w["predicted_level"] for w in wells])) if total > 0 else 0.0
    mean_rel = float(np.mean([w["source_reliability_score"] for w in wells])) if total > 0 else 0.0
    
    return CountySummaryKPIs(
        total_wells=total,
        critical_risk_count=critical_count,
        moderate_risk_count=moderate_count,
        low_risk_count=low_count,
        high_recharge_count=high_recharge,
        mean_predicted_depth_ft=round(mean_depth, 2),
        mean_system_reliability=round(mean_rel, 3),
        high_uncertainty_alerts=high_uncertainty
    )
