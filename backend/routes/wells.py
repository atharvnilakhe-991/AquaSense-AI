"""
Wells API Endpoints
Member 3 - Backend & Database
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from backend.database import db
from backend.schemas import WellPredictionItem, WellDetailResponse, ObservationRecord

router = APIRouter(prefix="/api/v1/wells", tags=["Wells"])


@router.get("", response_model=List[WellPredictionItem])
def list_wells(
    risk_level: Optional[str] = Query(None, description="Filter by risk: Low, Moderate, Critical"),
    recharge_potential: Optional[str] = Query(None, description="Filter by recharge: Low, Moderate, High"),
    min_confidence: Optional[float] = Query(0.0, ge=0.0, le=1.0),
    search: Optional[str] = Query(None, description="Search by well_id")
):
    """
    Returns all monitored wells in Phelps County with current predicted groundwater levels.
    """
    wells = db.get_all_wells()
    
    if risk_level:
        wells = [w for w in wells if w["risk_level"].lower() == risk_level.lower()]
    if recharge_potential:
        wells = [w for w in wells if w["recharge_potential"].lower() == recharge_potential.lower()]
    if min_confidence > 0:
        wells = [w for w in wells if w["confidence"] >= min_confidence]
    if search:
        wells = [w for w in wells if search.lower() in w["well_id"].lower()]
        
    return [
        WellPredictionItem(
            well_id=w["well_id"],
            latitude=w["latitude"],
            longitude=w["longitude"],
            observed_level=w["observed_level"],
            predicted_level=w["predicted_level"],
            risk_level=w["risk_level"],
            recharge_potential=w["recharge_potential"],
            confidence=w["confidence"],
            surface_elevation_m=w["surface_elevation_m"],
            year=w["year"],
            depletion_velocity_ft_yr=w["depletion_velocity_ft_yr"],
            monitoring_priority_score=w["monitoring_priority_score"],
            high_uncertainty_warning=w["high_uncertainty_warning"],
            source_reliability_score=w["source_reliability_score"]
        )
        for w in wells
    ]


@router.get("/{well_id}", response_model=WellDetailResponse)
def get_well_detail(well_id: str):
    """
    Returns complete metadata, historical time series hydrograph observations,
    and AMDFE modality reliability weights for a specific well.
    """
    well = db.get_well(well_id)
    if not well:
        raise HTTPException(status_code=404, detail=f"Well '{well_id}' not found.")
        
    pred_item = WellPredictionItem(
        well_id=well["well_id"],
        latitude=well["latitude"],
        longitude=well["longitude"],
        observed_level=well["observed_level"],
        predicted_level=well["predicted_level"],
        risk_level=well["risk_level"],
        recharge_potential=well["recharge_potential"],
        confidence=well["confidence"],
        surface_elevation_m=well["surface_elevation_m"],
        year=well["year"],
        depletion_velocity_ft_yr=well["depletion_velocity_ft_yr"],
        monitoring_priority_score=well["monitoring_priority_score"],
        high_uncertainty_warning=well["high_uncertainty_warning"],
        source_reliability_score=well["source_reliability_score"]
    )
    
    hist_objs = [ObservationRecord(**item) for item in well["history"]]
    
    return WellDetailResponse(
        well_id=well["well_id"],
        latitude=well["latitude"],
        longitude=well["longitude"],
        surface_elevation_m=well["surface_elevation_m"],
        current_prediction=pred_item,
        historical_observations=hist_objs,
        modality_weights=well["modality_weights"]
    )
