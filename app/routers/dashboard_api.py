from fastapi import APIRouter, Query
from typing import Optional
from app.services.data_adapter import data_adapter

router = APIRouter(tags=["AquaSense Dashboard & Scientific Data"])


@router.get("/api/v1/wells")
@router.get("/wells/summary")
def get_wells(
    risk_filter: Optional[str] = Query(None, alias="risk"),
    search: Optional[str] = Query(None)
):
    """Returns all 170 monitored wells with real hydrogeological and ML decision tiers."""
    return data_adapter.get_wells(risk_filter=risk_filter, search=search)


@router.get("/api/v1/wells/{well_id}")
@router.get("/wells/summary/{well_id}")
def get_well_detail(well_id: str):
    """Returns specific well detail with predictions and decision support action."""
    return data_adapter.get_well_by_id(well_id)


@router.get("/api/v1/groundwater")
@router.get("/groundwater/overview")
def get_groundwater_data():
    """Provides core groundwater statistics, facts, and KPI card metrics."""
    return data_adapter.get_groundwater_data()


@router.get("/api/v1/predictions")
@router.get("/predictions/overview")
def get_predictions(
    timeframe: str = Query("all"),
    well_id: Optional[str] = Query(None, alias="well_id")
):
    """Provides ML prediction horizons, model performance benchmarks, and uncertainty envelopes."""
    return data_adapter.get_predictions_data(timeframe=timeframe, well_id=well_id)


@router.get("/api/v1/predictions/{well_id}")
def get_well_prediction_detail(
    well_id: str,
    timeframe: str = Query("all")
):
    """Provides ML prediction horizons and uncertainty envelopes for a specific well."""
    return data_adapter.get_predictions_data(timeframe=timeframe, well_id=well_id)


@router.get("/api/v1/recharge")
@router.get("/recharge")
def get_recharge_data():
    """Provides recharge categories, hydro-meteorological factors, and RRPI metrics."""
    return data_adapter.get_recharge_data()


@router.get("/api/v1/risk")
@router.get("/risk")
def get_risk_data():
    """Provides risk distributions and operational alerts."""
    return data_adapter.get_risk_data()


@router.get("/api/v1/amdfe")
@router.get("/amdfe")
def get_amdfe_data():
    """Provides Member 2 AMDFE sensor modalities, SNR reliability metrics, and fusion weights."""
    return data_adapter.get_amdfe_data()


@router.get("/api/v1/xai")
@router.get("/xai")
def get_xai_data():
    """Provides Member 4 Explainable AI (TreeSHAP) attributions and decision directives."""
    return data_adapter.get_xai_data()


@router.get("/api/v1/health")
@router.get("/health")
def health_check():
    """System health check endpoint."""
    return {
        "status": "healthy",
        "service": "AquaSense AI Integrated Platform",
        "version": "1.0.0",
        "scientific_data_loaded": True,
        "monitored_stations": 170,
        "observations_analyzed": 3943
    }
