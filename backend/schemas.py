"""
Pydantic Schemas & Data Contracts
Member 3 - Backend & Database

Implements the official Section 13 Integration Contract and API response schemas.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class WellBase(BaseModel):
    well_id: str = Field(..., description="Unique well identifier")
    latitude: float = Field(..., description="Latitude coordinate (WGS84)")
    longitude: float = Field(..., description="Longitude coordinate (WGS84)")
    surface_elevation_m: float = Field(..., description="Surface elevation in meters")


class WellPredictionItem(BaseModel):
    """
    Official Section 13 Integration Contract Object
    """
    well_id: str
    latitude: float
    longitude: float
    observed_level: Optional[float] = None
    predicted_level: float
    risk_level: str = Field(..., description="Low, Moderate, Critical")
    recharge_potential: str = Field(..., description="Low, Moderate, High")
    confidence: float = Field(..., description="Confidence score in [0.0, 1.0]")
    
    # Extended intelligence fields
    surface_elevation_m: Optional[float] = None
    year: Optional[int] = 2024
    depletion_velocity_ft_yr: Optional[float] = 0.0
    monitoring_priority_score: Optional[float] = 50.0
    high_uncertainty_warning: Optional[bool] = False
    source_reliability_score: Optional[float] = 0.85


class PredictionRequest(BaseModel):
    latitude: float = Field(..., ge=39.5, le=41.5)
    longitude: float = Field(..., ge=-100.5, le=-98.0)
    surface_elevation_m: float = Field(..., ge=500.0, le=950.0)
    annual_precipitation_mm: float = Field(default=620.0, ge=100.0, le=2000.0)
    annual_temperature_c: float = Field(default=10.8, ge=-5.0, le=35.0)
    annual_relative_humidity_pct: float = Field(default=68.0, ge=10.0, le=100.0)
    annual_wind_speed_ms: float = Field(default=4.6, ge=0.5, le=25.0)
    annual_solar_radiation_mj: float = Field(default=16.0, ge=5.0, le=35.0)
    days_since_prior_obs: float = Field(default=365.0, ge=0.0)
    prior_observed_level: Optional[float] = Field(default=120.0)


class ObservationRecord(BaseModel):
    year: int
    observed_level: float
    annual_precipitation_mm: float
    annual_temperature_c: float


class WellDetailResponse(BaseModel):
    well_id: str
    latitude: float
    longitude: float
    surface_elevation_m: float
    current_prediction: WellPredictionItem
    historical_observations: List[ObservationRecord]
    modality_weights: Dict[str, float]


class ShapFeatureContribution(BaseModel):
    feature: str
    value: float
    attribution: float
    direction: str


class WellXaiResponse(BaseModel):
    well_id: str
    predicted_level: float
    base_value: float
    contributions: List[ShapFeatureContribution]


class GlobalXaiResponse(BaseModel):
    model_name: str
    feature_importances: List[Dict[str, Any]]


class CountySummaryKPIs(BaseModel):
    total_wells: int
    critical_risk_count: int
    moderate_risk_count: int
    low_risk_count: int
    high_recharge_count: int
    mean_predicted_depth_ft: float
    mean_system_reliability: float
    high_uncertainty_alerts: int
