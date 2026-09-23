"""
Unit & Integration Tests for Member 3: FastAPI Backend & Section 13 Contract
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.database import db

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "AquaSense AI Backend"


def test_section_13_contract_compliance():
    """
    Validates that the prediction item strictly fulfills the Section 13 integration contract:
    {
      "well_id": "...",
      "latitude": ...,
      "longitude": ...,
      "observed_level": ...,
      "predicted_level": ...,
      "risk_level": ...,
      "recharge_potential": ...,
      "confidence": ...
    }
    """
    res = client.get("/api/v1/wells")
    assert res.status_code == 200
    wells = res.json()
    assert len(wells) > 0
    
    first = wells[0]
    required_keys = [
        "well_id",
        "latitude",
        "longitude",
        "observed_level",
        "predicted_level",
        "risk_level",
        "recharge_potential",
        "confidence"
    ]
    for key in required_keys:
        assert key in first, f"Missing required Section 13 contract key: {key}"
        assert first[key] is not None or key == "observed_level"


def test_custom_prediction_endpoint():
    payload = {
        "latitude": 40.48,
        "longitude": -99.38,
        "surface_elevation_m": 715.0,
        "annual_precipitation_mm": 620.0,
        "annual_temperature_c": 11.0,
        "annual_relative_humidity_pct": 68.0,
        "annual_wind_speed_ms": 4.5,
        "annual_solar_radiation_mj": 16.0,
        "days_since_prior_obs": 365.0,
        "prior_observed_level": 120.0
    }
    res = client.post("/api/v1/predictions", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "predicted_level" in data
    assert "risk_level" in data
    assert "recharge_potential" in data
    assert 0.0 <= data["confidence"] <= 1.0


def test_global_xai_endpoint():
    res = client.get("/api/v1/xai/global")
    assert res.status_code == 200
    data = res.json()
    assert "feature_importances" in data
    assert len(data["feature_importances"]) > 0
