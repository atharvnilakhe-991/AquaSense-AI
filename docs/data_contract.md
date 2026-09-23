# AquaSense AI: Multi-Member Integration Contract & Interfaces

## 1. Architecture Handoff Chain

```
Member 2 (AMDFE) -> Member 4 (AI/ML & XAI) -> Member 3 (Backend API) -> Member 1 (Frontend GIS)
```

---

## 2. Member 2 -> Member 4 (AMDFE Clean Data Contract)
**File Module:** `amdf/fusion.py` -> `AdaptiveMultimodalDataFusionEngine.process_and_fuse()`

Input dataframe has raw well measurements + NASA POWER weather.
Output dataframe contains:
- `well_id`: String
- `year`: Integer
- `latitude`, `longitude`: Float (WGS84)
- `surface_elevation_m`: Float (meters)
- `observed_level`: Float (groundwater depth in feet)
- `annual_precipitation_mm`: Float (mm)
- `annual_temperature_c`: Float (°C)
- `annual_relative_humidity_pct`: Float (%)
- `annual_wind_speed_ms`: Float (m/s)
- `annual_solar_radiation_mj`: Float (MJ/m²/day)
- `temporal_gap_years`: Float (causal elapsed time)
- `sampling_regularity_index`: Float `[0.0, 1.0]`
- `composite_reliability`: Float `[0.0, 1.0]`
- `groundwater_history_weight`, `meteorological_weight`, `topographic_weight`: Float

---

## 3. Member 4 -> Member 3 (ML Model & Intelligence Contract)
**File Module:** `ml/models.py`, `ml/recharge.py`, `ml/risk.py`, `ml/explainability.py`

Member 4 provides:
1. `AquaSenseModelTrainer` instance (LightGBM preferred model).
2. `estimate_recharge_potential(...)`: Returns `{ recharge_score: float, recharge_potential: 'Low'|'Moderate'|'High' }`.
3. `assess_well_risk(...)`: Returns `{ risk_level: 'Low'|'Moderate'|'Critical', depletion_velocity: float, monitoring_priority_score: float, high_uncertainty_warning: bool }`.
4. `ShapExplainerService`: Computes local SHAP attributions per feature.

---

## 4. Member 3 -> Member 1 (REST API Contract)
**File Module:** `backend/routes/`

### Primary Well Prediction Object (Section 13 Format):
```json
{
  "well_id": "WELL-NE-018",
  "latitude": 40.4812,
  "longitude": -99.3789,
  "observed_level": 122.8,
  "predicted_level": 124.6,
  "risk_level": "Moderate",
  "recharge_potential": "High",
  "confidence": 0.89,
  "surface_elevation_m": 714.0,
  "year": 2024,
  "depletion_velocity_ft_yr": 0.62,
  "monitoring_priority_score": 38.5,
  "high_uncertainty_warning": false,
  "source_reliability_score": 0.89
}
```

### Endpoints:
- `GET /api/v1/wells`: Query/filter list of monitored wells.
- `GET /api/v1/wells/{well_id}`: Single well time-series hydrograph & AMDFE weights.
- `POST /api/v1/predict`: Hyperlocal prediction for new query coordinates.
- `GET /api/v1/predictions/summary`: County-level aggregates & alerts.
- `GET /api/v1/xai/global`: Global feature importance list.
- `GET /api/v1/xai/local/{well_id}`: SHAP local attribution list.

---

## 5. Strict Target Leakage Safeguards
1. **Never use groundwater-derived TIFF values as independent predictors.**
2. **Never allow test-well observations into training sets (use GroupKFold by `well_id`).**
3. **Never compute lag features forward in time (enforce $t_{prev} < t_{target}$).**
