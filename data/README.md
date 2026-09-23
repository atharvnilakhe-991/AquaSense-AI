# AquaSense AI Data Repository

## Dataset Overview (Phelps County, Nebraska)

- **Total Observations:** 3,844 records
- **Spatial Coverage:** 170 distinct monitoring wells in Phelps County, NE, USA
- **Temporal Alignment:** 2000–2024
- **Environmental Data:** NASA POWER Hydro-Meteorological parameters

### Predictor Modalities & Features
1. **Geospatial & Topography:**
   - `latitude`: Geographic coordinate (WGS84)
   - `longitude`: Geographic coordinate (WGS84)
   - `surface_elevation_m`: Surface elevation above sea level in meters
2. **Hydro-Meteorological (NASA POWER):**
   - `annual_temperature_c`: Mean annual 2-meter air temperature (°C)
   - `annual_precipitation_mm`: Total annual precipitation (mm)
   - `annual_relative_humidity_pct`: Relative humidity at 2m (%)
   - `annual_wind_speed_ms`: Wind speed at 10m (m/s)
   - `annual_solar_radiation_mj`: All-sky surface solar radiation (MJ/m²/day)
3. **AMDFE Reliability & Irregularity Features:**
   - `days_since_prior_obs`: Days elapsed since previous historical measurement at the well
   - `historical_obs_density`: Observation frequency (measurements per decade)
   - `source_reliability_score`: Composite AMDFE reliability index in range `[0.0, 1.0]`

---

## ⚠️ Critical Data Integrity & Leakage Prevention Policy

1. **Zero Target-Derived TIFF Predictors:**
   Groundwater-derived interpolation TIFFs (e.g., IDW, Kriging) must **NEVER** be sampled or treated as independent ML predictors if they were generated using target well measurements. Doing so introduces severe target leakage and invalidates scientific generalization.
2. **Causal Temporal Features Only:**
   Historical lag features for any observation at time $t$ must only reference measurements at $t_{prev} < t$. Future measurements ($t_{next} > t$) must never be accessed during imputation, aggregation, or feature calculation.
3. **Grouped Unseen-Well Cross-Validation:**
   To evaluate generalization to newly drilled or unmonitored locations, splits must be grouped strictly by `well_id` (`GroupKFold`), ensuring that no test well is represented in the training set.
