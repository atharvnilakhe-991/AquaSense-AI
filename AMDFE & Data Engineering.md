# Member 2 — AMDFE & Data Engineering

## Role
Core **Adaptive Multimodal Data Fusion Engine (AMDFE)** and data-processing pipeline.

## Objective
Build a leakage-safe pipeline that combines heterogeneous environmental data while accounting for data quality, reliability, missingness, and irregular monitoring.

## Tasks

### 1. Data Ingestion
Integrate groundwater, weather, elevation/topography, and independently available satellite/environmental data.

### 2. Data Quality Assessment
Calculate:
- Missing-value ratio
- Duplicate rate
- Temporal coverage
- Spatial coverage
- Outlier indicators
- Observation density

### 3. Reliability Scoring
Develop a reproducible reliability score using justified factors such as completeness, temporal consistency, spatial consistency, observation density, and measurement quality.

### 4. Adaptive Weight Assignment
Assign data-source/modality weights based on reliability instead of fixed weights.

Conceptual starting point:

`w_m = R_m / sum(R_k)`

The final formulation must be justified experimentally.

### 5. Missing-Data Recovery
Implement safe interpolation/fallback methods and missingness indicators. Do not use future information in temporal prediction.

### 6. Feature Engineering
Create spatial, temporal, environmental, reliability, and observation-density features.

### 7. Unified Fusion Dataset
Produce a documented, reproducible ML-ready dataset with units and source/reliability metadata.

### 8. Ablation Datasets
Prepare comparisons for:
- Groundwater-only
- Groundwater + weather
- Fixed-weight fusion
- Reliability-weighted fusion
- Full adaptive fusion

### 9. Leakage Prevention
Never use target-derived groundwater TIFF values as independent predictors, future observations, or test-set information during preprocessing/tuning.

## Deliverables
AMDFE pipeline, quality report, reliability module, adaptive-weight module, missing-data recovery, feature engineering, fused dataset, ablation datasets, mathematical documentation, and tests.

## Dependencies
Member 4 consumes the processed data; Member 3 exposes required data; Member 1 may visualize quality/reliability.

## Research Note
Do not claim AMDFE as novel until formal literature/prior-art review confirms the distinction.
