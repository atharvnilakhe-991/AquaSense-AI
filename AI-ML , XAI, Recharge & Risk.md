# Member 4 — AI/ML, XAI, Recharge & Risk

## Role
Advanced ML, model evaluation, XAI, recharge prediction, uncertainty/error analysis, and decision-support logic.

> This is also the Group Leader's technical role, including coordination of integration between ML, AMDFE, backend, and frontend.

## Research Direction
**Observation-Aware Adaptive Multimodal Data Fusion for Groundwater Prediction Under Irregular and Incomplete Monitoring**

The central investigation is whether observation history, reliability, missingness, and adaptive multimodal weighting improve realistic groundwater prediction.

## Current ML Dataset
`data/processed/ml_validation/groundwater_ml_leakage_safe.csv`

Primary predictors:
- Latitude
- Longitude
- Surface elevation
- Annual temperature
- Annual precipitation
- Annual humidity
- Annual wind speed
- Annual solar radiation

`TIFF_Value` must remain excluded as a primary predictor because the groundwater TIFFs were generated from groundwater observations and therefore contain target-derived information.

## Completed ML Work
Models:
- Random Forest
- XGBoost
- LightGBM

Current unseen-well results:

| Model | R² | MAE | RMSE |
|---|---:|---:|---:|
| Random Forest | 0.7848 | 14.53 | 22.97 |
| XGBoost | 0.8046 | 12.75 | 21.89 |
| LightGBM | **0.8107** | **12.75** | **21.55** |

LightGBM is the current preferred candidate, but stronger repeated/grouped validation is still required.

## Completed Error Analysis
Generalization R² gaps:
- RF: 0.2119
- XGBoost: 0.1910
- LightGBM: 0.1839

Groundwater-level vs absolute-error correlations:
- RF: 0.4095
- XGBoost: 0.4204
- LightGBM: 0.5341

These show that error varies across groundwater-level regimes.

## Current Task — Irregular Observation Features
Script:
`src/08_4_irregular_observation_features.py`

Output:
`data/processed/advanced_ml/irregular_observation/groundwater_irregular_observation_features.csv`

Features include:
- Previous_WatLevel
- Previous2_WatLevel
- Days_Since_Previous
- Years_Since_Previous
- Previous_Observation_Count
- Rolling_Mean_3
- Rolling_Std_3
- Previous_Level_Change
- Recent_Trend
- Historical_Mean
- Historical_Std
- Historical_Min
- Historical_Max
- Long_Gap_Flag
- Very_Long_Gap_Flag
- Observation_Density

All history features must use only observations before the current target.

## Next Tasks

### 1. Observation-Aware ML
Compare:
- Environmental-only
- Environmental + observation history
- Environmental + history + reliability features

### 2. Strong Validation
Use:
- Temporal holdout
- Unseen-well holdout
- Repeated grouped well splits
- Spatial/spatial-block validation where practical
- Cold-start unseen wells
- Warm-start unseen wells

### 3. Ablation
Measure the contribution of environmental variables, observation history, reliability, adaptive fusion, and individual modalities.

### 4. XAI
Implement SHAP for:
- Global importance
- Summary plots
- Individual predictions
- Feature contribution tables
- Explanations by risk level

### 5. Uncertainty
Investigate prediction intervals, quantile models, ensemble spread, or residual-based uncertainty. Do not call uncertainty a calibrated probability without calibration.

### 6. Recharge Prediction
Develop a separate recharge estimation/model using justified variables such as precipitation, temperature, groundwater trend, soil/land-use, elevation, and other approved hydro-environmental features.

Outputs:
- Recharge score/value
- Low/Moderate/High potential
- Supporting explanation

### 7. Risk & Decision Support
Produce:
- Depletion risk
- Recharge potential
- High-uncertainty warning
- Monitoring priority

Thresholds must be documented and scientifically justified.

## Deliverables
Leakage-safe ML pipeline, advanced model comparison, generalization/error analysis, observation-aware model, ablation study, SHAP/XAI, uncertainty analysis, recharge model, risk logic, saved outputs, and model documentation.

## Integration
Provide Member 3 with stable prediction/recharge/risk/XAI/uncertainty schemas and Member 1 with sample JSON.

## Research Caution
RF, XGBoost, LightGBM, SHAP, satellite data, or multimodal inputs alone are not sufficient novelty claims. Novelty must be established through literature/prior-art review and demonstrated experimentally.
