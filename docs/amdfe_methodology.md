# Adaptive Multimodal Data Fusion Engine (AMDFE): Methodology & Mathematical Specification
**AquaSense-AI Research Pipeline**
**Target**: SCI / Q1 Hydroinformatics & Applied AI Literature
**Date**: September 2026

---

## 1. Problem Formulation & Motivation

Groundwater level dynamics are driven by complex, coupled hydrogeological processes spanning multiple spatial and temporal scales. When constructing machine learning models for regional groundwater depth prediction, researchers must integrate heterogeneous environmental data sources:
- In-situ piezometric well monitoring records (irregular, sparse, variable gap intervals)
- Meteorological observations (precipitation, temperature, solar radiation, humidity, wind)
- Topographic and spatial surface indicators (elevation, coordinates)
- Remote sensing and pedological datasets (optical/SAR indices, soil hydraulic properties)

Standard machine learning practices typically adopt **naive concatenation** (early fusion), appending all raw features into a single matrix. This naive approach suffers from three major flaws in hydrogeological settings:
1. **Heterogeneous Measurement Reliability**: Modalities with high noise, irregular sampling, or large historical gaps are treated with equal confidence as dense, continuous observations.
2. **Causal Temporal Leakage**: Standard practices frequently attach calendar-year weather aggregates to mid-year groundwater measurements, allowing future precipitation (e.g., autumn rain) to leak into spring predictions.
3. **Target-Derived Contamination**: Spatial interpolation rasters (e.g., Kriging surfaces constructed from the full target observation network) are mistakenly included as predictors, invalidating generalization claims.

**AMDFE** addresses these challenges by introducing a leakage-safe, quality-aware, context-adaptive preprocessing and fusion layer between raw heterogeneous observations and downstream non-linear machine learning regressors.

---

## 2. Modality Definitions & Data Architecture

AMDFE formalizes input variables into 4 distinct modalities:

```
+--------------------------------------------------------------------------------------------------+
|                                    AMDFE MODALITY ARCHITECTURE                                   |
+--------------------------------------------------------------------------------------------------+
| Modality A: Groundwater History & Dynamic State                                                  |
|   - Variables: Previous_WatLevel, Previous2_WatLevel, Days_Since_Previous, Years_Since_Previous,     |
|                Rolling_Mean_3, Rolling_Std_3, Previous_Level_Change, Recent_Trend,               |
|                Historical_Mean, Historical_Std, Historical_Min, Historical_Max,                  |
|                Previous_Observation_Count, Observation_Density, Long_Gap_Flag, Very_Long_Gap_Flag|
|   - Precedence: Strictly prior observation dates (t_prior < t_current). Zero current-date info. |
+--------------------------------------------------------------------------------------------------+
| Modality B: Environmental & Meteorological Covariates                                            |
|   - Variables: Annual_Temperature_Mean, Annual_Precipitation_Total, Annual_Humidity_Mean,        |
|                Annual_WindSpeed_Mean, Annual_SolarRadiation_Mean                                 |
|   - Precedence: Strictly completed prior annual period (Y - 1) relative to measurement year Y.   |
+--------------------------------------------------------------------------------------------------+
| Modality C: Spatial Coordinates & Topographic Context                                            |
|   - Variables: LatDD, LongDD, Surf_Elev                                                          |
|   - Precedence: Static site invariant attributes.                                                |
+--------------------------------------------------------------------------------------------------+
| Modality D: Earth Observation & Pedology (SoilGrids / Sentinel)                                   |
|   - Variables: [Plug-in Interface]                                                               |
|   - Status: UNAVAILABLE_IN_CURRENT_REPOSITORY (Explicitly assigned zero weight, no fake values). |
+--------------------------------------------------------------------------------------------------+
```

---

## 3. Data Quality Assessment Dimensions

For each available modality $m \in \{A, B, C\}$, AMDFE evaluates four objective, observable quality dimensions:

1. **Completeness ($C_m \in [0, 1]$)**:
   $$C_m = 1 - \frac{1}{N \cdot D_m} \sum_{i=1}^N \sum_{j=1}^{D_m} \mathbb{I}(X_{i,j}^{(m)} \text{ is null})$$
   where $D_m$ is the dimensionality of modality $m$, and $N$ is the number of training samples.

2. **Temporal Consistency ($T_m \in [0, 1]$)**:
   - For Modality A: $T_A = \text{WarmStart\_Ratio} \cdot (1 - 0.5 \cdot \text{LongGap\_Ratio})$, penalizing wells with persistent monitoring gaps ($>365$ days).
   - For Modality B: $T_B = \text{Available\_Prior\_Year\_Fraction}$, reflecting the fraction of observations where the lagged completed period $Y-1$ was available.
   - For Modality C: $T_C = 1.0$ (static geodetic attributes).

3. **Spatial Coverage ($S_m \in [0, 1]$)**:
   Quantifies domain validity and coordinate boundary compliance.

4. **Observation Density ($O_m \in [0, 1]$)**:
   Normalized measurement frequency across the regional well network.

5. **Measurement Quality ($M_m$)**:
   Explicit sensor precision error bars. Because the raw dataset does not provide sensor calibration metadata, $M_m$ is recorded as `unavailable` rather than being assigned a fabricated score.

---

## 4. Reliability Score Formulation & Sensitivity Analysis

### 4.1 Global Source Reliability ($R_{\text{global}, m}$)
Source-level reliability is aggregated from available normalized quality dimensions. To enforce conservative bounds where a deficiency in any dimension penalizes the source, AMDFE adopts the **geometric mean** as its primary formulation:
$$R_{\text{global}, m} = \left( \prod_{j \in \{C, T, S, O\}} Q_{m, j} \right)^{1/4}$$

### 4.2 Aggregation Sensitivity
To ensure AMDFE is not an ad-hoc hand-tuned score, we conduct a sensitivity analysis comparing three classical Pythagorean aggregations:
- **Arithmetic Mean**: $R_{\text{arith}, m} = \frac{1}{K} \sum_{j=1}^K Q_{m, j}$ (Linear additive)
- **Geometric Mean**: $R_{\text{geom}, m} = \left( \prod_{j=1}^K Q_{m, j} \right)^{1/K}$ (Multiplicative conservative)
- **Harmonic Mean**: $R_{\text{harm}, m} = \frac{K}{\sum_{j=1}^K Q_{m, j}^{-1}}$ (Strict bottleneck penalty)

### 4.3 Context-Aware Row Availability ($A_{i, m}$)
Global reliability is modulated at sample $i$ by local observation availability $A_{i, m} \in [0, 1]$:
- **Modality A (Groundwater)**:
  $$A_{i, A} = \begin{cases} 0.4 + 0.4 \exp\left(-\frac{\Delta t_i}{730}\right) + 0.2 \min\left(1.0, \frac{N_{\text{prior}, i}}{5}\right), & \text{if } N_{\text{prior}, i} > 0 \\ 0.25, & \text{if cold-start } (N_{\text{prior}, i} = 0) \end{cases}$$
- **Modality B (Weather)**: $A_{i, B} = \mathbb{I}(\text{Weather of } Y-1 \text{ is complete})$
- **Modality C (Spatial)**: $A_{i, C} = \mathbb{I}(\text{Coordinates \& Elevation non-null})$
- **Modality D (Remote Sensing)**: $A_{i, D} = 0.0$

### 4.4 Dynamic Sample Reliability ($R_{i, m}$)
$$R_{i, m} = R_{\text{global}, m} \cdot A_{i, m}$$

---

## 5. Adaptive Modality Weighting & Fusion Mechanism

For each observation row $i$, the normalized modality weights are assigned by the partition of unity over active modalities $\mathcal{M}$:
$$w_{i, m} = \frac{R_{i, m}}{\sum_{k \in \mathcal{M}} R_{i, k}}$$

### Fallback Policy:
If $\sum_{k \in \mathcal{M}} R_{i, k} = 0$ (extreme edge case where all modalities are unavailable), AMDFE applies a uniform fallback weight $w_{i, m} = \frac{1}{|\mathcal{M}|}$ and asserts `Fallback_Flag = 1`.

### Feature Transformation & Scaling:
1. Imputation is fitted strictly on the **TRAIN partition** using feature medians.
2. Continuous features in each modality are standardized using **TRAIN statistics only**:
   $$Z(X_{i, j}^{(m)}) = \frac{X_{i, j}^{(m)} - \mu_{j, \text{train}}^{(m)}}{\sigma_{j, \text{train}}^{(m)}}$$
3. Weighted fused representations are generated:
   $${X'}_{i, j}^{(m)} = w_{i, m} \cdot Z(X_{i, j}^{(m)})$$
4. Traceable metadata retained in the output schema:
   - Original unweighted features $X_{i, j}^{(m)}$
   - Standardized features $Z(X_{i, j}^{(m)})$
   - Fused representations ${X'}_{i, j}^{(m)}$
   - Reliability scores $R_{i, m}$ and availability factors $A_{i, m}$
   - Dynamic weights $w_{i, m}$
   - Missingness indicators $\text{Missing}_{X_{i, j}}$
   - Traceability metadata (`Weather_Year_Used`, `Weather_Available_At_Prediction`, `Fallback_Flag`)

---

## 6. Core Ablation Configurations (B0 – B4)

To isolate the source of predictive gains and test scientific hypotheses, AMDFE defines 5 strictly separated ablation configurations:

| Suite | Configuration Name | Modalities | Weighting Strategy | Mathematical Formulation |
| :---: | :--- | :--- | :--- | :--- |
| **B0** | Groundwater Only | Modality A | Unweighted | $X = Z(X_A)$ |
| **B1** | Groundwater + Weather | Modalities A, B | Unweighted Concat | $X = [Z(X_A), Z(X_B)]$ |
| **B2** | Fixed-Weight Multimodal Fusion | Modalities A, B, C | Equal Fixed Contribution | ${X'}_m = \frac{1}{3} Z(X_m)$ |
| **B3** | Global Reliability-Weighted Fusion | Modalities A, B, C | Static Dataset Reliability | ${X'}_m = \bar{w}_m Z(X_m), \quad \bar{w}_m = \frac{R_{\text{global}, m}}{\sum R_k}$ |
| **B4** | Full Context-Adaptive AMDFE | Modalities A, B, C | Dynamic Sample-Aware | ${X'}_{i, m} = w_{i, m} Z(X_{i, m})$ |

---

## 7. Data Leakage Prevention Protocol

AMDFE enforces 10 automated assertion checks:
1. **AMDFE-LC-01**: Target `WatLevel` excluded from predictor matrices.
2. **AMDFE-LC-02**: `TIFF_Value` and target-derived Kriging rasters strictly purged.
3. **AMDFE-LC-03**: No future groundwater observations used in lag calculations ($t_{\text{prior}} < t_{\text{current}}$).
4. **AMDFE-LC-04**: $\Delta t \ge 0$ strictly non-negative.
5. **AMDFE-LC-05**: Meteorological features derived strictly from completed period $Y-1$.
6. **AMDFE-LC-06**: Zero mid-year lookahead from uncompleted calendar-year weather.
7. **AMDFE-LC-07**: Unseen-well holdout isolates train and test wells with 0 overlap.
8. **AMDFE-LC-08**: Imputers, scalers, and reliability transformers fit on TRAIN partition only.
9. **AMDFE-LC-09**: No target-derived raster in any output schema.
10. **AMDFE-LC-10**: Deterministic reproducibility across repeated random seeds.

---

## 8. Current Limitations & Extensibility

1. **Absence of Remote Sensing Covariates**:
   - Sentinel-2 optical indices (NDVI, NDWI) and GRACE-FO terrestrial water storage anomalies are not yet localized in this repository. Modality D is maintained as an open plug-in interface.
2. **Coarse Meteorological Temporal Granularity**:
   - Annual weather aggregates are currently available. While lagging to $Y-1$ prevents temporal leakage, integrating daily/monthly ERA5-Land reanalysis is the planned next iteration.
3. **Static Elevation**:
   - `DEM_Elevation` is currently unavailable; wellhead surface elevation `Surf_Elev` is utilized as the primary topographic covariate.
