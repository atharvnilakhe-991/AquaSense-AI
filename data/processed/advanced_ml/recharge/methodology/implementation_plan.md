# Phase 08.10 Methodology Design Document
## Hyperlocal Recharge-Response & Recharge-Potential Assessment

> **Phase Status**: METHODOLOGY APPROVED FOR IMPLEMENTATION — FINAL REVISION  
> **Proposed Script (Future)**: `src/08_10_recharge_prediction.py`  
> **Proposed Output Directory (Future)**: `data/processed/advanced_ml/recharge/`  
> **Locked Phases**: Phases 08.4, 08.5, 08.6, 08.7, 08.8, and 08.9 are strictly **LOCKED** (Read-Only)  
> **Notice**: Implementation is not yet authorized. The methodology has been revised and is awaiting final approval.

---

## 1. Phase Objective

Phase 08.10 establishes the scientific methodology for assessing localized groundwater-level response to precipitation and evaluating recharge potential under irregular monitoring in Phelps County, Nebraska.

### Core Scientific Questions:
1. *“Given antecedent precipitation pulses, atmospheric conditions, topography, and pre-existing water table depth, what is the expected groundwater-level response over the monitoring interval?”*
2. *“How does antecedent groundwater state correlate with the observed water-level response to surface weather drivers?”*
3. *“What empirical recharge potential can be characterized across geographic sub-regions under varying hydro-climatic conditions?”*

### Scope Exclusion:
Risk categorization, regulatory vulnerability thresholds, and groundwater management policy rules are strictly excluded from Phase 08.10. They belong exclusively to **Phase 08.11 (Risk & Decision Support)**, which will consume the recharge-response estimates produced here.

---

## 2. Data Inventory & Physical Availability Audit

A rigorous inventory of the raw and processed project assets was conducted:

### Available Datasets:
1. **Physical Groundwater Monitoring Observations** (`data/processed/advanced_ml/irregular_observation/groundwater_irregular_observation_features.csv`):
   - 3,844 physical monitoring records across 170 unique wells in Phelps County, Nebraska (2000–2024).
   - Irregular monitoring cadence: predominantly semi-annual (spring/fall) and annual measurements.
   - Primary state variable from prior phases: `WatLevel` (depth to water table below land surface in feet).
2. **Continuous Daily Weather Observations** (`data/raw/weather/NASA_POWER_2000_2025.csv`):
   - 9,132 consecutive daily records from 2000-01-01 through 2024-12-31 at 40.52°N, 99.41°W.
   - Coverage: Exactly 321 out of 321 unique groundwater observation dates (100%) are completely covered.

### Atmospheric & Evaporative Variable Classification:
- **Category A — Directly Available NASA POWER Variables**:
  - `T2M`: Daily mean temperature at 2 meters (°C)
  - `PRECTOTCORR`: Corrected total precipitation (mm/day)
  - `RH2M`: Relative humidity at 2 meters (%)
  - `WS2M`: Wind speed at 2 meters (m/s)
  - `ALLSKY_SFC_SW_DWN`: All-sky surface downward shortwave irradiance (MJ/m²/day)
- **Category B — Potentially Derived Variables (Future Enhancement Only)**:
  - Direct Potential Evapotranspiration (PET) observations are **NOT** present in the dataset.
  - While daily temperature, solar radiation, humidity, and wind speed exist, physical PET calculation (e.g., FAO-56 Penman-Monteith) cannot be defensibly derived without additional site-specific parameters including surface albedo, canopy resistance, aerodynamic roughness, and crop coefficients.
  - Consequently, direct PET is not available and is therefore excluded from the primary predictor set.
  - Instead, atmospheric and evaporative conditions are characterized strictly through demonstrably available raw variables (`T2M`, `ALLSKY_SFC_SW_DWN`, `RH2M`, `WS2M`).
  - Future PET derivation may be evaluated as a hydrogeological enhancement if the required assumptions, input variables, and parameterization can be independently justified.

### Absent Datasets & Physical Gaps:
1. **Zero Direct Recharge Measurements**:
   - There are NO lysimeters, soil percolation meters, isotopic tracer flux records, or seepage meters in the dataset.
2. **Missing Aquifer Storage Parameters**:
   - Well-specific specific-yield/storage parameters are not available in the current project dataset.
   - Continuous water-level hydrographs (e.g., hourly/daily pressure transducers) are absent.
3. **Unmeasured Agricultural Pumping**:
   - Agricultural pumping may confound groundwater-level response, but pumping observations are not available at the required well-specific temporal resolution in the current project dataset.

---

## 3. Core Scientific Problem & Recharge Claim Boundaries

The project dataset does NOT contain a direct measured recharge-rate target.

### Claim Boundaries:
The current dataset supports:
- Groundwater-response modelling
- Precipitation-response analysis
- Recharge-potential assessment

The current dataset does NOT directly support:
- Measured recharge-rate prediction
- Exact recharge volume
- Aquifer recharge flux in mm/year
- Physically calibrated recharge without additional hydrogeological parameters

> [!WARNING]
> **Groundwater Response $\neq$ Direct Recharge**:
> Groundwater-level change ($\Delta h$) is affected by precipitation infiltration, agricultural pumping, evapotranspiration, regional groundwater gradient flow, boundary conditions, well construction, and measurement timing.
> Therefore, $\Delta h$ must be explicitly recognized as a measured groundwater-head/depth response, and must **NEVER** be silently renamed as or equated to physical recharge.

---

## 4. Phase Architecture: Warm-Start vs. Cold-Start Distinction

Because groundwater monitoring is irregular, the methodology establishes a fundamental structural bifurcation between warm-start and cold-start regimes:

```
                               PHASE 08.10
                                    │
          ┌─────────────────────────┴─────────────────────────┐
          ▼                                                   ▼
     WARM-START                                          COLD-START
  • Previous observation exists                       • No previous observation exists
  • Target = Δh groundwater response                  • No Δh target (cannot be calculated)
  • Supervised Machine Learning                       • Diagnostic Assessment Only
  • Predictors: Antecedent state +                    • Predictors: Purely environmental +
    precipitation exposure                              climate-precipitation proxy
```

### 1. Primary Supervised Track (Warm-Start):
- Condition: A previous observation for the well exists ($\text{Previous\_Observation\_Count} > 0$).
- Target: Net Groundwater Head Response ($\Delta h$).
- Predictors: Environmental weather exposure + antecedent groundwater state.
- Methodology: Supervised machine learning.

### 2. Secondary Diagnostic Track (Cold-Start):
- Condition: First observation of a well ($\text{Previous\_Observation\_Count} == 0$).
- Target: **NO SUPERVISED $\Delta h$ TARGET**. Cold-start observations have no supervised $\Delta h$ target and must not be assigned fabricated $\Delta h$ targets.
- Methodology: Cold-start observations may be used only for a diagnostic hydro-climatic recharge-potential assessment based on the predefined RRPI formulation.
- Validation Limit: Because no direct recharge observations exist in the project dataset, RRPI cannot be validated as measured recharge or evaluated using conventional supervised prediction metrics against a recharge-ground-truth target.

---

## 5. Target Formulation: Groundwater Head Response ($\Delta h$)

For eligible warm-start observations, the supervised target is formulated as:
$$\Delta h_i = \text{Previous\_WatLevel}_i - \text{WatLevel}_i \quad [\text{ft}]$$

### Sign Convention Verification Requirement:
> [!IMPORTANT]
> **Sign Convention Rule**:
> The sign convention must be verified against the source definition of `WatLevel` before implementation:
> - In this dataset, `WatLevel` represents depth to water table below land surface in feet.
> - Under this convention, if depth decreases ($\text{WatLevel}_i < \text{Previous\_WatLevel}_i$), the water table has risen toward the land surface.
> - **Positive $\Delta h > 0$**:
>   Shallower/rising groundwater response. This may be consistent with recharge or groundwater recovery, but the available dataset does not permit attributing the response uniquely to recharge.
> - **Negative $\Delta h < 0$**:
>   Deeper/falling groundwater response, potentially associated with drawdown or drainage; the model does not attribute causality to a specific process.
> - $\Delta h$ is a measured groundwater-head/depth response; it does not itself measure recharge.

### Sample Count Qualification:
Final sample counts will be computed after target construction and eligibility filtering. The first observation of each well (170 cold-start records) cannot contribute a warm-start $\Delta h$ target and will be evaluated exclusively under the secondary diagnostic track.

---

## 6. Secondary Diagnostic Track: Rainfall-Response Recharge Potential Index (RRPI)

For diagnostic assessment across both warm and cold settings, Phase 08.10 defines a conceptual **Rainfall-Response Recharge Potential Index (RRPI)** based only on demonstrably available environmental variables.

### Conceptual Definition:
RRPI is an interpretable, standardized hydro-climatic proxy measuring surface precipitation exposure and atmospheric conditions over specified antecedent periods:
$$\text{RRPI} = g\Big( \text{Precipitation Exposure},\; \text{Atmospheric Drivers},\; \text{Seasonality} \Big)$$

### Candidate Components (Available in NASA POWER):
- Antecedent precipitation accumulations (short-term and multi-week)
- Rainfall frequency (number of rain days $> 1\text{ mm}$)
- Rainfall intensity (maximum daily precipitation)
- Daily temperature (`T2M`)
- Solar radiation (`ALLSKY_SFC_SW_DWN`)
- Seasonality (measurement timing)
- Dry-period duration (consecutive days without rainfall)

### Scientific Boundaries of RRPI:
- **Diagnostic Hydro-Climatic Index**: RRPI is a diagnostic hydro-climatic index; RRPI is **NOT** measured recharge and is **NOT** recharge flux.
- **No Volumetric Flux**: RRPI is not expressed in mm/year unless future physical calibration data become available.
- **Validation Limit**: Because no direct recharge observations exist in the project dataset, RRPI cannot be validated as measured recharge or evaluated using conventional supervised prediction metrics against a recharge-ground-truth target. No ground-truth recharge accuracy claim will be made.
- **Cold-Start Target Rule**: Cold-start observations have no supervised $\Delta h$ target and must not be assigned fabricated $\Delta h$ targets; they are evaluated exclusively through this diagnostic track.
- The exact mathematical formulation remains an implementation-stage decision after the baseline data audit.

---

## 7. Anti-Circularity Architecture & Leakage Prevention

To ensure valid supervised learning of $\Delta h_i$, strict anti-circularity boundaries are enforced:

> [!CAUTION]
> **Mandatory Anti-Circularity Safeguards**:
> 1. Current water level $\text{WatLevel}_i$ must be **STRICTLY EXCLUDED** from the predictor feature matrix. Predicting $\Delta h_i = y_{i-1} - y_i$ with $y_i$ present in the feature set would constitute fatal target leakage.
> 2. `Previous_WatLevel` ($y_{i-1}$) is permitted as an **antecedent groundwater-state indicator**, representing pre-event aquifer conditions.
> 3. `Previous_Level_Change` ($y_{i-1} - y_{i-2}$) represents prior change and does not encode the current target $\Delta h_i$.
> 4. All predictors must be derived strictly prior to or on the measurement date.

### Leakage Control Gates (RCH-LC-01 through RCH-LC-10):

| Gate ID | Gate Name | Verification Check | PASS Criteria | FAIL Criteria |
| :---: | :--- | :--- | :--- | :--- |
| **RCH-LC-01** | Future Weather Isolation | Timestamps of weather records | $\max(\text{Date}_{\text{weather}}) \le \text{DateMsr}_i$ for all features | Weather dated $> \text{DateMsr}_i$ used |
| **RCH-LC-02** | Future Groundwater Isolation | Timestamps of groundwater observations | No observation dated $> \text{DateMsr}_i$ used in predictors | Groundwater dated $> \text{DateMsr}_i$ used |
| **RCH-LC-03** | Target Exclusion from Features | Predictor matrix composition | Current `WatLevel` strictly absent from predictors | Current `WatLevel` present in features |
| **RCH-LC-04** | Same-Event Circularity Check | Prior water level reference | $\text{Previous\_WatLevel}$ strictly represents $y_{i-1}$, not $y_i$ | Target subtracted from itself |
| **RCH-LC-05** | Temporal Partition Precedence | Train/cal/test dates | $\max(\text{Date}_{\text{train}}) < \min(\text{Date}_{\text{cal}}) < \min(\text{Date}_{\text{test}})$ | Temporal partition overlap |
| **RCH-LC-06** | Well-Level Separation | Well IDs across partitions | $\text{Wells}_{\text{train}} \cap \text{Wells}_{\text{cal}} \cap \text{Wells}_{\text{test}} = \emptyset$ (10 seeds) | Well overlap across sets |
| **RCH-LC-07** | Spatial Fold Isolation | Spatial cluster boundaries | Held-out cluster wells strictly absent from train/cal | Spatial fold contamination |
| **RCH-LC-08** | Static Raster Exclusion | Prohibited raster variable | `TIFF_Value` strictly absent from all feature sets | `TIFF_Value` present in predictors |
| **RCH-LC-09** | Scaling Parameter Isolation | Standardization scalers | Scalers fit strictly on $D_{\text{train}}$; zero test leakage | Test data used to fit scalers |
| **RCH-LC-10** | Diagnostic Construction Independence | RRPI formulation | RRPI formula independent of future observations | Future data used in RRPI |

---

## 8. Feature Engineering: Precipitation Lags & Environmental Exposure

### 1. Cumulative Monitoring Interval Exposure:
- `Precip_Interval_Sum`: Cumulative precipitation exposure between the previous groundwater observation ($\text{DateMsr}_{i-1}$) and the current prediction date ($\text{DateMsr}_i$):
  $$\text{Precip\_Interval\_Sum}_i = \sum_{t = \text{DateMsr}_{i-1} + 1}^{\text{DateMsr}_i} \text{Daily\_Precip}_t$$
  *Explicit Clarification*: It is a predictor of environmental exposure and must not be interpreted as measured recharge.

### 2. Multi-Scale Precipitation Windows & Multicollinearity Management:
Candidate fixed precipitation accumulation windows (engineered from continuous daily NASA POWER data):
- `Precip_7D_Sum`, `Precip_14D_Sum`, `Precip_30D_Sum`, `Precip_60D_Sum`, `Precip_90D_Sum`, `Precip_180D_Sum`.

> [!NOTE]
> **Multicollinearity Management**:
> Fixed precipitation windows (7D, 14D, 30D, 60D, 90D, 180D) may be strongly correlated.
> They will not be assumed to be independent.
> Instead, they will be evaluated systematically through:
> - Predefined feature groups
> - Feature ablation
> - Model feature importance
> - Stability analysis across partitions
> rather than selecting windows ad-hoc because they improve a single split.

### 3. Atmospheric & Evaporative Conditions (Demonstrably Available Variables):
- `Temp_Mean_30D`, `Temp_Max_30D`: Antecedent thermal indicators from `T2M`.
- `Radiation_30D_Sum`: Cumulative downward shortwave solar radiation from `ALLSKY_SFC_SW_DWN`.
- `Humidity_Mean_30D`: Relative humidity indicator from `RH2M`.
- `WindSpeed_Mean_30D`: Wind speed indicator from `WS2M`.

### 4. Antecedent Groundwater State Indicators:
- `Previous_WatLevel`: Antecedent groundwater-state indicator (pre-existing water table depth).
- `Surf_Elev`: Surface elevation (topographic context).
- `Days_Since_Previous`: Duration of the accumulation interval in days.
- `Season_Code`: Categorical indicator of measurement season (`Spring` vs. `Fall`).
- `Rolling_Mean_3`, `Historical_Mean`: Long-term antecedent groundwater state baselines.

---

## 9. Baseline Models & Performance Criteria

To evaluate whether machine learning models provide genuine predictive value beyond simple relationships:

### Predefined Baselines:
1. **Baseline B0 (Historical Mean Response)**:
   $$\hat{\Delta h}_{\text{B0}} = \frac{1}{N_{\text{train}}} \sum_{i \in \text{train}} \Delta h_i$$
2. **Baseline B1 (Precipitation-Only Linear Model)**:
   $$\hat{\Delta h}_{\text{B1}} = \beta_0 + \beta_1 \cdot \text{Precip\_Interval\_Sum}_i$$
   *(Answers: “Does machine learning outperform a simple linear precipitation exposure model?”)*
3. **Baseline B2 (Seasonal Precipitation Bivariate Model)**:
   $$\hat{\Delta h}_{\text{B2}} = \beta_0 + \beta_1 \cdot \text{Precip\_Interval\_Sum}_i + \beta_2 \cdot \text{Is\_Spring}_i$$
   *(Answers: “Does machine learning outperform rainfall exposure conditioned on seasonal timing?”)*

### Performance Assessment Criteria:
Machine learning models should demonstrate consistent empirical error reduction relative to predefined baselines across the specified validation regimes. Arbitrary hypothesis tests or p-value requirements are not used.

Performance reporting metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Median Absolute Error (Median AE)
- 90th Percentile Absolute Error (P90 AE)
- 95th Percentile Absolute Error (P95 AE)
- Coefficient of Determination ($R^2$, where appropriate)

---

## 10. Feature Ablation Experiment Design (R0 through R5)

A structured ablation experiment will isolate the incremental value of precipitation, climate, and antecedent groundwater state:

- **Configuration R0 (Baseline)**: Historical Mean Response ($\bar{\Delta h}$).
- **Configuration R1 (Precipitation Exposure Only)**: `Precip_Interval_Sum`, `Precip_30D_Sum`, `Precip_90D_Sum`.
- **Configuration R2 (Precipitation + Topography/Geography)**: R1 + `LatDD`, `LongDD`, `Surf_Elev`.
- **Configuration R3 (Precipitation + Atmospheric Conditions)**: R2 + `Temp_Mean_30D`, `Radiation_30D_Sum`, `Humidity_Mean_30D`, `WindSpeed_Mean_30D`.
- **Configuration R4 (Precipitation + Antecedent Groundwater State)**: R3 + `Previous_WatLevel`, `Days_Since_Previous`.
- **Configuration R5 (Full Hydro-Climatic History Configuration)**: R4 + `Rolling_Mean_3`, `Historical_Mean`, `Previous_Level_Change`, `Season_Code`.

This ablation distinguishes rainfall information from antecedent groundwater state memory.

---

## 11. Validation Design & Principles Inheritance

Phase 08.10 inherits the **VALIDATION PRINCIPLES** established in Phases 08.6 and 08.9:

> [!IMPORTANT]
> **Validation Inheritance Scope**:
> Phase 08.10 inherits the validation *methodology* (temporal holdout, 10-seed repeated grouped-well holdout, 5-fold spatial cluster holdout, strict well isolation).
> It does **NOT** automatically inherit:
> - Trained models
> - Calibration quantiles
> - Interval widths
> - Uncertainty parameters
> from Phase 08.9. The uncertainty framework will be adapted to the new recharge-response target after the models are trained.

### Validation Partitions:
1. **Temporal Holdout Evaluation**:
   - Model Training: 2000–2019.
   - Uncertainty Calibration: 2020–2022.
   - Final Evaluation: 2023–2024.
2. **Repeated Grouped-Well Validation (10 Seeds)**:
   - Evaluated across seeds `[42, 101, 202, 303, 404, 505, 606, 707, 808, 909]`.
   - Strict well-level partitioning: 108 Training Wells, 28 Calibration Wells, 34 Test Wells (zero well overlap).
3. **Spatial Cluster Validation (5 Folds)**:
   - Standardized coordinate $k$-means 5 regional clusters across Phelps County.

---

## 12. Candidate Models

Candidate models for supervised warm-start prediction:
1. **Random Forest Regressor**: Handles non-linear interactions without strong parametric assumptions.
2. **XGBoost Regressor**: Gradient boosting with regularized tree splits.
3. **LightGBM Regressor**: Fast histogram-based gradient boosting.
4. **Regularized Linear Baseline (Ridge/Lasso)**: Interpretable benchmark.

*No model ranking or selection is assumed prior to empirical evaluation.*

---

## 13. Explainable AI (XAI) Plan

Reusing the TreeSHAP principles from Phase 08.8:
- **Global Feature Importance**: Evaluated on held-out test wells.
- **Precipitation Dependence Analysis**: Evaluates whether model sensitivity to rainfall shows threshold behavior.
- **Antecedent State Interactions**: Tests whether antecedent groundwater depth dampens or amplifies response.
- **Strict Non-Causal Framing**: SHAP values measure algorithmic feature attribution under empirical data distributions, not physical or hydrological causality.

---

## 14. Physical Sanity Checks

Before accepting model predictions, four physical consistency tests must be audited:
1. **Response Direction Under High Precipitation**: During non-pumping dormant periods (Spring), heavy precipitation should not systematically predict large water table declines.
2. **Antecedent State Consistency**: Observed response relationships with pre-existing depth should be hydrogeologically plausible across shallow vs. deep settings.
3. **Seasonal Sensitivity**: Predictions should reflect empirical seasonal patterns (recharge opportunity vs. drawdown cycles).
4. **Bounded Range Fidelity**: Predicted head response $\hat{\Delta h}$ must remain within physically observed aquifer ranges.

---

## 15. AMDFE Integration Contract

Conceptual integration boundary between the Adaptive Multimodal Data Fusion Engine (AMDFE) and Phase 08.10:

```
[ AMDFE Layer ]
   │  Quality-controlled multimodal weather and environmental inputs
   ▼
[ Phase 08.10: Recharge-Response Engine ]
   │  Warm-Start: Supervised Head Response Δh
   │  Cold-Start: Environmental Recharge Potential Index (RRPI)
   ▼
[ Uncertainty & Applicability Layer (Adapted from Phase 08.9) ]
   │  Prediction intervals, staleness indicators, spatial extrapolation flags
   ▼
[ Phase 08.11: Risk & Decision Support Layer ]
```

### Conceptual Boundaries:
- **AMDFE Reliability**: Data quality and sensor agreement of environmental inputs.
- **Model Uncertainty**: Statistical dispersion of predicted response given inputs.
- **Spatial Applicability**: Proximity to training data distributions.

---

## 16. Proposed Future Directory & Artifact Structure

When implemented, Phase 08.10 outputs will reside in `data/processed/advanced_ml/recharge/`:

```text
data/processed/advanced_ml/recharge/
    methodology/
        implementation_plan.md                # Approved methodology design document
    README.md                                 # Full scientific report and findings
    walkthrough.md                            # Comprehensive execution walkthrough
    recharge_target_definition.csv            # Formal schema and statistics of Δh and RRPI
    recharge_feature_dictionary.csv           # Dictionary of all engineered precipitation lag features
    recharge_model_results.csv                # Overall metrics across models and baselines
    recharge_predictions_temporal.csv         # Temporal holdout predictions
    recharge_predictions_repeated.csv         # 10-seed repeated holdout predictions
    recharge_predictions_spatial.csv          # 5-fold spatial cluster predictions
    recharge_ablation_results.csv             # Metrics across R0–R5 feature blocks
    recharge_physical_validation.csv          # Audit results of physical sanity checks
    recharge_leakage_validation.csv           # Audit log for Gates RCH-LC-01 through RCH-LC-10
    recharge_xai/                             # TreeSHAP importance and dependence CSVs
    recharge_uncertainty/                     # Adapted conformal evaluation CSVs
```

---

## 17. Scientific Limitations

1. **Unmeasured Agricultural Pumping**: Irrigation pumping is not metered at daily resolution in this dataset. Summer drawdowns reflect extraction, confounding precipitation recharge estimation.
2. **Absence of Specific Yield ($S_y$)**: Direct volumetric conversion from head change ($\Delta h$) to recharge depth ($R = S_y \Delta h$) cannot be performed reliably without site-specific pumping test data.
3. **Irregular Measurement Sparsity**: Measurements occur 1–2 times per year, integrating months of complex hydrological fluxes rather than isolating individual recharge events.
4. **Geographic Confinement**: Models and relationships are calibrated specifically to Phelps County, Nebraska, and cannot be transferred outside the local hydrogeological setting without local re-calibration.

---

## 18. Acceptance Criteria Checklist

Phase 08.10 methodology is ready for implementation review only if:
- [x] Unsourced numerical claims removed (specific yield range, pumping well counts).
- [x] Direct PET excluded from primary predictor set; future derivation clearly distinguished as hydrogeologic enhancement.
- [x] `Previous_WatLevel` designated as antecedent groundwater-state indicator (not vadose transit delay).
- [x] Sign convention verification requirement explicitly documented for $\Delta h$ with non-causal interpretation.
- [x] Warm-start (supervised $\Delta h$) vs. cold-start (diagnostic RRPI) explicitly separated; no fabricated targets.
- [x] RRPI defined conceptually based only on demonstrably available environmental variables without ground-truth accuracy claims.
- [x] Arbitrary statistical significance / p-value requirements removed in favor of consistent empirical error reduction.
- [x] Hard-coded sample counts replaced with post-filtering eligibility rule.
- [x] `Precip_Interval_Sum` defined as environmental exposure (not measured recharge).
- [x] Multicollinear precipitation windows handled via feature groups, ablation, and stability analysis.
- [x] Validation principles inherited without claiming automatic inheritance of 08.9 quantiles.
- [x] Recharge claim boundaries strictly defined.
- [x] Zero code executed, zero models trained, and prior locked phases remain 100% untouched.

---

## 19. Explicit Status Statement

```text
================================================================================
PHASE 08.10 STATUS:
METHODOLOGY APPROVED FOR IMPLEMENTATION — FINAL REVISION

Implementation is not yet authorized. The methodology has been revised and is
awaiting final approval.

- No code executed
- No model trained
- No dataset modified
- No Phase 08.4–08.9 artifact modified
- No experimental results generated
================================================================================
```
