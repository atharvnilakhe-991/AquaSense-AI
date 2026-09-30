# Phase 08.10 — Stage 3 Methodology Design Document
## Operational Groundwater Response Prediction, Uncertainty Quantification, XAI Interaction Profiling & Dual-Track Decision Support

> **Phase Status**: STAGE 3 METHODOLOGY DESIGN (REVISED & SCIENTIFICALLY ALIGNED) — AWAITING USER APPROVAL  
> **Proposed Implementation Script (Future)**: `src/08_10_stage3_decision_support.py`  
> **Proposed Output Directory (Future)**: `data/processed/advanced_ml/recharge/stage3/`  
> **Locked Predecessor Artifacts**: Phases 08.4–08.9, Stage 1/1.1, and Stage 2 are strictly **LOCKED** (Read-Only)  
> **Protected Methodology Plan**: `data/processed/advanced_ml/recharge/methodology/implementation_plan.md` is strictly **PROTECTED** (Read-Only)  
> **Authoritative Predecessor Reports**:
> - `data/processed/advanced_ml/recharge/stage1/README.md`
> - `data/processed/advanced_ml/recharge/stage2/README.md`
> - `data/processed/advanced_ml/recharge/stage2/stage2_rrpi_methodology.md`

---

## 1. Stage 3 Objective

The objective of Phase 08.10 Stage 3 is to establish an operational, uncertainty-aware, and explainable decision-support framework that characterizes groundwater-level response ($\Delta h$) and relative hydro-climatic potential across Phelps County, Nebraska, without asserting physical recharge flux or causal infiltration rates.

Stage 3 operationalizes the empirical findings of Stage 2 into a dual-track decision-support system:
1. **Warm-Start Track ($N = 3,674$)**: Supervised empirical groundwater response ($\Delta h$) prediction with:
   - Split-conformal uncertainty quantification (empirically evaluated prediction interval coverage),
   - Observation staleness and feature applicability bounding,
   - TreeSHAP feature interaction profiling (investigating model-estimated interactions between antecedent groundwater state and precipitation exposure), and
   - Standardized what-if scenario stress-testing with explicit out-of-distribution (OOD) support checking (evaluating model sensitivity under synthetic hydro-climatic sequences).
2. **Cold-Start Track ($N = 170$)**: Uncalibrated diagnostic relative hydro-climatic potential index (RRPI) combined with spatial extrapolation distance to monitored infrastructure.
3. **Proposed Decision-Support Framework (Conceptual / Not Yet Validated)**: Multi-criteria classification of monitoring locations into actionable monitoring and recharge-support tiers, strictly distinguishing empirical response from measured recharge and requiring empirical threshold calibration prior to operational adoption.

---

## 2. Relationship to Stage 1 and Stage 2

Stage 3 builds directly upon the audited foundation of Stages 1 and 2 while avoiding redundant experimentation:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1 & 1.1: DATA AUDIT & PHYSICAL BOUNDARIES (LOCKED)                               │
│ • Validated 3,844 physical monitoring records across 170 wells (2000–2024).            │
│ • Enforced strict D-1 prior-day weather cutoff across 100% of observations.            │
│ • Formulated empirical Δh target = Previous_WatLevel - WatLevel.                       │
│ • Separated warm-start (N=3,674) from cold-start (N=170, Δh = NaN).                   │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
┌──────────────────────────────────────────▼─────────────────────────────────────────────┐
│ STAGE 2: ABLATION, BENCHMARKING & RRPI CONSTRUCTION (LOCKED)                          │
│ • Multi-scale precipitation ablation (P7 to P180): P90 achieved lowest MAE (1.1945 ft).│
│ • Feature-family ablation: Set 4 (G+T+H+P_windows+Interval, 23 feats) selected.       │
│ • Disentangled gap duration vs. interval precipitation (r = 0.9089 collinearity).      │
│ • Multi-regime validation: Temporal (MAE 1.156–1.206 ft), Repeated Grouped, Spatial.  │
│ • Extreme Δh robustness audit (24 >3*IQR observations: 18 Train, 6 Val, 0 Test).       │
│ • Formulated approved 8-feature diagnostic RRPI on 2000–2019 reference baseline.       │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
┌──────────────────────────────────────────▼─────────────────────────────────────────────┐
│ STAGE 3: OPERATIONAL PREDICTION, UNCERTAINTY, XAI & DECISION SUPPORT (PROPOSED)        │
│ • Reuses exact Stage 2 Set 4 feature configuration (23 features) — NO redundant search.│
│ • Adapts split-conformal prediction intervals to the empirical Δh target.              │
│ • TreeSHAP attribution and model-estimated depth × precipitation interaction.          │
│ • Observation staleness and feature applicability bounding.                            │
│ • Scenario-based sensitivity stress-testing with historical support/OOD audit.         │
│ • Dual-track decision-support matrix with empirical threshold calibration policy.      │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

Stage 3 does **NOT**:
- Re-run feature-selection or ablation sweeps.
- Re-tune ML hyperparameters.
- Recalculate RRPI reference population parameters.
- Pre-select a "best model" prior to comprehensive validation.

---

## 3. Warm-Start Target

$$\Delta h_i = \text{Previous\_WatLevel}_i - \text{WatLevel}_i \quad [\text{ft}]$$
- **Eligibility**: Evaluated strictly on $N = 3,674$ warm-start observations ($\text{Previous\_Observation\_Count} \ge 1$).
- **Sign Convention**:
  - $\Delta h > 0$: Water table shallower / rising (reduced depth below ground surface).
  - $\Delta h < 0$: Water table deeper / falling (increased depth below ground surface).
- **Physical Boundary**: $\Delta h$ is an **empirical groundwater-level response** resulting from all simultaneous hydrological processes (precipitation infiltration, evapotranspiration, lateral gradient flow, unmetered agricultural pumping extraction, and recovery). It is **NOT** measured recharge, recharge flux, or recharge volume. Zero direct recharge targets are fabricated.

---

## 4. Cold-Start Role

- **Eligibility**: Evaluated on $N = 170$ cold-start observations ($\text{Previous\_Observation\_Count} == 0$).
- **Strict Target Boundary**: Cold-start observations have **NO SUPERVISED TARGET** ($\Delta h = \text{NaN}$). No pseudo-labels, imputed values, or synthetic history features are ever created.
- **Diagnostic Role**: Evaluated strictly through the approved 8-feature **Recharge Response-Potential Index (RRPI)** combined with spatial proximity to monitored infrastructure.
- **Role Distinction**:
  - **Cold-Start Track**: Uncalibrated hydro-climatic diagnostic & data-collection prioritization.
  - **Warm-Start Track**: Supervised empirical groundwater-response prediction.

---

## 5. Correct Feature Architecture (Stage 2 Set 4 Reconciliation)

Stage 3 inherits the exact feature configuration of **Stage 2 Set 4** (`SET_4_Precip`), which contains **exactly 23 features**:

```text
SET_4 Features (Total: 23)
├── Group G (Geography, 2): LatDD, LongDD
├── Group T (Topography, 1): Surf_Elev
├── Group H (Prior Groundwater State, 13):
│   ├── Previous_WatLevel
│   ├── Previous2_WatLevel
│   ├── Days_Since_Previous
│   ├── Years_Since_Previous
│   ├── Previous_Observation_Count
│   ├── Rolling_Mean_3
│   ├── Rolling_Std_3
│   ├── Previous_Level_Change
│   ├── Recent_Trend
│   ├── Historical_Mean
│   ├── Historical_Std
│   ├── Historical_Min
│   └── Historical_Max
├── Group P_WINDOWS (Precipitation Windows, 6):
│   ├── Precip_7D_Sum
│   ├── Precip_14D_Sum
│   ├── Precip_30D_Sum
│   ├── Precip_60D_Sum
│   ├── Precip_90D_Sum
│   └── Precip_180D_Sum
└── Group P_INTERVAL (Interval Precipitation, 1):
    └── Precip_Interval_Sum
```

### Complete Candidate Feature Space Reconciliation:
The total candidate feature space in Stage 2 comprised 31 features (`ALL_CANDIDATE_FEATURES`):
- **Set 4 Features (23)**: Defined above.
- **Group P_EVENTS (3)**: `Rain_Days_30D`, `Max_Daily_Precip_30D`, `Dry_Spell_Days`. (Excluded from Set 4; evaluated in Set 5 / RRPI).
- **Group A (Atmospheric Context, 5)**: `Temp_Mean_30D`, `Temp_Max_30D`, `Radiation_30D_Sum`, `Humidity_Mean_30D`, `WindSpeed_Mean_30D`. (Excluded from Set 4 due to Stage 2 ablation evidence showing temporal overfitting across multi-month gaps; evaluated in Set 5 / RRPI).
- **Observation_Density**: Present in the raw Stage 1 dataset, but explicitly excluded from Stage 2 modeling feature groups due to near-perfect collinearity with `Years_Since_Previous` and `Previous_Observation_Count`.

---

## 6. Prediction-Time Information Boundary

Strict information boundaries are maintained to prevent data leakage:
1. **Weather Information Cutoff**: Daily NASA POWER weather data must strictly satisfy $\max(\text{Date}_{\text{weather}}) \le \text{DateMsr}_i - 1\text{ day}$ across 100% of observations. Same-day weather ($D=0$) is strictly prohibited.
2. **Groundwater Temporal Precedence**: All prior water level and historical metrics are computed strictly using observations dated $< \text{DateMsr}_i$.
3. **Prohibited Variables**:
   - Current `WatLevel` ($y_i$) and target $\Delta h$ are strictly excluded from predictors.
   - Static interpolation raster `TIFF_Value` is strictly excluded.
   - Future groundwater observations or future weather are strictly excluded.
4. **Duplicate-Date Sequencing**: Records with identical well IDs and measurement dates share antecedent state and weather cutoffs; they are never treated as sequential transitions.

---

## 7. Validation Architecture

Stage 3 maintains full continuity with established multi-regime validation principles:
1. **Chronological Temporal Holdout**:
   - Training Partition: 2000–2019 ($N_{\text{train}} = 3,086$)
   - Uncertainty Calibration Partition: 2020–2022 ($N_{\text{cal}} = 408$)
   - Final Out-of-Sample Evaluation: 2023–2024 ($N_{\text{test}} = 180$)
2. **Repeated Grouped-Well Validation**:
   - 10 Random Holdout Seeds: `[42, 101, 202, 303, 404, 505, 606, 707, 808, 909]`
   - 20% well holdout per seed (136 Train / 34 Test wells, zero well overlap)
3. **Spatial Cluster Validation**:
   - 5 spatial folds via standardized coordinate KMeans across Phelps County
4. **Applicability-Disaggregated Reporting**:
   - Performance and interval width reported separately across gap duration cohorts:
     - Semi-annual ($\le 200$ days)
     - Annual ($201–400$ days)
     - Stale ($> 400$ days)

---

## 8. Candidate Model Architecture & Selection Policy

### Candidate Models:
To maintain consistency with Stage 2, candidate supervised models remain standardized:
1. **Random Forest Regressor**: `n_estimators=200`, `max_depth=12`, `min_samples_split=5`, `min_samples_leaf=2`, `random_state=42`.
2. **XGBoost Regressor**: `n_estimators=200`, `max_depth=5`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`.
3. **LightGBM Regressor**: `n_estimators=200`, `max_depth=5`, `learning_rate=0.05`, `num_leaves=31`, `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`.

### Model Selection Policy:
- Random Forest, XGBoost, and LightGBM remain candidate models throughout Stage 3.
- Models will be evaluated across all three validation regimes (temporal, repeated grouped-well, spatial).
- **No model is pre-selected as scientifically superior** based solely on single-split Stage 2 results.
- If a single model is selected as the primary analysis engine for downstream SHAP or scenario evaluations, it will be designated strictly as:
  > *"analysis engine selected for interpretability and scenario consistency under empirical evaluation,"*
  rather than claiming universal hydrogeological superiority.
- Zero uncontrolled hyperparameter sweeps will be conducted.

---

## 9. Uncertainty Quantification (Conformal Prediction)

Split-conformal prediction intervals are constructed for the empirical $\Delta h$ target:
1. **Calibration Set**: Chronological validation partition 2020–2022 ($N_{\text{cal}} = 408$).
2. **Nonconformity Score**: Absolute residual $s_i = |\Delta h_i - \hat{\Delta h}_i|$.
3. **Conformal Cutoff**:
   $$\hat{q}_{1-\alpha} = \text{Quantile}\left(\{s_i\}_{i \in \text{Cal}}, \frac{\lceil (n_{\text{cal}} + 1)(1 - \alpha) \rceil}{n_{\text{cal}}}\right)$$
4. **Evaluation Partition**: Chronological temporal test partition 2023–2024 ($N_{\text{test}} = 180$).
5. **Coverage Levels**: Calibrate both nominal 90% ($\alpha = 0.10$) and 95% ($\alpha = 0.05$) intervals.
6. **Empirical Evaluation Requirement**: Empirical test coverage must be calculated and verified rather than assumed ($1 - \alpha \ge 0.88$ empirical threshold).

> [!IMPORTANT]
> **Interpretation Boundary**: Conformal prediction intervals quantify the **statistical predictive uncertainty / model error** under empirical observation distributions. They do **NOT** represent physical aquifer uncertainty, recharge uncertainty, groundwater-flow uncertainty, or uncertainty in hydrogeological parameters (e.g., transmissivity, specific yield).

---

## 10. Observation Staleness Policy

- In Phelps County, monitoring is irregular, predominantly semi-annual and annual.
- A threshold of $\text{Days\_Since\_Previous} > 365\text{ days}$ is adopted as a **predefined operational diagnostic flag**, not a universal hydrogeological cutoff.
- Stage 3 will evaluate model errors and interval widths across gap duration cohorts ($\le 200\text{ d}$, $201–400\text{ d}$, $> 400\text{ d}$) to provide empirical evidence for gap-dependent uncertainty degradation.

---

## 11. Spatial Applicability & Extrapolation Policy

- Spatial applicability is assessed for all predictions:
  - Warm-start wells evaluate feature Mahalanobis distance to training cluster centroids.
  - Cold-start wells compute Euclidean distance to the nearest monitored well location.
- Proposed spatial distance flags (e.g., $5\text{ km}$) are treated strictly as **candidate diagnostic boundaries requiring empirical calibration**, rather than validated hydrogeological boundaries.

---

## 12. RRPI Role & Scientific Boundaries

The Recharge Response-Potential Index (RRPI) preserves the approved Stage 2 formulation exactly:
- **8 Features**:
  - Moisture Delivery: `Precip_30D_Sum`, `Precip_180D_Sum`, `Rain_Days_30D`, `Humidity_Mean_30D`
  - Atmospheric Demand: `Temp_Mean_30D`, `Radiation_30D_Sum`, `WindSpeed_Mean_30D`, `Dry_Spell_Days`
- **Formula**:
  $$S_{\text{moist}} = \text{mean}(\widetilde{\mathbf{x}}_{\text{moist}}), \quad S_{\text{demand}} = \text{mean}(\widetilde{\mathbf{x}}_{\text{demand}})$$
  $$\text{RRPI} = 50 \times \left(1 + S_{\text{moist}} - S_{\text{demand}}\right) \quad \in [0, 100]$$
- **Calibration**: Reference population min/max fitted strictly on 2000–2019 baseline (`stage2_rrpi_reference_population.csv`).
- **Interpretation**: RRPI is an **uncalibrated, relative, hydro-climatic diagnostic index**. It reflects *"favorable hydro-climatic conditions for recharge-related assessment"*.
- **Prohibitions**: RRPI is **NOT** measured recharge, recharge flux, recharge volume, physical recharge probability, or calibrated infiltration rate.

---

## 13. SHAP Interpretation Boundaries (Model-Level Structure Only)

TreeSHAP attribution will be evaluated on the held-out temporal test set ($N = 180$):
1. **Global Feature Attribution**: Ranks features by mean absolute SHAP contribution to $\hat{\Delta h}$.
2. **Model-Estimated Interaction Profiling**:
   - Examines the model-estimated interaction between antecedent groundwater state (`Previous_WatLevel`) and precipitation exposure (`Precip_90D_Sum` and `Precip_Interval_Sum`).
   - Identifies non-linear patterns learned by the tree ensemble.
3. **Strict Non-Causal Boundary**:
   - The analysis describes **model-estimated relationships** only.
   - SHAP values measure algorithmic feature contributions under empirical data distributions; they do **NOT** prove physical aquifer buffering, recharge mechanisms, hydrodynamic processes, or causal relationships.

---

## 14. Scenario Stress-Testing Framework (Validity & Support Auditing)

To evaluate model sensitivity to synthetic weather sequences without presenting outputs as weather forecasts:

### A. Plausibility & Internal Consistency:
Synthetic scenarios must not create impossible variable combinations (e.g., high precipitation paired with high dry spell duration). All 23 feature values must be physically consistent.

### B. Empirical Support & Out-of-Distribution (OOD) Auditing:
Every scenario input vector will be checked against the historical training distribution (2000–2019):
- **Within Observed Support**: All feature values fall within historical $[\min, \max]$ range and within 90% multivariate confidence ellipsoid.
- **Near-Boundary Support**: Values between 90th and 99th historical percentiles.
- **Outside Observed Support (OOD)**: Any value exceeding historical empirical extrema. Predictions falling into this category must be **explicitly flagged as extrapolative**.

### C. Standardized Synthetic Scenarios:
1. **Scenario S1 (Normal Seasonal Baseline)**: Median seasonal precipitation and median atmospheric exposure.
2. **Scenario S2 (Severe Dry Spell Sensitivity)**: Zero precipitation over 90 days ($P_{90} = 0\text{ mm}$), historical 90th percentile temperature and solar radiation.
3. **Scenario S3 (Moderate Precipitation Pulse)**: Historical 75th percentile precipitation exposure.
4. **Scenario S4 (Extreme Precipitation Pulse)**: Historical 95th percentile precipitation exposure.

### D. Framing:
Scenario outputs are explicitly designated as **"model-based sensitivity analysis"** or **"what-if hydro-climatic stress testing"**. They are **NOT** actual future groundwater forecasts or hydrological simulations.

---

## 15. Proposed Decision-Support Framework — Not Yet Validated

The decision-support framework organizes monitoring locations into conceptual operational tiers:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ PROPOSED DECISION-SUPPORT FRAMEWORK (CONCEPTUAL / NOT YET VALIDATED)                   │
├─────────┬───────────────────────────────┬──────────────────────────────────────────────┤
│ Tier    │ Conceptual Qualification      │ Operational Purpose                          │
├─────────┼───────────────────────────────┼──────────────────────────────────────────────┤
│ Tier 1  │ Warm-start, positive Δh rise, │ Identifies wells exhibiting empirical rise   │
│         │ narrow prediction interval    │ under favorable antecedent exposure.         │
├─────────┼───────────────────────────────┼──────────────────────────────────────────────┤
│ Tier 2  │ Warm-start, near-zero Δh,     │ Identifies stable water table conditions,    │
│         │ narrow prediction interval    │ buffered by deep table or moderate exposure. │
├─────────┼───────────────────────────────┼──────────────────────────────────────────────┤
│ Tier 3  │ Warm-start, negative Δh,      │ Identifies empirical water table decline     │
│         │ narrow prediction interval    │ under dry or extraction conditions.          │
├─────────┼───────────────────────────────┼──────────────────────────────────────────────┤
│ Tier 4  │ Warm-start, wide interval OR  │ Flagged for field measurement priority due   │
│         │ stale observation gap         │ to high statistical predictive uncertainty.  │
├─────────┼───────────────────────────────┼──────────────────────────────────────────────┤
│ Tier 5  │ Cold-start, high RRPI,        │ High diagnostic hydro-climatic favorability; │
│         │ near monitored network        │ candidate for new physical monitoring.       │
├─────────┼───────────────────────────────┼──────────────────────────────────────────────┤
│ Tier 6  │ Cold-start, low RRPI OR       │ Low diagnostic favorability or high spatial  │
│         │ high spatial distance         │ extrapolation distance; secondary priority.  │
└─────────┴───────────────────────────────┴──────────────────────────────────────────────┘
```

> [!WARNING]
> This framework is **conceptual and not yet validated**. It serves as an operational decision-support tool, **NOT** a physical recharge metering system.

---

## 16. Threshold Calibration Policy

All numerical thresholds proposed in the decision matrix are designated as **candidate thresholds requiring empirical calibration**:
1. **RRPI Boundary (e.g., 50)**: Must be calibrated against empirical distribution percentiles (e.g., median or interquartile bounds) of the reference population rather than assumed a priori.
2. **Spatial Distance Boundary (e.g., 5 km)**: Must be calibrated against spatial cluster semivariograms and nearest-neighbor distance distributions in Phelps County.
3. **Uncertainty Interval Width (e.g., $W_{90} = 3.0\text{ ft}$)**: Must be calibrated against the empirical distribution of conformal interval widths on the calibration set.
4. **Staleness Boundary (e.g., 365 days)**: Retained as a provisional diagnostic flag representing the modal annual monitoring cadence, but evaluated empirically across gap duration cohorts.

---

## 17. Proposed Novelty & Contributions Beyond Stage 2

| Dimension | Stage 2 Status | Proposed Stage 3 Contribution | Scientific Benefit |
| :--- | :--- | :--- | :--- |
| **Uncertainty Quantification** | None (point predictions only) | Split-conformal prediction intervals for empirical $\Delta h$ (90% & 95%) | Provides statistically valid prediction intervals for water managers |
| **Observation Staleness** | Collinearity acknowledged ($r=0.91$) | Bounded applicability flags conditioned on gap length and Mahalanobis distance | Distinguishes reliable predictions from stale extrapolations |
| **Model Interpretability** | Baseline feature importances | TreeSHAP attribution + model-estimated depth $\times$ precipitation interaction | Explains learned non-linear aquifer relationships |
| **Sensitivity Stress-Testing** | None | 4-scenario synthetic weather exposure testing with historical support checks | Evaluates model behavior under dry vs. wet conditions |
| **Decision-Support Synthesis** | Disconnected warm ML & cold RRPI | Integrated 6-tier operational decision matrix with empirical calibration policy | Provides actionable guidance for monitoring network prioritization |

*No claims of "first-ever", "revolutionary", or "accurate recharge measurement" are made.*

---

## 18. 14-Gate Anti-Circularity & Leakage Audit Matrix

| Gate ID | Gate Name | Verification Check | PASS Criteria | FAIL Criteria |
| :---: | :--- | :--- | :--- | :--- |
| **RCH3-LC-01** | Prior-Day Weather Cutoff | Timestamps of weather records | $\max(\text{Date}_{\text{weather}}) \le \text{DateMsr}_i - 1\text{ day}$ for 100% of features | Same-day or future weather used |
| **RCH3-LC-02** | Groundwater Temporal Precedence | Observation timestamps | $\text{DateMsr}_{\text{prior}} < \text{DateMsr}_i$ for all warm-start rows | Prior date $\ge$ current date |
| **RCH3-LC-03** | Target Segregation (`WatLevel`) | Predictor matrix composition | Current `WatLevel` strictly absent from predictors | Current `WatLevel` in features |
| **RCH3-LC-04** | Target Segregation ($\Delta h$) | Predictor matrix composition | $\Delta h$ strictly absent from predictors | $\Delta h$ in features |
| **RCH3-LC-05** | Static Raster Exclusion | Feature dictionary | `TIFF_Value` strictly absent from dataset and models | `TIFF_Value` present in features |
| **RCH3-LC-06** | Grouped Well Separation | Well IDs across partitions | Zero well overlap between train and test (10 seeds) | Well ID in both train and test |
| **RCH3-LC-07** | Spatial Fold Isolation | Spatial cluster boundaries | Zero well overlap across all 5 spatial cluster folds | Well ID in multiple folds |
| **RCH3-LC-08** | Transformation Leakage Isolation | Preprocessing pipelines | All scalers, imputers, and quantiles fit strictly on train | Test data used to fit transforms |
| **RCH3-LC-09** | Cold-Start Target Integrity | Target array for cold wells | All 170 cold-start records have 100% NaN for $\Delta h$ | Non-NaN target assigned to cold well |
| **RCH3-LC-10** | Conformal Calibration Independence | Split timestamps | $\max(\text{Date}_{\text{cal}}) < \min(\text{Date}_{\text{test}})$ (2020–2022 vs. 2023–2024) | Calibration/test overlap |
| **RCH3-LC-11** | RRPI Reference Population Fidelity | Parameter source | RRPI parameters strictly from 2000–2019 baseline | Post-2019 data in RRPI reference |
| **RCH3-LC-12** | Cold-Start History Fabrication Check | Cold-start feature matrix | Zero synthetic/imputed history assigned to cold wells | History features created for cold wells |
| **RCH3-LC-13** | Duplicate-Date Sequencing Check | Same-date monitoring records | Same-date records never treated as sequential transitions | Sequential $\Delta h$ computed on same date |
| **RCH3-LC-14** | Baseline Phase Preservation | Pre/post MD5 checksums | All locked files from Phases 08.4–08.9, Stage 1, Stage 2 identical | Any locked artifact modified |

---

## 19. Claim Boundaries

```text
┌──────────────────────────────────────────────┬──────────────────────────────────────────────┐
│ PERMITTED SCIENTIFIC TERMINOLOGY             │ PROHIBITED CLAIMS & TERMINOLOGY              │
├──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ • Empirical groundwater head response (Δh)   │ • Measured recharge or recharge amount       │
│ • Net water-level change between observations│ • Recharge flux in mm/year or inches/year    │
│ • Observation-gap precipitation exposure     │ • Aquifer recharge volume                    │
│ • Relative hydro-climatic diagnostic (RRPI)  │ • Physical infiltration rate                 │
│ • Favorable hydro-climatic conditions        │ • Infiltration travel time                   │
│ • Statistical prediction interval            │ • Physical aquifer storage parameter (Sy)    │
│ • Model-estimated feature attribution (SHAP) │ • Causal hydrodynamic flow attribution       │
│ • What-if hydro-climatic stress testing      │ • Physical recharge prediction               │
│ • Candidate decision-support tier            │ • Universal hydrological recharge model      │
└──────────────────────────────────────────────┴──────────────────────────────────────────────┘
```

---

## 20. Proposed Stage 3 Computational Outputs

| # | Artifact Filename | Type | Content & Purpose | Split / Cohort |
| :---: | :--- | :---: | :--- | :--- |
| 1 | `stage3_response_predictions_temporal.csv` | CSV | $\Delta h$, $\hat{\Delta h}$, residuals, $90\%$ conformal bounds | Temporal Test ($N=180$) |
| 2 | `stage3_response_predictions_repeated.csv` | CSV | $\hat{\Delta h}$ across 10 repeated grouped seeds | 10 seeds ($20\%$ holdout) |
| 3 | `stage3_response_predictions_spatial.csv` | CSV | $\hat{\Delta h}$ across 5 spatial cluster folds | 5 spatial cluster folds |
| 4 | `stage3_uncertainty_calibration.csv` | CSV | Nonconformity scores, $\hat{q}_{0.90}$, $\hat{q}_{0.95}$ | Calibration set (2020–2022, $N=408$) |
| 5 | `stage3_uncertainty_test_evaluation.csv` | CSV | Coverage, interval width, sharpness by gap cohort | Temporal Test ($N=180$) |
| 6 | `stage3_shap_global_importance.csv` | CSV | Mean absolute SHAP values per feature | Temporal Test ($N=180$) |
| 7 | `stage3_shap_interaction_profiles.csv` | CSV | Model-estimated SHAP interaction values | Temporal Test ($N=180$) |
| 8 | `stage3_scenario_stress_testing.csv` | CSV | Predicted $\hat{\Delta h}$ and support/OOD flags across 4 scenarios | All 170 wells, latest state |
| 9 | `stage3_decision_support_matrix.csv` | CSV | Classification into Decision Tiers 1–6 | All 170 wells (warm and cold) |
| 10 | `stage3_leakage_audit.csv` | CSV | PASS/FAIL audit across 14 leakage gates | Entire modeling pipeline |
| 11 | `stage3_model_summary.csv` | CSV | Summary of metrics across all models and regimes | Aggregated results |
| 12 | `README.md` | Doc | Comprehensive scientific report and findings | All Stage 3 outputs |
| 13 | `walkthrough.md` | Doc | Execution walkthrough and verification audit | All Stage 3 outputs |

---

## 21. Open Questions Requiring Approval

1. **Threshold Calibration Strategy**: Confirm approval to adopt empirical percentile-based calibration for RRPI and interval-width thresholds during implementation, rather than using fixed ad-hoc cutoffs?
2. **Analysis Engine for SHAP**: Confirm approval to select the primary analysis engine for SHAP and scenario stress-testing strictly after evaluating complete multi-regime validation tables?
3. **Scenario Out-of-Distribution Policy**: Confirm that any synthetic scenario input falling outside historical empirical support will be explicitly flagged as `OOD_Extrapolative` in `stage3_scenario_stress_testing.csv`?

---

## 22. Final Acceptance Checklist

- [x] Feature count reconciled against actual Stage 2 implementation (exactly 23 features in Set 4).
- [x] No unsupported feature added.
- [x] Warm-start target explicitly defined ($\Delta h = \text{Previous\_WatLevel} - \text{WatLevel}$).
- [x] Cold-start target not fabricated (strictly NaN).
- [x] Candidate models remain RF / XGB / LGBM.
- [x] No model preselected as scientifically superior.
- [x] Conformal intervals treated as statistical prediction intervals.
- [x] Coverage will be empirically evaluated ($1-\alpha \ge 0.88$ check).
- [x] Staleness threshold labeled provisional.
- [x] Spatial thresholds labeled provisional.
- [x] RRPI threshold labeled provisional.
- [x] Decision matrix labeled conceptual/unvalidated.
- [x] SHAP interaction claims limited to model-level interpretation.
- [x] No causal hydrogeological interpretation.
- [x] Scenario stress testing includes OOD/support checks.
- [x] Scenario results labeled sensitivity analysis.
- [x] No direct recharge ground truth fabricated.
- [x] No physical recharge flux claims.
- [x] Leakage gates defined (14 explicit gates).
- [x] Stage 2 remains untouched.
- [x] Original `implementation_plan.md` remains protected.
- [x] No Stage 3 computation performed.
