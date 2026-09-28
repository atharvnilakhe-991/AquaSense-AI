# Phase 08.10 — Stage 3 Implementation Report
## Operational Groundwater Response Prediction, Split-Conformal Uncertainty Quantification, TreeSHAP Interaction Profiling & Dual-Track Decision Support

**Phase Status**: STAGE 3 COMPLETE — DECISION SUPPORT & OPERATIONAL MODELLING VERIFIED  
**Primary Dataset**: `data/processed/advanced_ml/recharge/stage1/groundwater_recharge_response_stage1.csv`  
**Execution Script**: `src/08_10_stage3_decision_support.py`  
**Authoritative Methodology**: `data/processed/advanced_ml/recharge/methodology/stage3_methodology_plan.md`

---

## 1. Executive Summary & Operational Objectives

Stage 3 operationalizes the empirical groundwater response findings of Stage 2 into an **uncertainty-quantified, explainable, and applicability-bounded decision-support framework** across Phelps County, Nebraska:

1. **Reconciled Feature Architecture (Set 4, Exactly 23 Features)**:
   - Inherits the exact Stage 2 Set 4 specification: Geography (2), Topography (1), Prior Groundwater State (13), Rolling Precipitation Windows (6), and Interval Precipitation (1).
2. **Multi-Regime Validation Consistency**:
   - Evaluated across Chronological Temporal (2023–2024, N=180), Repeated Grouped-Well (10 seeds, 20% well holdout), and Spatial Cluster (5 folds).
   - Random Forest, XGBoost, and LightGBM evaluated; LightGBM selected as primary analysis engine for downstream interpretability and scenario sensitivity under empirical evaluation.
3. **Split-Conformal Uncertainty Quantification**:
   - Conformalized prediction intervals calibrated on 2020–2022 ($N=408$) achieved **98.89% empirical conformal coverage under the evaluated temporal split** on the 2023–2024 temporal test set for nominal 90% target (conformal cutoff $q_{90} = 4.7838$ ft).
   - Validated across semi-annual ($\le 200$ d), annual ($201–400$ d), and stale ($> 400$ d) observation gap cohorts.
4. **TreeSHAP Attribution & Model-Estimated Interactions**:
   - Global feature attribution identifies `Previous_Level_Change`, `Days_Since_Previous`, and `Precip_Interval_Sum` as top predictive contributors.
   - Exact 2-way TreeSHAP interaction profiling characterizes model-estimated interaction between antecedent water table depth and precipitation exposure without asserting physical causality.
5. **What-If Scenario Sensitivity Stress-Testing**:
   - Evaluated 4 standardized synthetic weather sequences across all 170 wells at their latest observed state with historical support/OOD auditing.
   - Characterizes empirical sensitivity across shallow ($< 30$ ft), intermediate ($30–100$ ft), and deep ($> 100$ ft) aquifer cohorts.
6. **Dual-Track Decision Support Matrix (Candidate Tiers)**:
   - Synthesizes warm-start predictions ($\hat{\Delta h}$, $W_{90}$, staleness flag) and cold-start diagnostics (RRPI, distance to monitored well) across all 170 wells into 6 candidate operational tiers.
7. **14-Gate Anti-Circularity & Leakage Audit Matrix**:
   - All 14 leakage gates report **PASS** with zero violations.

---

## 2. Master Model Validation Performance

| Validation Regime | Model | Sample Count | MAE (ft) | RMSE (ft) | R² | Median AE (ft) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Temporal Holdout (2023–2024)** | Random Forest | 180 | 1.2126 | 1.7136 | 0.2291 | 0.8326 |
| **Temporal Holdout (2023–2024)** | XGBoost | 180 | 1.2112 | 1.6922 | 0.2483 | 0.8361 |
| **Temporal Holdout (2023–2024)** | LightGBM | 180 | 1.1435 | 1.5854 | 0.3401 | 0.8793 |
| **Repeated Grouped-Well (10 Seeds)** | Random Forest | 748 | 1.0444 ± 0.0931 | 1.8288 ± 0.3541 | 0.5486 ± 0.0913 | — |
| **Repeated Grouped-Well (10 Seeds)** | XGBoost | 748 | 1.0983 ± 0.0898 | 1.8713 ± 0.3566 | 0.5275 ± 0.0926 | — |
| **Repeated Grouped-Well (10 Seeds)** | LightGBM | 748 | 1.1095 ± 0.0930 | 1.8824 ± 0.3547 | 0.5216 ± 0.0925 | — |
| **Spatial Cluster (5 Folds)** | Random Forest | 734 | 1.2511 ± 0.1258 | 1.9263 ± 0.3271 | 0.4430 ± 0.0700 | — |
| **Spatial Cluster (5 Folds)** | XGBoost | 734 | 1.2804 ± 0.1055 | 1.9706 ± 0.2958 | 0.4141 ± 0.0783 | — |
| **Spatial Cluster (5 Folds)** | LightGBM | 734 | 1.2797 ± 0.1420 | 1.9524 ± 0.3682 | 0.4284 ± 0.0941 | — |

---

## 3. Split-Conformal Uncertainty Calibration & Evaluation

- **Calibration Set (2020–2022, N=408)**:
  - LightGBM Conformal Cutoff $q_{90} = 4.7838$ ft (Prediction Interval Width $W_{90} = 9.5677$ ft)
  - LightGBM Conformal Cutoff $q_{95} = 6.1980$ ft (Prediction Interval Width $W_{95} = 12.3961$ ft)
- **Empirical Coverage on Temporal Test Set (2023–2024, N=180, LightGBM Engine)**:
  - Overall Temporal Test (N=180): **98.89%** empirical conformal coverage under the evaluated temporal split (nominal 90% target, PASS, MAE = 1.1435 ft).
  - Short Gap ($\le 200$ d, N=62): 98.39% empirical coverage, MAE = 1.1318 ft.
  - Annual Gap ($201–400$ d, N=86): 98.84% empirical coverage, MAE = 1.0804 ft.
  - Stale Gap ($> 400$ d, N=32): 100.00% empirical coverage, MAE = 1.3357 ft.

> [!NOTE]
> Conformal prediction intervals quantify the statistical predictive dispersion of model error under empirical observation distributions. They provide empirical conformal coverage under the evaluated temporal split and do **NOT** represent physical aquifer storage or transmissivity parameter uncertainty, nor do they claim unconditional future coverage guarantees.

---

## 4. TreeSHAP Attribution & Interaction Profiling

### Global Top 5 Predictive Features:
1. `Previous_Level_Change` (0.5601 ft): Prior local dynamic head trajectory.
2. `Days_Since_Previous` (0.4358 ft): Elapsed monitoring duration.
3. `Precip_Interval_Sum` (0.3736 ft): Cumulative rainfall exposure over the monitoring gap.
4. `Precip_60D_Sum` (0.2836 ft): Intermediate 60-day rainfall exposure.
5. `Recent_Trend` (0.1948 ft): Annualized prior rate of change.

### Model-Estimated Two-Way Interaction Profiling:
- **`Previous_WatLevel` × `Precip_Interval_Sum`**: Mean |SHAP_inter| = 0.040227 ft.
- **`Days_Since_Previous` × `Precip_Interval_Sum`**: Mean |SHAP_inter| = 0.101283 ft.
- **`Previous_WatLevel` × `Precip_90D_Sum`**: Mean |SHAP_inter| = 0.004246 ft.

*Scientific Boundary: Interaction values measure algorithmic structure learned by the predictive model; they do NOT prove physical causality or hydrodynamic mechanisms.*

---

## 5. What-If Scenario Stress-Testing Summary (170 Wells)

Scenarios represent **model-based what-if / sensitivity simulations under synthetic antecedent precipitation sequences**, NOT physical recharge simulations, guaranteed groundwater forecasts, or causal hydrological predictions.

Across 4 standardized synthetic weather sequences applied to each well's latest observed state:
- **S1 (Normal Seasonal Baseline)**: Historical median seasonal precipitation exposure. Shallow wells mean $\Delta h = +0.15$ ft; Deep wells mean $\Delta h = -0.12$ ft.
- **S2 (Severe Dry Spell Stress)**: Zero precipitation exposure across all antecedent windows ($0.0$ mm). Water table response shifts downward across all cohorts (Shallow mean $\Delta h = -0.62$ ft; Deep mean $\Delta h = -0.48$ ft).
- **S3 (Moderate Precipitation Pulse)**: Historical 75th percentile precipitation exposure. Positive response shift (Shallow mean $\Delta h = +0.84$ ft; Deep mean $\Delta h = +0.08$ ft).
- **S4 (Extreme Precipitation Pulse)**: Historical 95th percentile precipitation exposure. Enhanced response in shallow settings (Shallow mean $\Delta h = +1.42$ ft; Deep mean $\Delta h = +0.26$ ft).

> [!NOTE]
> **Out-of-Distribution (OOD) Support Audit & Methodological Limitation**:
> - OOD detection is based on univariate feature support checks against historical training ranges (2000–2019) and does **not** establish full multivariate distributional support.
> - **Scenario S2**: Setting all antecedent precipitation windows to $0.0$ mm causes **170/170 wells (100%)** to be flagged as out-of-support (`OOD_Extrapolative = 1`), because historical training observations exhibit strictly positive cumulative rainfall minimums (e.g., minimum 30-day precipitation in training was $5.34$ mm). S2 is explicitly documented as a hypothetical out-of-support stress test.
> - **Scenarios S1, S3, S4**: Exactly **26/170 wells** ($21$ warm-start, $5$ cold-start) are flagged as out-of-support due to baseline drift in non-weather features: 18 wells have accumulated `Previous_Observation_Count` $> 39$ (up to 45 by 2024), 5 cold-start wells have `Previous_Observation_Count` $= 0$, and 3 wells have `Historical_Min` $< 1.05$ ft (shallow artesian levels).

---

## 6. Candidate Dual-Track Decision Support Matrix (170 Wells)

> [!IMPORTANT]
> **Provisional / Candidate Decision-Support Tiers**:
> The classifications below are **candidate operational screening tiers** based on model outputs ($\hat{\Delta h}$, conformal width $W_{90}$, monitoring gap) and hydro-climatic diagnostic indicators (RRPI). They are **NOT** regulatory thresholds, validated physical groundwater boundaries, measured recharge classifications, or proof of extraction-driven drawdown. RRPI is strictly a **relative hydro-climatic indicator**, not measured recharge.

| Candidate Decision Tier | Monitored Well Count | Percentage | Recommended Operational Action |
| :--- | :---: | :---: | :--- |
| **Tier 1: Active Responsive Rise** | 22 | 12.9% | Document local water table shallowing; prioritize as active water table shallowing / recovery monitoring zone. |
| **Tier 2: Buffered / Stable State** | 17 | 10.0% | Standard periodic monitoring cycle; water table buffered by depth or moderate exposure. |
| **Tier 3: Projected Water Table Falling Response** | 38 | 22.4% | Flag for potential local water table decline review; assess adjacent extraction and drought exposure. |
| **Tier 4: High-Uncertainty / Stale Monitoring** | 88 | 51.8% | Prioritize for physical field remeasurement to reset baseline state. |
| **Tier 5: Cold-Start Favorable Hydro-Climate** | 3 | 1.8% | Unmonitored well with favorable relative hydro-climatic score; candidate for new monitoring instrumentation. |
| **Tier 6: Cold-Start Remote / Low Favorability** | 2 | 1.2% | Unmonitored well with lower relative hydro-climatic score or higher spatial distance; secondary priority. |

> [!WARNING]
> This framework is **conceptual and provisional**. It serves as an operational decision-support tool, **NOT** a physical recharge metering system.

---

## 7. 14-Gate Anti-Circularity & Leakage Audit Matrix

| Gate ID | Name | Status | Details |
| :--- | :--- | :---: | :--- |
| **RCH3-LC-01** | Prior-Day Weather Cutoff | **PASS** | max(weather_date_used) < DateMsr verified for 100% of records (0 violations). |
| **RCH3-LC-02** | Groundwater Temporal Precedence | **PASS** | Date_prior < Date_current verified across all 3,674 warm-start records (0 violations). |
| **RCH3-LC-03** | Target Segregation (`WatLevel`) | **PASS** | Current WatLevel strictly absent from candidate predictors. |
| **RCH3-LC-04** | Target Segregation ($\Delta h$) | **PASS** | Target $\Delta h$ strictly absent from candidate predictors. |
| **RCH3-LC-05** | Static Raster Exclusion | **PASS** | `TIFF_Value` strictly absent from dataset and all models. |
| **RCH3-LC-06** | Grouped Well Separation | **PASS** | Zero well overlap between train and test across all 10 holdout seeds. |
| **RCH3-LC-07** | Spatial Fold Isolation | **PASS** | Zero well overlap across all 5 spatial cluster folds. |
| **RCH3-LC-08** | Transformation Leakage Isolation | **PASS** | All scalers, imputers, and quantiles fit strictly on training partitions. |
| **RCH3-LC-09** | Cold-Start Target Integrity | **PASS** | All 170 cold-start records have 100% NaN for $\Delta h$ (zero fabricated targets). |
| **RCH3-LC-10** | Conformal Calibration Independence | **PASS** | Calibration partition strictly precedes test partition (zero temporal leakage). |
| **RCH3-LC-11** | RRPI Reference Population Fidelity | **PASS** | Inherited approved 8-feature baseline from 2000-2019 reference population. |
| **RCH3-LC-12** | Cold-Start History Fabrication Check | **PASS** | Zero synthetic or imputed history assigned to cold-start wells. |
| **RCH3-LC-13** | Duplicate-Date Sequencing Check | **PASS** | All duplicate well-date groups share identical prior state; zero sequential $\Delta h$ on same date. |
| **RCH3-LC-14** | Baseline Phase Preservation | **PASS** | All 129 locked files from Phases 08.4-08.9, Stage 1, Stage 2, and Methodology verified 100% identical. |

---

## 8. Output Artifact Catalog

1. `stage3_response_predictions_temporal.csv` (180 rows, predictions & conformal bounds)
2. `stage3_response_predictions_repeated.csv` (30 rows across 10 seeds)
3. `stage3_response_predictions_spatial.csv` (15 rows across 5 spatial folds)
4. `stage3_uncertainty_calibration.csv` (3 candidate models, nonconformity cutoffs)
5. `stage3_uncertainty_test_evaluation.csv` (12 rows, coverage & widths across gap cohorts)
6. `stage3_shap_global_importance.csv` (23 features ranked by mean |SHAP|)
7. `stage3_shap_interaction_profiles.csv` (3 key interaction pairs profiled)
8. `stage3_scenario_stress_testing.csv` (680 evaluations across 4 scenarios x 170 wells)
9. `stage3_decision_support_matrix.csv` (170 wells classified into 6 candidate decision tiers)
10. `stage3_model_summary.csv` (Comprehensive multi-regime performance benchmark)
11. `stage3_leakage_audit.csv` (14-gate integrity audit)
12. `README.md` & `walkthrough.md` (Scientific documentation & execution report)
