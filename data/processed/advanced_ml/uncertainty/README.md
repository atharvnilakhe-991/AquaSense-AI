# AquaSense AI — Phase 08.9: Uncertainty Quantification, Prediction Reliability & Applicability

## Executive Summary & Phase Status

Phase 08.9 establishes the empirical uncertainty bounds, predictive reliability indicators, and domain applicability diagnostics for the observation-aware machine learning models developed in AquaSense AI for Phelps County, Nebraska.

- **Status**: **EXECUTED, AUDITED, AND VALIDATED (GATES UG-01 THROUGH UG-10 PASSED)**
- **Pipeline Script**: [`src/08_9_uncertainty_quantification.py`](file:///c:/Users/athar/OneDrive/Desktop/EDI%20Project/src/08_9_uncertainty_quantification.py)
- **Primary Method**: **Stratified Split Conformal Prediction** (separate calibration for Warm-Start vs. Cold-Start operational regimes).
- **Secondary Benchmark**: **Conformalized Quantile Regression (CQR)** with gradient-boosted quantile trees.
- **Evaluation Regimes**: Temporal Holdout (2023–2024), 10-Seed Repeated Grouped-Well Holdouts, 5-Fold Spatial Cluster Holdouts.
- **Strict Boundary**: Risk estimation, hazard scoring, and decision rules are excluded and deferred to **Phase 08.11 (Risk & Decision Support)**.

---

## 1. Locked Foundation & Invariants

Phase 08.9 builds strictly on the locked and audited foundation of Phases 08.4–08.8:
- **Dataset**: 3,844 physical monitoring records across 170 unique wells in Phelps County, Nebraska (2000–2024).
- **Locked Models**: XGBoost A3 (primary point prediction model; 20 features) and Random Forest A3 (reference model for model disagreement).
- **Target Variable**: Depth to water (`WatLevel` in ft below surface).
- **Prohibited Variables**: Static baseline raster `TIFF_Value` strictly excluded.
- **Integrity Constraints**: Strict temporal precedence ($\text{Date}_{\text{prior}} < \text{Date}_{\text{pred}}$) and strict well-level train/cal/test separation.

---

## 2. Four-Part Inference Vector Results

For any query instance $x_i$, Phase 08.9 outputs:
$$\mathcal{T}(x_i) = \Big( \hat{y}(x_i),\; \mathcal{C}_{1-\alpha}(x_i),\; \mathcal{R}(x_i),\; \mathcal{A}(x_i) \Big)$$

1. **Point Prediction $\hat{y}(x_i)$**: Continuous groundwater depth estimate from locked XGBoost A3.
2. **Prediction Interval $\mathcal{C}_{1-\alpha}(x_i)$**: Calibrated intervals at 90% and 95% nominal confidence.
3. **Reliability Evidence $\mathcal{R}(x_i)$**: Empirical signals including `Interval_Width`, `Warm_Cold_Status`, `Days_Since_Previous`, `Observation_Density`, `Previous_Observation_Count`, `Model_Disagreement`, and `Calibration_Cohort_MAE`.
4. **Applicability Diagnostics $\mathcal{A}(x_i)$**: `Mahalanobis_Distance` (regularized), `Spatial_Distance_To_Train_Well_KM`, and `Environmental_Out_Of_Range_Count`.

---

## 3. Comprehensive Uncertainty Evaluation Matrix

Across all experimental regimes, Stratified Split Conformal Prediction and the CQR benchmark were evaluated across Overall, Warm-Start, and Cold-Start cohorts:

| Validation Experiment | Partition / Split | Cohort | Sample Count ($N$) | Point MAE (ft) | Conformal PICP 90% | Conformal Cov Dev 90% | Conformal MPIW 90% (ft) | Conformal PICP 95% | Conformal MPIW 95% (ft) | CQR PICP 90% | CQR MPIW 90% (ft) | CQR PICP 95% | CQR MPIW 95% (ft) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Temporal Holdout** | 2023–2024 Test Set | Overall | 188 | 1.54 | **97.34%** | +7.34% | **11.95** | **99.47%** | **13.86** | 98.94% | 43.43 | 100.00% | 96.60 |
| **Temporal Holdout** | 2023–2024 Test Set | Warm-Start | 180 | 1.48 | **97.78%** | +7.78% | **11.95** | **99.44%** | **13.86** | 98.89% | 42.76 | 100.00% | 96.06 |
| **Temporal Holdout** | 2023–2024 Test Set | Cold-Start | 8 | 2.94 | **87.50%** | -2.50% | **11.82** | **100.00%** | **13.86** | 100.00% | 58.54 | 100.00% | 108.77 |
| **Repeated Grouped-Well** | Pooled (10 Seeds) | Overall | 7,788 | 4.14 | **86.59%** | -3.41% | **18.15** | **91.45%** | **25.90** | 88.01% | 47.05 | 91.73% | 96.29 |
| **Repeated Grouped-Well** | Pooled (10 Seeds) | Warm-Start | 7,448 | 3.76 | **86.22%** | -3.78% | **14.11** | **91.30%** | **22.22** | 88.18% | 45.47 | 91.93% | 95.18 |
| **Repeated Grouped-Well** | Pooled (10 Seeds) | Cold-Start | 340 | 12.51 | **94.71%** | +4.71% | **106.51** | **94.71%** | **106.51** | 84.12% | 81.58 | 87.35% | 120.55 |
| **Spatial Cluster** | Pooled (5 Folds) | Overall | 3,844 | 5.06 | **77.34%** | -12.66% | **17.70** | **83.71%** | **22.39** | 77.39% | 47.32 | 78.90% | 94.98 |
| **Spatial Cluster** | Pooled (5 Folds) | Warm-Start | 3,674 | 4.45 | **76.84%** | -13.16% | **13.51** | **83.51%** | **18.41** | 77.57% | 45.72 | 79.18% | 93.87 |
| **Spatial Cluster** | Pooled (5 Folds) | Cold-Start | 170 | 18.10 | **88.24%** | -1.76% | **108.29** | **88.24%** | **108.45** | 73.53% | 81.77 | 72.94% | 118.88 |

---

## 4. Key Methodological Findings

### 1. Mandatory Need for Warm/Cold Calibration Stratification:
- Cold-start prediction error (MAE = $12.51\text{ ft}$ in unseen wells; $18.10\text{ ft}$ in spatial holdout) is over 3–4× higher than warm-start error (MAE = $3.76\text{ ft}$ and $4.45\text{ ft}$).
- Because warm-start observations comprise 95.6% of all records, an unstratified global quantile would have produced narrow intervals ($\sim 14\text{ ft}$), causing catastrophic under-coverage ($< 30\%$) on cold-start queries.
- Under Stratified Split Conformal Prediction, the cold-start calibration stratum expanded empirical intervals to $106.51\text{ ft}$, achieving **94.71% empirical coverage** across unseen cold-start wells and **88.24%** across held-out spatial clusters.

### 2. Method Selection: Stratified Split Conformal vs. CQR:
- **Sharpness**: Stratified Split Conformal achieves an MPIW 90% of **18.15 ft** on repeated holdouts, whereas CQR requires **47.05 ft** (2.6× wider). For 95% nominal confidence, Split Conformal requires **25.90 ft** vs. CQR's **96.29 ft** (3.7× wider).
- **Coverage**: Both methods achieve comparable overall coverage on repeated holdouts (86.59% vs. 88.01% for 90%; 91.45% vs. 91.73% for 95%).
- **Cold-Start Performance**: Stratified Split Conformal achieved **94.71%** coverage on cold-start unseen wells, whereas CQR achieved only **84.12%**.
- **Conclusion**: Stratified Split Conformal Prediction is designated as the primary uncertainty engine due to superior interval sharpness, deterministic stability, and higher cold-start fidelity. CQR is retained as a benchmark.

### 3. Spatial Exchangeability Breakdown:
- On spatial cluster validation, regional performance varied markedly across folds:
  - *Cluster 0 (Northwest)*: Conformal PICP 90% = **94.14%** (MPIW = 23.31 ft)
  - *Cluster 3 (South-Central)*: Conformal PICP 90% = **98.64%** (MPIW = 21.08 ft)
  - *Cluster 4 (Southwest)*: Conformal PICP 90% = **92.22%** (MPIW = 18.28 ft)
  - *Cluster 1 (North-Central)*: Conformal PICP 90% = **46.38%** (MAE = 8.90 ft, Bias = -6.63 ft)
  - *Cluster 2 (East)*: Conformal PICP 90% = **58.72%** (MAE = 6.72 ft, Bias = +4.84 ft)
- Clusters 1 and 2 experienced substantial spatial mean shifts. This empirically confirms the scientific caveat documented in Section 24: spatial dependence breaks ordinary exchangeability, demonstrating why uncalibrated universal coverage guarantees must never be claimed.

---

## 5. Reliability Evidence Empirical Analysis

Auditing the empirical relationship between candidate reliability signals and observed prediction accuracy revealed actionable patterns:

### Model Disagreement ($|\hat{y}_{\text{XGB}} - \hat{y}_{\text{RF}}|$):
- **Quartile 1 (Low Disagreement, N=2,908)**: Mean Absolute Error = **1.71 ft**, Median AE = **1.05 ft**, Empirical Coverage 90% = **96.94%**.
- **Quartile 2 (Med-Low Disagreement, N=2,908)**: Mean Absolute Error = **2.04 ft**, Median AE = **1.18 ft**, Empirical Coverage 90% = **95.39%**.
- **Quartile 3 (Med-High Disagreement, N=2,908)**: Mean Absolute Error = **3.17 ft**, Median AE = **1.85 ft**, Empirical Coverage 90% = **89.96%**.
- **Quartile 4 (High Disagreement, N=2,908)**: Mean Absolute Error = **10.86 ft**, Median AE = **7.32 ft**, Empirical Coverage 90% = **51.86%**.
- *Insight*: Algorithmic model disagreement between diverse architectures serves as a measurable proxy for empirical error magnitude.

### Monitoring Gap Duration (`Days_Since_Previous`):
- Active monitoring ($\le 6\text{ months}$): MAE = 4.60 ft (captures seasonal pumping volatility).
- Semi-annual to 2 years: MAE = 3.40–3.97 ft (stable annual trends).
- Extreme monitoring lapses ($> 5\text{ years}$, N=30): MAE rises to **5.44 ft**, and coverage drops to **70.0%**.

---

## 6. Domain Applicability Audit

Auditing model performance against spatial and environmental feature domains confirmed that spatial applicability provides a distinct diagnostic from prediction interval width; a prediction may have a relatively narrow interval while being spatially distant from the training network:

### Geodesic Proximity to Nearest Training Well:
- **0–2 km (Core Training Envelope, N=4,609)**: Mean AE = **2.04 ft**, Median AE = **1.13 ft**, Coverage 90% = **96.18%**, Coverage 95% = **98.29%**.
- **2–5 km (Intermediate Proximity, N=3,845)**: Mean AE = **5.05 ft**, Median AE = **2.39 ft**, Coverage 90% = **78.10%**, Coverage 95% = **86.71%**.
- **5–10 km (Periphery, N=2,351)**: Mean AE = **7.24 ft**, Median AE = **2.91 ft**, Coverage 90% = **73.50%**, Coverage 95% = **78.86%**.
- **10–20 km (Regional Extrapolation, N=827)**: Mean AE = **7.06 ft**, Median AE = **4.20 ft**, Coverage 90% = **66.87%**, Coverage 95% = **75.21%**.

*Scientific Demonstration*: Geographic distance beyond 2 km from previously monitored wells systematically increases residual error and causes coverage degradation. This confirms that applicability diagnostics provide essential context beyond prediction intervals alone.

---

## 7. Validation Gates Audit (UG-01 through UG-10)

| Gate ID | Gate Name | Verified Requirement | Status | Audit Findings |
| :---: | :--- | :--- | :---: | :--- |
| **UG-01** | Target Segregation | Target `WatLevel` and `TIFF_Value` strictly absent from predictor matrices | **PASS** | 100% verified across all feature matrices. |
| **UG-02** | Temporal Precedence | $\max(\text{Date}_{\text{cal}}) < \min(\text{Date}_{\text{test}})$; strict historical precedence | **PASS** | Cal max (2022-10-31) < Test min (2023-04-06); 0 temporal precedence violations. |
| **UG-03** | Well-Level Separation | Zero well overlap between train, cal, and test sets | **PASS** | 108 train, 28 cal, 34 test wells strictly disjoint across all 10 seeds. |
| **UG-04** | Spatial Separation | Held-out cluster wells isolated from train and cal | **PASS** | 0 held-out cluster wells leaked into train/cal across all 5 folds. |
| **UG-05** | Calibration/Test Separation | Row index disjointness ($D_{\text{cal}} \cap D_{\text{test}} = \emptyset$) | **PASS** | Exactly 0 overlapping rows in 100% of evaluations. |
| **UG-06** | Warm/Cold Semantic Integrity | Genuine NaNs preserved in cold-start rows | **PASS** | Cold-start rows retain genuine NaNs; zero synthetic filling. |
| **UG-07** | Interval Monotonicity | Bound ordering $\hat{L}_{1-\alpha} \le \hat{y} \le \hat{U}_{1-\alpha}$ | **PASS** | Exactly 0 bound inversions across 11,820 interval evaluations. |
| **UG-08** | Coverage Evaluation Correctness | Exact indicator evaluation against ground truth | **PASS** | PICP and coverage deviation calculated against true held-out $y$. |
| **UG-09** | Applicability Diagnostic Integrity | Applicability estimators fit strictly on train | **PASS** | $\hat{\mu}, \hat{\Sigma}$, and spatial KD-trees fitted exclusively on $D_{\text{train}}$. |
| **UG-10** | Baseline Preservation | Phases 08.4–08.8 artifacts 100% untouched | **PASS** | All 39 baseline CSV files verified untouched. |

---

## 8. Final Post-Execution Scientific Audit

A rigorous post-execution audit was conducted on all generated artifacts, logged in [`validation/final_post_execution_audit.csv`](file:///c:/Users/athar/OneDrive/Desktop/EDI%20Project/data/processed/advanced_ml/uncertainty/validation/final_post_execution_audit.csv):
- **Total Checks Evaluated**: 28 independent verification checks across all experimental partitions.
- **Audit Outcomes**:
  - **PASS**: 26 checks (100% numerical, architectural, and leakage-isolation criteria satisfied).
  - **DOCUMENTATION_ISSUE**: 2 checks (phrasing of spatial applicability and conservative language calibration; both corrected directly in documentation).
  - **METHODOLOGICAL_ISSUE**: 0 checks.
  - **DATA_ARTIFACT_ISSUE**: 0 checks.
- **Rerun Requirement**: None. All numerical results are verified to be mathematically exact and faithful to the approved methodology.

---

## 9. Directory Artifacts Reference

All generated outputs are located in `data/processed/advanced_ml/uncertainty/`:
- `calibration/conformal_calibration_summary.csv`: Quantiles and calibration MAEs per regime, split, and seed.
- `intervals/prediction_intervals_temporal.csv`: Instance-level intervals for the 2023–2024 temporal holdout.
- `intervals/prediction_intervals_repeated.csv`: Instance-level intervals across the 10 repeated grouped-well holdouts.
- `intervals/prediction_intervals_spatial.csv`: Instance-level intervals across the 5 spatial cluster folds.
- `evaluations/uncertainty_metrics_summary.csv`: Comprehensive PICP, coverage deviation, MPIW, and error benchmarks.
- `evaluations/warm_vs_cold_uncertainty.csv`: Stratified warm vs. cold uncertainty summary.
- `evaluations/spatial_uncertainty_analysis.csv`: Cluster-by-cluster spatial uncertainty report.
- `evaluations/cqr_benchmark_comparison.csv`: Comparative benchmark between Split Conformal and CQR.
- `reliability/reliability_evidence_summary.csv`: Empirical error and coverage distributions across reliability cohorts.
- `reliability/applicability_extrapolation_audit.csv`: Distance-based and Mahalanobis domain applicability audit.
- `validation/leakage_validation_report.csv`: Complete audit record for Gates UG-01 through UG-10.
- `validation/final_post_execution_audit.csv`: Independent post-execution scientific audit (28 checks).

