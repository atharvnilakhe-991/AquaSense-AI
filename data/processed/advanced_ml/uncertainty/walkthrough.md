# Phase 08.9 Walkthrough: Uncertainty Quantification, Prediction Reliability & Applicability

This walkthrough documents the full execution, audit verification, and key findings of Phase 08.9 in AquaSense AI.

---

## 1. Execution Overview

- **Script Executed**: [`src/08_9_uncertainty_quantification.py`](file:///c:/Users/athar/OneDrive/Desktop/EDI%20Project/src/08_9_uncertainty_quantification.py)
- **Primary Uncertainty Engine**: Stratified Split Conformal Prediction
- **Secondary Benchmark**: Conformalized Quantile Regression (CQR)
- **Status**: Completed in ~1.8 minutes with zero runtime warnings or errors.
- **Validation Gates**: 10 out of 10 gates passed (**UG-01 through UG-10**).

---

## 2. Experimental Verification Across Regimes

### A. Temporal Calibration (2000–2019 Fit | 2020–2022 Cal | 2023–2024 Test)
- **Model Training**: 3,247 historical observations (2000–2019).
- **Calibration Stratum**: 408 warm-start observations, 1 cold-start observation.
- **Quantiles Learned**:
  - Warm $Q_{90} = 5.98\text{ ft}$ (Interval Width = $11.95\text{ ft}$)
  - Warm $Q_{95} = 6.93\text{ ft}$ (Interval Width = $13.86\text{ ft}$)
- **Test Performance (2023–2024, N=188)**:
  - Point Prediction MAE: **1.54 ft** (RMSE = 2.10 ft, Bias = -0.21 ft).
  - Empirical Conformal PICP 90%: **97.34%** (Coverage Deviation: +7.34%).
  - Empirical Conformal PICP 95%: **99.47%** (Coverage Deviation: +4.47%).
  - CQR Benchmark: Achieved 98.94% coverage, but with an average width of **43.43 ft** (over 3.6× wider than Split Conformal).

### B. Repeated Grouped-Well Generalization (10 Seeds, 170 Wells)
- **Partitioning**: 108 Training Wells (~80% of Dev), 28 Calibration Wells (~20% of Dev), 34 Test Wells (Untouched).
- **Zero Well Overlap**: Verified across all 10 random seeds (`[42, 101, 202, 303, 404, 505, 606, 707, 808, 909]`).
- **Pooled Metrics (N=7,788)**:
  - Overall Conformal PICP 90%: **86.59%** (MPIW = 18.15 ft).
  - Overall Conformal PICP 95%: **91.45%** (MPIW = 25.90 ft).
  - Warm-Start Conformal PICP 90%: **86.22%** (MPIW = 14.11 ft, MAE = 3.76 ft).
  - Cold-Start Conformal PICP 90%: **94.71%** (MPIW = 106.51 ft, MAE = 12.51 ft).
- **Key Takeaway**: Cold-start prediction intervals dynamically expanded to 106.51 ft based on empirical calibration errors, securing **94.71% coverage** for unmonitored wells.

### C. Spatial Cluster Holdout (5 Folds)
- **Partitioning**: 4 Development Clusters (split into Train and Cal) and 1 Held-Out Evaluation Cluster.
- **Pooled Metrics (N=3,844)**:
  - Overall Conformal PICP 90%: **77.34%** (MPIW = 17.70 ft).
  - Overall Conformal PICP 95%: **83.71%** (MPIW = 22.39 ft).
  - Cluster 0 (Northwest): PICP 90% = **94.14%** (MPIW = 23.31 ft).
  - Cluster 3 (South-Central): PICP 90% = **98.64%** (MPIW = 21.08 ft).
  - Cluster 4 (Southwest): PICP 90% = **92.22%** (MPIW = 18.28 ft).
  - Cluster 1 (North-Central): PICP 90% = **46.38%** (Under-coverage due to -6.63 ft mean bias).
  - Cluster 2 (East): PICP 90% = **58.72%** (Under-coverage due to +4.84 ft mean bias).
- **Key Takeaway**: Strong regional variation highlights the impact of spatial autocorrelation and hydrostratigraphic shifts, proving the necessity of reporting empirical coverage rather than assuming textbook exchangeability.

---

## 3. Reliability & Applicability Insights

1. **Model Disagreement**:
   - Small disagreement ($\le Q2$): MAE is **1.71–2.04 ft**, with **95.4–96.9% coverage**.
   - Large disagreement ($Q4$): MAE spikes to **10.86 ft**, with coverage dropping to **51.86%**.
2. **Spatial Distance from Nearest Monitored Well**:
   - Within 2 km: MAE = **2.04 ft**, Coverage 90% = **96.18%**.
   - 2–5 km: MAE = **5.05 ft**, Coverage 90% = **78.10%**.
   - 5–10 km: MAE = **7.24 ft**, Coverage 90% = **73.50%**.
   - 10–20 km: MAE = **7.06 ft**, Coverage 90% = **66.87%**.

---

## 4. Validation Gates Audit (UG-01 through UG-10)

All 10 gates passed with zero infractions logged in [`validation/leakage_validation_report.csv`](file:///c:/Users/athar/OneDrive/Desktop/EDI%20Project/data/processed/advanced_ml/uncertainty/validation/leakage_validation_report.csv):
- `UG-01`: PASS (Target & TIFF strictly excluded)
- `UG-02`: PASS (Temporal precedence verified)
- `UG-03`: PASS (Zero well overlap across train, cal, test)
- `UG-04`: PASS (Spatial cluster isolation verified)
- `UG-05`: PASS (Calibration and test rows 100% disjoint)
- `UG-06`: PASS (Cold-start history preserved as genuine NaNs)
- `UG-07`: PASS (Zero bound inversions; $L \le \hat{y} \le U$ for 100% of rows)
- `UG-08`: PASS (Coverage evaluated exactly against true labels)
- `UG-09`: PASS (Applicability estimators trained strictly on $D_{\text{train}}$)
- `UG-10`: PASS (All 39 baseline files from Phases 08.4–08.8 verified untouched)

---

## 5. Final Post-Execution Scientific Audit (28 Checks)

A comprehensive scientific audit was executed across all generated CSV outputs, logged in [`validation/final_post_execution_audit.csv`](file:///c:/Users/athar/OneDrive/Desktop/EDI%20Project/data/processed/advanced_ml/uncertainty/validation/final_post_execution_audit.csv):
- **Total Checks Evaluated**: 28 checks covering empirical metrics, temporal precedence, well isolation, spatial cluster independence, cold-start distributions, and baseline phase preservation.
- **Audit Outcomes**:
  - **PASS**: 26 checks (100% numerical and structural validity).
  - **DOCUMENTATION_ISSUE**: 2 checks (phrasing of spatial applicability and conservative language calibration; both corrected).
  - **METHODOLOGICAL_ISSUE**: 0 checks.
  - **DATA_ARTIFACT_ISSUE**: 0 checks.
- **Final Decision**: **OPTION A: PHASE 08.9 — READY TO LOCK**.

