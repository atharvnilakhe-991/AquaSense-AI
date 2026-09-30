# AMDFE Version 2.1 Critical Audit & Statistical Correction Plan
**AquaSense-AI Research Pipeline**
**Branch**: `ml-training`
**Date**: September 2026

---

## 1. Audit of Version 2 Implementation & Statistical Inconsistencies

A rigorous audit of the AMDFE v2 implementation and generated artifacts identified the following issues requiring methodological and statistical correction:

### 1.1 Inconsistent Delta Definition in Bootstrap Reporting
- In v2 summary tables, LightGBM RMSE was reported as $9.87\text{ m}$ (A2) and $5.80\text{ m}$ (A5).
- However, the bootstrap paired difference table used an inverted sign convention ($\text{RMSE}_{\text{baseline}} - \text{RMSE}_{\text{candidate}} = +9.48\text{ m}$) without explicit sign declaration.
- **Correction in v2.1**: All statistical tables enforce a single, unambiguous mathematical convention:
  $$\Delta \text{RMSE}_{\text{method\_minus\_baseline}} = \text{RMSE}_{\text{method}} - \text{RMSE}_{\text{baseline}}$$
  $$\text{Improvement\_RMSE\_Percent} = 100 \times \frac{\text{RMSE}_{\text{baseline}} - \text{RMSE}_{\text{method}}}{\text{RMSE}_{\text{baseline}}}$$
  Under this convention, negative $\Delta \text{RMSE}$ strictly indicates error reduction.

### 1.2 Feature-Count Confounding Between A2, A5, and A6
- In v2, A2 had 23 features, A5 had 32 features (adding $R_m, A_{i,m}, w_{i,m}$ metadata and adaptive scaling), and A6 had 60 features (adding missingness and gap flags).
- Because A5 added metadata columns in addition to adaptive gating, the performance improvement could not be conclusively attributed to the adaptive gating mechanism alone vs. simply having more metadata features.
- **Correction in v2.1**: Implement a strictly matched-feature control suite:
  - **C0**: Unweighted multimodal features (A2 base, 23 cols).
  - **C1**: Unweighted base features + reliability metadata + observability metadata + weight columns (32 cols, **NO adaptive gating**).
  - **C2**: Exactly the same 32 feature columns as C1, with **adaptive modality scaling/gating** ($w_{i,m} \cdot Z(X_{i,m})$).
  - The difference $\mathbf{C2 - C1}$ isolates the pure marginal attribution of adaptive feature gating with zero feature-count disparity.

### 1.3 Observability Formulation Heuristic Sensitivity
- The v2 groundwater observability formulation used fixed parameters ($0.20 + 0.50 \exp(-\Delta t / 365.25) + 0.30 \min(N_{\text{prior}}/3, 1)$).
- **Correction in v2.1**: Conduct an exhaustive grid sensitivity evaluation over gap decay scales ($180, 365.25, 730$ days), prior observation saturation thresholds ($2, 3, 5$), and baseline weight mixtures without tuning on test partitions.

### 1.4 Cold-Start Generalization Disclosure
- On cold-start unseen wells ($N_{\text{prior}}=0$), groundwater lag features are null. Multimodal spatial models (A2) achieve $\text{RMSE} \approx 13.37\text{ m}$ while groundwater-only (A0) diverges ($\text{RMSE} \approx 52.23\text{ m}$).
- **Correction in v2.1**: Explicitly report that AMDFE is an observation-state-adaptive framework that improves warm-start and sparse-monitoring resilience, but does not solve zero-history cold-start spatial extrapolation.

---

## 2. Directory Architecture for v2.1

All v2.1 corrected artifacts are isolated in `data/processed/amdfe_v21/`:
- `fused/`: Canonical C0, C1, C2, A0, A1, A6 fused tables.
- `quality/`: Objective data quality dimensions.
- `reliability/`: Train-fitted source reliability scores.
- `observability/`: Per-sample point-in-time observability factors.
- `validation/`: 15-point automated assertion logs.
- `experiments/`: Master experiment matrix (E1–E10), raw runs, summaries.
- `robustness/`: Multi-mode degradation stress test results with checksum integrity.
- `statistics/`: Recomputed well-level cluster bootstrap distributions and 95% CIs.
