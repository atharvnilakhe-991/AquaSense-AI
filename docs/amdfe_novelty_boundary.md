# AMDFE Novelty Boundary & Scientific Distinction Matrix
**AquaSense-AI Research Pipeline**
**Target**: Peer-Reviewed Hydroinformatics Literature
**Date**: September 2026

---

## 1. Explicit Distinction of Scientific Layers

To maintain rigorous scientific reporting, the boundary between existing literature, software engineering, proposed methodology, and empirical evidence is explicitly categorized into 5 tiers:

```
+--------------------------------------------------------------------------------------------------+
| TIER A: EXISTING KNOWN TECHNIQUES (Prior Art)                                                    |
| - Multimodal data concatenation (Early Fusion) and ensemble stacking (Late Fusion).              |
| - Standard gradient boosting (LightGBM, XGBoost) and Random Forest regressors.                   |
| - Unseen-well spatial cross-validation (GroupKFold by well ID).                                  |
| - SHAP feature attribution and Monte Carlo uncertainty estimation.                               |
| - Median/mean feature imputation and basic standard scaling.                                     |
+--------------------------------------------------------------------------------------------------+
| TIER B: AQUASENSE ENGINEERING IMPLEMENTATION                                                     |
| - Modular scikit-learn compatible pipeline architecture (AMDFEPipelineV2).                       |
| - Automated 15-point data leakage assertion test harness (AMDFE-V2-01 to AMDFE-V2-15).           |
| - Deterministic reproducibility seed harness and automated experiment matrix execution.          |
| - Structured versioned data artifact persistence (data/processed/amdfe_v2/).                     |
+--------------------------------------------------------------------------------------------------+
| TIER C: PROPOSED AMDFE METHODOLOGICAL CONTRIBUTION                                               |
| - Mathematical decoupling of Source Reliability R_m (intrinsic source quality) from Context     |
|   Observability A_{i,m} (per-sample monitoring state).                                           |
| - Non-applicable quality dimension masking (preventing arbitrary zero or perfect penalties).    |
| - Two-stage adaptive weighting formulation: G_{i,m} = R_m^alpha * A_{i,m}^beta, normalized to 1.  |
| - Strict causal annual weather lag policy (Y - 1) preventing mid-year lookahead leakage.         |
| - Transparent controlled feature-count fairness ablation suite (A0 to A6).                       |
+--------------------------------------------------------------------------------------------------+
| TIER D: EXPERIMENTALLY DEMONSTRATED CONTRIBUTION                                                 |
| - Empirical validation on 170 real groundwater wells (2000–2024).                                |
| - Separation of Unseen-Well Warm-Start vs Unseen-Well Cold-Start generalization performance.      |
| - Controlled robustness evaluation under 3 degradation modes: Pointwise Loss, Burst Gap Loss,     |
|   and Source Outage (20%, 40%, 60% severities).                                                  |
| - Cluster-bootstrap statistical analysis by well with 95% confidence intervals and effect sizes. |
| - Quantified monitoring-density and temporal-gap cohort analysis.                                |
+--------------------------------------------------------------------------------------------------+
| TIER E: WHAT CANNOT YET BE CLAIMED (Out of Scope / Future Work)                                  |
| - Universal superiority across unmeasured hydrogeological settings (e.g. karst or deep confined).|
| - Multi-source remote sensing integration (Sentinel-1/2, GRACE) until localized in repository.   |
| - Sub-annual event-scale storm runoff response (requires daily ERA5-Land reanalysis).            |
+--------------------------------------------------------------------------------------------------+
```

---

## 2. Framing Guidelines for Publication

1. **Avoid Asserting Novelty by Name**:
   Do **not** state: *"AMDFE is a novel framework that outperforms existing baselines."*
   Instead state: *"We propose the Adaptive Multimodal Data Fusion Engine (AMDFE), which explicitly decouples source-level data quality from sample-level observation state, and evaluate its empirical utility under regular and degraded groundwater monitoring conditions."*

2. **Attribute Tree-Model Scaling Realities Honestly**:
   Acknowledge that constant global scaling provides minimal split-point changes for decision trees. The empirical value of AMDFE resides in **dynamic sample-level gating** ($w_{i,m}$) and **explicit observability metadata** under irregular sampling and data loss.

3. **Separate Cold-Start from Warm-Start Generalization**:
   Unseen-well evaluation with prior target history available is defined as **Unseen-Well Warm-Start**. Predictions for wells with zero prior observations are defined as **Unseen-Well Cold-Start**.
