# Phase 08.8: Explainable AI (XAI) & TreeSHAP Analysis Walkthrough
**Status: LOCKED**

---

## 1. Overview & Objectives

Phase 08.8 established the Explainable AI (XAI) foundation of the AquaSense machine learning module. Using exact TreeSHAP on **held-out generalization test partitions** from Phase 08.6 (10 repeated grouped-well splits and 5-fold spatial cluster holdouts), this study reveals the internal predictive mechanics of observation-aware groundwater modeling.

---

## 2. Key Findings

### A. Dominant Observation-History Attribution
- The 12-feature observation-history block drives **82.33%** of total predictive attribution on held-out test predictions (Mean |SHAP| = 52.1867 ft).
- Primary anchors:
  - `Previous_WatLevel`: Mean |SHAP| = 19.0914 ft (30.12% share)
  - `Rolling_Mean_3`: Mean |SHAP| = 9.7085 ft (15.32% share)
  - `Historical_Max`: Mean |SHAP| = 9.4093 ft (14.84% share)
  - `Historical_Min`: Mean |SHAP| = 6.6328 ft (10.46% share)
  - `Historical_Mean`: Mean |SHAP| = 6.1401 ft (9.69% share)
- Environmental baseline features contribute the remaining attribution:
  - Geography (`LatDD`, `LongDD`): **10.42%** share (Mean |SHAP| = 6.6018 ft)
  - Topography (`Surf_Elev`): **5.47%** share (Mean |SHAP| = 3.4674 ft)
  - Climate (5 PRISM covariates): **1.79%** share (Mean |SHAP| = 1.1305 ft)

### B. Warm-Start vs. Cold-Start Attribution Shift
- Under active monitoring (**Warm-Start**), temporal history features dictate prediction paths (>82% attribution).
- When history is legitimately absent (**Cold-Start**, `Previous_Observation_Count == 0`), predictive attribution shifts heavily into spatial coordinates (`LatDD`: 8.61% $\to$ 27.69%, +19.08% shift) and topography (`Surf_Elev`: 4.77% $\to$ 21.35%, +16.58% shift), explaining graceful model degradation to regional environmental head interpolation.
- *Cold-Start Semantic Integrity*: History features genuinely missing (`NaN`) evaluate through the learned default missing-branch split path; metadata explicitly tags `is_feature_available: False`.

### C. Multi-Seed Stability & Model Comparison
- **Rank Stability**: Multi-seed feature importance rankings achieve a mean Spearman rank correlation $\rho_s = 0.9547$ across all 10 seed holdouts (range: [0.9023, 0.9880]).
- **Random Forest A3**: Independently corroborates observation history dominance (`Previous_WatLevel` 27.16 ft, `Historical_Mean` 14.97 ft).
- **XGBoost A4**: Confirms observation-quality proxies receive $< 0.2\%$ attribution (`Previous_Observation_Count`: 0.1031 ft, `Observation_Density`: 0.0391 ft, gap flags: 0.0 ft), explaining their marginal empirical contribution in Phase 08.7.

---

## 3. Methodological Validation Audits

### A. Row-Level Prediction Reproduction
Row-level prediction reproduction between Phase 08.6 locked baselines and Phase 08.8 reproduced predictions was verified on 100% of held-out test observations:
- **Total rows compared**: 11,632 (7,788 repeated holdout rows + 3,844 spatial cluster holdout rows)
- **Maximum absolute difference**: 0.0000000000 ft
- **Mean absolute difference**: 0.0000000000 ft
- **Median absolute difference**: 0.0000000000 ft
- **Exceedances of $10^{-4}$ ft**: 0
- **Status**: **PASS**

### B. SHAP Additivity Precision
The original SHAP additivity target was <1e-4 ft. The audit found a maximum numerical discrepancy of 0.00036621 ft, with 1,050 of 11,632 observations exceeding 1e-4 ft and zero observations exceeding 1e-3 ft. The observed discrepancy is attributed to floating-point numerical accumulation in the float32 XGBoost/TreeSHAP implementation. A <1e-3 ft tolerance is therefore adopted as an implementation-level numerical tolerance.

This numerical tolerance is not a scientific confidence interval, prediction uncertainty bound, measurement-error estimate, or guarantee of prediction accuracy.

- **Maximum absolute error**: 0.00036621 ft
- **Mean absolute error**: 0.00003656 ft
- **Median absolute error**: 0.00002098 ft
- **P95 absolute error**: 0.00012207 ft
- **P99 absolute error**: 0.00019836 ft
- **Status**: **PASS UNDER APPROVED <1e-3 FT NUMERICAL TOLERANCE**

### C. Spatial Aggregation Integrity
- **Total Phelps County observations**: 3,844
- **Unique wells represented**: 170 / 170 (100%)
- **Total aggregated observations**: 3,844
- **Sample-count discrepancies**: 0
- **Training-observation contamination**: 0
- **Dominant predictors per well**: `Previous_WatLevel` (134 wells, 78.8%), `Surf_Elev` (20 wells, 11.8%), `LatDD` (15 wells, 8.8%), `Historical_Mean` (1 well, 0.6%)
- **Status**: **PASS**

---

## 4. Non-Causal Framing & Scope Boundaries

All feature attributions and sensitivities in this report reflect the learned decision behavior of gradient boosted decision trees under the specified evaluation design. They must not be interpreted as physical causality, hydrodynamic cause-and-effect, or direct proofs of groundwater recharge mechanisms. Furthermore, while spatial cluster holdout demonstrates regional transportability across Phelps County sub-regions, it does not imply universal geographic transportability to hydrogeologically distinct aquifer basins.

---

## 5. Artifact Directory

All Phase 08.8 artifacts are housed under `data/processed/advanced_ml/xai/`:
- `global/`: `xai_global_feature_importance.csv`, `xai_grouped_feature_importance.csv`, `shap_importance_bar.png`, `shap_summary_beeswarm.png`
- `warm_start/`: `xai_warm_start_importance.csv`
- `cold_start/`: `xai_cold_start_importance.csv`
- `comparisons/`: `xai_warm_vs_cold_comparison.csv`, `model_to_model_attribution_comparison.csv`, `warm_vs_cold_attribution_shift.png`
- `dependence/`: `xai_dependence_summary.csv`, 8 dependence plots with interaction color maps
- `local/`: `xai_local_explanations.csv`, `xai_local_explanations.json`, 6 waterfall plots
- `error_analysis/`: `xai_error_cohort_comparison.csv`, `error_associated_feature_patterns.png`
- `spatial/`: `xai_spatial_held_out_attribution.csv`, `spatial_dominant_feature_map.png`
- `validation/`: `xai_prediction_reproduction_audit.csv`, `leakage_validation_report.csv`, `xai_stability_analysis.csv`, `xai_physical_consistency_audit.csv`
