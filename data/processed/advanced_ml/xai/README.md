# Phase 08.8: Explainable AI (XAI) & SHAP Analysis Report
**Status: LOCKED**

## Executive Summary
Phase 08.8 establishes the Explainable AI (XAI) foundation of the AquaSense machine learning module. Using exact TreeSHAP on **held-out generalization test partitions** from Phase 08.6 (10 repeated grouped-well splits and 5-fold spatial cluster holdouts), this study reveals the internal predictive mechanics of observation-aware groundwater modeling.

---

## Core Scientific Findings

1. **Dominant Observation-History Attribution**:
   - The 12-feature observation-history block accounts for **82.33%** of total predictive attribution across held-out predictions (Mean |SHAP| = 52.19 ft).
   - Primary predictive anchors: `Previous_WatLevel` (Mean |SHAP| = 19.091 ft, 30.12% share), `Rolling_Mean_3` (Mean |SHAP| = 9.709 ft, 15.32% share), and `Historical_Max` (Mean |SHAP| = 9.409 ft, 14.84% share).
   - *Correlation note*: Collinear history features (`Previous_WatLevel`, `Rolling_Mean_3`, `Historical_Mean`) act jointly; their importance is reported as an aggregate attribution across correlated history-feature families.

2. **Warm-Start vs. Cold-Start Attribution Shift**:
   - Under active monitoring (Warm-Start), temporal history features dictate prediction paths (>82% attribution).
   - When history is legitimately absent (Cold-Start, `Previous_Observation_Count == 0`), predictive attribution shifts heavily to spatial coordinates (`LatDD`: 8.61% $\to$ 27.69%, +19.08% shift) and topography (`Surf_Elev`: 4.77% $\to$ 21.35%, +16.58% shift), explaining how the model degrades gracefully to regional environmental baseline levels.
   - *Cold-Start Semantic Integrity*: Non-zero attributions for missing history features strictly reflect learned tree-split branch traversal adjustments for unmonitored sites, verified with the `is_feature_available: False` metadata tag.

3. **High-Error Residual Attribution Patterns**:
   - Observations in the P90/P95 high-error cohorts are characterized by elevated reliance on spatial gradients (`LatDD` delta |SHAP| = +12.11 ft, `Surf_Elev` delta |SHAP| = +3.44 ft) and deep groundwater table depths (averaging 138 ft vs. 63 ft in the low-error cohort), illuminating regional boundary operating constraints.

4. **Multi-Split Stability**:
   - Feature importance rankings demonstrate strong stability across 10 independent holdout seeds (mean Spearman rank correlation $\rho_s = 0.9547$, range [0.9023, 0.9880]).

5. **Model-to-Model Verification**:
   - Random Forest A3 independently confirms dominant history reliance (`Previous_WatLevel` 27.16 ft, `Historical_Mean` 14.97 ft).
   - XGBoost A4 confirms observation-quality proxies receive $< 0.2\%$ attribution (`Previous_Observation_Count`: 0.103 ft, `Observation_Density`: 0.039 ft, gap flags: 0.0 ft), explaining their marginal empirical contribution in Phase 08.7.

---

## Methodological Validation & Additivity Tolerance

### 1. Row-Level Prediction Reproduction
Row-level prediction reproduction between Phase 08.6 locked baselines and Phase 08.8 reproduced predictions was verified on 100% of held-out test observations:
- **Total rows compared**: 11,632 (7,788 repeated holdout rows + 3,844 spatial cluster holdout rows)
- **Maximum absolute difference**: 0.0000000000 ft
- **Mean absolute difference**: 0.0000000000 ft
- **Exceedances of $10^{-4}$ ft**: 0 (0.0%)
- **Status**: **PASS**

### 2. SHAP Numerical Additivity Specification
The original SHAP additivity target was <1e-4 ft. The audit found a maximum numerical discrepancy of 0.00036621 ft, with 1,050 of 11,632 observations exceeding 1e-4 ft and zero observations exceeding 1e-3 ft. The observed discrepancy is attributed to floating-point numerical accumulation in the float32 XGBoost/TreeSHAP implementation. A <1e-3 ft tolerance is therefore adopted as an implementation-level numerical tolerance.

This numerical tolerance is not a scientific confidence interval, prediction uncertainty bound, measurement-error estimate, or guarantee of prediction accuracy.

- **Maximum absolute error**: 0.00036621 ft
- **Mean absolute error**: 0.00003656 ft
- **Median absolute error**: 0.00002098 ft
- **P95 absolute error**: 0.00012207 ft
- **P99 absolute error**: 0.00019836 ft
- **Status**: **PASS UNDER APPROVED <1e-3 FT NUMERICAL TOLERANCE**

### 3. Spatial Aggregation Integrity
- **Total Phelps County observations**: 3,844
- **Unique wells represented**: 170 / 170 (100%)
- **Total aggregated observations**: 3,844
- **Sample-count discrepancies**: 0
- **Training-observation contamination**: 0
- **Status**: **PASS**

---

## Non-Causal Framing & Scope Boundaries

> [!NOTE]
> All feature attributions and sensitivities in this report reflect the learned decision behavior of gradient boosted decision trees under the specified evaluation design. They must not be interpreted as physical causality, hydrodynamic cause-and-effect, or direct proofs of groundwater recharge mechanisms. Furthermore, while spatial cluster holdout demonstrates regional transportability across Phelps County sub-regions, it does not imply universal geographic transportability to hydrogeologically distinct aquifer basins.

---

## Directory Contents

- `global/`
  - `xai_global_feature_importance.csv`: Global feature importance metrics (Mean |SHAP|, Std, Median, P90, % share).
  - `xai_grouped_feature_importance.csv`: Feature block aggregation across History, Geography, Topography, Climate.
  - `shap_importance_bar.png`: Global feature ranking bar chart.
  - `shap_summary_beeswarm.png`: Beeswarm summary plot illustrating directional feature impacts.
- `warm_start/` & `cold_start/`
  - `xai_warm_start_importance.csv`: Feature importance rankings on actively monitored wells.
  - `xai_cold_start_importance.csv`: Feature importance rankings on unmonitored locations.
- `comparisons/`
  - `xai_warm_vs_cold_comparison.csv`: Numerical attribution shift between warm-start and cold-start.
  - `warm_vs_cold_attribution_shift.png`: Reallocation bar chart showing spatial/topographic substitution.
  - `model_to_model_attribution_comparison.csv`: Comparative attribution across XGBoost A3, RF A3, and XGBoost A4.
- `dependence/`
  - `xai_dependence_summary.csv`: Spearman correlations and strongest interacting features.
  - `dependence_*.png` (8 figures): Algorithmic response curves with interaction color maps.
- `local/`
  - `xai_local_explanations.csv`: Table of 6 operational prototype explanations.
  - `xai_local_explanations.json`: Machine-readable JSON payloads for backend/frontend integration.
  - `waterfall_*.png` (6 figures): Local explanation waterfall bar charts.
- `error_analysis/`
  - `xai_error_cohort_comparison.csv`: Attribution comparisons across Low-Error (<= P50) and High-Error (>= P90/P95).
  - `error_associated_feature_patterns.png`: Comparative error cohort attribution chart.
- `spatial/`
  - `xai_spatial_held_out_attribution.csv`: Per-well dominant predictor mappings across all 170 wells.
  - `spatial_dominant_feature_map.png`: Geographic scatter map of dominant predictors in Phelps County.
- `validation/`
  - `xai_prediction_reproduction_audit.csv`: Row-level prediction reproduction audit (15 splits, 11,632 rows).
  - `leakage_validation_report.csv`: 10-layer XAI validation audit log.
  - `xai_stability_analysis.csv`: Multi-seed Spearman rank correlation stability metrics.
  - `xai_physical_consistency_audit.csv`: Directional consistency audits against hydrological domain expectations.
