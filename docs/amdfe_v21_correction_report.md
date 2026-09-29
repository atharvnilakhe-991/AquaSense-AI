# AMDFE v2.1 Scientific Correction & Attribution Report

## 1. Executive Summary

This report documents the comprehensive scientific correction, attribution, and validation pass executed for the Adaptive Multimodal Data Fusion Engine (AMDFE v2.1).

All previous versioned results in `data/processed/amdfe/` (v1) and `data/processed/amdfe_v2/` (v2) remain completely intact and untouched. All corrected experiments, raw predictions, statistical bootstrap distributions, parameter sensitivity surfaces, and paper-ready tables have been generated and versioned in `data/processed/amdfe_v21/`.

---

## 2. Inventory of Statistical Inconsistencies Corrected

### Inconsistency 1: Inverted Delta Conventions in Bootstrap vs Ablation Tables
- **Previous Defect**: In AMDFE v2 reports, the main ablation table reported baseline RMSE $\approx 9.87\text{ m}$ and candidate RMSE $\approx 5.80\text{ m}$ (a reduction of $4.07\text{ m}$), while the bootstrap summary reported $\Delta\text{RMSE} = +9.48\text{ m}$ due to an inverted definition of the delta subtraction order and baseline reference.
- **Correction Applied**: Standardized one explicit, mathematically unambiguous convention across all v2.1 scripts, CSVs, and markdown reports:
  $$\Delta\text{RMSE}_{\text{method\_minus\_baseline}} = \text{RMSE}_{\text{method}} - \text{RMSE}_{\text{baseline}}$$
  $$\text{Improvement\_RMSE\_percent} = 100 \times \frac{\text{RMSE}_{\text{baseline}} - \text{RMSE}_{\text{method}}}{\text{RMSE}_{\text{baseline}}}$$
  Under this convention, a negative $\Delta\text{RMSE}$ strictly denotes error reduction (improvement).

### Inconsistency 2: Feature-Count Confounding in Ablation Comparisons
- **Previous Defect**: In v2, unweighted concatenation $A2$ had 23 features, while $A5$ had 32 features, and $A6$ had 60 features. This made it impossible to determine whether performance gains stemmed from the mathematical adaptive gating formulation or simply from supplying additional metadata columns.
- **Correction Applied**: Introduced matched-feature control suites ($C0, C1, C2$):
  - $C0$: Base unweighted features (23 cols)
  - $C1$: Base features + Reliability metadata + Observability metadata + Adaptive weight columns WITHOUT sample gating (32 cols)
  - $C2$: Identical 32 features WITH sample-level adaptive block gating ($w_{i,m} \cdot Z(X_{i,m})$)
  The isolated marginal effect of adaptive gating is strictly quantified by $C2 - C1$.

### Inconsistency 3: Observability Formulation Sensitivity
- **Previous Defect**: Groundwater observability factors in v2 relied on fixed heuristic constants ($\tau = 365.25\text{ d}, S = 3\text{ obs}, w = [0.2, 0.5, 0.3]$) without sensitivity testing.
- **Correction Applied**: Executed a multi-seed grid search over $\tau \in \{180, 365.25, 730\text{ days}\}$, $S \in \{2, 3, 5\text{ obs}\}$, and 4 weight mixtures on training/validation splits. Verified that the framework's relative ranking remains stable across reasonable parameterizations.

### Inconsistency 4: Cold-Start vs Warm-Start Spatial Generalization
- **Previous Defect**: Early summaries claimed broad spatial generalization without disclosing that models with temporal lag features degrade when an unseen well has zero historical measurements ($N_{\text{prior}} = 0$).
- **Correction Applied**: Separated unseen-well test evaluations into Cold-Start ($N_{\text{prior}} = 0$), Warm-Start ($N_{\text{prior}} \ge 1$), and Overall cohorts. Fully disclosed that AMDFE's core advantage is concentrated in warm-start and sparse-monitoring regimes rather than zero-history cold-start extrapolation.

---

## 3. Automated Validation & Test Suite

To prevent future regression or manual reporting drift, the following automated validation scripts and test suites were implemented:
1. `ml/experiments/validate_amdfe_v21_statistics.py`: Automated sanity checker verifying prediction reproducibility, formula consistency, and CI bounds. Exits with non-zero code on any discrepancy.
2. `ml/tests/test_amdfe_v21.py`: Unit test suite covering AMDFE-V21-01 through AMDFE-V21-15.
