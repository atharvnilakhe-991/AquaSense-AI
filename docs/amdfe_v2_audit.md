# AMDFE Version 2 Critical Audit & Research Hardening Plan
**AquaSense-AI Research Pipeline**
**Branch**: `ml-training`
**Date**: September 2026

---

## 1. Executive Summary

This document conducts a critical scientific audit of the initial AMDFE (v1) implementation and outlines the methodological hardening required for peer-reviewed journal submission (SCI / Q1 hydroinformatics & applied environmental AI literature). 

The goal of AMDFE v2 is **not** to artificially optimize RMSE or claim nominal novelty, but to establish a mathematically rigorous, leakage-safe, and empirically testable framework that addresses **sparse, irregular, and heterogeneous monitoring** in groundwater systems.

---

## 2. What AMDFE Currently Does (v1 Implementation)

1. **Multimodal Ingestion & Causal Weather Policy**:
   - Maps groundwater observations to completed prior calendar year weather ($Y-1$), preventing mid-year lookahead leakage.
   - Purges target-derived Kriging rasters (`TIFF_Value`).
2. **Quality & Reliability Scoring**:
   - Assesses 4 quality dimensions: Completeness ($C$), Temporal Consistency ($T$), Spatial Coverage ($S$), and Observation Density ($O$).
   - Aggregates source reliability via geometric mean: $R_{\text{global}, m} = (C \cdot T \cdot S \cdot O)^{1/4}$.
3. **Dynamic Sample Weighting & Block Standardization**:
   - Fits standardizers on the training partition only.
   - Modulates global reliability by sample availability $A_{i,m}$ using an empirical heuristic:
     $$A_{i, A} = 0.4 + 0.4 \exp(-\Delta t_i / 730) + 0.2 \min(1.0, N_{\text{prior}, i} / 5)$$
   - Computes normalized sample weights $w_{i,m} = R_{i,m} / \sum_k R_{i,k}$.
   - Scales standardized feature blocks: ${X'}_{i,m} = w_{i,m} \cdot Z(X_{i,m})$.
4. **Validation & Leakage Checks**:
   - Executes 10 automated leakage checks (AMDFE-LC-01 through LC-10).
   - Evaluates unseen-well spatial generalization across configurations B0–B4.

---

## 3. Scientific Weaknesses of AMDFE v1

| Area | Scientific Weakness in v1 | Methodological Correction in v2 |
| :--- | :--- | :--- |
| **Concept Conflation** | Conflated **Source Reliability** (intrinsic trustworthiness of a data source) with **Context Observability** (per-sample monitoring availability and gap state). | Formulate strict, independent mathematical definitions: $R_m$ (Source Reliability) vs. $A_{i,m}$ (Context Observability). |
| **Non-Applicable Dimensions** | Non-applicable quality dimensions for static/regional modalities (e.g. observation density for static spatial coords) were assigned default 1.0 or 0.5 without formal masking. | Mask non-applicable quality dimensions out of source reliability aggregation without penalizing or artificially inflating the score. |
| **Heuristic Formula** | The row availability formula $0.4 + 0.4 \exp(-\Delta t / 730) + 0.2 \min(N / 5)$ relied on arbitrary magic constants without justification. | Replace with transparent two-stage formulation $G_{i,m} = R_m^\alpha \cdot A_{i,m}^\beta$, normalized dynamically, with sensitivity evaluation over $\alpha, \beta$. |
| **Tree-Model Scaling Effect** | For tree models (LightGBM, XGBoost, RF), multiplying an entire feature block by a constant positive multiplier does not alter split points or rankings. | Implement a 4-tier control suite: unweighted concatenation vs. global scaling vs. metadata augmentation vs. sample-adaptive gating. |
| **Warm/Cold-Start Conflation** | Grouped unseen-well evaluation evaluated all test observations together, blending warm-start rows (which have historical well records) with true cold-start rows (first-time well observations). | Explicitly separate and report: (1) Temporal known-well, (2) Unseen-well warm-start, (3) Unseen-well cold-start. |
| **Degradation Testing** | Robustness stress test only evaluated random pointwise feature masking. | Implement 3 distinct degradation modes: Pointwise random loss, Burst temporal gap loss, and Source outage across multiple severities (20%, 40%, 60%). |
| **Statistical Rigor** | Reported standard deviation across seeds, but lacked cluster bootstrap by well and confidence intervals for paired differences. | Implement cluster bootstrap by well (accounting for intra-well correlation) and compute 95% CIs and Cohen's $d$ effect sizes. |
| **Terminology Precision** | Terminology previously referred to Pythagorean means and described `DEM_Elevation` as target-derived. | Correct terminology: Means are aggregation formulations; `DEM_Elevation` is an independent terrain feature excluded due to missingness in the repo, whereas `TIFF_Value` is target-derived. |

---

## 4. Strong Components Preserved from v1

1. **Strict Causal Weather Policy ($Y-1$)**:
   - Zero lookahead from uncompleted calendar-year meteorology is fully verified.
2. **Train-Only Parameter Fitting**:
   - Imputers, scalers, and quality baselines are fitted strictly on the training partition, with frozen transform on test sets.
3. **Automated Leakage Assertion Infrastructure**:
   - Automated checks ensuring no target `WatLevel` or Kriging rasters enter predictor matrices.
4. **Modality Definitions & Absence of Fake Data**:
   - Satellite and SoilGrids data remain marked as `UNAVAILABLE_IN_CURRENT_REPOSITORY` without synthetic fabrication.

---

## 5. AMDFE v2 Implementation Architecture

```
========================================================================================
                                AMDFE v2 ARCHITECTURE
========================================================================================

    [MODALITY A: GW History]    [MODALITY B: Weather (Y-1)]    [MODALITY C: Spatial]
               |                            |                             |
               +----------------------------+-----------------------------+
                                            |
                                            v
               [1. SOURCE RELIABILITY ENGINE: R_m (Train-Fitted Quality)]
                   - Completeness (C)
                   - Data Validity (V)
                   - Duplicate Integrity (D)
                   - Outlier Rate (O_rate)
                   - Temporal Coverage (T)
                   - Spatial Coverage (S)
                   (Non-applicable dimensions masked out; no fabricated metadata)
                                            |
                                            v
               [2. CONTEXT OBSERVABILITY ENGINE: A_{i,m} (Point-in-Time)]
                   - Prior observation count (N_{prior,i})
                   - Elapsed time / Gap interval (Delta t_i)
                   - Long gap / burst gap flags
                   - Meteorological availability flag
                   - Geodetic coordinate validity
                                            |
                                            v
               [3. ADAPTIVE FUSION ENGINE: G_{i,m} = R_m^alpha * A_{i,m}^beta]
                   - Dynamic weight: w_{i,m} = G_{i,m} / sum_k G_{i,k}
                   - Modality standardization: Z(X_{i,m})
                   - Adaptive gating: X'_{i,m} = w_{i,m} * Z(X_{i,m})
                   - Traceable metadata: w_{i,m}, R_m, A_{i,m}, missingness flags
                                            |
                                            v
               [4. ABLATION SUITE (A0 - A6) & MULTI-MODEL BENCHMARK]
                   - A0: Groundwater Only
                   - A1: Groundwater + Weather
                   - A2: Unweighted Multimodal Concatenation
                   - A3: Reliability Metadata Augmentation (No Gating)
                   - A4: Global Reliability Scaling
                   - A5: Context-Adaptive AMDFE
                   - A6: AMDFE + Robustness Metadata
========================================================================================
```

---

## 6. Git Safety & Versioning Policy

- All v1 results in `data/processed/amdfe/` remain strictly untouched.
- All historical results in `data/processed/advanced_ml/` remain strictly untouched.
- AMDFE v2 outputs will be written to a dedicated directory: `data/processed/amdfe_v2/`.
- Extended unit/integration tests will be added to `ml/tests/test_amdfe_v2.py`.
