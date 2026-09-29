# AMDFE Research Positioning & Scientific Framing
**AquaSense-AI Research Pipeline**
**Target**: SCI / Q1 Hydroinformatics & Applied AI Literature
**Date**: September 2026

---

## 1. Research Positioning Statement

> **"AMDFE is the proposed framework whose distinctiveness must be established empirically and through prior-art comparison."**

The objective of AMDFE is not to assert nominal novelty, but to provide a mathematically transparent, experimentally validated, and leakage-safe preprocessing/fusion methodology for heterogeneous environmental time series.

---

## 2. Distinction of Pipeline Layers

To maintain rigorous scientific reporting standards, the contribution of AMDFE is explicitly categorized across four operational tiers:

```
+--------------------------------------------------------------------------------------------------+
| TIER 1: ENGINEERING IMPLEMENTATION                                                               |
| - Scikit-learn compatible stateful pipeline (AMDFEPipeline).                                     |
| - Automated 10-point leakage validation suite (AMDFE-LC-01 to AMDFE-LC-10).                      |
| - Strict train-only fitting and frozen parameter transformation.                                  |
| - Reproducible benchmark execution and multi-seed repeated evaluation harness.                   |
+--------------------------------------------------------------------------------------------------+
| TIER 2: PROPOSED METHODOLOGICAL CONTRIBUTION                                                     |
| - Systematic Pythagorean quality aggregation for global source reliability R_{global,m}.         |
| - Context-aware sample availability modulation A_{i,m} reflecting monitoring gaps and cold-starts|
| - Partition-of-unity dynamic modality weighting w_{i,m} = R_{i,m} / \sum R_{i,k}.                |
| - Causal lagging policy for calendar-year meteorological aggregates (Y - 1).                     |
+--------------------------------------------------------------------------------------------------+
| TIER 3: EXPERIMENTALLY DEMONSTRATED CONTRIBUTION                                                 |
| - Empirical ablation comparison isolating fixed vs reliability vs adaptive fusion (B0 - B4).     |
| - Generalization performance on completely unseen wells (80/20 spatial holdout).                 |
| - Robustness under controlled synthetic degradation stress tests (Hypothesis H5).                |
| - Quantified uncertainty and effect size metrics across multiple random seeds.                   |
+--------------------------------------------------------------------------------------------------+
| TIER 4: FUTURE RESEARCH CLAIMS                                                                   |
| - Integration of sub-annual / daily ERA5-Land reanalysis products.                               |
| - Assimilation of Sentinel-2 optical indices and GRACE terrestrial water storage anomalies.     |
| - Physics-informed neural differential equation (PINN) coupling for aquifer hydraulic heads.    |
+--------------------------------------------------------------------------------------------------+
```

---

## 3. Prior-Art Context & Comparative Taxonomy

In contemporary machine learning for groundwater depth estimation, multimodal data integration strategies generally fall into three paradigms:

1. **Early Fusion (Feature Concatenation)**:
   - *Mechanism*: Concatenates all available raw or normalized features into a single matrix $\mathbf{X} = [\mathbf{X}_A, \mathbf{X}_B, \mathbf{X}_C]$.
   - *Limitation*: Fails to account for heterogeneous observational reliability or monitoring gaps; treats high-noise modalities identically to dense measurements.
2. **Late Fusion (Decision / Ensemble Level)**:
   - *Mechanism*: Trains separate sub-models on individual modalities and blends their output predictions $\hat{y} = \sum \alpha_m \hat{y}_m$.
   - *Limitation*: Destroys cross-modal feature interactions (e.g., the physical coupling between precipitation and antecedent water table depth).
3. **AMDFE (Adaptive Intermediate Fusion)**:
   - *Mechanism*: Standardizes modality blocks on training statistics, applies dynamic sample-level reliability weighting $w_{i,m} \cdot Z(\mathbf{X}_{i,m})$, and preserves cross-modal feature interactions while dynamically attenuating degraded or cold-start modalities.

---

## 4. Testable Scientific Hypotheses

The experimental evaluation of AMDFE is structured around five formal hypotheses:

- **Hypothesis H1 (Reliability Superiority)**:
  *Reliability-weighted multimodal fusion (B3) demonstrates superior or equivalent predictive accuracy compared to equal fixed-weight fusion (B2).*
- **Hypothesis H2 (Adaptive Context Robustness)**:
  *Context-adaptive AMDFE (B4) improves prediction stability and reduces high-error tails (P90/P95 error) when monitoring intervals and data availability vary across wells.*
- **Hypothesis H3 (Spatial Generalization)**:
  *AMDFE maintains or improves predictive performance on geographically distinct, unseen monitoring wells under strict spatial holdout.*
- **Hypothesis H4 (Monitoring-Density Resilience)**:
  *AMDFE reduces error variance across sparse vs dense observation cohorts by dynamically adjusting modality contributions for cold-start and long-gap records.*
- **Hypothesis H5 (Controlled Degradation Behavior)**:
  *Under controlled synthetic missingness stress tests, adaptive weighting degrades gracefully, maintaining lower RMSE increases than unweighted or fixed baselines.*

---

## 5. Honest Disclosure of Empirical Limitations

1. **Annual Weather Temporal Granularity**:
   - The currently available meteorological table consists of calendar-year aggregates. While the $Y-1$ causal policy guarantees zero temporal leakage, annual aggregates attenuate short-term storm recharge responses.
2. **Homogeneous Domain Characteristics**:
   - The current dataset represents a specific alluvial basin (170 wells). Transferability to fractured bedrock or karst aquifer systems requires multi-basin validation once external data repositories are ingested.
3. **Sensor Calibration Data**:
   - Sensor precision metadata was unavailable, necessitating reliance on observational completeness, temporal spacing, and density as primary empirical proxies for data quality.
