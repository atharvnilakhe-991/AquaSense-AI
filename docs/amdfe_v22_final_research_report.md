# Adaptive Multimodal Data Fusion Engine (AMDFE v2.2): Final Frozen Methodology & Comprehensive Research Report

---

## 1. Problem
Regional groundwater table depth estimation is fundamentally constrained by observational heterogeneity across monitoring networks. Groundwater observation records frequently suffer from irregular sampling intervals (spanning weeks to multiple years), data gaps, missing meteorological inputs, and variable monitoring density. Standard machine learning workflows either discard incomplete records, apply static unweighted early concatenation, or attempt uniform imputation, leading to error amplification when observation gaps occur.

---

## 2. Motivation
Conventional multimodal fusion methods treat all input channels with uniform structural confidence or rely on complex neural attention architectures that overfit when trained on sparse, irregular tabular monitoring data. The central motivation of AMDFE is to provide a lightweight, deterministic, and leakage-safe preprocessing layer that implements **sample-adaptive modality block gating** based on intrinsic data source quality and sample-specific observation latency.

---

## 3. Literature Positioning
Recent literature (2024–2026) highlights two key themes:
1. **Entity-Aware Regional ML**: Global models (e.g., Wunsch et al., 2024) predict across multiple wells but experience substantial performance drops on held-out spatial entities.
2. **Multi-View Attention**: Neural gating units (e.g., Zhang et al., 2025) fuse gridded satellite and meteorological rasters but require dense, uniform temporal grids.
AMDFE positions itself as a **tabular-native, decoupled two-stage fusion mechanism** ($R_m \times A_{i,m}$) tailored specifically for irregular, non-gridded groundwater monitoring records.

---

## 4. AMDFE Definition
AMDFE is a **modular feature preprocessing and fusion engine** that:
- Evaluates source-level intrinsic reliability ($R_m$) on training data.
- Evaluates sample-level context observability ($A_{i,m}$) dynamically per observation.
- Synthesizes normalized channel weights ($w_{i,m}$).
- Applies sample-adaptive modality block gating to standardized feature blocks ($w_{i,m} \cdot Z(X_{i,m})$) prior to estimator training.

AMDFE is **NOT** a standalone forecasting model, physical hydrodynamic simulator, recharge estimator, or telemetric hardware interface.

---

## 5. Mathematical Formulation
- **Source Reliability**:
  $$R_m = \left( \prod_{d \in \mathcal{Q}_m} Q_{m,d} \right)^{1 / |\mathcal{Q}_m|}$$
  *Rationale*: The geometric mean provides a non-compensatory aggregation index where severe failure in any fundamental quality dimension (e.g., validity $\to 0$) heavily penalizes the source score $R_m$, maintaining scale-invariance without claiming unique mathematical optimality.
- **Context Observability**:
  $$A_{i,A} = \begin{cases} 0.10 & \text{if } N_{\text{prior}} = 0 \\ \mathrm{clip}\left(0.20 + 0.50 \exp\left(-\frac{\Delta t_i}{365.25}\right) + 0.30 \min\left(\frac{N_{\text{prior}}}{3}, 1\right), \, 0.05, \, 1.0\right) & \text{if } N_{\text{prior}} \ge 1 \end{cases}$$
- **Effective Trust**: $G_{i,m} = R_m \cdot A_{i,m}$
- **Normalized Weight**: $w_{i,m} = \frac{M_{i,m} G_{i,m}}{\sum_k M_{i,k} G_{i,k}}$
- **Sample-Adaptive Modality Block Gating**: $F_{i,m,j} = w_{i,m} \cdot \left(\frac{X_{i,m,j} - \mu_{m,j}^{\text{train}}}{\sigma_{m,j}^{\text{train}}}\right)$

---

## 6. Data Modalities & Scope
1. **Modality A (Groundwater State)**: 16 features (lags, rolling statistics, time differences, historical summary statistics, observation density).
2. **Modality B (Weather Context)**: 5 features (completed prior calendar year $Y-1$ temperature, precipitation, humidity, wind, solar radiation).
3. **Modality C (Spatial Context)**: 3 features (`LatDD`, `LongDD`, `Surf_Elev`).
4. **Modality D (Earth Observation & SoilGrids)**: **NOT CURRENTLY INTEGRATED** ($M_D = 0, w_D = 0.0$; unlinked in current repository).

---

## 7. Source Reliability
Computed across completeness, validity, non-duplication, outlier regularity, temporal coverage, and spatial coverage on training data:
- $R_{\text{GW}} = 0.9244$ (or $0.9995$ on reference full partition)
- $R_{\text{Weather}} = 0.9880$
- $R_{\text{Spatial}} = 1.0000$

---

## 8. Context Observability
Point-in-time observation freshness factor $A_{i,m} \in [0, 1]$ adapts dynamically to sampling latency $\Delta t_i$ without lookahead. Extended multi-year gaps attenuate groundwater weight, shifting reliance to weather and topographic elevation.

---

## 9. Adaptive Weighting
Normalized weights $w_{i,m}$ guarantee $\sum_m w_{i,m} = 1.0$. If a modality is missing ($M_{i,m}=0$), $w_{i,m} = 0.0$, enforcing zero-trust without fabricating synthetic values.

---

## 10. Fusion Mechanism
Gating is applied to zero-mean, unit-variance standardized features. In decision trees, sample-dependent scaling alters sample ordering, compressing degraded modality variance toward zero and guiding tree splits toward reliable channels.

---

## 11. Missing-Data Handling
Median imputers and robust MAD outlier scalers are fit strictly on training partitions. Binary missingness indicators and fallback flags ensure deterministic execution without NaNs.

---

## 12. Leakage Prevention
- Target-derived rasters (`TIFF_Value`, `groundwater_depth_raster`) are completely purged.
- Weather records strictly utilize completed calendar year $Y-1$.
- Imputers, scalers, and quantile boundaries are fit on training folds only.

---

## 13. Experimental Design
The validation suite evaluates controlled matched-feature ablations across 5 random seeds:
- **C0**: Unweighted Base (23 features)
- **C1**: Matched Base + Metadata Control (32 features, ungated)
- **C2**: Core Context-Adaptive AMDFE (32 features, gated)
- **A0**: Groundwater Only (15 features)
- **A1**: Groundwater + Weather (20 features)
- **A6**: Core AMDFE + Robustness Extended (60 features)

---

## 14. Temporal Forward Evaluation (E1)
Evaluated on strict chronological split: **Train $\le 2017$** (2,996 rows), **Val 2018–2020** (347 rows), **Test 2021–2024** (501 rows across 121 active wells).
- LightGBM $C0$: $\text{RMSE} = 3.356\text{ m}, R^2 = 0.9972$
- LightGBM $C1$: $\text{RMSE} = 3.268\text{ m}, R^2 = 0.9974$
- LightGBM $C2$: $\text{RMSE} = 3.991\text{ m}, R^2 = 0.9961$
- Random Forest $C1$: $\text{RMSE} = 3.035\text{ m}, R^2 = 0.9977$
- **Interpretation**: On dense, regular historical data, ungated and metadata-augmented models perform on par with or marginally better than adaptive gating.

---

## 15. Unseen-Well Warm-Start Evaluation (E2)
On 3,738 unseen-well observations with prior history ($N_{\text{prior}} \ge 1$):
- LightGBM $C0$: $\text{RMSE} = 9.005\text{ m}, R^2 = 0.9773$
- LightGBM $C1$: $\text{RMSE} = 8.262\text{ m}, R^2 = 0.9809$
- LightGBM $C2$: $\text{RMSE} = \mathbf{5.208\text{ m}}, R^2 = 0.9924$
- LightGBM $A6$: $\text{RMSE} = \mathbf{4.036\text{ m}}, R^2 = 0.9954$
- **Statistical Inferential Bound**: Comparing $C1$ vs $C2$ on warm-start yields mean $\Delta\text{RMSE} = -1.387\text{ m}$ (point estimate favorable), with a 95% well-clustered CI of $[-4.483\text{ m}, +2.322\text{ m}]$ ($p=0.288$). This is classified as **directionally favorable but not statistically conclusive at $\alpha=0.05$**.

---

## 16. Unseen-Well Cold-Start Evaluation (E3)
On 170 unseen-well observations with zero prior history ($N_{\text{prior}} = 0$):
- Single-modality $A0$: $\text{RMSE} = 55.488\text{ m}, R^2 = -0.0128$ (Total collapse)
- Static spatial $C0$: $\mathbf{\text{RMSE} = 17.417\text{ m}}, R^2 = 0.9002$ (Spatial coords anchor predictions)
- Core AMDFE $C2$: $\text{RMSE} = 21.506\text{ m}, R^2 = 0.8478$
- **Finding**: AMDFE does not solve cold-start spatial extrapolation.

---

## 17. Monitoring-Density Cohorts (E4)
- **Low Density ($N_{\text{prior}} \le 1$)**: $C0\text{ RMSE} = 14.29\text{ m} \to C2\text{ RMSE} = 10.25\text{ m} \to A6\text{ RMSE} = 9.47\text{ m}$ ($>33\%$ error reduction).
- **Medium Density ($1 < N_{\text{prior}} \le 5$)**: $C0\text{ RMSE} = 5.39\text{ m} \to C2\text{ RMSE} = 4.23\text{ m}$.
- **High Density ($N_{\text{prior}} > 5$)**: $C0\text{ RMSE} = 4.41\text{ m} \to C2\text{ RMSE} = 3.59\text{ m}$.

---

## 18. Gap-Length Cohorts (E5)
- **Short Gap ($\le 180$ d)**: $C0\text{ RMSE} = 5.87\text{ m} \to C2\text{ RMSE} = 4.05\text{ m}$.
- **Moderate Gap (180–365 d)**: $C0\text{ RMSE} = 6.57\text{ m} \to C2\text{ RMSE} = 5.12\text{ m}$.
- **Long Gap ($> 365$ d)**: $C0\text{ RMSE} = 18.85\text{ m} \to C2\text{ RMSE} = 11.52\text{ m} \to A6\text{ RMSE} = 10.72\text{ m}$ (Mean $\Delta\text{RMSE} = -1.759\text{ m}$, 95% CI: $[-4.664\text{ m}, +1.314\text{ m}]$, $p=0.303$).

---

## 19. Controlled Observation-Loss Stress Experiments (E6, E7, E8)
*Note: Evaluated under controlled synthetic degradation masks with verified clean data SHA-256 immutability.*
- **Weather Pointwise Loss (60%)**: $C0$ degrades by $+14.5\%$ vs **$+1.9\%$** for $C2$.
- **Groundwater-History Loss (60%)**: $C0$ degrades by $+193.0\%$ vs **$+55.9\%$** for $C2$ and **$+44.5\%$** for $A6$.
- **Complete Groundwater-History Outage (100%)**: $C0$ degrades by $+366.1\%$ vs **$+163.7\%$** for $C2$.
- **Clean SHA-256 Immutability**: Verified invariant (`40cccd4ef142c27c...`).

---

## 20. Statistical Significance & Cluster Bootstrap
- **$C2$ vs $C1$ (Adaptive Gating Marginal Effect, 32 vs 32 features)**:
  - Baseline $C1$: $7.389\text{ m}$, Candidate $C2$: $6.535\text{ m}$
  - Observed $\Delta\text{RMSE} = \mathbf{-0.854\text{ m}}$, Improvement: $\mathbf{+11.56\%}$
  - 95% Cluster-Bootstrap CI: $[-4.072\text{ m}, +2.694\text{ m}]$ ($p = 0.170$).
  - Standard deviation across seeds drops from $5.75\text{ m}$ ($C1$) to $1.93\text{ m}$ ($C2$).
  - Classification: **Directionally favorable but not statistically conclusive at $\alpha=0.05$**.

---

## 21. Model-Independence Synthesis (E10)
- **LightGBM**: $C0\text{ RMSE} = 7.47\text{ m} \to C2\text{ RMSE} = 6.53\text{ m} \to A6\text{ RMSE} = 5.93\text{ m}$.
- **Random Forest**: $C0\text{ RMSE} = 5.45\text{ m} \to A6\text{ RMSE} = 5.41\text{ m}$ (Run std drops from $1.82\text{ m}$ to $0.83\text{ m}$).
- **XGBoost**: $C0\text{ RMSE} = 4.86\text{ m} \to A6\text{ RMSE} = 4.80\text{ m}$ (Lowest MAE = $2.23\text{ m}$).

---

## 22. Failure Cases
1. **Cold-Start Piezometers ($N_{\text{prior}}=0$)**: Without prior history, autoregressive lag channels collapse.
2. **Decision Tree Global Scale Invariance**: Constant global scalars do not modulate tree splits; sample-level variability is required.

---

## 23. Limitations
- Modality D (Earth Observation rasters and SoilGrids hydraulic conductivity) is unlinked in the present repository.
- Physical mass conservation is not mathematically enforced.

---

## 24. Prior-Art Boundary
AMDFE does not claim novelty for multimodal concatenation, missingness flags, or adaptive loss weighting. Its distinction lies in **decoupling intrinsic reliability from point-in-time observability for deterministic sample-adaptive modality block gating in tabular hydrogeological records**.

---

## 25. Empirically Supported Contribution
1. **Attenuated Degradation under Monitoring-Data Loss**: Substantial reduction in relative error amplification under controlled 20%–60% data loss experiments.
2. **Sparse Cohort Accuracy**: Directional error reduction during multi-year monitoring gaps ($>365\text{ d}$).
3. **Tail Error & Variance Stabilization**: P95 tail errors reduced from $>20.57\text{ m}$ to $<10.88\text{ m}$.

---

## 26. Unsupported Claims (Strictly Prohibited)
- *"AMDFE solves spatial cold-start extrapolation."*
- *"AMDFE integrates satellite imagery and digital soil maps."*
- *"AMDFE is a physics-informed mass-conserving model."*
- *"AMDFE is statistically proven superior across all models on dense clean data."*
- *"AMDFE is tested on live telemetric sensor hardware."*

---

## 27. Reproducibility Information
- **Pipeline Scripts**: `ml/amdfe/v2_pipeline.py`, `ml/experiments/run_amdfe_v21_benchmark.py`
- **Validation Script**: `ml/experiments/validate_amdfe_v22_logic.py`
- **Statistical Checker**: `ml/experiments/validate_amdfe_v21_statistics.py`
- **Test Suite**: 40 unit tests passing 100% across `ml/tests/`.

---

## 28. Final Conclusion & Methodology Freeze Decision
Under the formal decision tree protocol:
**Outcome**: **Case 2 / Case 3**.
- On clean benchmark data, adaptive gating is **directionally favorable** ($\Delta\text{RMSE} = -0.854\text{ m}$, $+11.56\%$ error reduction) and stabilizes run variance ($5.75\text{ m} \to 1.93\text{ m}$), with the 95% bootstrap CI containing zero.
- Under synthetic monitoring-data loss, observation outages, and multi-year gaps, AMDFE provides **controlled resilience and error attenuation**.

**The AMDFE v2.2 methodology is permanently frozen.**
