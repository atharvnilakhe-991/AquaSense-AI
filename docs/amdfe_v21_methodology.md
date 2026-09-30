# AMDFE v2.1 Methodology: Controlled Attribution, Observability & Validation

## 1. Executive Overview

The Adaptive Multimodal Data Fusion Engine (AMDFE) v2.1 represents a methodologically hardened, statistically corrected formulation for multimodal groundwater level forecasting under sparse, irregular, and heterogeneous monitoring regimes.

In previous iterations (v1 and v2), three critical methodological confounders were identified:
1. **Feature-Count Confounding**: Comparing unweighted baselines ($A2 = 23$ features) directly to adaptive fusion ($A5 = 32$ features, $A6 = 60$ features) entangled the effect of feature count with the effect of adaptive gating.
2. **Bootstrap Delta Convention Inconsistency**: An inversion in delta definitions led to discordant reports between main metric tables and bootstrap distribution summaries.
3. **Observability Formulation Heuristics**: Observability weights and decay constants lacked systematic parameter sensitivity bounds.

AMDFE v2.1 resolves all three challenges through:
- **Strict Matched-Feature Controls ($C0, C1, C2$)**: Isolating adaptive sample-level gating from metadata column additions while preserving identical 32-column representations.
- **Explicit Delta Conventions**: Enforcing uniform mathematical sign conventions ($\Delta\text{RMSE} = \text{RMSE}_{\text{method}} - \text{RMSE}_{\text{baseline}}$, $\text{Improvement \%} = 100 \times (\text{RMSE}_{\text{baseline}} - \text{RMSE}_{\text{method}}) / \text{RMSE}_{\text{baseline}}$).
- **Observability Parameter Sensitivity Grids**: Evaluating temporal decay scales ($180, 365.25, 730$ days), count saturations ($2, 3, 5$ observations), and weight allocations on training/validation folds before frozen test deployment.
- **Strict Cluster Bootstrap by Held-Out Well**: Resampling identical test wells over 1,000 replicates.

---

## 2. Modality Taxonomy & Feature Space

| Modality Key | Modality Name | Base Features | Description | Status |
|---|---|---|---|---|
| **Modality A** | Groundwater Temporal History & State | 15 | Lags (`Previous_WatLevel`, `Previous2_WatLevel`), rolling statistics (`Rolling_Mean_3`, `Rolling_Std_3`), gap descriptors (`Days_Since_Previous`, `Years_Since_Previous`, `Long_Gap_Flag`, `Very_Long_Gap_Flag`), historical summary statistics (`Historical_Mean`, `Historical_Std`, `Historical_Min`, `Historical_Max`), and observation density | Active |
| **Modality B** | Environmental & Weather Context | 5 | Leakage-safe completed period features from Year $Y-1$: `Annual_Temperature_Mean`, `Annual_Precipitation_Total`, `Annual_Humidity_Mean`, `Annual_WindSpeed_Mean`, `Annual_SolarRadiation_Mean` | Active |
| **Modality C** | Spatial & Topographic Context | 3 | Static geospatial coordinates and surface altitude: `LatDD`, `LongDD`, `Surf_Elev` | Active |
| **Modality D** | Earth Observation & SoilGrids | 0 | Satellite optical/SAR reflectance and physical soil hydraulic properties | UNAVAILABLE (Non-fabricated) |

---

## 3. Mathematical Formulation of AMDFE v2.1

### Stage 1: Global Source Reliability ($R_m$)
Evaluated across six applicable data quality dimensions on the training partition:
$$R_m = \left( \prod_{d \in \mathcal{D}_m} Q_{m,d} \right)^{1 / |\mathcal{D}_m|}$$
where $Q_{m,d} \in [0, 1]$ represents completeness, validity, non-duplication, outlier quality, temporal coverage, and spatial coverage.

### Stage 2: Sample-Level Context Observability ($A_{i,m}$)
Point-in-time observation availability for sample $i$ and modality $m$:
- **Modality A (Groundwater)**:
  $$A_{i,A} = \begin{cases} 
  0.10 & \text{if } N_{\text{prior}} = 0 \text{ (Cold-Start)} \\
  w_{\text{base}} + w_{\text{gap}} \exp\left(-\frac{\Delta t_i}{\tau}\right) + w_{\text{count}} \min\left(\frac{N_{\text{prior}}}{S}, 1.0\right) & \text{if } N_{\text{prior}} \ge 1 \text{ (Warm-Start)}
  \end{cases}$$
  where default frozen parameters are $\tau = 365.25\text{ days}$, $S = 3\text{ observations}$, $w_{\text{base}} = 0.20$, $w_{\text{gap}} = 0.50$, $w_{\text{count}} = 0.30$.
- **Modality B (Weather)**: $A_{i,B} \in \{0.0, 1.0\}$ indicating completed annual record availability.
- **Modality C (Spatial)**: $A_{i,C} \in \{0.0, 1.0\}$ indicating coordinate validity.

### Stage 3: Two-Stage Adaptive Modality Weights ($w_{i,m}$)
$$G_{i,m} = (R_m)^\alpha \cdot (A_{i,m})^\beta$$
$$w_{i,m} = \frac{G_{i,m}}{\sum_{k} G_{i,k}}$$
Default exponent scaling is $\alpha = 1.0, \beta = 1.0$.

### Stage 4: Standardized Block Feature Gating
For each standardized feature $Z(X_{i,m,j})$:
$$F_{i,m,j} = w_{i,m} \cdot Z(X_{i,m,j})$$

---

## 4. Matched-Feature Controlled Ablation Suite

To completely eliminate feature-count confounding, AMDFE v2.1 establishes the following experimental configurations:

| Config | Name | Base Features | Reliability Meta ($R_m$) | Observability Meta ($A_{i,m}$) | Weight Meta ($w_{i,m}$) | Gated Base Features | Total Features | Purpose |
|---|---|---|---|---|---|---|---|---|
| **C0** | Unweighted Base ($A2$) | 23 | 0 | 0 | 0 | No ($1.0 \cdot Z(X)$) | **23** | Standard concatenation baseline |
| **C1** | Base + Full Metadata | 23 | 3 | 3 | 3 | No ($1.0 \cdot Z(X)$) | **32** | Metadata addition without gating |
| **C2** | Core Context-Adaptive AMDFE | 23 | 3 | 3 | 3 | Yes ($w_{i,m} \cdot Z(X)$) | **32** | **Isolated marginal effect of gating** |
| **A0** | Groundwater Only | 15 | 0 | 0 | 0 | No | **15** | Single modality temporal state |
| **A1** | Groundwater + Weather | 20 | 0 | 0 | 0 | No | **20** | Dual modality unweighted |
| **A6** | Core AMDFE + Robustness Meta | 23 | 3 | 3 | 3 | Yes | **60** | Gated + 28 missingness/gap indicators |

### Key Experimental Hypothesis:
The true marginal attribution of the adaptive gating mechanism is measured solely by:
$$\Delta\text{RMSE}_{\text{gating}} = \text{RMSE}_{C2} - \text{RMSE}_{C1}$$
where $C1$ and $C2$ have the exact same 32 columns.
