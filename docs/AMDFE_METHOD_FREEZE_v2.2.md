# AMDFE METHODOLOGY FREEZE — VERSION 2.2
**FINAL METHODOLOGICAL & SCIENTIFIC SPECIFICATION**

---

## 1. Status of Methodology
**STATUS**: **PERMANENTLY FROZEN**.

As of Version 2.2, the mathematical formulation, feature allocation, reliability scoring, context observability equations, sample-adaptive modality block gating, and experimental evaluation protocols of the Adaptive Multimodal Data Fusion Engine (AMDFE) are permanently frozen.

No further modifications to the core fusion architecture, feature gating logic, or metric definitions are permitted unless:
1. New empirical physical modalities (e.g. validated Sentinel-2 satellite imagery) are integrated into the repository.
2. An unhandled data corruption or numerical failure is identified.
3. A formal peer-review critique requires a pre-specified ablation extension.

---

## 2. Canonical Mathematical Formulae

### 2.1 Intrinsic Source Reliability ($R_m$)
Evaluated exclusively on the training partition $\mathcal{D}_{\text{train}}$ across applicable quality dimensions $\mathcal{Q}_m$:
$$R_m = \left( \prod_{d \in \mathcal{Q}_m} Q_{m,d} \right)^{1 / |\mathcal{Q}_m|}$$
*Rationale*: Provides a non-compensatory aggregation index where severe failure in any fundamental quality dimension (e.g., validity $\to 0$) heavily penalizes the source score $R_m$, maintaining scale-invariance without claiming unique optimality.

### 2.2 Point-in-Time Context Observability ($A_{i,m}$)
Evaluated point-in-time for sample $i$ using strictly historical metadata prior to timestamp $t_i$:
$$A_{i,A} = \begin{cases} 
0.10 & \text{if } N_{\text{prior}} = 0 \\
\mathrm{clip}\left(0.20 + 0.50 \exp\left(-\frac{\Delta t_i}{365.25}\right) + 0.30 \min\left(\frac{N_{\text{prior}}}{3}, 1\right), \, 0.05, \, 1.0\right) & \text{if } N_{\text{prior}} \ge 1
\end{cases}$$
$$A_{i,B} = \mathbb{I}(\text{Weather Features Present for Year } Y_i - 1) \in \{0.0, 1.0\}$$
$$A_{i,C} = \mathbb{I}(\text{Coordinates } (\text{Lat}, \text{Long}, \text{Elev}) \text{ Valid}) \in \{0.0, 1.0\}$$
$$A_{i,D} = 0.0 \quad (\text{Enforced Zero-Trust for Unavailable Data})$$

### 2.3 Effective Trust Score ($G_{i,m}$)
$$G_{i,m} = (R_m)^\alpha \cdot (A_{i,m})^\beta \quad \text{with } \alpha = 1.0, \, \beta = 1.0 \implies G_{i,m} = R_m \cdot A_{i,m}$$

### 2.4 Normalized Dynamic Channel Weights ($w_{i,m}$)
$$w_{i,m} = \frac{M_{i,m} \cdot G_{i,m}}{\sum_{k} M_{i,k} \cdot G_{i,k}}$$
- $M_{i,m} = 0 \implies w_{i,m} = 0$.
- $\sum_m w_{i,m} = 1.0$ across active modalities.

### 2.5 Sample-Adaptive Modality Block Gating
$$F_{i,m,j} = w_{i,m} \cdot Z(X_{i,m,j}) = w_{i,m} \cdot \left(\frac{X_{i,m,j} - \mu_{m,j}^{\text{train}}}{\sigma_{m,j}^{\text{train}}}\right)$$

---

## 3. Canonical Feature Groups & Configurations

| Configuration | Feature Set Description | Gated Base Features | Metadata (Ungated) | Total Features |
|---|---|---|---|---|
| **C0** | Unweighted Multimodal Base ($A2$) | No ($1.0 \cdot Z$) | None | **23** |
| **C1** | Matched Base + Metadata Control | No ($1.0 \cdot Z$) | $R_m, A_{i,m}, w_{i,m}$ (9 cols) | **32** |
| **C2** | Core Context-Adaptive AMDFE | **Yes ($w_{i,m} \cdot Z$)** | $R_m, A_{i,m}, w_{i,m}$ (9 cols) | **32** |
| **A0** | Groundwater Only | No | None | **15** |
| **A1** | Groundwater + Weather | No | None | **20** |
| **A6** | Core AMDFE + Robustness Extended | Yes ($w_{i,m} \cdot Z$) | Meta + 28 indicator flags | **60** |

---

## 4. Evaluation Protocols & Decision Tree Outcome

### Primary Experimental Findings:
1. **Adaptive Gating Attribution ($C2 - C1$)**:
   - LightGBM Overall: $\Delta\text{RMSE} = -0.854\text{ m}$ ($+11.56\%$ point-estimate improvement, 32 vs 32 features), run standard deviation drops from $5.75\text{ m}$ to $1.93\text{ m}$.
   - 95% Cluster-Bootstrap CI: $[-4.072\text{ m}, +2.694\text{ m}]$ ($p = 0.170$).
   - Warm-Start Subgroup: $\Delta\text{RMSE} = -1.387\text{ m}$, 95% CI: $[-4.483\text{ m}, +2.322\text{ m}]$ ($p = 0.288$).
   - Classification: **Directionally favorable but not statistically conclusive at $\alpha=0.05$ on clean benchmarks**.
2. **Controlled Stress Experiments (Synthetic Monitoring-Data Loss)**:
   - Under 60% groundwater-history loss, unweighted baseline $C0$ degrades by $+193.0\%$ vs $+55.9\%$ for $C2$ and $+44.5\%$ for $A6$.
   - Under long observation gaps ($>365\text{ d}$), AMDFE lowers RMSE from $18.85\text{ m}$ to $11.52\text{ m}$ (mean $\Delta = -1.759\text{ m}$, 95% CI: $[-4.664\text{ m}, +1.314\text{ m}]$, $p = 0.303$).
3. **Cold-Start Spatial Boundary**:
   - When $N_{\text{prior}}=0$, static spatial baseline $C0$ achieves $\text{RMSE} \approx 17.42\text{ m}$ vs $21.51\text{ m}$ for $C2$. Data-driven temporal fusion collapses without historical monitoring.

---

## 5. Legitimate Claims vs Prohibited Overclaims

### Supported & Frozen Claims:
- **Decoupled Two-Stage Quality Synthesis**: Decoupling source reliability from sample observability resolves monitoring latency without penalizing high-quality fresh readings.
- **Attenuated Degradation under Monitoring-Data Loss**: Gating reduces relative error amplification under controlled 20%–60% data loss experiments and extended observation gaps.
- **Tail Error & Variance Control**: P95 tail errors on unseen wells are reduced from $>20.57\text{ m}$ to $<10.88\text{ m}$.

### STRICTLY PROHIBITED Claims:
- *"AMDFE solves spatial cold-start extrapolation"* (Refuted by empirical evidence).
- *"AMDFE integrates satellite imagery and soil hydraulic maps"* (Modality D is unlinked/unavailable).
- *"AMDFE is a physics-informed or mass-conserving hydrodynamic model"* (AMDFE is purely a tabular feature fusion engine).
- *"AMDFE is statistically proven superior across all models on dense clean data"* (Bootstrap CI includes zero on clean benchmarks).
- *"AMDFE was evaluated on live telemetric sensor hardware."*
