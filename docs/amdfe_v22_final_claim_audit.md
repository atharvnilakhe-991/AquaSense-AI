# AMDFE v2.2 Final Scientific Claim Audit & Boundary Verification

---

## 1. Executive Protocol

This audit establishes the definitive scientific boundary for AMDFE v2.2. Every methodological, performance, and robustness claim is evaluated under strict inferential rules:
- **Rule 1**: A 95% cluster-bootstrap confidence interval that contains zero **MUST NOT** be described as "statistically significant," "statistically proven," or "statistically established" at $\alpha=0.05$. It must be designated as **"directionally favorable but not statistically conclusive."**
- **Rule 2**: Terminology implying actual in-situ telemetry, transducers, or live sensor hardware outages must be replaced with accurate monitoring terms: **observation loss, monitoring-data loss, groundwater-history loss, observation gaps, and sparse/irregular monitoring**.
- **Rule 3**: Controlled stress simulations evaluated under single random masks must be explicitly identified as **controlled stress experiments**, not statistically established empirical properties.
- **Rule 4**: Physical limitations (cold-start underperformance and clean temporal parity) must be presented with full transparency.

---

## 2. Exhaustive Claim Audit Table

| Claim ID | Core Claim Statement | Experimental Evidence & Metrics | Statistical Support Status | Exact Paper-Safe Scientific Wording | May Appear in Paper? |
|---|---|---|---|---|---|
| **CLM-01** | *"Adaptive gating ($C2$) outperforms ungated metadata ($C1$) across unseen wells"* | Across 5 seeds on unseen wells (LightGBM): $C1\text{ RMSE} = 7.389\text{ m} \to C2\text{ RMSE} = 6.535\text{ m}$ ($\Delta\text{RMSE} = -0.854\text{ m}$). Run standard deviation drops from $5.748\text{ m}$ to $1.932\text{ m}$. Well-clustered 95% CI: $[-4.072\text{ m}, +2.694\text{ m}]$, $p=0.170$. | **DIRECTIONALLY FAVORABLE (Not statistically conclusive at $\alpha=0.05$)** | *"In controlled 32-feature ablations, sample-adaptive modality block gating yields directionally lower mean RMSE ($\Delta = -0.854\text{ m}$) and reduced run-to-run dispersion across random seeds, although the 95% well-clustered confidence interval contains zero."* | **YES (With safe wording only)** |
| **CLM-02** | *"AMDFE improves warm-start unseen-well prediction ($N_{\text{prior}} \ge 1$)"* | On warm-start unseen wells (LightGBM): $C1\text{ RMSE} = 6.250\text{ m} \to C2\text{ RMSE} = 4.863\text{ m}$ ($\Delta\text{RMSE} = -1.387\text{ m}$). 95% CI: $[-4.483\text{ m}, +2.322\text{ m}]$, $p=0.288$. | **DIRECTIONALLY FAVORABLE (Not statistically conclusive)** | *"On unseen wells with historical monitoring records, adaptive gating produces a point-estimate error reduction of $1.39\text{ m}$ over ungated controls, though between-well variance spans zero at the 95% confidence level."* | **YES (With safe wording only)** |
| **CLM-03** | *"AMDFE solves or improves spatial cold-start prediction ($N_{\text{prior}} = 0$)"* | On zero-history cold-start wells ($N_{\text{prior}}=0$): Static spatial $C0\text{ RMSE} = 17.417\text{ m}$ vs Core AMDFE $C2\text{ RMSE} = 21.506\text{ m}$. | **REFUTED / UNDERPERFORMS** | *"AMDFE does not solve spatial cold-start extrapolation. When zero historical observations exist, autoregressive lag features collapse, and static spatial coordinates provide a stronger predictive baseline."* | **YES (As an explicit limitation)** |
| **CLM-04** | *"AMDFE provides resilience under synthetic monitoring-data loss"* | Under controlled 60% groundwater observation loss, unweighted $C0$ degrades by $+193.0\%$ vs $+55.9\%$ for $C2$ and $+44.5\%$ for $A6$. | **SUPPORTED (Under controlled stress simulation)** | *"In controlled stress simulations of synthetic monitoring-data loss, sample-adaptive gating attenuates relative error degradation compared to unweighted concatenation baselines."* | **YES (Explicitly labeled as controlled stress experiment)** |
| **CLM-05** | *"AMDFE improves accuracy during long monitoring gaps ($>365\text{ d}$)"* | Long gap cohort ($>365\text{ d}$): $C0\text{ RMSE} = 5.492\text{ m} \to C2\text{ RMSE} = 3.733\text{ m}$ ($\Delta\text{RMSE} = -1.759\text{ m}$). 95% CI: $[-4.664\text{ m}, +1.314\text{ m}]$, $p=0.303$. | **DIRECTIONALLY FAVORABLE (Under sparse monitoring)** | *"During multi-year observation gaps, adaptive observability weighting attenuates stale lag reliance, yielding directionally lower error on sparse intervals."* | **YES (With safe wording only)** |
| **CLM-06** | *"AMDFE outperforms baselines on clean chronological temporal holdouts"* | Temporal holdout (2021–2024 test period): LightGBM $C1\text{ RMSE} = 3.268\text{ m} \to C2\text{ RMSE} = 3.991\text{ m}$. Random Forest $C1\text{ RMSE} = 3.035\text{ m} \to C2\text{ RMSE} = 3.662\text{ m}$. | **PARITY / SLIGHT UNDERPERFORMANCE ON CLEAN DENSE TEMPORAL DATA** | *"On dense, clean historical records with regular monitoring, unweighted and metadata-augmented models perform on par with or marginally better than adaptive gating."* | **YES (As an explicit parity finding)** |
| **CLM-07** | *"The framework fuses Sentinel-1, Sentinel-2, and SoilGrids data"* | Modality D features are unlinked in the repository schema ($M_D = 0, w_D = 0.0$). | **UNTESTED / NOT IMPLEMENTED** | *"Validation is conducted on in-situ historical observations, completed meteorological reanalysis, and static spatial coordinates; satellite imagery and soil hydraulic interfaces remain conceptual."* | **PROHIBITED TO CLAIM AS IMPLEMENTED** |
| **CLM-08** | *"AMDFE is a physics-informed or mass-conserving groundwater model"* | AMDFE is purely a tabular preprocessing transformation and does not solve partial differential flow equations. | **UNSUPPORTED / OUT OF SCOPE** | *"AMDFE is a data-driven feature engineering and fusion framework, not a numerical hydrodynamic solver."* | **PROHIBITED TO CLAIM** |

---

## 3. Methodological Rationale: Geometric Mean in Reliability Scoring

The intrinsic source reliability score is defined as the geometric mean over applicable quality dimensions:
$$R_m = \left( \prod_{d \in \mathcal{Q}_m} Q_{m,d} \right)^{1 / |\mathcal{Q}_m|}$$

### Scientific Rationale:
1. **Non-Compensatory Property**: Unlike an arithmetic mean, the geometric mean severely penalizes a data source if any single fundamental quality dimension collapses near zero (e.g. if $Q_{\text{validity}} \to 0$, $R_m \to 0$ regardless of high completeness).
2. **Scale Invariance & Multiplicative Consistency**: Ensures that quality dimensions on $[0, 1]$ scale proportionally into the downstream trust exponent $G_{i,m} = R_m \cdot A_{i,m}$.
3. **No Claim of Unique Optimality**: It is an intuitive, standard multi-criteria aggregation index rather than an intrinsically unique physical constant.

---

## 4. Final Conservative AMDFE Contribution Statement

> **"A reliability–observability decoupled feature-fusion framework in which source-level reliability and sample-level observability jointly determine sample-adaptive modality block gating, evaluated under clean, sparse, long-gap, and monitoring-data loss conditions."**
