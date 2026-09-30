# AMDFE v2.1 Novelty Boundary & Research Positioning

## 1. Context and Prior Art

Multimodal machine learning in hydrogeology typically relies on one of three conventional fusion paradigms:
1. **Early Concatenation**: Features from disparate sensors (piezometers, weather stations, digital elevation models) are concatenated into a flat vector $\mathbf{x} = [\mathbf{x}_A, \mathbf{x}_B, \mathbf{x}_C]$.
   - *Limitation*: Treats all input channels with equal structural confidence, ignoring observation gaps, sensor drift, and latency differences.
2. **Late Decision Fusion**: Separate models are trained per modality, and their predictions are averaged or stacked: $\hat{y} = \sum w_m \hat{y}_m$.
   - *Limitation*: Fails when individual modalities (such as weather or static terrain alone) lack sufficient inductive power to predict absolute water table depths independently.
3. **Deep Attention & Intermediate Fusion**: Complex neural attention mechanisms learn dynamic cross-modal weights.
   - *Limitation*: Requires massive, regularly sampled, dense spatio-temporal grids. In sparse, irregular, tabular telemetry regimes (such as regional groundwater monitoring networks with hundreds of disparate wells sampled 1–4 times per year), deep attention frequently overfits or collapses due to missing temporal tokens.

---

## 2. What the Proposed AMDFE Framework Does Differently

The proposed AMDFE framework introduces a **decoupled two-stage context-adaptive weighting mechanism** designed specifically for irregular tabular hydrogeological monitoring:

1. **Decoupling Intrinsic Source Reliability from Local Context Observability**:
   - **Source Reliability ($R_m$)**: Evaluates historical data quality, completeness, and sensor validity across a multi-dimensional quality tensor on training data.
   - **Context Observability ($A_{i,m}$)**: Evaluates sample-level point-in-time observation latency, temporal gap decay, and prior measurement density without lookahead.
   - *Advantage*: Prevents historical sensor failures from permanently penalizing high-frequency modern telemetry, and prevents a single recent observation from masking chronic sensor unreliability.

2. **Standardized Modality Block Gating ($w_{i,m} \cdot Z(X_{i,m})$)**:
   - Scales zero-mean, unit-variance normalized feature blocks dynamically at inference time based on observation confidence.
   - Modulates tree-based split criteria: when a modality is degraded, its variance is compressed toward zero, naturally guiding tree split algorithms toward more reliable modalities without requiring bespoke tree architectures.

---

## 3. What is Merely Engineering vs Scientifically Demonstrated

| Component | Engineering Implementation | Scientifically Demonstrated Contribution |
|---|---|---|
| **Data Ingestion & Imputation** | Median imputers, gap flags, missingness masks | Standard ML best practice; prevents pipeline crashes under NaN inputs. |
| **Leakage Prevention** | Purging target-derived rasters (`TIFF_Value`) and enforcing $Y-1$ weather boundaries | Essential validation integrity; eliminates artificial metric inflation. |
| **Source Reliability Scoring** | Multi-dimensional geometric mean of completeness/validity | Systematic heuristic for weighting sensor quality; parameter-insensitive. |
| **Adaptive Sample Gating** | $G_{i,m} = R_m^\alpha \cdot A_{i,m}^\beta$ applied per sample | **Demonstrated to improve tail-error stability ($P90, P95$) and reduce degradation under sensor dropout in controlled 32-feature comparisons ($C2$ vs $C1$).** |

---

## 4. What Remains Unproven & Future Boundaries

1. **Cold-Start Extrapolation**: AMDFE does *not* solve the spatial cold-start problem ($N_{\text{prior}} = 0$). When no prior temporal history exists at an unseen well, the model must rely entirely on static coordinates and regional weather.
2. **Satellite Image Fusion**: Interfaces for Sentinel-1/2 SAR/Optical imagery and SoilGrids are formulated but unverified with empirical rasters in the current repository.
3. **Physical Mass Conservation**: AMDFE is a data-driven feature fusion engine and does not enforce explicit hydrodynamic partial differential equations (e.g. Darcy's Law or Richards' Equation).
