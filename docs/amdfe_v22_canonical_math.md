# AMDFE v2.2 Canonical Mathematical Specification
**SINGLE SOURCE OF TRUTH FOR ALL AMDFE IMPLEMENTATIONS & DOCUMENTATION**

---

## 1. Mathematical Symbol Glossary

| Symbol | Definition | Domain | Invariance / Time Dependency |
|---|---|---|---|
| $m \in \{A, B, C\}$ | Modality index ($A$: Groundwater, $B$: Weather, $C$: Spatial) | Discrete Set | Static |
| $i \in \{1, \dots, N\}$ | Observation instance / sample index | Discrete Set | Point-in-time |
| $j \in \{1, \dots, K_m\}$ | Feature index within modality block $m$ | Discrete Set | Static |
| $R_m$ | Source-level intrinsic reliability score | $[0, 1]$ | Static per modality (fit on Train) |
| $M_{i,m}$ | Applicability / integration availability mask | $\{0, 1\}$ | Binary indicator per sample |
| $A_{i,m}$ | Context observability factor | $[0, 1]$ | Sample-dependent |
| $\alpha, \beta$ | Exponent parameters for reliability & observability | $\mathbb{R}^+$ | Frozen global hyperparameters ($\alpha=1.0, \beta=1.0$) |
| $G_{i,m}$ | Effective trust score | $[0, 1]$ | Sample-dependent |
| $w_{i,m}$ | Normalized adaptive modality weight | $[0, 1]$ | Sample-dependent ($\sum_m w_{i,m} = 1$) |
| $X_{i,m,j}$ | Raw feature value for sample $i$, modality $m$, feature $j$ | $\mathbb{R}$ | Input data |
| $\mu_{m,j}^{\text{train}}, \sigma_{m,j}^{\text{train}}$ | Training-set mean and standard deviation for feature $(m, j)$ | $\mathbb{R}, \mathbb{R}^+$ | Frozen from training fold |
| $Z(X_{i,m,j})$ | Zero-mean, unit-variance standardized feature | $\mathbb{R}$ | Standardized intermediate |
| $F_{i,m,j}$ | Fused adaptive predictor | $\mathbb{R}$ | Final predictor matrix column |

---

## 2. Canonical Equations

### 2.1 Stage 1: Intrinsic Source Reliability ($R_m$)
Evaluated exclusively on the training partition $\mathcal{D}_{\text{train}}$ across the set of applicable quality dimensions $\mathcal{Q}_m$:
$$R_m = \left( \prod_{d \in \mathcal{Q}_m} Q_{m,d} \right)^{1 / |\mathcal{Q}_m|}$$
where:
$$Q_{m,d} \in [0, 1] \quad \forall d \in \{\text{Completeness}, \text{Validity}, \text{Non-Duplication}, \text{Outlier Regularity}, \text{Temporal Continuity}, \text{Spatial Coverage}\}$$

---

### 2.2 Stage 2: Sample-Level Context Observability ($A_{i,m}$)
Evaluated point-in-time for sample $i$ using strictly historical metadata prior to timestamp $t_i$:

- **Modality A (Groundwater)**:
  $$A_{i,A} = \begin{cases} 
  0.10 & \text{if } N_{\text{prior}} = 0 \\
  \mathrm{clip}\left(0.20 + 0.50 \exp\left(-\frac{\Delta t_i}{365.25}\right) + 0.30 \min\left(\frac{N_{\text{prior}}}{3}, 1\right), \, 0.05, \, 1.0\right) & \text{if } N_{\text{prior}} \ge 1
  \end{cases}$$

- **Modality B (Weather)**:
  $$A_{i,B} = \mathbb{I}(\text{Weather Features Present for Year } Y_i - 1) \in \{0.0, 1.0\}$$

- **Modality C (Spatial & Topographic Context)**:
  $$A_{i,C} = \mathbb{I}(\text{LatDD, LongDD, Surf\_Elev Non-Null}) \in \{0.0, 1.0\}$$

- **Modality D (Earth Observation & SoilGrids)**:
  $$A_{i,D} = 0.0 \quad (\text{Enforced Zero-Trust for Unlinked Data})$$

---

### 2.3 Stage 3: Effective Trust Synthesis ($G_{i,m}$)
$$G_{i,m} = (R_m)^\alpha \cdot (A_{i,m})^\beta, \quad \text{with } \alpha = 1.0, \, \beta = 1.0 \implies G_{i,m} = R_m \cdot A_{i,m}$$

---

### 2.4 Stage 4: Normalized Dynamic Channel Weights ($w_{i,m}$)
For applicable modalities where $M_{i,m} = 1$:
$$w_{i,m} = \frac{M_{i,m} \cdot G_{i,m}}{\sum_{k} M_{i,k} \cdot G_{i,k}}$$

#### Invariant Constraints:
1. **Zero-Trust for Unavailable Sources**:
   $$M_{i,m} = 0 \implies w_{i,m} = 0$$
2. **Partition of Unity across Available Modalities**:
   $$\sum_{m \in \{m : M_{i,m}=1\}} w_{i,m} = 1.0 \quad \forall i$$
3. **Safe Fallback**: If $\sum_k M_{i,k} G_{i,k} \le 10^{-5}$, weights default to uniform allocation across applicable modalities:
   $$w_{i,m} = \frac{M_{i,m}}{\sum_k M_{i,k}}$$

---

### 2.5 Stage 5: Standardized Block Feature Gating
For each predictor $j$ in modality block $m$:
1. **Frozen Standard Normalization**:
   $$Z(X_{i,m,j}) = \frac{X_{i,m,j} - \mu_{m,j}^{\text{train}}}{\sigma_{m,j}^{\text{train}}}$$
2. **Multiplicative Gating**:
   $$F_{i,m,j} = w_{i,m} \cdot Z(X_{i,m,j})$$

---

## 3. Predictor Matrix Assembly (C0, C1, C2)

| Dataset Configuration | Gated Feature Columns | Metadata Columns (Ungated) | Total Feature Count | Mathematical Distinction |
|---|---|---|---|---|
| **C0 (Unweighted Base)** | $F_{i,m,j} = 1.0 \cdot Z(X_{i,m,j})$ (23 cols) | None | **23** | Pure unweighted concatenation |
| **C1 (Matched Base+Meta)**| $F_{i,m,j} = 1.0 \cdot Z(X_{i,m,j})$ (23 cols) | $R_m$ (3 cols), $A_{i,m}$ (3 cols), $w_{i,m}$ (3 cols) | **32** | Feature-matched control baseline |
| **C2 (Core Adaptive AMDFE)**| $F_{i,m,j} = w_{i,m} \cdot Z(X_{i,m,j})$ (23 cols) | $R_m$ (3 cols), $A_{i,m}$ (3 cols), $w_{i,m}$ (3 cols) | **32** | **Direct marginal test ($C2 - C1$)** |
