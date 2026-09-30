# AMDFE v2.2 Context Observability Mathematical Specification

## 1. Mathematical Formulation

For any given prediction instance $i$ and modality $m \in \{A, B, C\}$, the context observability factor $A_{i,m} \in [0, 1]$ quantifies the point-in-time freshness, availability, and completeness of that modality *specifically for sample $i$*.

### Modality A: Groundwater History & State Observability
$$A_{i,A} = \begin{cases} 
A_{\text{cold}} & \text{if } N_{\text{prior}} = 0 \\
\mathrm{clip}\left(w_{\text{base}} + w_{\text{gap}} \cdot f_{\text{gap}}(\Delta t_i) + w_{\text{count}} \cdot f_{\text{count}}(N_{\text{prior}}), \, 0.05, \, 1.0\right) & \text{if } N_{\text{prior}} \ge 1
\end{cases}$$

where:
- **Gap Decay Function**:
  $$f_{\text{gap}}(\Delta t_i) = \exp\left(-\frac{\Delta t_i}{\tau}\right)$$
- **Observation Density Saturation Function**:
  $$f_{\text{count}}(N_{\text{prior}}) = \min\left(\frac{N_{\text{prior}}}{S}, \, 1.0\right)$$

---

## 2. Parameter Definitions, Units, and Frozen Defaults

| Parameter | Symbol | Frozen Value | Physical Units | Functional Role in Formulation |
|---|---|---|---|---|
| **Cold-Start Baseline** | $A_{\text{cold}}$ | 0.10 | Dimensionless | Base confidence assigned when a well has zero historical observations ($N_{\text{prior}}=0$). Prevents numerical division by zero while heavily downweighting null autoregressive lag channels. |
| **Warm-Start Base Floor** | $w_{\text{base}}$ | 0.20 | Dimensionless | Baseline confidence guaranteed to any monitored well with at least one historical measurement. |
| **Temporal Gap Weight** | $w_{\text{gap}}$ | 0.50 | Dimensionless | Relative weight assigned to observation recentness. |
| **Observation Count Weight** | $w_{\text{count}}$ | 0.30 | Dimensionless | Relative weight assigned to the statistical depth/stability of the well's historical time series. |
| **Temporal Decay Scale** | $\tau$ | 365.25 | Days | Characteristic exponential decay scale (equivalent to an annualized hydrogeological memory scale). |
| **Saturation Threshold** | $S$ | 3.0 | Observations | Number of prior measurements required for local mean/variance statistics (`Rolling_Mean_3`, `Rolling_Std_3`) to reach full statistical validity. |
| **Elapsed Monitoring Interval**| $\Delta t_i$ | Sample-dependent | Days | Time elapsed since the most recent recorded measurement ($\Delta t_i = t_i - t_{i-1}$). |
| **Prior Observation Count** | $N_{\text{prior}}$| Sample-dependent | Count | Cumulative number of valid observations recorded at this specific well prior to timestamp $t_i$. |

---

## 3. Modality B & Modality C Observabilities

### Modality B: Environmental & Weather Context
$$A_{i,B} = \mathbb{I}(\text{Weather Available for Year } Y-1) \in \{0.0, 1.0\}$$
- Evaluates to $1.0$ when the completed meteorological reanalysis record for Year $Y-1$ is present and non-null.
- Evaluates to $0.0$ if weather telemetry is unavailable or missing.

### Modality C: Spatial & Topographic Context
$$A_{i,C} = \mathbb{I}(\text{Coordinates } (\text{Lat}, \text{Long}, \text{Elev}) \text{ Valid}) \in \{0.0, 1.0\}$$
- Evaluates to $1.0$ for all physical wells registered with valid geographic coordinates.

---

## 4. Behavior Across Boundary Conditions

### Case 1: Cold-Start Well ($N_{\text{prior}} = 0$)
- When an unseen well has no prior measurements, $N_{\text{prior}}=0$.
- $A_{i,A} = 0.10$.
- With $R_A = 0.9995, R_B = 0.9997, R_C = 1.0000$ and $A_{i,B} = 1.0, A_{i,C} = 1.0$:
  $$G_{i,A} = 0.10, \quad G_{i,B} \approx 1.00, \quad G_{i,C} = 1.00 \implies \sum G \approx 2.10$$
  $$w_{i,A} \approx 0.048, \quad w_{i,B} \approx 0.476, \quad w_{i,C} \approx 0.476$$
- **Effect**: Modality A features (imputed lags) are compressed by $95.2\%$, shifting predictive focus entirely to weather and spatial elevation.

### Case 2: Fresh Telemetry under Dense Monitoring ($\Delta t = 30\text{ d}, N_{\text{prior}} \ge 3$)
- $f_{\text{gap}}(30) = \exp(-30/365.25) \approx 0.921$.
- $f_{\text{count}}(3) = 1.000$.
- $A_{i,A} = 0.20 + 0.50(0.921) + 0.30(1.0) = 0.9605$.
- Normalized weights:
  $$w_{i,A} \approx 0.324, \quad w_{i,B} \approx 0.338, \quad w_{i,C} \approx 0.338$$
- **Effect**: All three modalities contribute with balanced, near-equal confidence.

### Case 3: Extended Monitoring Blackout ($\Delta t = 1,095\text{ d}$ / 3 Years, $N_{\text{prior}} \ge 3$)
- $f_{\text{gap}}(1095) = \exp(-1095/365.25) = \exp(-3) \approx 0.0498$.
- $f_{\text{count}}(3) = 1.000$.
- $A_{i,A} = 0.20 + 0.50(0.0498) + 0.30(1.0) = 0.5249$.
- Normalized weights:
  $$w_{i,A} \approx 0.208, \quad w_{i,B} \approx 0.396, \quad w_{i,C} \approx 0.396$$
- **Effect**: Modality A influence is attenuated by $\sim 36\%$, preventing stale multi-year lags from dominating current predictions.

---

## 5. Scientific Parameter Selection Protocol

1. **Strict Train-Only Fitting**: The decay constant $\tau = 365.25\text{ d}$ corresponds to the annual hydrological cycle of seasonal precipitation recharge.
2. **Empirical Sensitivity Verification**: As demonstrated in Table 12 of the research report, varying $\tau \in [180, 730\text{ d}]$, $S \in [2, 5\text{ obs}]$, and weight mixtures across 36 parameter configurations on training/validation splits yields stable mean RMSE within $[6.13, 6.73\text{ m}]$.
3. **Parameter Freezing**: All observability parameters were frozen *prior* to final independent test evaluations.
