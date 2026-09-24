# Stage 2: Relative Hydro-Climatic Response-Potential Index (RRPI) Methodology

## 1. Scientific Status & Boundary Constraints

The Recharge Response-Potential Index (RRPI) is an **UNCALIBRATED, RELATIVE, HYDRO-CLIMATIC DIAGNOSTIC INDEX**.

### Prohibited Interpretations:
RRPI is **NOT**:
- measured recharge,
- recharge flux,
- recharge volume,
- recharge rate,
- groundwater recharge amount,
- physically calibrated recharge potential.

RRPI is a transparent hypothesis-driven hydro-climatic index and is **NOT calibrated against measured recharge ground truth**. Direct recharge ground-truth measurements do not exist in this monitoring dataset.

### Permitted Scientific Language:
The score reflects **"relative hydro-climatic favorability"** or **"relative antecedent hydro-climatic condition"**.
- A high RRPI indicates that antecedent precipitation and atmospheric conditions are relatively more favorable according to the predefined mathematical index formulation.
- A low RRPI indicates that antecedent conditions are relatively less favorable according to that same formulation.
- RRPI does **NOT** prove or predict actual groundwater recharge.

---

## 2. Reference Population Calibration

All normalization parameters are estimated **exclusively from the 2000–2019 temporal training baseline** ($N = 3,247$ observations across 161 wells). No future validation (2020–2022) or test (2023–2024) observations may influence RRPI normalization parameters.

Reference parameter file: `stage2_rrpi_reference_population.csv`.

---

## 3. Mathematical Formulation

### A. Input Features & Directionality
1. **Moisture Delivery Dimension** ($S_{moist}$):
   - `Precip_30D_Sum` ($x_1$): 30-day antecedent rainfall exposure (+).
   - `Precip_180D_Sum` ($x_2$): 180-day seasonal rainfall exposure (+).
   - `Rain_Days_30D` ($x_3$): Number of rain days > 1mm in 30 days (+).
   - `Humidity_Mean_30D` ($x_4$): 30-day mean relative humidity (+).
2. **Atmospheric Demand Dimension** ($S_{demand}$):
   - `Temp_Mean_30D` ($z_1$): 30-day mean temperature (+).
   - `Radiation_30D_Sum` ($z_2$): 30-day solar irradiance (+).
   - `WindSpeed_Mean_30D` ($z_3$): 30-day mean wind speed (+).
   - `Dry_Spell_Days` ($z_4$): Consecutive dry days ending $D-1$ (+).

### B. Normalization Method
For each feature $u$, compute baseline empirical bounds:
$$\mu_{min}(u) = \min_{i \in Train} u_i, \quad \mu_{max}(u) = \max_{i \in Train} u_i$$
$$\tilde{u}_i = \text{clip}\left(\frac{u_i - \mu_{min}(u)}{\mu_{max}(u) - \mu_{min}(u)}, 0, 1\right)$$

### C. Dimension Aggregation
Equal weighting is applied across features within each dimension as a **transparent, uncalibrated design choice** rather than a learned or scientifically validated weighting:
$$S_{moist, i} = \frac{1}{4} \sum_{j=1}^4 \tilde{x}_{j, i} \quad \in [0, 1]$$
$$S_{demand, i} = \frac{1}{4} \sum_{k=1}^4 \tilde{z}_{k, i} \quad \in [0, 1]$$

### D. Final Score Transformation
$$\text{Balance}_i = S_{moist, i} - S_{demand, i} \quad \in [-1, 1]$$
$$\text{RRPI}_i = 50 \times \left(1 + S_{moist, i} - S_{demand, i}\right) \quad \in [0, 100]$$

---

## 4. Cold-Start Evaluation Summary ($N = 170$)

- **Min Score**: 27.62
- **Median Score**: 38.27
- **Mean Score**: 41.51
- **Max Score**: 60.16
- **Standard Deviation**: 8.95
- **Supervised $\Delta h$ Assigned**: Exactly 0 (100% NaN preserved).
