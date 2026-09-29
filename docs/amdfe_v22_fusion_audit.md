# AMDFE v2.2 Fusion Mechanics & Feature Allocation Audit

## 1. Executive Inspection of Adaptive Fusion Mechanics

This document provides an exhaustive, line-by-line structural audit of the AMDFE feature space, block segmentation, standardization order, and gating mechanics.

---

## 2. Feature-to-Modality Allocation Matrix

### Full Feature Inventory (C1, C2, and A6)

| Feature Index | Feature Name | Modality Assignment | Category | Normalized ($Z$-score)? | Gated by $w_{i,m}$ in C2? | Gated in C1? | Gated in A6? | Role in Machine Learning Pipeline |
|---|---|---|---|---|---|---|---|---|
| **1** | `Previous_WatLevel` | Modality A (Groundwater) | Base State | Yes ($\mu=0, \sigma=1$) | **Yes ($w_{i,A} \cdot Z$)** | No ($1.0 \cdot Z$) | Yes ($w_{i,A} \cdot Z$) | Immediate autoregressive lag |
| **2** | `Previous2_WatLevel` | Modality A (Groundwater) | Base State | Yes | **Yes ($w_{i,A} \cdot Z$)** | No | Yes | Second autoregressive lag |
| **3** | `Days_Since_Previous`| Modality A (Groundwater) | Base Temporal | Yes | **Yes ($w_{i,A} \cdot Z$)** | No | Yes | Time elapsed since last reading |
| **4** | `Years_Since_Previous`| Modality A (Groundwater)| Base Temporal | Yes | **Yes ($w_{i,A} \cdot Z$)** | No | Yes | Annualized elapsed time |
| **5** | `Rolling_Mean_3` | Modality A (Groundwater) | Base Statistics| Yes | **Yes ($w_{i,A} \cdot Z$)** | No | Yes | 3-step moving average |
| **6** | `Rolling_Std_3` | Modality A (Groundwater) | Base Statistics| Yes | **Yes ($w_{i,A} \cdot Z$)** | No | Yes | 3-step local volatility |
| **7** | `Previous_Level_Change`| Modality A (Groundwater)| Base Dynamics | Yes | **Yes ($w_{i,A} \cdot Z$)** | No | Yes | Difference between prior two states |
| **8** | `Recent_Trend` | Modality A (Groundwater) | Base Dynamics | Yes | **Yes ($w_{i,A} \cdot Z$)** | No | Yes | Directional slope |
| **9** | `Historical_Mean` | Modality A (Groundwater) | Base Summary | Yes | **Yes ($w_{i,A} \cdot Z$)** | No | Yes | All-time well mean water depth |
| **10** | `Historical_Std` | Modality A (Groundwater) | Base Summary | Yes | **Yes ($w_{i,A} \cdot Z$)** | No | Yes | All-time well depth variance |
| **11** | `Historical_Min` | Modality A (Groundwater) | Base Summary | Yes | **Yes ($w_{i,A} \cdot Z$)** | No | Yes | Historical shallowest level |
| **12** | `Historical_Max` | Modality A (Groundwater) | Base Summary | Yes | **Yes ($w_{i,A} \cdot Z$)** | No | Yes | Historical deepest level |
| **13** | `Previous_Observation_Count`| Modality A (GW)| Base Density | Yes | **Yes ($w_{i,A} \cdot Z$)** | No | Yes | Cumulative monitoring count |
| **14** | `Observation_Density`| Modality A (Groundwater)| Base Density | Yes | **Yes ($w_{i,A} \cdot Z$)** | No | Yes | Observations per monitored year |
| **15** | `Long_Gap_Flag` | Modality A (Groundwater) | Base Indicator| Yes | **Yes ($w_{i,A} \cdot Z$)** | No | Yes | Flag for $\Delta t > 365\text{ d}$ |
| **16** | `Very_Long_Gap_Flag`| Modality A (GW) | Base Indicator| Yes | **Yes ($w_{i,A} \cdot Z$)** | No | Yes | Flag for $\Delta t > 730\text{ d}$ |
| **17** | `Annual_Temperature_Mean`| Modality B (Weather)| Base Climate | Yes | **Yes ($w_{i,B} \cdot Z$)** | No | Yes | Mean annual temperature ($Y-1$) |
| **18** | `Annual_Precipitation_Total`| Modality B (Weather)| Base Climate | Yes | **Yes ($w_{i,B} \cdot Z$)** | No | Yes | Total annual precipitation ($Y-1$) |
| **19** | `Annual_Humidity_Mean`| Modality B (Weather) | Base Climate | Yes | **Yes ($w_{i,B} \cdot Z$)** | No | Yes | Mean relative humidity ($Y-1$) |
| **20** | `Annual_WindSpeed_Mean`| Modality B (Weather) | Base Climate | Yes | **Yes ($w_{i,B} \cdot Z$)** | No | Yes | Mean wind speed ($Y-1$) |
| **21** | `Annual_SolarRadiation_Mean`| Modality B (Weather)| Base Climate | Yes | **Yes ($w_{i,B} \cdot Z$)** | No | Yes | Mean solar radiation ($Y-1$) |
| **22** | `LatDD` | Modality C (Spatial) | Base Geospatial| Yes | **Yes ($w_{i,C} \cdot Z$)** | No | Yes | Latitude coordinate |
| **23** | `LongDD` | Modality C (Spatial) | Base Geospatial| Yes | **Yes ($w_{i,C} \cdot Z$)** | No | Yes | Longitude coordinate |
| **24** | `Surf_Elev` | Modality C (Spatial) | Base Geospatial| Yes | **Yes ($w_{i,C} \cdot Z$)** | No | Yes | Surface elevation |
| **25** | `R_modality_a` | Metadata | Reliability | No (Raw [0,1]) | **No (Ungated Meta)** | No | No | Intrinsic reliability of GW source |
| **26** | `R_modality_b` | Metadata | Reliability | No (Raw [0,1]) | **No (Ungated Meta)** | No | No | Intrinsic reliability of Weather source |
| **27** | `R_modality_c` | Metadata | Reliability | No (Raw [0,1]) | **No (Ungated Meta)** | No | No | Intrinsic reliability of Spatial source |
| **28** | `A_modality_a` | Metadata | Observability | No (Raw [0,1]) | **No (Ungated Meta)** | No | No | Sample GW observability |
| **29** | `A_modality_b` | Metadata | Observability | No (Raw [0,1]) | **No (Ungated Meta)** | No | No | Sample Weather observability |
| **30** | `A_modality_c` | Metadata | Observability | No (Raw [0,1]) | **No (Ungated Meta)** | No | No | Sample Spatial observability |
| **31** | `w_modality_a` | Metadata | Dynamic Weight | No (Raw [0,1]) | **No (Ungated Meta)** | No | No | Normalized adaptive weight for GW |
| **32** | `w_modality_b` | Metadata | Dynamic Weight | No (Raw [0,1]) | **No (Ungated Meta)** | No | No | Normalized adaptive weight for Weather |
| **33** | `w_modality_c` | Metadata | Dynamic Weight | No (Raw [0,1]) | **No (Ungated Meta)** | No | No | Normalized adaptive weight for Spatial |
| **34–60** | `Missing_*`, `Outlier_*` (28 cols) | Robustness Metadata | Indicator Flags| No (Binary/Z) | **No (A6 only)** | No | No (Raw) | Explicit robustness flags |

---

## 3. Detailed Audit of the 10 Core Fusion Questions

### 1. What is a modality block?
A modality block is a semantically grouped partition of raw predictor columns derived from a single distinct sensor or data generation mechanism:
- **Modality A Block**: 16 features representing piezometer time-series dynamics.
- **Modality B Block**: 5 features representing annual meteorological reanalysis.
- **Modality C Block**: 3 features representing geographic location and terrain altitude.

### 2. Which columns belong to each modality?
- Modality A: Features 1 through 16.
- Modality B: Features 17 through 21.
- Modality C: Features 22 through 24.
- Metadata: Columns 25 through 33 (supplied unscaled to both C1 and C2).

### 3. Are weights applied to modality blocks before downstream ML?
**Yes**. The adaptive weights $w_{i,m}$ are computed in the feature engineering pipeline and multiplied into the standardized feature columns *before* the matrix $(X, y)$ is passed to LightGBM, XGBoost, or Random Forest.

### 4. Are weights applied after standardization?
**Yes, strictly**. Standardization is performed first using frozen training set statistics:
$$Z(X_{i,m,j}) = \frac{X_{i,m,j} - \mu_{m,j}^{\text{train}}}{\sigma_{m,j}^{\text{train}}}$$
Then adaptive gating is applied:
$$F_{i,m,j} = w_{i,m} \cdot Z(X_{i,m,j})$$
This ensures that all base features have equal variance ($\sigma^2=1$) prior to weighting, so that multiplying by $w_{i,m}$ has an identical proportional compression across all channels.

### 5. Are metadata variables themselves being gated?
**No**. The reliability scores ($R_m$), observability factors ($A_{i,m}$), and adaptive weights ($w_{i,m}$) are appended as unscaled metadata columns. This preserves the direct raw interpretation of the confidence state for the tree split criteria.

### 6. Are $R_m$ and $A_{i,m}$ used as predictors AND as gating signals?
**Yes**. They serve a dual role:
1. **Control Signal**: They mathematically compute $w_{i,m}$ to scale the base feature blocks.
2. **Direct Metadata Feature**: They are supplied directly as features (in both C1 and C2) allowing tree splits on observation freshness.
Because both C1 and C2 contain these exact metadata columns, any empirical difference between C2 and C1 is strictly attributable to the **multiplicative gating of the base blocks**.

### 7. Does the decision-tree model actually respond to the gating?
**Yes**. In orthogonal decision trees, multiplying a feature by a constant global scalar has zero effect on split point ranking. However, multiplying by a **sample-dependent weight $w_{i,m}$** alters the relative ordering of samples along that feature axis, compressing degraded observations toward zero and forcing split algorithms to partition on more reliable channels.

### 8. Is the adaptive weight merely another feature transformation?
Mathematically, sample-level gating is a non-linear, sample-dependent coordinate transformation. However, unlike unconstrained transformations, its functional form is physically constrained by data source quality and observation recency.

### 9. Does a modality with $w_{i,m} \approx 0$ actually lose influence?
**Yes**. When $w_{i,m} \to 0$, $F_{i,m,j} \to 0$ for all features $j$ in modality $m$. The feature block variance collapses to 0, providing zero variance for the tree to split on, effectively masking the channel for that specific sample.

### 10. Are all modality blocks normalized comparably?
**Yes**. Every base feature is independently standardized to $\mathcal{N}(0, 1)$ using training-fold means and standard deviations before block scaling.
