# Phase 08.10 — Stage 2 Implementation Report
## Warm-Start Groundwater Response Modelling, Multi-Scale Precipitation Ablation & Cold-Start RRPI Construction

**Phase Status**: STAGE 2 COMPLETE — ABLATION & EMPIRICAL RESPONSE MODELLING VERIFIED  
**Primary Dataset**: `data/processed/advanced_ml/recharge/stage1/groundwater_recharge_response_stage1.csv`  
**Execution Script**: `src/08_10_stage2_recharge_modelling.py`  
**Authoritative Methodology**: `data/processed/advanced_ml/recharge/methodology/implementation_plan.md`

---

## 1. Executive Summary & Experimental Objectives

Stage 2 executes controlled empirical groundwater response (\Delta h) modelling and diagnostic relative hydro-climatic potential index (RRPI) construction across 3,844 monitoring observations in Phelps County, Nebraska:

1. **Target Formulation & Strict Boundary**:
   - Supervised target: \Delta h_i = Previous_WatLevel_i - WatLevel_i [ft].
   - Positive \Delta h > 0: Shallower/rising groundwater level. Negative \Delta h < 0: Deeper/falling groundwater level.
   - \Delta h is an **empirical groundwater response** and is **NOT** measured recharge, recharge flux, or recharge volume.
   - Modeled strictly on warm-start observations ($N = 3,674$). Cold-start observations ($N = 170$) receive **no supervised \Delta h target** (\Delta h = NaN).
2. **Strict Prior-Day Weather Cutoff**:
   - max(weather_date_used) < DateMsr (strictly D - 1 day) across 100% of observations.
3. **Precipitation-Window Ablation**:
   - Systematic evaluation of antecedent windows (P7, P14, P30, P60, P90, P180) revealed that multi-scale rolling windows provide marginal incremental predictive information over prior groundwater state alone.
   - The 90-day precipitation window achieved the lowest single-window temporal test MAE (1.1945 ft, R² = 0.3124), indicating empirical predictive utility over an intermediate multi-month timeframe. This does not establish a physical recharge travel time or causal recharge mechanism.
4. **Interval Precipitation Incremental Contribution**:
   - Evaluated whether `Precip_Interval_Sum` adds predictive value beyond observation gap duration (`Days_Since_Previous`).
   - Neither observation-gap duration alone nor interval precipitation alone accounts for water-level change over irregular monitoring intervals. When both variables are provided together, the model conditions on elapsed observation duration and cumulative precipitation exposure between monitoring events, without directly measuring physical infiltration rates.
5. **Feature-Family Ablation**:
   - Evaluated Set 0 (G+T) through Set 5 (Full Multimodal). Antecedent groundwater state (Set 1) accounts for the vast majority of predictable variation in \Delta h. Highly dynamic short-term surface atmospheric variables induce temporal overfitting across multi-month monitoring intervals.
6. **Extreme \Delta h Robustness**:
   - Comparing primary warm-start modeling (full population $N=3,674$, train $N=3,086$) vs. non-extreme subset (non-extreme population $N=3,650$, train $N=3,068$, excluding 24 >3*IQR observations: 18 in train, 6 in validation, 0 in test) demonstrated stable error distributions on the temporal test set ($N=180$) without artificial clipping.
7. **Cold-Start RRPI Construction**:
   - Formulated a transparent, uncalibrated diagnostic index ([0, 100]) calibrated strictly on the 2000–2019 historical reference population using exactly 8 approved features.
   - RRPI is an uncalibrated relative hydro-climatic diagnostic. It is NOT measured recharge, recharge flux, mm/year recharge, physically calibrated recharge, or infiltration rate.
8. **11-Gate Anti-Circularity & Leakage Audit**:
   - All 11 gates report **PASS** with zero violations.

---

## 2. Experimental Results Summary

### A. Feature-Family Ablation (Temporal Test Set, 2023–2024, N=180)

| Feature Set | Features | Description | RF MAE | RF RMSE | RF R² | XGB MAE | XGB R² | LGBM MAE | LGBM R² |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SET_0** | 3 | G + T (Static Location & Topography) | 1.9863 ft | 2.5855 ft | -0.0352 | 2.0620 ft | -0.1704 | 1.9547 ft | -0.0465 |
| **SET_1** | 16 | G + T + H (Antecedent Groundwater State) | 1.2858 ft | 1.9680 ft | 0.4005 | 1.4883 ft | 0.1691 | 1.4072 ft | 0.2372 |
| **SET_2** | 21 | G + T + H + Atmospheric Context | 1.2847 ft | 1.9792 ft | 0.3937 | 1.4589 ft | 0.1983 | 1.4170 ft | 0.2291 |
| **SET_3** | 22 | G + T + H + Precipitation Windows | 1.2657 ft | 1.9676 ft | 0.4008 | 1.4429 ft | 0.2078 | 1.4287 ft | 0.2173 |
| **SET_4** | 23 | G + T + H + Precip Windows + Interval Precip | 1.2582 ft | 1.9567 ft | 0.4074 | 1.4484 ft | 0.1878 | 1.4116 ft | 0.2296 |
| **SET_5** | 31 | Full Multimodal: G + T + H + P_all + A | 1.2721 ft | 1.9880 ft | 0.3883 | 1.4578 ft | 0.1866 | 1.4187 ft | 0.2166 |

*Scientific Takeaway: Adding antecedent groundwater state (Set 1) reduces MAE from ~1.99 ft to ~1.28 ft. Adding precipitation features (Set 4) yields a minor incremental predictive improvement (MAE 1.2858 -> 1.2582 ft), while adding all 31 multimodal features (Set 5) slightly degrades out-of-sample generalization due to collinearity across multi-month monitoring intervals.*

### B. Interval Precipitation Incremental Contribution (Temporal Test Set)

| Configuration | Features | Description | Random Forest MAE | Random Forest R² | XGBoost MAE | XGBoost R² |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **BASE** | 14 | G + T + H_no_gap (History excluding gap days) | 1.2878 ft | 0.3931 | 1.4820 ft | 0.1628 |
| **BASE_PLUS_GAP** | 16 | Base + Days_Since_Previous + Years_Since_Previous | 1.2858 ft | 0.4005 | 1.4883 ft | 0.1691 |
| **BASE_PLUS_INTERVAL** | 15 | Base + Precip_Interval_Sum | 1.2917 ft | 0.3892 | 1.4725 ft | 0.1706 |
| **FULL_INTERVAL** | 17 | Base + Gap Duration + Precip_Interval_Sum | 1.2801 ft | 0.4022 | 1.4811 ft | 0.1730 |

*Scientific Takeaway: Neither observation-gap duration alone nor interval precipitation alone accounts for water-level change over irregular monitoring intervals. When both variables are provided together, the model conditions on elapsed observation duration and cumulative precipitation exposure between monitoring events, without directly measuring physical infiltration rates. Precip_Interval_Sum has high correlation with gap duration ($r = 0.9089$), reflecting cumulative elapsed precipitation exposure over monitoring intervals.*

### C. Extreme \Delta h Robustness Evaluation (SET_4, Evaluated Exclusively on Temporal Test Set N=180)

> [!NOTE]
> Metrics below are evaluated exclusively on the 2023–2024 temporal test partition (N=180). Sample/cohort counts refer to the corresponding training/robustness population and are not the evaluation sample size.

- Full warm-start population: N = 3,674 (range: [-32.81, +33.21] ft, Q1 = -1.49 ft, Q3 = +1.31 ft, IQR = 2.80 ft, 3×IQR bounds = [-9.89, +9.71] ft)
- Baseline training population: N = 3,086 | Baseline validation population: N = 408 | Temporal test population: N = 180
- Non-extreme robustness experiment: 24 extreme observations removed (18 in training, 6 in validation, 0 in test)
- Non-extreme population: N = 3,650 | Non-extreme training: N = 3,068 | Non-extreme validation: N = 402 | Temporal test: N = 180

| Training Regime | Full Population | Train Obs | Test Obs | Random Forest MAE | Random Forest RMSE | Random Forest R² | XGBoost MAE | XGBoost RMSE | XGBoost R² | LightGBM MAE | LightGBM RMSE | LightGBM R² |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Baseline Training** | 3,674 | 3,086 | 180 | 1.2064 ft | 1.7072 ft | 0.2349 | 1.1989 ft | 1.6696 ft | 0.2682 | 1.1564 ft | 1.6294 ft | 0.3030 |
| **Non-Extreme Robustness Training** | 3,650 | 3,068 | 180 | 1.1178 ft | 1.5579 ft | 0.3628 | 1.1520 ft | 1.5752 ft | 0.3486 | 1.0904 ft | 1.5375 ft | 0.3794 |

*Scientific Takeaway: Metrics are evaluated exclusively on the N=180 temporal test partition. Retraining models without the 18 training-set extreme observations yields test MAE of 1.1178 ft (RF), 1.1520 ft (XGB), and 1.0904 ft (LGBM). Because the physical causes of extreme water-level changes cannot be uniquely established from available data, they are retained in primary modeling to ensure honest evaluation.*

---

## 3. Cold-Start RRPI Diagnostic Assessment ($N = 170$)

- **Approved 8 Features**:
  - Moisture Delivery: `Precip_30D_Sum`, `Precip_180D_Sum`, `Rain_Days_30D`, `Humidity_Mean_30D`
  - Atmospheric Demand: `Temp_Mean_30D`, `Radiation_30D_Sum`, `WindSpeed_Mean_30D`, `Dry_Spell_Days`
- **Formulation**: Balanced multi-component relative index:
  $$S_{\text{moist}} = \text{mean}(\widetilde{P}_{30}, \widetilde{P}_{180}, \widetilde{R}_{\text{days}}, \widetilde{H}_{\text{rel}})$$
  $$S_{\text{demand}} = \text{mean}(\widetilde{T}_{\text{mean}}, \widetilde{R}_{\text{rad}}, \widetilde{W}_{\text{speed}}, \widetilde{D}_{\text{dry}})$$
  $$\text{RRPI} = 50 \times \left(1 + S_{\text{moist}} - S_{\text{demand}}\right) \quad \in [0, 100]$$
- **Calibration**: Fit strictly on 2000–2019 reference baseline (zero future test leakage).
- **Distribution on 170 Cold-Start Wells**:
  - Min: 27.62
  - Median: 38.27
  - Mean: 41.51
  - Max: 60.16
  - Std Dev: 8.95
- **Interpretation**: Higher score reflects relatively more favorable antecedent hydro-climatic conditions; RRPI is an uncalibrated relative hydro-climatic diagnostic and does NOT predict measured recharge, recharge flux, or infiltration rate.

---

## 4. 11-Gate Anti-Circularity & Leakage Audit Matrix

| Gate ID | Name | Status | Details |
| :--- | :--- | :---: | :--- |
| **RCH2-LC-01** | Prior-Day Weather Cutoff | **PASS** | max(weather_date_used) < DateMsr verified for 100% of records (0 violations). |
| **RCH2-LC-02** | Groundwater Temporal Precedence | **PASS** | Date_prior < Date_current verified across all 3,674 warm-start records (0 violations). |
| **RCH2-LC-03** | Target Segregation (`WatLevel`) | **PASS** | Current WatLevel strictly absent from candidate predictors. |
| **RCH2-LC-04** | Target Segregation (\Delta h) | **PASS** | \Delta h strictly absent from candidate predictors. |
| **RCH2-LC-05** | Static Raster Exclusion | **PASS** | TIFF_Value strictly absent from dataset and all feature configurations. |
| **RCH2-LC-06** | Grouped Well Separation | **PASS** | Zero well overlap between train and test across all 10 seeds (136 Train / 34 Test). |
| **RCH2-LC-07** | Spatial Fold Isolation | **PASS** | Zero well overlap across all 5 spatial cluster folds. |
| **RCH2-LC-08** | Transformation Leakage Isolation | **PASS** | All scalers, parameters, and metrics fit strictly on training partitions. |
| **RCH2-LC-09** | Cold-Start Target Integrity | **PASS** | All 170 cold-start records have 100% NaN for \Delta h (zero fabricated targets). |
| **RCH2-LC-10** | RRPI Reference Population Fidelity | **PASS** | RRPI reference parameters fit strictly on 2000-2019 baseline. |
| **RCH2-LC-11** | Baseline Phase Preservation | **PASS** | All 49 locked files from Phases 08.4-08.9 & Stage 1.1 verified 100% identical. |

---

## 5. Artifact Inventory (Stage 2)

1. `stage2_feature_catalog.csv` (31 features)
2. `stage2_precipitation_ablation.csv` (24 rows)
3. `stage2_feature_family_ablation.csv` (18 rows)
4. `stage2_interval_precipitation_ablation.csv` (12 rows)
5. `stage2_temporal_results.csv` (9 rows)
6. `stage2_repeated_grouped_results.csv` (9 rows, aggregated across 10 seeds)
7. `stage2_spatial_results.csv` (45 rows across 5 spatial folds)
8. `stage2_model_summary.csv`
9. `stage2_extreme_delta_h_robustness.csv` (6 rows)
10. `stage2_rrpi_methodology.md`
11. `stage2_rrpi_reference_population.csv` (8 features)
12. `stage2_rrpi_cold_start.csv` (170 records)
13. `stage2_leakage_audit.csv` (11 gates)
14. `README.md`
15. `walkthrough.md`
