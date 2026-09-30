# AMDFE Implementation & Repository Audit
**AquaSense-AI Research Pipeline**
**Branch**: `ml-training`
**Date**: September 2026

---

## 1. Executive Summary & Objective

The **Adaptive Multimodal Data Fusion Engine (AMDFE)** is designed as a rigorous, leakage-safe preprocessing and multimodal fusion layer situated between heterogeneous hydrogeological/environmental data sources and downstream machine learning regressors (e.g., LightGBM, XGBoost, Random Forest).

AMDFE addresses the core research question:
> *"Can reliability- and observation-aware adaptive fusion improve groundwater prediction when heterogeneous environmental inputs exhibit disparate completeness, temporal availability, monitoring density, and spatial coverage?"*

This document audits the existing AquaSense-AI codebase, identifies reusable infrastructure, isolates critical data leakage failure modes (especially target-derived rasters and calendar-year weather lookahead), and defines the exact architectural integration points for AMDFE.

---

## 2. Existing Reusable Components

The existing repository contains well-structured modular components that are reused directly without duplication:

| Component | Path | Reusable Elements |
| :--- | :--- | :--- |
| **Model Factory** | `ml/models/factory.py` | Configured regressors: `get_random_forest_regressor()`, `get_xgboost_regressor()`, `get_lightgbm_regressor()`, `get_benchmark_models()`. Matches benchmark hyperparameters (e.g. LightGBM 500 trees, lr=0.05, leaves=31). |
| **Training Pipeline** | `ml/training/trainer.py` | `train_model()`, `train_all_benchmark_models()`. |
| **Evaluation Metrics** | `ml/evaluation/metrics.py` | `calculate_regression_metrics()` ($R^2$, MAE, RMSE, Pearson $r$), `evaluate_model_predictions()`, comparison tables. |
| **Data Split Logic** | `ml/data/loader.py` | `split_by_unseen_wells()` (strict 80/20 unseen well grouping via `CSD_ID` preventing spatial leakage), `split_by_temporal_cutoff()`. |
| **Feature Extraction** | `src/08_4_irregular_observation_features.py` | Prior-only groundwater state features (`Previous_WatLevel`, `Previous2_WatLevel`, `Days_Since_Previous`, `Years_Since_Previous`, `Rolling_Mean_3`, `Rolling_Std_3`, `Historical_Mean`, `Historical_Std`, `Historical_Min`, `Historical_Max`, `Previous_Observation_Count`, `Observation_Density`, `Long_Gap_Flag`, `Very_Long_Gap_Flag`). |

---

## 3. Existing Repository Limitations & Critical Risks

1. **Target-Derived Raster Leakage (`TIFF_Value`)**:
   - `TIFF_Value` present in `groundwater_integrated_dataset.csv` and `groundwater_ml_ready.csv` was generated from ordinary Kriging interpolation of groundwater observations across the full dataset.
   - **Mandatory Policy**: `TIFF_Value` must NEVER enter the feature matrix. Automated validation checks must enforce hard assertion errors.

2. **Temporal Lookahead in Annual Weather Aggregates**:
   - Existing datasets attach same-year calendar annual weather (e.g., `Annual_Precipitation_Total` for year $Y$) to observation dates (e.g., $Y\text{-04-15}$).
   - This violates causal precedence because precipitation occurring from May to December of year $Y$ was unobserved at the time of the April measurement.
   - **Mandatory Policy**: AMDFE maps each observation at year $Y$ to the latest completed annual period ($Y-1$), unless the observation timestamp demonstrably succeeds completion of year $Y$.

3. **Missing Static & Remote Sensing Modalities**:
   - `DEM_Elevation` is entirely null in existing datasets.
   - Satellite indices (NDVI, NDWI, GRACE) and SoilGrids physical parameters are currently absent from the local repository.
   - **Mandatory Policy**: Modality D (Satellite/Soil) is marked `UNAVAILABLE_IN_CURRENT_REPOSITORY` with modular plug-in interfaces. No fake or synthetic data is manufactured for the primary real-data pipeline.

---

## 4. Modality Definitions for AMDFE

AMDFE structures available inputs into 4 distinct modalities:

- **Modality A (Groundwater State / History)**:
  - Strict prior-only temporal lags: `Previous_WatLevel`, `Previous2_WatLevel`, `Days_Since_Previous`, `Years_Since_Previous`, `Rolling_Mean_3`, `Rolling_Std_3`, `Previous_Level_Change`, `Recent_Trend`, `Historical_Mean`, `Historical_Std`, `Historical_Min`, `Historical_Max`, `Previous_Observation_Count`, `Observation_Density`, `Long_Gap_Flag`, `Very_Long_Gap_Flag`.
- **Modality B (Environmental / Meteorological)**:
  - Leakage-safe lagged annual weather: `Annual_Temperature_Mean`, `Annual_Precipitation_Total`, `Annual_Humidity_Mean`, `Annual_WindSpeed_Mean`, `Annual_SolarRadiation_Mean` (evaluated from period $Y-1$).
- **Modality C (Spatial / Topographic Context)**:
  - Spatial coordinates and surface elevation: `LatDD`, `LongDD`, `Surf_Elev`.
- **Modality D (Earth Observation & Soil Physical Properties)**:
  - Extensible interface; marked unavailable in the current repository.

---

## 5. Files to Add vs. Files Intentionally Preserved

### Files Added:
1. `docs/amdfe_implementation_audit.md` (this audit document)
2. `docs/amdfe_methodology.md` (detailed scientific and mathematical formulation)
3. `docs/amdfe_research_positioning.md` (transparent research framing and hypothesis positioning)
4. `ml/amdfe/__init__.py`
5. `ml/amdfe/config.py` (modality configs, ablation keys B0–B4, leakage rules)
6. `ml/amdfe/ingestion.py` (leakage-safe ingestion and weather temporal matching)
7. `ml/amdfe/quality.py` (data quality dimension scoring $C, T, S, O, M$)
8. `ml/amdfe/reliability.py` (global & row-level reliability, aggregation sensitivity)
9. `ml/amdfe/missingness.py` (train-only fitted imputation and indicator generation)
10. `ml/amdfe/fusion.py` (adaptive weighting $w_{i,m}$, train-scaled block fusion)
11. `ml/amdfe/features.py` (feature matrix assembler for ablation suites B0–B4)
12. `ml/amdfe/pipeline.py` (scikit-learn compatible end-to-end `AMDFEPipeline`)
13. `ml/amdfe/validation.py` (automated 10-point leakage check suite)
14. `ml/experiments/run_amdfe_benchmark.py` (benchmark & ablation runner)
15. `ml/tests/test_amdfe_*.py` (unit and integration tests)
16. Processed datasets and reports in `data/processed/amdfe/`

### Files and Directories Left Strictly Untouched:
- `data/processed/integrated/` (historical integrated datasets)
- `data/processed/ml_validation/` (baseline validation datasets)
- `data/processed/advanced_ml/` (existing A0–A4 ablation, SHAP, recharge, uncertainty outputs)
- `src/01_*.py` through `src/08_10_*.py` (historical execution scripts)

---

## 6. Exact Integration Point of AMDFE

```
========================================================================================
                                AMDFE PIPELINE INTEGRATION
========================================================================================

           [Raw Groundwater Observations]        [Annual Weather Aggregates]
                        \                                    /
                         \                                  /
                          v                                v
                   [ml.amdfe.ingestion: Temporal Leakage-Safe Pairing]
                                          |
                                          v
                   [ml.data.loader: Unseen-Well / Temporal Splitting]
                                          |
                        +-----------------+-----------------+
                        | (TRAIN ONLY)                      | (VALIDATION / TEST)
                        v                                   v
             [ml.amdfe.quality: Quality Engine]      [Apply Frozen Scalers]
                        |                                   |
             [ml.amdfe.reliability: Reliability Fit] [Apply Frozen Weights]
                        |                                   |
             [ml.amdfe.missingness: Fit Imputers]    [Apply Frozen Imputers]
                        |                                   |
             [ml.amdfe.fusion: Standardize & Weight] <------+
                        |
                        v
          [ml.amdfe.features: Fused Representation (B0 - B4)]
                        |
                        v
         [ml.models.factory: LightGBM / XGBoost / Random Forest]
                        |
                        v
         [ml.evaluation.metrics: Generalization & Robustness Evaluation]
========================================================================================
```

AMDFE strictly fits all statistical transformers, imputers, scalers, and reliability baseline normalizations on the **TRAIN partition only**. Test and validation partitions are transformed using the frozen parameters without data snooping.
