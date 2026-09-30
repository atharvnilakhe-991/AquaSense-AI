# AquaSense AI — Machine Learning Module

## Hyperlocal Groundwater-Level Prediction & Model Evaluation
**Role:** Member 4 — AI/ML, Model Evaluation, XAI & Recharge  
**Target Branch:** `ml-training`  
**Status:** Experimental / Module-Level Baseline & Advanced Models

---

## 1. Purpose of the ML Module

The AquaSense AI Machine Learning module provides the core predictive intelligence for hyperlocal groundwater-level estimation. Groundwater monitoring networks frequently suffer from spatial sparsity, irregular observation intervals, and incomplete historical records.

The primary objective of this module is to:
1. Provide accurate, robust groundwater-level regression at localized monitoring wells.
2. Evaluate model generalization to completely **unseen wells** (spatial holdout) rather than relying solely on naive random row splits.
3. Establish a leakage-safe data and modeling pipeline free of target contamination.
4. Compare multiple gradient boosting and ensemble architectures (Random Forest, XGBoost, LightGBM) on standardized benchmark splits.
5. Provide a foundation for subsequent integration with Member 2's **Adaptive Multimodal Data Fusion Engine (AMDFE)**, explainable AI (SHAP), and recharge potential assessment.

> [!NOTE]
> The current results reported herein represent experimental, module-level findings. Final end-to-end system validation will take place following integration with AMDFE reliability scoring, backend APIs, and GIS dashboard services.

---

## 2. Dataset Used

The modeling dataset represents long-term monitoring across Phelps County, Nebraska, USA (High Plains / Ogallala Aquifer region):

- **Temporal Coverage:** 2000–2024 (25-year period aligned with NASA POWER meteorological data).
- **Monitoring Wells:** 170 distinct groundwater wells (`CSD_ID`).
- **Total Observations:** 3,844 verified groundwater-level measurements.
- **Dataset File:** `data/processed/ml_validation/groundwater_ml_leakage_safe.csv`

---

## 3. Target Variable & Input Features

### Target Variable
- **`WatLevel`**: Measured depth to groundwater level (ft) at the designated well and date.

### Primary Input Features (Leakage-Safe Predictors)
| Feature Name | Description | Source / Type |
|---|---|---|
| `LatDD` | Well Latitude (decimal degrees) | Spatial Coordinates |
| `LongDD` | Well Longitude (decimal degrees) | Spatial Coordinates |
| `Surf_Elev` | Surface Elevation (ft above sea level) | SRTM / Topography |
| `Annual_Temperature_Mean` | Annual Mean Air Temperature (°C) | NASA POWER Weather |
| `Annual_Precipitation_Total`| Total Annual Precipitation (mm) | NASA POWER Weather |
| `Annual_Humidity_Mean` | Annual Mean Relative Humidity (%) | NASA POWER Weather |
| `Annual_WindSpeed_Mean` | Annual Mean Wind Speed at 2m (m/s) | NASA POWER Weather |
| `Annual_SolarRadiation_Mean`| Annual Mean All-Sky Surface Solar Radiation (MJ/m²/day) | NASA POWER Weather |

---

## 4. Strict Leakage Prevention Approach

A critical scientific requirement of AquaSense AI is **target leakage prevention**:

1. **Exclusion of Interpolated Groundwater TIFFs:**  
   Groundwater depth rasters (`Groundwater_2000.tif` through `Groundwater_2025.tif`) were derived from the same well observations used as targets. Using values sampled from these rasters (`TIFF_Value`) as predictors artificially inflates model accuracy (producing artificial R² > 0.999), constituting severe target leakage. `TIFF_Value` is strictly banned and excluded from all feature sets.
2. **Strictly Prior Temporal Features:**  
   All observation-history features (lagged water levels, rolling trends, gap days) are engineered using *strictly earlier* observation dates ($t_{prev} < t$). Observations from the current date or future dates are never incorporated into past features.
3. **Grouped Spatial Holdout:**  
   To prevent spatial autocorrelation leakage, train/test splitting is grouped by unique well identifier (`CSD_ID`). Observations from the same well never appear in both the training set and testing set during unseen-well evaluation.

---

## 5. Preprocessing & Data Cleaning

- **Missing Value Audit:** Complete rows with valid coordinates, surface elevation, meteorological parameters, and water levels are audited. In the leakage-safe curated set (`3,844` rows), zero missing values exist across the 8 primary features.
- **Coordinate & Elevation Verification:** All well coordinates fall within Phelps County bounds (`-99.65°` to `-99.10° W`, `40.30°` to `40.65° N`), and elevations range between 670m and 765m.
- **Normalization / Scaling:** Tree-based models (Random Forest, XGBoost, LightGBM) natively handle disparate feature scales without requiring non-linear monotonic distortion.

---

## 6. Models Used & Hyperparameters

Three candidate ensemble regression architectures were evaluated under consistent random seeds (`random_state=42`):

### 1. LightGBM (`LGBMRegressor`) — Current Preferred Model
- `n_estimators`: 500
- `learning_rate`: 0.05
- `num_leaves`: 31
- `max_depth`: -1 (unconstrained depth with leaf-wise splitting)
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `random_state`: 42

### 2. XGBoost (`XGBRegressor`)
- `n_estimators`: 500
- `learning_rate`: 0.05
- `max_depth`: 6
- `min_child_weight`: 2
- `subsample`: 0.8
- `colsample_bytree`: 0.8
- `objective`: `reg:squarederror`
- `eval_metric`: `rmse`
- `random_state`: 42

### 3. Random Forest (`RandomForestRegressor`)
- `n_estimators`: 400
- `max_depth`: None
- `min_samples_split`: 2
- `min_samples_leaf`: 1
- `random_state`: 42
- `n_jobs`: -1

---

## 7. Evaluation Methodology & Current Results

### Unseen-Well Evaluation Protocol
- **Total Unique Wells:** 170 wells
- **Training Set:** 136 wells (80%), containing 3,066 observation records
- **Testing Set:** 34 wells (20%), containing 778 observation records
- **Well Overlap:** Exactly 0 overlapping wells

### Current Reproducible Benchmark Results

| Model | Training Wells | Testing Wells | Test R² | Test MAE (ft) | Test RMSE (ft) | Pearson $r$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **LightGBM** | 136 | 34 | **0.8107** | **12.75** | **21.55** | **0.9047** |
| **XGBoost** | 136 | 34 | **0.8046** | **12.75** | **21.89** | **0.8980** |
| **Random Forest** | 136 | 34 | **0.7848** | **14.53** | **22.97** | **0.8862** |

*Note: Results reproduced exactly via `python -m ml.experiments.run_benchmark` and verified in `data/processed/advanced_ml/unseen_well/unseen_well_model_comparison.csv`.*

---

## 8. Repository Organization

```text
ml/
├── __init__.py                 # ML package root
├── config.py                   # Centralized paths, features, target & split constants
├── data/
│   ├── __init__.py
│   └── loader.py               # Leakage-safe loader & unseen-well/temporal splitters
├── preprocessing/
│   ├── __init__.py
│   └── cleaning.py             # Data validation, integrity checks & null cleaning
├── features/
│   ├── __init__.py
│   └── engineering.py          # Environmental matrix & observation-history features
├── models/
│   ├── __init__.py
│   └── factory.py              # Tuned Random Forest, XGBoost & LightGBM model factories
├── training/
│   ├── __init__.py
│   └── trainer.py              # Fitting and model training pipeline
├── evaluation/
│   ├── __init__.py
│   └── metrics.py              # R², MAE, RMSE, Pearson r computation & comparison
├── experiments/
│   ├── __init__.py
│   └── run_benchmark.py        # Standalone benchmark execution runner
└── README.md                   # This documentation
```

### Full Research Scripts (`src/`)
In addition to the modular `ml/` package, the complete step-by-step experimental research pipeline is preserved in `src/`:
- `01_dataset_inspection.py` through `06_5_ml_dataset_validation.py`: Data cleaning, spatial GIS analysis, and leakage auditing.
- `07_ml_baseline_models.py`: Linear, Ridge, and Random Forest baselines.
- `08_advanced_ml_models.py` & `08_2_advanced_ml_unseen_wells.py`: Advanced tree ensemble benchmark on unseen wells.
- `08_3_generalization_error_analysis.py`: Generalization gap between random row splits and unseen well splits.
- `08_4_irregular_observation_features.py`: Historical temporal features under irregular sampling.
- `08_5_observation_aware_ml.py`: Observation-aware models combining environmental and historical modalities.
- `08_6_repeated_generalization_validation.py`: Multi-seed repeated grouped-well cross-validation.
- `08_7_feature_ablation.py`: Feature ablation isolating modality contributions.
- `08_8_explainable_ai_shap.py`: TreeSHAP global and local feature attribution.
- `08_9_uncertainty_quantification.py`: Prediction intervals and residual uncertainty.
- `08_10_stage1_data_audit.py` to `08_10_stage3_decision_support.py`: Stage 1–3 recharge modeling and decision support.

---

## 9. How to Run the Training Pipeline

### Prerequisites
Activate the Python virtual environment and ensure dependencies are installed:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Run the Modular Benchmark
Execute the automated unseen-well benchmark runner:
```powershell
python -m ml.experiments.run_benchmark
```

### Run Step-by-Step Research Scripts
```powershell
python src/08_2_advanced_ml_unseen_wells.py
python src/08_3_generalization_error_analysis.py
python src/08_5_observation_aware_ml.py
```

---

## 10. Expected Output Files

Running the benchmark pipeline generates structured evaluation artifacts in `data/processed/advanced_ml/unseen_well/`:

- `unseen_well_model_comparison.csv`: Summary table comparing Test R², MAE, RMSE, and sample counts.
- `unseen_well_split.csv`: Well assignment tracking table (136 train wells vs 34 test wells).
- `lightgbm_unseen_well_predictions.csv`: Actual vs predicted groundwater levels for all 778 test observations.
- `xgboost_unseen_well_predictions.csv`: Predictions from the XGBoost model.
- `random_forest_unseen_well_predictions.csv`: Predictions from the Random Forest model.

---

## 11. Known Limitations & Next Steps

1. **Single Split vs. Repeated Grouped K-Fold:**  
   The primary benchmark results ($R^2 \approx 0.8107$) are based on a fixed 80/20 unseen well split (`random_state=42`). Continued validation uses repeated multi-seed splits (`src/08_6_repeated_generalization_validation.py`) to confirm robustness against split variance.
2. **Environmental-Only Cold-Start:**  
   When predicting a completely new well with zero prior monitoring history (cold-start), predictions rely strictly on spatial coordinates and weather averages. Warm-start performance improves significantly once prior measurements are incorporated.
3. **Downstream Integration:**  
   These module-level models will receive quality-weighted feature vectors and missing-value flags once Member 2's AMDFE reliability module is finalized. Schema contracts with Backend (Member 3) and Frontend (Member 1) will then be locked.
