# ================================================================
# AquaSense AI - ML BASELINE MODELS
# Step 07: Leakage-Safe Baseline Modeling
# ================================================================
#
# Experiments:
#   A. Spatial Baseline
#   B. Environmental Baseline
#
# Models:
#   1. Linear Regression
#   2. Random Forest
#   3. XGBoost
#   4. LightGBM
#
# Evaluation:
#   - Temporal split
#   - Well-group split
#
# Metrics:
#   - R2
#   - MAE
#   - RMSE
#
# IMPORTANT:
# TIFF_Value is intentionally NOT used in the primary models.
#
# ================================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

warnings.filterwarnings("ignore")


# ================================================================
# OPTIONAL BOOSTING LIBRARIES
# ================================================================

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("\nWARNING: XGBoost is not installed.")
    print("Install using:")
    print("pip install xgboost")


try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("\nWARNING: LightGBM is not installed.")
    print("Install using:")
    print("pip install lightgbm")


# ================================================================
# HEADER
# ================================================================

print("=" * 78)
print("AquaSense AI - ML BASELINE MODELS")
print("Step 07: Leakage-Safe Baseline Modeling")
print("=" * 78)


# ================================================================
# 1. PROJECT PATHS
# ================================================================

print("\n1. PROJECT PATHS")
print("-" * 78)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

INPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "ml_validation"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "ml_baseline"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_FILE = os.path.join(
    INPUT_DIR,
    "groundwater_ml_leakage_safe.csv"
)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nInput dataset:")
print(INPUT_FILE)

print("\nOutput directory:")
print(OUTPUT_DIR)


# ================================================================
# 2. CHECK INPUT DATASET
# ================================================================

print("\n\n2. LOCATING ML DATASET")
print("-" * 78)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nML dataset not found:\n{INPUT_FILE}\n"
        "\nRun 06_5_ml_dataset_validation.py first."
    )

print("\nML dataset found:")
print(INPUT_FILE)


# ================================================================
# 3. LOAD DATASET
# ================================================================

print("\n\n3. LOADING ML DATASET")
print("-" * 78)

df = pd.read_csv(INPUT_FILE)

print("\nRows loaded :", len(df))
print("Columns     :", len(df.columns))

print("\nColumns:")

for col in df.columns:
    print(" -", col)


# ================================================================
# 4. DEFINE FEATURES
# ================================================================

print("\n\n4. DEFINING ML FEATURES")
print("-" * 78)

TARGET = "WatLevel"

SPATIAL_FEATURES = [
    "LatDD",
    "LongDD",
    "Surf_Elev"
]

ENVIRONMENTAL_FEATURES = [
    "LatDD",
    "LongDD",
    "Surf_Elev",
    "Annual_Temperature_Mean",
    "Annual_Precipitation_Total",
    "Annual_Humidity_Mean",
    "Annual_WindSpeed_Mean",
    "Annual_SolarRadiation_Mean"
]

print("\nTarget:")
print(TARGET)

print("\nSpatial baseline features:")

for feature in SPATIAL_FEATURES:
    print(" -", feature)

print("\nEnvironmental baseline features:")

for feature in ENVIRONMENTAL_FEATURES:
    print(" -", feature)


# ================================================================
# 5. TARGET LEAKAGE CHECK
# ================================================================

print("\n\n5. TARGET LEAKAGE CHECK")
print("-" * 78)

if "TIFF_Value" in ENVIRONMENTAL_FEATURES:
    raise RuntimeError(
        "TIFF_Value must NOT be included in the primary model."
    )

print("\nTIFF_Value present in dataset:",
      "TIFF_Value" in df.columns)

print("TIFF_Value used as feature: NO")

print("\nReason:")
print("TIFF_Value was generated using Ordinary Kriging")
print("from groundwater observations and may contain")
print("target-derived information.")

print("\nPrimary models are therefore TIFF-free.")


# ================================================================
# 6. REQUIRED COLUMN CHECK
# ================================================================

print("\n\n6. REQUIRED COLUMN CHECK")
print("-" * 78)

required_columns = (
    ENVIRONMENTAL_FEATURES
    + [TARGET, "CSD_ID", "YearMsr"]
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("\nAll required columns are present.")


# ================================================================
# 7. BASIC CLEANING
# ================================================================

print("\n\n7. BASIC DATA CLEANING")
print("-" * 78)

df["YearMsr"] = pd.to_numeric(
    df["YearMsr"],
    errors="coerce"
)

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

for feature in ENVIRONMENTAL_FEATURES:
    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

before = len(df)

df = df.dropna(
    subset=["YearMsr", TARGET]
).copy()

after = len(df)

print("\nRows before cleaning :", before)
print("Rows after cleaning  :", after)
print("Rows removed         :", before - after)


# ================================================================
# 8. TEMPORAL SPLIT
# ================================================================

print("\n\n8. TEMPORAL TRAIN / VALIDATION / TEST SPLIT")
print("-" * 78)

TRAIN_START = 2000
TRAIN_END = 2019

VALIDATION_START = 2020
VALIDATION_END = 2022

TEST_START = 2023
TEST_END = 2024

train_df = df[
    (df["YearMsr"] >= TRAIN_START) &
    (df["YearMsr"] <= TRAIN_END)
].copy()

validation_df = df[
    (df["YearMsr"] >= VALIDATION_START) &
    (df["YearMsr"] <= VALIDATION_END)
].copy()

test_df = df[
    (df["YearMsr"] >= TEST_START) &
    (df["YearMsr"] <= TEST_END)
].copy()

print("\nTraining period:")
print(f"{TRAIN_START} - {TRAIN_END}")

print("\nValidation period:")
print(f"{VALIDATION_START} - {VALIDATION_END}")

print("\nTesting period:")
print(f"{TEST_START} - {TEST_END}")

print("\nTraining rows   :", len(train_df))
print("Validation rows :", len(validation_df))
print("Testing rows    :", len(test_df))

print("\nNOTE:")
print("2025 is excluded from the primary temporal test")
print("because 2025 weather features are currently unavailable.")


# ================================================================
# 9. TEMPORAL SPLIT VALIDATION
# ================================================================

if len(train_df) == 0:
    raise RuntimeError("Training dataset is empty.")

if len(validation_df) == 0:
    raise RuntimeError("Validation dataset is empty.")

if len(test_df) == 0:
    raise RuntimeError("Testing dataset is empty.")


# ================================================================
# 10. MODEL DEFINITIONS
# ================================================================

print("\n\n9. DEFINING ML MODELS")
print("-" * 78)

models = {}


# ------------------------------------------------
# Linear Regression
# ------------------------------------------------

models["Linear Regression"] = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LinearRegression()
        )
    ]
)


# ------------------------------------------------
# Random Forest
# ------------------------------------------------

models["Random Forest"] = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# ------------------------------------------------
# XGBoost
# ------------------------------------------------

if XGBOOST_AVAILABLE:

    models["XGBoost"] = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )


# ------------------------------------------------
# LightGBM
# ------------------------------------------------

if LIGHTGBM_AVAILABLE:

    models["LightGBM"] = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                LGBMRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=-1,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1
                )
            )
        ]
    )


print("\nModels selected:")

for name in models:
    print(" -", name)


# ================================================================
# 11. METRIC FUNCTION
# ================================================================

def calculate_metrics(y_true, y_pred):

    r2 = r2_score(
        y_true,
        y_pred
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    return r2, mae, rmse


# ================================================================
# 12. RUN TEMPORAL EXPERIMENTS
# ================================================================

print("\n\n10. TEMPORAL BASELINE EXPERIMENTS")
print("-" * 78)


def run_temporal_experiment(
    experiment_name,
    features
):

    print("\n")
    print("=" * 70)
    print(experiment_name)
    print("=" * 70)

    print("\nFeatures:")

    for feature in features:
        print(" -", feature)

    X_train = train_df[features]
    y_train = train_df[TARGET]

    X_validation = validation_df[features]
    y_validation = validation_df[TARGET]

    X_test = test_df[features]
    y_test = test_df[TARGET]

    results = []

    trained_models = {}

    for model_name, model in models.items():

        print("\nTraining:", model_name)

        model.fit(
            X_train,
            y_train
        )

        validation_pred = model.predict(
            X_validation
        )

        test_pred = model.predict(
            X_test
        )

        validation_r2, validation_mae, validation_rmse = (
            calculate_metrics(
                y_validation,
                validation_pred
            )
        )

        test_r2, test_mae, test_rmse = (
            calculate_metrics(
                y_test,
                test_pred
            )
        )

        print("\nValidation:")
        print(" R²   :", round(validation_r2, 6))
        print(" MAE  :", round(validation_mae, 6))
        print(" RMSE :", round(validation_rmse, 6))

        print("\nTest:")
        print(" R²   :", round(test_r2, 6))
        print(" MAE  :", round(test_mae, 6))
        print(" RMSE :", round(test_rmse, 6))

        results.append({
            "Experiment": experiment_name,
            "Model": model_name,

            "Validation_R2": validation_r2,
            "Validation_MAE": validation_mae,
            "Validation_RMSE": validation_rmse,

            "Test_R2": test_r2,
            "Test_MAE": test_mae,
            "Test_RMSE": test_rmse
        })

        trained_models[model_name] = model

    return (
        pd.DataFrame(results),
        trained_models
    )


# ================================================================
# 13. EXPERIMENT A
# ================================================================

spatial_results, spatial_models = run_temporal_experiment(
    "Experiment A - Spatial Baseline",
    SPATIAL_FEATURES
)


# ================================================================
# 14. EXPERIMENT B
# ================================================================

environmental_results, environmental_models = (
    run_temporal_experiment(
        "Experiment B - Environmental Baseline",
        ENVIRONMENTAL_FEATURES
    )
)


# ================================================================
# 15. COMBINE RESULTS
# ================================================================

print("\n\n11. COMBINING TEMPORAL RESULTS")
print("-" * 78)

temporal_results = pd.concat(
    [
        spatial_results,
        environmental_results
    ],
    ignore_index=True
)

print("\nTemporal results:")

print(
    temporal_results.to_string(
        index=False
    )
)


# ================================================================
# 16. SAVE TEMPORAL RESULTS
# ================================================================

temporal_results_file = os.path.join(
    OUTPUT_DIR,
    "temporal_baseline_results.csv"
)

temporal_results.to_csv(
    temporal_results_file,
    index=False
)

print("\nSaved:")
print(temporal_results_file)


# ================================================================
# 17. WELL-GROUP SPLIT
# ================================================================

print("\n\n12. WELL-GROUP SPLIT")
print("-" * 78)

print(
    "\nA well-group split ensures that observations"
    "\nfrom the same well do not appear in both"
    "\ntraining and testing datasets."
)

group_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

groups = df["CSD_ID"]

train_indices, group_test_indices = next(
    group_splitter.split(
        df,
        df[TARGET],
        groups
    )
)

group_train_df = df.iloc[
    train_indices
].copy()

group_test_df = df.iloc[
    group_test_indices
].copy()

print("\nGroup-training rows :", len(group_train_df))
print("Group-testing rows  :", len(group_test_df))

print(
    "\nTraining wells:",
    group_train_df["CSD_ID"].nunique()
)

print(
    "Testing wells :",
    group_test_df["CSD_ID"].nunique()
)


# ================================================================
# 18. RUN WELL-GROUP EXPERIMENTS
# ================================================================

print("\n\n13. WELL-GROUP BASELINE EXPERIMENTS")
print("-" * 78)


def run_group_experiment(
    experiment_name,
    features
):

    print("\n")
    print("=" * 70)
    print(experiment_name)
    print("=" * 70)

    X_train = group_train_df[features]
    y_train = group_train_df[TARGET]

    X_test = group_test_df[features]
    y_test = group_test_df[TARGET]

    results = []

    for model_name, original_model in models.items():

        print("\nTraining:", model_name)

        # Create a fresh model
        model = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                )
            ]
        )

        # Rebuild from the original pipeline
        if model_name == "Linear Regression":

            model = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    ),
                    (
                        "model",
                        LinearRegression()
                    )
                ]
            )

        elif model_name == "Random Forest":

            model = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=300,
                            random_state=42,
                            n_jobs=-1
                        )
                    )
                ]
            )

        elif model_name == "XGBoost" and XGBOOST_AVAILABLE:

            model = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),
                    (
                        "model",
                        XGBRegressor(
                            n_estimators=300,
                            max_depth=6,
                            learning_rate=0.05,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            objective="reg:squarederror",
                            random_state=42,
                            n_jobs=-1
                        )
                    )
                ]
            )

        elif model_name == "LightGBM" and LIGHTGBM_AVAILABLE:

            model = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),
                    (
                        "model",
                        LGBMRegressor(
                            n_estimators=300,
                            learning_rate=0.05,
                            num_leaves=31,
                            random_state=42,
                            n_jobs=-1,
                            verbosity=-1
                        )
                    )
                ]
            )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        r2, mae, rmse = calculate_metrics(
            y_test,
            predictions
        )

        print("\nR²   :", round(r2, 6))
        print("MAE  :", round(mae, 6))
        print("RMSE :", round(rmse, 6))

        results.append({
            "Experiment": experiment_name,
            "Model": model_name,
            "R2": r2,
            "MAE": mae,
            "RMSE": rmse
        })

    return pd.DataFrame(results)


group_spatial_results = run_group_experiment(
    "Experiment A - Spatial Baseline",
    SPATIAL_FEATURES
)

group_environmental_results = run_group_experiment(
    "Experiment B - Environmental Baseline",
    ENVIRONMENTAL_FEATURES
)


# ================================================================
# 19. SAVE GROUP RESULTS
# ================================================================

group_results = pd.concat(
    [
        group_spatial_results,
        group_environmental_results
    ],
    ignore_index=True
)

print("\n\n14. WELL-GROUP RESULTS")
print("-" * 78)

print(
    group_results.to_string(
        index=False
    )
)

group_results_file = os.path.join(
    OUTPUT_DIR,
    "well_group_baseline_results.csv"
)

group_results.to_csv(
    group_results_file,
    index=False
)

print("\nSaved:")
print(group_results_file)


# ================================================================
# 20. FEATURE IMPORTANCE
# ================================================================

print("\n\n15. FEATURE IMPORTANCE")
print("-" * 78)

feature_importance_records = []


def extract_feature_importance(
    trained_model,
    model_name,
    experiment_name,
    features
):

    try:

        estimator = trained_model.named_steps[
            "model"
        ]

        if not hasattr(
            estimator,
            "feature_importances_"
        ):
            return

        importances = (
            estimator.feature_importances_
        )

        for feature, importance in zip(
            features,
            importances
        ):

            feature_importance_records.append({
                "Experiment": experiment_name,
                "Model": model_name,
                "Feature": feature,
                "Importance": importance
            })

    except Exception as e:

        print(
            f"Could not extract importance for "
            f"{model_name}: {e}"
        )


for model_name, model in environmental_models.items():

    extract_feature_importance(
        model,
        model_name,
        "Experiment B - Environmental Baseline",
        ENVIRONMENTAL_FEATURES
    )


if feature_importance_records:

    feature_importance_df = pd.DataFrame(
        feature_importance_records
    )

    feature_importance_file = os.path.join(
        OUTPUT_DIR,
        "environmental_feature_importance.csv"
    )

    feature_importance_df.to_csv(
        feature_importance_file,
        index=False
    )

    print("\nSaved:")
    print(feature_importance_file)

else:

    print(
        "\nNo tree-model feature importance available."
    )


# ================================================================
# 21. SAVE BEST ENVIRONMENTAL MODELS
# ================================================================

print("\n\n16. SAVING TRAINED MODELS")
print("-" * 78)

for model_name, model in environmental_models.items():

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    model_file = os.path.join(
        OUTPUT_DIR,
        f"{safe_name}_environmental_model.joblib"
    )

    joblib.dump(
        model,
        model_file
    )

    print("\nSaved:")
    print(model_file)


# ================================================================
# 22. BEST TEMPORAL MODEL
# ================================================================

print("\n\n17. BEST TEMPORAL MODEL")
print("-" * 78)

environmental_temporal = temporal_results[
    temporal_results["Experiment"]
    == "Experiment B - Environmental Baseline"
].copy()

environmental_temporal = environmental_temporal.sort_values(
    by="Test_R2",
    ascending=False
)

best_temporal = environmental_temporal.iloc[0]

print("\nBest environmental baseline:")
print("Model:", best_temporal["Model"])
print("Test R²:", round(best_temporal["Test_R2"], 6))
print("Test MAE:", round(best_temporal["Test_MAE"], 6))
print("Test RMSE:", round(best_temporal["Test_RMSE"], 6))


# ================================================================
# 23. TEMPORAL MODEL COMPARISON PLOT
# ================================================================

print("\n\n18. CREATING MODEL COMPARISON PLOT")
print("-" * 78)

plot_df = environmental_temporal.copy()

plt.figure(figsize=(10, 6))

plt.bar(
    plot_df["Model"],
    plot_df["Test_R2"]
)

plt.xlabel("Model")
plt.ylabel("Test R²")
plt.title(
    "Environmental Baseline Models - Temporal Test R²"
)

plt.xticks(rotation=20)
plt.tight_layout()

plot_file = os.path.join(
    OUTPUT_DIR,
    "environmental_model_test_r2.png"
)

plt.savefig(
    plot_file,
    dpi=300
)

plt.close()

print("\nSaved:")
print(plot_file)


# ================================================================
# 24. MAE COMPARISON
# ================================================================

print("\n\n19. CREATING MAE COMPARISON")
print("-" * 78)

plt.figure(figsize=(10, 6))

plt.bar(
    plot_df["Model"],
    plot_df["Test_MAE"]
)

plt.xlabel("Model")
plt.ylabel("Test MAE")
plt.title(
    "Environmental Baseline Models - Temporal Test MAE"
)

plt.xticks(rotation=20)
plt.tight_layout()

mae_plot_file = os.path.join(
    OUTPUT_DIR,
    "environmental_model_test_mae.png"
)

plt.savefig(
    mae_plot_file,
    dpi=300
)

plt.close()

print("\nSaved:")
print(mae_plot_file)


# ================================================================
# 25. RMSE COMPARISON
# ================================================================

print("\n\n20. CREATING RMSE COMPARISON")
print("-" * 78)

plt.figure(figsize=(10, 6))

plt.bar(
    plot_df["Model"],
    plot_df["Test_RMSE"]
)

plt.xlabel("Model")
plt.ylabel("Test RMSE")
plt.title(
    "Environmental Baseline Models - Temporal Test RMSE"
)

plt.xticks(rotation=20)
plt.tight_layout()

rmse_plot_file = os.path.join(
    OUTPUT_DIR,
    "environmental_model_test_rmse.png"
)

plt.savefig(
    rmse_plot_file,
    dpi=300
)

plt.close()

print("\nSaved:")
print(rmse_plot_file)


# ================================================================
# 26. SAVE PREDICTIONS FROM BEST MODEL
# ================================================================

print("\n\n21. SAVING BEST MODEL PREDICTIONS")
print("-" * 78)

best_model_name = best_temporal["Model"]

best_model = environmental_models[
    best_model_name
]

best_predictions = best_model.predict(
    test_df[ENVIRONMENTAL_FEATURES]
)

prediction_df = test_df[
    [
        "CSD_ID",
        "YearMsr",
        "LatDD",
        "LongDD",
        TARGET
    ]
].copy()

prediction_df["Predicted_WatLevel"] = (
    best_predictions
)

prediction_df["Absolute_Error"] = (
    np.abs(
        prediction_df[TARGET]
        - prediction_df["Predicted_WatLevel"]
    )
)

prediction_file = os.path.join(
    OUTPUT_DIR,
    "best_environmental_model_test_predictions.csv"
)

prediction_df.to_csv(
    prediction_file,
    index=False
)

print("\nBest model:")
print(best_model_name)

print("\nSaved:")
print(prediction_file)


# ================================================================
# 27. FINAL SUMMARY
# ================================================================

print("\n\n")
print("=" * 78)
print("AquaSense AI - STEP 07 COMPLETED")
print("=" * 78)

print("\nDATASET")
print("-" * 78)

print("Original ML rows :", len(df))
print("Training rows    :", len(train_df))
print("Validation rows  :", len(validation_df))
print("Testing rows     :", len(test_df))

print("\nPRIMARY FEATURES")
print("-" * 78)

for feature in ENVIRONMENTAL_FEATURES:
    print(" -", feature)

print("\nTARGET")
print("-" * 78)

print(TARGET)

print("\nTIFF LEAKAGE")
print("-" * 78)

print("TIFF_Value used in primary model: NO")

print("\nBEST ENVIRONMENTAL MODEL")
print("-" * 78)

print("Model    :", best_model_name)
print(
    "Test R²  :",
    round(best_temporal["Test_R2"], 6)
)
print(
    "Test MAE :",
    round(best_temporal["Test_MAE"], 6)
)
print(
    "Test RMSE:",
    round(best_temporal["Test_RMSE"], 6)
)

print("\nOUTPUT DIRECTORY")
print("-" * 78)

print(OUTPUT_DIR)

print("\nNext stage:")
print("Step 07 results should be reviewed before")
print("proceeding to AMDFE / advanced modeling.")

print("\n")
print("=" * 78)
print("AquaSense AI - ANALYSIS FINISHED")
print("=" * 78)