import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_validation"
    / "groundwater_ml_leakage_safe.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "advanced_ml"
    / "unseen_well"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

FEATURES = [
    "LatDD",
    "LongDD",
    "Surf_Elev",
    "Annual_Temperature_Mean",
    "Annual_Precipitation_Total",
    "Annual_Humidity_Mean",
    "Annual_WindSpeed_Mean",
    "Annual_SolarRadiation_Mean"
]

TARGET = "WatLevel"

GROUP_COLUMN = "CSD_ID"


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 08.2 - ADVANCED ML: UNSEEN-WELL EVALUATION")
print("=" * 70)

print("\nLoading dataset...")
print(f"Input: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print(f"\nDataset shape: {df.shape}")
print(f"Unique wells: {df[GROUP_COLUMN].nunique()}")


# ============================================================
# 4. VERIFY COLUMNS
# ============================================================

required_columns = (
    FEATURES
    + [TARGET, GROUP_COLUMN]
)

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

before = len(df)

df = df.dropna(
    subset=FEATURES + [TARGET]
).copy()

after = len(df)

print("\nMissing-value handling:")
print(f"Rows before : {before}")
print(f"Rows after  : {after}")
print(f"Rows removed: {before - after}")


# ============================================================
# 6. IDENTIFY UNIQUE WELLS
# ============================================================

unique_wells = (
    df[GROUP_COLUMN]
    .drop_duplicates()
    .values
)

print("\nTotal unique wells:", len(unique_wells))


# ============================================================
# 7. SPLIT WELLS
# ============================================================

train_wells, test_wells = train_test_split(
    unique_wells,
    test_size=0.20,
    random_state=RANDOM_STATE
)


# ============================================================
# 8. CREATE TRAIN / TEST DATASETS
# ============================================================

train_df = df[
    df[GROUP_COLUMN].isin(train_wells)
].copy()

test_df = df[
    df[GROUP_COLUMN].isin(test_wells)
].copy()


print("\n" + "=" * 70)
print("UNSEEN-WELL SPLIT")
print("=" * 70)

print(
    f"Training wells: {len(train_wells)}"
)

print(
    f"Testing wells : {len(test_wells)}"
)

print(
    f"Training rows : {len(train_df)}"
)

print(
    f"Testing rows  : {len(test_df)}"
)


# ============================================================
# 9. VERIFY NO WELL OVERLAP
# ============================================================

train_well_set = set(train_wells)
test_well_set = set(test_wells)

overlap = train_well_set.intersection(
    test_well_set
)

print(
    f"\nTrain/Test well overlap: {len(overlap)}"
)

if len(overlap) > 0:

    raise RuntimeError(
        "ERROR: Train and test wells overlap!"
    )

print(
    "[OK] No well appears in both training and testing."
)


# ============================================================
# 10. CREATE FEATURE MATRICES
# ============================================================

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


# ============================================================
# 11. MODEL DEFINITIONS
# ============================================================

models = {

    "Random Forest": RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1
    )
}


# ============================================================
# 12. METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

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


# ============================================================
# 13. TRAIN AND EVALUATE MODELS
# ============================================================

results = []


for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    r2, mae, rmse = calculate_metrics(
        y_test,
        predictions
    )

    print("\nUnseen-Well Test Performance:")

    print(
        f"R²   : {r2:.6f}"
    )

    print(
        f"MAE  : {mae:.6f}"
    )

    print(
        f"RMSE : {rmse:.6f}"
    )

    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    results.append({

        "Model": model_name,

        "Training_Wells":
            len(train_wells),

        "Testing_Wells":
            len(test_wells),

        "Training_Rows":
            len(train_df),

        "Testing_Rows":
            len(test_df),

        "Test_R2":
            r2,

        "Test_MAE":
            mae,

        "Test_RMSE":
            rmse
    })


    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    prediction_df = test_df[
        [
            "CSD_ID",
            "County",
            "LatDD",
            "LongDD",
            "YearMsr",
            "DateMsr",
            "WatLevel"
        ]
    ].copy()

    prediction_df[
        "Predicted_WatLevel"
    ] = predictions

    prediction_df[
        "Prediction_Error"
    ] = (
        prediction_df["WatLevel"]
        - prediction_df["Predicted_WatLevel"]
    )

    prediction_df[
        "Absolute_Error"
    ] = np.abs(
        prediction_df["Prediction_Error"]
    )

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    prediction_file = (
        OUTPUT_DIR
        / f"{safe_name}_unseen_well_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_file,
        index=False
    )

    print(
        f"\nPrediction file saved:"
    )

    print(prediction_file)


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    if hasattr(
        model,
        "feature_importances_"
    ):

        importance_df = pd.DataFrame({

            "Feature":
                FEATURES,

            "Importance":
                model.feature_importances_
        })

        importance_df = (
            importance_df
            .sort_values(
                by="Importance",
                ascending=False
            )
            .reset_index(drop=True)
        )

        importance_file = (
            OUTPUT_DIR
            / f"{safe_name}_unseen_well_feature_importance.csv"
        )

        importance_df.to_csv(
            importance_file,
            index=False
        )

        print("\nTop Features:")

        print(
            importance_df
            .head(8)
            .to_string(index=False)
        )


# ============================================================
# 14. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = (
    results_df
    .sort_values(
        by="Test_R2",
        ascending=False
    )
    .reset_index(drop=True)
)


comparison_file = (
    OUTPUT_DIR
    / "unseen_well_model_comparison.csv"
)

results_df.to_csv(
    comparison_file,
    index=False
)


# ============================================================
# 15. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("UNSEEN-WELL MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 16. BEST MODEL
# ============================================================

best_model = results_df.iloc[0]

print("\n" + "=" * 70)
print("BEST UNSEEN-WELL MODEL")
print("=" * 70)

print(
    f"Model: {best_model['Model']}"
)

print(
    f"Test R²: "
    f"{best_model['Test_R2']:.6f}"
)

print(
    f"Test MAE: "
    f"{best_model['Test_MAE']:.6f}"
)

print(
    f"Test RMSE: "
    f"{best_model['Test_RMSE']:.6f}"
)


# ============================================================
# 17. SAVE WELL SPLIT
# ============================================================

train_well_df = pd.DataFrame({
    "CSD_ID": train_wells,
    "Split": "TRAIN"
})

test_well_df = pd.DataFrame({
    "CSD_ID": test_wells,
    "Split": "TEST"
})

well_split_df = pd.concat(
    [
        train_well_df,
        test_well_df
    ],
    ignore_index=True
)

well_split_file = (
    OUTPUT_DIR
    / "unseen_well_split.csv"
)

well_split_df.to_csv(
    well_split_file,
    index=False
)


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(
    f"\nOutput directory:\n{OUTPUT_DIR}"
)

print(
    f"\nModel comparison:\n{comparison_file}"
)

print(
    f"\nWell split:\n{well_split_file}"
)

print("\nStep 08.2 completed successfully.")