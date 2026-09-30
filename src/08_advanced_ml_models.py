import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 08 - ADVANCED ML MODELS")
print("=" * 70)

print("\nLoading dataset...")
print(f"Input: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print(f"\nDataset shape: {df.shape}")
print(f"Number of wells: {df['CSD_ID'].nunique()}")


# ============================================================
# 3. FEATURES AND TARGET
# ============================================================

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


# ============================================================
# 4. VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = FEATURES + [TARGET, "CSD_ID", "YearMsr"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 5. REMOVE ROWS WITH MISSING ML FEATURES
# ============================================================

before = len(df)

df = df.dropna(
    subset=FEATURES + [TARGET]
).copy()

after = len(df)

print("\nMissing-value handling:")
print(f"Rows before: {before}")
print(f"Rows after : {after}")
print(f"Rows removed: {before - after}")


# ============================================================
# 6. SORT DATA TEMPORALLY
# ============================================================

df = df.sort_values(
    by=["YearMsr", "CSD_ID"]
).reset_index(drop=True)


# ============================================================
# 7. TEMPORAL SPLIT
# ============================================================

train_df = df[
    df["YearMsr"] <= 2019
].copy()

validation_df = df[
    (df["YearMsr"] >= 2020) &
    (df["YearMsr"] <= 2022)
].copy()

test_df = df[
    (df["YearMsr"] >= 2023) &
    (df["YearMsr"] <= 2024)
].copy()


print("\n" + "=" * 70)
print("TEMPORAL SPLIT")
print("=" * 70)

print(
    f"Training   : {len(train_df)} rows "
    f"({train_df['YearMsr'].min()}-{train_df['YearMsr'].max()})"
)

print(
    f"Validation : {len(validation_df)} rows "
    f"({validation_df['YearMsr'].min()}-{validation_df['YearMsr'].max()})"
)

print(
    f"Testing    : {len(test_df)} rows "
    f"({test_df['YearMsr'].min()}-{test_df['YearMsr'].max()})"
)


X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_val = validation_df[FEATURES]
y_val = validation_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]


# ============================================================
# 8. MODEL DEFINITIONS
# ============================================================

models = {

    "Random Forest": RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
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
        random_state=42,
        n_jobs=-1
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
}


# ============================================================
# 9. METRIC FUNCTION
# ============================================================

def calculate_metrics(y_true, y_pred):

    r2 = r2_score(y_true, y_pred)

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
# 10. TRAIN + VALIDATE + TEST
# ============================================================

results = []

prediction_files = []


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
    # VALIDATION
    # --------------------------------------------------------

    val_predictions = model.predict(
        X_val
    )

    val_r2, val_mae, val_rmse = calculate_metrics(
        y_val,
        val_predictions
    )

    print("\nValidation Performance:")
    print(f"R²   : {val_r2:.6f}")
    print(f"MAE  : {val_mae:.6f}")
    print(f"RMSE : {val_rmse:.6f}")

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_predictions = model.predict(
        X_test
    )

    test_r2, test_mae, test_rmse = calculate_metrics(
        y_test,
        test_predictions
    )

    print("\nTest Performance:")
    print(f"R²   : {test_r2:.6f}")
    print(f"MAE  : {test_mae:.6f}")
    print(f"RMSE : {test_rmse:.6f}")

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results.append({
        "Model": model_name,
        "Validation_R2": val_r2,
        "Validation_MAE": val_mae,
        "Validation_RMSE": val_rmse,
        "Test_R2": test_r2,
        "Test_MAE": test_mae,
        "Test_RMSE": test_rmse
    })

    # --------------------------------------------------------
    # SAVE TEST PREDICTIONS
    # --------------------------------------------------------

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
    ] = test_predictions

    prediction_df[
        "Prediction_Error"
    ] = (
        prediction_df["WatLevel"]
        - prediction_df["Predicted_WatLevel"]
    )

    safe_name = model_name.lower().replace(
        " ",
        "_"
    )

    prediction_file = (
        OUTPUT_DIR
        / f"{safe_name}_test_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_file,
        index=False
    )

    prediction_files.append(
        str(prediction_file)
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    if hasattr(model, "feature_importances_"):

        importance_df = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": model.feature_importances_
        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        )

        importance_file = (
            OUTPUT_DIR
            / f"{safe_name}_feature_importance.csv"
        )

        importance_df.to_csv(
            importance_file,
            index=False
        )

        print("\nTop Features:")

        print(
            importance_df.head(5).to_string(
                index=False
            )
        )


# ============================================================
# 11. SAVE MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Test_R2",
    ascending=False
)

results_file = (
    OUTPUT_DIR
    / "advanced_model_comparison.csv"
)

results_df.to_csv(
    results_file,
    index=False
)


# ============================================================
# 12. DISPLAY FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL ADVANCED MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 13. BEST MODEL
# ============================================================

best_model = results_df.iloc[0]

print("\n" + "=" * 70)
print("BEST TEMPORAL MODEL")
print("=" * 70)

print(
    f"Model: {best_model['Model']}"
)

print(
    f"Validation R²: "
    f"{best_model['Validation_R2']:.6f}"
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
# 14. OUTPUT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(
    f"\nResults directory:\n{OUTPUT_DIR}"
)

print(
    f"\nComparison file:\n{results_file}"
)

for file in prediction_files:
    print(
        f"Prediction file:\n{file}"
    )

print("\nStep 08 temporal experiment completed.")