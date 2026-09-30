import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 08.3 GENERALIZATION & ERROR ANALYSIS
# AquaSense AI - Member 4
# ============================================================

# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ml_validation",
    "groundwater_ml_leakage_safe.csv"
)

ADVANCED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "advanced_ml"
)

UNSEEN_DIR = os.path.join(
    ADVANCED_DIR,
    "unseen_well"
)

OUTPUT_DIR = os.path.join(
    ADVANCED_DIR,
    "generalization_analysis"
)

# ------------------------------------------------------------
# PATH CHECK
# ------------------------------------------------------------

print("=" * 70)
print("PATH CHECK")
print("=" * 70)

print("BASE_DIR:")
print(BASE_DIR)

print("\nDATA_FILE:")
print(DATA_FILE)

print("\nADVANCED_DIR:")
print(ADVANCED_DIR)

print("\nOUTPUT_DIR:")
print(OUTPUT_DIR)

# Check project directory
if not os.path.exists(BASE_DIR):
    raise FileNotFoundError(
        f"Project directory does not exist:\n{BASE_DIR}"
    )

# Check input dataset
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Input dataset does not exist:\n{DATA_FILE}"
    )

# Check advanced ML directory
if not os.path.exists(ADVANCED_DIR):
    raise FileNotFoundError(
        f"Advanced ML directory does not exist:\n{ADVANCED_DIR}"
    )

# Create only the required output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\nPath check successful.")
print("=" * 70)


# ============================================================
# 1. LOAD SOURCE DATA
# ============================================================

print("=" * 70)
print("08.3 GENERALIZATION & ERROR ANALYSIS")
print("=" * 70)

print("\n[1] Loading leakage-safe dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# 2. CALCULATE OBSERVATION COUNT PER WELL
# ============================================================

print("\n[2] Calculating historical observation count per well...")

well_counts = (
    df.groupby("CSD_ID")
    .size()
    .reset_index(name="Observation_Count")
)

print("Number of wells:", len(well_counts))

print("\nObservation count statistics:")
print(well_counts["Observation_Count"].describe())


# ============================================================
# 3. MODEL PREDICTION FILES
# ============================================================

models = {
    "Random Forest": {
        "temporal": os.path.join(
            ADVANCED_DIR,
            "random_forest_test_predictions.csv"
        ),
        "unseen": os.path.join(
            UNSEEN_DIR,
            "random_forest_unseen_well_predictions.csv"
        )
    },

    "XGBoost": {
        "temporal": os.path.join(
            ADVANCED_DIR,
            "xgboost_test_predictions.csv"
        ),
        "unseen": os.path.join(
            UNSEEN_DIR,
            "xgboost_unseen_well_predictions.csv"
        )
    },

    "LightGBM": {
        "temporal": os.path.join(
            ADVANCED_DIR,
            "lightgbm_test_predictions.csv"
        ),
        "unseen": os.path.join(
            UNSEEN_DIR,
            "lightgbm_unseen_well_predictions.csv"
        )
    }
}

# ============================================================
# 4. HELPER FUNCTION
# ============================================================

def calculate_metrics(data):
    """
    Calculate error statistics from prediction dataframe.
    """

    y_true = data["Actual"]
    y_pred = data["Predicted"]

    error = y_pred - y_true
    abs_error = np.abs(error)

    mae = np.mean(abs_error)
    rmse = np.sqrt(np.mean(error ** 2))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2 = 1 - (ss_res / ss_tot)

    return {
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "Mean_Error": np.mean(error),
        "Median_Absolute_Error": np.median(abs_error),
        "P90_Absolute_Error": np.percentile(abs_error, 90),
        "P95_Absolute_Error": np.percentile(abs_error, 95),
        "Max_Absolute_Error": np.max(abs_error)
    }


# ============================================================
# 5. LOAD AND STANDARDIZE PREDICTIONS
# ============================================================

print("\n[3] Loading prediction files...")

prediction_data = {}

for model_name, paths in models.items():

    print(f"\n--- {model_name} ---")

    model_results = {}

    for split_name, file_path in paths.items():

        if not os.path.exists(file_path):
            print("WARNING: File not found:")
            print(file_path)
            continue

        temp = pd.read_csv(file_path)

        print(
            split_name,
            "rows:",
            len(temp),
            "| columns:",
            list(temp.columns)
        )

        # ----------------------------------------------------
        # Detect actual/predicted column names
        # ----------------------------------------------------

        actual_candidates = [
            "Actual",
            "Actual_WatLevel",
            "WatLevel",
            "y_true"
        ]

        predicted_candidates = [
            "Predicted",
            "Predicted_WatLevel",
            "Prediction",
            "y_pred"
        ]

        actual_col = None
        predicted_col = None

        for col in actual_candidates:
            if col in temp.columns:
                actual_col = col
                break

        for col in predicted_candidates:
            if col in temp.columns:
                predicted_col = col
                break

        if actual_col is None or predicted_col is None:

            print("ERROR: Could not identify prediction columns.")
            print("Available columns:", list(temp.columns))

            continue

        temp = temp.rename(
            columns={
                actual_col: "Actual",
                predicted_col: "Predicted"
            }
        )

        # Use existing prediction error if available
        if "Prediction_Error" in temp.columns:
            temp["Error"] = temp["Prediction_Error"]
        else:
            temp["Error"] = temp["Predicted"] - temp["Actual"]

        temp["Absolute_Error"] = np.abs(temp["Error"])

        model_results[split_name] = temp

    prediction_data[model_name] = model_results


# ============================================================
# 6. TEMPORAL VS UNSEEN-WELL COMPARISON
# ============================================================

print("\n[4] Creating temporal vs unseen-well comparison...")

comparison_rows = []

for model_name, splits in prediction_data.items():

    for split_name, data in splits.items():

        metrics = calculate_metrics(data)

        comparison_rows.append({
            "Model": model_name,
            "Split": split_name,
            **metrics
        })


comparison_df = pd.DataFrame(comparison_rows)

print("\nPerformance comparison:")
print(comparison_df.to_string(index=False))

comparison_file = os.path.join(
    OUTPUT_DIR,
    "temporal_vs_unseen_comparison.csv"
)

comparison_df.to_csv(
    comparison_file,
    index=False
)


# ============================================================
# 7. GENERALIZATION GAP
# ============================================================

print("\n[5] Calculating generalization gap...")

gap_rows = []

for model_name in models.keys():

    model_df = comparison_df[
        comparison_df["Model"] == model_name
    ]

    temporal_row = model_df[
        model_df["Split"] == "temporal"
    ]

    unseen_row = model_df[
        model_df["Split"] == "unseen"
    ]

    if len(temporal_row) == 0 or len(unseen_row) == 0:
        continue

    temporal_r2 = temporal_row["R2"].iloc[0]
    unseen_r2 = unseen_row["R2"].iloc[0]

    temporal_mae = temporal_row["MAE"].iloc[0]
    unseen_mae = unseen_row["MAE"].iloc[0]

    gap_rows.append({
        "Model": model_name,
        "Temporal_R2": temporal_r2,
        "Unseen_Well_R2": unseen_r2,
        "R2_Generalization_Gap": temporal_r2 - unseen_r2,
        "Temporal_MAE": temporal_mae,
        "Unseen_Well_MAE": unseen_mae,
        "MAE_Increase": unseen_mae - temporal_mae
    })


gap_df = pd.DataFrame(gap_rows)

print("\nGeneralization gap:")
print(gap_df.to_string(index=False))

gap_file = os.path.join(
    OUTPUT_DIR,
    "generalization_gap.csv"
)

gap_df.to_csv(
    gap_file,
    index=False
)


# ============================================================
# 8. ERROR DISTRIBUTION
# ============================================================

print("\n[6] Analyzing error distributions...")

error_rows = []

for model_name, splits in prediction_data.items():

    for split_name, data in splits.items():

        metrics = calculate_metrics(data)

        error_rows.append({
            "Model": model_name,
            "Split": split_name,
            "Mean_Error": metrics["Mean_Error"],
            "Median_Absolute_Error": metrics["Median_Absolute_Error"],
            "P90_Absolute_Error": metrics["P90_Absolute_Error"],
            "P95_Absolute_Error": metrics["P95_Absolute_Error"],
            "Max_Absolute_Error": metrics["Max_Absolute_Error"]
        })


error_df = pd.DataFrame(error_rows)

error_file = os.path.join(
    OUTPUT_DIR,
    "error_statistics.csv"
)

error_df.to_csv(
    error_file,
    index=False
)

print("\nError statistics:")
print(error_df.to_string(index=False))


# ============================================================
# 9. ANALYZE UNSEEN-WELL PERFORMANCE
# ============================================================

print("\n[7] Analyzing unseen-well errors...")

for model_name, splits in prediction_data.items():

    if "unseen" not in splits:
        continue

    data = splits["unseen"].copy()

    # --------------------------------------------------------
    # Merge observation count
    # --------------------------------------------------------

    if "CSD_ID" in data.columns:

        data = data.merge(
            well_counts,
            on="CSD_ID",
            how="left"
        )

    # --------------------------------------------------------
    # Sort worst observations
    # --------------------------------------------------------

    worst = data.sort_values(
        "Absolute_Error",
        ascending=False
    ).head(20)

    worst_file = os.path.join(
        OUTPUT_DIR,
        model_name.lower().replace(" ", "_")
        + "_worst_predictions.csv"
    )

    worst.to_csv(
        worst_file,
        index=False
    )

    # --------------------------------------------------------
    # Per-well performance
    # --------------------------------------------------------

    if "CSD_ID" in data.columns:

        well_error = (
            data.groupby("CSD_ID")
            .agg(
                Mean_Absolute_Error=(
                    "Absolute_Error",
                    "mean"
                ),
                RMSE=(
                    "Error",
                    lambda x: np.sqrt(np.mean(x ** 2))
                ),
                Max_Absolute_Error=(
                    "Absolute_Error",
                    "max"
                ),
                Observation_Count=(
                    "Observation_Count",
                    "first"
                ),
                Latitude=(
                    "LatDD",
                    "first"
                ) if "LatDD" in data.columns else (
                    "CSD_ID",
                    "size"
                ),
                Longitude=(
                    "LongDD",
                    "first"
                ) if "LongDD" in data.columns else (
                    "CSD_ID",
                    "size"
                )
            )
            .reset_index()
        )

        well_error = well_error.sort_values(
            "Mean_Absolute_Error",
            ascending=False
        )

        well_error_file = os.path.join(
            OUTPUT_DIR,
            model_name.lower().replace(" ", "_")
            + "_well_error_summary.csv"
        )

        well_error.to_csv(
            well_error_file,
            index=False
        )

        print(
            f"{model_name}: analyzed",
            len(well_error),
            "wells"
        )


# ============================================================
# 10. ERROR BY YEAR
# ============================================================

print("\n[8] Analyzing error by year...")

for model_name, splits in prediction_data.items():

    if "unseen" not in splits:
        continue

    data = splits["unseen"].copy()

    if "YearMsr" not in data.columns:
        continue

    yearly = (
        data.groupby("YearMsr")
        .agg(
            Mean_Absolute_Error=(
                "Absolute_Error",
                "mean"
            ),
            RMSE=(
                "Error",
                lambda x: np.sqrt(np.mean(x ** 2))
            ),
            Mean_Error=(
                "Error",
                "mean"
            ),
            Number_of_Observations=(
                "Error",
                "size"
            )
        )
        .reset_index()
    )

    yearly_file = os.path.join(
        OUTPUT_DIR,
        model_name.lower().replace(" ", "_")
        + "_error_by_year.csv"
    )

    yearly.to_csv(
        yearly_file,
        index=False
    )

    print(f"\n{model_name} error by year:")
    print(yearly.to_string(index=False))


# ============================================================
# 11. ERROR VS OBSERVATION COUNT
# ============================================================

print("\n[9] Analyzing error vs observation count...")

observation_relationship_rows = []

for model_name, splits in prediction_data.items():

    if "unseen" not in splits:
        continue

    data = splits["unseen"].copy()

    if "CSD_ID" not in data.columns:
        continue

    data = data.merge(
        well_counts,
        on="CSD_ID",
        how="left"
    )

    if "Observation_Count" not in data.columns:
        continue

    relationship = data[
        [
            "Observation_Count",
            "Absolute_Error"
        ]
    ].corr().iloc[0, 1]

    observation_relationship_rows.append({
        "Model": model_name,
        "Correlation_Observation_Count_vs_Absolute_Error":
            relationship
    })


observation_relationship_df = pd.DataFrame(
    observation_relationship_rows
)

observation_file = os.path.join(
    OUTPUT_DIR,
    "observation_count_error_relationship.csv"
)

observation_relationship_df.to_csv(
    observation_file,
    index=False
)

print("\nObservation count vs error correlation:")
print(
    observation_relationship_df.to_string(index=False)
)


# ============================================================
# 12. ERROR VS OBSERVED GROUNDWATER LEVEL
# ============================================================

print("\n[10] Analyzing error vs groundwater level...")

level_rows = []

for model_name, splits in prediction_data.items():

    if "unseen" not in splits:
        continue

    data = splits["unseen"]

    correlation = data[
        [
            "Actual",
            "Absolute_Error"
        ]
    ].corr().iloc[0, 1]

    level_rows.append({
        "Model": model_name,
        "Correlation_Actual_Level_vs_Absolute_Error":
            correlation
    })


level_df = pd.DataFrame(level_rows)

level_file = os.path.join(
    OUTPUT_DIR,
    "groundwater_level_error_relationship.csv"
)

level_df.to_csv(
    level_file,
    index=False
)

print("\nGroundwater level vs error correlation:")
print(
    level_df.to_string(index=False)
)


# ============================================================
# 13. SPATIAL ERROR DATA
# ============================================================

print("\n[11] Preparing spatial error data...")

for model_name, splits in prediction_data.items():

    if "unseen" not in splits:
        continue

    data = splits["unseen"].copy()

    spatial_columns = [
        "CSD_ID",
        "LatDD",
        "LongDD",
        "Actual",
        "Predicted",
        "Error",
        "Absolute_Error"
    ]

    spatial_columns = [
        col for col in spatial_columns
        if col in data.columns
    ]

    spatial = data[spatial_columns].copy()

    spatial_file = os.path.join(
        OUTPUT_DIR,
        model_name.lower().replace(" ", "_")
        + "_spatial_errors.csv"
    )

    spatial.to_csv(
        spatial_file,
        index=False
    )


# ============================================================
# 14. PLOT R2 COMPARISON
# ============================================================

print("\n[12] Creating performance plots...")

if not comparison_df.empty:

    pivot_r2 = comparison_df.pivot(
        index="Model",
        columns="Split",
        values="R2"
    )

    pivot_r2.plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.ylabel("R²")
    plt.title("Temporal vs Unseen-Well R²")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "r2_temporal_vs_unseen.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 15. PLOT MAE COMPARISON
# ============================================================

if not comparison_df.empty:

    pivot_mae = comparison_df.pivot(
        index="Model",
        columns="Split",
        values="MAE"
    )

    pivot_mae.plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.ylabel("MAE")
    plt.title("Temporal vs Unseen-Well MAE")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "mae_temporal_vs_unseen.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 16. PLOT UNSEEN ERROR DISTRIBUTION
# ============================================================

for model_name, splits in prediction_data.items():

    if "unseen" not in splits:
        continue

    data = splits["unseen"]

    plt.figure(figsize=(10, 6))

    plt.hist(
        data["Error"],
        bins=30
    )

    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")

    plt.title(
        f"{model_name} - Unseen-Well Error Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            model_name.lower().replace(" ", "_")
            + "_error_distribution.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 17. OBSERVED VS PREDICTED
# ============================================================

for model_name, splits in prediction_data.items():

    if "unseen" not in splits:
        continue

    data = splits["unseen"]

    plt.figure(figsize=(7, 7))

    plt.scatter(
        data["Actual"],
        data["Predicted"],
        alpha=0.6
    )

    minimum = min(
        data["Actual"].min(),
        data["Predicted"].min()
    )

    maximum = max(
        data["Actual"].max(),
        data["Predicted"].max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum]
    )

    plt.xlabel("Observed Groundwater Level")
    plt.ylabel("Predicted Groundwater Level")

    plt.title(
        f"{model_name} - Unseen-Well Predictions"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            model_name.lower().replace(" ", "_")
            + "_observed_vs_predicted.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 18. SPATIAL ERROR PLOT
# ============================================================

for model_name, splits in prediction_data.items():

    if "unseen" not in splits:
        continue

    data = splits["unseen"]

    if "LatDD" not in data.columns or "LongDD" not in data.columns:
        continue

    plt.figure(figsize=(9, 7))

    scatter = plt.scatter(
        data["LongDD"],
        data["LatDD"],
        c=data["Absolute_Error"],
        s=35,
        alpha=0.8
    )

    plt.colorbar(
        scatter,
        label="Absolute Error"
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.title(
        f"{model_name} - Spatial Distribution of Prediction Error"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            model_name.lower().replace(" ", "_")
            + "_spatial_error.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 19. ERROR VS OBSERVATION COUNT PLOT
# ============================================================

for model_name, splits in prediction_data.items():

    if "unseen" not in splits:
        continue

    data = splits["unseen"].copy()

    if "CSD_ID" not in data.columns:
        continue

    data = data.merge(
        well_counts,
        on="CSD_ID",
        how="left"
    )

    plt.figure(figsize=(9, 6))

    plt.scatter(
        data["Observation_Count"],
        data["Absolute_Error"],
        alpha=0.6
    )

    plt.xlabel("Historical Observation Count per Well")
    plt.ylabel("Absolute Prediction Error")

    plt.title(
        f"{model_name} - Error vs Observation Count"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            model_name.lower().replace(" ", "_")
            + "_error_vs_observation_count.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("08.3 ANALYSIS COMPLETED")
print("=" * 70)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated analysis files include:")

for file in sorted(os.listdir(OUTPUT_DIR)):
    print(" -", file)

print("\nKey files:")
print(" - temporal_vs_unseen_comparison.csv")
print(" - generalization_gap.csv")
print(" - error_statistics.csv")
print(" - observation_count_error_relationship.csv")
print(" - groundwater_level_error_relationship.csv")

print("\nNext step:")
print("Review the complete terminal output before moving to")
print("irregular-observation feature engineering.")

print("=" * 70)