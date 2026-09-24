# =============================================================================
# AquaSense AI - ML DATASET & LEAKAGE VALIDATION
# Step 06.5
#
# Purpose:
#   Validate the integrated dataset before machine learning.
#
# Main objectives:
#   1. Check dataset integrity
#   2. Detect duplicate well/year observations
#   3. Analyse temporal structure
#   4. Analyse repeated wells
#   5. Detect potential target leakage
#   6. Separate safe and unsafe ML features
#   7. Check missing values
#   8. Create leakage-safe ML dataset
#   9. Recommend appropriate train/test strategies
#
# Target:
#   WatLevel
#
# Important:
#   TIFF_Value is NOT used in the primary ML dataset because the TIFFs
#   were generated using Ordinary Kriging from groundwater observations.
#
# =============================================================================


import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

print("=" * 78)
print("AquaSense AI - ML DATASET & LEAKAGE VALIDATION")
print("Step 06.5: Pre-ML Dataset Audit")
print("=" * 78)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

INTEGRATED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "integrated"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "ml_validation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("\n1. PROJECT PATHS")
print("-" * 78)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nIntegrated dataset directory:")
print(INTEGRATED_DIR)

print("\nValidation output directory:")
print(OUTPUT_DIR)


# =============================================================================
# 2. LOCATE INTEGRATED DATASET
# =============================================================================

print("\n\n2. LOCATING INTEGRATED DATASET")
print("-" * 78)


integrated_file = os.path.join(
    INTEGRATED_DIR,
    "groundwater_integrated_dataset.csv"
)

ml_ready_file = os.path.join(
    INTEGRATED_DIR,
    "groundwater_ml_ready.csv"
)


if not os.path.exists(integrated_file):

    print("\nERROR:")
    print("Integrated dataset was not found.")

    print("\nExpected file:")
    print(integrated_file)

    sys.exit(1)


print("\nIntegrated dataset found:")
print(integrated_file)


# =============================================================================
# 3. LOAD DATASET
# =============================================================================

print("\n\n3. LOADING INTEGRATED DATASET")
print("-" * 78)


df = pd.read_csv(integrated_file)


print("\nRows loaded :", len(df))
print("Columns     :", len(df.columns))

print("\nColumns:")

for column in df.columns:
    print(" -", column)


# =============================================================================
# 4. REQUIRED COLUMN CHECK
# =============================================================================

print("\n\n4. REQUIRED COLUMN CHECK")
print("-" * 78)


required_columns = [
    "CSD_ID",
    "County",
    "LatDD",
    "LongDD",
    "YearMsr",
    "DateMsr",
    "Season",
    "WatLevel",
    "Surf_Elev",
    "TIFF_Value",
    "Annual_Temperature_Mean",
    "Annual_Precipitation_Total",
    "Annual_Humidity_Mean",
    "Annual_WindSpeed_Mean",
    "Annual_SolarRadiation_Mean"
]


missing_required = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_required:

    print("\nERROR:")
    print("Missing required columns:")

    for column in missing_required:
        print(" -", column)

    sys.exit(1)


print("\nAll required columns are present.")


# =============================================================================
# 5. BASIC DATASET INFORMATION
# =============================================================================

print("\n\n5. BASIC DATASET INFORMATION")
print("-" * 78)


print("\nDataset shape:")
print(df.shape)


print("\nDuplicate complete rows:")
print(df.duplicated().sum())


print("\nData types:")

print(df.dtypes)


# =============================================================================
# 6. TARGET VARIABLE INSPECTION
# =============================================================================

print("\n\n6. TARGET VARIABLE INSPECTION")
print("-" * 78)


TARGET = "WatLevel"


print("\nTarget variable:")
print(TARGET)


print("\nMissing target values:")
print(df[TARGET].isna().sum())


print("\nTarget statistics:")
print(df[TARGET].describe())


# =============================================================================
# 7. MISSING VALUE ANALYSIS
# =============================================================================

print("\n\n7. MISSING VALUE ANALYSIS")
print("-" * 78)


missing_report = pd.DataFrame({
    "Column": df.columns,
    "Missing_Count": [
        df[column].isna().sum()
        for column in df.columns
    ]
})


missing_report["Missing_Percent"] = (
    missing_report["Missing_Count"]
    / len(df)
    * 100
)


print("\nMissing value report:")

print(
    missing_report.to_string(index=False)
)


missing_report_file = os.path.join(
    OUTPUT_DIR,
    "ml_validation_missing_values.csv"
)


missing_report.to_csv(
    missing_report_file,
    index=False
)


print("\nSaved:")
print(missing_report_file)


# =============================================================================
# 8. UNIQUE WELL ANALYSIS
# =============================================================================

print("\n\n8. UNIQUE WELL ANALYSIS")
print("-" * 78)


unique_wells = df["CSD_ID"].nunique()


print("\nUnique wells:")
print(unique_wells)


observations_per_well = (
    df.groupby("CSD_ID")
    .size()
    .reset_index(name="Observation_Count")
)


print("\nObservations per well statistics:")

print(
    observations_per_well["Observation_Count"].describe()
)


print("\nWells with more than one observation:")

multi_observation_wells = (
    observations_per_well[
        observations_per_well["Observation_Count"] > 1
    ]
)


print(len(multi_observation_wells))


well_observation_file = os.path.join(
    OUTPUT_DIR,
    "observations_per_well.csv"
)


observations_per_well.to_csv(
    well_observation_file,
    index=False
)


print("\nSaved:")
print(well_observation_file)


# =============================================================================
# 9. CSD_ID + YEAR DUPLICATE ANALYSIS
# =============================================================================

print("\n\n9. WELL + YEAR DUPLICATE ANALYSIS")
print("-" * 78)


well_year_counts = (
    df.groupby(
        ["CSD_ID", "YearMsr"]
    )
    .size()
    .reset_index(name="Observation_Count")
)


duplicate_well_years = well_year_counts[
    well_year_counts["Observation_Count"] > 1
]


print("\nTotal unique CSD_ID + Year combinations:")
print(len(well_year_counts))


print("\nDuplicate CSD_ID + Year combinations:")
print(len(duplicate_well_years))


print("\nTotal rows belonging to duplicated combinations:")

if len(duplicate_well_years) > 0:

    duplicate_keys = duplicate_well_years[
        ["CSD_ID", "YearMsr"]
    ]

    duplicate_rows = df.merge(
        duplicate_keys,
        on=["CSD_ID", "YearMsr"],
        how="inner"
    )

    print(len(duplicate_rows))

else:

    print(0)


duplicate_file = os.path.join(
    OUTPUT_DIR,
    "duplicate_well_year_combinations.csv"
)


duplicate_well_years.to_csv(
    duplicate_file,
    index=False
)


print("\nSaved:")
print(duplicate_file)


# =============================================================================
# 10. TEMPORAL COVERAGE
# =============================================================================

print("\n\n10. TEMPORAL COVERAGE")
print("-" * 78)


min_year = int(df["YearMsr"].min())
max_year = int(df["YearMsr"].max())


print("\nGroundwater period:")
print(min_year, "to", max_year)


year_counts = (
    df.groupby("YearMsr")
    .size()
    .reset_index(name="Observation_Count")
)


print("\nObservations by year:")

print(
    year_counts.to_string(index=False)
)


year_file = os.path.join(
    OUTPUT_DIR,
    "observations_by_year.csv"
)


year_counts.to_csv(
    year_file,
    index=False
)


print("\nSaved:")
print(year_file)


# =============================================================================
# 11. TEMPORAL TRAIN / TEST RECOMMENDATION
# =============================================================================

print("\n\n11. TEMPORAL TRAIN / TEST STRATEGY")
print("-" * 78)


print("\nRecommended temporal evaluation:")

print("""
Training period:
2000 - 2019

Validation period:
2020 - 2022

Testing period:
2023 - 2025
""")


train_mask = df["YearMsr"] <= 2019
validation_mask = (
    (df["YearMsr"] >= 2020)
    & (df["YearMsr"] <= 2022)
)
test_mask = df["YearMsr"] >= 2023


print("Training rows :", train_mask.sum())
print("Validation rows:", validation_mask.sum())
print("Testing rows  :", test_mask.sum())


# =============================================================================
# 12. SPATIAL INFORMATION CHECK
# =============================================================================

print("\n\n12. SPATIAL FEATURE CHECK")
print("-" * 78)


spatial_features = [
    "LatDD",
    "LongDD",
    "Surf_Elev"
]


for feature in spatial_features:

    print(
        f"\n{feature}:"
    )

    print(
        "  Min:",
        df[feature].min()
    )

    print(
        "  Max:",
        df[feature].max()
    )

    print(
        "  Missing:",
        df[feature].isna().sum()
    )


# =============================================================================
# 13. WEATHER FEATURE CHECK
# =============================================================================

print("\n\n13. WEATHER FEATURE CHECK")
print("-" * 78)


weather_features = [
    "Annual_Temperature_Mean",
    "Annual_Precipitation_Total",
    "Annual_Humidity_Mean",
    "Annual_WindSpeed_Mean",
    "Annual_SolarRadiation_Mean"
]


for feature in weather_features:

    print(
        f"\n{feature}:"
    )

    print(
        "  Missing:",
        df[feature].isna().sum()
    )

    if df[feature].notna().any():

        print(
            "  Min:",
            round(df[feature].min(), 4)
        )

        print(
            "  Max:",
            round(df[feature].max(), 4)
        )


# =============================================================================
# 14. TIFF LEAKAGE ANALYSIS
# =============================================================================

print("\n\n14. TIFF TARGET LEAKAGE ANALYSIS")
print("-" * 78)


print("""
TIFF_Value is being treated as a potentially leaked feature.

Reason:

The annual groundwater TIFF files were generated using
Ordinary Kriging from groundwater observations.

Therefore:

Groundwater observations
        ↓
Ordinary Kriging
        ↓
TIFF_Value
        ↓
ML Model
        ↓
WatLevel

This creates a possible target-information pathway.
""")


tiff_correlation = df[
    [TARGET, "TIFF_Value"]
].corr().iloc[0, 1]


print("\nWatLevel vs TIFF_Value correlation:")
print(round(tiff_correlation, 6))


if abs(tiff_correlation) >= 0.8:

    print("""
WARNING:
TIFF_Value has a very strong relationship with the target.

It will NOT be included in the primary ML feature set.
""")

elif abs(tiff_correlation) >= 0.5:

    print("""
CAUTION:
TIFF_Value has a strong relationship with the target.

It will NOT be included in the primary ML feature set.
""")

else:

    print("""
TIFF_Value does not show an extremely strong correlation,
but it is still considered potentially leaked because of
its generation methodology.
""")


# =============================================================================
# 15. SAFE VS UNSAFE FEATURE CLASSIFICATION
# =============================================================================

print("\n\n15. FEATURE SAFETY CLASSIFICATION")
print("-" * 78)


safe_features = [
    "LatDD",
    "LongDD",
    "Surf_Elev",
    "Annual_Temperature_Mean",
    "Annual_Precipitation_Total",
    "Annual_Humidity_Mean",
    "Annual_WindSpeed_Mean",
    "Annual_SolarRadiation_Mean"
]


leakage_features = [
    "TIFF_Value"
]


metadata_features = [
    "CSD_ID",
    "County",
    "DateMsr",
    "Season"
]


print("\nSAFE PRIMARY ML FEATURES:")

for feature in safe_features:
    print("  [SAFE]   ", feature)


print("\nPOTENTIAL LEAKAGE FEATURES:")

for feature in leakage_features:
    print("  [LEAKAGE]", feature)


print("\nMETADATA / IDENTIFIER FEATURES:")

for feature in metadata_features:
    print("  [META]   ", feature)


# =============================================================================
# 16. CORRELATION ANALYSIS
# =============================================================================

print("\n\n16. FEATURE-TARGET CORRELATION")
print("-" * 78)


numeric_features = safe_features + leakage_features


correlation_rows = []


for feature in numeric_features:

    correlation = df[
        [feature, TARGET]
    ].corr().iloc[0, 1]

    correlation_rows.append({
        "Feature": feature,
        "Target": TARGET,
        "Correlation": correlation,
        "Absolute_Correlation": abs(correlation)
    })


correlation_df = pd.DataFrame(
    correlation_rows
).sort_values(
    "Absolute_Correlation",
    ascending=False
)


print(
    correlation_df.to_string(index=False)
)


correlation_file = os.path.join(
    OUTPUT_DIR,
    "ml_feature_target_correlations.csv"
)


correlation_df.to_csv(
    correlation_file,
    index=False
)


print("\nSaved:")
print(correlation_file)


# =============================================================================
# 17. SAFE FEATURE CORRELATION MATRIX
# =============================================================================

print("\n\n17. SAFE FEATURE CORRELATION MATRIX")
print("-" * 78)


safe_correlation = df[
    safe_features + [TARGET]
].corr()


print(
    safe_correlation.round(4)
)


safe_correlation_file = os.path.join(
    OUTPUT_DIR,
    "safe_feature_correlation_matrix.csv"
)


safe_correlation.to_csv(
    safe_correlation_file
)


print("\nSaved:")
print(safe_correlation_file)


# =============================================================================
# 18. HIGHLY CORRELATED PREDICTOR CHECK
# =============================================================================

print("\n\n18. MULTICOLLINEARITY CHECK")
print("-" * 78)


feature_correlation = df[
    safe_features
].corr()


high_correlation_pairs = []


for i in range(len(safe_features)):

    for j in range(i + 1, len(safe_features)):

        feature_a = safe_features[i]
        feature_b = safe_features[j]

        value = feature_correlation.loc[
            feature_a,
            feature_b
        ]

        if abs(value) >= 0.80:

            high_correlation_pairs.append({
                "Feature_A": feature_a,
                "Feature_B": feature_b,
                "Correlation": value
            })


if high_correlation_pairs:

    high_corr_df = pd.DataFrame(
        high_correlation_pairs
    )

    print("\nHighly correlated predictor pairs:")

    print(
        high_corr_df.to_string(index=False)
    )

else:

    high_corr_df = pd.DataFrame(
        columns=[
            "Feature_A",
            "Feature_B",
            "Correlation"
        ]
    )

    print(
        "\nNo predictor pairs with absolute correlation >= 0.80."
    )


high_corr_file = os.path.join(
    OUTPUT_DIR,
    "high_predictor_correlation_pairs.csv"
)


high_corr_df.to_csv(
    high_corr_file,
    index=False
)


print("\nSaved:")
print(high_corr_file)


# =============================================================================
# 19. 2025 WEATHER CHECK
# =============================================================================

print("\n\n19. WEATHER COVERAGE CHECK")
print("-" * 78)


weather_missing_years = []


for year in sorted(df["YearMsr"].unique()):

    weather_rows = df[
        df["YearMsr"] == year
    ][weather_features]

    if weather_rows.isna().all().all():

        weather_missing_years.append(year)


print("\nYears without weather features:")

print(weather_missing_years)


if 2025 in weather_missing_years:

    print("""
WARNING:
2025 groundwater observations do not currently have
annual NASA POWER weather features.

Primary ML dataset will therefore exclude rows with
missing required weather values.

Obtaining 2025 weather data later is recommended.
""")


# =============================================================================
# 20. CREATE PRIMARY ML DATASET
# =============================================================================

print("\n\n20. CREATING LEAKAGE-SAFE ML DATASET")
print("-" * 78)


primary_ml_columns = [
    "CSD_ID",
    "County",
    "LatDD",
    "LongDD",
    "YearMsr",
    "DateMsr",
    "WatLevel",
    "Surf_Elev",

    "Annual_Temperature_Mean",
    "Annual_Precipitation_Total",
    "Annual_Humidity_Mean",
    "Annual_WindSpeed_Mean",
    "Annual_SolarRadiation_Mean"
]


ml_df = df[
    primary_ml_columns
].copy()


print("\nPrimary ML columns:")

for column in ml_df.columns:
    print(" -", column)


before_cleaning = len(ml_df)


ml_df = ml_df.dropna(
    subset=safe_features + [TARGET]
).copy()


after_cleaning = len(ml_df)


print("\nRows before required-feature cleaning:")
print(before_cleaning)


print("\nRows after required-feature cleaning:")
print(after_cleaning)


print("\nRows removed:")
print(before_cleaning - after_cleaning)


# =============================================================================
# 21. DUPLICATE CHECK ON ML DATASET
# =============================================================================

print("\n\n21. DUPLICATE CHECK ON ML DATASET")
print("-" * 78)


duplicate_complete = ml_df.duplicated().sum()


duplicate_well_year = ml_df.duplicated(
    subset=["CSD_ID", "YearMsr"]
).sum()


print("\nDuplicate complete rows:")
print(duplicate_complete)


print("\nDuplicate CSD_ID + Year rows:")
print(duplicate_well_year)


print("""
IMPORTANT:
Duplicate CSD_ID + Year rows are NOT automatically removed.

They may represent multiple groundwater observations
from the same well during the same year.

They must be handled carefully during model evaluation.
""")


# =============================================================================
# 22. CREATE GROUP IDENTIFIER
# =============================================================================

print("\n\n22. CREATING GROUP IDENTIFIER")
print("-" * 78)


ml_df["Well_Group"] = (
    ml_df["CSD_ID"]
    .astype(str)
)


print("\nWell_Group created.")

print(
    "Unique groups:",
    ml_df["Well_Group"].nunique()
)


# =============================================================================
# 23. CREATE TEMPORAL SPLIT LABEL
# =============================================================================

print("\n\n23. CREATING TEMPORAL SPLIT LABEL")
print("-" * 78)


def assign_temporal_split(year):

    if year <= 2019:
        return "train"

    elif year <= 2022:
        return "validation"

    else:
        return "test"


ml_df["Temporal_Split"] = (
    ml_df["YearMsr"]
    .apply(assign_temporal_split)
)


print("\nTemporal split distribution:")

print(
    ml_df["Temporal_Split"]
    .value_counts()
    .sort_index()
)


# =============================================================================
# 24. FINAL ML FEATURE LIST
# =============================================================================

print("\n\n24. FINAL PRIMARY ML FEATURE LIST")
print("-" * 78)


print("""
Primary ML features:

1. LatDD
2. LongDD
3. Surf_Elev
4. Annual_Temperature_Mean
5. Annual_Precipitation_Total
6. Annual_Humidity_Mean
7. Annual_WindSpeed_Mean
8. Annual_SolarRadiation_Mean

Target:

WatLevel

Excluded from primary ML:

TIFF_Value
""")


# =============================================================================
# 25. SAVE LEAKAGE-SAFE DATASET
# =============================================================================

print("\n\n25. SAVING LEAKAGE-SAFE ML DATASET")
print("-" * 78)


safe_ml_file = os.path.join(
    OUTPUT_DIR,
    "groundwater_ml_leakage_safe.csv"
)


ml_df.to_csv(
    safe_ml_file,
    index=False
)


print("\nSaved:")
print(safe_ml_file)


# =============================================================================
# 26. CREATE FEATURE CONFIGURATION
# =============================================================================

print("\n\n26. SAVING FEATURE CONFIGURATION")
print("-" * 78)


feature_config = pd.DataFrame({

    "Feature": (
        safe_features
        + leakage_features
        + metadata_features
    ),

    "Role": (
        ["Primary_ML_Feature"] * len(safe_features)
        + ["Potential_Target_Leakage"] * len(leakage_features)
        + ["Metadata"] * len(metadata_features)
    )
})


feature_config_file = os.path.join(
    OUTPUT_DIR,
    "feature_classification.csv"
)


feature_config.to_csv(
    feature_config_file,
    index=False
)


print("\nSaved:")
print(feature_config_file)


# =============================================================================
# 27. FINAL DATASET SUMMARY
# =============================================================================

print("\n\n27. FINAL DATASET SUMMARY")
print("-" * 78)


print("\nOriginal integrated rows:")
print(len(df))


print("\nLeakage-safe ML rows:")
print(len(ml_df))


print("\nUnique wells:")
print(ml_df["CSD_ID"].nunique())


print("\nPrimary ML features:")
print(len(safe_features))


print("\nTarget:")
print(TARGET)


print("\nTIFF_Value included in primary ML?")
print("NO")


print("\nTemporal split:")

print(
    ml_df["Temporal_Split"]
    .value_counts()
    .sort_index()
)


# =============================================================================
# 28. RECOMMENDED EXPERIMENTS
# =============================================================================

print("\n\n28. RECOMMENDED ML EXPERIMENTS")
print("-" * 78)


print("""
EXPERIMENT A - SPATIAL BASELINE
--------------------------------

Features:
    LatDD
    LongDD
    Surf_Elev

Target:
    WatLevel


EXPERIMENT B - ENVIRONMENTAL BASELINE
-------------------------------------

Features:
    LatDD
    LongDD
    Surf_Elev
    Annual_Temperature_Mean
    Annual_Precipitation_Total
    Annual_Humidity_Mean
    Annual_WindSpeed_Mean
    Annual_SolarRadiation_Mean

Target:
    WatLevel


EXPERIMENT C - TIFF LEAKAGE ABLATION
------------------------------------

Features:
    Experiment B features
    +
    TIFF_Value

Purpose:
    Demonstrate the effect of target-derived spatial information.

IMPORTANT:
    This should NOT be presented as the independent primary model.


EXPERIMENT D - PROPOSED AMDFE
-----------------------------

Independent environmental sources
        ↓
Data Quality Assessment
        ↓
Reliability Scores
        ↓
Dynamic Weight Assignment
        ↓
Adaptive Multimodal Data Fusion
        ↓
Groundwater Prediction
""")


# =============================================================================
# 29. FINAL RECOMMENDATION
# =============================================================================

print("\n\n29. FINAL RECOMMENDATION")
print("-" * 78)


print("""
PRIMARY MODELING DATASET:
    groundwater_ml_leakage_safe.csv


DO NOT USE TIFF_Value IN THE PRIMARY MODEL.

Reason:
    TIFF_Value was generated through Ordinary Kriging from
    groundwater observations and therefore contains information
    derived from the prediction target.


RECOMMENDED EVALUATION:

1. Random split
   -> Basic reference only

2. Well-group split
   -> Tests generalization to unseen wells

3. Temporal split
   -> Tests future prediction capability

4. Spatial holdout
   -> Tests generalization to unseen geographic regions


NEXT STEP:

    STEP 07 - ML BASELINE MODELS

Use:
    groundwater_ml_leakage_safe.csv

Start with:
    Experiment A - Spatial Baseline
    Experiment B - Environmental Baseline

Then separately perform:
    Experiment C - TIFF leakage ablation
""")


# =============================================================================
# 30. COMPLETION
# =============================================================================

print("\n\n" + "=" * 78)
print("AquaSense AI - ML DATASET VALIDATION COMPLETED")
print("=" * 78)


print("\nOriginal observations:")
print(len(df))


print("\nLeakage-safe ML observations:")
print(len(ml_df))


print("\nPrimary ML features:")
print(len(safe_features))


print("\nTarget:")
print(TARGET)


print("\nTIFF leakage feature:")
print("TIFF_Value -> EXCLUDED")


print("\nOutput directory:")
print(OUTPUT_DIR)


print("\nLeakage-safe dataset:")
print(safe_ml_file)


print("\nNext stage:")
print("07_ml_baseline_models.py")


print("\n" + "=" * 78)