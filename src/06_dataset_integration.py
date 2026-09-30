# ==============================================================
# AquaSense AI
# 06_dataset_integration.py
#
# Purpose:
# Integrate:
#   1. Groundwater well observations
#   2. Corrected annual groundwater TIFF values
#   3. NASA POWER annual weather features
#   4. Elevation / DEM data when available
#
# This script DOES NOT train an ML model.
# It prepares and validates the final ML-ready dataset.
# ==============================================================


import os
import re
import glob
import warnings

import numpy as np
import pandas as pd
import rasterio

from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


# ==============================================================
# 1. PROJECT PATHS
# ==============================================================

print("=" * 78)
print("AquaSense AI - DATASET INTEGRATION")
print("Groundwater + Corrected TIFF + Weather + Elevation")
print("=" * 78)


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


GROUNDWATER_DIRECTORY = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "groundwater"
)


WEATHER_DIRECTORY = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "weather"
)


ELEVATION_DIRECTORY = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "elevation"
)


TIFF_ANALYSIS_DIRECTORY = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "tiff_analysis"
)


OUTPUT_DIRECTORY = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "integrated"
)


os.makedirs(
    OUTPUT_DIRECTORY,
    exist_ok=True
)


print("\n1. PROJECT PATHS")
print("-" * 78)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nGroundwater directory:")
print(GROUNDWATER_DIRECTORY)

print("\nWeather directory:")
print(WEATHER_DIRECTORY)

print("\nElevation directory:")
print(ELEVATION_DIRECTORY)

print("\nTIFF analysis directory:")
print(TIFF_ANALYSIS_DIRECTORY)

print("\nOutput directory:")
print(OUTPUT_DIRECTORY)


# ==============================================================
# 2. FIND GROUNDWATER CSV
# ==============================================================

print("\n\n2. LOCATING GROUNDWATER DATA")
print("-" * 78)


groundwater_file = os.path.join(
    GROUNDWATER_DIRECTORY,
    "Groundwater_Clean.csv"
)


if not os.path.exists(groundwater_file):

    raise FileNotFoundError(
        "Groundwater_Clean.csv was not found:\n"
        + groundwater_file
    )


print("\nGroundwater CSV found:")
print(groundwater_file)


# ==============================================================
# 3. LOAD GROUNDWATER DATA
# ==============================================================

print("\n\n3. LOADING GROUNDWATER DATA")
print("-" * 78)


groundwater_df = pd.read_csv(
    groundwater_file
)


print("\nRows loaded :", len(groundwater_df))
print("Columns     :", len(groundwater_df.columns))


print("\nColumns:")

for column in groundwater_df.columns:
    print(" -", column)


# ==============================================================
# 4. CHECK REQUIRED GROUNDWATER COLUMNS
# ==============================================================

required_groundwater_columns = [
    "CSD_ID",
    "County",
    "LatDD",
    "LongDD",
    "Surf_Elev",
    "YearMsr",
    "DateMsr",
    "Season",
    "WatLevel"
]


missing_columns = [
    column
    for column in required_groundwater_columns
    if column not in groundwater_df.columns
]


if missing_columns:

    raise ValueError(
        "Missing groundwater columns:\n"
        + str(missing_columns)
    )


print("\nAll required groundwater columns are present.")


# ==============================================================
# 5. CLEAN BASIC TYPES
# ==============================================================

print("\n\n4. PREPARING GROUNDWATER DATA")
print("-" * 78)


groundwater_df["YearMsr"] = pd.to_numeric(
    groundwater_df["YearMsr"],
    errors="coerce"
)


groundwater_df["LatDD"] = pd.to_numeric(
    groundwater_df["LatDD"],
    errors="coerce"
)


groundwater_df["LongDD"] = pd.to_numeric(
    groundwater_df["LongDD"],
    errors="coerce"
)


groundwater_df["Surf_Elev"] = pd.to_numeric(
    groundwater_df["Surf_Elev"],
    errors="coerce"
)


groundwater_df["WatLevel"] = pd.to_numeric(
    groundwater_df["WatLevel"],
    errors="coerce"
)


groundwater_df = groundwater_df.dropna(
    subset=[
        "YearMsr",
        "LatDD",
        "LongDD",
        "WatLevel"
    ]
).copy()


groundwater_df["YearMsr"] = (
    groundwater_df["YearMsr"]
    .astype(int)
)


print("\nGroundwater observations after basic cleaning:")
print(len(groundwater_df))


# ==============================================================
# 6. FIND CORRECTED TIFF VALIDATION FILE
# ==============================================================

print("\n\n5. LOCATING CORRECTED TIFF VALIDATION DATA")
print("-" * 78)


corrected_validation_file = os.path.join(
    TIFF_ANALYSIS_DIRECTORY,
    "tiff_well_validation_corrected.csv"
)


if not os.path.exists(corrected_validation_file):

    raise FileNotFoundError(
        "\nCorrected TIFF validation file was not found:\n"
        + corrected_validation_file
        + "\n\nRun 05_tiff_spatial_analysis.py first."
    )


print("\nCorrected TIFF validation file found:")
print(corrected_validation_file)


# ==============================================================
# 7. LOAD CORRECTED TIFF VALUES
# ==============================================================

print("\n\n6. LOADING CORRECTED TIFF VALUES")
print("-" * 78)


tiff_df = pd.read_csv(
    corrected_validation_file
)


print("\nRows loaded :", len(tiff_df))

print("\nColumns:")

for column in tiff_df.columns:
    print(" -", column)


# ==============================================================
# 8. CHECK TIFF REQUIRED COLUMNS
# ==============================================================

required_tiff_columns = [
    "CSD_ID",
    "YearMsr",
    "TIFF_Value"
]


missing_tiff_columns = [
    column
    for column in required_tiff_columns
    if column not in tiff_df.columns
]


if missing_tiff_columns:

    raise ValueError(
        "Missing corrected TIFF columns:\n"
        + str(missing_tiff_columns)
    )


# ==============================================================
# 9. CLEAN TIFF DATA
# ==============================================================

tiff_df["YearMsr"] = pd.to_numeric(
    tiff_df["YearMsr"],
    errors="coerce"
)


tiff_df["TIFF_Value"] = pd.to_numeric(
    tiff_df["TIFF_Value"],
    errors="coerce"
)


tiff_df = tiff_df.dropna(
    subset=[
        "CSD_ID",
        "YearMsr",
        "TIFF_Value"
    ]
).copy()


tiff_df["YearMsr"] = (
    tiff_df["YearMsr"]
    .astype(int)
)


print("\nValid TIFF observations:")
print(len(tiff_df))


# ==============================================================
# 10. CHECK TIFF DUPLICATES
# ==============================================================

print("\n\n7. CHECKING TIFF OBSERVATION DUPLICATES")
print("-" * 78)


tiff_duplicates = tiff_df.duplicated(
    subset=[
        "CSD_ID",
        "YearMsr"
    ]
).sum()


print("\nDuplicate CSD_ID + Year combinations:")
print(tiff_duplicates)


if tiff_duplicates > 0:

    print(
        "\nWARNING: Duplicate TIFF observations detected."
    )

    tiff_df = (
        tiff_df
        .sort_values(
            ["CSD_ID", "YearMsr"]
        )
        .drop_duplicates(
            subset=[
                "CSD_ID",
                "YearMsr"
            ],
            keep="first"
        )
    )


# ==============================================================
# 11. MERGE GROUNDWATER + TIFF
# ==============================================================

print("\n\n8. MERGING GROUNDWATER WITH CORRECTED TIFF")
print("-" * 78)


integrated_df = groundwater_df.merge(
    tiff_df[
        [
            "CSD_ID",
            "YearMsr",
            "TIFF_Value"
        ]
    ],
    on=[
        "CSD_ID",
        "YearMsr"
    ],
    how="left"
)


print("\nRows after merge:")
print(len(integrated_df))


missing_tiff = (
    integrated_df["TIFF_Value"]
    .isna()
    .sum()
)


print("\nMissing TIFF values:")
print(missing_tiff)


# ==============================================================
# 12. TIFF VALIDATION AFTER MERGE
# ==============================================================

print("\n\n9. TIFF VALIDATION AFTER MERGE")
print("-" * 78)


tiff_valid = integrated_df.dropna(
    subset=[
        "WatLevel",
        "TIFF_Value"
    ]
)


if len(tiff_valid) > 1:

    correlation = (
        tiff_valid["WatLevel"]
        .corr(tiff_valid["TIFF_Value"])
    )

    r2 = (
        1
        -
        (
            np.sum(
                (
                    tiff_valid["WatLevel"]
                    -
                    tiff_valid["TIFF_Value"]
                ) ** 2
            )
            /
            np.sum(
                (
                    tiff_valid["WatLevel"]
                    -
                    tiff_valid["WatLevel"].mean()
                ) ** 2
            )
        )
    )

    mae = mean_absolute_error(
        tiff_valid["WatLevel"],
        tiff_valid["TIFF_Value"]
    )

    rmse = np.sqrt(
        mean_squared_error(
            tiff_valid["WatLevel"],
            tiff_valid["TIFF_Value"]
        )
    )

    bias = (
        tiff_valid["TIFF_Value"]
        -
        tiff_valid["WatLevel"]
    ).mean()

    print("\nCorrelation :", round(correlation, 6))
    print("R²          :", round(r2, 6))
    print("MAE         :", round(mae, 6))
    print("RMSE        :", round(rmse, 6))
    print("Mean Bias   :", round(bias, 6))


# ==============================================================
# 13. FIND NASA POWER WEATHER FILE
# ==============================================================

print("\n\n10. LOCATING NASA POWER WEATHER DATA")
print("-" * 78)


weather_candidates = [
    os.path.join(
        WEATHER_DIRECTORY,
        "NASA_POWER_2000_2025.csv"
    ),
    os.path.join(
        WEATHER_DIRECTORY,
        "NASA_POWER_2000_2024.csv"
    )
]


weather_file = None


for candidate in weather_candidates:

    if os.path.exists(candidate):

        weather_file = candidate
        break


if weather_file is None:

    raise FileNotFoundError(
        "\nNASA POWER weather CSV was not found in:\n"
        + WEATHER_DIRECTORY
    )


print("\nWeather file found:")
print(weather_file)


# ==============================================================
# 14. LOAD NASA POWER DATA
# ==============================================================

print("\n\n11. LOADING NASA POWER DATA")
print("-" * 78)


# NASA POWER CSV contains a metadata header.
# Find the actual CSV header automatically.

with open(
    weather_file,
    "r",
    encoding="utf-8"
) as file:

    lines = file.readlines()


header_line_index = None


for i, line in enumerate(lines):

    if line.strip().startswith(
        "YEAR,DOY"
    ):

        header_line_index = i
        break


if header_line_index is None:

    raise ValueError(
        "Could not locate NASA POWER CSV header."
    )


weather_df = pd.read_csv(
    weather_file,
    skiprows=header_line_index
)


print("\nRows loaded:")
print(len(weather_df))


print("\nColumns:")

for column in weather_df.columns:
    print(" -", column)


# ==============================================================
# 15. CHECK WEATHER COLUMNS
# ==============================================================

required_weather_columns = [
    "YEAR",
    "DOY",
    "T2M",
    "PRECTOTCORR",
    "RH2M",
    "WS2M",
    "ALLSKY_SFC_SW_DWN"
]


missing_weather_columns = [
    column
    for column in required_weather_columns
    if column not in weather_df.columns
]


if missing_weather_columns:

    raise ValueError(
        "Missing NASA POWER columns:\n"
        + str(missing_weather_columns)
    )


print("\nAll required NASA POWER columns are present.")


# ==============================================================
# 16. CLEAN WEATHER DATA
# ==============================================================

print("\n\n12. CLEANING WEATHER DATA")
print("-" * 78)


numeric_weather_columns = [
    "YEAR",
    "DOY",
    "T2M",
    "PRECTOTCORR",
    "RH2M",
    "WS2M",
    "ALLSKY_SFC_SW_DWN"
]


for column in numeric_weather_columns:

    weather_df[column] = pd.to_numeric(
        weather_df[column],
        errors="coerce"
    )


# NASA POWER uses -999 for missing values.

weather_df = weather_df.replace(
    -999,
    np.nan
)


print("\nMissing values before aggregation:")

print(
    weather_df[
        numeric_weather_columns
    ]
    .isna()
    .sum()
)


# ==============================================================
# 17. ANNUAL WEATHER AGGREGATION
# ==============================================================

print("\n\n13. CREATING ANNUAL WEATHER FEATURES")
print("-" * 78)


annual_weather = (
    weather_df
    .groupby("YEAR")
    .agg(
        Annual_Temperature_Mean=(
            "T2M",
            "mean"
        ),

        Annual_Precipitation_Total=(
            "PRECTOTCORR",
            "sum"
        ),

        Annual_Humidity_Mean=(
            "RH2M",
            "mean"
        ),

        Annual_WindSpeed_Mean=(
            "WS2M",
            "mean"
        ),

        Annual_SolarRadiation_Mean=(
            "ALLSKY_SFC_SW_DWN",
            "mean"
        )
    )
    .reset_index()
)


annual_weather = annual_weather.rename(
    columns={
        "YEAR": "YearMsr"
    }
)


print("\nAnnual weather rows:")
print(len(annual_weather))


print("\nAnnual weather years:")

print(
    annual_weather["YearMsr"]
    .min(),
    "to",
    annual_weather["YearMsr"].max()
)


print("\nAnnual weather preview:")

print(
    annual_weather.head()
    .to_string(index=False)
)


# ==============================================================
# 18. MERGE WEATHER
# ==============================================================

print("\n\n14. MERGING ANNUAL WEATHER FEATURES")
print("-" * 78)


integrated_df = integrated_df.merge(
    annual_weather,
    on="YearMsr",
    how="left"
)


weather_features = [
    "Annual_Temperature_Mean",
    "Annual_Precipitation_Total",
    "Annual_Humidity_Mean",
    "Annual_WindSpeed_Mean",
    "Annual_SolarRadiation_Mean"
]


print("\nWeather missing-value count:")

for column in weather_features:

    print(
        f"{column}: "
        f"{integrated_df[column].isna().sum()}"
    )


# ==============================================================
# 19. FIND ELEVATION / DEM FILES
# ==============================================================

print("\n\n15. SEARCHING FOR ELEVATION / DEM DATA")
print("-" * 78)


elevation_files = []


if os.path.exists(
    ELEVATION_DIRECTORY
):

    elevation_patterns = [
        "*.tif",
        "*.tiff"
    ]

    for pattern in elevation_patterns:

        elevation_files.extend(
            glob.glob(
                os.path.join(
                    ELEVATION_DIRECTORY,
                    pattern
                )
            )
        )


elevation_files = sorted(
    list(
        set(
            elevation_files
        )
    )
)


print(
    "\nElevation raster files found:",
    len(elevation_files)
)


for file in elevation_files:

    print(
        " -",
        os.path.basename(file)
    )


# ==============================================================
# 20. ELEVATION EXTRACTION
# ==============================================================

if len(elevation_files) == 0:

    print(
        "\nNo elevation TIFF was found."
    )

    print(
        "The existing Surf_Elev column will be retained."
    )

    integrated_df[
        "DEM_Elevation"
    ] = np.nan


else:

    print(
        "\nElevation raster will be sampled "
        "at groundwater well locations."
    )


    elevation_file = elevation_files[0]


    with rasterio.open(
        elevation_file
    ) as src:

        print("\nElevation CRS:")
        print(src.crs)

        print("\nElevation dimensions:")
        print(
            src.width,
            "x",
            src.height
        )

        coordinates = list(
            zip(
                integrated_df["LongDD"],
                integrated_df["LatDD"]
            )
        )


        elevation_values = []


        for value in src.sample(
            coordinates
        ):

            pixel_value = value[0]


            if src.nodata is not None:

                if np.isclose(
                    pixel_value,
                    src.nodata
                ):

                    pixel_value = np.nan


            elevation_values.append(
                pixel_value
            )


    integrated_df[
        "DEM_Elevation"
    ] = elevation_values


    print(
        "\nDEM values extracted:",
        integrated_df[
            "DEM_Elevation"
        ].notna().sum()
    )


    print(
        "\nMissing DEM values:",
        integrated_df[
            "DEM_Elevation"
        ].isna().sum()
    )


# ==============================================================
# 21. TEMPORAL ALIGNMENT CHECK
# ==============================================================

print("\n\n16. TEMPORAL ALIGNMENT CHECK")
print("-" * 78)


print("\nGroundwater years:")
print(
    groundwater_df["YearMsr"].min(),
    "to",
    groundwater_df["YearMsr"].max()
)


print("\nTIFF years:")
print(
    tiff_df["YearMsr"].min(),
    "to",
    tiff_df["YearMsr"].max()
)


print("\nWeather years:")
print(
    annual_weather["YearMsr"].min(),
    "to",
    annual_weather["YearMsr"].max()
)


groundwater_years = set(
    groundwater_df["YearMsr"]
)


tiff_years = set(
    tiff_df["YearMsr"]
)


weather_years = set(
    annual_weather["YearMsr"]
)


print("\nYears missing from TIFF:")

print(
    sorted(
        groundwater_years
        -
        tiff_years
    )
)


print("\nYears missing from weather:")

print(
    sorted(
        groundwater_years
        -
        weather_years
    )
)


# ==============================================================
# 22. SPATIAL ALIGNMENT CHECK
# ==============================================================

print("\n\n17. SPATIAL ALIGNMENT CHECK")
print("-" * 78)


print("\nLatitude range:")

print(
    round(
        integrated_df["LatDD"].min(),
        6
    ),
    "to",
    round(
        integrated_df["LatDD"].max(),
        6
    )
)


print("\nLongitude range:")

print(
    round(
        integrated_df["LongDD"].min(),
        6
    ),
    "to",
    round(
        integrated_df["LongDD"].max(),
        6
    )
)


print("\nCoordinate missing values:")

print(
    "Latitude:",
    integrated_df["LatDD"].isna().sum()
)


print(
    "Longitude:",
    integrated_df["LongDD"].isna().sum()
)


# ==============================================================
# 23. FINAL FEATURE LIST
# ==============================================================

print("\n\n18. FINAL FEATURE DATASET")
print("-" * 78)


feature_columns = [
    "CSD_ID",
    "County",
    "LatDD",
    "LongDD",
    "YearMsr",
    "DateMsr",
    "Season",

    # Target
    "WatLevel",

    # Ground surface elevation
    "Surf_Elev",

    # Corrected spatial groundwater feature
    "TIFF_Value",

    # Weather
    "Annual_Temperature_Mean",
    "Annual_Precipitation_Total",
    "Annual_Humidity_Mean",
    "Annual_WindSpeed_Mean",
    "Annual_SolarRadiation_Mean",

    # DEM
    "DEM_Elevation"
]


# Keep only columns that exist.

feature_columns = [
    column
    for column in feature_columns
    if column in integrated_df.columns
]


final_df = integrated_df[
    feature_columns
].copy()


print("\nFinal columns:")

for column in final_df.columns:

    print(
        " -",
        column
    )


print(
    "\nFinal dataset shape:"
)

print(
    final_df.shape
)


# ==============================================================
# 24. MISSING VALUE REPORT
# ==============================================================

print("\n\n19. FINAL MISSING VALUE REPORT")
print("-" * 78)


missing_report = pd.DataFrame({
    "Column":
        final_df.columns,

    "Missing_Count":
        [
            final_df[column].isna().sum()
            for column in final_df.columns
        ]
})


missing_report[
    "Missing_Percent"
] = (
    missing_report["Missing_Count"]
    /
    len(final_df)
    *
    100
)


print(
    missing_report.to_string(
        index=False
    )
)


# ==============================================================
# 25. CHECK FOR DUPLICATES
# ==============================================================

print("\n\n20. FINAL DUPLICATE CHECK")
print("-" * 78)


duplicate_count = final_df.duplicated(
    subset=[
        "CSD_ID",
        "YearMsr"
    ]
).sum()


print(
    "\nDuplicate CSD_ID + Year:",
    duplicate_count
)


# ==============================================================
# 26. FEATURE STATISTICS
# ==============================================================

print("\n\n21. FEATURE STATISTICS")
print("-" * 78)


numeric_feature_columns = [
    column
    for column in [
        "WatLevel",
        "Surf_Elev",
        "TIFF_Value",
        "Annual_Temperature_Mean",
        "Annual_Precipitation_Total",
        "Annual_Humidity_Mean",
        "Annual_WindSpeed_Mean",
        "Annual_SolarRadiation_Mean",
        "DEM_Elevation"
    ]
    if column in final_df.columns
]


print(
    final_df[
        numeric_feature_columns
    ]
    .describe()
    .round(4)
    .to_string()
)


# ==============================================================
# 27. CORRELATION MATRIX
# ==============================================================

print("\n\n22. FEATURE CORRELATION MATRIX")
print("-" * 78)


correlation_matrix = (
    final_df[
        numeric_feature_columns
    ]
    .corr()
)


print(
    correlation_matrix
    .round(4)
    .to_string()
)


# ==============================================================
# 28. CHECK TARGET LEAKAGE
# ==============================================================

print("\n\n23. TARGET LEAKAGE CHECK")
print("-" * 78)


print(
    "\nTarget variable:"
)

print(
    "WatLevel"
)


print(
    "\nPotentially dangerous features:"
)


dangerous_features = [
    "WatLevel",
    "DepthToWater",
    "Groundwater_Level",
    "Target",
    "Predicted_WatLevel"
]


for column in dangerous_features:

    if column in final_df.columns:

        print(
            "WARNING:",
            column,
            "is present."
        )


print(
    "\nTIFF_Value is retained as a spatial predictor."
)


# ==============================================================
# 29. CREATE ML DATASET
# ==============================================================

print("\n\n24. CREATING ML-READY DATASET")
print("-" * 78)


ml_columns = [
    "CSD_ID",
    "YearMsr",
    "LatDD",
    "LongDD",
    "Surf_Elev",
    "TIFF_Value",
    "Annual_Temperature_Mean",
    "Annual_Precipitation_Total",
    "Annual_Humidity_Mean",
    "Annual_WindSpeed_Mean",
    "Annual_SolarRadiation_Mean",
    "DEM_Elevation",
    "WatLevel"
]


ml_columns = [
    column
    for column in ml_columns
    if column in final_df.columns
]


ml_df = final_df[
    ml_columns
].copy()


print(
    "\nML dataset shape before removing missing rows:"
)

print(
    ml_df.shape
)


# Only remove rows missing essential ML variables.

essential_ml_columns = [
    "WatLevel",
    "TIFF_Value",
    "Annual_Temperature_Mean",
    "Annual_Precipitation_Total",
    "Annual_Humidity_Mean",
    "Annual_WindSpeed_Mean",
    "Annual_SolarRadiation_Mean"
]


essential_ml_columns = [
    column
    for column in essential_ml_columns
    if column in ml_df.columns
]


ml_complete_df = ml_df.dropna(
    subset=essential_ml_columns
).copy()


print(
    "\nML dataset shape after required-feature cleaning:"
)

print(
    ml_complete_df.shape
)


print(
    "\nRows removed:"
)

print(
    len(ml_df)
    -
    len(ml_complete_df)
)


# ==============================================================
# 30. SAVE INTEGRATED DATASET
# ==============================================================

print("\n\n25. SAVING INTEGRATED DATASET")
print("-" * 78)


integrated_file = os.path.join(
    OUTPUT_DIRECTORY,
    "groundwater_integrated_dataset.csv"
)


final_df.to_csv(
    integrated_file,
    index=False
)


print(
    "\nSaved:"
)

print(
    integrated_file
)


# ==============================================================
# 31. SAVE ML DATASET
# ==============================================================

ml_file = os.path.join(
    OUTPUT_DIRECTORY,
    "groundwater_ml_ready.csv"
)


ml_complete_df.to_csv(
    ml_file,
    index=False
)


print(
    "\nSaved:"
)

print(
    ml_file
)


# ==============================================================
# 32. SAVE WEATHER FEATURES
# ==============================================================

weather_output_file = os.path.join(
    OUTPUT_DIRECTORY,
    "annual_weather_features.csv"
)


annual_weather.to_csv(
    weather_output_file,
    index=False
)


print(
    "\nSaved:"
)

print(
    weather_output_file
)


# ==============================================================
# 33. SAVE MISSING VALUE REPORT
# ==============================================================

missing_report_file = os.path.join(
    OUTPUT_DIRECTORY,
    "integration_missing_value_report.csv"
)


missing_report.to_csv(
    missing_report_file,
    index=False
)


print(
    "\nSaved:"
)

print(
    missing_report_file
)


# ==============================================================
# 34. SAVE CORRELATION MATRIX
# ==============================================================

correlation_file = os.path.join(
    OUTPUT_DIRECTORY,
    "integration_correlation_matrix.csv"
)


correlation_matrix.to_csv(
    correlation_file
)


print(
    "\nSaved:"
)

print(
    correlation_file
)


# ==============================================================
# 35. FINAL SUMMARY
# ==============================================================

print("\n\n")
print("=" * 78)
print("AquaSense AI - DATASET INTEGRATION COMPLETED")
print("=" * 78)


print(
    "\nGroundwater observations:",
    len(groundwater_df)
)


print(
    "Integrated dataset rows:",
    len(final_df)
)


print(
    "ML-ready rows:",
    len(ml_complete_df)
)


print(
    "Feature columns:",
    len(
        ml_complete_df.columns
    )
)


print(
    "\nIntegrated dataset:"
)

print(
    integrated_file
)


print(
    "\nML-ready dataset:"
)

print(
    ml_file
)


print(
    "\nNext stage:"
)

print(
    "07_ml_baseline_models.py"
)


print("\n")
print("=" * 78)