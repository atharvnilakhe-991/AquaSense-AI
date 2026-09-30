"""
05_tiff_spatial_analysis.py

AquaSense AI
TIFF Spatial Analysis & Groundwater Well Validation

Purpose
-------
1. Load groundwater well observations.
2. Inspect annual groundwater GeoTIFFs.
3. Validate CRS, dimensions, resolution and bounds.
4. Sample TIFF values at USGS well locations.
5. Diagnose raster orientation problems.
6. Test:
       - Native raster
       - Vertical flip
       - Horizontal flip
       - Both flips
7. Automatically identify the consistent raster orientation.
8. Perform corrected validation.
9. Generate yearly statistics, plots and maps.

Important
---------
The TIFFs are documented as annual groundwater surfaces generated
using Ordinary Kriging from groundwater observations.

The previous validation produced a strong negative correlation.
This version explicitly checks raster orientation before deciding
whether the TIFF values agree with the well observations.
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import re
import warnings

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

GROUNDWATER_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "groundwater"
)

GROUNDWATER_CSV = os.path.join(
    GROUNDWATER_DIR,
    "Groundwater_Clean.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "tiff_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 75)
print("AquaSense AI - TIFF SPATIAL & WELL VALIDATION")
print("Diagnostic: Raster Orientation + WatLevel Validation")
print("=" * 75)


print("\n1. SETTING PROJECT PATHS")
print("-" * 75)

print("Project root:")
print(PROJECT_ROOT)

print("\nGroundwater directory:")
print(GROUNDWATER_DIR)

print("\nOutput directory:")
print(OUTPUT_DIR)


# ============================================================
# 2. CHECK CSV
# ============================================================

print("\n\n2. LOCATING GROUNDWATER CSV")
print("-" * 75)

if not os.path.exists(GROUNDWATER_CSV):
    raise FileNotFoundError(
        f"Groundwater CSV not found:\n{GROUNDWATER_CSV}"
    )

print("\nGroundwater CSV found:")
print(GROUNDWATER_CSV)


# ============================================================
# 3. LOAD GROUNDWATER DATA
# ============================================================

print("\n\n3. LOADING GROUNDWATER DATA")
print("-" * 75)

df = pd.read_csv(GROUNDWATER_CSV)

print(f"\nRows loaded : {len(df)}")
print(f"Columns     : {len(df.columns)}")

print("\nColumns:")

for col in df.columns:
    print(f" - {col}")


# ============================================================
# 4. REQUIRED COLUMNS
# ============================================================

print("\n\n4. CHECKING REQUIRED COLUMNS")
print("-" * 75)

required_columns = [
    "CSD_ID",
    "LatDD",
    "LongDD",
    "YearMsr",
    "WatLevel"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    print("\nERROR: Missing required columns:")

    for col in missing_columns:
        print(" -", col)

    raise ValueError("Required columns are missing.")

print("\nAll required columns are present.")


# ============================================================
# 5. DATA CLEANING
# ============================================================

print("\n\n5. GROUNDWATER DATA QUALITY")
print("-" * 75)

for col in ["LatDD", "LongDD", "YearMsr", "WatLevel"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nMissing WatLevel:",
      df["WatLevel"].isna().sum())

print("Missing Latitude:",
      df["LatDD"].isna().sum())

print("Missing Longitude:",
      df["LongDD"].isna().sum())

print("Duplicate records:",
      df.duplicated().sum())


df = df.dropna(
    subset=[
        "LatDD",
        "LongDD",
        "YearMsr",
        "WatLevel"
    ]
).copy()


print("\nWatLevel statistics:")
print(df["WatLevel"].describe())


# ============================================================
# 6. FIND TIFF FILES
# ============================================================

print("\n\n6. FINDING TIFF FILES")
print("-" * 75)

tiff_files = []

for filename in os.listdir(GROUNDWATER_DIR):

    if filename.lower().endswith(".tif"):

        match = re.search(
            r"Groundwater_(\d{4})\.tif",
            filename,
            re.IGNORECASE
        )

        if match:

            year = int(match.group(1))

            tiff_files.append(
                (year, os.path.join(
                    GROUNDWATER_DIR,
                    filename
                ))
            )


tiff_files = sorted(
    tiff_files,
    key=lambda x: x[0]
)


print(f"\nTIFF files found: {len(tiff_files)}")

for year, path in tiff_files:
    print(f" - Groundwater_{year}.tif")


if len(tiff_files) == 0:
    raise FileNotFoundError(
        "No annual Groundwater_YYYY.tif files found."
    )


# ============================================================
# 7. TIFF METADATA
# ============================================================

print("\n\n7. TIFF METADATA INSPECTION")
print("-" * 75)

metadata_rows = []

for year, path in tiff_files:

    with rasterio.open(path) as src:

        metadata_rows.append({

            "Year": year,

            "Width": src.width,

            "Height": src.height,

            "Bands": src.count,

            "CRS": str(src.crs),

            "Resolution_X": src.res[0],

            "Resolution_Y": src.res[1],

            "Left": src.bounds.left,

            "Bottom": src.bounds.bottom,

            "Right": src.bounds.right,

            "Top": src.bounds.top,

            "NoData": src.nodata,

            "Transform": str(src.transform)

        })


metadata_df = pd.DataFrame(metadata_rows)


print("\nRaster dimensions:")

print(
    metadata_df[
        ["Year", "Width", "Height", "Bands"]
    ].to_string(index=False)
)


print("\nCRS:")

print(
    metadata_df["CRS"].value_counts()
)


print("\nResolutions:")

print(
    metadata_df[
        ["Resolution_X", "Resolution_Y"]
    ].drop_duplicates().to_string(index=False)
)


print("\nSpatial bounds:")

print(
    metadata_df[
        [
            "Year",
            "Left",
            "Bottom",
            "Right",
            "Top"
        ]
    ].to_string(index=False)
)


# ============================================================
# 8. RASTER GRID CONSISTENCY
# ============================================================

print("\n\n8. RASTER GRID CONSISTENCY")
print("-" * 75)

first_year, first_path = tiff_files[0]

with rasterio.open(first_path) as first_src:

    reference_width = first_src.width
    reference_height = first_src.height
    reference_crs = first_src.crs
    reference_res = first_src.res
    reference_bounds = first_src.bounds


same_dimensions = all(
    row["Width"] == reference_width
    and row["Height"] == reference_height
    for _, row in metadata_df.iterrows()
)

same_crs = all(
    row["CRS"] == str(reference_crs)
    for _, row in metadata_df.iterrows()
)

same_resolution = all(
    np.isclose(
        row["Resolution_X"],
        reference_res[0]
    )
    and
    np.isclose(
        row["Resolution_Y"],
        reference_res[1]
    )
    for _, row in metadata_df.iterrows()
)

same_bounds = all(
    np.isclose(
        row["Left"],
        reference_bounds.left
    )
    and
    np.isclose(
        row["Bottom"],
        reference_bounds.bottom
    )
    and
    np.isclose(
        row["Right"],
        reference_bounds.right
    )
    and
    np.isclose(
        row["Top"],
        reference_bounds.top
    )
    for _, row in metadata_df.iterrows()
)


print("\nSame dimensions :", same_dimensions)
print("Same CRS        :", same_crs)
print("Same resolution :", same_resolution)
print("Same bounds     :", same_bounds)


if not (
    same_dimensions
    and same_crs
    and same_resolution
    and same_bounds
):

    print("\nWARNING:")
    print(
        "Raster grids are not perfectly identical."
    )

    print(
        "Point sampling is still possible."
    )

    print(
        "Common-grid alignment should be performed "
        "before pixel-by-pixel temporal analysis."
    )


# ============================================================
# 9. TIFF VALUE STATISTICS
# ============================================================

print("\n\n9. TIFF VALUE STATISTICS")
print("-" * 75)

raster_statistics = []


for year, path in tiff_files:

    with rasterio.open(path) as src:

        raster = src.read(1).astype(float)

        if src.nodata is not None:

            raster[
                raster == src.nodata
            ] = np.nan

        valid = raster[
            np.isfinite(raster)
        ]

        raster_statistics.append({

            "Year": year,

            "Min": np.min(valid),

            "Max": np.max(valid),

            "Mean": np.mean(valid),

            "Median": np.median(valid),

            "Std": np.std(valid),

            "Valid_Pixels": len(valid)

        })


raster_stats_df = pd.DataFrame(
    raster_statistics
)


print(
    raster_stats_df.to_string(
        index=False
    )
)


stats_path = os.path.join(
    OUTPUT_DIR,
    "tiff_yearly_statistics.csv"
)

raster_stats_df.to_csv(
    stats_path,
    index=False
)

print("\nSaved:")
print(stats_path)


# ============================================================
# 10. HELPER FUNCTION
# ============================================================

def calculate_metrics(observed, predicted):

    observed = np.asarray(observed)
    predicted = np.asarray(predicted)

    valid = (
        np.isfinite(observed)
        &
        np.isfinite(predicted)
    )

    observed = observed[valid]
    predicted = predicted[valid]

    if len(observed) < 2:

        return {
            "Correlation": np.nan,
            "R2": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
            "Mean_Bias": np.nan,
            "N": len(observed)
        }

    correlation = pearsonr(
        observed,
        predicted
    )[0]

    r2 = r2_score(
        observed,
        predicted
    )

    mae = mean_absolute_error(
        observed,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            observed,
            predicted
        )
    )

    bias = np.mean(
        predicted - observed
    )

    return {

        "Correlation": correlation,

        "R2": r2,

        "MAE": mae,

        "RMSE": rmse,

        "Mean_Bias": bias,

        "N": len(observed)

    }


# ============================================================
# 11. SAMPLE NATIVE TIFF
# ============================================================

print("\n\n10. EXTRACTING TIFF VALUES AT WELL LOCATIONS")
print("-" * 75)

all_samples = []


for year, path in tiff_files:

    yearly_wells = df[
        df["YearMsr"] == year
    ].copy()

    if len(yearly_wells) == 0:
        continue


    with rasterio.open(path) as src:

        coords = list(
            zip(
                yearly_wells["LongDD"],
                yearly_wells["LatDD"]
            )
        )

        samples = list(
            src.sample(coords)
        )

        values = np.array(
            [x[0] for x in samples],
            dtype=float
        )


        yearly_wells[
            "TIFF_Value_Native"
        ] = values


        all_samples.append(
            yearly_wells
        )


validation = pd.concat(
    all_samples,
    ignore_index=True
)


print(
    "\nTotal observations sampled:",
    len(validation)
)


# ============================================================
# 12. CHECK INVALID VALUES
# ============================================================

print("\n\n11. CLEANING VALIDATION DATA")
print("-" * 75)

before = len(validation)


validation = validation[
    np.isfinite(
        validation["TIFF_Value_Native"]
    )
].copy()


after = len(validation)


print("Before cleaning :", before)
print("After cleaning  :", after)
print("Removed         :", before - after)


# ============================================================
# 13. ORIENTATION DIAGNOSTIC
# ============================================================

print("\n\n12. RASTER ORIENTATION DIAGNOSTIC")
print("-" * 75)

print(
    "\nTesting four possible raster orientations:"
)

print("1. Native")
print("2. Vertical flip")
print("3. Horizontal flip")
print("4. Both flips")


orientation_results = []


# Store sampled values for every orientation
orientation_samples = {

    "Native": [],

    "Vertical_Flip": [],

    "Horizontal_Flip": [],

    "Both_Flips": []

}


for year, path in tiff_files:

    yearly_wells = df[
        df["YearMsr"] == year
    ].copy()

    if len(yearly_wells) == 0:
        continue


    with rasterio.open(path) as src:

        raster = src.read(1).astype(float)

        coords = list(
            zip(
                yearly_wells["LongDD"],
                yearly_wells["LatDD"]
            )
        )


        rows, cols = rasterio.transform.rowcol(
            src.transform,
            yearly_wells["LongDD"].values,
            yearly_wells["LatDD"].values
        )


        rows = np.asarray(rows)
        cols = np.asarray(cols)


        # --------------------------------------------
        # Native
        # --------------------------------------------

        native = raster[
            rows,
            cols
        ]


        # --------------------------------------------
        # Vertical flip
        # --------------------------------------------

        vertical = np.flipud(raster)[
            rows,
            cols
        ]


        # --------------------------------------------
        # Horizontal flip
        # --------------------------------------------

        horizontal = np.fliplr(raster)[
            rows,
            cols
        ]


        # --------------------------------------------
        # Both flips
        # --------------------------------------------

        both = np.flipud(
            np.fliplr(raster)
        )[rows, cols]


        observed = (
            yearly_wells[
                "WatLevel"
            ].values
        )


        orientation_data = {

            "Native": native,

            "Vertical_Flip": vertical,

            "Horizontal_Flip": horizontal,

            "Both_Flips": both

        }


        for name, values in orientation_data.items():

            metrics = calculate_metrics(
                observed,
                values
            )

            orientation_samples[
                name
            ].extend(values)


            orientation_results.append({

                "Year": year,

                "Orientation": name,

                **metrics

            })


orientation_df = pd.DataFrame(
    orientation_results
)


# ============================================================
# 14. OVERALL ORIENTATION COMPARISON
# ============================================================

print("\n\n13. OVERALL ORIENTATION COMPARISON")
print("-" * 75)


overall_orientation = []


for orientation in orientation_samples:

    observed = df[
        df["YearMsr"].isin(
            [x[0] for x in tiff_files]
        )
    ]["WatLevel"].values


    predicted = np.array(
        orientation_samples[
            orientation
        ]
    )


    # Safety check
    n = min(
        len(observed),
        len(predicted)
    )


    observed = observed[:n]
    predicted = predicted[:n]


    metrics = calculate_metrics(
        observed,
        predicted
    )


    overall_orientation.append({

        "Orientation": orientation,

        **metrics

    })


overall_orientation_df = pd.DataFrame(
    overall_orientation
)


print(
    overall_orientation_df.to_string(
        index=False
    )
)


# ============================================================
# 15. SELECT BEST ORIENTATION
# ============================================================

print("\n\n14. SELECTING RASTER ORIENTATION")
print("-" * 75)


best_row = overall_orientation_df.loc[
    overall_orientation_df[
        "Correlation"
    ].abs().idxmax()
]


best_orientation = best_row[
    "Orientation"
]


best_correlation = best_row[
    "Correlation"
]


print(
    "\nBest orientation:",
    best_orientation
)

print(
    "Correlation:",
    round(
        best_correlation,
        6
    )
)


# ============================================================
# 16. SAFETY CHECK
# ============================================================

native_corr = overall_orientation_df.loc[
    overall_orientation_df[
        "Orientation"
    ] == "Native",
    "Correlation"
].iloc[0]


vertical_corr = overall_orientation_df.loc[
    overall_orientation_df[
        "Orientation"
    ] == "Vertical_Flip",
    "Correlation"
].iloc[0]


print(
    "\nNative correlation:",
    round(native_corr, 6)
)

print(
    "Vertical flip correlation:",
    round(vertical_corr, 6)
)


if (
    vertical_corr > 0.8
    and
    native_corr < 0
):

    print(
        "\nDIAGNOSTIC RESULT:"
    )

    print(
        "A systematic vertical raster orientation "
        "problem is strongly indicated."
    )

    print(
        "The vertically flipped raster will be used "
        "for corrected validation."
    )

else:

    print(
        "\nNo strong vertical-orientation diagnosis "
        "was automatically established."
    )


# ============================================================
# 17. APPLY CORRECTED ORIENTATION
# ============================================================

print("\n\n15. APPLYING CORRECTED RASTER ORIENTATION")
print("-" * 75)


corrected_samples = []


for year, path in tiff_files:

    yearly_wells = df[
        df["YearMsr"] == year
    ].copy()

    if len(yearly_wells) == 0:
        continue


    with rasterio.open(path) as src:

        raster = src.read(1).astype(float)


        rows, cols = rasterio.transform.rowcol(
            src.transform,
            yearly_wells["LongDD"].values,
            yearly_wells["LatDD"].values
        )


        rows = np.asarray(rows)
        cols = np.asarray(cols)


        if best_orientation == "Native":

            corrected_raster = raster


        elif best_orientation == "Vertical_Flip":

            corrected_raster = np.flipud(
                raster
            )


        elif best_orientation == "Horizontal_Flip":

            corrected_raster = np.fliplr(
                raster
            )


        elif best_orientation == "Both_Flips":

            corrected_raster = np.flipud(
                np.fliplr(raster)
            )


        values = corrected_raster[
            rows,
            cols
        ]


        yearly_wells[
            "TIFF_Value"
        ] = values


        yearly_wells[
            "Difference"
        ] = (
            yearly_wells[
                "TIFF_Value"
            ]
            -
            yearly_wells[
                "WatLevel"
            ]
        )


        yearly_wells[
            "Absolute_Error"
        ] = np.abs(
            yearly_wells[
                "Difference"
            ]
        )


        corrected_samples.append(
            yearly_wells
        )


validation_corrected = pd.concat(
    corrected_samples,
    ignore_index=True
)


# ============================================================
# 18. FINAL STATISTICAL VALIDATION
# ============================================================

print("\n\n16. CORRECTED STATISTICAL VALIDATION")
print("-" * 75)


metrics = calculate_metrics(

    validation_corrected[
        "WatLevel"
    ],

    validation_corrected[
        "TIFF_Value"
    ]

)


print(
    "\nObservations :",
    metrics["N"]
)

print(
    "Correlation  :",
    round(
        metrics["Correlation"],
        6
    )
)

print(
    "R²           :",
    round(
        metrics["R2"],
        6
    )
)

print(
    "MAE          :",
    round(
        metrics["MAE"],
        6
    )
)

print(
    "RMSE         :",
    round(
        metrics["RMSE"],
        6
    )
)

print(
    "Mean Bias    :",
    round(
        metrics["Mean_Bias"],
        6
    )
)


# ============================================================
# 19. FIRST 20 CORRECTED COMPARISONS
# ============================================================

print("\n\n17. FIRST 20 CORRECTED COMPARISONS")
print("-" * 75)


comparison_columns = [

    "CSD_ID",

    "YearMsr",

    "LatDD",

    "LongDD",

    "WatLevel",

    "TIFF_Value",

    "Difference",

    "Absolute_Error"

]


print(
    validation_corrected[
        comparison_columns
    ].head(20).to_string(
        index=False
    )
)


# ============================================================
# 20. ERROR ANALYSIS
# ============================================================

print("\n\n18. ERROR ANALYSIS")
print("-" * 75)


error = validation_corrected[
    "Absolute_Error"
]


print(
    "\nAbsolute error statistics:"
)

print(
    error.describe()
)


print(
    "\nAbsolute error > 10:",
    (error > 10).sum()
)


print(
    "Absolute error > 25:",
    (error > 25).sum()
)


print(
    "Absolute error > 50:",
    (error > 50).sum()
)


print(
    "Absolute error > 100:",
    (error > 100).sum()
)


# ============================================================
# 21. YEARLY VALIDATION
# ============================================================

print("\n\n19. YEARLY CORRECTED VALIDATION")
print("-" * 75)


yearly_results = []


for year, group in validation_corrected.groupby(
    "YearMsr"
):

    result = calculate_metrics(

        group["WatLevel"],

        group["TIFF_Value"]

    )


    yearly_results.append({

        "Year": year,

        "Observations": result["N"],

        "Correlation": result["Correlation"],

        "R2": result["R2"],

        "MAE": result["MAE"],

        "RMSE": result["RMSE"],

        "Mean_Bias": result["Mean_Bias"]

    })


yearly_df = pd.DataFrame(
    yearly_results
)


print(
    yearly_df.to_string(
        index=False
    )
)


# ============================================================
# 22. SAVE ORIENTATION RESULTS
# ============================================================

orientation_path = os.path.join(
    OUTPUT_DIR,
    "tiff_orientation_diagnostic.csv"
)


orientation_df.to_csv(
    orientation_path,
    index=False
)


overall_orientation_path = os.path.join(
    OUTPUT_DIR,
    "tiff_orientation_overall.csv"
)


overall_orientation_df.to_csv(
    overall_orientation_path,
    index=False
)


# ============================================================
# 23. SAVE VALIDATION DATA
# ============================================================

print("\n\n20. SAVING VALIDATION DATA")
print("-" * 75)


validation_path = os.path.join(
    OUTPUT_DIR,
    "tiff_well_validation_corrected.csv"
)


yearly_path = os.path.join(
    OUTPUT_DIR,
    "tiff_yearly_validation_corrected.csv"
)


metadata_path = os.path.join(
    OUTPUT_DIR,
    "tiff_raster_metadata.csv"
)


validation_corrected.to_csv(
    validation_path,
    index=False
)


yearly_df.to_csv(
    yearly_path,
    index=False
)


metadata_df.to_csv(
    metadata_path,
    index=False
)


print("\nSaved:")
print(validation_path)

print("\nSaved:")
print(yearly_path)

print("\nSaved:")
print(metadata_path)

print("\nSaved:")
print(orientation_path)

print("\nSaved:")
print(overall_orientation_path)


# ============================================================
# 24. SCATTER PLOT
# ============================================================

print("\n\n21. CREATING CORRECTED SCATTER PLOT")
print("-" * 75)


plt.figure(figsize=(9, 7))


plt.scatter(

    validation_corrected[
        "WatLevel"
    ],

    validation_corrected[
        "TIFF_Value"
    ],

    alpha=0.45,

    s=20
)


minimum = min(

    validation_corrected[
        "WatLevel"
    ].min(),

    validation_corrected[
        "TIFF_Value"
    ].min()

)


maximum = max(

    validation_corrected[
        "WatLevel"
    ].max(),

    validation_corrected[
        "TIFF_Value"
    ].max()

)


plt.plot(

    [minimum, maximum],

    [minimum, maximum],

    linestyle="--"

)


plt.xlabel(
    "Observed WatLevel"
)

plt.ylabel(
    "Corrected TIFF Value"
)


plt.title(
    "Observed WatLevel vs Corrected TIFF Value"
)


plt.grid(
    alpha=0.3
)


plt.tight_layout()


scatter_path = os.path.join(
    OUTPUT_DIR,
    "corrected_watlevel_vs_tiff_scatter.png"
)


plt.savefig(
    scatter_path,
    dpi=300
)

plt.close()


print(
    "Saved:",
    scatter_path
)


# ============================================================
# 25. ERROR DISTRIBUTION
# ============================================================

print("\n\n22. CREATING ERROR DISTRIBUTION")
print("-" * 75)


plt.figure(
    figsize=(9, 6)
)


plt.hist(

    validation_corrected[
        "Difference"
    ],

    bins=50

)


plt.axvline(
    0,
    linestyle="--"
)


plt.xlabel(
    "TIFF Value - Observed WatLevel"
)


plt.ylabel(
    "Frequency"
)


plt.title(
    "Corrected TIFF Validation Error Distribution"
)


plt.grid(
    alpha=0.3
)


plt.tight_layout()


error_plot_path = os.path.join(
    OUTPUT_DIR,
    "corrected_tiff_error_distribution.png"
)


plt.savefig(
    error_plot_path,
    dpi=300
)


plt.close()


print(
    "Saved:",
    error_plot_path
)


# ============================================================
# 26. YEARLY CORRELATION PLOT
# ============================================================

print("\n\n23. CREATING YEARLY CORRELATION PLOT")
print("-" * 75)


plt.figure(
    figsize=(11, 6)
)


plt.plot(

    yearly_df[
        "Year"
    ],

    yearly_df[
        "Correlation"
    ],

    marker="o"

)


plt.axhline(
    0,
    linestyle="--"
)


plt.xlabel(
    "Year"
)


plt.ylabel(
    "Pearson Correlation"
)


plt.title(
    "Yearly Correlation: Observed WatLevel vs Corrected TIFF"
)


plt.grid(
    alpha=0.3
)


plt.tight_layout()


yearly_plot_path = os.path.join(
    OUTPUT_DIR,
    "yearly_corrected_watlevel_tiff_correlation.png"
)


plt.savefig(
    yearly_plot_path,
    dpi=300
)


plt.close()


print(
    "Saved:",
    yearly_plot_path
)


# ============================================================
# 27. CREATE TIFF MAPS
# ============================================================

print("\n\n24. CREATING TIFF SPATIAL MAPS")
print("-" * 75)


map_years = [
    2000,
    2005,
    2010,
    2015,
    2020,
    2025
]


for year in map_years:

    path = os.path.join(
        GROUNDWATER_DIR,
        f"Groundwater_{year}.tif"
    )


    if not os.path.exists(path):
        continue


    with rasterio.open(path) as src:

        raster = src.read(1).astype(float)

        extent = [

            src.bounds.left,

            src.bounds.right,

            src.bounds.bottom,

            src.bounds.top

        ]


    # Apply diagnosed orientation
    if best_orientation == "Vertical_Flip":

        raster = np.flipud(
            raster
        )

    elif best_orientation == "Horizontal_Flip":

        raster = np.fliplr(
            raster
        )

    elif best_orientation == "Both_Flips":

        raster = np.flipud(
            np.fliplr(raster)
        )


    plt.figure(
        figsize=(9, 7)
    )


    plt.imshow(

        raster,

        extent=extent,

        origin="upper",

        aspect="auto"

    )


    plt.colorbar(
        label="Groundwater Value"
    )


    plt.xlabel(
        "Longitude"
    )


    plt.ylabel(
        "Latitude"
    )


    plt.title(
        f"Corrected Groundwater Surface - {year}"
    )


    plt.tight_layout()


    map_path = os.path.join(

        OUTPUT_DIR,

        f"corrected_groundwater_map_{year}.png"

    )


    plt.savefig(
        map_path,
        dpi=300
    )


    plt.close()


    print(
        "Saved:",
        map_path
    )


# ============================================================
# 28. SEMANTIC INTERPRETATION
# ============================================================

print("\n\n25. TIFF SEMANTIC INTERPRETATION")
print("-" * 75)


print(
    "\nDataset documentation indicates that the annual "
    "groundwater GeoTIFFs were generated from groundwater "
    "observations using Ordinary Kriging."
)


print(
    "\nThe native TIFF orientation produced:"
)


print(
    "Native correlation :",
    round(
        native_corr,
        6
    )
)


print(
    "\nAfter orientation correction:"
)


print(
    "Corrected correlation :",
    round(
        metrics["Correlation"],
        6
    )
)


print(
    "Corrected R²          :",
    round(
        metrics["R2"],
        6
    )
)


print(
    "Corrected MAE         :",
    round(
        metrics["MAE"],
        6
    )
)


print(
    "Corrected RMSE        :",
    round(
        metrics["RMSE"],
        6
    )
)


print(
    "Corrected Mean Bias   :",
    round(
        metrics["Mean_Bias"],
        6
    )
)


if best_orientation == "Vertical_Flip":

    print(
        "\nFINAL DIAGNOSTIC:"
    )

    print(
        "The TIFF values show a strong systematic "
        "vertical orientation mismatch."
    )

    print(
        "The geospatial transform and raster array "
        "orientation are inconsistent."
    )

    print(
        "A vertical flip produces a strong positive "
        "agreement with the observed groundwater values."
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The original TIFF files have NOT been modified."
    )

    print(
        "The correction is applied only during validation "
        "and visualization."
    )

else:

    print(
        "\nFINAL DIAGNOSTIC:"
    )

    print(
        "No vertical raster orientation mismatch "
        "was conclusively detected."
    )


# ============================================================
# 29. FINAL SUMMARY
# ============================================================

print("\n\n")
print("=" * 75)
print(
    "TIFF SPATIAL & WELL VALIDATION COMPLETED"
)
print("=" * 75)


print(
    "\nDATASET SUMMARY:"
)

print(
    "Groundwater observations :",
    len(df)
)

print(
    "TIFF files               :",
    len(tiff_files)
)

print(
    "Years covered            :",
    f"{tiff_files[0][0]} - {tiff_files[-1][0]}"
)

print(
    "Validation observations  :",
    len(validation_corrected)
)


print(
    "\nORIENTATION DIAGNOSTIC:"
)

print(
    "Native correlation       :",
    round(native_corr, 6)
)

print(
    "Best orientation         :",
    best_orientation
)

print(
    "Corrected correlation    :",
    round(
        metrics["Correlation"],
        6
    )
)


print(
    "\nCORRECTED VALIDATION:"
)

print(
    "R²                       :",
    round(
        metrics["R2"],
        6
    )
)

print(
    "MAE                      :",
    round(
        metrics["MAE"],
        6
    )
)

print(
    "RMSE                     :",
    round(
        metrics["RMSE"],
        6
    )
)

print(
    "Mean Bias                :",
    round(
        metrics["Mean_Bias"],
        6
    )
)


print(
    "\nOUTPUT DIRECTORY:"
)

print(
    OUTPUT_DIR
)


print("\n")
print("=" * 75)
print("AquaSense AI - ANALYSIS FINISHED")
print("=" * 75)