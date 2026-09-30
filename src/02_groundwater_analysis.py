import pandas as pd
from pathlib import Path

print("=" * 70)
print("AquaSense AI - GROUNDWATER DATA ANALYSIS")
print("=" * 70)

# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

groundwater_path = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "groundwater"
)

clean_file = groundwater_path / "Groundwater_Clean.csv"
full_file = groundwater_path / "Groundwater_2000_2025.csv"


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

clean_df = pd.read_csv(clean_file)
full_df = pd.read_csv(full_file)


# ---------------------------------------------------------
# CONVERT DATE COLUMNS
# ---------------------------------------------------------

clean_df["DateMsr"] = pd.to_datetime(
    clean_df["DateMsr"],
    errors="coerce"
)

full_df["DateMsr"] = pd.to_datetime(
    full_df["DateMsr"],
    errors="coerce"
)


# ---------------------------------------------------------
# 1. BASIC INFORMATION
# ---------------------------------------------------------

print("\n1. BASIC INFORMATION")
print("-" * 70)

print("Clean dataset rows:", len(clean_df))
print("Full dataset rows:", len(full_df))


# ---------------------------------------------------------
# 2. UNIQUE WELLS
# ---------------------------------------------------------

print("\n2. UNIQUE GROUNDWATER LOCATIONS")
print("-" * 70)

print(
    "Unique CSD_IDs:",
    clean_df["CSD_ID"].nunique()
)


# ---------------------------------------------------------
# 3. COUNTIES
# ---------------------------------------------------------

print("\n3. COUNTIES")
print("-" * 70)

print(
    "Number of counties:",
    clean_df["County"].nunique()
)

print(
    clean_df["County"].value_counts()
)


# ---------------------------------------------------------
# 4. DATE RANGE
# ---------------------------------------------------------

print("\n4. DATE RANGE")
print("-" * 70)

print(
    "Earliest measurement:",
    clean_df["DateMsr"].min()
)

print(
    "Latest measurement:",
    clean_df["DateMsr"].max()
)


# ---------------------------------------------------------
# 5. OBSERVATIONS PER YEAR
# ---------------------------------------------------------

print("\n5. OBSERVATIONS PER YEAR")
print("-" * 70)

print(
    clean_df["YearMsr"]
    .value_counts()
    .sort_index()
)


# ---------------------------------------------------------
# 6. WATER LEVEL STATISTICS
# ---------------------------------------------------------

print("\n6. WATER LEVEL STATISTICS")
print("-" * 70)

print(
    clean_df["WatLevel"].describe()
)


# ---------------------------------------------------------
# 7. MISSING VALUES
# ---------------------------------------------------------

print("\n7. MISSING VALUES")
print("-" * 70)

print(
    clean_df.isnull().sum()
)


# ---------------------------------------------------------
# 8. MISSING SEASON
# ---------------------------------------------------------

print("\n8. MISSING SEASON")
print("-" * 70)

missing_season = clean_df[
    clean_df["Season"].isna()
]

print(
    "Missing Season records:",
    len(missing_season)
)

print("\nFirst 10:")
print(
    missing_season[
        ["CSD_ID", "DateMsr", "YearMsr", "WatLevel"]
    ].head(10)
)


# ---------------------------------------------------------
# 9. SEASON DISTRIBUTION
# ---------------------------------------------------------

print("\n9. SEASON DISTRIBUTION")
print("-" * 70)

print(
    clean_df["Season"].value_counts(
        dropna=False
    )
)


# ---------------------------------------------------------
# 10. SPATIAL RANGE
# ---------------------------------------------------------

print("\n10. SPATIAL RANGE")
print("-" * 70)

print(
    "Latitude minimum:",
    clean_df["LatDD"].min()
)

print(
    "Latitude maximum:",
    clean_df["LatDD"].max()
)

print(
    "Longitude minimum:",
    clean_df["LongDD"].min()
)

print(
    "Longitude maximum:",
    clean_df["LongDD"].max()
)


# ---------------------------------------------------------
# 11. SURFACE ELEVATION
# ---------------------------------------------------------

print("\n11. SURFACE ELEVATION"
)

print("-" * 70)

print(
    clean_df["Surf_Elev"].describe()
)


# ---------------------------------------------------------
# 12. DUPLICATE WELL + DATE
# ---------------------------------------------------------

print("\n12. DUPLICATE WELL + DATE RECORDS")
print("-" * 70)

duplicate_mask = clean_df.duplicated(
    subset=["CSD_ID", "DateMsr"],
    keep=False
)

duplicate_records = clean_df[
    duplicate_mask
].sort_values(
    ["CSD_ID", "DateMsr"]
)

print(
    "Duplicate CSD_ID + DateMsr records:",
    len(duplicate_records)
)

print("\nDuplicate records:")
print(duplicate_records.to_string(index=False))


# ---------------------------------------------------------
# 13. COMPARE DATASET SIZES
# ---------------------------------------------------------

print("\n13. COMPARING GROUNDWATER FILES")
print("-" * 70)

print(
    "Clean rows:",
    len(clean_df)
)

print(
    "Full rows:",
    len(full_df)
)

print(
    "Difference:",
    len(full_df) - len(clean_df)
)


# ---------------------------------------------------------
# 14. FIND RECORDS ONLY IN FULL DATASET
# ---------------------------------------------------------

print("\n14. RECORDS ONLY IN FULL DATASET")
print("-" * 70)

comparison_columns = [
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

extra_records = full_df.merge(
    clean_df[comparison_columns],
    on=comparison_columns,
    how="left",
    indicator=True
)

extra_records = extra_records[
    extra_records["_merge"] == "left_only"
]

print(
    "Records only in full dataset:",
    len(extra_records)
)

print(
    extra_records[
        comparison_columns
    ].to_string(index=False)
)


# ---------------------------------------------------------
# 15. CHECK WHETHER CLEAN IS SUBSET OF FULL
# ---------------------------------------------------------

print("\n15. DATASET RELATIONSHIP")
print("-" * 70)

clean_in_full = clean_df.merge(
    full_df[comparison_columns],
    on=comparison_columns,
    how="left",
    indicator=True
)

missing_from_full = clean_in_full[
    clean_in_full["_merge"] == "left_only"
]

print(
    "Clean records missing from full dataset:",
    len(missing_from_full)
)


# ---------------------------------------------------------
# END
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("GROUNDWATER ANALYSIS COMPLETED")
print("=" * 70)