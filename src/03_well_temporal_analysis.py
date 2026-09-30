import pandas as pd
from pathlib import Path

print("=" * 70)
print("AquaSense AI - WELL TEMPORAL ANALYSIS")
print("=" * 70)

# ---------------------------------------------------------
# PATH
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

groundwater_file = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "groundwater"
    / "Groundwater_Clean.csv"
)

df = pd.read_csv(groundwater_file)

df["DateMsr"] = pd.to_datetime(
    df["DateMsr"],
    errors="coerce"
)

# ---------------------------------------------------------
# 1. OBSERVATIONS PER WELL
# ---------------------------------------------------------

print("\n1. OBSERVATIONS PER WELL")
print("-" * 70)

well_counts = (
    df.groupby("CSD_ID")
    .size()
    .sort_values()
)

print(well_counts.describe())

print("\nWells with fewest observations:")
print(well_counts.head(10))

print("\nWells with most observations:")
print(well_counts.tail(10))


# ---------------------------------------------------------
# 2. TEMPORAL COVERAGE PER WELL
# ---------------------------------------------------------

print("\n2. TEMPORAL COVERAGE PER WELL")
print("-" * 70)

well_dates = df.groupby("CSD_ID")["DateMsr"].agg(
    ["min", "max", "count"]
)

well_dates["years_covered"] = (
    well_dates["max"].dt.year -
    well_dates["min"].dt.year +
    1
)

print(well_dates.describe())


# ---------------------------------------------------------
# 3. WELLS BY NUMBER OF YEARS
# ---------------------------------------------------------

print("\n3. WELLS WITH LONGEST TEMPORAL COVERAGE")
print("-" * 70)

print(
    well_dates
    .sort_values("years_covered", ascending=False)
    .head(20)
)


# ---------------------------------------------------------
# 4. OBSERVATION FREQUENCY
# ---------------------------------------------------------

print("\n4. OBSERVATIONS PER WELL")
print("-" * 70)

frequency = (
    df.groupby("CSD_ID")
    .size()
    .value_counts()
    .sort_index()
)

print(frequency)


# ---------------------------------------------------------
# 5. SAMPLE WELL HISTORIES
# ---------------------------------------------------------

print("\n5. SAMPLE WELL HISTORIES")
print("-" * 70)

sample_wells = df["CSD_ID"].drop_duplicates().head(5)

for well in sample_wells:

    print("\n" + "-" * 50)
    print("CSD_ID:", well)

    well_data = (
        df[df["CSD_ID"] == well]
        .sort_values("DateMsr")
    )

    print(
        well_data[
            ["DateMsr", "YearMsr", "Season", "WatLevel"]
        ].to_string(index=False)
    )


# ---------------------------------------------------------
# 6. MISSING YEARS PER WELL
# ---------------------------------------------------------

print("\n6. TEMPORAL SPARSITY")
print("-" * 70)

sparse_count = 0

for well, group in df.groupby("CSD_ID"):

    years = sorted(group["YearMsr"].unique())

    if len(years) > 1:

        expected = set(
            range(
                min(years),
                max(years) + 1
            )
        )

        missing = expected - set(years)

        if len(missing) > 0:
            sparse_count += 1

print(
    "Wells with missing years:",
    sparse_count
)


# ---------------------------------------------------------
# END
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("WELL TEMPORAL ANALYSIS COMPLETED")
print("=" * 70)