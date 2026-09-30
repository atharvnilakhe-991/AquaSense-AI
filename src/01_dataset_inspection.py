import pandas as pd
from pathlib import Path

print("=" * 60)
print("AquaSense AI - DATASET INSPECTION")
print("=" * 60)

# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

groundwater_path = PROJECT_ROOT / "data" / "raw" / "groundwater"
weather_path = PROJECT_ROOT / "data" / "raw" / "weather"


# ---------------------------------------------------------
# 1. AVAILABLE FILES
# ---------------------------------------------------------

print("\n1. AVAILABLE GROUNDWATER FILES")
print("-" * 60)

for file in groundwater_path.iterdir():
    print(" -", file.name)

print("\nAVAILABLE WEATHER FILES")
print("-" * 60)

for file in weather_path.iterdir():
    print(" -", file.name)


# ---------------------------------------------------------
# 2. LOAD GROUNDWATER CSV FILES
# ---------------------------------------------------------

print("\n\n2. LOADING GROUNDWATER DATA")
print("-" * 60)

clean_file = groundwater_path / "Groundwater_Clean.csv"
full_file = groundwater_path / "Groundwater_2000_2025.csv"

groundwater_clean = pd.read_csv(clean_file)
groundwater_full = pd.read_csv(full_file)

print("Groundwater_Clean.csv loaded successfully.")
print("Groundwater_2000_2025.csv loaded successfully.")


# ---------------------------------------------------------
# 3. DATASET SIZE
# ---------------------------------------------------------

print("\n\n3. DATASET SIZE")
print("-" * 60)

print("\nGroundwater_Clean.csv")
print("Rows:", groundwater_clean.shape[0])
print("Columns:", groundwater_clean.shape[1])

print("\nGroundwater_2000_2025.csv")
print("Rows:", groundwater_full.shape[0])
print("Columns:", groundwater_full.shape[1])


# ---------------------------------------------------------
# 4. COLUMN NAMES
# ---------------------------------------------------------

print("\n\n4. COLUMN NAMES")
print("-" * 60)

print("\nGroundwater_Clean.csv:")
for column in groundwater_clean.columns:
    print(" -", column)

print("\nGroundwater_2000_2025.csv:")
for column in groundwater_full.columns:
    print(" -", column)


# ---------------------------------------------------------
# 5. FIRST FIVE RECORDS
# ---------------------------------------------------------

print("\n\n5. FIRST FIVE RECORDS")
print("-" * 60)

print("\nGroundwater_Clean.csv:")
print(groundwater_clean.head())

print("\nGroundwater_2000_2025.csv:")
print(groundwater_full.head())


# ---------------------------------------------------------
# 6. DATA TYPES
# ---------------------------------------------------------

print("\n\n6. DATA TYPES")
print("-" * 60)

print("\nGroundwater_Clean.csv:")
print(groundwater_clean.dtypes)

print("\nGroundwater_2000_2025.csv:")
print(groundwater_full.dtypes)


# ---------------------------------------------------------
# 7. MISSING VALUES
# ---------------------------------------------------------

print("\n\n7. MISSING VALUES")
print("-" * 60)

print("\nGroundwater_Clean.csv:")
print(groundwater_clean.isnull().sum())

print("\nGroundwater_2000_2025.csv:")
print(groundwater_full.isnull().sum())


# ---------------------------------------------------------
# 8. DUPLICATES
# ---------------------------------------------------------

print("\n\n8. DUPLICATE RECORDS")
print("-" * 60)

print(
    "Groundwater_Clean:",
    groundwater_clean.duplicated().sum()
)

print(
    "Groundwater_2000_2025:",
    groundwater_full.duplicated().sum()
)


# ---------------------------------------------------------
# 9. WEATHER DATA
# ---------------------------------------------------------

print("\n\n9. NASA POWER WEATHER DATA")
print("-" * 60)

weather_file = weather_path / "NASA_POWER_2000_2025.csv"

weather = pd.read_csv(weather_file,skiprows=13)

print("Rows:", weather.shape[0])
print("Columns:", weather.shape[1])

print("\nColumn names:")
for column in weather.columns:
    print(" -", column)

print("\nFirst five records:")
print(weather.head())

print("\nData types:")
print(weather.dtypes)

print("\nMissing values:")
print(weather.isnull().sum())


# ---------------------------------------------------------
# END
# ---------------------------------------------------------

print("\n\n" + "=" * 60)
print("DATASET INSPECTION COMPLETED")
print("=" * 60)