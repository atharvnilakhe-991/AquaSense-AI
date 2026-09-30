"""
AquaSense AI - ML Configuration
Centralized configuration, paths, features, and model parameters.
"""

from pathlib import Path

# Project root (dynamically resolved)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LEAKAGE_SAFE_DATASET = PROCESSED_DATA_DIR / "ml_validation" / "groundwater_ml_leakage_safe.csv"
OUTPUT_DIR = PROCESSED_DATA_DIR / "advanced_ml"
UNSEEN_WELL_OUTPUT_DIR = OUTPUT_DIR / "unseen_well"

# Seed & Split
RANDOM_STATE = 42
TEST_WELL_RATIO = 0.20  # 80/20 unseen well split

# Primary Environmental & Spatial Features (Leakage-Safe)
PRIMARY_FEATURES = [
    "LatDD",
    "LongDD",
    "Surf_Elev",
    "Annual_Temperature_Mean",
    "Annual_Precipitation_Total",
    "Annual_Humidity_Mean",
    "Annual_WindSpeed_Mean",
    "Annual_SolarRadiation_Mean",
]

# Target & Well Identifier
TARGET_COLUMN = "WatLevel"
WELL_ID_COLUMN = "CSD_ID"
DATE_COLUMN = "DateMsr"
YEAR_COLUMN = "Year"

# Explicitly Excluded Features (Target Leakage Prevention)
LEAKAGE_COLUMNS = [
    "TIFF_Value",
    "tiff_value",
    "groundwater_depth_raster",
]
