"""
AquaSense AI - AMDFE Configuration (Version 2)
Defines modality feature sets, paths, leakage protection rules, and ablation suites (A0 - A6).
"""

from pathlib import Path
from ml.config import PROJECT_ROOT, DATA_DIR, PROCESSED_DATA_DIR, RANDOM_STATE

# AMDFE v1 Output Directories
AMDFE_DIR = PROCESSED_DATA_DIR / "amdfe"
AMDFE_QUALITY_DIR = AMDFE_DIR / "quality"
AMDFE_RELIABILITY_DIR = AMDFE_DIR / "research"
AMDFE_MISSINGNESS_DIR = AMDFE_DIR / "missingness"
AMDFE_FUSED_DIR = AMDFE_DIR / "fused"
AMDFE_VALIDATION_DIR = AMDFE_DIR / "validation"
AMDFE_EXPERIMENTS_DIR = AMDFE_DIR / "experiments"

# AMDFE v2 Dedicated Versioned Output Directories
AMDFE_V2_DIR = PROCESSED_DATA_DIR / "amdfe_v2"
AMDFE_V2_QUALITY_DIR = AMDFE_V2_DIR / "quality"
AMDFE_V2_RELIABILITY_DIR = AMDFE_V2_DIR / "reliability"
AMDFE_V2_OBSERVABILITY_DIR = AMDFE_V2_DIR / "observability"
AMDFE_V2_FUSED_DIR = AMDFE_V2_DIR / "fused"
AMDFE_V2_MISSINGNESS_DIR = AMDFE_V2_DIR / "missingness"
AMDFE_V2_VALIDATION_DIR = AMDFE_V2_DIR / "validation"
AMDFE_V2_EXPERIMENTS_DIR = AMDFE_V2_DIR / "experiments"
AMDFE_V2_ROBUSTNESS_DIR = AMDFE_V2_DIR / "robustness"

# AMDFE v2.1 Dedicated Correction & Attribution Directories
AMDFE_V21_DIR = PROCESSED_DATA_DIR / "amdfe_v21"
AMDFE_V21_QUALITY_DIR = AMDFE_V21_DIR / "quality"
AMDFE_V21_RELIABILITY_DIR = AMDFE_V21_DIR / "reliability"
AMDFE_V21_OBSERVABILITY_DIR = AMDFE_V21_DIR / "observability"
AMDFE_V21_FUSED_DIR = AMDFE_V21_DIR / "fused"
AMDFE_V21_VALIDATION_DIR = AMDFE_V21_DIR / "validation"
AMDFE_V21_EXPERIMENTS_DIR = AMDFE_V21_DIR / "experiments"
AMDFE_V21_ROBUSTNESS_DIR = AMDFE_V21_DIR / "robustness"
AMDFE_V21_STATISTICS_DIR = AMDFE_V21_DIR / "statistics"

# Source Data Paths
INTEGRATED_DATASET_PATH = PROCESSED_DATA_DIR / "integrated" / "groundwater_integrated_dataset.csv"
ANNUAL_WEATHER_PATH = PROCESSED_DATA_DIR / "integrated" / "annual_weather_features.csv"
IRREGULAR_FEATURES_PATH = (
    PROCESSED_DATA_DIR / "advanced_ml" / "irregular_observation" / "groundwater_irregular_observation_features.csv"
)

# Identifiers and Target
TARGET_COLUMN = "WatLevel"
WELL_ID_COLUMN = "CSD_ID"
DATE_COLUMN = "DateMsr"
YEAR_COLUMN = "YearMsr"

# =============================================================================
# MODALITY DEFINITIONS
# =============================================================================

# Modality A: Groundwater History / Prior State (Strict prior observations)
MODALITY_A_FEATURES = [
    "Previous_WatLevel",
    "Previous2_WatLevel",
    "Days_Since_Previous",
    "Years_Since_Previous",
    "Rolling_Mean_3",
    "Rolling_Std_3",
    "Previous_Level_Change",
    "Recent_Trend",
    "Historical_Mean",
    "Historical_Std",
    "Historical_Min",
    "Historical_Max",
    "Previous_Observation_Count",
    "Observation_Density",
    "Long_Gap_Flag",
    "Very_Long_Gap_Flag",
]

# Modality B: Environmental / Weather (Strict leakage-safe completed year Y-1)
MODALITY_B_FEATURES = [
    "Annual_Temperature_Mean",
    "Annual_Precipitation_Total",
    "Annual_Humidity_Mean",
    "Annual_WindSpeed_Mean",
    "Annual_SolarRadiation_Mean",
]

# Modality C: Spatial / Topographic Context
MODALITY_C_FEATURES = [
    "LatDD",
    "LongDD",
    "Surf_Elev",
]

# Modality D: Earth Observation / SoilGrids (Currently unavailable interface)
MODALITY_D_FEATURES = []
MODALITY_D_STATUS = "UNAVAILABLE_IN_CURRENT_REPOSITORY"

# Explicit Banned / Leakage Columns
LEAKAGE_COLUMNS = [
    "TIFF_Value",
    "tiff_value",
    "DEM_Elevation",
    "dem_elevation",
    "groundwater_depth_raster",
]

# =============================================================================
# AMDFE v2 ABLATION SUITE (A0 - A6)
# =============================================================================

ABLATION_CONFIGS_V2 = {
    "A0": {
        "name": "Groundwater Only",
        "description": "Groundwater temporal history/state features only (Modality A).",
        "modalities": ["modality_a"],
        "fusion_type": "single",
        "has_reliability_weights": False,
        "has_reliability_meta": False,
        "has_observability_meta": False,
    },
    "A1": {
        "name": "Groundwater + Weather",
        "description": "Groundwater history + leakage-safe completed period weather (Modalities A + B).",
        "modalities": ["modality_a", "modality_b"],
        "fusion_type": "concat",
        "has_reliability_weights": False,
        "has_reliability_meta": False,
        "has_observability_meta": False,
    },
    "A2": {
        "name": "Unweighted Multimodal Concatenation",
        "description": "All available modalities (A, B, C) concatenated without weighting or reliability metadata.",
        "modalities": ["modality_a", "modality_b", "modality_c"],
        "fusion_type": "concat",
        "has_reliability_weights": False,
        "has_reliability_meta": False,
        "has_observability_meta": False,
    },
    "A3": {
        "name": "Reliability Metadata Augmentation",
        "description": "Same multimodal features (A, B, C) with static source reliability scores R_m appended, NO adaptive gating.",
        "modalities": ["modality_a", "modality_b", "modality_c"],
        "fusion_type": "metadata_only",
        "has_reliability_weights": False,
        "has_reliability_meta": True,
        "has_observability_meta": False,
    },
    "A4": {
        "name": "Global Reliability Scaling",
        "description": "Same multimodal features scaled by constant global modality weights w_m = R_m / sum(R_k).",
        "modalities": ["modality_a", "modality_b", "modality_c"],
        "fusion_type": "global_reliability",
        "has_reliability_weights": True,
        "has_reliability_meta": True,
        "has_observability_meta": False,
    },
    "A5": {
        "name": "Context-Adaptive AMDFE",
        "description": "Full two-stage adaptive fusion: G_{i,m} = R_m * A_{i,m}, w_{i,m} = G_{i,m} / sum(G_{i,k}), scaling standardized blocks.",
        "modalities": ["modality_a", "modality_b", "modality_c"],
        "fusion_type": "adaptive_amdfe",
        "has_reliability_weights": True,
        "has_reliability_meta": True,
        "has_observability_meta": True,
    },
    "A6": {
        "name": "AMDFE + Robustness Metadata",
        "description": "A5 plus explicit missingness indicators, gap flags, and monitoring-density metadata features.",
        "modalities": ["modality_a", "modality_b", "modality_c"],
        "fusion_type": "adaptive_amdfe_plus_meta",
        "has_reliability_weights": True,
        "has_reliability_meta": True,
        "has_observability_meta": True,
    },
}
ABLATION_CONFIGS = {
    "B0": {"name": "Groundwater Only", "modalities": ["modality_a"], "fusion_type": "single", "has_reliability_weights": False},
    "B1": {"name": "Groundwater + Weather", "modalities": ["modality_a", "modality_b"], "fusion_type": "concat", "has_reliability_weights": False},
    "B2": {"name": "Unweighted Multimodal", "modalities": ["modality_a", "modality_b", "modality_c"], "fusion_type": "concat", "has_reliability_weights": False},
    "B3": {"name": "Reliability Weighted", "modalities": ["modality_a", "modality_b", "modality_c"], "fusion_type": "reliability_weighted", "has_reliability_weights": True},
    "B4": {"name": "Full Adaptive AMDFE", "modalities": ["modality_a", "modality_b", "modality_c"], "fusion_type": "adaptive_gated", "has_reliability_weights": True},
    "B5": {"name": "Robustness Extended", "modalities": ["modality_a", "modality_b", "modality_c"], "fusion_type": "adaptive_plus_meta", "has_reliability_weights": True},
}

# =============================================================================
# AMDFE v2.1 MATCHED-FEATURE ATTRIBUTION ABLATION SUITE (C0, C1, C2, A0, A1, A6)
# =============================================================================

ABLATION_CONFIGS_V21 = {
    "C0": {
        "name": "Unweighted Multimodal Base (A2)",
        "description": "Base modalities A, B, C concatenated without weighting or metadata (23 features).",
        "modalities": ["modality_a", "modality_b", "modality_c"],
        "fusion_type": "concat",
        "has_reliability_weights": False,
        "has_reliability_meta": False,
        "has_observability_meta": False,
    },
    "C1": {
        "name": "Multimodal Base + Full Metadata (No Gating)",
        "description": "Base modalities A, B, C + R_m metadata + A_{i,m} metadata + w_{i,m} metadata WITHOUT adaptive scaling (32 features).",
        "modalities": ["modality_a", "modality_b", "modality_c"],
        "fusion_type": "metadata_only_all",
        "has_reliability_weights": True,
        "has_reliability_meta": True,
        "has_observability_meta": True,
    },
    "C2": {
        "name": "Core Context-Adaptive AMDFE (Gated)",
        "description": "Same exact 32 features as C1, with adaptive sample-level block gating w_{i,m} * Z(X_{i,m}).",
        "modalities": ["modality_a", "modality_b", "modality_c"],
        "fusion_type": "adaptive_amdfe",
        "has_reliability_weights": True,
        "has_reliability_meta": True,
        "has_observability_meta": True,
    },
    "A0": {
        "name": "Groundwater Only",
        "description": "Groundwater history features only (15 features).",
        "modalities": ["modality_a"],
        "fusion_type": "single",
        "has_reliability_weights": False,
        "has_reliability_meta": False,
        "has_observability_meta": False,
    },
    "A1": {
        "name": "Groundwater + Weather",
        "description": "Groundwater history + completed period weather (20 features).",
        "modalities": ["modality_a", "modality_b"],
        "fusion_type": "concat",
        "has_reliability_weights": False,
        "has_reliability_meta": False,
        "has_observability_meta": False,
    },
    "A6": {
        "name": "Core AMDFE + Robustness Metadata",
        "description": "Core adaptive AMDFE (C2) + explicit missingness indicators, gap flags, and outlier flags (60 features).",
        "modalities": ["modality_a", "modality_b", "modality_c"],
        "fusion_type": "adaptive_amdfe_plus_meta",
        "has_reliability_weights": True,
        "has_reliability_meta": True,
        "has_observability_meta": True,
    },
}
