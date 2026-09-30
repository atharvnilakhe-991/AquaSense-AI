"""
AquaSense AI - Adaptive Multimodal Data Fusion Engine (AMDFE)
A research-grade preprocessing and multimodal fusion framework for groundwater prediction.
"""

from ml.amdfe.config import (
    AMDFE_DIR,
    AMDFE_QUALITY_DIR,
    AMDFE_RELIABILITY_DIR,
    AMDFE_MISSINGNESS_DIR,
    AMDFE_FUSED_DIR,
    AMDFE_VALIDATION_DIR,
    AMDFE_EXPERIMENTS_DIR,
    MODALITY_A_FEATURES,
    MODALITY_B_FEATURES,
    MODALITY_C_FEATURES,
    MODALITY_D_FEATURES,
    TARGET_COLUMN,
    WELL_ID_COLUMN,
    DATE_COLUMN,
    YEAR_COLUMN,
    LEAKAGE_COLUMNS,
)
from ml.amdfe.pipeline import AMDFEPipeline
from ml.amdfe.validation import run_leakage_validation_suite

__all__ = [
    "AMDFEPipeline",
    "run_leakage_validation_suite",
    "MODALITY_A_FEATURES",
    "MODALITY_B_FEATURES",
    "MODALITY_C_FEATURES",
    "MODALITY_D_FEATURES",
    "TARGET_COLUMN",
    "WELL_ID_COLUMN",
    "DATE_COLUMN",
    "YEAR_COLUMN",
    "LEAKAGE_COLUMNS",
]
