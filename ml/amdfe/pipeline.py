"""
AquaSense AI - AMDFE Pipeline
Encapsulates complete end-to-end adaptive fusion pipeline with strict train/test isolation,
frozen parameter persistence, and reproducible transformation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from ml.amdfe.config import (
    AMDFE_FUSED_DIR,
    MODALITY_A_FEATURES,
    MODALITY_B_FEATURES,
    MODALITY_C_FEATURES,
    TARGET_COLUMN,
    WELL_ID_COLUMN,
    DATE_COLUMN,
    YEAR_COLUMN,
    LEAKAGE_COLUMNS,
    ABLATION_CONFIGS,
)
from ml.amdfe.quality import run_data_quality_assessment
from ml.amdfe.reliability import (
    compute_global_reliability_scores,
    compute_context_aware_reliability,
    evaluate_reliability_aggregation_sensitivity,
)
from ml.amdfe.missingness import LeakageSafeImputer, audit_missingness_and_imputation
from ml.amdfe.fusion import ModalityBlockScaler, compute_adaptive_weights, fuse_modality_blocks


class AMDFEPipeline:
    """
    Research-grade Adaptive Multimodal Data Fusion Engine (AMDFE).

    Guarantees:
    - Zero data leakage between partitions (all scalers, imputers, global quality fit on TRAIN only).
    - Transparent reliability scores and dynamic weighting.
    - Full traceability of raw, standardized, and fused representations.
    """

    def __init__(self, aggregation_method: str = "geometric"):
        self.aggregation_method = aggregation_method
        self.imputer = LeakageSafeImputer(strategy="median")
        self.scaler = ModalityBlockScaler()
        self.global_reliability: Dict[str, float] = {}
        self.quality_summary: Optional[pd.DataFrame] = None
        self.is_fitted: bool = False

        self.modality_feature_map = {
            "modality_a": MODALITY_A_FEATURES,
            "modality_b": MODALITY_B_FEATURES,
            "modality_c": MODALITY_C_FEATURES,
        }

    def fit(self, train_df: pd.DataFrame, save_artifacts: bool = True) -> "AMDFEPipeline":
        """
        Fits all statistical transformations strictly on the training partition.
        """
        # Purge leakage columns
        train_clean = train_df.copy()
        for col in LEAKAGE_COLUMNS:
            if col in train_clean.columns:
                train_clean = train_clean.drop(columns=[col])

        # 1. Quality Assessment on Train
        self.quality_summary = run_data_quality_assessment(train_clean, save_reports=save_artifacts)

        # 2. Global Reliability on Train
        self.global_reliability = compute_global_reliability_scores(
            train_clean, aggregation=self.aggregation_method
        )
        if save_artifacts:
            evaluate_reliability_aggregation_sensitivity(train_clean, save_results=True)

        # 3. Fit Imputer on Train
        all_features = MODALITY_A_FEATURES + MODALITY_B_FEATURES + MODALITY_C_FEATURES
        self.imputer.fit(train_clean, all_features)

        # 4. Impute Train to fit Scalers
        train_imputed = self.imputer.transform(train_clean, all_features)
        self.scaler.fit(train_imputed, self.modality_feature_map)

        if save_artifacts:
            audit_missingness_and_imputation(
                train_clean, test_df=None, imputer=self.imputer, save_reports=True
            )

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame, config_key: str = "B4") -> pd.DataFrame:
        """
        Transforms input partition (train, validation, or test) using frozen parameters.
        """
        if not self.is_fitted:
            raise RuntimeError("AMDFEPipeline must be fitted on training data before calling transform!")

        data = df.copy()
        # Enforce leakage check
        for col in LEAKAGE_COLUMNS:
            if col in data.columns:
                data = data.drop(columns=[col])

        cfg = ABLATION_CONFIGS.get(config_key, ABLATION_CONFIGS["B4"])
        active_modalities = cfg["modalities"]
        fusion_type = cfg["fusion_type"]

        # 1. Compute Row Availability and Reliability using train-fitted global scores
        rel_df = compute_context_aware_reliability(data, self.global_reliability)
        for col in rel_df.columns:
            data[col] = rel_df[col]

        # 2. Compute Normalized Adaptive Weights
        weights_df, fallback_flag = compute_adaptive_weights(
            rel_df, active_modalities=active_modalities
        )
        for col in weights_df.columns:
            data[col] = weights_df[col]

        # 3. Apply Frozen Imputation & Indicators
        all_features = MODALITY_A_FEATURES + MODALITY_B_FEATURES + MODALITY_C_FEATURES
        data_imputed = self.imputer.transform(data, all_features)

        # 4. Apply Frozen Standardization
        data_std = self.scaler.transform(data_imputed, self.modality_feature_map)

        # 5. Apply Fusion
        data_fused = fuse_modality_blocks(
            data_std,
            weights_df=weights_df,
            active_modalities=active_modalities,
            fusion_mode=fusion_type,
        )

        return data_fused

    def fit_transform(self, train_df: pd.DataFrame, config_key: str = "B4") -> pd.DataFrame:
        """
        Fits on train and transforms train.
        """
        return self.fit(train_df, save_artifacts=False).transform(train_df, config_key=config_key)


def generate_all_fused_datasets(save_to_disk: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Generates and saves the 5 canonical fused datasets (B0 - B4) to data/processed/amdfe/fused/.
    """
    from ml.amdfe.ingestion import ingest_amdfe_dataset

    print("[AMDFE] Ingesting multimodal dataset...")
    df_raw = ingest_amdfe_dataset(enforce_strict_leakage_check=True, save_audit=True)

    pipeline = AMDFEPipeline(aggregation_method="geometric")
    print("[AMDFE] Fitting pipeline on full dataset for reference fused tables...")
    pipeline.fit(df_raw, save_artifacts=True)

    fused_datasets = {}
    config_filename_map = {
        "B0": "groundwater_only.csv",
        "B1": "groundwater_weather.csv",
        "B2": "fixed_weight_fusion.csv",
        "B3": "reliability_weighted_fusion.csv",
        "B4": "adaptive_amdfe_fusion.csv",
    }

    if save_to_disk:
        AMDFE_FUSED_DIR.mkdir(parents=True, exist_ok=True)

    for cfg_key, filename in config_filename_map.items():
        print(f"[AMDFE] Generating fused dataset for {cfg_key} ({ABLATION_CONFIGS[cfg_key]['name']})...")
        fused = pipeline.transform(df_raw, config_key=cfg_key)
        
        # Verify TIFF_Value never present
        assert "TIFF_Value" not in fused.columns, "TIFF_Value leakage detected!"
        assert "tiff_value" not in fused.columns, "tiff_value leakage detected!"

        fused_datasets[cfg_key] = fused

        if save_to_disk:
            out_path = AMDFE_FUSED_DIR / filename
            fused.to_csv(out_path, index=False)
            print(f"  -> Saved to: {out_path} ({len(fused)} rows, {len(fused.columns)} cols)")

    return fused_datasets
