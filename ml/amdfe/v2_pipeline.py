"""
AquaSense AI - AMDFE Version 2 Stateful Pipeline
Encapsulates complete research-grade preprocessing and fusion layer with strict
train-only parameter fitting, non-applicable dimension masking, and frozen transform logic.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from ml.amdfe.config import (
    AMDFE_V2_FUSED_DIR,
    AMDFE_V2_QUALITY_DIR,
    AMDFE_V2_RELIABILITY_DIR,
    AMDFE_V2_OBSERVABILITY_DIR,
    AMDFE_V2_MISSINGNESS_DIR,
    MODALITY_A_FEATURES,
    MODALITY_B_FEATURES,
    MODALITY_C_FEATURES,
    TARGET_COLUMN,
    WELL_ID_COLUMN,
    DATE_COLUMN,
    YEAR_COLUMN,
    LEAKAGE_COLUMNS,
    ABLATION_CONFIGS_V2,
    ABLATION_CONFIGS_V21,
)
from ml.amdfe.v2_reliability import (
    compute_v2_source_reliability,
    evaluate_v2_reliability_sensitivity,
    evaluate_modality_quality_dimensions,
)
from ml.amdfe.v2_observability import ContextObservabilityEngine
from ml.amdfe.v2_outliers import RobustOutlierDetector
from ml.amdfe.missingness import LeakageSafeImputer, audit_missingness_and_imputation
from ml.amdfe.fusion import ModalityBlockScaler
from ml.amdfe.v2_fusion import compute_v2_adaptive_weights, fuse_v2_modality_blocks


class AMDFEPipelineV2:
    """
    AMDFE Version 2 Pipeline.
    Strictly guarantees:
    - Decoupled Source Reliability R_m from Context Observability A_{i,m}.
    - Train-only fitted imputers, scalers, quantile cohort boundaries, and robust MAD outlier scales.
    - Zero future observation lookahead or test snooping.
    - Transparent feature tracking across ablations A0 - A6.
    """

    def __init__(self, aggregation_method: str = "geometric", alpha: float = 1.0, beta: float = 1.0):
        self.aggregation_method = aggregation_method
        self.alpha = alpha
        self.beta = beta
        
        self.imputer = LeakageSafeImputer(strategy="median")
        self.scaler = ModalityBlockScaler()
        self.observability_engine = ContextObservabilityEngine()
        self.outlier_detector = RobustOutlierDetector(threshold=3.5)
        
        self.global_reliability: Dict[str, float] = {}
        self.quality_dimensions: Dict[str, Dict[str, Any]] = {}
        self.is_fitted: bool = False

        self.modality_feature_map = {
            "modality_a": MODALITY_A_FEATURES,
            "modality_b": MODALITY_B_FEATURES,
            "modality_c": MODALITY_C_FEATURES,
        }

    def fit(self, train_df: pd.DataFrame, save_artifacts: bool = True) -> "AMDFEPipelineV2":
        """
        Fits all statistical parameters on the training partition only.
        """
        train_clean = train_df.copy()
        for col in LEAKAGE_COLUMNS:
            if col in train_clean.columns:
                train_clean = train_clean.drop(columns=[col])

        # 1. Evaluate Source Reliability on Train
        self.quality_dimensions = evaluate_modality_quality_dimensions(train_clean)
        self.global_reliability = compute_v2_source_reliability(
            train_clean, aggregation_method=self.aggregation_method
        )
        
        if save_artifacts:
            evaluate_v2_reliability_sensitivity(train_clean, save_outputs=True)
            # Save quality dimensions table
            AMDFE_V2_QUALITY_DIR.mkdir(parents=True, exist_ok=True)
            qual_records = []
            for m, d in self.quality_dimensions.items():
                qual_records.append({
                    "Modality": m,
                    "Completeness_C": d.get("Completeness_C"),
                    "Validity_V": d.get("Validity_V"),
                    "Duplicate_D": d.get("Duplicate_D"),
                    "Outlier_Quality_O": d.get("Outlier_Quality_O"),
                    "Temporal_Coverage_T": d.get("Temporal_Coverage_T"),
                    "Spatial_Coverage_S": d.get("Spatial_Coverage_S"),
                    "Measurement_Quality_M": d.get("Measurement_Quality_M"),
                    "Applicable_Dimensions_Count": len(d.get("applicable_dimensions", [])),
                })
            pd.DataFrame(qual_records).to_csv(AMDFE_V2_QUALITY_DIR / "modality_quality_v2.csv", index=False)

        # 2. Fit Observability Quantile Boundaries on Train
        self.observability_engine.fit(train_clean)

        # 3. Fit Robust Outlier Detector on Train
        self.outlier_detector.fit(train_clean, feature_cols=["Previous_WatLevel"])

        # 4. Fit Imputer on Train
        all_features = MODALITY_A_FEATURES + MODALITY_B_FEATURES + MODALITY_C_FEATURES
        self.imputer.fit(train_clean, all_features)

        # 5. Fit Scaler on Imputed Train
        train_imputed = self.imputer.transform(train_clean, all_features)
        self.scaler.fit(train_imputed, self.modality_feature_map)

        if save_artifacts:
            audit_missingness_and_imputation(
                train_clean, test_df=None, imputer=self.imputer, save_reports=True
            )

        self.is_fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        config_key: str = "A5",
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Transforms any partition using frozen training statistics.
        """
        if not self.is_fitted:
            raise RuntimeError("AMDFEPipelineV2 must be fitted on training data before calling transform!")

        data = df.copy()
        for col in LEAKAGE_COLUMNS:
            if col in data.columns:
                data = data.drop(columns=[col])

        if config_key in ABLATION_CONFIGS_V21:
            cfg = ABLATION_CONFIGS_V21[config_key]
        elif config_key in ABLATION_CONFIGS_V2:
            cfg = ABLATION_CONFIGS_V2[config_key]
        else:
            cfg = ABLATION_CONFIGS_V21.get("C2", ABLATION_CONFIGS_V2["A5"])

        active_modalities = cfg["modalities"]
        fusion_type = cfg["fusion_type"]

        a_param = alpha if alpha is not None else self.alpha
        b_param = beta if beta is not None else self.beta

        # 1. Observability Factors & Cohorts
        obs_df = self.observability_engine.transform(data)
        for col in obs_df.columns:
            data[col] = obs_df[col]

        # 2. Robust Outlier Indicators
        out_df = self.outlier_detector.transform(data, feature_cols=["Previous_WatLevel"])
        for col in out_df.columns:
            data[col] = out_df[col]

        # 3. Two-Stage Adaptive Weights
        weights_df, fallback_flag = compute_v2_adaptive_weights(
            reliability_scores=self.global_reliability,
            observability_df=obs_df,
            active_modalities=active_modalities,
            alpha=a_param,
            beta=b_param,
        )
        for col in weights_df.columns:
            data[col] = weights_df[col]

        # 4. Impute Missing Values with Frozen Statistics & Add Indicators
        all_features = MODALITY_A_FEATURES + MODALITY_B_FEATURES + MODALITY_C_FEATURES
        data_imputed = self.imputer.transform(data, all_features)

        # 5. Standardize Features with Frozen Scalers
        data_std = self.scaler.transform(data_imputed, self.modality_feature_map)

        # 6. Apply Fusion Transformation
        data_fused = fuse_v2_modality_blocks(
            data_std,
            weights_df=weights_df,
            reliability_scores=self.global_reliability,
            active_modalities=active_modalities,
            fusion_type=fusion_type,
        )

        return data_fused

    def fit_transform(self, train_df: pd.DataFrame, config_key: str = "A5") -> pd.DataFrame:
        return self.fit(train_df, save_artifacts=False).transform(train_df, config_key=config_key)


def generate_all_v2_fused_datasets(save_to_disk: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Generates and saves canonical A0 - A6 fused datasets to data/processed/amdfe_v2/fused/.
    """
    from ml.amdfe.ingestion import ingest_amdfe_dataset

    print("[AMDFE v2] Ingesting multimodal dataset...")
    df_raw = ingest_amdfe_dataset(enforce_strict_leakage_check=True, save_audit=True)

    pipeline = AMDFEPipelineV2(aggregation_method="geometric")
    print("[AMDFE v2] Fitting pipeline on full reference dataset...")
    pipeline.fit(df_raw, save_artifacts=True)

    config_filename_map = {
        "A0": "groundwater_only_a0.csv",
        "A1": "groundwater_weather_a1.csv",
        "A2": "unweighted_multimodal_a2.csv",
        "A3": "reliability_meta_augmented_a3.csv",
        "A4": "global_reliability_scaled_a4.csv",
        "A5": "context_adaptive_amdfe_a5.csv",
        "A6": "amdfe_robustness_metadata_a6.csv",
    }

    fused_datasets = {}
    if save_to_disk:
        AMDFE_V2_FUSED_DIR.mkdir(parents=True, exist_ok=True)

    for cfg_key, filename in config_filename_map.items():
        print(f"[AMDFE v2] Generating fused dataset for {cfg_key} ({ABLATION_CONFIGS_V2[cfg_key]['name']})...")
        fused = pipeline.transform(df_raw, config_key=cfg_key)

        assert "TIFF_Value" not in fused.columns, "TIFF_Value leakage detected!"
        assert "tiff_value" not in fused.columns, "tiff_value leakage detected!"

        fused_datasets[cfg_key] = fused
        if save_to_disk:
            out_path = AMDFE_V2_FUSED_DIR / filename
            fused.to_csv(out_path, index=False)
            print(f"  -> Saved {cfg_key} to: {out_path} ({len(fused)} rows, {len(fused.columns)} cols)")

    return fused_datasets


def generate_all_v21_fused_datasets(save_to_disk: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Generates and saves canonical C0, C1, C2, A0, A1, A6 fused datasets to data/processed/amdfe_v21/fused/.
    """
    from ml.amdfe.config import AMDFE_V21_FUSED_DIR, AMDFE_V21_QUALITY_DIR, AMDFE_V21_RELIABILITY_DIR, AMDFE_V21_OBSERVABILITY_DIR
    from ml.amdfe.ingestion import ingest_amdfe_dataset

    print("[AMDFE v2.1] Ingesting multimodal dataset...")
    df_raw = ingest_amdfe_dataset(enforce_strict_leakage_check=True, save_audit=True)

    pipeline = AMDFEPipelineV2(aggregation_method="geometric")
    print("[AMDFE v2.1] Fitting pipeline on full reference dataset...")
    pipeline.fit(df_raw, save_artifacts=False)

    if save_to_disk:
        AMDFE_V21_QUALITY_DIR.mkdir(parents=True, exist_ok=True)
        AMDFE_V21_RELIABILITY_DIR.mkdir(parents=True, exist_ok=True)
        AMDFE_V21_OBSERVABILITY_DIR.mkdir(parents=True, exist_ok=True)
        AMDFE_V21_FUSED_DIR.mkdir(parents=True, exist_ok=True)

        # Save quality dimensions
        qual_records = []
        for m, d in pipeline.quality_dimensions.items():
            qual_records.append({
                "Modality": m,
                "Completeness_C": d.get("Completeness_C"),
                "Validity_V": d.get("Validity_V"),
                "Duplicate_D": d.get("Duplicate_D"),
                "Outlier_Quality_O": d.get("Outlier_Quality_O"),
                "Temporal_Coverage_T": d.get("Temporal_Coverage_T"),
                "Spatial_Coverage_S": d.get("Spatial_Coverage_S"),
                "Measurement_Quality_M": d.get("Measurement_Quality_M"),
                "Applicable_Dimensions_Count": len(d.get("applicable_dimensions", [])),
            })
        pd.DataFrame(qual_records).to_csv(AMDFE_V21_QUALITY_DIR / "modality_quality_v21.csv", index=False)

        # Save reliability scores
        rel_records = [{"Modality": m, "Reliability_Score": r} for m, r in pipeline.global_reliability.items()]
        pd.DataFrame(rel_records).to_csv(AMDFE_V21_RELIABILITY_DIR / "source_reliability_v21.csv", index=False)

    config_filename_map = {
        "C0": "c0_unweighted_base.csv",
        "C1": "c1_base_plus_metadata_ungated.csv",
        "C2": "c2_core_amdfe_gated.csv",
        "A0": "a0_groundwater_only.csv",
        "A1": "a1_groundwater_weather.csv",
        "A6": "a6_amdfe_robustness_extended.csv",
    }

    fused_datasets = {}
    for cfg_key, filename in config_filename_map.items():
        print(f"[AMDFE v2.1] Generating fused dataset for {cfg_key} ({ABLATION_CONFIGS_V21[cfg_key]['name']})...")
        fused = pipeline.transform(df_raw, config_key=cfg_key)

        assert "TIFF_Value" not in fused.columns, "TIFF_Value leakage detected!"
        assert "tiff_value" not in fused.columns, "tiff_value leakage detected!"

        fused_datasets[cfg_key] = fused
        if save_to_disk:
            out_path = AMDFE_V21_FUSED_DIR / filename
            fused.to_csv(out_path, index=False)
            print(f"  -> Saved {cfg_key} to: {out_path} ({len(fused)} rows, {len(fused.columns)} cols)")

    return fused_datasets

