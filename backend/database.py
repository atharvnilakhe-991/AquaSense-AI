"""
Data Repository & Model Cache Layer
Member 3 - Backend & Database

Manages the preloaded Phelps County hydrogeological database,
AMDFE pipeline state, and trained ML predictor.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from amdf.fusion import AdaptiveMultimodalDataFusionEngine
from ml.dataset import load_or_generate_phelps_dataset, FEATURE_COLUMNS, TARGET_COLUMN
from ml.models import AquaSenseModelTrainer
from ml.recharge import estimate_recharge_potential
from ml.risk import calculate_depletion_velocity, assess_well_risk
from ml.explainability import ShapExplainerService


class AquaDatabase:
    """
    In-memory hydrogeological repository and inference service.
    Can be connected directly to PostgreSQL/Supabase when configured.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.raw_df: Optional[pd.DataFrame] = None
        self.fused_df: Optional[pd.DataFrame] = None
        self.amdfe = AdaptiveMultimodalDataFusionEngine()
        self.trainer = AquaSenseModelTrainer(model_name="LightGBM")
        self.explainer_service: Optional[ShapExplainerService] = None
        
        self.wells_cache: Dict[str, Dict[str, Any]] = {}
        self.is_initialized = False

    def initialize(self):
        """Loads data, runs AMDFE fusion, trains ML model, and caches predictions."""
        if self.is_initialized:
            return

        # 1. Ingest dataset (170 wells, 3,844 rows)
        self.raw_df = load_or_generate_phelps_dataset(data_dir=self.data_dir)
        
        # 2. AMDFE Fusion (Quality + Causal Lags + Reliability)
        self.amdfe.fit_priors(self.raw_df)
        self.fused_df = self.amdfe.process_and_fuse(self.raw_df)
        
        # 3. Train Model
        X = self.fused_df[FEATURE_COLUMNS].values
        y = self.fused_df[TARGET_COLUMN].values
        self.trainer.fit(X, y, feature_names=FEATURE_COLUMNS)
        
        # 4. Initialize XAI SHAP Explainer
        self.explainer_service = ShapExplainerService(self.trainer.model, FEATURE_COLUMNS)
        
        # 5. Build Well Cache (Latest prediction, history, and metrics per well)
        self._build_wells_cache()
        self.is_initialized = True

    def _build_wells_cache(self):
        grouped = self.fused_df.groupby("well_id")
        
        for well_id, group in grouped:
            sorted_g = group.sort_values("year")
            latest = sorted_g.iloc[-1]
            
            # Predict level using trained model
            X_latest = latest[FEATURE_COLUMNS].values.reshape(1, -1)
            pred_level = float(self.trainer.predict(X_latest)[0])
            obs_level = float(latest["observed_level"])
            
            # Historical trajectory & slope
            obs_history = sorted_g["observed_level"].tolist()
            years = sorted_g["year"].tolist()
            depletion_vel = calculate_depletion_velocity(obs_history, years)
            
            # Recharge potential
            recharge_info = estimate_recharge_potential(
                annual_precip_mm=float(latest["annual_precipitation_mm"]),
                surface_elevation_m=float(latest["surface_elevation_m"]),
                relative_humidity_pct=float(latest["annual_relative_humidity_pct"]),
            )
            
            # Risk & Monitoring Priority
            rel_score = float(latest.get("composite_reliability", 0.85))
            risk_info = assess_well_risk(
                predicted_level=pred_level,
                baseline_level=obs_history[0],
                depletion_velocity=depletion_vel,
                reliability_score=rel_score,
                recharge_potential=recharge_info["recharge_potential"]
            )
            
            # History list
            hist_records = []
            for _, row in sorted_g.iterrows():
                hist_records.append({
                    "year": int(row["year"]),
                    "observed_level": round(float(row["observed_level"]), 2),
                    "annual_precipitation_mm": round(float(row["annual_precipitation_mm"]), 1),
                    "annual_temperature_c": round(float(row["annual_temperature_c"]), 1)
                })

            self.wells_cache[well_id] = {
                "well_id": well_id,
                "latitude": float(latest["latitude"]),
                "longitude": float(latest["longitude"]),
                "surface_elevation_m": float(latest["surface_elevation_m"]),
                "observed_level": round(obs_level, 2),
                "predicted_level": round(pred_level, 2),
                "risk_level": risk_info["risk_level"],
                "recharge_potential": recharge_info["recharge_potential"],
                "confidence": risk_info["confidence_score"],
                "depletion_velocity_ft_yr": risk_info["depletion_velocity_ft_per_year"],
                "monitoring_priority_score": risk_info["monitoring_priority_score"],
                "high_uncertainty_warning": risk_info["high_uncertainty_warning"],
                "source_reliability_score": round(rel_score, 3),
                "recharge_score": recharge_info["recharge_score"],
                "year": int(latest["year"]),
                "history": hist_records,
                "latest_features": latest[FEATURE_COLUMNS].to_dict(),
                "modality_weights": {
                    "groundwater_history": round(float(latest.get("groundwater_history_weight", 0.45)), 3),
                    "meteorological": round(float(latest.get("meteorological_weight", 0.35)), 3),
                    "topographic": round(float(latest.get("topographic_weight", 0.20)), 3),
                }
            }

    def get_all_wells(self) -> List[Dict[str, Any]]:
        if not self.is_initialized:
            self.initialize()
        return list(self.wells_cache.values())

    def get_well(self, well_id: str) -> Optional[Dict[str, Any]]:
        if not self.is_initialized:
            self.initialize()
        return self.wells_cache.get(well_id)


# Global singleton instance
db = AquaDatabase()
