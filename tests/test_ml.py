"""
Unit Tests for Member 4: AI/ML, Unseen-Well Evaluation, Recharge & XAI
"""

import numpy as np
import pandas as pd
import pytest

from ml.dataset import generate_benchmark_phelps_dataset, FEATURE_COLUMNS, TARGET_COLUMN
from ml.models import AquaSenseModelTrainer
from ml.evaluate import evaluate_unseen_wells_group_kfold
from ml.recharge import estimate_recharge_potential, classify_recharge_tier
from ml.risk import assess_well_risk, calculate_depletion_velocity
from ml.explainability import ShapExplainerService


def test_dataset_generator_specs():
    df = generate_benchmark_phelps_dataset(seed=42)
    assert len(df) == 3844
    assert df["well_id"].nunique() == 170
    assert df["year"].min() == 2000
    assert df["year"].max() == 2024
    
    # Assert zero leaked TIFF predictors
    leaked = [c for c in df.columns if "tiff" in c.lower() or "idw" in c.lower()]
    assert len(leaked) == 0


def test_model_trainer_and_shap():
    # Synthetic quick train
    X = np.random.normal(size=(50, len(FEATURE_COLUMNS)))
    y = 120.0 + np.random.normal(scale=10.0, size=50)
    
    trainer = AquaSenseModelTrainer(model_name="Random Forest", random_state=42)
    trainer.fit(X, y, feature_names=FEATURE_COLUMNS)
    preds = trainer.predict(X[:5])
    
    assert len(preds) == 5
    assert all(np.isfinite(preds))
    
    # XAI
    explainer = ShapExplainerService(trainer.model, FEATURE_COLUMNS)
    local_exp = explainer.explain_local_instance(X[0])
    assert len(local_exp) == len(FEATURE_COLUMNS)
    assert "attribution" in local_exp[0]


def test_recharge_and_risk_modules():
    recharge = estimate_recharge_potential(
        annual_precip_mm=850.0,  # high precip
        surface_elevation_m=710.0,
        relative_humidity_pct=72.0
    )
    assert recharge["recharge_potential"] == "High"
    assert recharge["recharge_score"] >= 65.0
    
    risk = assess_well_risk(
        predicted_level=175.0,  # very deep
        baseline_level=150.0,
        depletion_velocity=1.5,  # fast drawdown
        reliability_score=0.85,
        recharge_potential="Low"
    )
    assert risk["risk_level"] == "Critical"
    assert risk["monitoring_priority_score"] > 60.0
