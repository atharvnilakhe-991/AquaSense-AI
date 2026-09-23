"""
Explainable AI (XAI) Endpoints
Member 4 / Member 3 - XAI & Backend Integration
"""

from fastapi import APIRouter, HTTPException
import numpy as np
from backend.database import db
from backend.schemas import GlobalXaiResponse, WellXaiResponse, ShapFeatureContribution
from ml.dataset import FEATURE_COLUMNS

router = APIRouter(prefix="/api/v1/xai", tags=["XAI"])


@router.get("/global", response_model=GlobalXaiResponse)
def get_global_xai_importance():
    """
    Returns global feature importance of predictors using SHAP / LightGBM split-gain importance.
    """
    if not db.is_initialized:
        db.initialize()
        
    X_sample = db.fused_df[FEATURE_COLUMNS].values[:200]
    importances = db.explainer_service.explain_global(X_sample)
    
    return GlobalXaiResponse(
        model_name=db.trainer.model_name,
        feature_importances=importances
    )


@router.get("/local/{well_id}", response_model=WellXaiResponse)
def get_local_well_explanation(well_id: str):
    """
    Returns local per-well SHAP attribution breakdown (force/waterfall values)
    explaining why the groundwater level was predicted at this depth.
    """
    well = db.get_well(well_id)
    if not well:
        raise HTTPException(status_code=404, detail=f"Well '{well_id}' not found.")
        
    features_dict = well["latest_features"]
    x_single = np.array([features_dict[f] for f in FEATURE_COLUMNS])
    
    explanations = db.explainer_service.explain_local_instance(x_single, base_value=120.0)
    
    contrib_objs = [
        ShapFeatureContribution(
            feature=e["feature"],
            value=e["value"],
            attribution=e["attribution"],
            direction=e["direction"]
        )
        for e in explanations
    ]
    
    return WellXaiResponse(
        well_id=well_id,
        predicted_level=well["predicted_level"],
        base_value=120.0,
        contributions=contrib_objs
    )
