from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.models.models import Prediction
from app.schemas.user import PredictionCreate, PredictionResponse


router = APIRouter(
    prefix="/predictions",
    tags=["Predictions"]
)


@router.post("/", response_model=PredictionResponse)
def create_prediction(
    data: PredictionCreate,
    db: Session = Depends(get_db)
):
    new_prediction = Prediction(
        well_id=data.well_id,
        prediction_date=data.prediction_date,
        predicted_groundwater=data.predicted_groundwater,
        recharge_estimate=data.recharge_estimate,
        risk_level=data.risk_level,
        model_name=data.model_name
    )

    db.add(new_prediction)
    db.commit()
    db.refresh(new_prediction)

    return new_prediction


@router.get("/", response_model=list[PredictionResponse])
def get_predictions(db: Session = Depends(get_db)):
    predictions = db.query(Prediction).all()

    return predictions