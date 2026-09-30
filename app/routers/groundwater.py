from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.models.models import GroundwaterData
from app.schemas.user import GroundwaterDataCreate, GroundwaterDataResponse


router = APIRouter(
    prefix="/groundwater",
    tags=["Groundwater"]
)


@router.post("/", response_model=GroundwaterDataResponse)
def create_groundwater_data(
    data: GroundwaterDataCreate,
    db: Session = Depends(get_db)
):
    new_data = GroundwaterData(
        well_id=data.well_id,
        observation_date=data.observation_date,
        groundwater_level=data.groundwater_level,
        latitude=data.latitude,
        longitude=data.longitude
    )

    db.add(new_data)
    db.commit()
    db.refresh(new_data)

    return new_data


@router.get("/", response_model=list[GroundwaterDataResponse])
def get_groundwater_data(db: Session = Depends(get_db)):
    groundwater_data = db.query(GroundwaterData).all()

    return groundwater_data