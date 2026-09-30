from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.models.models import SpatialData
from app.schemas.user import SpatialDataCreate, SpatialDataResponse


router = APIRouter(
    prefix="/spatial",
    tags=["Spatial Data"]
)


@router.post("/", response_model=SpatialDataResponse)
def create_spatial_data(
    data: SpatialDataCreate,
    db: Session = Depends(get_db)
):
    new_data = SpatialData(
        dataset_name=data.dataset_name,
        dataset_type=data.dataset_type,
        acquisition_date=data.acquisition_date,
        file_path=data.file_path,
        latitude=data.latitude,
        longitude=data.longitude,
        description=data.description
    )

    db.add(new_data)
    db.commit()
    db.refresh(new_data)

    return new_data


@router.get("/", response_model=list[SpatialDataResponse])
def get_spatial_data(db: Session = Depends(get_db)):
    spatial_data = db.query(SpatialData).all()

    return spatial_data