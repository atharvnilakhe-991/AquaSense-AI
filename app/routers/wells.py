from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.models.models import Well
from app.schemas.user import WellCreate, WellResponse
from app.auth.dependencies import get_current_user


router = APIRouter(
    prefix="/wells",
    tags=["Wells"]
)


@router.post("/", response_model=WellResponse)
def create_well(well: WellCreate, db: Session = Depends(get_db)):

    new_well = Well(
        well_name=well.well_name,
        latitude=well.latitude,
        longitude=well.longitude
    )

    db.add(new_well)
    db.commit()
    db.refresh(new_well)

    return new_well

@router.get("/", response_model=list[WellResponse])
def get_wells(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):

    wells = db.query(Well).all()

    return wells