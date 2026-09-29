from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.models.models import Alert
from app.schemas.user import AlertCreate, AlertResponse

router = APIRouter(prefix="/alerts", tags=["Alerts"])


@router.post("/", response_model=AlertResponse)
def create_alert(data: AlertCreate, db: Session = Depends(get_db)):
    new_alert = Alert(
        well_id=data.well_id,
        alert_type=data.alert_type,
        message=data.message,
        severity=data.severity,
        status=data.status,
        created_at=data.created_at
    )

    db.add(new_alert)
    db.commit()
    db.refresh(new_alert)

    return new_alert


@router.get("/", response_model=list[AlertResponse])
def get_alerts(db: Session = Depends(get_db)):
    alerts = db.query(Alert).all()
    return alerts