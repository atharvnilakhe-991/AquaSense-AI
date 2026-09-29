from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.models.models import WeatherData
from app.schemas.user import WeatherDataCreate, WeatherDataResponse


router = APIRouter(
    prefix="/weather",
    tags=["Weather"]
)


@router.post("/", response_model=WeatherDataResponse)
def create_weather_data(
    data: WeatherDataCreate,
    db: Session = Depends(get_db)
):
    new_data = WeatherData(
        observation_date=data.observation_date,
        latitude=data.latitude,
        longitude=data.longitude,
        temperature=data.temperature,
        precipitation=data.precipitation,
        humidity=data.humidity,
        solar_radiation=data.solar_radiation
    )

    db.add(new_data)
    db.commit()
    db.refresh(new_data)

    return new_data


@router.get("/", response_model=list[WeatherDataResponse])
def get_weather_data(db: Session = Depends(get_db)):
    weather_data = db.query(WeatherData).all()

    return weather_data