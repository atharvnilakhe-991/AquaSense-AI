from pydantic import BaseModel, EmailStr
from datetime import date, datetime

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: str = "user"


class UserResponse(BaseModel):
    id: int
    name: str
    email: EmailStr
    role: str

    class Config:
        from_attributes = True

class WellCreate(BaseModel):
    well_name: str
    latitude: float
    longitude: float


class WellResponse(BaseModel):
    id: int
    well_name: str
    latitude: float
    longitude: float

    class Config:
        from_attributes = True

class GroundwaterDataCreate(BaseModel):
    well_id: int
    observation_date: date
    groundwater_level: float | None = None
    latitude: float | None = None
    longitude: float | None = None


class GroundwaterDataResponse(BaseModel):
    id: int
    well_id: int
    observation_date: date
    groundwater_level: float | None = None
    latitude: float | None = None
    longitude: float | None = None

    class Config:
        from_attributes = True

class WeatherDataCreate(BaseModel):
    observation_date: date
    latitude: float | None = None
    longitude: float | None = None
    temperature: float | None = None
    precipitation: float | None = None
    humidity: float | None = None
    solar_radiation: float | None = None


class WeatherDataResponse(BaseModel):
    id: int
    observation_date: date
    latitude: float | None = None
    longitude: float | None = None
    temperature: float | None = None
    precipitation: float | None = None
    humidity: float | None = None
    solar_radiation: float | None = None

    class Config:
        from_attributes = True

class PredictionCreate(BaseModel):
    well_id: int
    prediction_date: date
    predicted_groundwater: float | None = None
    recharge_estimate: float | None = None
    risk_level: str | None = None
    model_name: str | None = None


class PredictionResponse(BaseModel):
    id: int
    well_id: int
    prediction_date: date
    predicted_groundwater: float | None = None
    recharge_estimate: float | None = None
    risk_level: str | None = None
    model_name: str | None = None

    class Config:
        from_attributes = True

class SpatialDataCreate(BaseModel):
    dataset_name: str | None = None
    dataset_type: str | None = None
    acquisition_date: date | None = None
    file_path: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    description: str | None = None


class SpatialDataResponse(BaseModel):
    id: int
    dataset_name: str | None = None
    dataset_type: str | None = None
    acquisition_date: date | None = None
    file_path: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    description: str | None = None

    class Config:
        from_attributes = True

class AlertCreate(BaseModel):
    well_id: int
    alert_type: str | None = None
    message: str | None = None
    severity: str | None = None
    status: str | None = None
    created_at: datetime | None = None


class AlertResponse(BaseModel):
    id: int
    well_id: int
    alert_type: str | None = None
    message: str | None = None
    severity: str | None = None
    status: str | None = None
    created_at: datetime | None = None

    class Config:
        from_attributes = True

class LoginRequest(BaseModel):
    email: EmailStr
    password: str