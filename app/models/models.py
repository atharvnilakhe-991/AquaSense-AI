from sqlalchemy import Column, Integer, String, Float, Date, Numeric, DateTime
from app.database.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False, index=True)
    password = Column(String, nullable=False)
    role = Column(String, default="user")


class Well(Base):
    __tablename__ = "wells"

    id = Column(Integer, primary_key=True, index=True)

    well_name = Column(
        "well_id",
        String(50),
        nullable=True
    )

    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

class GroundwaterData(Base):
    __tablename__ = "groundwater_data"

    id = Column(Integer, primary_key=True, index=True)

    well_id = Column(Integer, nullable=False)

    observation_date = Column(Date, nullable=False)

    groundwater_level = Column(Numeric(10, 3), nullable=True)

    latitude = Column(Numeric(10, 7), nullable=True)

    longitude = Column(Numeric(10, 7), nullable=True)

class WeatherData(Base):
    __tablename__ = "weather_data"

    id = Column(Integer, primary_key=True, index=True)

    observation_date = Column(Date, nullable=False)

    latitude = Column(Numeric(10, 7), nullable=True)

    longitude = Column(Numeric(10, 7), nullable=True)

    temperature = Column(Numeric(10, 3), nullable=True)

    precipitation = Column(Numeric(10, 3), nullable=True)

    humidity = Column(Numeric(10, 3), nullable=True)

    solar_radiation = Column(Numeric(10, 3), nullable=True)

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)

    well_id = Column(Integer, nullable=False)

    prediction_date = Column(Date, nullable=False)

    predicted_groundwater = Column(
        Numeric(10, 3),
        nullable=True
    )

    recharge_estimate = Column(
        Numeric(10, 3),
        nullable=True
    )

    risk_level = Column(
        String,
        nullable=True
    )

    model_name = Column(
        String,
        nullable=True
    )

class SpatialData(Base):
    __tablename__ = "spatial_data"

    id = Column(Integer, primary_key=True, index=True)

    dataset_name = Column(String, nullable=True)

    dataset_type = Column(String, nullable=True)

    acquisition_date = Column(Date, nullable=True)

    file_path = Column(String, nullable=True)

    latitude = Column(Numeric(10, 7), nullable=True)

    longitude = Column(Numeric(10, 7), nullable=True)

    description = Column(String, nullable=True)

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)

    well_id = Column(Integer, nullable=False)

    alert_type = Column(String, nullable=True)

    message = Column(String, nullable=True)

    severity = Column(String, nullable=True)

    status = Column(String, nullable=True)

    created_at = Column(DateTime, nullable=True)