from fastapi import FastAPI
from app.database.database import Base, engine
from app.models import models
from app.routers.users import router as user_router
from app.routers.wells import router as well_router
from app.routers.groundwater import router as groundwater_router
from app.routers.weather import router as weather_router
from app.routers.predictions import router as prediction_router
from app.routers.spatial import router as spatial_router
from app.routers.alerts import router as alert_router

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AquaSense AI API",
    description="Backend API for AquaSense AI",
    version="1.0.0"
)

app.include_router(user_router)
app.include_router(well_router)
app.include_router(groundwater_router)
app.include_router(weather_router)
app.include_router(prediction_router)
app.include_router(spatial_router)
app.include_router(alert_router)


@app.get("/")
def home():
    return {
        "message": "AquaSense AI Backend is running"
    }