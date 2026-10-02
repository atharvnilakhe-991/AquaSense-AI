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

from fastapi.middleware.cors import CORSMiddleware
from app.routers.dashboard_api import router as dashboard_router

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AquaSense AI API",
    description="Backend API for AquaSense AI Hyperlocal Groundwater Intelligence Platform",
    version="1.0.0"
)

import os
from dotenv import load_dotenv

load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()

if ENVIRONMENT == "production":
    frontend_origin_env = os.getenv("FRONTEND_ORIGIN")
    if not frontend_origin_env:
        raise RuntimeError(
            "Production CORS configuration error: FRONTEND_ORIGIN must be explicitly configured in production."
        )
    cors_origins = [origin.strip() for origin in frontend_origin_env.split(",") if origin.strip()]
else:
    cors_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://testserver",
    ]
    frontend_origin_env = os.getenv("FRONTEND_ORIGIN")
    if frontend_origin_env:
        for origin in frontend_origin_env.split(","):
            if origin.strip() and origin.strip() not in cors_origins:
                cors_origins.append(origin.strip())

# Configure CORS for frontend clients (development localhosts or explicit production origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Dashboard & Scientific Integration APIs
app.include_router(dashboard_router)

# Mount Database Entities Routers
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