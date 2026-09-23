"""
AquaSense AI - Main FastAPI Application
Member 3 - Backend & Integration Layer
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.routes.wells import router as wells_router
from backend.routes.predictions import router as predictions_router
from backend.routes.xai import router as xai_router
from backend.database import db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Pre-initialize dataset, AMDFE fusion & ML model
    print("[AquaSense AI] Initializing AMDFE data fusion and pre-training ML predictor...")
    try:
        db.initialize()
        print(f"[AquaSense AI] Ready! Monitored wells cached: {len(db.wells_cache)}")
    except Exception as e:
        print(f"[AquaSense AI] Warning during eager initialization: {e}")
    yield


app = FastAPI(
    title="AquaSense AI - Groundwater Intelligence Platform",
    description="Adaptive Multimodal AI for Hyperlocal Groundwater Prediction, Recharge Assessment & Explainable Decision Support.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for dashboard interactions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Member 3 REST Routers
app.include_router(wells_router)
app.include_router(predictions_router)
app.include_router(xai_router)


@app.get("/health", tags=["System"])
def health_check():
    """System health check and pipeline status."""
    return {
        "status": "healthy",
        "service": "AquaSense AI Backend",
        "version": "1.0.0",
        "database_initialized": db.is_initialized,
        "monitored_wells_count": len(db.wells_cache) if db.is_initialized else 0
    }


# Serve Frontend static build if present
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/", include_in_schema=False)
    def serve_frontend_index():
        index_file = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "AquaSense AI API running. Open /docs for OpenAPI specs."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
