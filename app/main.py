from fastapi import FastAPI
from app.routers.users import router as user_router

app = FastAPI(
    title="AquaSense AI API",
    description="Backend API for AquaSense AI",
    version="1.0.0"
)


app.include_router(user_router)


@app.get("/")
def home():
    return {
        "message": "AquaSense AI Backend is running"
    }