from datetime import datetime, timedelta
from jose import jwt


import os
from dotenv import load_dotenv

load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
_ENV_SECRET = os.getenv("SECRET_KEY")

if _ENV_SECRET:
    SECRET_KEY = _ENV_SECRET
elif ENVIRONMENT == "production":
    raise RuntimeError(
        "Production security configuration error: SECRET_KEY must be explicitly set in environment variables."
    )
else:
    # Explicit development fallback for local authentication testing
    SECRET_KEY = "aquasense_dev_secret_key_local_only"

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


def create_access_token(data: dict):
    to_encode = data.copy()

    expire = datetime.utcnow() + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    to_encode.update({"exp": expire})

    return jwt.encode(
        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM
    )