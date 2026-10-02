from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from urllib.parse import quote_plus
import os
from dotenv import load_dotenv

load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()

DATABASE_USER = os.getenv("DATABASE_USER")
raw_password = os.getenv("DATABASE_PASSWORD")
DATABASE_HOST = os.getenv("DATABASE_HOST")
DATABASE_PORT = os.getenv("DATABASE_PORT")
DATABASE_NAME = os.getenv("DATABASE_NAME")

db_vars = {
    "DATABASE_USER": DATABASE_USER,
    "DATABASE_PASSWORD": raw_password,
    "DATABASE_HOST": DATABASE_HOST,
    "DATABASE_PORT": DATABASE_PORT,
    "DATABASE_NAME": DATABASE_NAME,
}
missing_vars = [k for k, v in db_vars.items() if not v]

custom_url = os.getenv("DATABASE_URL")

if not missing_vars:
    DATABASE_PASSWORD = quote_plus(raw_password)
    DATABASE_URL = (
        f"postgresql+psycopg2://"
        f"{DATABASE_USER}:{DATABASE_PASSWORD}@"
        f"{DATABASE_HOST}:{DATABASE_PORT}/"
        f"{DATABASE_NAME}"
    )
    connect_args = {}
elif custom_url and not custom_url.startswith("sqlite"):
    DATABASE_URL = custom_url
    connect_args = {}
elif ENVIRONMENT == "production":
    raise RuntimeError(
        "Production database configuration error: SQLite fallback is strictly prohibited in production. "
        f"Missing required PostgreSQL configuration ({', '.join(missing_vars)})."
    )
else:
    # Development mode: allow SQLite fallback
    DATABASE_URL = custom_url if custom_url else "sqlite:///./aquasense_dev.db"
    connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
    if len(missing_vars) < len(db_vars) and not custom_url:
        print(
            f"[AquaSense Database] Notice: Incomplete PostgreSQL configuration ({', '.join(missing_vars)}). "
            f"Using local development SQLite ({DATABASE_URL})."
        )

engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()