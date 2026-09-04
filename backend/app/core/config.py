from pydantic_settings import BaseSettings
from typing import List
import os
from pathlib import Path

# Base directory paths
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
BASE_DIR = BACKEND_DIR.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "TruthLens"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    ENVIRONMENT: str = "development"
    
    # Server
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    
    # Database
    DATABASE_URL: str = f"sqlite:///{BACKEND_DIR}/truthlens.db"
    
    # Optional external credentials
    GOOGLE_FACT_CHECK_API_KEY: str = ""
    SERPAPI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    
    # Engine Settings
    DEFAULT_RETRIEVAL_MODE: str = "hybrid" # live, offline, hybrid
    MAX_CLAIMS_PER_ARTICLE: int = 6
    REQUEST_TIMEOUT_SECONDS: int = 15
    MAX_TEXT_LENGTH: int = 25000
    
    # Models directory
    MODELS_DIR: str = str(BACKEND_DIR / "models")
    DATA_DIR: str = str(BASE_DIR / "data")

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
