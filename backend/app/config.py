"""
Configuration management using Pydantic Settings
"""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings"""

    # Application
    APP_NAME: str = "IFB Service Intelligence"
    DEBUG: bool = True
    API_VERSION: str = "v1"

    # Database
    DATABASE_URL: str = "sqlite:///./ifb_service.db"
    # For PostgreSQL in production:
    # DATABASE_URL: str = "postgresql://user:password@localhost:5432/ifb_service"

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # Lovable AI frontend (React)
        "http://localhost:5173",  # Vite dev server
        "http://localhost:8080",  # Alternative frontend port
    ]

    # OpenAI
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4"

    # ML Models
    MODEL_PATH: str = "./app/ml/models"
    RETRAIN_SCHEDULE: str = "0 0 * * 0"  # Weekly on Sunday midnight

    # File Upload
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".csv", ".xlsx"]
    UPLOAD_DIR: str = "./data/uploads"

    # Forecasting
    DEFAULT_FORECAST_PERIODS: int = 90  # days
    MIN_TRAINING_DATA: int = 30  # minimum days of data needed

    # Security
    API_KEY: str = "your-secret-api-key-change-in-production"
    SECRET_KEY: str = "your-secret-key-for-jwt-change-in-production"

    # Pagination
    DEFAULT_PAGE_SIZE: int = 50
    MAX_PAGE_SIZE: int = 1000

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings instance
settings = Settings()
