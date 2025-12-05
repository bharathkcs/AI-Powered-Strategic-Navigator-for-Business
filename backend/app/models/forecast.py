"""
Database models for forecasts
"""

from sqlalchemy import Column, Integer, String, Float, Date, DateTime, JSON
from sqlalchemy.sql import func
from app.database import Base


class Forecast(Base):
    """Forecast results model"""
    __tablename__ = "forecasts"

    id = Column(Integer, primary_key=True, index=True)

    # Forecast metadata
    forecast_type = Column(String, index=True)  # service_volume, parts_demand, revenue, warranty
    period = Column(Integer)  # 30, 60, or 90 days
    created_date = Column(Date, index=True)

    # Forecast data (JSON array of predictions)
    forecast_data = Column(JSON)  # [{date: "2024-01-01", value: 150}, ...]

    # Model information
    model_type = Column(String)  # gradient_boosting, random_forest
    model_metrics = Column(JSON)  # {mae: 10.5, rmse: 15.2, mape: 5.3, r2: 0.85}

    # Location-specific (optional)
    location = Column(String, nullable=True, index=True)
    franchise_id = Column(String, nullable=True, index=True)

    # AI insights
    insights = Column(String)  # AI-generated insights

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<Forecast {self.forecast_type} - {self.period}d>"
