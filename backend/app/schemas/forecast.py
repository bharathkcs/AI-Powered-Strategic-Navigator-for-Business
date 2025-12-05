"""
Pydantic schemas for forecasting
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import date


class ForecastDataPoint(BaseModel):
    """Single forecast data point"""
    date: date
    value: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None


class ForecastRequest(BaseModel):
    """Request schema for generating forecasts"""
    forecast_type: Literal["service_volume", "parts_demand", "revenue", "warranty"] = Field(
        ...,
        description="Type of forecast to generate"
    )
    period: Literal[30, 60, 90] = Field(
        default=90,
        description="Forecast period in days"
    )
    model_type: Literal["gradient_boosting", "random_forest"] = Field(
        default="gradient_boosting",
        description="ML model to use"
    )
    location: Optional[str] = Field(
        None,
        description="Specific location (optional)"
    )
    franchise_id: Optional[str] = Field(
        None,
        description="Specific franchise (optional)"
    )


class ModelMetrics(BaseModel):
    """Model performance metrics"""
    mae: float = Field(..., description="Mean Absolute Error")
    rmse: float = Field(..., description="Root Mean Squared Error")
    mape: float = Field(..., description="Mean Absolute Percentage Error")
    r2: float = Field(..., description="R-squared score")


class ForecastResponse(BaseModel):
    """Response schema for forecasts"""
    forecast_id: int
    forecast_type: str
    period: int
    model_type: str
    forecast_data: List[ForecastDataPoint]
    model_metrics: ModelMetrics
    insights: Optional[str] = None
    created_at: date

    class Config:
        from_attributes = True
