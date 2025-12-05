"""
Forecasting API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import pandas as pd
from datetime import date
from typing import List

from app.database import get_db
from app.schemas.forecast import ForecastRequest, ForecastResponse, ModelMetrics, ForecastDataPoint
from app.models import ServiceRecord, Forecast
from app.ml.demand_forecaster import DemandForecaster
from app.utils.ai_insights import generate_forecast_insights

router = APIRouter()


@router.post("/generate", response_model=ForecastResponse)
async def generate_forecast(
    request: ForecastRequest,
    db: Session = Depends(get_db)
):
    """
    Generate demand forecast for specified period (30/60/90 days)

    - **forecast_type**: service_volume, parts_demand, revenue, or warranty
    - **period**: 30, 60, or 90 days
    - **model_type**: gradient_boosting or random_forest
    - **location**: Optional location filter
    - **franchise_id**: Optional franchise filter
    """
    try:
        # Fetch historical data
        query = db.query(ServiceRecord)

        # Apply filters
        if request.location:
            query = query.filter(ServiceRecord.location == request.location)
        if request.franchise_id:
            query = query.filter(ServiceRecord.franchise_id == request.franchise_id)

        records = query.all()

        if len(records) < 30:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Insufficient data: need at least 30 records, found {len(records)}"
            )

        # Prepare data for forecasting
        df = pd.DataFrame([
            {
                'date': r.service_date,
                'value': getattr(r, _get_target_column(request.forecast_type))
            }
            for r in records
        ])

        # Aggregate by date
        df_daily = df.groupby('date')['value'].sum().reset_index()

        # Generate forecast
        forecaster = DemandForecaster()
        forecast_result = forecaster.forecast_demand(
            data=df_daily,
            forecast_type=request.forecast_type,
            periods=request.period,
            model_type=request.model_type
        )

        # Generate AI insights
        insights = await generate_forecast_insights(
            forecast_type=request.forecast_type,
            forecast_data=forecast_result['forecast_data'],
            metrics=forecast_result['model_metrics'],
            period=request.period
        )

        # Save forecast to database
        forecast_db = Forecast(
            forecast_type=request.forecast_type,
            period=request.period,
            created_date=date.today(),
            forecast_data=forecast_result['forecast_data'],
            model_type=request.model_type,
            model_metrics=forecast_result['model_metrics'],
            location=request.location,
            franchise_id=request.franchise_id,
            insights=insights
        )
        db.add(forecast_db)
        db.commit()
        db.refresh(forecast_db)

        # Format response
        return ForecastResponse(
            forecast_id=forecast_db.id,
            forecast_type=forecast_db.forecast_type,
            period=forecast_db.period,
            model_type=forecast_db.model_type,
            forecast_data=[
                ForecastDataPoint(**dp) for dp in forecast_db.forecast_data
            ],
            model_metrics=ModelMetrics(**forecast_db.model_metrics),
            insights=forecast_db.insights,
            created_at=forecast_db.created_date
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forecasting error: {str(e)}"
        )


@router.get("/history", response_model=List[ForecastResponse])
async def get_forecast_history(
    forecast_type: str = None,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get historical forecasts

    - **forecast_type**: Optional filter by type
    - **limit**: Maximum number of forecasts to return (default: 10)
    """
    query = db.query(Forecast).order_by(Forecast.created_at.desc())

    if forecast_type:
        query = query.filter(Forecast.forecast_type == forecast_type)

    forecasts = query.limit(limit).all()

    return [
        ForecastResponse(
            forecast_id=f.id,
            forecast_type=f.forecast_type,
            period=f.period,
            model_type=f.model_type,
            forecast_data=[ForecastDataPoint(**dp) for dp in f.forecast_data],
            model_metrics=ModelMetrics(**f.model_metrics),
            insights=f.insights,
            created_at=f.created_date
        )
        for f in forecasts
    ]


def _get_target_column(forecast_type: str) -> str:
    """Map forecast type to database column"""
    mapping = {
        "service_volume": "service_id",  # Will count
        "parts_demand": "parts_used",
        "revenue": "total_revenue",
        "warranty": "warranty_claim"
    }
    return mapping.get(forecast_type, "service_revenue")
