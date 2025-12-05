"""Analytics API endpoints - Simplified version"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database import get_db
from app.models import ServiceRecord
from app.schemas.analytics import AnalyticsSummary

router = APIRouter()

@router.get("/summary", response_model=AnalyticsSummary)
async def get_analytics_summary(db: Session = Depends(get_db)):
    """Get executive summary analytics"""
    total_calls = db.query(func.count(ServiceRecord.id)).scalar()
    total_service_rev = db.query(func.sum(ServiceRecord.service_revenue)).scalar() or 0
    total_parts_rev = db.query(func.sum(ServiceRecord.parts_revenue)).scalar() or 0
    avg_value = db.query(func.avg(ServiceRecord.service_revenue)).scalar() or 0
    warranty_count = db.query(func.count(ServiceRecord.id)).filter(
        ServiceRecord.warranty_claim == True
    ).scalar()
    avg_satisfaction = db.query(func.avg(ServiceRecord.customer_satisfaction)).scalar() or 0

    return AnalyticsSummary(
        total_service_calls=total_calls,
        total_service_revenue=float(total_service_rev),
        total_parts_revenue=float(total_parts_rev),
        avg_service_value=float(avg_value),
        warranty_claims=warranty_count,
        warranty_claim_rate=float(warranty_count / total_calls * 100) if total_calls > 0 else 0,
        avg_customer_satisfaction=float(avg_satisfaction),
        top_locations=[],
        insights=None
    )
