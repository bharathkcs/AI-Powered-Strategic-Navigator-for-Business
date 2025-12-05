"""Pydantic schemas for API contracts"""

from app.schemas.service import (
    ServiceRecordBase,
    ServiceRecordCreate,
    ServiceRecordResponse,
    ServiceUploadResponse
)
from app.schemas.forecast import (
    ForecastRequest,
    ForecastResponse,
    ForecastDataPoint
)
from app.schemas.analytics import (
    AnalyticsSummary,
    LocationPerformance,
    FranchisePerformance
)

__all__ = [
    "ServiceRecordBase",
    "ServiceRecordCreate",
    "ServiceRecordResponse",
    "ServiceUploadResponse",
    "ForecastRequest",
    "ForecastResponse",
    "ForecastDataPoint",
    "AnalyticsSummary",
    "LocationPerformance",
    "FranchisePerformance"
]
