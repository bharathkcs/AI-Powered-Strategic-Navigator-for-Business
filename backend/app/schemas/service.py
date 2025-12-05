"""
Pydantic schemas for service records
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import date, datetime


class ServiceRecordBase(BaseModel):
    """Base schema for service records"""
    service_id: str = Field(..., description="Unique service ID")
    service_date: date = Field(..., description="Date of service")
    location: Optional[str] = None
    branch: Optional[str] = None
    region: Optional[str] = None
    franchise_id: Optional[str] = None
    product_category: Optional[str] = None
    service_type: Optional[str] = None
    technician_id: Optional[str] = None
    customer_id: Optional[str] = None
    service_duration: Optional[float] = None
    service_cost: Optional[float] = None
    service_revenue: Optional[float] = None
    part_name: Optional[str] = None
    parts_used: Optional[int] = 0
    parts_cost: Optional[float] = 0.0
    parts_revenue: Optional[float] = 0.0
    total_revenue: Optional[float] = None
    warranty_claim: Optional[bool] = False
    customer_satisfaction: Optional[float] = None
    product_age_months: Optional[int] = None
    first_call_resolution: Optional[bool] = True
    priority: Optional[int] = 2


class ServiceRecordCreate(ServiceRecordBase):
    """Schema for creating a service record"""
    pass


class ServiceRecordResponse(ServiceRecordBase):
    """Schema for service record response"""
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ServiceUploadResponse(BaseModel):
    """Response schema for bulk upload"""
    success: bool
    records_created: int
    records_failed: int
    errors: Optional[list] = []
    message: str
