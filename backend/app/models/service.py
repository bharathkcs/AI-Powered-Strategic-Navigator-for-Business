"""
Database models for service records
"""

from sqlalchemy import Column, Integer, String, Float, Date, Boolean, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.database import Base


class ServiceRecord(Base):
    """Service call record model"""
    __tablename__ = "service_records"

    id = Column(Integer, primary_key=True, index=True)
    service_id = Column(String, unique=True, index=True, nullable=False)
    service_date = Column(Date, nullable=False, index=True)

    # Location information
    location = Column(String, index=True)
    branch = Column(String)
    region = Column(String, index=True)
    franchise_id = Column(String, ForeignKey("franchises.franchise_id"), index=True)

    # Product and service information
    product_category = Column(String, index=True)
    service_type = Column(String, index=True)
    technician_id = Column(String, index=True)
    customer_id = Column(String)

    # Service metrics
    service_duration = Column(Float)  # hours
    service_cost = Column(Float)
    service_revenue = Column(Float)

    # Parts information
    part_name = Column(String)
    parts_used = Column(Integer, default=0)
    parts_cost = Column(Float, default=0.0)
    parts_revenue = Column(Float, default=0.0)

    # Additional metrics
    total_revenue = Column(Float)
    warranty_claim = Column(Boolean, default=False, index=True)
    customer_satisfaction = Column(Float)  # 1-5 scale
    product_age_months = Column(Integer)
    first_call_resolution = Column(Boolean, default=True)
    priority = Column(Integer, default=2)  # 1-4 scale

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<ServiceRecord {self.service_id}>"
