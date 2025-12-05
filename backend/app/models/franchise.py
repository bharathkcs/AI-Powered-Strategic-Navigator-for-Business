"""
Database models for franchise partners
"""

from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from app.database import Base


class Franchise(Base):
    """Franchise partner model"""
    __tablename__ = "franchises"

    id = Column(Integer, primary_key=True, index=True)
    franchise_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)

    # Location
    region = Column(String, index=True)
    state = Column(String)
    city = Column(String)

    # Contact information
    contact_person = Column(String)
    email = Column(String)
    phone = Column(String)

    # Performance metrics (calculated periodically)
    performance_score = Column(Float, default=0.0)
    tier = Column(String, default="Silver")  # Platinum, Gold, Silver

    # Status
    is_active = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self):
        return f"<Franchise {self.franchise_id}: {self.name}>"
