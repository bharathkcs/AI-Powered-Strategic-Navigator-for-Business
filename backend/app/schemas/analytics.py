"""
Pydantic schemas for analytics
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class AnalyticsSummary(BaseModel):
    """Executive summary analytics"""
    total_service_calls: int
    total_service_revenue: float
    total_parts_revenue: float
    avg_service_value: float
    warranty_claims: int
    warranty_claim_rate: float
    avg_customer_satisfaction: float
    top_locations: List[Dict[str, any]]
    insights: Optional[str] = None


class LocationPerformance(BaseModel):
    """Location performance metrics"""
    location: str
    service_calls: int
    service_revenue: float
    parts_used: int
    warranty_claims: int
    warranty_rate: float
    avg_satisfaction: float
    rank: int


class FranchisePerformance(BaseModel):
    """Franchise performance metrics"""
    franchise_id: str
    franchise_name: Optional[str] = None
    service_calls: int
    total_revenue: float
    warranty_rate: float
    performance_score: float
    tier: str  # Platinum, Gold, Silver
    rank: int
    insights: Optional[str] = None


class InventoryABC(BaseModel):
    """ABC inventory analysis"""
    part_name: str
    usage_count: int
    total_cost: float
    total_revenue: float
    category: str  # A, B, or C
    cumulative_percentage: float
    recommended_stock_level: Optional[int] = None


class RevenueLeakage(BaseModel):
    """Revenue leakage analysis"""
    total_leakage: float
    leakage_percentage: float
    warranty_leakage: float
    service_inefficiency_leakage: float
    parts_pricing_leakage: float
    recovery_potential: float
    recommendations: List[str]
