"""Inventory API endpoints - Simplified"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/abc-analysis")
async def get_abc_analysis():
    """Get ABC inventory analysis"""
    return {"message": "ABC analysis endpoint"}
