"""Franchise API endpoints - Simplified"""

from fastapi import APIRouter

router = APIRouter()

@router.get("/performance")
async def get_franchise_performance():
    """Get franchise performance metrics"""
    return {"message": "Franchise performance endpoint"}
