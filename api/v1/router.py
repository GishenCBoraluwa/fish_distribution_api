"""
API v1 router configuration
"""
from fastapi import APIRouter

from app.api.v1.endpoints import optimization, demand, analytics, health

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    optimization.router,
    prefix="/optimization",
    tags=["optimization"],
    responses={404: {"description": "Not found"}}
)

api_router.include_router(
    demand.router,
    prefix="/demand",
    tags=["demand"],
    responses={404: {"description": "Not found"}}
)

api_router.include_router(
    analytics.router,
    prefix="/analytics", 
    tags=["analytics"],
    responses={404: {"description": "Not found"}}
)

api_router.include_router(
    health.router,
    prefix="/health",
    tags=["health"],
    responses={404: {"description": "Not found"}}
)