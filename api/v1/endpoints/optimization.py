"""
Optimization API endpoints
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status
from fastapi.security import HTTPBearer

from app.models.request_models import OptimizationRequest, OptimizationParameters
from app.models.response_models import OptimizationResponse, ErrorResponse
from app.services.optimization_service import OptimizationService
from app.api.dependencies import get_rate_limiter, verify_request_size
from app.core.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer(auto_error=False)
settings = get_settings()

# Initialize service (singleton pattern)
optimization_service = OptimizationService()


@router.post(
    "/optimize",
    response_model=OptimizationResponse,
    status_code=status.HTTP_200_OK,
    summary="Optimize Fish Distribution Routes",
    description="""
    Optimize delivery routes for fish distribution using the Artificial Bee Colony (ABC) algorithm.
    
    **Features:**
    - Multi-harbor assignment based on proximity and capacity
    - Freshness constraint handling
    - Truck capacity optimization
    - Distance minimization
    - Real-time route planning
    
    **Request Limits:**
    - Maximum 100 orders per request
    - Maximum 5 minute processing time
    - Rate limited to prevent abuse
    """,
    responses={
        200: {"description": "Optimization completed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        429: {"description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def optimize_routes(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    token: Optional[str] = Depends(security),
    _rate_limit: None = Depends(get_rate_limiter),
    _size_check: None = Depends(verify_request_size)
):
    """
    Optimize delivery routes for the given orders
    """
    try:
        # Validate request size
        if len(request.orders) > settings.max_orders_per_request:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Too many orders. Maximum {settings.max_orders_per_request} orders per request."
            )
        
        # Extract optimization parameters
        params = None
        if request.optimization_params:
            params = OptimizationParameters(**request.optimization_params)
        
        logger.info(f"Starting optimization for {len(request.orders)} orders")
        
        # Run optimization
        result = await optimization_service.optimize_distribution(request.orders, params)
        
        # Log result for monitoring
        if result.success:
            logger.info(
                f"Optimization completed: {result.summary.trucks_used} trucks, "
                f"{result.summary.total_distance}km, {result.summary.freshness_violations} violations"
            )
        else:
            logger.error(f"Optimization failed: {result.message}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Optimization endpoint error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during optimization"
        )


@router.get(
    "/parameters",
    summary="Get Default Optimization Parameters",
    description="Retrieve the default parameters used by the optimization algorithm"
)
async def get_default_parameters():
    """Get default optimization parameters"""
    return {
        "default_parameters": {
            "max_iterations": settings.abc_max_iterations,
            "n_bees": settings.abc_n_bees,
            "abandonment_limit": 10
        },
        "parameter_ranges": {
            "max_iterations": {"min": 10, "max": 500, "recommended": 100},
            "n_bees": {"min": 10, "max": 200, "recommended": 50},
            "abandonment_limit": {"min": 5, "max": 50, "recommended": 10}
        },
        "description": {
            "max_iterations": "Maximum number of ABC algorithm iterations",
            "n_bees": "Number of bees in the ABC colony",
            "abandonment_limit": "Number of trials before abandoning a solution"
        }
    }


@router.get(
    "/system-info",
    summary="Get System Information",
    description="Retrieve information about the optimization system capabilities"
)
async def get_system_info():
    """Get system information and capabilities"""
    return {
        "system_info": {
            "algorithm": "Artificial Bee Colony (ABC)",
            "max_orders_per_request": settings.max_orders_per_request,
            "supported_cities": [
                "Colombo", "Kandy", "Galle", "Jaffna", "Anuradhapura",
                "Negombo", "Kurunegala", "Ratnapura", "Batticaloa", "Trincomalee"
            ],
            "available_harbors": [
                {"name": "Negombo", "capacity_kg": 2000},
                {"name": "Colombo", "capacity_kg": 3000},
                {"name": "Beruwala", "capacity_kg": 1500}
            ],
            "truck_fleet": [
                {"id": "T001", "capacity_kg": 500, "cost_per_km": 0.5},
                {"id": "T002", "capacity_kg": 750, "cost_per_km": 0.6},
                {"id": "T003", "capacity_kg": 500, "cost_per_km": 0.5},
                {"id": "T004", "capacity_kg": 1000, "cost_per_km": 0.8},
                {"id": "T005", "capacity_kg": 750, "cost_per_km": 0.6}
            ]
        },
        "constraints": {
            "max_freshness_hours": 72,
            "max_order_quantity_kg": 5000,
            "min_order_quantity_kg": 0.1
        },
        "performance": {
            "average_processing_time_seconds": "10-60 depending on complexity",
            "timeout_seconds": settings.request_timeout_seconds
        }
    }