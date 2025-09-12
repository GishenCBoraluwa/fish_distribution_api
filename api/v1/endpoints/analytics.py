"""
Analytics API endpoints
"""
import logging
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer

from app.models.request_models import AnalyticsRequest
from app.models.response_models import AnalyticsResponse, ErrorResponse
from app.services.analytics_service import AnalyticsService
from app.api.dependencies import get_rate_limiter

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer(auto_error=False)

# Initialize service
analytics_service = AnalyticsService()


@router.post(
    "/generate",
    response_model=AnalyticsResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Analytics Report",
    description="""
    Generate comprehensive analytics from optimization results.
    
    **Analytics Include:**
    - Cost breakdown analysis
    - Efficiency metrics
    - Performance recommendations
    - Resource utilization assessment
    - Operational insights
    """,
    responses={
        200: {"description": "Analytics generated successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def generate_analytics(
    request: AnalyticsRequest,
    token: Optional[str] = Depends(security),
    _rate_limit: None = Depends(get_rate_limiter)
):
    """
    Generate comprehensive analytics from optimization results
    """
    try:
        logger.info("Generating analytics report")
        
        result = await analytics_service.generate_analytics(request)
        
        if result.success:
            logger.info("Analytics generation completed successfully")
        else:
            logger.error(f"Analytics generation failed: {result.message}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analytics endpoint error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during analytics generation"
        )


@router.post(
    "/compare-scenarios",
    summary="Compare Optimization Scenarios",
    description="Compare two different optimization scenarios to identify improvements"
)
async def compare_scenarios(
    base_results: Dict[str, Any],
    alternative_results: Dict[str, Any],
    token: Optional[str] = Depends(security),
    _rate_limit: None = Depends(get_rate_limiter)
):
    """Compare two optimization scenarios"""
    try:
        logger.info("Comparing optimization scenarios")
        
        comparison = await analytics_service.compare_scenarios(base_results, alternative_results)
        
        return {
            "success": True,
            "message": "Scenario comparison completed",
            "comparison_results": comparison,
            "timestamp": "2025-09-08T10:30:00Z"
        }
        
    except Exception as e:
        logger.error(f"Scenario comparison error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during scenario comparison"
        )


@router.post(
    "/performance-report",
    summary="Generate Performance Report",
    description="Generate a comprehensive performance report with KPIs and scores"
)
async def generate_performance_report(
    optimization_results: Dict[str, Any],
    token: Optional[str] = Depends(security),
    _rate_limit: None = Depends(get_rate_limiter)
):
    """Generate comprehensive performance report"""
    try:
        logger.info("Generating performance report")
        
        report = await analytics_service.generate_performance_report(optimization_results)
        
        return {
            "success": True,
            "message": "Performance report generated successfully",
            "report": report,
            "timestamp": "2025-09-08T10:30:00Z"
        }
        
    except Exception as e:
        logger.error(f"Performance report error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during performance report generation"
        )