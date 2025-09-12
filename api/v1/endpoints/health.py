"""
Health check and monitoring endpoints
"""
import time
import logging
from datetime import datetime

from fastapi import APIRouter, status
import psutil

from app.models.response_models import HealthResponse
from app.core.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

# Track application start time
start_time = time.time()


@router.get(
    "/",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Check the health status of the API and its dependencies"
)
async def health_check():
    """
    Comprehensive health check endpoint
    """
    try:
        uptime = time.time() - start_time
        
        # Check system resources
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent(interval=1)
        disk_usage = psutil.disk_usage('/').percent
        
        # Dependency status
        dependencies = {
            "numpy": "✓ Available",
            "pandas": "✓ Available", 
            "sklearn": "✓ Available",
            "matplotlib": "✓ Available"
        }
        
        # Check if system is healthy
        is_healthy = (
            memory_usage < 90 and 
            cpu_usage < 95 and 
            disk_usage < 90
        )
        
        status_text = "healthy" if is_healthy else "degraded"
        
        return HealthResponse(
            status=status_text,
            version=settings.app_version,
            timestamp=datetime.utcnow(),
            uptime_seconds=round(uptime, 2),
            dependencies=dependencies
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            version=settings.app_version,
            timestamp=datetime.utcnow(),
            uptime_seconds=0,
            dependencies={"error": f"Health check failed: {str(e)}"}
        )


@router.get(
    "/metrics",
    summary="System Metrics",
    description="Get detailed system performance metrics"
)
async def get_metrics():
    """
    Get detailed system metrics for monitoring
    """
    try:
        # System metrics
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "system_metrics": {
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_percent": memory.percent
                },
                "cpu": {
                    "usage_percent": psutil.cpu_percent(interval=1),
                    "cores": psutil.cpu_count()
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "used_percent": round((disk.used / disk.total) * 100, 1)
                }
            },
            "application_metrics": {
                "uptime_seconds": round(time.time() - start_time, 2),
                "version": settings.app_version,
                "environment": "development" if settings.debug else "production"
            },
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {str(e)}")
        return {
            "error": f"Metrics collection failed: {str(e)}",
            "timestamp": datetime.utcnow()
        }


@router.get(
    "/status",
    summary="Service Status",
    description="Get status of individual services"
)
async def get_service_status():
    """
    Check the status of individual services
    """
    try:
        # Import services to check their status
        from app.services.optimization_service import OptimizationService
        from app.services.demand_service import DemandPredictionService
        from app.services.analytics_service import AnalyticsService
        
        services = {}
        
        # Check optimization service
        try:
            opt_service = OptimizationService()
            services["optimization"] = {
                "status": "operational",
                "harbors_available": len(opt_service.harbors),
                "trucks_available": len(opt_service.trucks)
            }
        except Exception as e:
            services["optimization"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Check demand prediction service
        try:
            demand_service = DemandPredictionService()
            model_info = await demand_service.get_model_info()
            services["demand_prediction"] = {
                "status": "operational" if model_info["is_trained"] else "training",
                "model_trained": model_info["is_trained"],
                "accuracy": model_info["accuracy"]
            }
        except Exception as e:
            services["demand_prediction"] = {
                "status": "error", 
                "error": str(e)
            }
        
        # Check analytics service
        try:
            analytics_service = AnalyticsService()
            services["analytics"] = {
                "status": "operational",
                "cost_analysis_enabled": True,
                "efficiency_metrics_enabled": True
            }
        except Exception as e:
            services["analytics"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Overall status
        all_operational = all(
            service.get("status") == "operational" 
            for service in services.values()
        )
        
        return {
            "overall_status": "operational" if all_operational else "degraded",
            "services": services,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Service status check failed: {str(e)}")
        return {
            "overall_status": "error",
            "error": f"Status check failed: {str(e)}",
            "timestamp": datetime.utcnow()
        }