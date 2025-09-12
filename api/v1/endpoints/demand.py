"""
Demand prediction API endpoints
"""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer

from app.models.request_models import DemandPredictionRequest
from app.models.response_models import DemandPredictionResponse, ErrorResponse
from app.services.demand_service import DemandPredictionService
from app.api.dependencies import get_rate_limiter
from app.core.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer(auto_error=False)
settings = get_settings()

# Initialize service
demand_service = DemandPredictionService()


@router.post(
    "/predict",
    response_model=DemandPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Fish Demand",
    description="""
    Predict fish demand for specified cities using machine learning models.
    
    **Features:**
    - Random Forest regression model
    - Seasonal and weather factor consideration
    - Multiple city batch prediction
    - Confidence level assessment
    
    **Prediction Factors:**
    - Historical demand patterns
    - Seasonal variations
    - Day of week effects
    - Weather conditions (rainfall)
    - City-specific characteristics
    """,
    responses={
        200: {"description": "Prediction completed successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def predict_demand(
    request: DemandPredictionRequest,
    token: Optional[str] = Depends(security),
    _rate_limit: None = Depends(get_rate_limiter)
):
    """
    Predict fish demand for the specified cities and date
    """
    try:
        logger.info(f"Predicting demand for {len(request.cities)} cities on {request.prediction_date}")
        
        result = await demand_service.predict_demand(request)
        
        if result.success:
            total_predicted = sum(p.predicted_demand_kg for p in result.predictions)
            logger.info(f"Demand prediction completed. Total predicted: {total_predicted:.1f} kg")
        else:
            logger.error(f"Demand prediction failed: {result.message}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Demand prediction endpoint error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during demand prediction"
        )


@router.get(
    "/model-info",
    summary="Get Model Information",
    description="Retrieve information about the demand prediction model"
)
async def get_model_info():
    """Get demand prediction model information"""
    try:
        model_info = await demand_service.get_model_info()
        return {
            "model_info": model_info,
            "usage_guidelines": {
                "optimal_prediction_window": "1-7 days ahead",
                "confidence_levels": {
                    "high": "R² score > 0.8",
                    "medium": "R² score 0.6-0.8", 
                    "low": "R² score < 0.6"
                },
                "factors_considered": [
                    "Historical demand patterns",
                    "Seasonal variations",
                    "Day of week effects",
                    "Weather conditions",
                    "City characteristics"
                ]
            }
        }
    except Exception as e:
        logger.error(f"Model info retrieval error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve model information"
        )


@router.get(
    "/supported-cities",
    summary="Get Supported Cities",
    description="Get list of cities supported for demand prediction"
)
async def get_supported_cities():
    """Get list of supported cities for demand prediction"""
    return {
        "supported_cities": [
            {
                "name": "Colombo",
                "population": "~750,000",
                "typical_demand_kg": "400-600"
            },
            {
                "name": "Kandy", 
                "population": "~125,000",
                "typical_demand_kg": "250-350"
            },
            {
                "name": "Galle",
                "population": "~120,000", 
                "typical_demand_kg": "200-300"
            },
            {
                "name": "Jaffna",
                "population": "~90,000",
                "typical_demand_kg": "150-250"
            },
            {
                "name": "Anuradhapura",
                "population": "~65,000",
                "typical_demand_kg": "150-220"
            },
            {
                "name": "Negombo",
                "population": "~145,000",
                "typical_demand_kg": "300-500"
            },
            {
                "name": "Kurunegala",
                "population": "~100,000",
                "typical_demand_kg": "180-280"
            },
            {
                "name": "Ratnapura",
                "population": "~50,000",
                "typical_demand_kg": "120-180"
            },
            {
                "name": "Batticaloa",
                "population": "~95,000",
                "typical_demand_kg": "130-190"
            },
            {
                "name": "Trincomalee",
                "population": "~100,000",
                "typical_demand_kg": "140-200"
            }
        ],
        "total_cities": 10,
        "coverage": "Major Sri Lankan cities"
    }