"""
Pydantic models for API request validation
"""
from datetime import datetime
from typing import List, Optional, Dict
from pydantic import BaseModel, Field, validator
from enum import Enum


class OrderRequest(BaseModel):
    """Request model for a single order"""
    order_id: str = Field(..., min_length=1, max_length=50, description="Unique order identifier")
    city: str = Field(..., min_length=1, max_length=100, description="Destination city")
    quantity: float = Field(..., gt=0, le=5000, description="Order quantity in kg")
    freshness_limit: float = Field(..., gt=0, le=72, description="Maximum delivery time in hours")
    
    @validator('city')
    def validate_city(cls, v):
        """Validate city name format"""
        allowed_cities = {
            'Colombo', 'Kandy', 'Galle', 'Jaffna', 'Anuradhapura', 
            'Negombo', 'Kurunegala', 'Ratnapura', 'Batticaloa', 'Trincomalee'
        }
        if v not in allowed_cities:
            raise ValueError(f'City must be one of: {", ".join(allowed_cities)}')
        return v


class OptimizationRequest(BaseModel):
    """Request model for route optimization"""
    orders: List[OrderRequest] = Field(..., min_items=1, max_items=100, description="List of orders to optimize")
    optimization_params: Optional[Dict] = Field(default={}, description="Optional optimization parameters")
    
    class Config:
        json_schema_extra = {
            "example": {
                "orders": [
                    {
                        "order_id": "ORD001",
                        "city": "Kandy",
                        "quantity": 200.0,
                        "freshness_limit": 5.0
                    },
                    {
                        "order_id": "ORD002", 
                        "city": "Galle",
                        "quantity": 150.0,
                        "freshness_limit": 4.0
                    }
                ],
                "optimization_params": {
                    "max_iterations": 100,
                    "n_bees": 50
                }
            }
        }


class DemandPredictionRequest(BaseModel):
    """Request model for demand prediction"""
    cities: List[str] = Field(..., min_items=1, max_items=20, description="Cities to predict demand for")
    prediction_date: datetime = Field(..., description="Date for demand prediction")
    rainfall: Optional[float] = Field(default=5.0, ge=0, le=100, description="Expected rainfall in mm")
    
    @validator('cities', each_item=True)
    def validate_cities(cls, v):
        """Validate each city name"""
        allowed_cities = {
            'Colombo', 'Kandy', 'Galle', 'Jaffna', 'Anuradhapura', 
            'Negombo', 'Kurunegala', 'Ratnapura', 'Batticaloa', 'Trincomalee'
        }
        if v not in allowed_cities:
            raise ValueError(f'City must be one of: {", ".join(allowed_cities)}')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "cities": ["Colombo", "Kandy", "Galle"],
                "prediction_date": "2025-09-09T00:00:00",
                "rainfall": 5.0
            }
        }


class AnalyticsRequest(BaseModel):
    """Request model for analytics generation"""
    optimization_results: Dict = Field(..., description="Optimization results to analyze")
    include_cost_analysis: bool = Field(default=True, description="Include cost analysis in results")
    include_efficiency_metrics: bool = Field(default=True, description="Include efficiency metrics")
    
    class Config:
        json_schema_extra = {
            "example": {
                "optimization_results": {},
                "include_cost_analysis": True,
                "include_efficiency_metrics": True
            }
        }


class OptimizationParameters(BaseModel):
    """Model for optimization algorithm parameters"""
    max_iterations: int = Field(default=100, ge=10, le=500, description="Maximum ABC iterations")
    n_bees: int = Field(default=50, ge=10, le=200, description="Number of bees in ABC algorithm")
    abandonment_limit: int = Field(default=10, ge=5, le=50, description="Abandonment limit for ABC")
    
    class Config:
        json_schema_extra = {
            "example": {
                "max_iterations": 100,
                "n_bees": 50,
                "abandonment_limit": 10
            }
        }