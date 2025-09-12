"""
Pydantic models for API responses
"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class OrderInfo(BaseModel):
    """Information about a processed order"""
    id: str
    city: str
    quantity: float
    freshness_limit: float


class TruckRoute(BaseModel):
    """Information about a truck route"""
    truck_id: str
    assigned_harbor: str
    route: List[str]
    total_load_kg: float
    total_distance_km: float
    estimated_time_hours: float
    freshness_violations: List[str]
    orders: List[OrderInfo]


class OptimizationSummary(BaseModel):
    """Summary of optimization results"""
    total_orders: int
    trucks_used: int
    total_distance: float
    freshness_violations: int
    optimization_fitness: float


class OptimizationResponse(BaseModel):
    """Response model for route optimization"""
    success: bool
    message: str
    summary: OptimizationSummary
    delivery_plan: List[TruckRoute]
    processing_time_seconds: float
    timestamp: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Optimization completed successfully",
                "summary": {
                    "total_orders": 10,
                    "trucks_used": 3,
                    "total_distance": 485.5,
                    "freshness_violations": 0,
                    "optimization_fitness": 1250.0
                },
                "delivery_plan": [
                    {
                        "truck_id": "T001",
                        "assigned_harbor": "Negombo",
                        "route": ["Kandy", "Kurunegala"],
                        "total_load_kg": 420.0,
                        "total_distance_km": 165.2,
                        "estimated_time_hours": 3.3,
                        "freshness_violations": [],
                        "orders": []
                    }
                ],
                "processing_time_seconds": 12.5,
                "timestamp": "2025-09-08T10:30:00Z"
            }
        }


class DemandPrediction(BaseModel):
    """Demand prediction for a city"""
    city: str
    predicted_demand_kg: float
    confidence_level: str = "medium"


class DemandPredictionResponse(BaseModel):
    """Response model for demand prediction"""
    success: bool
    message: str
    predictions: List[DemandPrediction]
    prediction_date: datetime
    model_accuracy_score: Optional[float]
    processing_time_seconds: float
    timestamp: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Demand prediction completed successfully",
                "predictions": [
                    {
                        "city": "Colombo",
                        "predicted_demand_kg": 485.2,
                        "confidence_level": "high"
                    },
                    {
                        "city": "Kandy", 
                        "predicted_demand_kg": 295.8,
                        "confidence_level": "medium"
                    }
                ],
                "prediction_date": "2025-09-09T00:00:00Z",
                "model_accuracy_score": 0.875,
                "processing_time_seconds": 2.1,
                "timestamp": "2025-09-08T10:30:00Z"
            }
        }


class CostAnalysis(BaseModel):
    """Cost analysis breakdown"""
    fuel_cost_usd: float
    driver_cost_usd: float
    truck_cost_usd: float
    total_operational_cost_usd: float
    cost_per_kg_usd: float


class EfficiencyMetrics(BaseModel):
    """Efficiency metrics"""
    orders_per_truck: float
    kg_per_km: float
    violation_rate_percent: float
    capacity_utilization_percent: float


class AnalyticsResponse(BaseModel):
    """Response model for analytics"""
    success: bool
    message: str
    cost_analysis: Optional[CostAnalysis]
    efficiency_metrics: Optional[EfficiencyMetrics]
    recommendations: List[str]
    processing_time_seconds: float
    timestamp: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Analytics generated successfully",
                "cost_analysis": {
                    "fuel_cost_usd": 388.4,
                    "driver_cost_usd": 82.5,
                    "truck_cost_usd": 150.0,
                    "total_operational_cost_usd": 620.9,
                    "cost_per_kg_usd": 0.218
                },
                "efficiency_metrics": {
                    "orders_per_truck": 3.33,
                    "kg_per_km": 5.87,
                    "violation_rate_percent": 0.0,
                    "capacity_utilization_percent": 75.6
                },
                "recommendations": [
                    "Consider consolidating routes to improve truck utilization",
                    "Current freshness compliance is excellent"
                ],
                "processing_time_seconds": 1.2,
                "timestamp": "2025-09-08T10:30:00Z"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: datetime
    uptime_seconds: float
    dependencies: Dict[str, str]


class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = False
    error: str
    details: Optional[str] = None
    timestamp: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "Validation Error",
                "details": "Invalid city name provided",
                "timestamp": "2025-09-08T10:30:00Z"
            }
        }