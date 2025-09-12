"""
Custom exception handlers for the FastAPI application
"""
import logging
from datetime import datetime
from typing import Union

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class OptimizationException(Exception):
    """Custom exception for optimization errors"""
    def __init__(self, message: str, details: str = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class DemandPredictionException(Exception):
    """Custom exception for demand prediction errors"""
    def __init__(self, message: str, details: str = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


def setup_exception_handlers(app: FastAPI):
    """Setup global exception handlers"""
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions"""
        logger.warning(f"HTTP {exc.status_code} error: {exc.detail}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": exc.detail,
                "status_code": exc.status_code,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors"""
        logger.warning(f"Validation error: {exc.errors()}")
        
        # Format validation errors nicely
        errors = []
        for error in exc.errors():
            field = " -> ".join(str(loc) for loc in error["loc"])
            errors.append(f"{field}: {error['msg']}")
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "error": "Validation Error",
                "details": "; ".join(errors),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    @app.exception_handler(ValidationError)
    async def pydantic_validation_handler(request: Request, exc: ValidationError):
        """Handle Pydantic validation errors"""
        logger.warning(f"Pydantic validation error: {exc.errors()}")
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "error": "Data Validation Error",
                "details": str(exc),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    @app.exception_handler(OptimizationException)
    async def optimization_exception_handler(request: Request, exc: OptimizationException):
        """Handle optimization-specific errors"""
        logger.error(f"Optimization error: {exc.message}")
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": exc.message,
                "details": exc.details,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    @app.exception_handler(DemandPredictionException)
    async def demand_prediction_exception_handler(request: Request, exc: DemandPredictionException):
        """Handle demand prediction errors"""
        logger.error(f"Demand prediction error: {exc.message}")
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": exc.message,
                "details": exc.details,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all other exceptions"""
        logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "Internal Server Error",
                "details": "An unexpected error occurred. Please try again later.",
                "timestamp": datetime.utcnow().isoformat()
            }
        )