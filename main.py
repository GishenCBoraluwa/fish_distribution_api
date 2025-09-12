"""
Main FastAPI application for Fish Distribution Optimization API
"""
import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.core.config import get_settings
from app.core.exceptions import setup_exception_handlers
from app.api.v1.router import api_router
from app.utils.logging import setup_logging

# Global variables for tracking
start_time = time.time()
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    setup_logging()
    
    yield
    
    # Shutdown
    # Any cleanup code goes here


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-ready API for optimizing fish distribution routes using AI algorithms",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Allow all hosts for development
)

# Setup exception handlers
setup_exception_handlers(app)

# Include API routes
app.include_router(api_router, prefix=settings.api_v1_str)


@app.get("/")
async def root():
    """Root endpoint with basic API information"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "operational",
        "timestamp": datetime.utcnow(),
        "uptime_seconds": round(time.time() - start_time, 2),
        "docs_url": "/docs" if settings.debug else "Contact administrator for API documentation",
        "endpoints": {
            "optimization": f"{settings.api_v1_str}/optimization/",
            "demand_prediction": f"{settings.api_v1_str}/demand/",
            "analytics": f"{settings.api_v1_str}/analytics/",
            "health": f"{settings.api_v1_str}/health/"
        }
    }


@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "uptime_seconds": round(time.time() - start_time, 2)
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers,
        log_level=settings.log_level.lower()
    )