"""
Configuration settings for Fish Distribution Optimization API
"""
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Configuration
    app_name: str = "Fish Distribution Optimization API"
    app_version: str = "1.0.0"
    debug: bool = False
    api_v1_str: str = "/api/v1"
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Security
    secret_key: str = Field(..., description="Secret key for JWT tokens")
    access_token_expire_minutes: int = 60 * 24 * 8  # 8 days
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour in seconds
    
    # Google Maps API
    google_maps_api_key: Optional[str] = None
    
    # Redis Configuration (for caching and rate limiting)
    redis_url: str = "redis://localhost:6379"
    cache_expire_seconds: int = 3600  # 1 hour
    
    # Optimization Parameters
    abc_max_iterations: int = 100
    abc_n_bees: int = 50
    ml_model_cache_time: int = 86400  # 24 hours
    
    # Performance Settings
    max_orders_per_request: int = 100
    max_concurrent_requests: int = 10
    request_timeout_seconds: int = 300  # 5 minutes
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()