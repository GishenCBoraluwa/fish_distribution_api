"""
Logging configuration for the Fish Distribution API
"""
import logging
import logging.config
import sys
from pathlib import Path

from app.core.config import get_settings

settings = get_settings()


def setup_logging():
    """Setup logging configuration"""
    
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "default",
                "stream": sys.stdout
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "detailed",
                "filename": "logs/api.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": "logs/errors.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            }
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console", "file"],
                "level": settings.log_level,
                "propagate": False
            },
            "app": {
                "handlers": ["console", "file", "error_file"],
                "level": settings.log_level,
                "propagate": False
            },
            "uvicorn": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False
            }
        }
    }
    
    logging.config.dictConfig(logging_config)
    
    # Set up specific loggers
    logging.getLogger("app.services.optimization_service").setLevel(logging.INFO)
    logging.getLogger("app.services.demand_service").setLevel(logging.INFO)
    logging.getLogger("app.services.analytics_service").setLevel(logging.INFO)