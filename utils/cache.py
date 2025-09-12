"""
Caching utilities for improved performance
"""
import json
import logging
from typing import Any, Optional, Union
from datetime import datetime, timedelta
import hashlib
import pickle

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class CacheManager:
    """Unified cache manager with Redis and in-memory fallback"""
    
    def __init__(self):
        self.redis_client = None
        self.memory_cache = {}
        self.cache_timestamps = {}
        
        if REDIS_AVAILABLE and settings.redis_url:
            try:
                self.redis_client = redis.from_url(settings.redis_url, decode_responses=False)
                # Test connection
                self.redis_client.ping()
                logger.info("Redis cache connection established")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, using memory cache")
                self.redis_client = None
        else:
            logger.info("Using in-memory cache (Redis not available)")
    
    def _generate_key(self, key: str, prefix: str = "fishapi") -> str:
        """Generate cache key with prefix"""
        return f"{prefix}:{key}"
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for caching"""
        return pickle.dumps(value)
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize cached value"""
        return pickle.loads(data)
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        cache_key = self._generate_key(key)
        
        try:
            if self.redis_client:
                # Try Redis first
                data = self.redis_client.get(cache_key)
                if data:
                    return self._deserialize_value(data)
            else:
                # Use memory cache
                if cache_key in self.memory_cache:
                    # Check if expired
                    timestamp = self.cache_timestamps.get(cache_key)
                    if timestamp and datetime.now() - timestamp < timedelta(seconds=settings.cache_expire_seconds):
                        return self.memory_cache[cache_key]
                    else:
                        # Remove expired entry
                        self.memory_cache.pop(cache_key, None)
                        self.cache_timestamps.pop(cache_key, None)
        
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        
        return default
    
    async def set(self, key: str, value: Any, expire_seconds: Optional[int] = None) -> bool:
        """Set value in cache"""
        cache_key = self._generate_key(key)
        expire_time = expire_seconds or settings.cache_expire_seconds
        
        try:
            if self.redis_client:
                # Use Redis
                serialized_value = self._serialize_value(value)
                return self.redis_client.setex(cache_key, expire_time, serialized_value)
            else:
                # Use memory cache
                self.memory_cache[cache_key] = value
                self.cache_timestamps[cache_key] = datetime.now()
                return True
        
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        cache_key = self._generate_key(key)
        
        try:
            if self.redis_client:
                return bool(self.redis_client.delete(cache_key))
            else:
                self.memory_cache.pop(cache_key, None)
                self.cache_timestamps.pop(cache_key, None)
                return True
        
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            if self.redis_client:
                # Clear only our keys
                keys = self.redis_client.keys("fishapi:*")
                if keys:
                    return bool(self.redis_client.delete(*keys))
            else:
                self.memory_cache.clear()
                self.cache_timestamps.clear()
                return True
        
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    def generate_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        # Create a string representation of arguments
        key_parts = []
        
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                key_parts.append(str(hash(str(arg))))
        
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (str, int, float, bool)):
                key_parts.append(f"{k}:{v}")
            else:
                key_parts.append(f"{k}:{hash(str(v))}")
        
        # Create hash of the key parts
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()


# Global cache manager instance
cache_manager = CacheManager()


def cache_key_for_demand_prediction(cities: list, date: str, rainfall: float) -> str:
    """Generate cache key for demand predictions"""
    return cache_manager.generate_cache_key("demand", sorted(cities), date, rainfall)


def cache_key_for_optimization(orders_hash: str, params: dict) -> str:
    """Generate cache key for optimization results"""
    return cache_manager.generate_cache_key("optimization", orders_hash, params)


def cache_key_for_routes(origins: list, destinations: list) -> str:
    """Generate cache key for route calculations"""
    return cache_manager.generate_cache_key("routes", sorted(origins), sorted(destinations))


async def cached_function(cache_key: str, func, *args, expire_seconds: int = None, **kwargs):
    """Generic caching wrapper for functions"""
    # Try to get from cache first
    result = await cache_manager.get(cache_key)
    
    if result is not None:
        logger.debug(f"Cache hit for key: {cache_key}")
        return result
    
    # Execute function and cache result
    logger.debug(f"Cache miss for key: {cache_key}")
    result = await func(*args, **kwargs) if callable(func) else func
    
    # Cache the result
    await cache_manager.set(cache_key, result, expire_seconds)
    
    return result