"""
API dependencies for rate limiting, authentication, and validation
"""
import time
from typing import Dict, Any
import asyncio
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status
from app.core.config import get_settings

settings = get_settings()

# In-memory rate limiter (for production, use Redis)
class InMemoryRateLimiter:
    def __init__(self):
        self.requests = defaultdict(deque)
        self.lock = asyncio.Lock()
    
    async def is_allowed(self, identifier: str, window_seconds: int = 3600, max_requests: int = 100) -> bool:
        async with self.lock:
            now = time.time()
            # Clean old requests
            while self.requests[identifier] and self.requests[identifier][0] < now - window_seconds:
                self.requests[identifier].popleft()
            
            # Check if under limit
            if len(self.requests[identifier]) < max_requests:
                self.requests[identifier].append(now)
                return True
            
            return False

# Global rate limiter instance
rate_limiter = InMemoryRateLimiter()


async def get_rate_limiter(request: Request):
    """Rate limiting dependency"""
    # Use IP address as identifier (in production, consider user tokens)
    client_ip = request.client.host
    
    allowed = await rate_limiter.is_allowed(
        client_ip,
        settings.rate_limit_window,
        settings.rate_limit_requests
    )
    
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {settings.rate_limit_requests} requests per hour."
        )
    
    return True


async def verify_request_size(request: Request):
    """Verify request size to prevent DoS attacks"""
    # Get content length
    content_length = request.headers.get("content-length")
    
    if content_length:
        content_length = int(content_length)
        max_size = 10 * 1024 * 1024  # 10 MB limit
        
        if content_length > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Request too large. Maximum size is 10MB."
            )
    
    return True


def get_current_user():
    """Placeholder for user authentication (implement as needed)"""
    # In production, implement proper JWT token validation
    return {"user_id": "anonymous", "role": "user"}