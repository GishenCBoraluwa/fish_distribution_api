"""
Security utilities for authentication and authorization
"""
import secrets
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status

from app.core.config import get_settings

settings = get_settings()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration
ALGORITHM = "HS256"


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=ALGORITHM)
    
    return encoded_jwt


def verify_token(token: str):
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
        
        return {"user_id": user_id}
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash password"""
    return pwd_context.hash(password)


def generate_api_key() -> str:
    """Generate secure API key"""
    return secrets.token_urlsafe(32)


def verify_api_key(api_key: str) -> bool:
    """Verify API key (placeholder implementation)"""
    # In production, store API keys in database
    valid_keys = ["demo-api-key", "test-api-key"]
    return api_key in valid_keys