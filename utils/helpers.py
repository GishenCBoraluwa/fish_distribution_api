"""
Helper utilities for common operations
"""
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


def generate_hash(data: Any) -> str:
    """Generate MD5 hash from data"""
    if isinstance(data, dict):
        # Sort dict for consistent hashing
        data_str = json.dumps(data, sort_keys=True)
    elif isinstance(data, list):
        # Convert list to string
        data_str = json.dumps(sorted(data) if all(isinstance(x, (str, int, float)) for x in data) else data)
    else:
        data_str = str(data)
    
    return hashlib.md5(data_str.encode()).hexdigest()


def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency amount"""
    return f"{amount:.2f} {currency}"


def format_weight(weight_kg: float) -> str:
    """Format weight in kg"""
    return f"{weight_kg:.1f} kg"


def format_distance(distance_km: float) -> str:
    """Format distance in km"""
    return f"{distance_km:.1f} km"


def format_time(hours: float) -> str:
    """Format time duration"""
    if hours < 1:
        minutes = int(hours * 60)
        return f"{minutes} min"
    else:
        return f"{hours:.1f} hrs"


def validate_coordinates(lat: float, lng: float) -> bool:
    """Validate geographic coordinates"""
    return -90 <= lat <= 90 and -180 <= lng <= 180


def calculate_percentage(value: float, total: float) -> float:
    """Calculate percentage safely"""
    if total == 0:
        return 0
    return (value / total) * 100


def safe_divide(numerator: float, denominator: float, default: float = 0) -> float:
    """Perform division with zero check"""
    return numerator / denominator if denominator != 0 else default


def normalize_city_name(city: str) -> str:
    """Normalize city name for consistent comparison"""
    return city.strip().title()


def is_within_business_hours(hour: int, start: int = 6, end: int = 18) -> bool:
    """Check if hour is within business hours"""
    return start <= hour < end


def get_utc_timestamp() -> str:
    """Get current UTC timestamp in ISO format"""
    return datetime.now(timezone.utc).isoformat()


def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """Parse ISO timestamp string"""
    try:
        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except ValueError:
        logger.warning(f"Invalid timestamp format: {timestamp_str}")
        return None


def truncate_string(text: str, max_length: int = 100) -> str:
    """Truncate string to max length"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def remove_duplicates(lst: List[Any]) -> List[Any]:
    """Remove duplicates while preserving order"""
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]


def merge_dicts(*dicts) -> Dict[str, Any]:
    """Merge multiple dictionaries"""
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result


class Timer:
    """Simple context manager for timing operations"""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, *args):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        logger.info(f"{self.description} completed in {duration:.2f} seconds")
    
    @property
    def duration(self) -> float:
        """Get duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed operations"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying in {delay}s")
                    await asyncio.sleep(delay)
            return None
        return wrapper
    return decorator


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations"""
    import re
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    # Limit length
    return filename[:255] if filename else 'unnamed'


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def validate_email(email: str) -> bool:
    """Basic email validation"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    return re.match(pattern, email) is not None


def generate_uuid() -> str:
    """Generate UUID4 string"""
    import uuid
    return str(uuid.uuid4())


def round_to_nearest(value: float, nearest: float = 0.5) -> float:
    """Round value to nearest specified increment"""
    return round(value / nearest) * nearest