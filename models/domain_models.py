"""
Domain models for business logic
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
from datetime import datetime


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class TruckStatus(Enum):
    """Truck status enumeration"""
    AVAILABLE = "available"
    IN_USE = "in_use"
    MAINTENANCE = "maintenance"
    OUT_OF_SERVICE = "out_of_service"


@dataclass
class Location:
    """Geographic location"""
    latitude: float
    longitude: float
    name: str
    
    def distance_to(self, other: 'Location') -> float:
        """Calculate distance to another location using Haversine formula"""
        import math
        
        R = 6371  # Earth's radius in km
        
        lat1, lng1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lng2 = math.radians(other.latitude), math.radians(other.longitude)
        
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1) * math.cos(lat2) * math.sin(dlng/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c


@dataclass
class Order:
    """Fish delivery order domain model"""
    order_id: str
    city: str
    quantity: float  # kg
    freshness_limit: float  # hours
    status: OrderStatus = OrderStatus.PENDING
    assigned_harbor: Optional[str] = None
    assigned_truck: Optional[str] = None
    priority: int = 1  # 1-5, 5 being highest
    created_at: datetime = None
    location: Optional[Location] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def is_expired(self) -> bool:
        """Check if order has exceeded freshness limit"""
        if not self.created_at:
            return False
        
        elapsed_hours = (datetime.utcnow() - self.created_at).total_seconds() / 3600
        return elapsed_hours > self.freshness_limit


@dataclass
class Harbor:
    """Fishery harbor domain model"""
    name: str
    location: Location
    capacity: float  # max kg per day
    available_fish: float  # current available fish
    operating_hours: Tuple[int, int] = (6, 18)  # 6 AM to 6 PM
    
    def __post_init__(self):
        if self.available_fish is None:
            self.available_fish = self.capacity
    
    def can_fulfill_order(self, order: Order) -> bool:
        """Check if harbor can fulfill an order"""
        return self.available_fish >= order.quantity
    
    def reserve_fish(self, quantity: float) -> bool:
        """Reserve fish for an order"""
        if self.available_fish >= quantity:
            self.available_fish -= quantity
            return True
        return False
    
    def utilization_rate(self) -> float:
        """Calculate harbor utilization rate"""
        if self.capacity == 0:
            return 0
        return (self.capacity - self.available_fish) / self.capacity


@dataclass
class Truck:
    """Delivery truck domain model"""
    truck_id: str
    capacity: float  # max kg
    cost_per_km: float
    current_load: float = 0
    status: TruckStatus = TruckStatus.AVAILABLE
    current_location: Optional[Location] = None
    route: List[str] = None
    
    def __post_init__(self):
        if self.route is None:
            self.route = []
    
    def available_capacity(self) -> float:
        """Get remaining capacity"""
        return self.capacity - self.current_load
    
    def can_load_order(self, order: Order) -> bool:
        """Check if truck can load an order"""
        return self.available_capacity() >= order.quantity
    
    def load_order(self, order: Order) -> bool:
        """Load an order onto truck"""
        if self.can_load_order(order):
            self.current_load += order.quantity
            order.assigned_truck = self.truck_id
            return True
        return False
    
    def utilization_rate(self) -> float:
        """Calculate truck utilization rate"""
        if self.capacity == 0:
            return 0
        return self.current_load / self.capacity


@dataclass
class Route:
    """Delivery route domain model"""
    route_id: str
    truck_id: str
    harbor_id: str
    orders: List[Order]
    total_distance: float = 0
    estimated_time: float = 0  # hours
    status: str = "planned"
    
    def calculate_metrics(self, locations: dict) -> None:
        """Calculate route metrics"""
        if not self.orders:
            return
        
        # Calculate total distance
        harbor_location = locations.get(self.harbor_id)
        if not harbor_location:
            return
        
        current_location = harbor_location
        total_distance = 0
        
        for order in self.orders:
            next_location = locations.get(order.city)
            if next_location:
                total_distance += current_location.distance_to(next_location)
                current_location = next_location
        
        self.total_distance = total_distance
        self.estimated_time = total_distance / 50  # Assuming 50 km/h average speed
    
    def total_weight(self) -> float:
        """Calculate total weight of orders"""
        return sum(order.quantity for order in self.orders)
    
    def has_freshness_violations(self) -> bool:
        """Check if route has any freshness violations"""
        current_time = 0
        
        for order in self.orders:
            current_time += 2  # Assume 2 hours per delivery
            if current_time > order.freshness_limit:
                return True
        
        return False


@dataclass
class OptimizationResult:
    """Optimization result domain model"""
    routes: List[Route]
    total_distance: float
    total_time: float
    trucks_used: int
    freshness_violations: int
    fitness_score: float
    processing_time: float
    
    def summary(self) -> dict:
        """Get optimization summary"""
        return {
            "total_routes": len(self.routes),
            "total_distance": self.total_distance,
            "total_time": self.total_time,
            "trucks_used": self.trucks_used,
            "freshness_violations": self.freshness_violations,
            "fitness_score": self.fitness_score,
            "processing_time": self.processing_time
        }