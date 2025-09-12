"""
Route calculation and management service
"""
import logging
from typing import List, Dict, Tuple, Optional
import httpx
import asyncio
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RouteService:
    """Service for route calculations and management"""
    
    def __init__(self):
        self.google_maps_api_key = settings.google_maps_api_key
        self.base_url = "https://maps.googleapis.com/maps/api/distancematrix/json"
        
        # Sri Lankan city coordinates (cached for fallback)
        self.city_coordinates = {
            'Colombo': (6.9271, 79.8612),
            'Kandy': (7.2906, 80.6337),
            'Galle': (6.0535, 80.2210),
            'Jaffna': (9.6615, 80.0255),
            'Anuradhapura': (8.3114, 80.4037),
            'Negombo': (7.2083, 79.8358),
            'Kurunegala': (7.4818, 80.3609),
            'Ratnapura': (6.6828, 80.4047),
            'Batticaloa': (7.7102, 81.6924),
            'Trincomalee': (8.5874, 81.2152)
        }
        
        # Harbor coordinates
        self.harbor_coordinates = {
            'Negombo': (7.2083, 79.8358),
            'Colombo': (6.9271, 79.8612),
            'Beruwala': (6.4788, 79.9828)
        }
    
    async def calculate_distance_matrix(
        self,
        origins: List[str],
        destinations: List[str],
        use_api: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """Calculate distance matrix between origins and destinations"""
        
        if use_api and self.google_maps_api_key:
            try:
                return await self._get_api_distances(origins, destinations)
            except Exception as e:
                logger.warning(f"Google Maps API failed: {e}, falling back to haversine")
                return self._calculate_haversine_distances(origins, destinations)
        else:
            return self._calculate_haversine_distances(origins, destinations)
    
    async def _get_api_distances(
        self,
        origins: List[str],
        destinations: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Get distances using Google Maps API"""
        
        # Convert location names to coordinates
        origin_coords = [self._get_coordinates(origin) for origin in origins]
        dest_coords = [self._get_coordinates(dest) for dest in destinations]
        
        # Format coordinates for API
        origins_str = "|".join([f"{lat},{lng}" for lat, lng in origin_coords])
        destinations_str = "|".join([f"{lat},{lng}" for lat, lng in dest_coords])
        
        params = {
            'origins': origins_str,
            'destinations': destinations_str,
            'key': self.google_maps_api_key,
            'units': 'metric',
            'mode': 'driving'
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
        
        if data['status'] != 'OK':
            raise Exception(f"API Error: {data['status']}")
        
        # Parse response into matrix
        matrix = {}
        for i, origin in enumerate(origins):
            matrix[origin] = {}
            for j, destination in enumerate(destinations):
                element = data['rows'][i]['elements'][j]
                
                if element['status'] == 'OK':
                    distance_km = element['distance']['value'] / 1000
                    matrix[origin][destination] = distance_km
                else:
                    # Fallback to haversine calculation
                    origin_coord = self._get_coordinates(origin)
                    dest_coord = self._get_coordinates(destination)
                    matrix[origin][destination] = self._haversine_distance(
                        origin_coord, dest_coord
                    )
        
        return matrix
    
    def _calculate_haversine_distances(
        self,
        origins: List[str],
        destinations: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate distances using Haversine formula"""
        
        matrix = {}
        for origin in origins:
            matrix[origin] = {}
            origin_coord = self._get_coordinates(origin)
            
            for destination in destinations:
                dest_coord = self._get_coordinates(destination)
                distance = self._haversine_distance(origin_coord, dest_coord)
                matrix[origin][destination] = distance
        
        return matrix
    
    def _get_coordinates(self, location: str) -> Tuple[float, float]:
        """Get coordinates for a location"""
        # Try harbor coordinates first
        if location in self.harbor_coordinates:
            return self.harbor_coordinates[location]
        
        # Try city coordinates
        if location in self.city_coordinates:
            return self.city_coordinates[location]
        
        # Default to Colombo
        logger.warning(f"Unknown location: {location}, using Colombo coordinates")
        return self.city_coordinates['Colombo']
    
    def _haversine_distance(
        self,
        coord1: Tuple[float, float],
        coord2: Tuple[float, float]
    ) -> float:
        """Calculate Haversine distance between two coordinates"""
        import math
        
        lat1, lng1 = coord1
        lat2, lng2 = coord2
        
        R = 6371  # Earth's radius in km
        
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    async def optimize_route_order(
        self,
        start_location: str,
        destinations: List[str]
    ) -> List[str]:
        """Optimize the order of destinations for a single route"""
        
        if len(destinations) <= 2:
            return destinations
        
        # Calculate distance matrix
        all_locations = [start_location] + destinations
        distance_matrix = await self.calculate_distance_matrix(all_locations, all_locations)
        
        # Simple greedy nearest neighbor algorithm
        route = []
        current_location = start_location
        remaining_destinations = destinations.copy()
        
        while remaining_destinations:
            nearest_dest = min(
                remaining_destinations,
                key=lambda dest: distance_matrix[current_location][dest]
            )
            
            route.append(nearest_dest)
            remaining_destinations.remove(nearest_dest)
            current_location = nearest_dest
        
        return route
    
    def calculate_route_stats(
        self,
        route: List[str],
        distance_matrix: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate route statistics"""
        
        if len(route) < 2:
            return {
                'total_distance': 0,
                'estimated_time': 0,
                'average_speed': 50  # km/h
            }
        
        total_distance = 0
        for i in range(len(route) - 1):
            from_location = route[i]
            to_location = route[i + 1]
            
            if from_location in distance_matrix and to_location in distance_matrix[from_location]:
                total_distance += distance_matrix[from_location][to_location]
        
        # Estimate time (assuming 50 km/h average + 30 min per stop)
        driving_time = total_distance / 50  # hours
        stop_time = (len(route) - 1) * 0.5  # 30 minutes per stop
        total_time = driving_time + stop_time
        
        return {
            'total_distance': round(total_distance, 2),
            'estimated_time': round(total_time, 2),
            'average_speed': 50
        }
    
    def get_supported_locations(self) -> Dict[str, List[str]]:
        """Get list of supported locations"""
        return {
            'cities': list(self.city_coordinates.keys()),
            'harbors': list(self.harbor_coordinates.keys())
        }