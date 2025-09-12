"""
Core optimization service - refactored from original code for production use
"""
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
import time

from app.models.request_models import OrderRequest, OptimizationParameters
from app.models.response_models import OptimizationResponse, OptimizationSummary, TruckRoute, OrderInfo
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class Order:
    """Domain model for orders"""
    def __init__(self, order_id: str, city: str, quantity: float, freshness_limit: float):
        self.order_id = order_id
        self.city = city
        self.quantity = quantity
        self.freshness_limit = freshness_limit
        self.assigned_harbor = None


class Harbor:
    """Domain model for harbors"""
    def __init__(self, name: str, location: Tuple[float, float], capacity: float):
        self.name = name
        self.location = location
        self.capacity = capacity
        self.available_fish = capacity


class Truck:
    """Domain model for trucks"""
    def __init__(self, truck_id: str, capacity: float, cost_per_km: float):
        self.truck_id = truck_id
        self.capacity = capacity
        self.cost_per_km = cost_per_km


class OptimizedABC:
    """Optimized Artificial Bee Colony algorithm for production use"""
    
    def __init__(self, n_bees: int = 50, max_iterations: int = 100):
        self.n_bees = n_bees
        self.max_iterations = max_iterations
        self.limit = 10
        
        # Sri Lankan city coordinates (cached)
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
        
        # Harbor coordinates (cached)
        self.harbor_coordinates = {
            'Negombo': (7.2083, 79.8358),
            'Colombo': (6.9271, 79.8612),
            'Beruwala': (6.4788, 79.9828)
        }
    
    async def optimize_routes(self, orders: List[Order], harbors: List[Harbor], 
                            trucks: List[Truck]) -> Dict:
        """Async wrapper for route optimization"""
        loop = asyncio.get_event_loop()
        
        # Run optimization in thread pool to avoid blocking
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = await loop.run_in_executor(
                executor, self._optimize_routes_sync, orders, harbors, trucks
            )
        
        return result
    
    def _optimize_routes_sync(self, orders: List[Order], harbors: List[Harbor], 
                            trucks: List[Truck]) -> Dict:
        """Synchronous route optimization implementation"""
        start_time = time.time()
        
        # Assign orders to harbors
        self._assign_orders_to_harbors(orders, harbors)
        
        # Initialize population
        population = []
        for _ in range(self.n_bees):
            solution = self._generate_random_solution(orders, trucks)
            fitness = self._calculate_fitness(solution)
            population.append({'solution': solution, 'fitness': fitness, 'trial': 0})
        
        best_solution = None
        best_fitness = float('inf')
        
        # ABC optimization loop
        for iteration in range(self.max_iterations):
            # Employed bees phase
            for i in range(self.n_bees // 2):
                new_solution = self._generate_neighbor_solution(population[i]['solution'])
                new_fitness = self._calculate_fitness(new_solution)
                
                if new_fitness < population[i]['fitness']:
                    population[i] = {'solution': new_solution, 'fitness': new_fitness, 'trial': 0}
                else:
                    population[i]['trial'] += 1
            
            # Onlooker bees phase
            fitness_values = [bee['fitness'] for bee in population]
            probabilities = self._calculate_probabilities(fitness_values)
            
            for _ in range(self.n_bees // 2):
                selected_index = self._roulette_wheel_selection(probabilities)
                new_solution = self._generate_neighbor_solution(population[selected_index]['solution'])
                new_fitness = self._calculate_fitness(new_solution)
                
                if new_fitness < population[selected_index]['fitness']:
                    population[selected_index] = {'solution': new_solution, 'fitness': new_fitness, 'trial': 0}
                else:
                    population[selected_index]['trial'] += 1
            
            # Scout bees phase
            for i, bee in enumerate(population):
                if bee['trial'] > self.limit:
                    new_solution = self._generate_random_solution(orders, trucks)
                    new_fitness = self._calculate_fitness(new_solution)
                    population[i] = {'solution': new_solution, 'fitness': new_fitness, 'trial': 0}
            
            # Update best solution
            current_best = min(population, key=lambda x: x['fitness'])
            if current_best['fitness'] < best_fitness:
                best_fitness = current_best['fitness']
                best_solution = current_best['solution'].copy()
        
        processing_time = time.time() - start_time
        
        return {
            'solution': best_solution,
            'fitness': best_fitness,
            'total_distance': self._calculate_total_distance(best_solution),
            'trucks_used': len([truck for truck in best_solution if best_solution[truck]]),
            'freshness_violations': self._calculate_freshness_violations(best_solution),
            'processing_time': processing_time
        }
    
    def _assign_orders_to_harbors(self, orders: List[Order], harbors: List[Harbor]):
        """Assign orders to harbors based on proximity and capacity"""
        for order in orders:
            city_location = self.city_coordinates[order.city]
            
            # Calculate distances to all harbors
            distances = []
            for harbor in harbors:
                dist = self._haversine_distance(city_location, harbor.location)
                distances.append((dist, harbor))
            
            # Sort by distance
            distances.sort(key=lambda x: x[0])
            
            # Assign to closest harbor with capacity
            for _, harbor in distances:
                if harbor.available_fish >= order.quantity:
                    order.assigned_harbor = harbor.name
                    harbor.available_fish -= order.quantity
                    break
            
            # Fallback to closest harbor if none have capacity
            if not order.assigned_harbor:
                order.assigned_harbor = distances[0][1].name
    
    def _generate_random_solution(self, orders: List[Order], trucks: List[Truck]) -> Dict:
        """Generate random truck assignments"""
        solution = {truck.truck_id: [] for truck in trucks}
        
        for order in orders:
            available_trucks = [
                truck for truck in trucks 
                if sum(o.quantity for o in solution[truck.truck_id]) + order.quantity <= truck.capacity
            ]
            
            if available_trucks:
                selected_truck = random.choice(available_trucks)
                solution[selected_truck.truck_id].append(order)
            else:
                # Assign to truck with most remaining capacity
                best_truck = max(
                    trucks, 
                    key=lambda t: t.capacity - sum(o.quantity for o in solution[t.truck_id])
                )
                solution[best_truck.truck_id].append(order)
        
        return solution
    
    def _generate_neighbor_solution(self, solution: Dict) -> Dict:
        """Generate neighbor by swapping orders"""
        new_solution = {truck_id: orders.copy() for truck_id, orders in solution.items()}
        
        trucks_with_orders = [truck_id for truck_id, orders in new_solution.items() if orders]
        
        if len(trucks_with_orders) >= 2:
            truck1, truck2 = random.sample(trucks_with_orders, 2)
            
            if new_solution[truck1] and new_solution[truck2]:
                order1 = random.choice(new_solution[truck1])
                order2 = random.choice(new_solution[truck2])
                
                new_solution[truck1].remove(order1)
                new_solution[truck2].remove(order2)
                new_solution[truck1].append(order2)
                new_solution[truck2].append(order1)
        
        return new_solution
    
    def _calculate_fitness(self, solution: Dict) -> float:
        """Calculate solution fitness"""
        total_distance = self._calculate_total_distance(solution)
        truck_usage_cost = len([truck for truck in solution if solution[truck]]) * 100
        freshness_penalty = self._calculate_freshness_violations(solution) * 1000
        
        return total_distance + truck_usage_cost + freshness_penalty
    
    def _calculate_total_distance(self, solution: Dict) -> float:
        """Calculate total route distance"""
        total_distance = 0
        
        for truck_id, orders in solution.items():
            if not orders:
                continue
            
            # Get harbor location
            harbor_coords = self.harbor_coordinates.get(
                orders[0].assigned_harbor, 
                (6.9271, 79.8612)  # Fallback to Colombo
            )
            
            # Calculate route distance
            current_location = harbor_coords
            for order in orders:
                next_location = self.city_coordinates[order.city]
                distance = self._haversine_distance(current_location, next_location)
                total_distance += distance
                current_location = next_location
        
        return total_distance
    
    def _calculate_freshness_violations(self, solution: Dict) -> int:
        """Calculate freshness violations"""
        violations = 0
        
        for truck_id, orders in solution.items():
            if not orders:
                continue
            
            harbor_coords = self.harbor_coordinates.get(
                orders[0].assigned_harbor, 
                (6.9271, 79.8612)
            )
            
            current_time = 0
            current_location = harbor_coords
            
            for order in orders:
                next_location = self.city_coordinates[order.city]
                travel_distance = self._haversine_distance(current_location, next_location)
                travel_time = travel_distance / 50  # 50 km/h average speed
                current_time += travel_time
                
                if current_time > order.freshness_limit:
                    violations += 1
                
                current_location = next_location
        
        return violations
    
    def _calculate_probabilities(self, fitness_values: List[float]) -> List[float]:
        """Calculate selection probabilities"""
        if not fitness_values:
            return []
        
        max_fitness = max(fitness_values)
        adjusted_fitness = [max_fitness - f + 1 for f in fitness_values]
        total_fitness = sum(adjusted_fitness)
        
        if total_fitness == 0:
            return [1.0 / len(fitness_values)] * len(fitness_values)
        
        return [f / total_fitness for f in adjusted_fitness]
    
    def _roulette_wheel_selection(self, probabilities: List[float]) -> int:
        """Roulette wheel selection"""
        if not probabilities:
            return 0
        
        r = random.random()
        cumulative_prob = 0
        
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if r <= cumulative_prob:
                return i
        
        return len(probabilities) - 1
    
    def _haversine_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate Haversine distance"""
        lat1, lng1 = coord1
        lat2, lng2 = coord2
        
        R = 6371  # Earth's radius in km
        
        dlat = np.radians(lat2 - lat1)
        dlng = np.radians(lng2 - lng1)
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlng/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c


class OptimizationService:
    """Main optimization service for production use"""
    
    def __init__(self):
        self.harbors = [
            Harbor("Negombo", (7.2083, 79.8358), 2000),
            Harbor("Colombo", (6.9271, 79.8612), 3000),
            Harbor("Beruwala", (6.4788, 79.9828), 1500)
        ]
        
        self.trucks = [
            Truck("T001", 500, 0.5),
            Truck("T002", 750, 0.6),
            Truck("T003", 500, 0.5),
            Truck("T004", 1000, 0.8),
            Truck("T005", 750, 0.6)
        ]
    
    async def optimize_distribution(
        self, 
        order_requests: List[OrderRequest],
        params: OptimizationParameters = None
    ) -> OptimizationResponse:
        """Main optimization entry point"""
        try:
            start_time = time.time()
            
            # Convert request models to domain models
            orders = [
                Order(req.order_id, req.city, req.quantity, req.freshness_limit)
                for req in order_requests
            ]
            
            # Reset harbor capacities
            for harbor in self.harbors:
                harbor.available_fish = harbor.capacity
            
            # Initialize ABC optimizer with parameters
            if params:
                abc_optimizer = OptimizedABC(
                    n_bees=params.n_bees,
                    max_iterations=params.max_iterations
                )
            else:
                abc_optimizer = OptimizedABC()
            
            # Run optimization
            result = await abc_optimizer.optimize_routes(orders, self.harbors, self.trucks)
            
            # Generate delivery plan
            delivery_plan = self._generate_delivery_plan(result['solution'])
            
            # Create summary
            summary = OptimizationSummary(
                total_orders=len(orders),
                trucks_used=result['trucks_used'],
                total_distance=round(result['total_distance'], 2),
                freshness_violations=result['freshness_violations'],
                optimization_fitness=round(result['fitness'], 2)
            )
            
            processing_time = time.time() - start_time
            
            return OptimizationResponse(
                success=True,
                message="Optimization completed successfully",
                summary=summary,
                delivery_plan=delivery_plan,
                processing_time_seconds=round(processing_time, 2),
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            return OptimizationResponse(
                success=False,
                message=f"Optimization failed: {str(e)}",
                summary=OptimizationSummary(
                    total_orders=len(order_requests),
                    trucks_used=0,
                    total_distance=0,
                    freshness_violations=0,
                    optimization_fitness=0
                ),
                delivery_plan=[],
                processing_time_seconds=round(time.time() - start_time, 2),
                timestamp=datetime.utcnow()
            )
    
    def _generate_delivery_plan(self, solution: Dict) -> List[TruckRoute]:
        """Generate delivery plan from optimization solution"""
        delivery_plan = []
        
        for truck_id, orders in solution.items():
            if not orders:
                continue
            
            # Calculate route details
            harbor_name = orders[0].assigned_harbor if orders else "Colombo"
            route_cities = [order.city for order in orders]
            total_load = sum(order.quantity for order in orders)
            
            # Calculate distance and time (simplified)
            total_distance = self._calculate_route_distance(harbor_name, route_cities)
            estimated_time = total_distance / 50  # 50 km/h average
            
            # Check for freshness violations
            violations = self._check_freshness_violations(harbor_name, orders)
            
            # Convert orders to response format
            order_infos = [
                OrderInfo(
                    id=order.order_id,
                    city=order.city,
                    quantity=order.quantity,
                    freshness_limit=order.freshness_limit
                )
                for order in orders
            ]
            
            delivery_plan.append(TruckRoute(
                truck_id=truck_id,
                assigned_harbor=harbor_name,
                route=route_cities,
                total_load_kg=round(total_load, 2),
                total_distance_km=round(total_distance, 2),
                estimated_time_hours=round(estimated_time, 2),
                freshness_violations=violations,
                orders=order_infos
            ))
        
        return delivery_plan
    
    def _calculate_route_distance(self, harbor_name: str, cities: List[str]) -> float:
        """Calculate total route distance"""
        if not cities:
            return 0
        
        harbor_coords = {
            'Negombo': (7.2083, 79.8358),
            'Colombo': (6.9271, 79.8612),
            'Beruwala': (6.4788, 79.9828)
        }
        
        city_coords = {
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
        
        current_location = harbor_coords.get(harbor_name, (6.9271, 79.8612))
        total_distance = 0
        
        for city in cities:
            next_location = city_coords[city]
            distance = self._haversine_distance(current_location, next_location)
            total_distance += distance
            current_location = next_location
        
        return total_distance
    
    def _check_freshness_violations(self, harbor_name: str, orders: List[Order]) -> List[str]:
        """Check for freshness violations"""
        violations = []
        current_time = 0
        
        for order in orders:
            # Simplified travel time calculation
            current_time += 2  # Assume 2 hours per delivery on average
            
            if current_time > order.freshness_limit:
                violations.append(order.order_id)
        
        return violations
    
    def _haversine_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate Haversine distance"""
        lat1, lng1 = coord1
        lat2, lng2 = coord2
        
        R = 6371  # Earth's radius in km
        
        dlat = np.radians(lat2 - lat1)
        dlng = np.radians(lng2 - lng1)
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlng/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c