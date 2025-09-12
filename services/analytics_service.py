"""
Analytics service for performance analysis and reporting
"""
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

from app.models.request_models import AnalyticsRequest
from app.models.response_models import (
    AnalyticsResponse, CostAnalysis, EfficiencyMetrics
)

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Service for generating analytics and performance reports"""
    
    def __init__(self):
        # Cost parameters (configurable)
        self.fuel_cost_per_km = 0.8  # USD per km
        self.driver_cost_per_hour = 5.0  # USD per hour
        self.truck_fixed_cost = 50.0  # USD per truck per day
    
    async def generate_analytics(self, request: AnalyticsRequest) -> AnalyticsResponse:
        """Generate comprehensive analytics from optimization results"""
        try:
            start_time = time.time()
            
            optimization_data = request.optimization_results
            
            # Extract basic metrics
            summary = optimization_data.get('summary', {})
            delivery_plan = optimization_data.get('delivery_plan', [])
            
            cost_analysis = None
            efficiency_metrics = None
            recommendations = []
            
            # Generate cost analysis if requested
            if request.include_cost_analysis and summary and delivery_plan:
                cost_analysis = self._calculate_cost_analysis(summary, delivery_plan)
            
            # Generate efficiency metrics if requested
            if request.include_efficiency_metrics and summary and delivery_plan:
                efficiency_metrics = self._calculate_efficiency_metrics(summary, delivery_plan)
            
            # Generate recommendations
            if summary and delivery_plan:
                recommendations = self._generate_recommendations(
                    summary, delivery_plan, efficiency_metrics
                )
            
            processing_time = time.time() - start_time
            
            return AnalyticsResponse(
                success=True,
                message="Analytics generated successfully",
                cost_analysis=cost_analysis,
                efficiency_metrics=efficiency_metrics,
                recommendations=recommendations,
                processing_time_seconds=round(processing_time, 2),
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Analytics generation failed: {str(e)}")
            return AnalyticsResponse(
                success=False,
                message=f"Analytics generation failed: {str(e)}",
                cost_analysis=None,
                efficiency_metrics=None,
                recommendations=[],
                processing_time_seconds=round(time.time() - start_time, 2),
                timestamp=datetime.utcnow()
            )
    
    def _calculate_cost_analysis(self, summary: Dict, delivery_plan: List[Dict]) -> CostAnalysis:
        """Calculate detailed cost analysis"""
        try:
            total_distance = summary.get('total_distance', 0)
            trucks_used = summary.get('trucks_used', 0)
            
            # Calculate total estimated time
            total_time = sum(plan.get('estimated_time_hours', 0) for plan in delivery_plan)
            
            # Calculate costs
            fuel_cost = total_distance * self.fuel_cost_per_km
            driver_cost = total_time * self.driver_cost_per_hour
            truck_cost = trucks_used * self.truck_fixed_cost
            total_cost = fuel_cost + driver_cost + truck_cost
            
            # Calculate cost per kg
            total_quantity = sum(plan.get('total_load_kg', 0) for plan in delivery_plan)
            cost_per_kg = total_cost / total_quantity if total_quantity > 0 else 0
            
            return CostAnalysis(
                fuel_cost_usd=round(fuel_cost, 2),
                driver_cost_usd=round(driver_cost, 2),
                truck_cost_usd=round(truck_cost, 2),
                total_operational_cost_usd=round(total_cost, 2),
                cost_per_kg_usd=round(cost_per_kg, 3)
            )
            
        except Exception as e:
            logger.error(f"Cost analysis calculation failed: {e}")
            return CostAnalysis(
                fuel_cost_usd=0,
                driver_cost_usd=0,
                truck_cost_usd=0,
                total_operational_cost_usd=0,
                cost_per_kg_usd=0
            )
    
    def _calculate_efficiency_metrics(self, summary: Dict, delivery_plan: List[Dict]) -> EfficiencyMetrics:
        """Calculate efficiency metrics"""
        try:
            total_orders = summary.get('total_orders', 0)
            trucks_used = summary.get('trucks_used', 1)  # Avoid division by zero
            total_distance = summary.get('total_distance', 0)
            freshness_violations = summary.get('freshness_violations', 0)
            
            # Calculate total quantity and capacity
            total_quantity = sum(plan.get('total_load_kg', 0) for plan in delivery_plan)
            
            # Assume truck capacity (could be made configurable)
            truck_capacities = {'T001': 500, 'T002': 750, 'T003': 500, 'T004': 1000, 'T005': 750}
            total_capacity_used = 0
            total_capacity_available = 0
            
            for plan in delivery_plan:
                truck_id = plan.get('truck_id', '')
                capacity = truck_capacities.get(truck_id, 500)
                total_capacity_available += capacity
                total_capacity_used += plan.get('total_load_kg', 0)
            
            # Calculate metrics
            orders_per_truck = total_orders / trucks_used if trucks_used > 0 else 0
            kg_per_km = total_quantity / total_distance if total_distance > 0 else 0
            violation_rate = (freshness_violations / total_orders * 100) if total_orders > 0 else 0
            capacity_utilization = (total_capacity_used / total_capacity_available * 100) if total_capacity_available > 0 else 0
            
            return EfficiencyMetrics(
                orders_per_truck=round(orders_per_truck, 2),
                kg_per_km=round(kg_per_km, 2),
                violation_rate_percent=round(violation_rate, 1),
                capacity_utilization_percent=round(capacity_utilization, 1)
            )
            
        except Exception as e:
            logger.error(f"Efficiency metrics calculation failed: {e}")
            return EfficiencyMetrics(
                orders_per_truck=0,
                kg_per_km=0,
                violation_rate_percent=0,
                capacity_utilization_percent=0
            )
    
    def _generate_recommendations(
        self, 
        summary: Dict, 
        delivery_plan: List[Dict], 
        efficiency_metrics: Optional[EfficiencyMetrics]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        try:
            # Freshness violations check
            freshness_violations = summary.get('freshness_violations', 0)
            if freshness_violations == 0:
                recommendations.append("✓ Excellent freshness compliance - all deliveries within time limits")
            else:
                recommendations.append(
                    f"⚠ {freshness_violations} freshness violations detected. Consider revising routes or adding trucks for time-sensitive orders"
                )
            
            # Capacity utilization check
            if efficiency_metrics:
                capacity_util = efficiency_metrics.capacity_utilization_percent
                
                if capacity_util < 60:
                    recommendations.append(
                        f"• Low truck utilization ({capacity_util:.1f}%). Consider consolidating routes to improve efficiency"
                    )
                elif capacity_util > 90:
                    recommendations.append(
                        f"• Very high truck utilization ({capacity_util:.1f}%). Consider adding trucks to reduce overloading"
                    )
                else:
                    recommendations.append(
                        f"✓ Good truck utilization ({capacity_util:.1f}%) - well balanced"
                    )
                
                # Distance efficiency check
                avg_distance_per_truck = summary.get('total_distance', 0) / summary.get('trucks_used', 1)
                if avg_distance_per_truck > 200:
                    recommendations.append(
                        f"• High average distance per truck ({avg_distance_per_truck:.1f} km). Consider opening additional distribution points"
                    )
                
                # Order distribution check
                if efficiency_metrics.orders_per_truck < 2:
                    recommendations.append(
                        f"• Low orders per truck ({efficiency_metrics.orders_per_truck:.1f}). Consider batching more orders per route"
                    )
                elif efficiency_metrics.orders_per_truck > 6:
                    recommendations.append(
                        f"• High orders per truck ({efficiency_metrics.orders_per_truck:.1f}). Consider splitting routes to maintain service quality"
                    )
            
            # Harbor utilization analysis
            harbor_loads = {}
            for plan in delivery_plan:
                harbor = plan.get('assigned_harbor', 'Unknown')
                harbor_loads[harbor] = harbor_loads.get(harbor, 0) + plan.get('total_load_kg', 0)
            
            if len(harbor_loads) > 1:
                max_load = max(harbor_loads.values())
                min_load = min(harbor_loads.values())
                if max_load > min_load * 2:  # Significant imbalance
                    recommendations.append(
                        "• Harbor load imbalance detected. Consider redistributing orders across harbors"
                    )
            
            # Cost efficiency recommendations
            total_cost = sum([
                summary.get('total_distance', 0) * self.fuel_cost_per_km,
                sum(plan.get('estimated_time_hours', 0) for plan in delivery_plan) * self.driver_cost_per_hour,
                summary.get('trucks_used', 0) * self.truck_fixed_cost
            ])
            
            total_quantity = sum(plan.get('total_load_kg', 0) for plan in delivery_plan)
            if total_quantity > 0:
                cost_per_kg = total_cost / total_quantity
                if cost_per_kg > 0.5:  # High cost threshold
                    recommendations.append(
                        f"• High operational cost per kg (${cost_per_kg:.3f}). Review route efficiency and truck utilization"
                    )
            
            # Default recommendation if none generated
            if not recommendations:
                recommendations.append("• Operations appear well-optimized. Continue monitoring for improvements")
                
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            recommendations.append("• Unable to generate specific recommendations due to data processing error")
        
        return recommendations
    
    async def compare_scenarios(self, base_results: Dict, alternative_results: Dict) -> Dict:
        """Compare two optimization scenarios"""
        try:
            base_summary = base_results.get('summary', {})
            alt_summary = alternative_results.get('summary', {})
            
            comparison = {
                'trucks_used': {
                    'base': base_summary.get('trucks_used', 0),
                    'alternative': alt_summary.get('trucks_used', 0),
                    'difference': alt_summary.get('trucks_used', 0) - base_summary.get('trucks_used', 0),
                    'improvement_percent': self._calculate_improvement_percent(
                        base_summary.get('trucks_used', 0), alt_summary.get('trucks_used', 0)
                    )
                },
                'total_distance': {
                    'base': base_summary.get('total_distance', 0),
                    'alternative': alt_summary.get('total_distance', 0),
                    'difference': alt_summary.get('total_distance', 0) - base_summary.get('total_distance', 0),
                    'improvement_percent': self._calculate_improvement_percent(
                        base_summary.get('total_distance', 0), alt_summary.get('total_distance', 0)
                    )
                },
                'freshness_violations': {
                    'base': base_summary.get('freshness_violations', 0),
                    'alternative': alt_summary.get('freshness_violations', 0),
                    'difference': alt_summary.get('freshness_violations', 0) - base_summary.get('freshness_violations', 0),
                    'improvement_percent': self._calculate_improvement_percent(
                        base_summary.get('freshness_violations', 0), alt_summary.get('freshness_violations', 0)
                    )
                }
            }
            
            # Determine which scenario is better
            better_scenario = self._determine_better_scenario(comparison)
            
            return {
                'comparison': comparison,
                'recommendation': better_scenario,
                'summary': self._generate_comparison_summary(comparison)
            }
            
        except Exception as e:
            logger.error(f"Scenario comparison failed: {e}")
            return {
                'error': f"Comparison failed: {str(e)}",
                'comparison': {},
                'recommendation': None,
                'summary': []
            }
    
    def _calculate_improvement_percent(self, base_value: float, alternative_value: float) -> float:
        """Calculate improvement percentage (negative means worse)"""
        if base_value == 0:
            return 0 if alternative_value == 0 else 100
        
        return round(((base_value - alternative_value) / base_value) * 100, 1)
    
    def _determine_better_scenario(self, comparison: Dict) -> str:
        """Determine which scenario performs better overall"""
        try:
            # Weight different metrics (lower is better for all these metrics)
            truck_improvement = comparison['trucks_used']['improvement_percent']
            distance_improvement = comparison['total_distance']['improvement_percent']
            violation_improvement = comparison['freshness_violations']['improvement_percent']
            
            # Weighted score (violations have highest priority)
            weighted_score = (
                truck_improvement * 0.2 +
                distance_improvement * 0.3 +
                violation_improvement * 0.5
            )
            
            if weighted_score > 5:  # Significant improvement threshold
                return "alternative"
            elif weighted_score < -5:  # Significant degradation threshold
                return "base"
            else:
                return "similar"
                
        except Exception:
            return "unknown"
    
    def _generate_comparison_summary(self, comparison: Dict) -> List[str]:
        """Generate summary of scenario comparison"""
        summary = []
        
        try:
            for metric, values in comparison.items():
                metric_name = metric.replace('_', ' ').title()
                improvement = values.get('improvement_percent', 0)
                
                if improvement > 5:
                    summary.append(f"{metric_name}: {improvement}% better in alternative scenario")
                elif improvement < -5:
                    summary.append(f"{metric_name}: {abs(improvement)}% worse in alternative scenario")
                else:
                    summary.append(f"{metric_name}: Similar performance in both scenarios")
                    
        except Exception:
            summary.append("Unable to generate detailed comparison summary")
        
        return summary
    
    async def generate_performance_report(self, optimization_results: Dict) -> Dict:
        """Generate comprehensive performance report"""
        try:
            summary = optimization_results.get('summary', {})
            delivery_plan = optimization_results.get('delivery_plan', [])
            
            # Key performance indicators
            kpis = {
                'operational_efficiency': self._calculate_operational_efficiency(summary, delivery_plan),
                'cost_efficiency': self._calculate_cost_efficiency(summary, delivery_plan),
                'service_quality': self._calculate_service_quality(summary, delivery_plan),
                'resource_utilization': self._calculate_resource_utilization(summary, delivery_plan)
            }
            
            # Overall score (0-100)
            overall_score = sum(kpis.values()) / len(kpis)
            
            return {
                'overall_score': round(overall_score, 1),
                'kpis': kpis,
                'grade': self._calculate_grade(overall_score),
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Performance report generation failed: {e}")
            return {
                'error': f"Report generation failed: {str(e)}",
                'overall_score': 0,
                'kpis': {},
                'grade': 'N/A'
            }
    
    def _calculate_operational_efficiency(self, summary: Dict, delivery_plan: List[Dict]) -> float:
        """Calculate operational efficiency score (0-100)"""
        try:
            total_distance = summary.get('total_distance', 0)
            trucks_used = summary.get('trucks_used', 1)
            total_orders = summary.get('total_orders', 0)
            
            # Efficiency metrics
            avg_distance_per_truck = total_distance / trucks_used if trucks_used > 0 else float('inf')
            orders_per_truck = total_orders / trucks_used if trucks_used > 0 else 0
            
            # Score based on reasonable benchmarks
            distance_score = max(0, min(100, 100 - (avg_distance_per_truck - 100) * 0.5))  # 100km is ideal
            order_score = min(100, orders_per_truck * 25)  # 4 orders per truck is ideal
            
            return (distance_score + order_score) / 2
            
        except Exception:
            return 0
    
    def _calculate_cost_efficiency(self, summary: Dict, delivery_plan: List[Dict]) -> float:
        """Calculate cost efficiency score (0-100)"""
        try:
            total_quantity = sum(plan.get('total_load_kg', 0) for plan in delivery_plan)
            total_distance = summary.get('total_distance', 0)
            
            if total_quantity == 0 or total_distance == 0:
                return 0
            
            kg_per_km = total_quantity / total_distance
            
            # Higher kg/km is better (more efficient)
            # Score based on benchmark of 5 kg/km being ideal
            score = min(100, (kg_per_km / 5) * 100)
            return score
            
        except Exception:
            return 0
    
    def _calculate_service_quality(self, summary: Dict, delivery_plan: List[Dict]) -> float:
        """Calculate service quality score (0-100)"""
        try:
            total_orders = summary.get('total_orders', 0)
            violations = summary.get('freshness_violations', 0)
            
            if total_orders == 0:
                return 100  # No orders, no violations
            
            violation_rate = violations / total_orders
            
            # Perfect score for no violations, decreasing with violations
            score = max(0, 100 - (violation_rate * 200))  # Penalties are severe for violations
            return score
            
        except Exception:
            return 0
    
    def _calculate_resource_utilization(self, summary: Dict, delivery_plan: List[Dict]) -> float:
        """Calculate resource utilization score (0-100)"""
        try:
            # Estimate total available capacity
            truck_capacities = {'T001': 500, 'T002': 750, 'T003': 500, 'T004': 1000, 'T005': 750}
            
            total_used_capacity = sum(plan.get('total_load_kg', 0) for plan in delivery_plan)
            total_available_capacity = 0
            
            for plan in delivery_plan:
                truck_id = plan.get('truck_id', '')
                capacity = truck_capacities.get(truck_id, 500)
                total_available_capacity += capacity
            
            if total_available_capacity == 0:
                return 0
            
            utilization_rate = total_used_capacity / total_available_capacity
            
            # Optimal utilization is around 75-85%
            if 0.75 <= utilization_rate <= 0.85:
                return 100
            elif utilization_rate < 0.75:
                return utilization_rate / 0.75 * 100
            else:  # Over 85%
                return max(0, 100 - (utilization_rate - 0.85) * 200)
            
        except Exception:
            return 0
    
    def _calculate_grade(self, overall_score: float) -> str:
        """Convert numerical score to letter grade"""
        if overall_score >= 90:
            return 'A'
        elif overall_score >= 80:
            return 'B'
        elif overall_score >= 70:
            return 'C'
        elif overall_score >= 60:
            return 'D'
        else:
            return 'F'