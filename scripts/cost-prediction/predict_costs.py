#!/usr/bin/env python3
"""
LLM deployment cost prediction with seasonal patterns and market trends.
Supports on-premise TCO and GPU service provider models.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import argparse
from dataclasses import dataclass, asdict
import logging

@dataclass
class DeploymentProfile:
    """Profile for an LLM deployment."""
    name: str
    model_size: str  # "8b", "70b", etc.
    gpu_type: str
    gpu_count: int
    deployment_type: str  # "on_premise" or "gpu_service"
    expected_requests_per_hour: int
    quantization: str = "fp16"
    utilization_target: float = 0.7

@dataclass
class CostPrediction:
    """Cost prediction results."""
    deployment_name: str
    hourly_cost: float
    monthly_cost: float
    yearly_cost: float
    cost_per_request: float
    break_even_vs_cloud: Optional[int] = None  # Months to break even

class CostPredictor:
    def __init__(self, pricing_data_path: str = "pricing_data.json"):
        """Initialize with current pricing data."""
        self.pricing_data = self._load_pricing_data(pricing_data_path)
        self.logger = logging.getLogger(__name__)
        
    def _load_pricing_data(self, path: str) -> Dict:
        """Load pricing data with market adjustments."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Generate default pricing with realistic patterns
            return self._generate_default_pricing()
    
    def _generate_default_pricing(self) -> Dict:
        """Generate realistic pricing data."""
        base_date = datetime.now()
        
        # GPU depreciation over time (Moore's law-ish)
        monthly_decline = 0.02  # 2% monthly price decline
        
        pricing = {
            "on_premise": {
                "gpu_purchase": {
                    "a100_40gb": self._apply_market_trend(15000, base_date, monthly_decline),
                    "a100_80gb": self._apply_market_trend(20000, base_date, monthly_decline),
                    "h100_80gb": self._apply_market_trend(30000, base_date, monthly_decline),
                    "v100_32gb": self._apply_market_trend(8000, base_date, monthly_decline)
                },
                "power_cost_kwh": 0.12,
                "datacenter_pue": 1.5,
                "rack_cost_monthly": 500,
                "staff_cost_hourly": 150
            },
            "gpu_service": {
                "hourly_rates": {
                    "a100_40gb": self._apply_seasonal_pattern(1.80, base_date),
                    "a100_80gb": self._apply_seasonal_pattern(2.40, base_date),
                    "h100_80gb": self._apply_seasonal_pattern(4.20, base_date),
                    "v100_32gb": self._apply_seasonal_pattern(0.90, base_date)
                },
                "spot_discount": 0.3,  # 30% discount for spot
                "commitment_discounts": {
                    "1_month": 0.1,
                    "6_month": 0.2,
                    "12_month": 0.3
                }
            },
            "last_updated": base_date.isoformat()
        }
        
        return pricing
    
    def _apply_market_trend(self, base_price: float, date: datetime, 
                          monthly_decline: float) -> float:
        """Apply market trend to pricing."""
        # Random variation ±10%
        variation = np.random.uniform(0.9, 1.1)
        return base_price * variation * (1 - monthly_decline)
    
    def _apply_seasonal_pattern(self, base_price: float, date: datetime) -> float:
        """Apply seasonal patterns to cloud pricing."""
        # Higher prices in Q4 (holiday season), lower in Q2
        month = date.month
        seasonal_multiplier = {
            1: 1.0, 2: 0.95, 3: 0.95,  # Q1
            4: 0.90, 5: 0.90, 6: 0.92,  # Q2 (lowest)
            7: 0.95, 8: 0.98, 9: 1.0,   # Q3
            10: 1.05, 11: 1.08, 12: 1.10  # Q4 (highest)
        }
        
        return base_price * seasonal_multiplier.get(month, 1.0)
    
    def predict_costs(self, profile: DeploymentProfile, 
                     duration_months: int = 12) -> CostPrediction:
        """Predict costs for a deployment profile."""
        
        if profile.deployment_type == "on_premise":
            hourly_cost = self._calculate_on_premise_hourly(profile)
        else:
            hourly_cost = self._calculate_gpu_service_hourly(profile)
        
        # Apply quantization savings
        quantization_savings = {
            "fp16": 1.0,
            "int8": 0.35,  # 65% savings
            "int4": 0.20   # 80% savings
        }
        
        hourly_cost *= quantization_savings.get(profile.quantization, 1.0)
        
        # Calculate costs
        monthly_cost = hourly_cost * 730
        yearly_cost = monthly_cost * 12
        cost_per_request = hourly_cost / profile.expected_requests_per_hour if profile.expected_requests_per_hour > 0 else 0
        
        # Calculate break-even for on-premise
        break_even_months = None
        if profile.deployment_type == "on_premise":
            cloud_hourly = self._calculate_gpu_service_hourly(profile)
            monthly_savings = (cloud_hourly - hourly_cost) * 730
            
            if monthly_savings > 0:
                total_investment = self._calculate_initial_investment(profile)
                break_even_months = int(total_investment / monthly_savings)
        
        return CostPrediction(
            deployment_name=profile.name,
            hourly_cost=round(hourly_cost, 3),
            monthly_cost=round(monthly_cost, 2),
            yearly_cost=round(yearly_cost, 2),
            cost_per_request=round(cost_per_request, 6),
            break_even_vs_cloud=break_even_months
        )
    
    def _calculate_on_premise_hourly(self, profile: DeploymentProfile) -> float:
        """Calculate on-premise hourly costs."""
        
        gpu_prices = self.pricing_data["on_premise"]["gpu_purchase"]
        gpu_price = gpu_prices.get(profile.gpu_type, 15000)
        
        # 5-year depreciation
        depreciation_hourly = (gpu_price * profile.gpu_count) / (5 * 365 * 24)
        
        # Power costs
        gpu_power_watts = {
            "a100_40gb": 400,
            "a100_80gb": 400,
            "h100_80gb": 700,
            "v100_32gb": 300
        }
        
        power_kw = gpu_power_watts.get(profile.gpu_type, 400) / 1000
        pue = self.pricing_data["on_premise"]["datacenter_pue"]
        power_cost_hourly = (power_kw * profile.gpu_count * pue * 
                           self.pricing_data["on_premise"]["power_cost_kwh"])
        
        # Infrastructure costs
        rack_hourly = self.pricing_data["on_premise"]["rack_cost_monthly"] / 730
        staff_hourly = self.pricing_data["on_premise"]["staff_cost_hourly"] / 20  # Shared
        
        total_hourly = depreciation_hourly + power_cost_hourly + rack_hourly + staff_hourly
        
        return total_hourly
    
    def _calculate_gpu_service_hourly(self, profile: DeploymentProfile) -> float:
        """Calculate GPU service hourly costs."""
        
        rates = self.pricing_data["gpu_service"]["hourly_rates"]
        base_rate = rates.get(profile.gpu_type, 1.80)
        
        # Apply commitment discounts if specified
        # For now, assume on-demand pricing
        
        total_hourly = base_rate * profile.gpu_count
        
        return total_hourly
    
    def _calculate_initial_investment(self, profile: DeploymentProfile) -> float:
        """Calculate initial investment for on-premise."""
        
        gpu_prices = self.pricing_data["on_premise"]["gpu_purchase"]
        gpu_price = gpu_prices.get(profile.gpu_type, 15000)
        
        # Include setup costs (estimated 20% of hardware)
        setup_multiplier = 1.2
        
        return gpu_price * profile.gpu_count * setup_multiplier
    
    def generate_forecast(self, profiles: List[DeploymentProfile], 
                         months: int = 24) -> pd.DataFrame:
        """Generate cost forecast for multiple profiles."""
        
        results = []
        
        for month in range(months):
            month_date = datetime.now() + timedelta(days=30 * month)
            
            for profile in profiles:
                # Adjust for growth
                growth_rate = 1.05 ** (month / 12)  # 5% annual growth
                adjusted_profile = DeploymentProfile(
                    name=profile.name,
                    model_size=profile.model_size,
                    gpu_type=profile.gpu_type,
                    gpu_count=profile.gpu_count,
                    deployment_type=profile.deployment_type,
                    expected_requests_per_hour=int(profile.expected_requests_per_hour * growth_rate),
                    quantization=profile.quantization,
                    utilization_target=profile.utilization_target
                )
                
                prediction = self.predict_costs(adjusted_profile)
                
                results.append({
                    'month': month,
                    'date': month_date,
                    'deployment': profile.name,
                    'monthly_cost': prediction.monthly_cost,
                    'requests_per_hour': adjusted_profile.expected_requests_per_hour,
                    'cost_per_request': prediction.cost_per_request
                })
        
        return pd.DataFrame(results)

def main():
    """Example cost predictions."""
    
    parser = argparse.ArgumentParser(description='Predict LLM deployment costs')
    parser.add_argument('--profile', type=str, help='Deployment profile JSON')
    parser.add_argument('--months', type=int, default=12, help='Forecast months')
    parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    # Example profiles
    profiles = [
        DeploymentProfile(
            name="llama-8b-prod",
            model_size="8b",
            gpu_type="a100_40gb",
            gpu_count=2,
            deployment_type="gpu_service",
            expected_requests_per_hour=1000,
            quantization="int8"
        ),
        DeploymentProfile(
            name="llama-70b-enterprise",
            model_size="70b",
            gpu_type="h100_80gb",
            gpu_count=8,
            deployment_type="on_premise",
            expected_requests_per_hour=5000,
            quantization="fp16"
        )
    ]
    
    predictor = CostPredictor()
    
    print("💰 LLM Deployment Cost Predictions\n")
    
    for profile in profiles:
        prediction = predictor.predict_costs(profile)
        
        print(f"📊 {profile.name}:")
        print(f"   Hourly Cost: ${prediction.hourly_cost}")
        print(f"   Monthly Cost: ${prediction.monthly_cost:,.0f}")
        print(f"   Yearly Cost: ${prediction.yearly_cost:,.0f}")
        print(f"   Cost per Request: ${prediction.cost_per_request:.6f}")
        
        if prediction.break_even_vs_cloud:
            print(f"   Break-even vs Cloud: {prediction.break_even_vs_cloud} months")
        
        print()
    
    # Generate forecast
    if args.months > 1:
        print(f"📈 Generating {args.months}-month forecast...")
        forecast = predictor.generate_forecast(profiles, args.months)
        
        if args.output:
            forecast.to_csv(args.output, index=False)
            print(f"✅ Forecast saved to {args.output}")

if __name__ == "__main__":
    main()