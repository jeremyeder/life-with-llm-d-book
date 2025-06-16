#!/usr/bin/env python3
"""
Comprehensive test suite for cost calculation accuracy.
Tests pricing models, quantization savings, and forecasting.
"""

import unittest
import json
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
from typing import Dict, List

# Import the module we're testing
from predict_costs import CostPredictor, DeploymentProfile, CostPrediction

class TestCostCalculations(unittest.TestCase):
    """Test suite for cost calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test pricing data
        self.test_pricing = {
            "on_premise": {
                "gpu_purchase": {
                    "a100_40gb": 15000,
                    "a100_80gb": 20000,
                    "h100_80gb": 30000,
                    "v100_32gb": 8000
                },
                "power_cost_kwh": 0.10,
                "datacenter_pue": 1.5,
                "rack_cost_monthly": 500,
                "staff_cost_hourly": 150
            },
            "gpu_service": {
                "hourly_rates": {
                    "a100_40gb": 1.80,
                    "a100_80gb": 2.40,
                    "h100_80gb": 4.20,
                    "v100_32gb": 0.90
                },
                "spot_discount": 0.3,
                "commitment_discounts": {
                    "1_month": 0.1,
                    "6_month": 0.2,
                    "12_month": 0.3
                }
            },
            "last_updated": datetime.now().isoformat()
        }
        
        # Create temporary file with test pricing
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.test_pricing, self.temp_file)
        self.temp_file.close()
        
        # Initialize predictor with test data
        self.predictor = CostPredictor(self.temp_file.name)
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_file.name)
    
    def test_on_premise_calculation_accuracy(self):
        """Test on-premise cost calculations."""
        profile = DeploymentProfile(
            name="test-on-prem",
            model_size="8b",
            gpu_type="a100_40gb",
            gpu_count=2,
            deployment_type="on_premise",
            expected_requests_per_hour=1000,
            quantization="fp16"
        )
        
        prediction = self.predictor.predict_costs(profile)
        
        # Manual calculation verification
        # Depreciation: (15000 * 2) / (5 * 365 * 24) = $0.684/hour
        expected_depreciation = (15000 * 2) / (5 * 365 * 24)
        
        # Power: 400W * 2 * 1.5 PUE * 0.10/kWh = $0.12/hour
        expected_power = (0.4 * 2 * 1.5 * 0.10)
        
        # Infrastructure: 500/730 + 150/20 = $8.18/hour
        expected_infra = 500/730 + 150/20
        
        expected_total = expected_depreciation + expected_power + expected_infra
        
        # Allow 1% tolerance for rounding
        self.assertAlmostEqual(prediction.hourly_cost, expected_total, places=2)
    
    def test_gpu_service_calculation_accuracy(self):
        """Test GPU service cost calculations."""
        profile = DeploymentProfile(
            name="test-gpu-service",
            model_size="70b",
            gpu_type="h100_80gb",
            gpu_count=4,
            deployment_type="gpu_service",
            expected_requests_per_hour=5000,
            quantization="fp16"
        )
        
        prediction = self.predictor.predict_costs(profile)
        
        # Expected: 4.20 * 4 = $16.80/hour
        expected_hourly = 4.20 * 4
        
        self.assertEqual(prediction.hourly_cost, expected_hourly)
        self.assertEqual(prediction.monthly_cost, expected_hourly * 730)
    
    def test_quantization_savings(self):
        """Test quantization cost savings calculations."""
        base_profile = DeploymentProfile(
            name="test-quant",
            model_size="8b",
            gpu_type="a100_40gb",
            gpu_count=1,
            deployment_type="gpu_service",
            expected_requests_per_hour=1000,
            quantization="fp16"
        )
        
        # Test different quantization levels
        quantization_tests = [
            ("fp16", 1.0),      # No savings
            ("int8", 0.35),     # 65% savings
            ("int4", 0.20)      # 80% savings
        ]
        
        for quant, expected_multiplier in quantization_tests:
            profile = DeploymentProfile(
                name=f"test-{quant}",
                model_size="8b",
                gpu_type="a100_40gb",
                gpu_count=1,
                deployment_type="gpu_service",
                expected_requests_per_hour=1000,
                quantization=quant
            )
            
            prediction = self.predictor.predict_costs(profile)
            expected_cost = 1.80 * expected_multiplier
            
            self.assertAlmostEqual(prediction.hourly_cost, expected_cost, places=3)
    
    def test_cost_per_request_calculation(self):
        """Test cost per request calculations."""
        test_cases = [
            (100, 1.80, 0.018),    # Low volume
            (1000, 1.80, 0.0018),  # Medium volume
            (10000, 1.80, 0.00018) # High volume
        ]
        
        for requests_per_hour, hourly_cost, expected_cpr in test_cases:
            profile = DeploymentProfile(
                name="test-cpr",
                model_size="8b",
                gpu_type="a100_40gb",
                gpu_count=1,
                deployment_type="gpu_service",
                expected_requests_per_hour=requests_per_hour,
                quantization="fp16"
            )
            
            prediction = self.predictor.predict_costs(profile)
            
            self.assertAlmostEqual(prediction.cost_per_request, expected_cpr, places=6)
    
    def test_break_even_calculation(self):
        """Test break-even calculation for on-premise vs cloud."""
        profile = DeploymentProfile(
            name="test-breakeven",
            model_size="8b",
            gpu_type="a100_40gb",
            gpu_count=4,
            deployment_type="on_premise",
            expected_requests_per_hour=5000,
            quantization="fp16"
        )
        
        prediction = self.predictor.predict_costs(profile)
        
        # Should have break-even calculation
        self.assertIsNotNone(prediction.break_even_vs_cloud)
        
        # Verify it's reasonable (between 6-36 months typically)
        self.assertGreaterEqual(prediction.break_even_vs_cloud, 6)
        self.assertLessEqual(prediction.break_even_vs_cloud, 36)
    
    def test_seasonal_pricing_patterns(self):
        """Test seasonal pricing variations."""
        # Test Q4 vs Q2 pricing
        predictor = CostPredictor()  # Use default pricing
        
        # Simulate Q4 (December)
        december = datetime(2024, 12, 15)
        q4_price = predictor._apply_seasonal_pattern(1.80, december)
        
        # Simulate Q2 (May)
        may = datetime(2024, 5, 15)
        q2_price = predictor._apply_seasonal_pattern(1.80, may)
        
        # Q4 should be more expensive than Q2
        self.assertGreater(q4_price, q2_price)
        
        # Verify expected multipliers
        self.assertAlmostEqual(q4_price, 1.80 * 1.10, places=2)  # +10% in Dec
        self.assertAlmostEqual(q2_price, 1.80 * 0.90, places=2)  # -10% in May
    
    def test_forecast_generation(self):
        """Test multi-month forecast generation."""
        profiles = [
            DeploymentProfile(
                name="test-forecast",
                model_size="8b",
                gpu_type="a100_40gb",
                gpu_count=2,
                deployment_type="gpu_service",
                expected_requests_per_hour=1000,
                quantization="int8"
            )
        ]
        
        forecast = self.predictor.generate_forecast(profiles, months=12)
        
        # Verify forecast structure
        self.assertEqual(len(forecast), 12)
        self.assertIn('monthly_cost', forecast.columns)
        self.assertIn('cost_per_request', forecast.columns)
        
        # Verify growth is applied
        first_month_requests = forecast.iloc[0]['requests_per_hour']
        last_month_requests = forecast.iloc[-1]['requests_per_hour']
        
        # Should show ~5% growth over 12 months
        expected_growth = first_month_requests * 1.05
        self.assertAlmostEqual(last_month_requests, expected_growth, delta=50)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Zero requests
        profile = DeploymentProfile(
            name="test-zero",
            model_size="8b",
            gpu_type="a100_40gb",
            gpu_count=1,
            deployment_type="gpu_service",
            expected_requests_per_hour=0,
            quantization="fp16"
        )
        
        prediction = self.predictor.predict_costs(profile)
        self.assertEqual(prediction.cost_per_request, 0)
        
        # Unknown GPU type
        profile = DeploymentProfile(
            name="test-unknown",
            model_size="8b",
            gpu_type="unknown_gpu",
            gpu_count=1,
            deployment_type="gpu_service",
            expected_requests_per_hour=1000,
            quantization="fp16"
        )
        
        # Should use default pricing
        prediction = self.predictor.predict_costs(profile)
        self.assertGreater(prediction.hourly_cost, 0)

class TestPropertyBasedCostCalculations(unittest.TestCase):
    """Property-based tests for cost calculations."""
    
    def setUp(self):
        """Set up property-based tests."""
        self.predictor = CostPredictor()
    
    def test_cost_monotonicity(self):
        """Test that costs increase monotonically with resources."""
        base_profile = DeploymentProfile(
            name="test-mono",
            model_size="8b",
            gpu_type="a100_40gb",
            gpu_count=1,
            deployment_type="gpu_service",
            expected_requests_per_hour=1000,
            quantization="fp16"
        )
        
        costs = []
        for gpu_count in range(1, 5):
            profile = DeploymentProfile(
                name=f"test-mono-{gpu_count}",
                model_size="8b",
                gpu_type="a100_40gb",
                gpu_count=gpu_count,
                deployment_type="gpu_service",
                expected_requests_per_hour=1000,
                quantization="fp16"
            )
            
            prediction = self.predictor.predict_costs(profile)
            costs.append(prediction.hourly_cost)
        
        # Costs should increase monotonically
        for i in range(1, len(costs)):
            self.assertGreater(costs[i], costs[i-1])
    
    def test_quantization_ordering(self):
        """Test that quantization savings are properly ordered."""
        quantization_levels = ["fp16", "int8", "int4"]
        costs = []
        
        for quant in quantization_levels:
            profile = DeploymentProfile(
                name=f"test-quant-{quant}",
                model_size="8b",
                gpu_type="a100_40gb",
                gpu_count=1,
                deployment_type="gpu_service",
                expected_requests_per_hour=1000,
                quantization=quant
            )
            
            prediction = self.predictor.predict_costs(profile)
            costs.append(prediction.hourly_cost)
        
        # Costs should decrease with more aggressive quantization
        self.assertGreater(costs[0], costs[1])  # fp16 > int8
        self.assertGreater(costs[1], costs[2])  # int8 > int4
    
    def test_cost_consistency(self):
        """Test that repeated calculations produce consistent results."""
        profile = DeploymentProfile(
            name="test-consistency",
            model_size="8b",
            gpu_type="a100_40gb",
            gpu_count=2,
            deployment_type="on_premise",
            expected_requests_per_hour=1000,
            quantization="fp16"
        )
        
        # Calculate costs multiple times
        costs = []
        for _ in range(10):
            predictor = CostPredictor()
            prediction = predictor.predict_costs(profile)
            costs.append(prediction.hourly_cost)
        
        # All costs should be within 10% (due to random variations)
        mean_cost = np.mean(costs)
        for cost in costs:
            self.assertAlmostEqual(cost, mean_cost, delta=mean_cost * 0.1)

def run_tests():
    """Run all tests and generate report."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCostCalculations))
    suite.addTests(loader.loadTestsFromTestCase(TestPropertyBasedCostCalculations))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)