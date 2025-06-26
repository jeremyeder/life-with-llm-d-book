#!/usr/bin/env python3
"""
Comprehensive Test Suite for LLM Cost Calculator

Tests cost modeling, GPU requirement calculation, provider comparison,
and ROI analysis functionality for LLM deployment optimization.

Coverage:
- Memory requirement calculations and scaling
- GPU sizing optimization for throughput and memory
- Multi-cloud cost comparison and optimization
- Cost breakdown analysis and forecasting
- Performance vs cost tradeoff analysis
"""

import pytest
import math
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

# Add the docs directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "docs"))

try:
    from cost_optimization.llm_cost_calculator import (
        LLMCostCalculator, ModelSpecs, GPUSpecs, DeploymentConfig,
        CloudProvider, GPUType
    )
except ImportError:
    # For testing, define minimal versions
    from enum import Enum
    from dataclasses import dataclass
    
    class CloudProvider(Enum):
        COREWEAVE = "coreweave"
        LAMBDA_LABS = "lambda_labs"
        AWS = "aws"
        GCP = "gcp"
        AZURE = "azure"
    
    class GPUType(Enum):
        A100_40GB = "a100_40gb"
        A100_80GB = "a100_80gb"
        H100_80GB = "h100_80gb"
        V100_32GB = "v100_32gb"
        T4_16GB = "t4_16gb"
    
    @dataclass
    class ModelSpecs:
        name: str
        parameters: int
        memory_per_param_bytes: float = 2.0
        context_length: int = 4096
        batch_size: int = 1
    
    @dataclass
    class GPUSpecs:
        gpu_type: GPUType
        memory_gb: int
        compute_capability: float
        hourly_cost_usd: float
        provider: CloudProvider
    
    @dataclass
    class DeploymentConfig:
        model: ModelSpecs
        gpu: GPUSpecs
        replicas: int
        utilization_target: float = 0.7
        availability_zone: str = "us-east-1"
    
    # Skip test if module not available
    LLMCostCalculator = None


class TestModelSpecs:
    """Test cases for ModelSpecs dataclass."""
    
    def test_model_specs_creation(self):
        """Test creating ModelSpecs instance."""
        model = ModelSpecs(
            name="test-model",
            parameters=7_000_000_000,
            memory_per_param_bytes=2.0,
            context_length=4096,
            batch_size=1
        )
        
        assert model.name == "test-model"
        assert model.parameters == 7_000_000_000
        assert model.memory_per_param_bytes == 2.0
        assert model.context_length == 4096
        assert model.batch_size == 1
    
    def test_model_specs_defaults(self):
        """Test ModelSpecs default values."""
        model = ModelSpecs(name="test", parameters=1_000_000)
        
        assert model.memory_per_param_bytes == 2.0  # FP16 default
        assert model.context_length == 4096
        assert model.batch_size == 1


class TestGPUSpecs:
    """Test cases for GPUSpecs dataclass."""
    
    def test_gpu_specs_creation(self):
        """Test creating GPUSpecs instance."""
        gpu = GPUSpecs(
            gpu_type=GPUType.A100_40GB,
            memory_gb=40,
            compute_capability=8.0,
            hourly_cost_usd=1.89,
            provider=CloudProvider.COREWEAVE
        )
        
        assert gpu.gpu_type == GPUType.A100_40GB
        assert gpu.memory_gb == 40
        assert gpu.compute_capability == 8.0
        assert gpu.hourly_cost_usd == 1.89
        assert gpu.provider == CloudProvider.COREWEAVE


class TestLLMCostCalculator:
    """Test cases for LLMCostCalculator class."""
    
    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
        if LLMCostCalculator is None:
            pytest.skip("LLMCostCalculator module not available")
        return LLMCostCalculator()
    
    @pytest.fixture
    def small_model(self):
        """Create a small model for testing."""
        return ModelSpecs(
            name="small-model",
            parameters=1_000_000_000,  # 1B parameters
            memory_per_param_bytes=2.0,
            context_length=2048,
            batch_size=1
        )
    
    @pytest.fixture
    def large_model(self):
        """Create a large model for testing."""
        return ModelSpecs(
            name="large-model",
            parameters=70_000_000_000,  # 70B parameters
            memory_per_param_bytes=2.0,
            context_length=4096,
            batch_size=4
        )
    
    def test_calculator_initialization(self, calculator):
        """Test calculator initializes with pricing data."""
        assert CloudProvider.COREWEAVE in calculator.gpu_pricing
        assert CloudProvider.AWS in calculator.gpu_pricing
        assert GPUType.A100_40GB in calculator.gpu_pricing[CloudProvider.COREWEAVE]
        
        # Check standard models
        assert "llama-3.1-8b" in calculator.standard_models
        assert "llama-3.1-70b" in calculator.standard_models
    
    def test_calculate_memory_requirements_small_model(self, calculator, small_model):
        """Test memory calculation for small model."""
        memory_req = calculator.calculate_memory_requirements(small_model)
        
        # Check all components are present
        assert "model_memory_gb" in memory_req
        assert "kv_cache_gb" in memory_req
        assert "activation_memory_gb" in memory_req
        assert "system_overhead_gb" in memory_req
        assert "total_memory_gb" in memory_req
        
        # Verify calculations
        # Model memory: 1B params * 2 bytes / 1024^3
        expected_model_memory = (1_000_000_000 * 2.0) / (1024**3)
        assert abs(memory_req["model_memory_gb"] - expected_model_memory) < 0.01
        
        # System overhead should be 4GB
        assert memory_req["system_overhead_gb"] == 4.0
        
        # Total should be sum of all components
        total = sum([
            memory_req["model_memory_gb"],
            memory_req["kv_cache_gb"],
            memory_req["activation_memory_gb"],
            memory_req["system_overhead_gb"]
        ])
        assert abs(memory_req["total_memory_gb"] - total) < 0.01
    
    def test_calculate_memory_requirements_large_model(self, calculator, large_model):
        """Test memory calculation for large model with batching."""
        memory_req = calculator.calculate_memory_requirements(large_model)
        
        # Large model should require more memory
        assert memory_req["model_memory_gb"] > 100  # 70B params * 2 bytes
        
        # KV cache should scale with batch size
        single_batch_model = ModelSpecs(
            name="large-model",
            parameters=70_000_000_000,
            batch_size=1
        )
        single_batch_req = calculator.calculate_memory_requirements(single_batch_model)
        
        # Batch size 4 should have ~4x KV cache
        ratio = memory_req["kv_cache_gb"] / single_batch_req["kv_cache_gb"]
        assert abs(ratio - 4.0) < 0.1
    
    def test_calculate_gpu_requirements(self, calculator, small_model):
        """Test GPU requirement calculation."""
        target_rps = 10.0
        gpu_req = calculator.calculate_gpu_requirements(small_model, target_rps)
        
        # Check all fields are present
        assert "memory_requirements" in gpu_req
        assert "tokens_per_second_per_gpu" in gpu_req
        assert "required_gpus_throughput" in gpu_req
        assert "required_gpus_memory" in gpu_req
        assert "recommended_gpus" in gpu_req
        assert "recommended_gpu_memory" in gpu_req
        
        # Recommended GPUs should be max of throughput and memory requirements
        assert gpu_req["recommended_gpus"] == max(
            gpu_req["required_gpus_throughput"],
            gpu_req["required_gpus_memory"]
        )
    
    def test_calculate_gpu_requirements_standard_models(self, calculator):
        """Test GPU requirements for standard models."""
        llama_8b = calculator.standard_models["llama-3.1-8b"]
        gpu_req = calculator.calculate_gpu_requirements(llama_8b, 5.0)
        
        # Should recommend appropriate GPU memory
        assert gpu_req["recommended_gpu_memory"] in [40, 80]
        
        # Should have reasonable token throughput
        assert gpu_req["tokens_per_second_per_gpu"] > 0
        assert gpu_req["tokens_per_second_per_gpu"] == 150  # From base_tokens_per_second
    
    def test_calculate_deployment_cost(self, calculator, small_model):
        """Test deployment cost calculation."""
        gpu_specs = GPUSpecs(
            gpu_type=GPUType.A100_40GB,
            memory_gb=40,
            compute_capability=8.0,
            hourly_cost_usd=1.89,
            provider=CloudProvider.COREWEAVE
        )
        
        config = DeploymentConfig(
            model=small_model,
            gpu=gpu_specs,
            replicas=2,
            utilization_target=0.7
        )
        
        # Mock the additional cost methods
        calculator.calculate_storage_cost = lambda model: 50.0
        calculator.estimate_network_cost = lambda config: 100.0
        
        costs = calculator.calculate_deployment_cost(config)
        
        # Check cost components
        assert "monthly_gpu_cost" in costs
        assert "storage_cost" in costs
        assert "network_cost" in costs
        assert "management_overhead" in costs
        assert "total_monthly_cost" in costs
        
        # Verify GPU cost calculation
        expected_gpu_cost = 1.89 * 2 * 730  # hourly * replicas * hours
        assert costs["monthly_gpu_cost"] == expected_gpu_cost
        
        # Verify overhead is 5%
        assert costs["management_overhead"] == expected_gpu_cost * 0.05
    
    @pytest.mark.parametrize("provider,gpu_type,expected_cost", [
        (CloudProvider.COREWEAVE, GPUType.A100_40GB, 1.89),
        (CloudProvider.COREWEAVE, GPUType.H100_80GB, 4.29),
        (CloudProvider.AWS, GPUType.V100_32GB, 2.04),
        (CloudProvider.GCP, GPUType.A100_40GB, 2.93),
    ])
    def test_gpu_pricing(self, calculator, provider, gpu_type, expected_cost):
        """Test GPU pricing data is correct."""
        assert calculator.gpu_pricing[provider][gpu_type] == expected_cost
    
    def test_memory_scaling_with_context(self, calculator):
        """Test memory requirements scale with context length."""
        model_short = ModelSpecs(
            name="test",
            parameters=1_000_000_000,
            context_length=1024
        )
        model_long = ModelSpecs(
            name="test",
            parameters=1_000_000_000,
            context_length=8192
        )
        
        mem_short = calculator.calculate_memory_requirements(model_short)
        mem_long = calculator.calculate_memory_requirements(model_long)
        
        # Longer context should require more KV cache
        assert mem_long["kv_cache_gb"] > mem_short["kv_cache_gb"]
        
        # Ratio should be approximately 8x (8192/1024)
        ratio = mem_long["kv_cache_gb"] / mem_short["kv_cache_gb"]
        assert abs(ratio - 8.0) < 0.1
    
    def test_gpu_requirements_high_throughput(self, calculator, small_model):
        """Test GPU requirements for high throughput scenarios."""
        low_rps = 1.0
        high_rps = 100.0
        
        low_req = calculator.calculate_gpu_requirements(small_model, low_rps)
        high_req = calculator.calculate_gpu_requirements(small_model, high_rps)
        
        # Higher RPS should require more GPUs
        assert high_req["required_gpus_throughput"] > low_req["required_gpus_throughput"]
        
        # Memory requirements should be the same
        assert high_req["required_gpus_memory"] == low_req["required_gpus_memory"]
    
    def test_storage_cost_calculation(self, calculator, small_model):
        """Test storage cost calculation based on model size."""
        storage_cost = calculator.calculate_storage_cost(small_model)
        
        # Calculate expected cost
        model_size_gb = (small_model.parameters * small_model.memory_per_param_bytes) / (1024**3)
        expected_cost = model_size_gb * 3 * 0.023  # 3 replicas * $0.023/GB/month
        
        assert abs(storage_cost - expected_cost) < 0.01
        assert storage_cost > 0
    
    def test_network_cost_estimation(self, calculator):
        """Test network cost estimation for deployment."""
        config = DeploymentConfig(
            model=ModelSpecs("test", 1_000_000_000),
            gpu=GPUSpecs(GPUType.A100_40GB, 40, 8.0, 1.89, CloudProvider.COREWEAVE),
            replicas=3
        )
        
        network_cost = calculator.estimate_network_cost(config)
        
        # Expected: 3 replicas * 100GB * $0.05/GB
        expected_cost = 3 * 100 * 0.05
        assert network_cost == expected_cost
    
    def test_monthly_requests_estimation(self, calculator):
        """Test monthly request volume estimation."""
        config = DeploymentConfig(
            model=ModelSpecs("test", 1_000_000_000),
            gpu=GPUSpecs(GPUType.A100_40GB, 40, 8.0, 1.89, CloudProvider.COREWEAVE),
            replicas=2,
            utilization_target=0.8
        )
        
        requests = calculator.estimate_monthly_requests(config)
        
        # Expected: 100 req/hr/replica * 0.8 utilization * 2 replicas * 730 hours
        expected_requests = int(100 * 0.8 * 2 * 730)
        assert requests == expected_requests
    
    def test_compare_providers_basic(self, calculator):
        """Test basic provider comparison functionality."""
        comparison = calculator.compare_providers("llama-3.1-8b", target_rps=5)
        
        # Should return results
        assert len(comparison) > 0
        
        # Check structure of first result
        first_result = list(comparison.values())[0]
        assert "provider" in first_result
        assert "gpu_type" in first_result
        assert "monthly_cost" in first_result
        assert "cost_per_request" in first_result
        assert "gpu_count" in first_result
        
        # Results should be sorted by cost (ascending)
        costs = [result["monthly_cost"] for result in comparison.values()]
        assert costs == sorted(costs)
    
    def test_compare_providers_large_model(self, calculator):
        """Test provider comparison for large models requiring 80GB GPUs."""
        comparison = calculator.compare_providers("llama-3.1-70b", target_rps=2)
        
        # Should have results
        assert len(comparison) > 0
        
        # Should recommend 80GB GPUs for large models
        gpu_types = [result["gpu_type"] for result in comparison.values()]
        assert any("80gb" in gpu_type for gpu_type in gpu_types)


class TestCostOptimizationScenarios:
    """Test cost optimization scenarios and recommendations."""
    
    @pytest.fixture
    def calculator(self):
        if LLMCostCalculator is None:
            pytest.skip("LLMCostCalculator module not available")
        return LLMCostCalculator()
    
    def test_cost_efficiency_analysis(self, calculator):
        """Test cost efficiency across different model sizes."""
        models = ["mistral-7b", "llama-3.1-8b", "codellama-13b"]
        target_rps = 5
        
        costs = {}
        for model_name in models:
            comparison = calculator.compare_providers(model_name, target_rps)
            best_option = list(comparison.values())[0]
            costs[model_name] = best_option["cost_per_request"]
        
        # Larger models should generally cost more per request
        assert costs["codellama-13b"] >= costs["llama-3.1-8b"]
        assert costs["llama-3.1-8b"] >= costs["mistral-7b"]
    
    def test_throughput_scaling_costs(self, calculator):
        """Test how costs scale with throughput requirements."""
        model_name = "llama-3.1-8b"
        throughputs = [1, 5, 10, 20]
        
        costs_by_throughput = {}
        for rps in throughputs:
            comparison = calculator.compare_providers(model_name, rps)
            best_option = list(comparison.values())[0]
            costs_by_throughput[rps] = best_option["monthly_cost"]
        
        # Higher throughput should generally require higher costs
        assert costs_by_throughput[20] > costs_by_throughput[1]
        assert costs_by_throughput[10] > costs_by_throughput[5]
    
    def test_gpu_memory_optimization(self, calculator):
        """Test GPU memory optimization recommendations."""
        # Small model should work with 40GB
        small_model = calculator.standard_models["mistral-7b"]
        small_req = calculator.calculate_gpu_requirements(small_model, 5)
        
        # Large model should require 80GB
        large_model = calculator.standard_models["llama-3.1-70b"]
        large_req = calculator.calculate_gpu_requirements(large_model, 5)
        
        # Memory recommendations should be appropriate
        assert small_req["recommended_gpu_memory"] <= 40
        assert large_req["recommended_gpu_memory"] == 80
    
    def test_batch_size_impact(self, calculator):
        """Test impact of batch size on memory requirements."""
        base_model = ModelSpecs("test", 7_000_000_000, batch_size=1)
        batched_model = ModelSpecs("test", 7_000_000_000, batch_size=8)
        
        base_mem = calculator.calculate_memory_requirements(base_model)
        batched_mem = calculator.calculate_memory_requirements(batched_model)
        
        # Batched model should require more KV cache memory
        assert batched_mem["kv_cache_gb"] > base_mem["kv_cache_gb"]
        
        # Should scale approximately with batch size
        ratio = batched_mem["kv_cache_gb"] / base_mem["kv_cache_gb"]
        assert 7 < ratio < 9  # Approximately 8x
    
    def test_cost_breakdown_accuracy(self, calculator):
        """Test accuracy of cost breakdown percentages."""
        model = calculator.standard_models["llama-3.1-8b"]
        gpu_specs = GPUSpecs(
            gpu_type=GPUType.A100_40GB,
            memory_gb=40,
            compute_capability=8.0,
            hourly_cost_usd=2.00,
            provider=CloudProvider.COREWEAVE
        )
        
        config = DeploymentConfig(
            model=model,
            gpu=gpu_specs,
            replicas=2
        )
        
        costs = calculator.calculate_deployment_cost(config)
        breakdown = costs["cost_breakdown_pct"]
        
        # Percentages should sum to 100%
        total_pct = sum(breakdown.values())
        assert abs(total_pct - 100.0) < 0.01
        
        # GPU compute should be the largest component
        assert breakdown["gpu_compute"] > 50  # Should be majority of cost


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def calculator(self):
        if LLMCostCalculator is None:
            pytest.skip("LLMCostCalculator module not available")
        return LLMCostCalculator()
    
    def test_invalid_model_name(self, calculator):
        """Test handling of invalid model names."""
        with pytest.raises(KeyError):
            calculator.compare_providers("nonexistent-model", target_rps=5)
    
    def test_zero_throughput_handling(self, calculator):
        """Test handling of zero throughput requirements."""
        model = calculator.standard_models["llama-3.1-8b"]
        gpu_req = calculator.calculate_gpu_requirements(model, target_throughput_rps=0)
        
        # Should handle gracefully
        assert gpu_req["required_gpus_throughput"] >= 0
        assert gpu_req["recommended_gpus"] >= 1  # At least 1 GPU for memory
    
    def test_extreme_throughput_handling(self, calculator):
        """Test handling of extreme throughput requirements."""
        model = calculator.standard_models["llama-3.1-8b"]
        gpu_req = calculator.calculate_gpu_requirements(model, target_throughput_rps=1000)
        
        # Should handle gracefully
        assert gpu_req["recommended_gpus"] > 10  # Should require many GPUs
        assert gpu_req["recommended_gpus"] < 1000  # But not unreasonably many
    
    def test_unsupported_gpu_provider_combo(self, calculator):
        """Test handling of unsupported GPU/provider combinations."""
        # Create config with T4 on CoreWeave (not in pricing table)
        invalid_gpu = GPUSpecs(
            gpu_type=GPUType.T4_16GB,
            memory_gb=16,
            compute_capability=7.5,
            hourly_cost_usd=0.50,
            provider=CloudProvider.COREWEAVE
        )
        
        invalid_config = DeploymentConfig(
            model=calculator.standard_models["llama-3.1-8b"],
            gpu=invalid_gpu,
            replicas=1
        )
        
        # Should raise KeyError for unsupported combination
        with pytest.raises(KeyError):
            calculator.calculate_deployment_cost(invalid_config)
    
    def test_very_large_model_memory(self, calculator):
        """Test handling of extremely large models."""
        huge_model = ModelSpecs(
            name="huge-model",
            parameters=1_000_000_000_000,  # 1T parameters
            memory_per_param_bytes=2.0
        )
        
        memory_req = calculator.calculate_memory_requirements(huge_model)
        gpu_req = calculator.calculate_gpu_requirements(huge_model, 1)
        
        # Should require many GPUs
        assert memory_req["total_memory_gb"] > 1000
        assert gpu_req["required_gpus_memory"] > 10


class TestPerformanceAndScaling:
    """Test performance characteristics and scaling behavior."""
    
    @pytest.fixture
    def calculator(self):
        if LLMCostCalculator is None:
            pytest.skip("LLMCostCalculator module not available")
        return LLMCostCalculator()
    
    def test_calculation_precision(self, calculator):
        """Test calculation precision across different scales."""
        sizes = [1_000_000, 100_000_000, 10_000_000_000]
        
        for params in sizes:
            model = ModelSpecs(f"test-{params}", params, 2.0)
            memory_req = calculator.calculate_memory_requirements(model)
            
            # Memory should scale linearly with parameters
            expected_model_memory = (params * 2.0) / (1024**3)
            assert abs(memory_req["model_memory_gb"] - expected_model_memory) < 0.001
    
    def test_batch_comparison_consistency(self, calculator):
        """Test consistency of batch provider comparisons."""
        models = ["llama-3.1-8b", "mistral-7b"]
        throughputs = [1, 5, 10]
        
        # Run comparisons multiple times
        results_1 = {}
        results_2 = {}
        
        for model in models:
            for rps in throughputs:
                results_1[f"{model}_{rps}"] = calculator.compare_providers(model, rps)
                results_2[f"{model}_{rps}"] = calculator.compare_providers(model, rps)
        
        # Results should be identical
        for key in results_1:
            assert results_1[key] == results_2[key]
    
    def test_memory_calculation_consistency(self, calculator):
        """Test memory calculations are consistent."""
        model = calculator.standard_models["llama-3.1-8b"]
        
        # Calculate memory requirements multiple times
        mem_1 = calculator.calculate_memory_requirements(model)
        mem_2 = calculator.calculate_memory_requirements(model)
        
        # Should be identical
        assert mem_1 == mem_2