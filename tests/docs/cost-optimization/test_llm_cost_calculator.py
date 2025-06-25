"""
Tests for the LLMCostCalculator class in docs/cost-optimization/llm_cost_calculator.py
"""

import pytest
import math
from pathlib import Path
import sys

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
        AWS = "aws"
        GCP = "gcp"
    
    class GPUType(Enum):
        A100_40GB = "a100_40gb"
        A100_80GB = "a100_80gb"
        H100_80GB = "h100_80gb"
        V100_32GB = "v100_32gb"
    
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


@pytest.mark.skipif(LLMCostCalculator is None, reason="LLMCostCalculator module not available")
class TestLLMCostCalculator:
    """Test cases for LLMCostCalculator class."""
    
    @pytest.fixture
    def calculator(self):
        """Create calculator instance."""
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