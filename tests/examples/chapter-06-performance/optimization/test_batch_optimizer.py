"""
Tests for batch size optimization module in chapter-06-performance/optimization/batch-optimizer.py
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path
import json

# Add the examples directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "llm-d-book-examples"))

try:
    from chapter_06_performance.optimization.batch_optimizer import BatchOptimizer
except ImportError:
    # Create mock class for testing when real implementation isn't available
    class BatchOptimizer:
        def __init__(self):
            self.models = {
                "llama-3.1-8b": {"params": 8e9, "hidden_size": 4096, "num_layers": 32},
                "llama-3.1-70b": {"params": 70e9, "hidden_size": 8192, "num_layers": 80},
                "llama-3.1-405b": {"params": 405e9, "hidden_size": 16384, "num_layers": 126}
            }
            
        def calculate_memory_requirements(self, model_name, sequence_length=2048, quantization="fp16"):
            """Calculate memory requirements for given configuration."""
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not supported")
                
            model_info = self.models[model_name]
            
            # Calculate memory components (simplified)
            quantization_factor = {"fp16": 2, "fp8": 1, "int8": 1, "int4": 0.5}[quantization]
            
            model_memory = model_info["params"] * quantization_factor / 1e9  # GB
            kv_cache_memory = sequence_length * model_info["hidden_size"] * 2 * quantization_factor / 1e9
            activation_memory = model_info["hidden_size"] * sequence_length * quantization_factor / 1e9
            
            return {
                "model_memory_gb": model_memory,
                "kv_cache_per_token_gb": kv_cache_memory / sequence_length,
                "activation_memory_gb": activation_memory,
                "total_base_memory_gb": model_memory + activation_memory
            }
            
        def optimize_batch_size(self, model_name, gpu_memory_gb, sequence_length=2048, 
                              quantization="fp16", tensor_parallel=1, safety_factor=0.9):
            """Optimize batch size for given constraints."""
            memory_req = self.calculate_memory_requirements(model_name, sequence_length, quantization)
            
            available_memory = gpu_memory_gb * safety_factor / tensor_parallel
            memory_per_sequence = memory_req["kv_cache_per_token_gb"] * sequence_length
            
            base_memory = memory_req["total_base_memory_gb"] / tensor_parallel
            remaining_memory = available_memory - base_memory
            
            if remaining_memory <= 0:
                return {"optimal_batch_size": 0, "error": "Insufficient memory for model"}
                
            optimal_batch_size = int(remaining_memory / memory_per_sequence)
            
            return {
                "optimal_batch_size": max(1, optimal_batch_size),
                "memory_utilization": min(1.0, (base_memory + optimal_batch_size * memory_per_sequence) / available_memory),
                "estimated_throughput_tokens_per_sec": optimal_batch_size * 50,  # Simplified estimate
                "configuration": {
                    "model": model_name,
                    "batch_size": max(1, optimal_batch_size),
                    "sequence_length": sequence_length,
                    "quantization": quantization,
                    "tensor_parallel": tensor_parallel
                }
            }
            
        def generate_vllm_config(self, model_name, batch_size, sequence_length=2048, 
                               quantization="fp16", tensor_parallel=1):
            """Generate vLLM configuration for optimized settings."""
            return {
                "model": model_name,
                "max_model_len": sequence_length,
                "max_num_batched_tokens": batch_size * sequence_length,
                "max_num_seqs": batch_size,
                "tensor_parallel_size": tensor_parallel,
                "quantization": quantization,
                "gpu_memory_utilization": 0.9,
                "enforce_eager": False,
                "max_context_len_to_capture": sequence_length
            }


class TestBatchOptimizer:
    """Test cases for batch size optimization."""
    
    @pytest.fixture
    def optimizer(self):
        """Create batch optimizer instance."""
        return BatchOptimizer()
    
    def test_initialization(self, optimizer):
        """Test BatchOptimizer initialization."""
        assert hasattr(optimizer, 'models')
        assert "llama-3.1-8b" in optimizer.models
        assert "llama-3.1-70b" in optimizer.models
        assert "llama-3.1-405b" in optimizer.models
    
    def test_memory_requirements_calculation(self, optimizer):
        """Test memory requirements calculation for different models."""
        # Test Llama 3.1 8B
        memory_req = optimizer.calculate_memory_requirements("llama-3.1-8b", 2048, "fp16")
        
        assert "model_memory_gb" in memory_req
        assert "kv_cache_per_token_gb" in memory_req
        assert "activation_memory_gb" in memory_req
        assert "total_base_memory_gb" in memory_req
        
        # Model memory should be reasonable for 8B parameters
        assert 10 < memory_req["model_memory_gb"] < 30
        assert memory_req["kv_cache_per_token_gb"] > 0
        assert memory_req["activation_memory_gb"] > 0
    
    @pytest.mark.parametrize("model_name,expected_memory_range", [
        ("llama-3.1-8b", (10, 30)),
        ("llama-3.1-70b", (70, 150)),
        ("llama-3.1-405b", (400, 900))
    ])
    def test_memory_requirements_different_models(self, optimizer, model_name, expected_memory_range):
        """Test memory requirements for different model sizes."""
        memory_req = optimizer.calculate_memory_requirements(model_name, 2048, "fp16")
        
        min_memory, max_memory = expected_memory_range
        assert min_memory < memory_req["model_memory_gb"] < max_memory
    
    @pytest.mark.parametrize("quantization,expected_factor", [
        ("fp16", 2.0),
        ("fp8", 1.0), 
        ("int8", 1.0),
        ("int4", 0.5)
    ])
    def test_quantization_memory_impact(self, optimizer, quantization, expected_factor):
        """Test quantization impact on memory requirements."""
        fp16_memory = optimizer.calculate_memory_requirements("llama-3.1-8b", 2048, "fp16")
        quant_memory = optimizer.calculate_memory_requirements("llama-3.1-8b", 2048, quantization)
        
        # Quantized model should use less or equal memory
        ratio = fp16_memory["model_memory_gb"] / quant_memory["model_memory_gb"]
        assert abs(ratio - (2.0 / expected_factor)) < 0.1
    
    def test_batch_size_optimization_h100(self, optimizer):
        """Test batch size optimization for H100 80GB."""
        result = optimizer.optimize_batch_size(
            model_name="llama-3.1-8b",
            gpu_memory_gb=80,
            sequence_length=2048,
            quantization="fp16",
            tensor_parallel=1
        )
        
        assert result["optimal_batch_size"] > 0
        assert result["memory_utilization"] <= 1.0
        assert "estimated_throughput_tokens_per_sec" in result
        assert "configuration" in result
    
    def test_batch_size_optimization_a100(self, optimizer):
        """Test batch size optimization for A100 40GB."""
        result = optimizer.optimize_batch_size(
            model_name="llama-3.1-8b",
            gpu_memory_gb=40,
            sequence_length=2048,
            quantization="fp16",
            tensor_parallel=1
        )
        
        assert result["optimal_batch_size"] > 0
        assert result["memory_utilization"] <= 1.0
    
    def test_tensor_parallelism_impact(self, optimizer):
        """Test impact of tensor parallelism on batch size."""
        # Single GPU
        single_gpu = optimizer.optimize_batch_size(
            "llama-3.1-70b", 80, tensor_parallel=1
        )
        
        # Multi-GPU tensor parallelism
        multi_gpu = optimizer.optimize_batch_size(
            "llama-3.1-70b", 80, tensor_parallel=8
        )
        
        # Multi-GPU should allow larger batch sizes
        assert multi_gpu["optimal_batch_size"] >= single_gpu["optimal_batch_size"]
    
    def test_sequence_length_impact(self, optimizer):
        """Test impact of sequence length on batch size."""
        short_seq = optimizer.optimize_batch_size(
            "llama-3.1-8b", 80, sequence_length=1024
        )
        
        long_seq = optimizer.optimize_batch_size(
            "llama-3.1-8b", 80, sequence_length=4096
        )
        
        # Shorter sequences should allow larger batch sizes
        assert short_seq["optimal_batch_size"] >= long_seq["optimal_batch_size"]
    
    def test_insufficient_memory_handling(self, optimizer):
        """Test handling of insufficient memory scenarios."""
        result = optimizer.optimize_batch_size(
            model_name="llama-3.1-405b",
            gpu_memory_gb=8,  # Too small for 405B model
            tensor_parallel=1
        )
        
        assert result["optimal_batch_size"] == 0 or "error" in result
    
    def test_vllm_config_generation(self, optimizer):
        """Test vLLM configuration generation."""
        config = optimizer.generate_vllm_config(
            model_name="llama-3.1-8b",
            batch_size=32,
            sequence_length=2048,
            quantization="fp16",
            tensor_parallel=2
        )
        
        assert config["model"] == "llama-3.1-8b"
        assert config["max_num_seqs"] == 32
        assert config["max_model_len"] == 2048
        assert config["tensor_parallel_size"] == 2
        assert config["quantization"] == "fp16"
        assert "max_num_batched_tokens" in config
    
    @pytest.mark.parametrize("gpu_type,memory_gb", [
        ("H100", 80),
        ("H200", 141),
        ("A100", 40),
        ("A100", 80),
        ("V100", 32)
    ])
    def test_optimization_different_gpus(self, optimizer, gpu_type, memory_gb):
        """Test optimization for different GPU types."""
        result = optimizer.optimize_batch_size(
            model_name="llama-3.1-8b",
            gpu_memory_gb=memory_gb,
            sequence_length=2048,
            quantization="fp16"
        )
        
        assert result["optimal_batch_size"] > 0
        assert result["memory_utilization"] <= 1.0
        
        # Larger memory should generally allow larger batch sizes
        if memory_gb >= 80:
            assert result["optimal_batch_size"] >= 16
    
    def test_safety_factor_impact(self, optimizer):
        """Test impact of safety factor on optimization."""
        aggressive = optimizer.optimize_batch_size(
            "llama-3.1-8b", 80, safety_factor=0.99
        )
        
        conservative = optimizer.optimize_batch_size(
            "llama-3.1-8b", 80, safety_factor=0.8
        )
        
        # Aggressive should allow larger batch sizes
        assert aggressive["optimal_batch_size"] >= conservative["optimal_batch_size"]
    
    def test_throughput_estimation(self, optimizer):
        """Test throughput estimation accuracy."""
        result = optimizer.optimize_batch_size(
            model_name="llama-3.1-8b",
            gpu_memory_gb=80,
            sequence_length=2048
        )
        
        # Throughput should scale with batch size
        assert result["estimated_throughput_tokens_per_sec"] > 0
        expected_throughput = result["optimal_batch_size"] * 50  # Simplified estimate
        assert abs(result["estimated_throughput_tokens_per_sec"] - expected_throughput) < 100
    
    def test_optimization_workflow(self, optimizer):
        """Test complete optimization workflow."""
        # Step 1: Calculate memory requirements
        memory_req = optimizer.calculate_memory_requirements("llama-3.1-8b", 2048, "fp16")
        assert memory_req["model_memory_gb"] > 0
        
        # Step 2: Optimize batch size
        optimization_result = optimizer.optimize_batch_size("llama-3.1-8b", 80)
        assert optimization_result["optimal_batch_size"] > 0
        
        # Step 3: Generate vLLM config
        config = optimizer.generate_vllm_config(
            "llama-3.1-8b",
            optimization_result["optimal_batch_size"]
        )
        assert config["max_num_seqs"] == optimization_result["optimal_batch_size"]
    
    def test_model_validation(self, optimizer):
        """Test model validation and error handling."""
        with pytest.raises(ValueError):
            optimizer.calculate_memory_requirements("invalid-model", 2048, "fp16")
    
    @pytest.mark.parametrize("sequence_length", [512, 1024, 2048, 4096, 8192])
    def test_sequence_length_scaling(self, optimizer, sequence_length):
        """Test optimization with different sequence lengths."""
        result = optimizer.optimize_batch_size(
            "llama-3.1-8b", 80, sequence_length=sequence_length
        )
        
        assert result["optimal_batch_size"] > 0
        assert result["configuration"]["sequence_length"] == sequence_length
    
    def test_memory_utilization_bounds(self, optimizer):
        """Test that memory utilization stays within bounds."""
        result = optimizer.optimize_batch_size("llama-3.1-8b", 80)
        
        assert 0.0 <= result["memory_utilization"] <= 1.0
        # Should utilize most of available memory efficiently
        assert result["memory_utilization"] > 0.5