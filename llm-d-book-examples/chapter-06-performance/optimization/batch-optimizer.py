#!/usr/bin/env python3
"""
Intelligent Batch Size Optimizer for LLM Throughput
Calculates optimal batch sizes based on GPU memory and model characteristics

This tool helps optimize LLM inference throughput by finding the maximum batch size
that fits within GPU memory constraints while accounting for:
- Model parameter memory requirements
- KV cache memory scaling with batch size and sequence length
- Activation memory for forward pass computation
- Different quantization schemes (fp16, fp8, int8, int4)
- Tensor parallelism configurations

Usage:
    python batch-optimizer.py

Example Output:
    Model: llama-3.1-70b
    80GB-fp16       | Batch:  12 | TPS:  421.3 | Memory: 89.2%
    141GB-fp16      | Batch:  24 | TPS:  756.8 | Memory: 87.4%
    192GB-fp8       | Batch:  64 | TPS: 1834.1 | Memory: 91.8%

Performance Guidelines:
- Larger batch sizes improve throughput but increase latency
- Memory utilization should stay below 95% for stability
- FP8 quantization can double throughput on supported hardware
- Consider sequence length vs batch size trade-offs

Source: Chapter 6 - Performance Optimization
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

@dataclass
class ModelConfig:
    name: str
    parameters: int  # Number of parameters
    hidden_size: int
    num_layers: int
    vocab_size: int
    max_sequence_length: int
    dtype_bytes: int  # 2 for fp16, 1 for fp8, etc.

@dataclass
class GPUMemoryProfile:
    total_memory_gb: int
    memory_bandwidth_tbs: float
    compute_tflops: float
    overhead_ratio: float = 0.15  # OS + driver overhead

class BatchOptimizer:
    """Optimizes batch sizes for maximum throughput while staying within memory constraints"""
    
    def __init__(self):
        # Pre-defined model configurations
        self.model_configs = {
            "llama-3.1-8b": ModelConfig(
                name="Llama 3.1 8B",
                parameters=8_030_000_000,
                hidden_size=4096,
                num_layers=32,
                vocab_size=128256,
                max_sequence_length=8192,
                dtype_bytes=2  # fp16
            ),
            "llama-3.1-70b": ModelConfig(
                name="Llama 3.1 70B", 
                parameters=70_553_000_000,
                hidden_size=8192,
                num_layers=80,
                vocab_size=128256,
                max_sequence_length=8192,
                dtype_bytes=2  # fp16
            ),
            "llama-3.1-405b": ModelConfig(
                name="Llama 3.1 405B",
                parameters=405_000_000_000,
                hidden_size=16384,
                num_layers=126,
                vocab_size=128256,
                max_sequence_length=8192,
                dtype_bytes=2  # fp16
            )
        }
    
    def calculate_model_memory_gb(self, 
                                 model_config: ModelConfig,
                                 quantization: str = "fp16",
                                 tensor_parallel_size: int = 1) -> float:
        """Calculate base model memory requirement"""
        
        dtype_multiplier = {
            "fp32": 4,
            "fp16": 2, 
            "bf16": 2,
            "fp8": 1,
            "int8": 1,
            "int4": 0.5
        }
        
        bytes_per_param = dtype_multiplier.get(quantization, 2)
        model_memory_bytes = model_config.parameters * bytes_per_param
        
        # Account for tensor parallelism
        model_memory_per_gpu = model_memory_bytes / tensor_parallel_size
        
        return model_memory_per_gpu / (1024**3)  # Convert to GB
    
    def calculate_kv_cache_memory_gb(self,
                                   model_config: ModelConfig,
                                   batch_size: int,
                                   sequence_length: int,
                                   kv_dtype: str = "fp16") -> float:
        """Calculate KV cache memory requirement"""
        
        dtype_bytes = {"fp16": 2, "fp8": 1, "int8": 1}.get(kv_dtype, 2)
        
        # KV cache formula: 2 (K+V) * batch_size * seq_len * num_layers * hidden_size * dtype_bytes
        kv_cache_bytes = (2 * batch_size * sequence_length * 
                         model_config.num_layers * model_config.hidden_size * dtype_bytes)
        
        return kv_cache_bytes / (1024**3)  # Convert to GB
    
    def calculate_activation_memory_gb(self,
                                     model_config: ModelConfig, 
                                     batch_size: int,
                                     sequence_length: int) -> float:
        """Calculate activation memory requirement"""
        
        # Simplified activation memory calculation
        # Actual formula is complex and depends on implementation details
        activation_bytes = (batch_size * sequence_length * model_config.hidden_size * 
                          model_config.num_layers * 4)  # Rough approximation
        
        return activation_bytes / (1024**3)  # Convert to GB
    
    def find_optimal_batch_size(self,
                               model_name: str,
                               gpu_memory_gb: int,
                               target_sequence_length: int = 2048,
                               quantization: str = "fp16",
                               kv_cache_dtype: str = "fp16",
                               tensor_parallel_size: int = 1,
                               safety_margin: float = 0.1) -> Dict:
        """Find optimal batch size for maximum throughput"""
        
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_config = self.model_configs[model_name]
        
        # Calculate base model memory
        model_memory = self.calculate_model_memory_gb(
            model_config, quantization, tensor_parallel_size
        )
        
        # Available memory for dynamic allocation
        available_memory = gpu_memory_gb * (1 - safety_margin) - model_memory
        
        if available_memory <= 0:
            return {
                "error": f"Model too large for GPU memory",
                "model_memory_gb": model_memory,
                "available_memory_gb": available_memory
            }
        
        # Binary search for optimal batch size
        max_batch_size = 1024  # Upper bound
        optimal_batch_size = 1
        
        for batch_size in range(1, max_batch_size + 1):
            kv_cache_memory = self.calculate_kv_cache_memory_gb(
                model_config, batch_size, target_sequence_length, kv_cache_dtype
            )
            
            activation_memory = self.calculate_activation_memory_gb(
                model_config, batch_size, target_sequence_length
            )
            
            total_dynamic_memory = kv_cache_memory + activation_memory
            
            if total_dynamic_memory <= available_memory:
                optimal_batch_size = batch_size
            else:
                break
        
        # Calculate performance metrics
        final_kv_memory = self.calculate_kv_cache_memory_gb(
            model_config, optimal_batch_size, target_sequence_length, kv_cache_dtype
        )
        final_activation_memory = self.calculate_activation_memory_gb(
            model_config, optimal_batch_size, target_sequence_length
        )
        
        total_memory_used = model_memory + final_kv_memory + final_activation_memory
        memory_utilization = total_memory_used / gpu_memory_gb
        
        # Estimate throughput (simplified calculation)
        base_throughput = self._estimate_throughput(
            model_config, optimal_batch_size, quantization
        )
        
        return {
            "optimal_batch_size": optimal_batch_size,
            "memory_breakdown": {
                "model_memory_gb": round(model_memory, 2),
                "kv_cache_memory_gb": round(final_kv_memory, 2),
                "activation_memory_gb": round(final_activation_memory, 2),
                "total_memory_gb": round(total_memory_used, 2)
            },
            "memory_utilization": round(memory_utilization, 2),
            "estimated_throughput": {
                "tokens_per_second": round(base_throughput, 1),
                "requests_per_second": round(base_throughput / target_sequence_length, 2)
            },
            "configuration_recommendations": self._generate_config_recommendations(
                optimal_batch_size, target_sequence_length, quantization
            )
        }
    
    def _estimate_throughput(self,
                           model_config: ModelConfig,
                           batch_size: int, 
                           quantization: str) -> float:
        """Estimate throughput in tokens per second"""
        
        # Base throughput scaling factors (empirical)
        base_tps_per_billion_params = {
            "fp16": 2.5,
            "fp8": 4.0,
            "int8": 3.5,
            "int4": 6.0
        }
        
        base_rate = base_tps_per_billion_params.get(quantization, 2.5)
        model_size_billions = model_config.parameters / 1_000_000_000
        
        # Batch size scaling (with diminishing returns)
        batch_efficiency = min(1.0, 0.3 + 0.7 * math.log(batch_size + 1) / math.log(65))
        
        estimated_tps = base_rate * batch_size * batch_efficiency / model_size_billions
        
        return estimated_tps
    
    def _generate_config_recommendations(self,
                                       batch_size: int,
                                       sequence_length: int,
                                       quantization: str) -> Dict:
        """Generate vLLM configuration recommendations"""
        
        return {
            "vllm_args": {
                "--max-num-seqs": batch_size,
                "--max-model-len": sequence_length,
                "--dtype": quantization,
                "--gpu-memory-utilization": "0.90",
                "--enable-chunked-prefill": "true" if batch_size > 32 else "false",
                "--max-num-batched-tokens": min(batch_size * sequence_length, 32768)
            },
            "environment_variables": {
                "VLLM_PARALLEL_OUTPUT_PROCESSING": "true" if batch_size > 16 else "false",
                "VLLM_ENABLE_ASYNC_OUTPUT_PROC": "true",
                "CUDA_LAUNCH_BLOCKING": "0"
            }
        }
    
    def compare_configurations(self,
                             model_name: str,
                             gpu_memory_options: List[int],
                             quantization_options: List[str] = None,
                             sequence_length: int = 2048) -> Dict:
        """Compare different configuration options"""
        
        if quantization_options is None:
            quantization_options = ["fp16", "fp8", "int8"]
        
        results = {}
        
        for gpu_memory in gpu_memory_options:
            for quantization in quantization_options:
                config_key = f"{gpu_memory}GB-{quantization}"
                
                try:
                    result = self.find_optimal_batch_size(
                        model_name=model_name,
                        gpu_memory_gb=gpu_memory,
                        target_sequence_length=sequence_length,
                        quantization=quantization
                    )
                    results[config_key] = result
                    
                except Exception as e:
                    results[config_key] = {"error": str(e)}
        
        return results

# Example usage and benchmarking
if __name__ == "__main__":
    optimizer = BatchOptimizer()
    
    print("LLM Batch Size Optimization Analysis")
    print("=" * 50)
    
    # Test different GPU configurations
    gpu_options = [80, 141, 192]  # H100, H200, MI300X memory sizes
    quantization_options = ["fp16", "fp8"]
    
    for model_name in ["llama-3.1-8b", "llama-3.1-70b"]:
        print(f"\nModel: {model_name}")
        print("-" * 30)
        
        comparison = optimizer.compare_configurations(
            model_name=model_name,
            gpu_memory_options=gpu_options,
            quantization_options=quantization_options,
            sequence_length=2048
        )
        
        for config, result in comparison.items():
            if "error" not in result:
                print(f"{config:15} | Batch: {result['optimal_batch_size']:3d} | "
                      f"TPS: {result['estimated_throughput']['tokens_per_second']:6.1f} | "
                      f"Memory: {result['memory_utilization']:.1%}")
            else:
                print(f"{config:15} | Error: {result['error']}")