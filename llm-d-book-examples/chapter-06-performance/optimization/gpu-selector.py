#!/usr/bin/env python3
"""
GPU Selection Framework for LLM Inference
Helps select optimal GPU configuration based on model requirements

This tool evaluates NVIDIA and AMD GPUs across multiple dimensions:
- Performance (compute, memory bandwidth)
- Cost efficiency (TFLOPS per dollar)
- Memory efficiency (capacity per watt)
- Tensor parallelism requirements

Usage:
    python gpu-selector.py

Example Output:
    GPU Recommendation Results:
    {
      "recommended_gpu": "H100",
      "tensor_parallel_size": 2,
      "estimated_memory_usage": 70.0,
      "performance_analysis": {
        "estimated_tokens_per_second": 42.1,
        "estimated_latency_ms": 23.8,
        "cost_per_1k_tokens": 0.0297
      }
    }

Performance Baseline Estimates:
- Small models (7B-13B): ~150 tokens/sec baseline
- Medium models (30B-40B): ~80 tokens/sec baseline  
- Large models (70B-80B): ~40 tokens/sec baseline
- XLarge models (175B+): ~15 tokens/sec baseline

Source: Chapter 6 - Performance Optimization
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum

class ModelSize(Enum):
    SMALL = "7B-13B"      # 7B, 8B, 13B parameter models
    MEDIUM = "30B-40B"    # 30B, 34B parameter models  
    LARGE = "70B-80B"     # 70B, 80B parameter models
    XLARGE = "175B+"      # 175B, 405B+ parameter models

@dataclass
class GPUSpec:
    name: str
    memory_gb: int
    memory_bandwidth_tbs: float  # TB/s
    compute_tflops_fp8: float
    power_watts: int
    cost_per_hour: float  # USD per hour
    tensor_parallel_max: int
    
    def memory_efficiency_score(self) -> float:
        """Calculate memory efficiency (GB per watt)"""
        return self.memory_gb / self.power_watts
    
    def cost_efficiency_score(self) -> float:
        """Calculate cost efficiency (TFLOPS per dollar per hour)"""
        return self.compute_tflops_fp8 / self.cost_per_hour
    
    def performance_score(self, model_size: ModelSize) -> float:
        """Calculate performance score for specific model size"""
        base_score = self.compute_tflops_fp8 * self.memory_bandwidth_tbs
        
        # Adjust for model size requirements
        size_multiplier = {
            ModelSize.SMALL: 1.0,
            ModelSize.MEDIUM: 0.9,   # Slight penalty for memory pressure
            ModelSize.LARGE: 0.8,    # Memory becomes limiting factor
            ModelSize.XLARGE: 0.7    # Multi-GPU coordination overhead
        }
        
        return base_score * size_multiplier[model_size]

# Current generation GPU specifications
GPU_CATALOG = {
    # NVIDIA Hopper Generation
    "H100": GPUSpec(
        name="NVIDIA H100",
        memory_gb=80,
        memory_bandwidth_tbs=3.35,
        compute_tflops_fp8=1979,
        power_watts=700,
        cost_per_hour=4.50,
        tensor_parallel_max=8
    ),
    
    "H200": GPUSpec(
        name="NVIDIA H200", 
        memory_gb=141,
        memory_bandwidth_tbs=4.8,
        compute_tflops_fp8=1979,  # Same compute as H100
        power_watts=700,
        cost_per_hour=6.20,
        tensor_parallel_max=8
    ),
    
    # NVIDIA Blackwell Generation (Projected)
    "B200": GPUSpec(
        name="NVIDIA B200",
        memory_gb=192,
        memory_bandwidth_tbs=8.0,
        compute_tflops_fp8=4500,  # Estimated 2.3x H100
        power_watts=1000,
        cost_per_hour=9.50,  # Projected pricing
        tensor_parallel_max=16
    ),
    
    # AMD Instinct Generation
    "MI300X": GPUSpec(
        name="AMD MI300X",
        memory_gb=192,
        memory_bandwidth_tbs=5.2,
        compute_tflops_fp8=1307,  # FP8 performance
        power_watts=750,
        cost_per_hour=4.80,
        tensor_parallel_max=8
    ),
    
    "MI350X": GPUSpec(  # Next-gen AMD
        name="AMD MI350X",
        memory_gb=288,
        memory_bandwidth_tbs=8.0,
        compute_tflops_fp8=2100,  # Projected improvement
        power_watts=800,
        cost_per_hour=7.20,
        tensor_parallel_max=8
    )
}

class GPUSelector:
    def __init__(self):
        self.gpu_catalog = GPU_CATALOG
    
    def recommend_gpu(self, 
                     model_size: ModelSize,
                     priority: str = "balanced",  # "latency", "throughput", "cost"
                     max_cost_per_hour: float = None,
                     required_memory_gb: int = None) -> Dict:
        """
        Recommend optimal GPU configuration
        
        Args:
            model_size: Target model size category
            priority: Optimization priority
            max_cost_per_hour: Maximum acceptable cost
            required_memory_gb: Minimum memory requirement
            
        Returns:
            Dictionary with recommendations and analysis
        """
        
        # Filter GPUs by constraints
        candidates = []
        for gpu_name, gpu_spec in self.gpu_catalog.items():
            # Apply cost filter
            if max_cost_per_hour and gpu_spec.cost_per_hour > max_cost_per_hour:
                continue
                
            # Apply memory filter  
            if required_memory_gb and gpu_spec.memory_gb < required_memory_gb:
                continue
                
            candidates.append((gpu_name, gpu_spec))
        
        if not candidates:
            return {"error": "No GPUs meet the specified constraints"}
        
        # Score candidates based on priority
        scored_candidates = []
        for gpu_name, gpu_spec in candidates:
            if priority == "latency":
                score = gpu_spec.performance_score(model_size) * 0.7 + \
                       gpu_spec.memory_bandwidth_tbs * 0.3
            elif priority == "throughput":
                score = gpu_spec.compute_tflops_fp8 * 0.6 + \
                       gpu_spec.memory_gb * 0.4
            elif priority == "cost":
                score = gpu_spec.cost_efficiency_score() * 0.8 + \
                       gpu_spec.memory_efficiency_score() * 0.2
            else:  # balanced
                score = (gpu_spec.performance_score(model_size) * 0.4 +
                        gpu_spec.cost_efficiency_score() * 0.3 +
                        gpu_spec.memory_efficiency_score() * 0.3)
            
            scored_candidates.append((score, gpu_name, gpu_spec))
        
        # Sort by score (highest first)
        scored_candidates.sort(reverse=True)
        
        # Calculate tensor parallelism recommendation
        top_gpu = scored_candidates[0][2]
        tp_recommendation = self._calculate_tensor_parallelism(model_size, top_gpu)
        
        return {
            "recommended_gpu": scored_candidates[0][1],
            "gpu_specs": scored_candidates[0][2],
            "tensor_parallel_size": tp_recommendation["tp_size"],
            "estimated_memory_usage": tp_recommendation["memory_usage_gb"],
            "alternatives": [
                {
                    "gpu": name,
                    "score": score,
                    "cost_per_hour": spec.cost_per_hour,
                    "memory_gb": spec.memory_gb
                }
                for score, name, spec in scored_candidates[1:4]  # Top 3 alternatives
            ],
            "performance_analysis": self._analyze_performance(
                scored_candidates[0][2], model_size, tp_recommendation
            )
        }
    
    def _calculate_tensor_parallelism(self, model_size: ModelSize, gpu_spec: GPUSpec) -> Dict:
        """Calculate optimal tensor parallelism configuration"""
        
        # Estimated memory requirements (includes KV cache, attention, etc.)
        memory_requirements = {
            ModelSize.SMALL: {"7B": 14, "8B": 16, "13B": 26},
            ModelSize.MEDIUM: {"30B": 60, "34B": 68},
            ModelSize.LARGE: {"70B": 140, "80B": 160},
            ModelSize.XLARGE: {"175B": 350, "405B": 810}
        }
        
        # Use largest model in category for conservative estimate
        if model_size == ModelSize.SMALL:
            required_memory = memory_requirements[model_size]["13B"]
        elif model_size == ModelSize.MEDIUM:
            required_memory = memory_requirements[model_size]["34B"]
        elif model_size == ModelSize.LARGE:
            required_memory = memory_requirements[model_size]["80B"]
        else:  # XLARGE
            required_memory = memory_requirements[model_size]["405B"]
        
        # Calculate minimum TP size
        tp_size = max(1, (required_memory + gpu_spec.memory_gb - 1) // gpu_spec.memory_gb)
        tp_size = min(tp_size, gpu_spec.tensor_parallel_max)
        
        # Ensure TP size is power of 2 for optimal performance
        tp_size = 2 ** (tp_size - 1).bit_length() if tp_size > 1 else 1
        
        memory_per_gpu = required_memory / tp_size
        
        return {
            "tp_size": tp_size,
            "memory_usage_gb": memory_per_gpu,
            "memory_utilization": memory_per_gpu / gpu_spec.memory_gb
        }
    
    def _analyze_performance(self, gpu_spec: GPUSpec, model_size: ModelSize, tp_config: Dict) -> Dict:
        """Analyze expected performance characteristics"""
        
        # Baseline performance estimates (tokens/second)
        baseline_performance = {
            ModelSize.SMALL: 150,
            ModelSize.MEDIUM: 80, 
            ModelSize.LARGE: 40,
            ModelSize.XLARGE: 15
        }
        
        base_tps = baseline_performance[model_size]
        
        # Adjust for GPU performance
        gpu_multiplier = gpu_spec.compute_tflops_fp8 / 1000  # Normalize to H100 baseline
        
        # Adjust for tensor parallelism overhead
        tp_efficiency = 0.95 ** (tp_config["tp_size"] - 1)
        
        estimated_tps = base_tps * gpu_multiplier * tp_efficiency
        
        return {
            "estimated_tokens_per_second": round(estimated_tps, 1),
            "estimated_latency_ms": round(1000 / estimated_tps, 1),
            "tensor_parallel_efficiency": round(tp_efficiency, 3),
            "memory_utilization": round(tp_config["memory_utilization"], 2),
            "cost_per_1k_tokens": round(gpu_spec.cost_per_hour / (estimated_tps * 3.6), 4)
        }

# Example usage and testing
if __name__ == "__main__":
    selector = GPUSelector()
    
    # Example: Find best GPU for 70B model with cost constraint
    recommendation = selector.recommend_gpu(
        model_size=ModelSize.LARGE,
        priority="balanced",
        max_cost_per_hour=7.0,
        required_memory_gb=100
    )
    
    print("GPU Recommendation Results:")
    print(json.dumps(recommendation, indent=2, default=str))
    
    # Example: Compare all GPUs for different model sizes
    print("\n" + "="*60)
    print("Performance Comparison Across Model Sizes")
    print("="*60)
    
    for model_size in ModelSize:
        print(f"\n{model_size.value} Models:")
        rec = selector.recommend_gpu(model_size=model_size, priority="balanced")
        if "error" not in rec:
            print(f"  Recommended: {rec['recommended_gpu']}")
            print(f"  TP Size: {rec['tensor_parallel_size']}")
            print(f"  Performance: {rec['performance_analysis']['estimated_tokens_per_second']} tokens/sec")
            print(f"  Cost: ${rec['performance_analysis']['cost_per_1k_tokens']}/1K tokens")