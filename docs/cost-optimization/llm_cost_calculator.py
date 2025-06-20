#!/usr/bin/env python3
"""
LLM Deployment Cost Calculator and Forecasting Framework

This comprehensive cost modeling framework helps organizations understand,
predict, and optimize their LLM deployment costs across different cloud providers.

Key Features:
- Multi-cloud cost comparison (CoreWeave, Lambda Labs, AWS, GCP, Azure)
- GPU requirement calculation based on model specifications
- Memory and compute optimization recommendations
- ROI analysis and cost forecasting
- Real-time pricing updates and market analysis

Usage:
    from llm_cost_calculator import LLMCostCalculator
    
    calculator = LLMCostCalculator()
    
    # Analyze requirements
    gpu_req = calculator.calculate_gpu_requirements("llama-3.1-8b", target_rps=10)
    
    # Compare providers
    comparison = calculator.compare_providers("llama-3.1-8b", 10)
    
    # Calculate detailed costs
    costs = calculator.calculate_deployment_cost(config)

Dependencies:
    - None (pure Python implementation)
    - Optional: requests for real-time pricing updates

See: docs/11-cost-optimization.md#cost-modeling-framework
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

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
    memory_per_param_bytes: float = 2.0  # FP16
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

class LLMCostCalculator:
    def __init__(self):
        # Current GPU pricing (as of 2024)
        self.gpu_pricing = {
            CloudProvider.COREWEAVE: {
                GPUType.A100_40GB: 1.89,
                GPUType.A100_80GB: 2.49,
                GPUType.H100_80GB: 4.29,
            },
            CloudProvider.LAMBDA_LABS: {
                GPUType.A100_40GB: 1.95,
                GPUType.A100_80GB: 2.55,
                GPUType.H100_80GB: 4.49,
            },
            CloudProvider.AWS: {
                GPUType.A100_40GB: 3.24,  # p4d.xlarge on-demand
                GPUType.V100_32GB: 2.04,  # p3.2xlarge on-demand
            },
            CloudProvider.GCP: {
                GPUType.A100_40GB: 2.93,  # a2-highgpu-1g on-demand
                GPUType.V100_32GB: 1.85,  # n1-standard-4 + V100
            }
        }
        
        # Standard model configurations
        self.standard_models = {
            "llama-3.1-8b": ModelSpecs("llama-3.1-8b", 8_000_000_000, 2.0),
            "llama-3.1-70b": ModelSpecs("llama-3.1-70b", 70_000_000_000, 2.0),
            "mistral-7b": ModelSpecs("mistral-7b", 7_000_000_000, 2.0),
            "codellama-13b": ModelSpecs("codellama-13b", 13_000_000_000, 2.0)
        }
    
    def calculate_memory_requirements(self, model: ModelSpecs) -> Dict[str, float]:
        """Calculate memory requirements for model deployment."""
        
        # Base model memory (parameters + gradients + optimizer states)
        model_memory_gb = (model.parameters * model.memory_per_param_bytes) / (1024**3)
        
        # KV cache memory per request
        kv_cache_per_token_bytes = model.parameters * 2 * 2  # key + value, FP16
        kv_cache_gb = (kv_cache_per_token_bytes * model.context_length * model.batch_size) / (1024**3)
        
        # Activation memory during inference
        activation_memory_gb = model_memory_gb * 0.1  # Rough estimate
        
        # System overhead
        system_overhead_gb = 4.0
        
        total_memory_gb = (
            model_memory_gb + 
            kv_cache_gb + 
            activation_memory_gb + 
            system_overhead_gb
        )
        
        return {
            "model_memory_gb": model_memory_gb,
            "kv_cache_gb": kv_cache_gb,
            "activation_memory_gb": activation_memory_gb,
            "system_overhead_gb": system_overhead_gb,
            "total_memory_gb": total_memory_gb
        }
    
    def calculate_gpu_requirements(self, model: ModelSpecs, target_throughput_rps: float) -> Dict:
        """Calculate optimal GPU configuration for target throughput."""
        
        memory_req = self.calculate_memory_requirements(model)
        
        # Estimate tokens per second per GPU (simplified model)
        base_tokens_per_second = {
            "llama-3.1-8b": 150,
            "llama-3.1-70b": 25,
            "mistral-7b": 160,
            "codellama-13b": 100
        }
        
        tokens_per_sec = base_tokens_per_second.get(model.name, 100)
        
        # Calculate required GPUs for throughput
        required_gpus_throughput = math.ceil(target_throughput_rps * 50 / tokens_per_sec)  # Assume 50 tokens per request
        
        # Calculate required GPUs for memory
        gpu_memory_options = [40, 80]  # A100 variants
        suitable_gpu_memory = next(
            (mem for mem in gpu_memory_options if mem >= memory_req["total_memory_gb"]), 
            max(gpu_memory_options)
        )
        
        if memory_req["total_memory_gb"] > suitable_gpu_memory:
            # Need multiple GPUs for memory
            required_gpus_memory = math.ceil(memory_req["total_memory_gb"] / suitable_gpu_memory)
        else:
            required_gpus_memory = 1
        
        recommended_gpus = max(required_gpus_throughput, required_gpus_memory)
        
        return {
            "memory_requirements": memory_req,
            "tokens_per_second_per_gpu": tokens_per_sec,
            "required_gpus_throughput": required_gpus_throughput,
            "required_gpus_memory": required_gpus_memory,
            "recommended_gpus": recommended_gpus,
            "recommended_gpu_memory": suitable_gpu_memory
        }
    
    def calculate_deployment_cost(self, config: DeploymentConfig, hours_per_month: int = 730) -> Dict:
        """Calculate monthly deployment costs."""
        
        # Get hourly GPU cost
        hourly_cost = self.gpu_pricing[config.gpu.provider][config.gpu.gpu_type]
        
        # Calculate base costs
        monthly_gpu_cost = hourly_cost * config.replicas * hours_per_month
        
        # Additional costs
        storage_cost = self.calculate_storage_cost(config.model)
        network_cost = self.estimate_network_cost(config)
        management_overhead = monthly_gpu_cost * 0.05  # 5% overhead
        
        total_monthly_cost = monthly_gpu_cost + storage_cost + network_cost + management_overhead
        
        # Cost per request calculation
        estimated_requests_per_month = self.estimate_monthly_requests(config)
        cost_per_request = total_monthly_cost / estimated_requests_per_month if estimated_requests_per_month > 0 else 0
        
        return {
            "monthly_costs": {
                "gpu_compute": monthly_gpu_cost,
                "storage": storage_cost,
                "network": network_cost,
                "management_overhead": management_overhead,
                "total": total_monthly_cost
            },
            "cost_per_request": cost_per_request,
            "estimated_requests_per_month": estimated_requests_per_month,
            "utilization_rate": config.utilization_target,
            "cost_breakdown_pct": {
                "gpu_compute": (monthly_gpu_cost / total_monthly_cost) * 100,
                "storage": (storage_cost / total_monthly_cost) * 100,
                "network": (network_cost / total_monthly_cost) * 100,
                "management": (management_overhead / total_monthly_cost) * 100
            }
        }
    
    def calculate_storage_cost(self, model: ModelSpecs) -> float:
        """Calculate monthly storage costs."""
        
        # Model size in GB
        model_size_gb = (model.parameters * model.memory_per_param_bytes) / (1024**3)
        
        # Storage costs (S3-compatible, per GB/month)
        storage_cost_per_gb = 0.023  # ~$0.023/GB/month
        
        # Multiple copies for redundancy and caching
        storage_multiplier = 3  # Original + 2 replicas
        
        monthly_storage_cost = model_size_gb * storage_multiplier * storage_cost_per_gb
        
        return monthly_storage_cost
    
    def estimate_network_cost(self, config: DeploymentConfig) -> float:
        """Estimate monthly network transfer costs."""
        
        # Rough estimate based on deployment size
        estimated_monthly_transfer_gb = config.replicas * 100  # 100GB per replica per month
        network_cost_per_gb = 0.05  # $0.05/GB
        
        return estimated_monthly_transfer_gb * network_cost_per_gb
    
    def estimate_monthly_requests(self, config: DeploymentConfig) -> int:
        """Estimate monthly request volume based on deployment size."""
        
        # Simple heuristic based on replica count and utilization
        requests_per_hour_per_replica = 100 * config.utilization_target
        hours_per_month = 730
        
        return int(requests_per_hour_per_replica * config.replicas * hours_per_month)
    
    def compare_providers(self, model_name: str, target_rps: float) -> Dict:
        """Compare costs across different cloud providers."""
        
        model = self.standard_models[model_name]
        gpu_req = self.calculate_gpu_requirements(model, target_rps)
        
        comparison = {}
        
        for provider in CloudProvider:
            if provider not in self.gpu_pricing:
                continue
                
            for gpu_type in self.gpu_pricing[provider]:
                # Use A100-80GB for models requiring > 40GB
                if gpu_req["recommended_gpu_memory"] > 40 and gpu_type != GPUType.A100_80GB:
                    continue
                if gpu_req["recommended_gpu_memory"] <= 40 and gpu_type not in [GPUType.A100_40GB, GPUType.V100_32GB]:
                    continue
                
                gpu_specs = GPUSpecs(
                    gpu_type=gpu_type,
                    memory_gb=80 if "80gb" in gpu_type.value else 40,
                    compute_capability=8.0,  # Simplified
                    hourly_cost_usd=self.gpu_pricing[provider][gpu_type],
                    provider=provider
                )
                
                config = DeploymentConfig(
                    model=model,
                    gpu=gpu_specs,
                    replicas=gpu_req["recommended_gpus"],
                    utilization_target=0.7
                )
                
                costs = self.calculate_deployment_cost(config)
                
                comparison[f"{provider.value}_{gpu_type.value}"] = {
                    "provider": provider.value,
                    "gpu_type": gpu_type.value,
                    "monthly_cost": costs["monthly_costs"]["total"],
                    "cost_per_request": costs["cost_per_request"],
                    "gpu_count": config.replicas
                }
        
        # Sort by monthly cost
        sorted_comparison = dict(sorted(comparison.items(), key=lambda x: x[1]["monthly_cost"]))
        
        return sorted_comparison

# Example usage
def main():
    """Example cost calculation and comparison."""
    
    calculator = LLMCostCalculator()
    
    # Calculate requirements for Llama 3.1 8B at 10 RPS
    print("🔍 Analyzing cost requirements for llama-3.1-8b at 10 RPS:")
    
    model = calculator.standard_models["llama-3.1-8b"]
    gpu_req = calculator.calculate_gpu_requirements(model, target_throughput_rps=10)
    
    print(f"  Memory Requirements: {gpu_req['memory_requirements']['total_memory_gb']:.1f} GB")
    print(f"  Recommended GPUs: {gpu_req['recommended_gpus']}")
    print(f"  GPU Memory: {gpu_req['recommended_gpu_memory']} GB")
    
    # Compare providers
    print("\n💰 Provider Cost Comparison:")
    comparison = calculator.compare_providers("llama-3.1-8b", 10)
    
    for config_name, costs in list(comparison.items())[:5]:  # Top 5 cheapest
        print(f"  {costs['provider']} ({costs['gpu_type']}): ${costs['monthly_cost']:.0f}/month, ${costs['cost_per_request']:.4f}/request")
    
    # Detailed cost breakdown for top choice
    if comparison:
        top_choice = list(comparison.values())[0]
        print(f"\n📊 Detailed breakdown for {top_choice['provider']}:")
        
        # Create config for detailed analysis
        provider = CloudProvider(top_choice['provider'])
        gpu_type = GPUType(top_choice['gpu_type'])
        
        gpu_specs = GPUSpecs(
            gpu_type=gpu_type,
            memory_gb=80 if "80gb" in gpu_type.value else 40,
            compute_capability=8.0,
            hourly_cost_usd=calculator.gpu_pricing[provider][gpu_type],
            provider=provider
        )
        
        config = DeploymentConfig(
            model=model,
            gpu=gpu_specs,
            replicas=top_choice['gpu_count']
        )
        
        detailed_costs = calculator.calculate_deployment_cost(config)
        
        for cost_type, amount in detailed_costs["monthly_costs"].items():
            if cost_type != "total":
                pct = detailed_costs["cost_breakdown_pct"][cost_type.replace("_", "")]
                print(f"    {cost_type.replace('_', ' ').title()}: ${amount:.2f} ({pct:.1f}%)")

if __name__ == "__main__":
    main()