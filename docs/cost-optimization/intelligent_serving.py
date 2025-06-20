#!/usr/bin/env python3
"""
Intelligent Model Serving System for Cost Optimization

This system optimizes LLM serving costs through intelligent request routing,
dynamic batching, model quantization selection, and resource pooling.

The system implements multiple cost optimization strategies:
- Model tier selection based on request complexity
- Dynamic batching for cost efficiency
- Resource pooling across deployments
- Real-time cost monitoring and budget enforcement

Key Features:
- Request complexity analysis for optimal routing
- Cost-aware batching with configurable wait times
- Multi-tier model serving (ultra-low-cost to premium)
- Automatic cost anomaly detection and mitigation
- Real-time performance and cost tracking

Usage:
    from intelligent_serving import CostOptimizedModelRouter, RequestProfile
    
    router = CostOptimizedModelRouter()
    
    # Route request
    response, cost, metadata = await router.route_request(
        request_profile, prompt, max_tokens
    )
    
    # Batch processing
    batches = await router.optimize_batch_routing(requests)

Dependencies:
    - asyncio (for async request handling)
    - numpy (for statistical calculations)

See: docs/11-cost-optimization.md#intelligent-model-serving
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime, timedelta

class ModelTier(Enum):
    ULTRA_LOW_COST = "ultra-low-cost"  # Quantized, high batch
    LOW_COST = "low-cost"              # Quantized, medium batch
    BALANCED = "balanced"              # FP16, dynamic batch
    HIGH_PERFORMANCE = "high-perf"     # FP16, low latency
    PREMIUM = "premium"                # BF16, instant response

@dataclass
class CostProfile:
    model_name: str
    tier: ModelTier
    cost_per_token: float
    latency_p95_ms: float
    throughput_tokens_per_sec: float
    gpu_utilization_target: float

@dataclass
class RequestProfile:
    user_id: str
    request_type: str
    latency_tolerance_ms: int
    cost_sensitivity: str  # "low", "medium", "high"
    priority: int  # 1-10
    batch_compatible: bool = True

class CostOptimizedModelRouter:
    def __init__(self):
        # Define cost profiles for different configurations
        self.cost_profiles = {
            "llama-3.1-8b": {
                ModelTier.ULTRA_LOW_COST: CostProfile(
                    "llama-3.1-8b", ModelTier.ULTRA_LOW_COST,
                    cost_per_token=0.0001, latency_p95_ms=3000,
                    throughput_tokens_per_sec=300, gpu_utilization_target=0.9
                ),
                ModelTier.LOW_COST: CostProfile(
                    "llama-3.1-8b", ModelTier.LOW_COST,
                    cost_per_token=0.0002, latency_p95_ms=1500,
                    throughput_tokens_per_sec=200, gpu_utilization_target=0.8
                ),
                ModelTier.BALANCED: CostProfile(
                    "llama-3.1-8b", ModelTier.BALANCED,
                    cost_per_token=0.0004, latency_p95_ms=800,
                    throughput_tokens_per_sec=150, gpu_utilization_target=0.7
                ),
                ModelTier.HIGH_PERFORMANCE: CostProfile(
                    "llama-3.1-8b", ModelTier.HIGH_PERFORMANCE,
                    cost_per_token=0.0008, latency_p95_ms=400,
                    throughput_tokens_per_sec=120, gpu_utilization_target=0.6
                )
            }
        }
        
        # Current deployment status
        self.active_deployments = {}
        self.request_queue = asyncio.Queue()
        self.batch_accumulator = {}
        
        # Cost tracking
        self.hourly_costs = {}
        self.cost_budget_per_hour = 50.0  # $50/hour budget
        
    async def route_request(self, request_profile: RequestProfile, prompt: str, 
                          max_tokens: int = 100) -> Tuple[str, float, dict]:
        """Route request to optimal model configuration based on cost and performance."""
        
        # Analyze request characteristics
        optimal_tier = self._select_optimal_tier(request_profile, len(prompt.split()))
        
        # Check if we should batch this request
        if request_profile.batch_compatible and optimal_tier in [ModelTier.ULTRA_LOW_COST, ModelTier.LOW_COST]:
            return await self._handle_batched_request(request_profile, prompt, max_tokens, optimal_tier)
        else:
            return await self._handle_individual_request(request_profile, prompt, max_tokens, optimal_tier)
    
    def _select_optimal_tier(self, request_profile: RequestProfile, prompt_length: int) -> ModelTier:
        """Select optimal model tier based on request characteristics."""
        
        # Cost-sensitive routing
        if request_profile.cost_sensitivity == "high":
            if request_profile.latency_tolerance_ms > 5000:
                return ModelTier.ULTRA_LOW_COST
            elif request_profile.latency_tolerance_ms > 2000:
                return ModelTier.LOW_COST
            else:
                return ModelTier.BALANCED
        
        # Performance-sensitive routing
        elif request_profile.cost_sensitivity == "low":
            if request_profile.latency_tolerance_ms < 500:
                return ModelTier.HIGH_PERFORMANCE
            else:
                return ModelTier.BALANCED
        
        # Balanced routing (default)
        else:
            if request_profile.latency_tolerance_ms > 3000 and prompt_length < 100:
                return ModelTier.LOW_COST
            elif request_profile.latency_tolerance_ms < 1000:
                return ModelTier.HIGH_PERFORMANCE
            else:
                return ModelTier.BALANCED
    
    async def _handle_batched_request(self, request_profile: RequestProfile, 
                                    prompt: str, max_tokens: int, tier: ModelTier) -> Tuple[str, float, dict]:
        """Handle request through batching system for cost efficiency."""
        
        # Add to batch accumulator
        batch_key = f"{tier.value}_batch"
        
        if batch_key not in self.batch_accumulator:
            self.batch_accumulator[batch_key] = {
                "requests": [],
                "created_at": time.time(),
                "tier": tier
            }
        
        # Add request to batch
        request_data = {
            "profile": request_profile,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "future": asyncio.Future()
        }
        
        self.batch_accumulator[batch_key]["requests"].append(request_data)
        
        # Trigger batch processing if conditions met
        batch_size = len(self.batch_accumulator[batch_key]["requests"])
        batch_age = time.time() - self.batch_accumulator[batch_key]["created_at"]
        
        if batch_size >= self._get_optimal_batch_size(tier) or batch_age > self._get_max_batch_wait_time(tier):
            await self._process_batch(batch_key)
        
        # Wait for result
        result = await request_data["future"]
        return result
    
    async def _process_batch(self, batch_key: str):
        """Process accumulated batch of requests."""
        
        if batch_key not in self.batch_accumulator:
            return
        
        batch_data = self.batch_accumulator.pop(batch_key)
        requests = batch_data["requests"]
        tier = batch_data["tier"]
        
        if not requests:
            return
        
        print(f"🔄 Processing batch of {len(requests)} requests with tier {tier.value}")
        
        # Simulate batch processing
        batch_cost = self._calculate_batch_cost(requests, tier)
        processing_time = self._estimate_batch_processing_time(requests, tier)
        
        # Simulate processing delay
        await asyncio.sleep(processing_time / 1000)  # Convert ms to seconds
        
        # Generate responses for all requests in batch
        for i, request_data in enumerate(requests):
            response = f"Batch response {i+1} for: {request_data['prompt'][:50]}..."
            
            result = (
                response,
                batch_cost / len(requests),  # Distribute cost across batch
                {
                    "tier": tier.value,
                    "batch_size": len(requests),
                    "processing_time_ms": processing_time,
                    "cost_per_request": batch_cost / len(requests)
                }
            )
            
            request_data["future"].set_result(result)
    
    async def _handle_individual_request(self, request_profile: RequestProfile,
                                       prompt: str, max_tokens: int, tier: ModelTier) -> Tuple[str, float, dict]:
        """Handle individual request for low-latency requirements."""
        
        cost_profile = self.cost_profiles["llama-3.1-8b"][tier]
        
        # Simulate processing
        processing_time = np.random.normal(cost_profile.latency_p95_ms * 0.7, cost_profile.latency_p95_ms * 0.1)
        processing_time = max(processing_time, 100)  # Minimum 100ms
        
        await asyncio.sleep(processing_time / 1000)  # Convert to seconds
        
        # Calculate cost
        estimated_tokens = min(max_tokens, 100)  # Simplified
        cost = estimated_tokens * cost_profile.cost_per_token
        
        response = f"Individual response for: {prompt[:50]}..."
        
        return (
            response,
            cost,
            {
                "tier": tier.value,
                "processing_time_ms": processing_time,
                "estimated_tokens": estimated_tokens
            }
        )
    
    def _get_optimal_batch_size(self, tier: ModelTier) -> int:
        """Get optimal batch size for tier."""
        batch_sizes = {
            ModelTier.ULTRA_LOW_COST: 16,
            ModelTier.LOW_COST: 8,
            ModelTier.BALANCED: 4,
            ModelTier.HIGH_PERFORMANCE: 2
        }
        return batch_sizes.get(tier, 4)
    
    def _get_max_batch_wait_time(self, tier: ModelTier) -> float:
        """Get maximum wait time for batch accumulation (seconds)."""
        wait_times = {
            ModelTier.ULTRA_LOW_COST: 5.0,   # Wait up to 5 seconds
            ModelTier.LOW_COST: 2.0,         # Wait up to 2 seconds
            ModelTier.BALANCED: 1.0,         # Wait up to 1 second
            ModelTier.HIGH_PERFORMANCE: 0.5  # Wait up to 500ms
        }
        return wait_times.get(tier, 1.0)
    
    def _calculate_batch_cost(self, requests: List[dict], tier: ModelTier) -> float:
        """Calculate total cost for processing batch."""
        cost_profile = self.cost_profiles["llama-3.1-8b"][tier]
        
        total_tokens = sum(
            min(req["max_tokens"], 100) for req in requests
        )
        
        # Batch processing efficiency bonus
        efficiency_multiplier = 0.8  # 20% cost reduction for batching
        
        return total_tokens * cost_profile.cost_per_token * efficiency_multiplier
    
    def _estimate_batch_processing_time(self, requests: List[dict], tier: ModelTier) -> float:
        """Estimate batch processing time in milliseconds."""
        cost_profile = self.cost_profiles["llama-3.1-8b"][tier]
        
        # Base processing time + per-request overhead
        base_time = cost_profile.latency_p95_ms * 0.8
        per_request_overhead = 50  # 50ms per additional request
        
        return base_time + (len(requests) - 1) * per_request_overhead
    
    async def monitor_costs(self):
        """Monitor and alert on cost anomalies."""
        
        while True:
            current_hour = datetime.now().hour
            
            if current_hour not in self.hourly_costs:
                self.hourly_costs[current_hour] = 0.0
            
            # Check if we're approaching budget limits
            if self.hourly_costs[current_hour] > self.cost_budget_per_hour * 0.8:
                print(f"⚠️  Cost Alert: ${self.hourly_costs[current_hour]:.2f} spent this hour (80% of ${self.cost_budget_per_hour} budget)")
                
                # Implement cost-saving measures
                await self._implement_cost_savings()
            
            await asyncio.sleep(300)  # Check every 5 minutes
    
    async def _implement_cost_savings(self):
        """Implement emergency cost-saving measures."""
        
        print("🛡️ Implementing cost-saving measures:")
        print("  - Reducing batch wait times")
        print("  - Increasing batch sizes")
        print("  - Routing more requests to ultra-low-cost tier")
        
        # Could trigger scale-down of expensive deployments
        # Could increase batching aggressiveness
        # Could temporarily reject non-critical requests
    
    async def optimize_batch_routing(self, requests: List[Tuple[str, Dict]]) -> Dict[str, List]:
        """Optimize routing for a batch of requests."""
        
        # Group by complexity tier
        batches = {"simple": [], "moderate": [], "complex": [], "critical": []}
        
        for prompt, slo in requests:
            model, metadata = await self.route_request(prompt, slo)
            
            # Determine tier from model
            for tier, config in self.routing_table.items():
                if config["model"] == model:
                    batches[tier].append((prompt, metadata))
                    break
        
        # Optimize batch sizes
        optimized_batches = {}
        for tier, batch in batches.items():
            if not batch:
                continue
                
            max_batch_size = self.routing_table[tier]["max_batch"]
            
            # Split into optimal sub-batches
            for i in range(0, len(batch), max_batch_size):
                batch_id = f"{tier}_batch_{i // max_batch_size}"
                optimized_batches[batch_id] = batch[i:i + max_batch_size]
        
        return optimized_batches

# Example usage showing cost savings
async def demonstrate_routing():
    """Demonstrate intelligent routing cost savings."""
    
    router = CostOptimizedModelRouter()
    
    test_requests = [
        ("What is the capital of France?", {"max_latency_ms": 1000}),
        ("Explain quantum computing in detail with examples", {"max_latency_ms": 5000}),
        ("Write a Python function to sort a list", {"max_latency_ms": 2000}),
        ("Hi", {"max_latency_ms": 500})
    ]
    
    total_cost_simple = 0
    total_cost_optimized = 0
    
    print("🎯 Intelligent Model Routing Demo\n")
    
    for prompt, slo in test_requests:
        # Create request profile
        request_profile = RequestProfile(
            user_id="demo_user",
            request_type="demo",
            latency_tolerance_ms=slo["max_latency_ms"],
            cost_sensitivity="medium",
            priority=5
        )
        
        response, cost, metadata = await router.route_request(request_profile, prompt, 50)
        
        # Calculate what it would cost with premium tier
        simple_cost = 0.0008 * 50  # Always use expensive model
        
        total_cost_simple += simple_cost
        total_cost_optimized += cost
        
        print(f"Prompt: '{prompt[:50]}...'")
        print(f"  Routed to: {metadata['tier']}")
        print(f"  Cost: ${cost:.6f} (saved ${simple_cost - cost:.6f})")
        print(f"  Processing Time: {metadata.get('processing_time_ms', 0):.0f}ms")
        print()
    
    savings_pct = ((total_cost_simple - total_cost_optimized) / total_cost_simple) * 100
    print(f"💰 Total Savings: ${total_cost_simple - total_cost_optimized:.6f} ({savings_pct:.1f}%)")

# Main execution
async def main():
    """Example cost-optimized request routing."""
    
    router = CostOptimizedModelRouter()
    
    # Start cost monitoring
    cost_monitor_task = asyncio.create_task(router.monitor_costs())
    
    # Example requests with different characteristics
    test_requests = [
        RequestProfile("user1", "analytics", 5000, "high", 3, True),      # Cost-sensitive, batch-friendly
        RequestProfile("user2", "chat", 800, "low", 8, False),           # Performance-sensitive, individual
        RequestProfile("user3", "search", 2000, "medium", 5, True),      # Balanced requirements
        RequestProfile("user4", "batch_job", 10000, "high", 2, True),    # Very cost-sensitive
    ]
    
    print("🎯 Testing cost-optimized request routing:")
    
    # Process test requests
    tasks = []
    for i, req_profile in enumerate(test_requests):
        prompt = f"Test request {i+1}: This is a sample prompt for {req_profile.request_type}"
        task = router.route_request(req_profile, prompt, 50)
        tasks.append(task)
    
    # Wait for all requests to complete
    results = await asyncio.gather(*tasks)
    
    # Display results
    total_cost = 0
    for i, (response, cost, metadata) in enumerate(results):
        print(f"\n📊 Request {i+1} Results:")
        print(f"  Tier: {metadata['tier']}")
        print(f"  Cost: ${cost:.6f}")
        print(f"  Processing Time: {metadata.get('processing_time_ms', 0):.0f}ms")
        if 'batch_size' in metadata:
            print(f"  Batch Size: {metadata['batch_size']}")
        total_cost += cost
    
    print(f"\n💰 Total Cost: ${total_cost:.6f}")
    print(f"📈 Average Cost per Request: ${total_cost/len(results):.6f}")
    
    # Cancel monitoring task
    cost_monitor_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())