#!/usr/bin/env python3
"""
Advanced Dynamic Model Router for llm-d

This router leverages llm-d's unique features to optimize costs through
intelligent request routing based on complexity analysis and SLO requirements.

Key llm-d Features Utilized:
- Speculative decoding for latency reduction
- Memory pooling across deployments
- Request complexity analysis
- Dynamic batch optimization
- Cost-aware routing decisions

The router analyzes each request's complexity and routes it to the most
cost-effective model that can meet the quality and latency requirements.

Usage:
    from dynamic_router import LLMDDynamicRouter

    router = LLMDDynamicRouter()

    # Route single request
    model, metadata = await router.route_request(prompt, slo_requirements)

    # Optimize batch routing
    batches = await router.optimize_batch_routing(requests)

Dependencies:
    - asyncio (for async operations)
    - numpy (for statistical calculations)

See: docs/11-cost-optimization.md#dynamic-model-routing
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class RequestComplexity:
    """Analyze request complexity for routing."""

    prompt_length: int
    expected_output_length: int
    complexity_score: float  # 0-1
    requires_reasoning: bool
    domain: str  # "general", "code", "math", etc.


class LLMDDynamicRouter:
    """
    Dynamic router leveraging llm-d's unique features:
    - Speculative decoding for latency reduction
    - Memory pooling across deployments
    - Request complexity analysis
    """

    def __init__(self):
        self.routing_table = {
            "simple": {
                "model": "llama-3.1-8b-int8",
                "max_batch": 32,
                "cost_per_token": 0.0001,
            },
            "moderate": {
                "model": "llama-3.1-8b-fp16",
                "max_batch": 16,
                "cost_per_token": 0.0004,
            },
            "complex": {
                "model": "llama-3.1-70b-int8",
                "max_batch": 4,
                "cost_per_token": 0.002,
            },
            "critical": {
                "model": "llama-3.1-70b-fp16",
                "max_batch": 1,
                "cost_per_token": 0.008,
            },
        }

        # llm-d specific: Speculative decoding pairs
        self.speculative_pairs = {
            "llama-3.1-70b-fp16": "llama-3.1-8b-int8",  # Draft model
            "llama-3.1-70b-int8": "llama-3.1-8b-int4",
        }

    def analyze_request_complexity(
        self, prompt: str, context: Optional[Dict] = None
    ) -> RequestComplexity:
        """Analyze request to determine optimal routing."""

        # Simple complexity heuristics
        prompt_length = len(prompt.split())

        # Check for reasoning indicators
        reasoning_keywords = ["explain", "why", "how", "analyze", "compare", "evaluate"]
        requires_reasoning = any(
            keyword in prompt.lower() for keyword in reasoning_keywords
        )

        # Estimate output length
        if "write" in prompt.lower() or "generate" in prompt.lower():
            expected_output_length = prompt_length * 5
        elif "summarize" in prompt.lower():
            expected_output_length = prompt_length // 3
        else:
            expected_output_length = prompt_length * 2

        # Calculate complexity score
        complexity_score = min(
            1.0,
            (
                (prompt_length / 500) * 0.3  # Length factor
                + (requires_reasoning * 0.4)  # Reasoning factor
                + (expected_output_length / 1000) * 0.3  # Output factor
            ),
        )

        # Determine domain
        domain = "general"
        if "```" in prompt or "code" in prompt.lower():
            domain = "code"
        elif any(
            math_term in prompt.lower()
            for math_term in ["equation", "calculate", "solve"]
        ):
            domain = "math"

        return RequestComplexity(
            prompt_length=prompt_length,
            expected_output_length=expected_output_length,
            complexity_score=complexity_score,
            requires_reasoning=requires_reasoning,
            domain=domain,
        )

    async def route_request(
        self, prompt: str, slo_requirements: Dict[str, float]
    ) -> Tuple[str, Dict]:
        """Route request to optimal model based on complexity and SLOs."""

        complexity = self.analyze_request_complexity(prompt)

        # Determine routing tier based on complexity and SLOs
        if (
            complexity.complexity_score < 0.3
            and slo_requirements.get("max_latency_ms", 1000) > 500
        ):
            tier = "simple"
        elif complexity.complexity_score < 0.6:
            tier = "moderate"
        elif complexity.complexity_score < 0.8:
            tier = "complex"
        else:
            tier = "critical"

        # Override for specific domains
        if complexity.domain == "code" and complexity.requires_reasoning:
            tier = "complex"  # Code generation needs better models

        route_config = self.routing_table[tier]

        # Enable speculative decoding for expensive models
        if tier in ["complex", "critical"]:
            draft_model = self.speculative_pairs.get(route_config["model"])
            if draft_model:
                route_config["speculative_decoding"] = {
                    "enabled": True,
                    "draft_model": draft_model,
                    "verification_batch_size": 4,
                }

        # Memory pooling optimization
        route_config["memory_pool"] = {
            "enabled": True,
            "pool_name": f"{tier}_pool",
            "max_allocation_gb": 100,
        }

        return route_config["model"], {
            "config": route_config,
            "complexity": complexity,
            "estimated_cost": route_config["cost_per_token"]
            * complexity.expected_output_length,
        }

    async def optimize_batch_routing(
        self, requests: List[Tuple[str, Dict]]
    ) -> Dict[str, List]:
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
                optimized_batches[batch_id] = batch[i : i + max_batch_size]

        return optimized_batches


# Example usage showing cost savings
async def demonstrate_routing():
    """Demonstrate dynamic routing cost savings."""

    router = LLMDDynamicRouter()

    test_requests = [
        ("What is the capital of France?", {"max_latency_ms": 1000}),
        ("Explain quantum computing in detail with examples", {"max_latency_ms": 5000}),
        ("Write a Python function to sort a list", {"max_latency_ms": 2000}),
        ("Hi", {"max_latency_ms": 500}),
    ]

    total_cost_simple = 0
    total_cost_optimized = 0

    print("🎯 Dynamic Model Routing Demo\n")

    for prompt, slo in test_requests:
        model, metadata = await router.route_request(prompt, slo)

        # Calculate costs
        simple_cost = (
            0.008 * metadata["complexity"].expected_output_length
        )  # Always use expensive model
        optimized_cost = metadata["estimated_cost"]

        total_cost_simple += simple_cost
        total_cost_optimized += optimized_cost

        print(f"Prompt: '{prompt[:50]}...'")
        print(f"  Complexity: {metadata['complexity'].complexity_score:.2f}")
        print(f"  Routed to: {model}")
        print(
            f"  Cost: ${optimized_cost:.6f} (saved ${simple_cost - optimized_cost:.6f})"
        )

        if "speculative_decoding" in metadata["config"]:
            print(
                f"  ⚡ Speculative decoding enabled with {metadata['config']['speculative_decoding']['draft_model']}"
            )

        print()

    savings_pct = ((total_cost_simple - total_cost_optimized) / total_cost_simple) * 100
    print(
        f"💰 Total Savings: ${total_cost_simple - total_cost_optimized:.6f} ({savings_pct:.1f}%)"
    )


if __name__ == "__main__":
    asyncio.run(demonstrate_routing())
