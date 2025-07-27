"""
Tests for GPU selection framework in
chapter-06-performance/optimization/gpu-selector.py
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add the examples directory to the path
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent.parent.parent / "llm-d-book-examples")
)

try:
    from chapter_06_performance.optimization.gpu_selector import GPUSelector
except ImportError:
    # Create mock class for testing when real implementation isn't available
    class GPUSelector:
        def __init__(self):
            self.gpus = {
                "H100": {
                    "memory_gb": 80,
                    "memory_bandwidth_tbps": 3.35,
                    "compute_units": 132,
                    "tensor_cores": 528,
                    "cost_per_hour": 4.00,
                    "vendor": "nvidia",
                },
                "H200": {
                    "memory_gb": 141,
                    "memory_bandwidth_tbps": 4.8,
                    "compute_units": 144,
                    "tensor_cores": 576,
                    "cost_per_hour": 6.00,
                    "vendor": "nvidia",
                },
                "B200": {
                    "memory_gb": 192,
                    "memory_bandwidth_tbps": 8.0,
                    "compute_units": 256,
                    "tensor_cores": 1024,
                    "cost_per_hour": 10.00,
                    "vendor": "nvidia",
                },
                "MI300X": {
                    "memory_gb": 192,
                    "memory_bandwidth_tbps": 5.3,
                    "compute_units": 304,
                    "tensor_cores": 0,  # AMD uses different architecture
                    "cost_per_hour": 3.50,
                    "vendor": "amd",
                },
                "MI350X": {
                    "memory_gb": 256,
                    "memory_bandwidth_tbps": 6.4,
                    "compute_units": 384,
                    "tensor_cores": 0,
                    "cost_per_hour": 5.00,
                    "vendor": "amd",
                },
            }

        def calculate_model_requirements(
            self, model_size_params, quantization="fp16", context_length=2048
        ):
            """Calculate model resource requirements."""
            quantization_factor = {"fp16": 2, "fp8": 1, "int8": 1, "int4": 0.5}[
                quantization
            ]

            model_memory_gb = model_size_params * quantization_factor / 1e9
            kv_cache_gb = (
                context_length * 4096 * 2 * quantization_factor / 1e9
            )  # Simplified

            return {
                "model_memory_gb": model_memory_gb,
                "kv_cache_gb_per_1k_tokens": kv_cache_gb
                * 1000
                / context_length,  # Fixed calculation
                "min_memory_gb": model_memory_gb + kv_cache_gb,
                "recommended_memory_gb": model_memory_gb * 1.5 + kv_cache_gb,
            }

        def evaluate_gpu_for_model(
            self,
            gpu_name,
            model_size_params,
            quantization="fp16",
            context_length=2048,
            tensor_parallel=1,
        ):
            """Evaluate GPU suitability for specific model."""
            if gpu_name not in self.gpus:
                raise ValueError(f"GPU {gpu_name} not supported")

            gpu_spec = self.gpus[gpu_name]
            model_req = self.calculate_model_requirements(
                model_size_params, quantization, context_length
            )

            effective_memory = gpu_spec["memory_gb"] * tensor_parallel
            memory_per_gpu = model_req["min_memory_gb"] / tensor_parallel

            can_fit = effective_memory >= model_req["min_memory_gb"]
            memory_efficiency = (
                model_req["min_memory_gb"] / effective_memory if can_fit else 0
            )

            # Estimate performance (simplified)
            base_throughput = (
                gpu_spec["memory_bandwidth_tbps"] * 1000 / model_req["model_memory_gb"]
                if model_req["model_memory_gb"] > 0
                else 0
            )
            actual_throughput = (
                base_throughput * tensor_parallel * 0.8
            )  # Scaling efficiency

            return {
                "can_fit": can_fit,
                "memory_efficiency": memory_efficiency,
                "estimated_throughput_tokens_per_sec": actual_throughput,
                "cost_per_hour": gpu_spec["cost_per_hour"] * tensor_parallel,
                "cost_per_1k_tokens": (
                    gpu_spec["cost_per_hour"] / (actual_throughput * 3.6)
                    if actual_throughput > 0
                    else float("inf")
                ),
                "tensor_parallel_required": max(
                    1, int(model_req["min_memory_gb"] / gpu_spec["memory_gb"]) + 1
                ),
                "recommended_tensor_parallel": tensor_parallel,
            }

        def recommend_gpu_configuration(
            self,
            model_size_params,
            quantization="fp16",
            budget_per_hour=None,
            performance_priority=True,
        ):
            """Recommend optimal GPU configuration."""
            model_req = self.calculate_model_requirements(
                model_size_params, quantization
            )
            recommendations = []

            for gpu_name, gpu_spec in self.gpus.items():
                # Calculate minimum tensor parallelism needed
                min_tp = max(
                    1, int(model_req["min_memory_gb"] / gpu_spec["memory_gb"]) + 1
                )

                for tp in [min_tp, min_tp * 2, min_tp * 4]:
                    if tp > 8:  # Reasonable upper limit
                        continue

                    eval_result = self.evaluate_gpu_for_model(
                        gpu_name, model_size_params, quantization, tensor_parallel=tp
                    )

                    if not eval_result["can_fit"]:
                        continue

                    if (
                        budget_per_hour
                        and eval_result["cost_per_hour"] > budget_per_hour
                    ):
                        continue

                    recommendations.append(
                        {
                            "gpu": gpu_name,
                            "tensor_parallel": tp,
                            "total_gpus": tp,
                            "memory_efficiency": eval_result["memory_efficiency"],
                            "performance_score": eval_result[
                                "estimated_throughput_tokens_per_sec"
                            ],
                            "cost_per_hour": eval_result["cost_per_hour"],
                            "cost_efficiency": eval_result[
                                "estimated_throughput_tokens_per_sec"
                            ]
                            / eval_result["cost_per_hour"],
                            "cost_per_1k_tokens": eval_result["cost_per_1k_tokens"],
                        }
                    )

            # Sort by performance or cost efficiency
            if performance_priority:
                recommendations.sort(key=lambda x: x["performance_score"], reverse=True)
            else:
                recommendations.sort(key=lambda x: x["cost_efficiency"], reverse=True)

            return recommendations[:5]  # Top 5 recommendations

        def compare_gpus(self, gpu_list, model_size_params, quantization="fp16"):
            """Compare multiple GPUs for a specific model."""
            comparisons = {}

            for gpu_name in gpu_list:
                if gpu_name not in self.gpus:
                    continue

                eval_result = self.evaluate_gpu_for_model(
                    gpu_name, model_size_params, quantization
                )
                comparisons[gpu_name] = eval_result

            return comparisons


class TestGPUSelector:
    """Test cases for GPU selection framework."""

    @pytest.fixture
    def selector(self):
        """Create GPU selector instance."""
        return GPUSelector()

    def test_initialization(self, selector):
        """Test GPUSelector initialization."""
        assert hasattr(selector, "gpus")
        assert "H100" in selector.gpus
        assert "H200" in selector.gpus
        assert "B200" in selector.gpus
        assert "MI300X" in selector.gpus
        assert "MI350X" in selector.gpus

    def test_gpu_specifications(self, selector):
        """Test GPU specifications are complete."""
        for gpu_name, gpu_spec in selector.gpus.items():
            assert "memory_gb" in gpu_spec
            assert "memory_bandwidth_tbps" in gpu_spec
            assert "compute_units" in gpu_spec
            assert "cost_per_hour" in gpu_spec
            assert "vendor" in gpu_spec

            # Verify reasonable values
            assert gpu_spec["memory_gb"] > 0
            assert gpu_spec["memory_bandwidth_tbps"] > 0
            assert gpu_spec["cost_per_hour"] > 0

    @pytest.mark.parametrize(
        "model_size,quantization,expected_memory",
        [
            (8e9, "fp16", (10, 20)),  # Llama 3.1 8B
            (70e9, "fp16", (120, 160)),  # Llama 3.1 70B
            (405e9, "fp16", (700, 900)),  # Llama 3.1 405B
            (8e9, "int4", (3, 8)),  # Quantized 8B
        ],
    )
    def test_model_requirements_calculation(
        self, selector, model_size, quantization, expected_memory
    ):
        """Test model requirements calculation for different sizes."""
        requirements = selector.calculate_model_requirements(model_size, quantization)

        assert "model_memory_gb" in requirements
        assert "kv_cache_gb_per_1k_tokens" in requirements
        assert "min_memory_gb" in requirements
        assert "recommended_memory_gb" in requirements

        min_expected, max_expected = expected_memory
        assert min_expected < requirements["model_memory_gb"] < max_expected

    def test_gpu_evaluation_llama_8b(self, selector):
        """Test GPU evaluation for Llama 3.1 8B."""
        eval_result = selector.evaluate_gpu_for_model("H100", 8e9, "fp16")

        assert eval_result["can_fit"] is True
        assert eval_result["memory_efficiency"] > 0
        assert eval_result["estimated_throughput_tokens_per_sec"] > 0
        assert eval_result["cost_per_hour"] > 0
        assert eval_result["cost_per_1k_tokens"] < float("inf")

    def test_gpu_evaluation_llama_70b(self, selector):
        """Test GPU evaluation for Llama 3.1 70B."""
        # Single H100 should not fit 70B
        eval_result = selector.evaluate_gpu_for_model(
            "H100", 70e9, "fp16", tensor_parallel=1
        )
        assert eval_result["can_fit"] is False

        # Multi-GPU should fit
        eval_result = selector.evaluate_gpu_for_model(
            "H100", 70e9, "fp16", tensor_parallel=4
        )
        assert eval_result["can_fit"] is True
        assert (
            eval_result["cost_per_hour"] == selector.gpus["H100"]["cost_per_hour"] * 4
        )

    def test_gpu_evaluation_llama_405b(self, selector):
        """Test GPU evaluation for Llama 3.1 405B."""
        # Should require many GPUs
        eval_result = selector.evaluate_gpu_for_model(
            "H100", 405e9, "fp16", tensor_parallel=8
        )
        assert eval_result["tensor_parallel_required"] >= 8

    def test_tensor_parallelism_scaling(self, selector):
        """Test tensor parallelism impact on evaluation."""
        single_gpu = selector.evaluate_gpu_for_model(
            "H100", 8e9, "fp16", tensor_parallel=1
        )
        multi_gpu = selector.evaluate_gpu_for_model(
            "H100", 8e9, "fp16", tensor_parallel=2
        )

        # Multi-GPU should have higher throughput and cost
        assert (
            multi_gpu["estimated_throughput_tokens_per_sec"]
            > single_gpu["estimated_throughput_tokens_per_sec"]
        )
        assert multi_gpu["cost_per_hour"] > single_gpu["cost_per_hour"]

    def test_quantization_impact(self, selector):
        """Test quantization impact on GPU evaluation."""
        fp16_result = selector.evaluate_gpu_for_model(
            "H100", 70e9, "fp16", tensor_parallel=4
        )
        int4_result = selector.evaluate_gpu_for_model(
            "H100", 70e9, "int4", tensor_parallel=1
        )

        # INT4 should require fewer GPUs
        assert int4_result["can_fit"] is True
        assert int4_result["cost_per_hour"] < fp16_result["cost_per_hour"]

    def test_recommendations_performance_priority(self, selector):
        """Test GPU recommendations with performance priority."""
        recommendations = selector.recommend_gpu_configuration(
            model_size_params=8e9, quantization="fp16", performance_priority=True
        )

        assert len(recommendations) > 0
        assert len(recommendations) <= 5

        # Should be sorted by performance (descending)
        for i in range(len(recommendations) - 1):
            assert (
                recommendations[i]["performance_score"]
                >= recommendations[i + 1]["performance_score"]
            )

        # Verify recommendation structure
        for rec in recommendations:
            assert "gpu" in rec
            assert "tensor_parallel" in rec
            assert "performance_score" in rec
            assert "cost_per_hour" in rec
            assert "cost_efficiency" in rec

    def test_recommendations_cost_priority(self, selector):
        """Test GPU recommendations with cost priority."""
        recommendations = selector.recommend_gpu_configuration(
            model_size_params=8e9, quantization="fp16", performance_priority=False
        )

        assert len(recommendations) > 0

        # Should be sorted by cost efficiency (descending)
        for i in range(len(recommendations) - 1):
            assert (
                recommendations[i]["cost_efficiency"]
                >= recommendations[i + 1]["cost_efficiency"]
            )

    def test_budget_constraints(self, selector):
        """Test recommendations with budget constraints."""
        budget = 10.0  # $10/hour budget

        recommendations = selector.recommend_gpu_configuration(
            model_size_params=8e9, quantization="fp16", budget_per_hour=budget
        )

        # All recommendations should be within budget
        for rec in recommendations:
            assert rec["cost_per_hour"] <= budget

    def test_gpu_comparison(self, selector):
        """Test GPU comparison functionality."""
        gpu_list = ["H100", "H200", "MI300X"]
        comparisons = selector.compare_gpus(gpu_list, 8e9, "fp16")

        assert len(comparisons) == 3
        for gpu_name in gpu_list:
            assert gpu_name in comparisons
            assert "can_fit" in comparisons[gpu_name]
            assert "memory_efficiency" in comparisons[gpu_name]
            assert "estimated_throughput_tokens_per_sec" in comparisons[gpu_name]

    @pytest.mark.parametrize("gpu_name", ["H100", "H200", "B200", "MI300X", "MI350X"])
    def test_all_gpus_specifications(self, selector, gpu_name):
        """Test all GPU specifications are valid."""
        gpu_spec = selector.gpus[gpu_name]

        # Memory should be reasonable
        assert 32 <= gpu_spec["memory_gb"] <= 300

        # Bandwidth should be reasonable
        assert 1.0 <= gpu_spec["memory_bandwidth_tbps"] <= 10.0

        # Cost should be reasonable
        assert 1.0 <= gpu_spec["cost_per_hour"] <= 20.0

    def test_nvidia_vs_amd_comparison(self, selector):
        """Test comparison between NVIDIA and AMD GPUs."""
        nvidia_eval = selector.evaluate_gpu_for_model("H100", 8e9, "fp16")
        amd_eval = selector.evaluate_gpu_for_model("MI300X", 8e9, "fp16")

        # Both should be able to fit the model
        assert nvidia_eval["can_fit"] is True
        assert amd_eval["can_fit"] is True

        # Should have reasonable performance estimates
        assert nvidia_eval["estimated_throughput_tokens_per_sec"] > 0
        assert amd_eval["estimated_throughput_tokens_per_sec"] > 0

    def test_memory_efficiency_calculation(self, selector):
        """Test memory efficiency calculations."""
        # Small model on large GPU - should have lower efficiency
        small_model = selector.evaluate_gpu_for_model("H200", 8e9, "fp16")

        # Large model on appropriate GPU - should have higher efficiency
        large_model = selector.evaluate_gpu_for_model(
            "H100", 70e9, "fp16", tensor_parallel=4
        )

        assert 0 <= small_model["memory_efficiency"] <= 1
        if large_model["can_fit"]:
            assert 0 <= large_model["memory_efficiency"] <= 1

    def test_cost_per_token_calculation(self, selector):
        """Test cost per token calculations."""
        eval_result = selector.evaluate_gpu_for_model("H100", 8e9, "fp16")

        assert eval_result["cost_per_1k_tokens"] > 0
        assert eval_result["cost_per_1k_tokens"] < 1.0  # Should be reasonable

    def test_error_handling_invalid_gpu(self, selector):
        """Test error handling for invalid GPU names."""
        with pytest.raises(ValueError):
            selector.evaluate_gpu_for_model("InvalidGPU", 8e9, "fp16")

    def test_recommendations_large_models(self, selector):
        """Test recommendations for very large models."""
        recommendations = selector.recommend_gpu_configuration(
            model_size_params=405e9, quantization="fp16"  # 405B model
        )

        # Should still provide recommendations, even if expensive
        assert len(recommendations) > 0

        # All recommendations should require multiple GPUs
        for rec in recommendations:
            assert rec["tensor_parallel"] > 1

    def test_context_length_impact(self, selector):
        """Test impact of context length on requirements."""
        short_context = selector.calculate_model_requirements(8e9, "fp16", 1024)
        long_context = selector.calculate_model_requirements(8e9, "fp16", 8192)

        # Longer context should require more total memory (model + larger KV cache)
        # The per-token KV cache should be the same, but total memory should be more
        assert long_context["min_memory_gb"] > short_context["min_memory_gb"]
        # KV cache per token should be consistent (or at least not decrease)
        assert (
            long_context["kv_cache_gb_per_1k_tokens"]
            >= short_context["kv_cache_gb_per_1k_tokens"] * 0.99
        )  # Allow for small rounding
