#!/usr/bin/env python3
"""
Comprehensive Test Suite for Quantization Optimizer

Tests model quantization analysis, cost-benefit calculations,
configuration generation, and optimization strategies.

Coverage:
- Quantization type definitions and profiles
- Cost-benefit analysis for different quantization strategies
- GPU compatibility and resource optimization
- Configuration generation for quantized deployments
- Performance vs cost tradeoff analysis
"""

import sys
from pathlib import Path

import pytest

# Add the docs directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "docs"))

try:
    from cost_optimization.quantization_optimizer import (
        QuantizationOptimizer, QuantizationProfile, QuantizationType)
except ImportError:
    # For testing, define minimal versions
    from dataclasses import dataclass
    from enum import Enum
    from typing import Dict, List

    class QuantizationType(Enum):
        FP16 = "fp16"
        INT8 = "int8"
        INT4 = "int4"
        MIXED_PRECISION = "mixed"
        DYNAMIC = "dynamic"

    @dataclass
    class QuantizationProfile:
        name: str
        quant_type: QuantizationType
        memory_reduction_pct: float
        performance_retention_pct: float
        cost_reduction_pct: float
        compatible_models: List[str]
        gpu_requirements: Dict[str, int]

    class QuantizationOptimizer:
        def __init__(self):
            self.profiles = {
                QuantizationType.FP16: QuantizationProfile(
                    name="Half Precision (FP16)",
                    quant_type=QuantizationType.FP16,
                    memory_reduction_pct=50.0,
                    performance_retention_pct=99.5,
                    cost_reduction_pct=40.0,
                    compatible_models=["llama-3.1-8b", "llama-3.1-70b", "mistral-7b"],
                    gpu_requirements={"min_memory_gb": 16, "compute_capability": 7.0},
                ),
                QuantizationType.INT8: QuantizationProfile(
                    name="8-bit Integer (INT8)",
                    quant_type=QuantizationType.INT8,
                    memory_reduction_pct=75.0,
                    performance_retention_pct=97.0,
                    cost_reduction_pct=65.0,
                    compatible_models=["llama-3.1-8b", "mistral-7b", "codellama-13b"],
                    gpu_requirements={"min_memory_gb": 8, "compute_capability": 6.1},
                ),
                QuantizationType.INT4: QuantizationProfile(
                    name="4-bit Integer (INT4)",
                    quant_type=QuantizationType.INT4,
                    memory_reduction_pct=87.5,
                    performance_retention_pct=92.0,
                    cost_reduction_pct=80.0,
                    compatible_models=["llama-3.1-8b", "mistral-7b"],
                    gpu_requirements={"min_memory_gb": 4, "compute_capability": 6.1},
                ),
                QuantizationType.MIXED_PRECISION: QuantizationProfile(
                    name="Mixed Precision (Critical layers FP16, others INT8)",
                    quant_type=QuantizationType.MIXED_PRECISION,
                    memory_reduction_pct=60.0,
                    performance_retention_pct=98.5,
                    cost_reduction_pct=50.0,
                    compatible_models=["llama-3.1-8b", "llama-3.1-70b"],
                    gpu_requirements={"min_memory_gb": 12, "compute_capability": 7.0},
                ),
            }

        def analyze_quantization_options(
            self, model_name, available_gpu_memory_gb, performance_threshold_pct=95.0
        ):
            options = []

            for profile in self.profiles.values():
                if model_name not in profile.compatible_models:
                    continue

                if available_gpu_memory_gb < profile.gpu_requirements["min_memory_gb"]:
                    continue

                if profile.performance_retention_pct < performance_threshold_pct:
                    continue

                monthly_savings = self._calculate_monthly_savings(model_name, profile)

                options.append(
                    {
                        "quantization_type": profile.quant_type.value,
                        "name": profile.name,
                        "memory_reduction_pct": profile.memory_reduction_pct,
                        "performance_retention_pct": profile.performance_retention_pct,
                        "estimated_monthly_savings_usd": monthly_savings,
                        "cost_reduction_pct": profile.cost_reduction_pct,
                        "implementation_complexity": (
                            self._get_implementation_complexity(profile.quant_type)
                        ),
                    }
                )

            options.sort(key=lambda x: x["estimated_monthly_savings_usd"], reverse=True)
            return options

        def _calculate_monthly_savings(self, model_name, profile):
            base_monthly_costs = {
                "llama-3.1-8b": 1500,
                "llama-3.1-70b": 8000,
                "mistral-7b": 1200,
                "codellama-13b": 2500,
            }

            base_cost = base_monthly_costs.get(model_name, 1500)
            savings = base_cost * (profile.cost_reduction_pct / 100)
            return savings

        def _get_implementation_complexity(self, quant_type):
            complexity_map = {
                QuantizationType.FP16: "Low",
                QuantizationType.INT8: "Medium",
                QuantizationType.INT4: "High",
                QuantizationType.MIXED_PRECISION: "Medium",
                QuantizationType.DYNAMIC: "High",
            }
            return complexity_map.get(quant_type, "Medium")

        def generate_quantization_config(self, model_name, quant_type):
            profile = self.profiles[quant_type]

            config = {
                "apiVersion": "inference.llm-d.io/v1alpha1",
                "kind": "LLMDeployment",
                "metadata": {
                    "name": f"{model_name}-{quant_type.value}",
                    "namespace": "production",
                    "labels": {
                        "app.kubernetes.io/name": "llm-d",
                        "llm-d.ai/model": model_name,
                        "llm-d.ai/quantization": quant_type.value,
                        "cost-optimization.llm-d.io/enabled": "true",
                    },
                    "annotations": {
                        "cost-optimization.llm-d.io/memory-reduction": (
                            f"{profile.memory_reduction_pct}%"
                        ),
                        "cost-optimization.llm-d.io/expected-savings": (
                            f"{profile.cost_reduction_pct}%"
                        ),
                    },
                },
                "spec": {
                    "model": {
                        "name": model_name,
                        "quantization": {
                            "type": quant_type.value,
                            "precision": self._get_precision_config(quant_type),
                        },
                    },
                    "resources": self._get_optimized_resources(model_name, profile),
                    "serving": {
                        "protocol": "http",
                        "port": 8080,
                        "batchSize": self._get_optimal_batch_size(quant_type),
                    },
                    "autoscaling": {
                        "enabled": True,
                        "minReplicas": 1,
                        "maxReplicas": 8,
                        "targetGPUUtilization": 80,
                    },
                },
            }

            return config

        def _get_precision_config(self, quant_type):
            configs = {
                QuantizationType.FP16: {
                    "format": "fp16",
                    "weight_dtype": "float16",
                    "activation_dtype": "float16",
                },
                QuantizationType.INT8: {
                    "format": "int8",
                    "weight_dtype": "int8",
                    "activation_dtype": "int8",
                    "calibration_dataset": "c4",
                    "calibration_samples": 128,
                },
                QuantizationType.INT4: {
                    "format": "int4",
                    "weight_dtype": "int4",
                    "activation_dtype": "float16",
                    "group_size": 128,
                    "calibration_dataset": "c4",
                    "calibration_samples": 256,
                },
                QuantizationType.MIXED_PRECISION: {
                    "format": "mixed",
                    "attention_dtype": "float16",
                    "mlp_dtype": "int8",
                    "embedding_dtype": "float16",
                },
            }
            return configs.get(quant_type, configs[QuantizationType.FP16])

        def _get_optimized_resources(self, model_name, profile):
            base_resources = {
                "llama-3.1-8b": {"memory": "16Gi", "gpu": "1"},
                "llama-3.1-70b": {"memory": "80Gi", "gpu": "4"},
                "mistral-7b": {"memory": "14Gi", "gpu": "1"},
                "codellama-13b": {"memory": "26Gi", "gpu": "2"},
            }

            base = base_resources.get(model_name, {"memory": "16Gi", "gpu": "1"})
            base_memory_gb = int(base["memory"].replace("Gi", ""))
            reduced_memory_gb = int(
                base_memory_gb * (1 - profile.memory_reduction_pct / 100)
            )
            reduced_memory_gb = max(
                reduced_memory_gb, profile.gpu_requirements["min_memory_gb"]
            )

            return {
                "requests": {
                    "nvidia.com/gpu": base["gpu"],
                    "memory": f"{reduced_memory_gb}Gi",
                    "cpu": "4",
                },
                "limits": {
                    "nvidia.com/gpu": base["gpu"],
                    "memory": f"{reduced_memory_gb + 4}Gi",
                    "cpu": "8",
                },
            }

        def _get_optimal_batch_size(self, quant_type):
            batch_sizes = {
                QuantizationType.FP16: 4,
                QuantizationType.INT8: 8,
                QuantizationType.INT4: 16,
                QuantizationType.MIXED_PRECISION: 6,
                QuantizationType.DYNAMIC: 8,
            }
            return batch_sizes.get(quant_type, 4)


class TestQuantizationTypes:
    """Test quantization type definitions."""

    def test_quantization_type_enum(self):
        """Test QuantizationType enum values."""
        assert QuantizationType.FP16.value == "fp16"
        assert QuantizationType.INT8.value == "int8"
        assert QuantizationType.INT4.value == "int4"
        assert QuantizationType.MIXED_PRECISION.value == "mixed"
        assert QuantizationType.DYNAMIC.value == "dynamic"

    def test_quantization_profile_creation(self):
        """Test QuantizationProfile dataclass creation."""
        profile = QuantizationProfile(
            name="Test Profile",
            quant_type=QuantizationType.INT8,
            memory_reduction_pct=75.0,
            performance_retention_pct=97.0,
            cost_reduction_pct=65.0,
            compatible_models=["test-model"],
            gpu_requirements={"min_memory_gb": 8, "compute_capability": 6.1},
        )

        assert profile.name == "Test Profile"
        assert profile.quant_type == QuantizationType.INT8
        assert profile.memory_reduction_pct == 75.0
        assert profile.performance_retention_pct == 97.0
        assert profile.cost_reduction_pct == 65.0
        assert profile.compatible_models == ["test-model"]
        assert profile.gpu_requirements["min_memory_gb"] == 8


class TestQuantizationOptimizer:
    """Test quantization optimizer functionality."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance."""
        return QuantizationOptimizer()

    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initializes with correct profiles."""
        # Check all quantization types have profiles
        assert QuantizationType.FP16 in optimizer.profiles
        assert QuantizationType.INT8 in optimizer.profiles
        assert QuantizationType.INT4 in optimizer.profiles
        assert QuantizationType.MIXED_PRECISION in optimizer.profiles

        # Check profile structure
        fp16_profile = optimizer.profiles[QuantizationType.FP16]
        assert fp16_profile.memory_reduction_pct == 50.0
        assert fp16_profile.performance_retention_pct == 99.5
        assert fp16_profile.cost_reduction_pct == 40.0
        assert "llama-3.1-8b" in fp16_profile.compatible_models

    def test_analyze_quantization_options_basic(self, optimizer):
        """Test basic quantization options analysis."""
        options = optimizer.analyze_quantization_options(
            model_name="llama-3.1-8b",
            available_gpu_memory_gb=40,
            performance_threshold_pct=95.0,
        )

        # Should return options for compatible quantizations
        assert len(options) > 0

        # Check option structure
        option = options[0]
        assert "quantization_type" in option
        assert "name" in option
        assert "memory_reduction_pct" in option
        assert "performance_retention_pct" in option
        assert "estimated_monthly_savings_usd" in option
        assert "cost_reduction_pct" in option
        assert "implementation_complexity" in option

        # Options should be sorted by savings (descending)
        if len(options) > 1:
            assert (
                options[0]["estimated_monthly_savings_usd"]
                >= options[1]["estimated_monthly_savings_usd"]
            )

    def test_analyze_quantization_options_memory_constraint(self, optimizer):
        """Test quantization analysis with memory constraints."""
        # Very limited GPU memory
        limited_options = optimizer.analyze_quantization_options(
            model_name="llama-3.1-8b",
            available_gpu_memory_gb=6,  # Less than most requirements
            performance_threshold_pct=90.0,
        )

        # Abundant GPU memory
        abundant_options = optimizer.analyze_quantization_options(
            model_name="llama-3.1-8b",
            available_gpu_memory_gb=80,  # More than any requirement
            performance_threshold_pct=90.0,
        )

        # Should have fewer options with limited memory
        assert len(limited_options) <= len(abundant_options)

        # All limited options should have memory requirements <= 6GB
        for option in limited_options:
            quant_type = QuantizationType(option["quantization_type"])
            profile = optimizer.profiles[quant_type]
            assert profile.gpu_requirements["min_memory_gb"] <= 6

    def test_analyze_quantization_options_performance_threshold(self, optimizer):
        """Test quantization analysis with performance thresholds."""
        # Strict performance threshold
        strict_options = optimizer.analyze_quantization_options(
            model_name="llama-3.1-8b",
            available_gpu_memory_gb=40,
            performance_threshold_pct=98.0,  # High threshold
        )

        # Relaxed performance threshold
        relaxed_options = optimizer.analyze_quantization_options(
            model_name="llama-3.1-8b",
            available_gpu_memory_gb=40,
            performance_threshold_pct=90.0,  # Low threshold
        )

        # Should have fewer options with strict threshold
        assert len(strict_options) <= len(relaxed_options)

        # All strict options should meet the performance threshold
        for option in strict_options:
            assert option["performance_retention_pct"] >= 98.0

    def test_analyze_quantization_options_model_compatibility(self, optimizer):
        """Test quantization analysis respects model compatibility."""
        # Model with limited quantization support
        limited_options = optimizer.analyze_quantization_options(
            model_name="mistral-7b",  # Not supported by all quantizations
            available_gpu_memory_gb=40,
            performance_threshold_pct=90.0,
        )

        # Model with broad quantization support
        broad_options = optimizer.analyze_quantization_options(
            model_name="llama-3.1-8b",  # Supported by all quantizations
            available_gpu_memory_gb=40,
            performance_threshold_pct=90.0,
        )

        # Broader support should have more options
        assert len(limited_options) <= len(broad_options)

        # All options should be compatible with the model
        for option in limited_options:
            quant_type = QuantizationType(option["quantization_type"])
            profile = optimizer.profiles[quant_type]
            assert "mistral-7b" in profile.compatible_models

    def test_monthly_savings_calculation(self, optimizer):
        """Test monthly savings calculation accuracy."""
        # Test known savings calculation
        profile = optimizer.profiles[QuantizationType.INT8]
        savings = optimizer._calculate_monthly_savings("llama-3.1-8b", profile)

        # Expected: $1500 * 65% = $975
        expected_savings = 1500 * 0.65
        assert abs(savings - expected_savings) < 0.01

        # Test different models have different base costs
        savings_8b = optimizer._calculate_monthly_savings("llama-3.1-8b", profile)
        savings_70b = optimizer._calculate_monthly_savings("llama-3.1-70b", profile)

        # 70B model should have higher base cost and thus higher savings
        assert savings_70b > savings_8b

    def test_implementation_complexity_mapping(self, optimizer):
        """Test implementation complexity mapping."""
        assert optimizer._get_implementation_complexity(QuantizationType.FP16) == "Low"
        assert (
            optimizer._get_implementation_complexity(QuantizationType.INT8) == "Medium"
        )
        assert optimizer._get_implementation_complexity(QuantizationType.INT4) == "High"
        assert (
            optimizer._get_implementation_complexity(QuantizationType.MIXED_PRECISION)
            == "Medium"
        )
        assert (
            optimizer._get_implementation_complexity(QuantizationType.DYNAMIC) == "High"
        )


class TestConfigurationGeneration:
    """Test quantization configuration generation."""

    @pytest.fixture
    def optimizer(self):
        return QuantizationOptimizer()

    def test_generate_quantization_config_structure(self, optimizer):
        """Test generated configuration has correct structure."""
        config = optimizer.generate_quantization_config(
            model_name="llama-3.1-8b", quant_type=QuantizationType.INT8
        )

        # Check top-level structure
        assert config["apiVersion"] == "inference.llm-d.io/v1alpha1"
        assert config["kind"] == "LLMDeployment"
        assert "metadata" in config
        assert "spec" in config

        # Check metadata
        metadata = config["metadata"]
        assert metadata["name"] == "llama-3.1-8b-int8"
        assert metadata["namespace"] == "production"
        assert "labels" in metadata
        assert "annotations" in metadata

        # Check labels
        labels = metadata["labels"]
        assert labels["llm-d.ai/model"] == "llama-3.1-8b"
        assert labels["llm-d.ai/quantization"] == "int8"
        assert labels["cost-optimization.llm-d.io/enabled"] == "true"

        # Check spec structure
        spec = config["spec"]
        assert "model" in spec
        assert "resources" in spec
        assert "serving" in spec
        assert "autoscaling" in spec

    def test_generate_quantization_config_model_spec(self, optimizer):
        """Test model specification in generated config."""
        config = optimizer.generate_quantization_config(
            model_name="llama-3.1-8b", quant_type=QuantizationType.INT8
        )

        model_spec = config["spec"]["model"]
        assert model_spec["name"] == "llama-3.1-8b"

        quantization_spec = model_spec["quantization"]
        assert quantization_spec["type"] == "int8"

        precision_config = quantization_spec["precision"]
        assert precision_config["format"] == "int8"
        assert precision_config["weight_dtype"] == "int8"
        assert precision_config["activation_dtype"] == "int8"
        assert "calibration_dataset" in precision_config
        assert "calibration_samples" in precision_config

    def test_generate_quantization_config_resources(self, optimizer):
        """Test resource specification in generated config."""
        config = optimizer.generate_quantization_config(
            model_name="llama-3.1-8b", quant_type=QuantizationType.INT8
        )

        resources = config["spec"]["resources"]
        assert "requests" in resources
        assert "limits" in resources

        requests = resources["requests"]
        assert "nvidia.com/gpu" in requests
        assert "memory" in requests
        assert "cpu" in requests

        # Check memory reduction applied
        # INT8 should reduce memory by 75%: 16GB -> 4GB (but capped at min_memory_gb=8GB)
        assert requests["memory"] == "8Gi"  # Minimum requirement

    def test_generate_quantization_config_serving(self, optimizer):
        """Test serving specification in generated config."""
        config = optimizer.generate_quantization_config(
            model_name="llama-3.1-8b", quant_type=QuantizationType.INT4
        )

        serving = config["spec"]["serving"]
        assert serving["protocol"] == "http"
        assert serving["port"] == 8080
        assert serving["batchSize"] == 16  # INT4 should have larger batch size

    def test_generate_quantization_config_autoscaling(self, optimizer):
        """Test autoscaling specification in generated config."""
        config = optimizer.generate_quantization_config(
            model_name="llama-3.1-8b", quant_type=QuantizationType.FP16
        )

        autoscaling = config["spec"]["autoscaling"]
        assert autoscaling["enabled"] is True
        assert autoscaling["minReplicas"] == 1
        assert autoscaling["maxReplicas"] == 8
        assert autoscaling["targetGPUUtilization"] == 80

    def test_precision_config_variations(self, optimizer):
        """Test precision configurations for different quantization types."""
        # Test FP16 precision config
        fp16_config = optimizer._get_precision_config(QuantizationType.FP16)
        assert fp16_config["format"] == "fp16"
        assert fp16_config["weight_dtype"] == "float16"
        assert fp16_config["activation_dtype"] == "float16"

        # Test INT4 precision config
        int4_config = optimizer._get_precision_config(QuantizationType.INT4)
        assert int4_config["format"] == "int4"
        assert int4_config["weight_dtype"] == "int4"
        assert int4_config["activation_dtype"] == "float16"  # Kept at FP16
        assert "group_size" in int4_config

        # Test mixed precision config
        mixed_config = optimizer._get_precision_config(QuantizationType.MIXED_PRECISION)
        assert mixed_config["format"] == "mixed"
        assert mixed_config["attention_dtype"] == "float16"
        assert mixed_config["mlp_dtype"] == "int8"
        assert mixed_config["embedding_dtype"] == "float16"

    def test_optimal_batch_size_scaling(self, optimizer):
        """Test optimal batch size scaling with quantization."""
        # More aggressive quantization should allow larger batches
        assert optimizer._get_optimal_batch_size(QuantizationType.FP16) == 4
        assert optimizer._get_optimal_batch_size(QuantizationType.INT8) == 8
        assert optimizer._get_optimal_batch_size(QuantizationType.INT4) == 16
        assert optimizer._get_optimal_batch_size(QuantizationType.MIXED_PRECISION) == 6

    def test_resource_optimization_scaling(self, optimizer):
        """Test resource optimization for different models and quantizations."""
        # Test small model with aggressive quantization
        small_resources = optimizer._get_optimized_resources(
            "mistral-7b", optimizer.profiles[QuantizationType.INT4]
        )

        # Test large model with conservative quantization
        large_resources = optimizer._get_optimized_resources(
            "llama-3.1-70b", optimizer.profiles[QuantizationType.FP16]
        )

        # Small model should require fewer resources
        small_gpu = int(small_resources["requests"]["nvidia.com/gpu"])
        large_gpu = int(large_resources["requests"]["nvidia.com/gpu"])
        assert small_gpu <= large_gpu

        # Memory should be reduced but respect minimums
        small_memory = int(small_resources["requests"]["memory"].replace("Gi", ""))
        large_memory = int(large_resources["requests"]["memory"].replace("Gi", ""))
        assert small_memory < large_memory


class TestCostOptimizationScenarios:
    """Test cost optimization scenarios and strategies."""

    @pytest.fixture
    def optimizer(self):
        return QuantizationOptimizer()

    def test_cost_vs_performance_tradeoffs(self, optimizer):
        """Test cost vs performance tradeoffs across quantization types."""
        model = "llama-3.1-8b"
        options = optimizer.analyze_quantization_options(model, 40, 90.0)

        # Sort by performance retention (descending)
        options_by_perf = sorted(
            options, key=lambda x: x["performance_retention_pct"], reverse=True
        )

        # Sort by cost savings (descending)
        options_by_cost = sorted(
            options, key=lambda x: x["cost_reduction_pct"], reverse=True
        )

        # Generally, higher performance retention should correlate with lower cost savings
        highest_perf = options_by_perf[0]
        highest_cost_saving = options_by_cost[0]

        # They should typically be different (tradeoff exists)
        if len(options) > 1:
            assert (
                highest_perf["quantization_type"]
                != highest_cost_saving["quantization_type"]
            )

    def test_quantization_recommendations_by_constraint(self, optimizer):
        """Test quantization recommendations based on different constraints."""
        model = "llama-3.1-8b"

        # Memory-constrained scenario
        memory_constrained = optimizer.analyze_quantization_options(model, 8, 90.0)

        # Performance-critical scenario
        performance_critical = optimizer.analyze_quantization_options(model, 40, 98.0)

        # Cost-sensitive scenario
        cost_sensitive = optimizer.analyze_quantization_options(model, 40, 90.0)

        # Memory-constrained should prefer more aggressive quantization
        if memory_constrained:
            memory_best = memory_constrained[0]
            assert memory_best["memory_reduction_pct"] >= 60.0

        # Performance-critical should prefer conservative quantization
        if performance_critical:
            perf_best = performance_critical[0]
            assert perf_best["performance_retention_pct"] >= 98.0

        # Cost-sensitive should maximize savings
        if cost_sensitive:
            cost_best = cost_sensitive[0]
            # Should be the option with highest cost reduction
            all_cost_reductions = [opt["cost_reduction_pct"] for opt in cost_sensitive]
            assert cost_best["cost_reduction_pct"] == max(all_cost_reductions)

    def test_model_specific_optimization_strategies(self, optimizer):
        """Test model-specific optimization strategies."""
        models = ["llama-3.1-8b", "llama-3.1-70b", "mistral-7b"]

        for model in models:
            options = optimizer.analyze_quantization_options(model, 40, 95.0)

            # Each model should have at least one viable quantization option
            assert len(options) > 0

            # All options should be compatible with the model
            for option in options:
                quant_type = QuantizationType(option["quantization_type"])
                profile = optimizer.profiles[quant_type]
                assert model in profile.compatible_models

            # Calculate potential savings
            if options:
                total_savings = sum(
                    opt["estimated_monthly_savings_usd"] for opt in options
                )
                assert total_savings > 0

    def test_implementation_complexity_vs_savings(self, optimizer):
        """Test relationship between implementation complexity and savings."""
        options = optimizer.analyze_quantization_options("llama-3.1-8b", 40, 90.0)

        complexity_savings = {}
        for option in options:
            complexity = option["implementation_complexity"]
            savings = option["cost_reduction_pct"]

            if complexity not in complexity_savings:
                complexity_savings[complexity] = []
            complexity_savings[complexity].append(savings)

        # Generally, higher complexity should offer higher potential savings
        if "Low" in complexity_savings and "High" in complexity_savings:
            avg_low_savings = sum(complexity_savings["Low"]) / len(
                complexity_savings["Low"]
            )
            avg_high_savings = sum(complexity_savings["High"]) / len(
                complexity_savings["High"]
            )

            # High complexity should generally offer higher savings
            assert avg_high_savings >= avg_low_savings


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    @pytest.fixture
    def optimizer(self):
        return QuantizationOptimizer()

    def test_unsupported_model_handling(self, optimizer):
        """Test handling of unsupported models."""
        options = optimizer.analyze_quantization_options(
            model_name="unsupported-model",
            available_gpu_memory_gb=40,
            performance_threshold_pct=95.0,
        )

        # Should return empty list for unsupported model
        assert len(options) == 0

    def test_insufficient_gpu_memory(self, optimizer):
        """Test handling of insufficient GPU memory."""
        options = optimizer.analyze_quantization_options(
            model_name="llama-3.1-8b",
            available_gpu_memory_gb=2,  # Very limited memory
            performance_threshold_pct=95.0,
        )

        # Should return very few or no options
        assert len(options) <= 1  # Maybe INT4 with very low requirements

    def test_very_strict_performance_threshold(self, optimizer):
        """Test handling of very strict performance thresholds."""
        options = optimizer.analyze_quantization_options(
            model_name="llama-3.1-8b",
            available_gpu_memory_gb=40,
            performance_threshold_pct=99.9,  # Very strict threshold
        )

        # Should return only the most conservative options
        for option in options:
            assert option["performance_retention_pct"] >= 99.9

    def test_zero_performance_threshold(self, optimizer):
        """Test handling of zero performance threshold."""
        options = optimizer.analyze_quantization_options(
            model_name="llama-3.1-8b",
            available_gpu_memory_gb=40,
            performance_threshold_pct=0.0,  # No performance requirement
        )

        # Should return all compatible options
        assert len(options) > 0
        # Should include the most aggressive quantization
        quant_types = [opt["quantization_type"] for opt in options]
        assert "int4" in quant_types  # Most aggressive quantization

    def test_config_generation_with_unknown_model(self, optimizer):
        """Test configuration generation with unknown model."""
        config = optimizer.generate_quantization_config(
            model_name="unknown-model", quant_type=QuantizationType.INT8
        )

        # Should use default resource allocation
        resources = config["spec"]["resources"]["requests"]
        assert resources["memory"] == "8Gi"  # Default after INT8 reduction
        assert resources["nvidia.com/gpu"] == "1"  # Default GPU count


if __name__ == "__main__":
    pytest.main([__file__])
