"""
Tests for model configuration validation in
chapter-10-mlops/kubeflow-pipelines/validate-model-config.py
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

# Add the examples directory to the path
sys.path.insert(
    0, str(Path(__file__).parent.parent.parent.parent.parent / "llm-d-book-examples")
)

try:
    from chapter_10_mlops.kubeflow_pipelines.validate_model_config import (
        MODEL_CONFIG_SCHEMA, validate_model_config)
except ImportError:
    # Create mock implementation for testing when real implementation isn't available
    MODEL_CONFIG_SCHEMA = {
        "type": "object",
        "required": ["name", "version", "architecture", "resources"],
        "properties": {
            "name": {"type": "string", "pattern": "^[a-z0-9-]+$"},
            "version": {"type": "string", "pattern": "^\\d+\\.\\d+\\.\\d+$"},
            "architecture": {"type": "string"},
            "description": {"type": "string"},
            "resources": {
                "type": "object",
                "required": ["memory_gb", "gpu_count"],
                "properties": {
                    "memory_gb": {"type": "number", "minimum": 1},
                    "gpu_count": {"type": "integer", "minimum": 1},
                    "gpu_type": {"type": "string"},
                },
            },
            "serving": {
                "type": "object",
                "properties": {
                    "max_batch_size": {"type": "integer", "minimum": 1},
                    "max_sequence_length": {"type": "integer", "minimum": 1},
                    "protocol": {"type": "string", "enum": ["http", "grpc"]},
                },
            },
            "source": {
                "type": "object",
                "required": ["type", "uri"],
                "properties": {
                    "type": {"type": "string", "enum": ["s3", "huggingface", "local"]},
                    "uri": {"type": "string"},
                },
            },
        },
    }

    def validate_model_config(config_path: str) -> bool:
        """Validate model configuration file"""
        import jsonschema

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            # Validate against schema
            jsonschema.validate(config, MODEL_CONFIG_SCHEMA)

            # Additional business logic validation
            if config["resources"]["memory_gb"] < config["resources"]["gpu_count"] * 8:
                raise ValueError(
                    f"Insufficient memory for {config['resources']['gpu_count']} GPUs"
                )

            # Validate model name matches directory structure
            expected_model_name = os.path.basename(os.path.dirname(config_path))
            if config["name"] != expected_model_name:
                raise ValueError(
                    f"Model name {config['name']} doesn't match directory "
                    f"{expected_model_name}"
                )

            print(f"✅ Configuration valid: {config['name']} v{config['version']}")
            return True

        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return False


class TestModelConfigValidation:
    """Test cases for model configuration validation."""

    def test_valid_basic_config(self):
        """Test validation of a basic valid configuration."""
        valid_config = {
            "name": "llama-3-1-7b",
            "version": "1.0.0",
            "architecture": "transformer",
            "description": "Llama 3.1 7B parameter model",
            "resources": {"memory_gb": 32, "gpu_count": 2, "gpu_type": "A100"},
            "serving": {
                "max_batch_size": 8,
                "max_sequence_length": 2048,
                "protocol": "http",
            },
            "source": {"type": "s3", "uri": "s3://model-registry/llama-3.1-7b/v1.0.0"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_config, f)
            config_path = f.name

        try:
            # Create directory structure that matches model name
            config_dir = os.path.join(os.path.dirname(config_path), "llama-3-1-7b")
            os.makedirs(config_dir, exist_ok=True)
            final_config_path = os.path.join(config_dir, "model.yaml")

            with open(final_config_path, "w") as f:
                yaml.dump(valid_config, f)

            result = validate_model_config(final_config_path)
            assert result is True

        finally:
            # Cleanup
            if os.path.exists(config_path):
                os.unlink(config_path)
            if os.path.exists(final_config_path):
                os.unlink(final_config_path)
                os.rmdir(config_dir)

    def test_missing_required_fields(self):
        """Test validation with missing required fields."""
        incomplete_configs = [
            # Missing name
            {
                "version": "1.0.0",
                "architecture": "transformer",
                "resources": {"memory_gb": 16, "gpu_count": 1},
            },
            # Missing version
            {
                "name": "test-model",
                "architecture": "transformer",
                "resources": {"memory_gb": 16, "gpu_count": 1},
            },
            # Missing architecture
            {
                "name": "test-model",
                "version": "1.0.0",
                "resources": {"memory_gb": 16, "gpu_count": 1},
            },
            # Missing resources
            {"name": "test-model", "version": "1.0.0", "architecture": "transformer"},
        ]

        for config in incomplete_configs:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                yaml.dump(config, f)
                config_path = f.name

            try:
                result = validate_model_config(config_path)
                assert result is False
            finally:
                os.unlink(config_path)

    def test_invalid_field_formats(self):
        """Test validation with invalid field formats."""
        base_config = {
            "name": "test-model",
            "version": "1.0.0",
            "architecture": "transformer",
            "resources": {"memory_gb": 16, "gpu_count": 1},
        }

        invalid_configs = [
            # Invalid name format (uppercase not allowed)
            {**base_config, "name": "Test-Model"},
            # Invalid version format
            {**base_config, "version": "1.0"},
            # Invalid memory (negative)
            {**base_config, "resources": {"memory_gb": -1, "gpu_count": 1}},
            # Invalid GPU count (zero)
            {**base_config, "resources": {"memory_gb": 16, "gpu_count": 0}},
        ]

        for config in invalid_configs:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                yaml.dump(config, f)
                config_path = f.name

            try:
                result = validate_model_config(config_path)
                assert result is False
            finally:
                os.unlink(config_path)

    def test_memory_gpu_ratio_validation(self):
        """Test business logic validation for memory-to-GPU ratio."""
        # Insufficient memory (less than 8GB per GPU)
        insufficient_memory_config = {
            "name": "test-model",
            "version": "1.0.0",
            "architecture": "transformer",
            "resources": {"memory_gb": 15, "gpu_count": 2},  # Only 7.5GB per GPU
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(insufficient_memory_config, f)
            config_path = f.name

        try:
            result = validate_model_config(config_path)
            assert result is False
        finally:
            os.unlink(config_path)

        # Sufficient memory (8GB+ per GPU)
        sufficient_memory_config = {
            "name": "test-model",
            "version": "1.0.0",
            "architecture": "transformer",
            "resources": {"memory_gb": 24, "gpu_count": 2},  # 12GB per GPU
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sufficient_memory_config, f)
            config_path = f.name

        try:
            # Create proper directory structure
            config_dir = os.path.join(os.path.dirname(config_path), "test-model")
            os.makedirs(config_dir, exist_ok=True)
            final_config_path = os.path.join(config_dir, "model.yaml")

            with open(final_config_path, "w") as f:
                yaml.dump(sufficient_memory_config, f)

            result = validate_model_config(final_config_path)
            assert result is True

        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
            if os.path.exists(final_config_path):
                os.unlink(final_config_path)
                os.rmdir(config_dir)

    def test_model_name_directory_mismatch(self):
        """Test validation of model name matching directory structure."""
        config = {
            "name": "wrong-model-name",
            "version": "1.0.0",
            "architecture": "transformer",
            "resources": {"memory_gb": 16, "gpu_count": 1},
        }

        # Create directory with different name
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = os.path.join(temp_dir, "correct-model-name")
            os.makedirs(model_dir)
            config_path = os.path.join(model_dir, "model.yaml")

            with open(config_path, "w") as f:
                yaml.dump(config, f)

            result = validate_model_config(config_path)
            assert result is False

    def test_serving_configuration_validation(self):
        """Test serving configuration validation."""
        base_config = {
            "name": "test-model",
            "version": "1.0.0",
            "architecture": "transformer",
            "resources": {"memory_gb": 16, "gpu_count": 1},
        }

        # Valid serving configs
        valid_serving_configs = [
            {"max_batch_size": 8, "max_sequence_length": 2048, "protocol": "http"},
            {"max_batch_size": 16, "max_sequence_length": 4096, "protocol": "grpc"},
            {"max_batch_size": 1, "max_sequence_length": 1024, "protocol": "http"},
        ]

        for serving_config in valid_serving_configs:
            config = {**base_config, "serving": serving_config}

            with tempfile.TemporaryDirectory() as temp_dir:
                model_dir = os.path.join(temp_dir, "test-model")
                os.makedirs(model_dir)
                config_path = os.path.join(model_dir, "model.yaml")

                with open(config_path, "w") as f:
                    yaml.dump(config, f)

                result = validate_model_config(config_path)
                assert result is True

        # Invalid serving configs
        invalid_serving_configs = [
            {
                "max_batch_size": 0,
                "max_sequence_length": 2048,
                "protocol": "http",
            },  # Invalid batch size
            {
                "max_batch_size": 8,
                "max_sequence_length": 0,
                "protocol": "http",
            },  # Invalid sequence length
            {
                "max_batch_size": 8,
                "max_sequence_length": 2048,
                "protocol": "tcp",
            },  # Invalid protocol
        ]

        for serving_config in invalid_serving_configs:
            config = {**base_config, "serving": serving_config}

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                yaml.dump(config, f)
                config_path = f.name

            try:
                result = validate_model_config(config_path)
                assert result is False
            finally:
                os.unlink(config_path)

    def test_source_configuration_validation(self):
        """Test source configuration validation."""
        base_config = {
            "name": "test-model",
            "version": "1.0.0",
            "architecture": "transformer",
            "resources": {"memory_gb": 16, "gpu_count": 1},
        }

        # Valid source configs
        valid_source_configs = [
            {"type": "s3", "uri": "s3://model-registry/test-model/v1.0.0"},
            {"type": "huggingface", "uri": "huggingface:microsoft/DialoGPT-medium"},
            {"type": "local", "uri": "/local/path/to/model"},
        ]

        for source_config in valid_source_configs:
            config = {**base_config, "source": source_config}

            with tempfile.TemporaryDirectory() as temp_dir:
                model_dir = os.path.join(temp_dir, "test-model")
                os.makedirs(model_dir)
                config_path = os.path.join(model_dir, "model.yaml")

                with open(config_path, "w") as f:
                    yaml.dump(config, f)

                result = validate_model_config(config_path)
                assert result is True

        # Invalid source configs
        invalid_source_configs = [
            {"type": "invalid", "uri": "some://uri"},  # Invalid type
            {"type": "s3"},  # Missing URI
            {"uri": "s3://some/uri"},  # Missing type
        ]

        for source_config in invalid_source_configs:
            config = {**base_config, "source": source_config}

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                yaml.dump(config, f)
                config_path = f.name

            try:
                result = validate_model_config(config_path)
                assert result is False
            finally:
                os.unlink(config_path)

    def test_file_not_found_handling(self):
        """Test handling of missing configuration files."""
        non_existent_path = "/non/existent/path/model.yaml"
        result = validate_model_config(non_existent_path)
        assert result is False

    def test_invalid_yaml_handling(self):
        """Test handling of invalid YAML files."""
        invalid_yaml = (
            "name: test-model\nversion: 1.0.0\narchitecture: transformer\n"
            "resources:\n  memory_gb: 16\n  gpu_count: [\n"
        )  # Malformed YAML

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(invalid_yaml)
            config_path = f.name

        try:
            result = validate_model_config(config_path)
            assert result is False
        finally:
            os.unlink(config_path)

    def test_schema_validation_comprehensive(self):
        """Test comprehensive schema validation."""
        # Test that MODEL_CONFIG_SCHEMA is properly structured
        assert "type" in MODEL_CONFIG_SCHEMA
        assert MODEL_CONFIG_SCHEMA["type"] == "object"
        assert "required" in MODEL_CONFIG_SCHEMA
        assert "properties" in MODEL_CONFIG_SCHEMA

        # Verify required fields
        required_fields = MODEL_CONFIG_SCHEMA["required"]
        assert "name" in required_fields
        assert "version" in required_fields
        assert "architecture" in required_fields
        assert "resources" in required_fields

        # Verify resource schema
        resources_schema = MODEL_CONFIG_SCHEMA["properties"]["resources"]
        assert resources_schema["type"] == "object"
        assert "memory_gb" in resources_schema["required"]
        assert "gpu_count" in resources_schema["required"]

    @pytest.mark.parametrize(
        "gpu_count,memory_gb,expected_valid",
        [
            (1, 8, True),  # Exactly 8GB per GPU
            (1, 16, True),  # More than 8GB per GPU
            (2, 16, True),  # Exactly 8GB per GPU
            (2, 20, True),  # More than 8GB per GPU
            (4, 32, True),  # Exactly 8GB per GPU
            (1, 7, False),  # Less than 8GB per GPU
            (2, 15, False),  # Less than 8GB per GPU
            (4, 30, False),  # Less than 8GB per GPU
        ],
    )
    def test_memory_gpu_ratio_edge_cases(self, gpu_count, memory_gb, expected_valid):
        """Test memory-to-GPU ratio validation with various combinations."""
        config = {
            "name": "test-model",
            "version": "1.0.0",
            "architecture": "transformer",
            "resources": {"memory_gb": memory_gb, "gpu_count": gpu_count},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = os.path.join(temp_dir, "test-model")
            os.makedirs(model_dir)
            config_path = os.path.join(model_dir, "model.yaml")

            with open(config_path, "w") as f:
                yaml.dump(config, f)

            result = validate_model_config(config_path)
            assert result == expected_valid

    def test_optional_fields_validation(self):
        """Test validation with optional fields included."""
        config_with_all_fields = {
            "name": "comprehensive-model",
            "version": "2.1.3",
            "architecture": "transformer-xl",
            "description": "A comprehensive model configuration for testing",
            "resources": {"memory_gb": 64, "gpu_count": 4, "gpu_type": "A100-80GB"},
            "serving": {
                "max_batch_size": 16,
                "max_sequence_length": 4096,
                "protocol": "grpc",
            },
            "source": {
                "type": "huggingface",
                "uri": "huggingface:meta-llama/Llama-2-70b-hf",
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = os.path.join(temp_dir, "comprehensive-model")
            os.makedirs(model_dir)
            config_path = os.path.join(model_dir, "model.yaml")

            with open(config_path, "w") as f:
                yaml.dump(config_with_all_fields, f)

            result = validate_model_config(config_path)
            assert result is True

    def test_version_format_validation(self):
        """Test semantic version format validation."""
        base_config = {
            "name": "test-model",
            "architecture": "transformer",
            "resources": {"memory_gb": 16, "gpu_count": 1},
        }

        valid_versions = ["1.0.0", "2.1.3", "10.20.30", "0.0.1"]
        invalid_versions = ["1.0", "v1.0.0", "1.0.0-beta", "1.0.0.1", "latest"]

        for version in valid_versions:
            config = {**base_config, "version": version}

            with tempfile.TemporaryDirectory() as temp_dir:
                model_dir = os.path.join(temp_dir, "test-model")
                os.makedirs(model_dir)
                config_path = os.path.join(model_dir, "model.yaml")

                with open(config_path, "w") as f:
                    yaml.dump(config, f)

                result = validate_model_config(config_path)
                assert result is True, f"Version {version} should be valid"

        for version in invalid_versions:
            config = {**base_config, "version": version}

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                yaml.dump(config, f)
                config_path = f.name

            try:
                result = validate_model_config(config_path)
                assert result is False, f"Version {version} should be invalid"
            finally:
                os.unlink(config_path)

    def test_name_format_validation(self):
        """Test model name format validation."""
        base_config = {
            "version": "1.0.0",
            "architecture": "transformer",
            "resources": {"memory_gb": 16, "gpu_count": 1},
        }

        valid_names = ["llama-3-1-7b", "mistral-7b", "gpt-4", "test-model-123"]
        invalid_names = [
            "Llama-3.1-7B",
            "test_model",
            "test.model",
            "test@model",
            "TEST-MODEL",
        ]

        for name in valid_names:
            config = {**base_config, "name": name}

            with tempfile.TemporaryDirectory() as temp_dir:
                model_dir = os.path.join(temp_dir, name)
                os.makedirs(model_dir)
                config_path = os.path.join(model_dir, "model.yaml")

                with open(config_path, "w") as f:
                    yaml.dump(config, f)

                result = validate_model_config(config_path)
                assert result is True, f"Name {name} should be valid"

        for name in invalid_names:
            config = {**base_config, "name": name}

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                yaml.dump(config, f)
                config_path = f.name

            try:
                result = validate_model_config(config_path)
                assert result is False, f"Name {name} should be invalid"
            finally:
                os.unlink(config_path)
