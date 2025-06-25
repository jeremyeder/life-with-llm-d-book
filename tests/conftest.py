"""
Global pytest configuration and fixtures for LLM-D book examples.

This module provides common test fixtures, mocks, and utilities used across
all test modules.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, Generator
# import asyncio  # Uncomment when needed


# Pytest plugins
# pytest_plugins = ["pytest_asyncio"]  # Uncomment when pytest-asyncio is installed


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_kubernetes_client():
    """Mock Kubernetes Python client."""
    with patch("kubernetes.client") as mock_client:
        # Mock common Kubernetes API objects
        mock_v1 = MagicMock()
        mock_apps_v1 = MagicMock()
        mock_custom_objects = MagicMock()
        
        mock_client.CoreV1Api.return_value = mock_v1
        mock_client.AppsV1Api.return_value = mock_apps_v1
        mock_client.CustomObjectsApi.return_value = mock_custom_objects
        
        # Mock common responses
        mock_v1.list_pod_for_all_namespaces.return_value.items = []
        mock_v1.list_node.return_value.items = []
        
        yield mock_client


@pytest.fixture
def mock_gpu_environment():
    """Mock GPU/CUDA environment variables and utilities."""
    gpu_env = {
        "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        "GPU_MEMORY_FRACTION": "0.9",
        "NVIDIA_VISIBLE_DEVICES": "all"
    }
    
    with patch.dict("os.environ", gpu_env):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=4):
                yield gpu_env


@pytest.fixture
def mock_llm_endpoint():
    """Mock LLM service endpoint responses."""
    with patch("requests.Session") as mock_session:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test-completion-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "llama3-8b",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Test response"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
        
        mock_session.return_value.post.return_value = mock_response
        mock_session.return_value.get.return_value = mock_response
        
        yield mock_session


@pytest.fixture
def sample_model_config() -> Dict[str, Any]:
    """Sample model configuration for testing."""
    return {
        "model_name": "llama-3.1-8b",
        "model_size_gb": 16,
        "context_length": 8192,
        "batch_size": 32,
        "dtype": "float16",
        "quantization": "int8",
        "tensor_parallel": 2,
        "pipeline_parallel": 1,
        "gpu_memory_fraction": 0.9,
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.9
    }


@pytest.fixture
def sample_metrics_data() -> Dict[str, Any]:
    """Sample metrics data for monitoring tests."""
    return {
        "timestamp": "2024-01-15T10:30:00Z",
        "gpu_utilization": {
            "gpu_0": 85.2,
            "gpu_1": 87.1,
            "gpu_2": 83.5,
            "gpu_3": 86.8
        },
        "memory_usage": {
            "gpu_0": {"used": 30720, "total": 40960},
            "gpu_1": {"used": 31232, "total": 40960},
            "gpu_2": {"used": 29696, "total": 40960},
            "gpu_3": {"used": 31744, "total": 40960}
        },
        "inference_metrics": {
            "requests_per_second": 125.4,
            "avg_latency_ms": 45.2,
            "p95_latency_ms": 89.3,
            "p99_latency_ms": 125.7,
            "tokens_per_second": 3762.5
        },
        "errors": {
            "cuda_oom": 0,
            "timeout": 2,
            "validation": 1
        }
    }


@pytest.fixture
def mock_prometheus_client():
    """Mock Prometheus client for metrics collection."""
    with patch("prometheus_client.Counter") as mock_counter:
        with patch("prometheus_client.Histogram") as mock_histogram:
            with patch("prometheus_client.Gauge") as mock_gauge:
                with patch("prometheus_client.Summary") as mock_summary:
                    yield {
                        "counter": mock_counter,
                        "histogram": mock_histogram,
                        "gauge": mock_gauge,
                        "summary": mock_summary
                    }


@pytest.fixture
def mock_mlflow_client():
    """Mock MLflow client for experiment tracking."""
    with patch("mlflow.start_run") as mock_start:
        with patch("mlflow.log_params") as mock_params:
            with patch("mlflow.log_metrics") as mock_metrics:
                with patch("mlflow.log_artifacts") as mock_artifacts:
                    yield {
                        "start_run": mock_start,
                        "log_params": mock_params,
                        "log_metrics": mock_metrics,
                        "log_artifacts": mock_artifacts
                    }


# Helper functions for tests
def assert_valid_gpu_config(config: Dict[str, Any]) -> None:
    """Assert that GPU configuration is valid."""
    assert "gpu_type" in config
    assert config.get("gpu_memory", 0) > 0
    assert 0 < config.get("gpu_memory_fraction", 0.9) <= 1.0
    

def assert_valid_model_response(response: Dict[str, Any]) -> None:
    """Assert that model response has required fields."""
    assert "id" in response
    assert "choices" in response
    assert len(response["choices"]) > 0
    assert "usage" in response
    

def create_mock_deployment_config(name: str = "test-deployment") -> Dict[str, Any]:
    """Create a mock Kubernetes deployment configuration."""
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": name,
            "namespace": "llm-d-models"
        },
        "spec": {
            "replicas": 2,
            "selector": {
                "matchLabels": {
                    "app": name
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": name
                    }
                },
                "spec": {
                    "containers": [{
                        "name": "model-server",
                        "image": "vllm/vllm-openai:latest",
                        "resources": {
                            "limits": {
                                "nvidia.com/gpu": "2",
                                "memory": "64Gi",
                                "cpu": "16"
                            }
                        }
                    }]
                }
            }
        }
    }


# Async test helpers (uncomment when asyncio is needed)
# @pytest.fixture
# def event_loop():
#     """Create an instance of the default event loop for the test session."""
#     import asyncio
#     loop = asyncio.get_event_loop_policy().new_event_loop()
#     yield loop
#     loop.close()