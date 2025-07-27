"""
Test helper utilities for LLM-D book examples.

Provides common testing utilities, mock data generators, and validation helpers.
"""

import json
import random
import string
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


class MockDataGenerator:
    """Generate mock data for testing."""

    @staticmethod
    def generate_gpu_metrics(num_gpus: int = 4) -> Dict[str, Any]:
        """Generate realistic GPU metrics data."""
        metrics = {}
        for i in range(num_gpus):
            metrics[f"gpu_{i}"] = {
                "utilization": random.uniform(70, 95),
                "memory_used_mb": random.randint(20000, 35000),
                "memory_total_mb": 40960,
                "temperature_c": random.randint(65, 85),
                "power_draw_w": random.randint(250, 350),
                "fan_speed_percent": random.randint(40, 80),
            }
        return metrics

    @staticmethod
    def generate_model_inference_request() -> Dict[str, Any]:
        """Generate a sample model inference request."""
        return {
            "model": "llama-3.1-8b",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is machine learning?"},
            ],
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": False,
            "request_id": f"req-{''.join(random.choices(string.ascii_lowercase, k=8))}",
        }

    @staticmethod
    def generate_training_metrics(num_steps: int = 100) -> List[Dict[str, Any]]:
        """Generate mock training metrics over time."""
        metrics = []
        base_loss = 2.5

        for step in range(num_steps):
            # Simulate decreasing loss with some noise
            loss = base_loss * (0.99**step) + random.uniform(-0.05, 0.05)
            metrics.append(
                {
                    "step": step,
                    "loss": max(0.1, loss),
                    "learning_rate": 1e-4 * (0.95 ** (step // 20)),
                    "gradient_norm": random.uniform(0.5, 2.0),
                    "tokens_per_second": random.uniform(3000, 4000),
                    "timestamp": (datetime.now() + timedelta(minutes=step)).isoformat(),
                }
            )

        return metrics

    @staticmethod
    def generate_cost_data(hours: int = 24) -> Dict[str, Any]:
        """Generate mock cost tracking data."""
        hourly_costs = []
        base_cost = 50.0  # Base hourly cost

        for hour in range(hours):
            # Simulate varying load throughout the day
            load_factor = 1.0 + 0.3 * random.random()
            if 9 <= hour <= 17:  # Peak hours
                load_factor *= 1.5

            hourly_costs.append(
                {
                    "hour": hour,
                    "gpu_cost": base_cost * load_factor,
                    "network_cost": base_cost * 0.1 * load_factor,
                    "storage_cost": base_cost * 0.05,
                    "total_cost": base_cost * 1.15 * load_factor,
                }
            )

        return {
            "period": "24h",
            "total_cost": sum(h["total_cost"] for h in hourly_costs),
            "hourly_breakdown": hourly_costs,
            "cost_per_request": random.uniform(0.001, 0.005),
            "requests_processed": random.randint(100000, 500000),
        }


class KubernetesResourceMocker:
    """Mock Kubernetes resources for testing."""

    @staticmethod
    def create_pod(
        name: str, namespace: str = "default", gpu_count: int = 1
    ) -> Dict[str, Any]:
        """Create a mock Kubernetes pod resource."""
        return {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": name,
                "namespace": namespace,
                "uid": f"uid-{name}",
                "creationTimestamp": datetime.now().isoformat(),
            },
            "spec": {
                "containers": [
                    {
                        "name": "model-server",
                        "image": "vllm/vllm-openai:latest",
                        "resources": {
                            "limits": {
                                "nvidia.com/gpu": str(gpu_count),
                                "memory": "32Gi",
                                "cpu": "8",
                            },
                            "requests": {
                                "nvidia.com/gpu": str(gpu_count),
                                "memory": "30Gi",
                                "cpu": "6",
                            },
                        },
                    }
                ]
            },
            "status": {
                "phase": "Running",
                "conditions": [
                    {
                        "type": "Ready",
                        "status": "True",
                        "lastTransitionTime": datetime.now().isoformat(),
                    }
                ],
                "containerStatuses": [
                    {
                        "ready": True,
                        "restartCount": 0,
                        "state": {"running": {"startedAt": datetime.now().isoformat()}},
                    }
                ],
            },
        }

    @staticmethod
    def create_service(name: str, namespace: str = "default") -> Dict[str, Any]:
        """Create a mock Kubernetes service resource."""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": name, "namespace": namespace},
            "spec": {
                "selector": {"app": name},
                "ports": [{"name": "http", "port": 8080, "targetPort": 8080}],
                "type": "ClusterIP",
            },
        }

    @staticmethod
    def create_configmap(name: str, data: Dict[str, str]) -> Dict[str, Any]:
        """Create a mock ConfigMap resource."""
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": name},
            "data": data,
        }


class ResponseValidator:
    """Validate API responses and data structures."""

    @staticmethod
    def validate_inference_response(response: Dict[str, Any]) -> bool:
        """Validate LLM inference response structure."""
        required_fields = ["id", "object", "created", "model", "choices", "usage"]

        if not all(field in response for field in required_fields):
            return False

        if not response["choices"] or len(response["choices"]) == 0:
            return False

        choice = response["choices"][0]
        if "message" not in choice or "content" not in choice["message"]:
            return False

        usage = response["usage"]
        if not all(
            key in usage
            for key in ["prompt_tokens", "completion_tokens", "total_tokens"]
        ):
            return False

        return True

    @staticmethod
    def validate_metrics(metrics: Dict[str, Any], required_keys: List[str]) -> bool:
        """Validate metrics dictionary contains required keys."""
        return all(key in metrics for key in required_keys)

    @staticmethod
    def validate_config_schema(config: Dict[str, Any], schema: Dict[str, type]) -> bool:
        """Validate configuration against a type schema."""
        for key, expected_type in schema.items():
            if key not in config:
                return False
            if not isinstance(config[key], expected_type):
                return False
        return True


def load_test_config(filename: str) -> Dict[str, Any]:
    """Load test configuration from YAML file."""
    test_configs_dir = Path(__file__).parent.parent / "fixtures" / "configs"
    config_path = test_configs_dir / filename

    if config_path.suffix == ".yaml":
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    elif config_path.suffix == ".json":
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")


def assert_response_time(func, max_time_seconds: float = 1.0) -> Any:
    """Assert function executes within time limit."""
    import time

    start = time.time()
    result = func()
    elapsed = time.time() - start

    assert (
        elapsed < max_time_seconds
    ), f"Function took {elapsed}s, max allowed: {max_time_seconds}s"
    return result


def create_temp_yaml_file(content: Dict[str, Any], temp_dir: Path) -> Path:
    """Create a temporary YAML file for testing."""
    temp_file = temp_dir / f"test_{random.randint(1000, 9999)}.yaml"
    with open(temp_file, "w") as f:
        yaml.dump(content, f)
    return temp_file


def compare_dicts_ignore_order(dict1: Dict, dict2: Dict) -> bool:
    """Compare two dictionaries ignoring order of lists."""
    if set(dict1.keys()) != set(dict2.keys()):
        return False

    for key in dict1:
        val1, val2 = dict1[key], dict2[key]

        if isinstance(val1, dict) and isinstance(val2, dict):
            if not compare_dicts_ignore_order(val1, val2):
                return False
        elif isinstance(val1, list) and isinstance(val2, list):
            if sorted(map(str, val1)) != sorted(map(str, val2)):
                return False
        else:
            if val1 != val2:
                return False

    return True
