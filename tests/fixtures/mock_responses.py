"""
Mock response fixtures for testing LLM-D examples.

This module provides pre-defined mock responses for various API calls and
system interactions used in tests.
"""

from typing import Dict, Any, List
from datetime import datetime


class MockLLMResponses:
    """Mock responses from LLM inference endpoints."""
    
    @staticmethod
    def chat_completion_success() -> Dict[str, Any]:
        """Successful chat completion response."""
        return {
            "id": "chatcmpl-test123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "llama-3.1-8b",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Machine learning is a subset of artificial intelligence..."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 45,
                "total_tokens": 60
            }
        }
    
    @staticmethod
    def chat_completion_error() -> Dict[str, Any]:
        """Error response from chat completion."""
        return {
            "error": {
                "message": "Model overloaded. Please retry after 2 seconds.",
                "type": "model_overloaded",
                "code": "model_overloaded"
            }
        }
    
    @staticmethod
    def embeddings_response() -> Dict[str, Any]:
        """Embeddings generation response."""
        return {
            "object": "list",
            "data": [{
                "object": "embedding",
                "embedding": [0.1, -0.2, 0.3] * 256,  # 768-dim embedding
                "index": 0
            }],
            "model": "llama-3.1-8b",
            "usage": {
                "prompt_tokens": 10,
                "total_tokens": 10
            }
        }
    
    @staticmethod
    def streaming_chunk() -> str:
        """Single chunk from streaming response."""
        return 'data: {"id":"chatcmpl-test","choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n'


class MockKubernetesResponses:
    """Mock responses from Kubernetes API."""
    
    @staticmethod
    def list_pods() -> Dict[str, Any]:
        """List of pods response."""
        return {
            "apiVersion": "v1",
            "kind": "PodList",
            "items": [
                {
                    "metadata": {
                        "name": "llm-server-1",
                        "namespace": "llm-models",
                        "uid": "abc123"
                    },
                    "status": {
                        "phase": "Running",
                        "conditions": [{"type": "Ready", "status": "True"}]
                    }
                },
                {
                    "metadata": {
                        "name": "llm-server-2",
                        "namespace": "llm-models",
                        "uid": "def456"
                    },
                    "status": {
                        "phase": "Running",
                        "conditions": [{"type": "Ready", "status": "True"}]
                    }
                }
            ]
        }
    
    @staticmethod
    def get_deployment() -> Dict[str, Any]:
        """Single deployment details."""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "llama-3-1-8b",
                "namespace": "llm-models"
            },
            "spec": {
                "replicas": 2
            },
            "status": {
                "replicas": 2,
                "readyReplicas": 2,
                "availableReplicas": 2
            }
        }
    
    @staticmethod
    def node_metrics() -> List[Dict[str, Any]]:
        """Node metrics for resource monitoring."""
        return [
            {
                "metadata": {"name": "gpu-node-1"},
                "usage": {
                    "cpu": "4000m",
                    "memory": "32Gi",
                    "nvidia.com/gpu": "2"
                }
            },
            {
                "metadata": {"name": "gpu-node-2"},
                "usage": {
                    "cpu": "6000m",
                    "memory": "48Gi",
                    "nvidia.com/gpu": "4"
                }
            }
        ]


class MockPrometheusResponses:
    """Mock responses from Prometheus queries."""
    
    @staticmethod
    def gpu_utilization_query() -> Dict[str, Any]:
        """GPU utilization metrics from Prometheus."""
        return {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {
                        "metric": {
                            "gpu": "0",
                            "instance": "gpu-node-1",
                            "job": "gpu-metrics"
                        },
                        "value": [1234567890, "85.5"]
                    },
                    {
                        "metric": {
                            "gpu": "1",
                            "instance": "gpu-node-1",
                            "job": "gpu-metrics"
                        },
                        "value": [1234567890, "92.3"]
                    }
                ]
            }
        }
    
    @staticmethod
    def request_rate_query() -> Dict[str, Any]:
        """Request rate metrics."""
        return {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [{
                    "metric": {"model": "llama-3.1-8b"},
                    "values": [
                        [1234567890, "125.5"],
                        [1234567950, "132.3"],
                        [1234568010, "128.7"]
                    ]
                }]
            }
        }


class MockMLFlowResponses:
    """Mock responses from MLflow tracking."""
    
    @staticmethod
    def create_run() -> Dict[str, Any]:
        """Create new MLflow run."""
        return {
            "run": {
                "info": {
                    "run_id": "test-run-123",
                    "experiment_id": "exp-456",
                    "status": "RUNNING",
                    "start_time": datetime.now().timestamp() * 1000,
                    "artifact_uri": "s3://bucket/mlflow/exp-456/test-run-123"
                }
            }
        }
    
    @staticmethod
    def get_experiment() -> Dict[str, Any]:
        """Get experiment details."""
        return {
            "experiment": {
                "experiment_id": "exp-456",
                "name": "llm-quantization-experiments",
                "artifact_location": "s3://bucket/mlflow/exp-456",
                "lifecycle_stage": "active"
            }
        }


class MockCostServiceResponses:
    """Mock responses from cost tracking services."""
    
    @staticmethod
    def hourly_costs() -> Dict[str, Any]:
        """Hourly cost breakdown."""
        return {
            "period": "2024-01-15T10:00:00Z",
            "costs": {
                "compute": {
                    "gpu": 125.50,
                    "cpu": 15.25,
                    "memory": 8.75
                },
                "network": {
                    "ingress": 2.50,
                    "egress": 12.25
                },
                "storage": {
                    "persistent": 5.00,
                    "object": 3.25
                },
                "total": 172.50
            },
            "resource_usage": {
                "gpu_hours": 32,
                "cpu_core_hours": 256,
                "memory_gb_hours": 1024,
                "requests_processed": 125000
            }
        }
    
    @staticmethod
    def cost_forecast() -> Dict[str, Any]:
        """Cost forecast for planning."""
        return {
            "forecast_period": "30d",
            "predicted_costs": {
                "optimistic": 4500.00,
                "likely": 5200.00,
                "pessimistic": 6100.00
            },
            "confidence_interval": 0.85,
            "factors": {
                "expected_growth": 0.15,
                "seasonal_adjustment": 1.05,
                "efficiency_improvements": 0.92
            }
        }