---
title: Model Lifecycle Management
description: Managing LLM model lifecycles with Kubeflow and llm-d
sidebar_position: 1
---

# Model Lifecycle Management

This section covers how to manage the complete lifecycle of Large Language Models using Kubeflow's model registry, Argo Workflows, and llm-d's deployment capabilities.

## Model Lifecycle Overview

### Stages of Model Lifecycle

```mermaid
graph LR
    A[Model Selection] --> B[Validation]
    B --> C[Staging Deployment]
    C --> D[Production Candidate]
    D --> E[Production Deployment]
    E --> F[Monitoring]
    F --> G[Retirement]
    
    F --> H[Performance Issues]
    H --> I[Rollback]
    I --> C
    
    F --> J[Model Updates]
    J --> B
```

### Key Principles

1. **Immutable Artifacts**: Models and configurations are versioned and immutable
2. **Progressive Promotion**: Models advance through environments with validation gates
3. **Automated Testing**: Each stage includes automated quality and performance checks
4. **Rollback Capability**: Quick rollback to previous known-good versions
5. **Audit Trail**: Complete history of deployments and changes

## Kubeflow Model Registry Integration

### Model Registry Setup

The Kubeflow Model Registry provides centralized model management with versioning, metadata, and lifecycle tracking.

```yaml
# kubeflow-model-registry.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-registry-config
  namespace: kubeflow
data:
  config.yaml: |
    registry:
      backend: "s3"
      s3_config:
        endpoint: "minio.kubeflow.svc.cluster.local:9000"
        bucket: "model-registry"
        access_key: "minio"
        secret_key: "minio123"
      
      metadata_store:
        type: "mysql"
        connection_string: "mysql://user:password@mysql.kubeflow.svc.cluster.local:3306/mlmetadata"
```

### Model Registration Pipeline

```python
# register_model.py - Model registration pipeline
from kfp import dsl, compiler
from kfp.client import Client
import os

@dsl.component(
    base_image="python:3.9",
    packages_to_install=["kubeflow-model-registry", "boto3"]
)
def register_model(
    model_name: str,
    model_version: str,
    model_uri: str,
    model_format: str = "pytorch",
    description: str = ""
) -> str:
    """Register a model in Kubeflow Model Registry"""
    from model_registry import ModelRegistry
    import json
    
    # Initialize registry client
    registry = ModelRegistry(
        server_address="model-registry-service.kubeflow.svc.cluster.local:8080"
    )
    
    # Create model metadata
    model_metadata = {
        "name": model_name,
        "version": model_version,
        "uri": model_uri,
        "format": model_format,
        "description": description,
        "framework": "pytorch",
        "tags": ["llm", "production-ready"]
    }
    
    # Register model
    registered_model = registry.register_model(
        name=model_name,
        version=model_version,
        model_uri=model_uri,
        metadata=model_metadata
    )
    
    return json.dumps({
        "model_id": registered_model.id,
        "version": registered_model.version,
        "status": "registered"
    })

@dsl.component(
    base_image="python:3.9",
    packages_to_install=["torch", "transformers", "requests"]
)
def validate_model(
    model_uri: str,
    validation_dataset: str,
    performance_threshold: float = 0.8
) -> str:
    """Validate model performance and compatibility"""
    import torch
    import json
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    try:
        # Load model for validation
        tokenizer = AutoTokenizer.from_pretrained(model_uri)
        model = AutoModelForCausalLM.from_pretrained(
            model_uri,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Basic functionality test
        test_input = "The quick brown fox"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=10,
                do_sample=False
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Performance validation (simplified)
        # In practice, this would run comprehensive benchmarks
        performance_score = 0.85  # Placeholder
        
        validation_result = {
            "status": "passed" if performance_score >= performance_threshold else "failed",
            "performance_score": performance_score,
            "test_output": generated_text,
            "model_size_gb": model.get_memory_footprint() / (1024**3)
        }
        
        return json.dumps(validation_result)
        
    except Exception as e:
        return json.dumps({
            "status": "failed",
            "error": str(e)
        })

@dsl.pipeline(
    name="model-registration-pipeline",
    description="Register and validate LLM models"
)
def model_registration_pipeline(
    model_name: str,
    model_version: str,
    model_uri: str,
    validation_dataset: str = "default"
):
    """Complete model registration and validation pipeline"""
    
    # Step 1: Validate model
    validation_task = validate_model(
        model_uri=model_uri,
        validation_dataset=validation_dataset
    )
    
    # Step 2: Register model (only if validation passes)
    with dsl.Condition(
        validation_task.outputs["status"] == "passed",
        name="validation-passed"
    ):
        registration_task = register_model(
            model_name=model_name,
            model_version=model_version,
            model_uri=model_uri,
            description=f"Validated model with score: {validation_task.outputs['performance_score']}"
        )

# Compile pipeline
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=model_registration_pipeline,
        package_path="model_registration_pipeline.yaml"
    )
```

### Model Deployment Pipeline

```python
# deploy_model.py - Model deployment pipeline
from kfp import dsl, compiler
import yaml

@dsl.component(
    base_image="bitnami/kubectl:latest",
    packages_to_install=["pyyaml"]
)
def deploy_llm_model(
    model_name: str,
    model_version: str,
    model_uri: str,
    namespace: str = "production",
    replicas: int = 1,
    gpu_count: int = 1,
    memory_limit: str = "32Gi"
) -> str:
    """Deploy LLM model using llm-d operator"""
    import yaml
    import subprocess
    import json
    
    # Create LLMDeployment manifest
    deployment_manifest = {
        "apiVersion": "inference.llm-d.io/v1alpha1",
        "kind": "LLMDeployment",
        "metadata": {
            "name": f"{model_name}-{model_version}".replace(".", "-").lower(),
            "namespace": namespace,
            "labels": {
                "model": model_name,
                "version": model_version,
                "managed-by": "kubeflow"
            }
        },
        "spec": {
            "model": {
                "name": model_name,
                "source": {
                    "modelUri": model_uri
                },
                "framework": "pytorch"
            },
            "replicas": replicas,
            "resources": {
                "requests": {
                    "nvidia.com/gpu": str(gpu_count),
                    "memory": memory_limit
                },
                "limits": {
                    "nvidia.com/gpu": str(gpu_count),
                    "memory": memory_limit
                }
            },
            "serving": {
                "protocol": "http",
                "port": 8080
            },
            "autoscaling": {
                "enabled": True,
                "minReplicas": replicas,
                "maxReplicas": replicas * 3,
                "targetGPUUtilization": 70
            }
        }
    }
    
    # Write manifest to file
    with open("/tmp/deployment.yaml", "w") as f:
        yaml.dump(deployment_manifest, f)
    
    # Apply deployment
    try:
        result = subprocess.run(
            ["kubectl", "apply", "-f", "/tmp/deployment.yaml"],
            capture_output=True,
            text=True,
            check=True
        )
        
        return json.dumps({
            "status": "deployed",
            "deployment_name": deployment_manifest["metadata"]["name"],
            "namespace": namespace,
            "kubectl_output": result.stdout
        })
        
    except subprocess.CalledProcessError as e:
        return json.dumps({
            "status": "failed",
            "error": e.stderr
        })

@dsl.component(
    base_image="curlimages/curl:latest"
)
def health_check(
    service_name: str,
    namespace: str,
    timeout_seconds: int = 300
) -> str:
    """Check if deployed model is healthy and responding"""
    import subprocess
    import time
    import json
    
    service_url = f"http://{service_name}.{namespace}.svc.cluster.local:8080"
    
    # Wait for service to be ready
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        try:
            result = subprocess.run(
                ["curl", "-f", f"{service_url}/health"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return json.dumps({
                    "status": "healthy",
                    "response_time": time.time() - start_time,
                    "health_response": result.stdout
                })
                
        except Exception:
            pass
        
        time.sleep(10)
    
    return json.dumps({
        "status": "unhealthy",
        "error": "Health check timeout"
    })

@dsl.pipeline(
    name="model-deployment-pipeline",
    description="Deploy LLM models with health checks"
)
def model_deployment_pipeline(
    model_name: str,
    model_version: str,
    model_uri: str,
    target_namespace: str = "staging",
    replicas: int = 1
):
    """Complete model deployment pipeline with validation"""
    
    # Deploy model
    deploy_task = deploy_llm_model(
        model_name=model_name,
        model_version=model_version,
        model_uri=model_uri,
        namespace=target_namespace,
        replicas=replicas
    )
    
    # Health check
    health_task = health_check(
        service_name=f"{model_name}-{model_version}".replace(".", "-").lower(),
        namespace=target_namespace
    )
    health_task.after(deploy_task)

# Compile pipeline
if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=model_deployment_pipeline,
        package_path="model_deployment_pipeline.yaml"
    )
```

## Environment Progression

### Environment Strategy

```yaml
# environments.yaml - Environment configuration
environments:
  development:
    namespace: "dev"
    gpu_nodes: "t4-nodes"
    replicas: 1
    auto_promote: true
    validation_required: false
    
  staging:
    namespace: "staging"
    gpu_nodes: "v100-nodes"
    replicas: 2
    auto_promote: false
    validation_required: true
    performance_threshold: 0.8
    
  production:
    namespace: "production"
    gpu_nodes: "a100-nodes"
    replicas: 3
    auto_promote: false
    validation_required: true
    performance_threshold: 0.9
    approval_required: true
```

### Promotion Workflow

```python
# promotion_pipeline.py - Environment promotion
from kfp import dsl, compiler

@dsl.component(
    base_image="python:3.9",
    packages_to_install=["kubernetes", "requests"]
)
def check_promotion_criteria(
    model_name: str,
    current_environment: str,
    target_environment: str
) -> str:
    """Check if model meets promotion criteria"""
    import json
    import requests
    from kubernetes import client, config
    
    config.load_incluster_config()
    v1 = client.CoreV1Api()
    
    criteria = {
        "staging_to_production": {
            "uptime_hours": 72,
            "error_rate_threshold": 0.01,
            "p95_latency_ms": 2000,
            "approval_required": True
        }
    }
    
    promotion_key = f"{current_environment}_to_{target_environment}"
    requirements = criteria.get(promotion_key, {})
    
    # Check metrics (simplified - would integrate with Prometheus)
    checks = {
        "uptime_check": True,  # Placeholder
        "error_rate_check": True,  # Placeholder
        "latency_check": True,  # Placeholder
        "approval_check": False  # Would check approval system
    }
    
    all_passed = all(checks.values())
    
    return json.dumps({
        "promotion_approved": all_passed,
        "checks": checks,
        "requirements": requirements
    })

@dsl.pipeline(
    name="model-promotion-pipeline",
    description="Promote model between environments"
)
def model_promotion_pipeline(
    model_name: str,
    model_version: str,
    source_environment: str,
    target_environment: str
):
    """Promote model from one environment to another"""
    
    # Check promotion criteria
    criteria_task = check_promotion_criteria(
        model_name=model_name,
        current_environment=source_environment,
        target_environment=target_environment
    )
    
    # Deploy to target environment if criteria met
    with dsl.Condition(
        criteria_task.outputs["promotion_approved"] == "true",
        name="promotion-approved"
    ):
        deployment_task = deploy_llm_model(
            model_name=model_name,
            model_version=model_version,
            model_uri=f"s3://model-registry/{model_name}/{model_version}",
            namespace=target_environment
        )
```

## Version Management

### Semantic Versioning for Models

```yaml
# Model versioning strategy
versioning:
  format: "MAJOR.MINOR.PATCH"
  
  # MAJOR: Breaking changes (architecture changes, incompatible API)
  # MINOR: New features (improved accuracy, new capabilities)
  # PATCH: Bug fixes (performance optimizations, small fixes)
  
  examples:
    - "1.0.0": Initial production release
    - "1.1.0": Improved accuracy with same API
    - "1.1.1": Performance optimization
    - "2.0.0": New architecture (e.g., Llama2 to Llama4)
```

### Model Metadata Schema

```python
# model_metadata.py - Standardized model metadata
from dataclasses import dataclass
from typing import Dict, List, Optional
import json

@dataclass
class ModelMetadata:
    """Standardized model metadata for registry"""
    
    # Basic information
    name: str
    version: str
    description: str
    
    # Technical specifications
    architecture: str  # e.g., "llama-3.1-7b"
    framework: str  # e.g., "pytorch"
    parameters: int  # e.g., 7000000000
    precision: str  # e.g., "fp16"
    
    # Resource requirements
    memory_gb: float
    gpu_memory_gb: float
    min_gpu_count: int
    recommended_gpu_type: str
    
    # Performance characteristics
    max_sequence_length: int
    throughput_tokens_per_second: Optional[float]
    latency_p95_ms: Optional[float]
    
    # Validation results
    accuracy_score: Optional[float]
    benchmark_results: Dict[str, float]
    
    # Lifecycle information
    created_at: str
    created_by: str
    training_dataset: Optional[str]
    parent_model: Optional[str]
    
    # Deployment configuration
    default_batch_size: int
    recommended_replicas: int
    
    def to_json(self) -> str:
        """Convert to JSON for registry storage"""
        return json.dumps(self.__dict__, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ModelMetadata':
        """Create from JSON"""
        data = json.loads(json_str)
        return cls(**data)

# Example usage
llama-3.1_7b_metadata = ModelMetadata(
    name="llama-3.1-7b",
    version="1.0.0",
    description="Llama4 7B parameter model optimized for chat",
    architecture="llama-3.1-7b",
    framework="pytorch",
    parameters=7000000000,
    precision="fp16",
    memory_gb=16.0,
    gpu_memory_gb=14.0,
    min_gpu_count=1,
    recommended_gpu_type="A100",
    max_sequence_length=4096,
    throughput_tokens_per_second=150.0,
    latency_p95_ms=800.0,
    accuracy_score=0.85,
    benchmark_results={
        "hellaswag": 0.78,
        "arc_challenge": 0.52,
        "truthfulqa": 0.41
    },
    created_at="2024-01-15T10:00:00Z",
    created_by="ml-team",
    training_dataset="custom-chat-dataset-v1",
    parent_model=None,
    default_batch_size=4,
    recommended_replicas=2
)
```

## Integration with llm-d

### Automatic Deployment Configuration

```python
# llm_d_integration.py - Generate llm-d configs from model metadata
def generate_llm_deployment(metadata: ModelMetadata, environment: str) -> dict:
    """Generate llm-d deployment from model metadata"""
    
    env_configs = {
        "dev": {"replicas": 1, "resources_multiplier": 0.5},
        "staging": {"replicas": 2, "resources_multiplier": 0.8},
        "production": {"replicas": metadata.recommended_replicas, "resources_multiplier": 1.0}
    }
    
    env_config = env_configs.get(environment, env_configs["production"])
    
    return {
        "apiVersion": "inference.llm-d.io/v1alpha1",
        "kind": "LLMDeployment",
        "metadata": {
            "name": f"{metadata.name}-{metadata.version}".replace(".", "-").lower(),
            "namespace": environment,
            "labels": {
                "model": metadata.name,
                "version": metadata.version,
                "architecture": metadata.architecture
            },
            "annotations": {
                "model-registry/id": f"{metadata.name}/{metadata.version}",
                "model-registry/parameters": str(metadata.parameters),
                "model-registry/precision": metadata.precision
            }
        },
        "spec": {
            "model": {
                "name": metadata.name,
                "source": {
                    "modelUri": f"s3://model-registry/{metadata.name}/{metadata.version}"
                },
                "framework": metadata.framework
            },
            "replicas": env_config["replicas"],
            "resources": {
                "requests": {
                    "nvidia.com/gpu": str(metadata.min_gpu_count),
                    "memory": f"{int(metadata.memory_gb * env_config['resources_multiplier'])}Gi"
                },
                "limits": {
                    "nvidia.com/gpu": str(metadata.min_gpu_count),
                    "memory": f"{int(metadata.memory_gb * env_config['resources_multiplier'])}Gi"
                }
            },
            "serving": {
                "batchSize": metadata.default_batch_size,
                "maxSequenceLength": metadata.max_sequence_length,
                "protocol": "http",
                "port": 8080
            },
            "autoscaling": {
                "enabled": True,
                "minReplicas": env_config["replicas"],
                "maxReplicas": env_config["replicas"] * 3,
                "targetGPUUtilization": 70
            }
        }
    }
```

## Rollback Strategies

### Blue-Green Deployment

```yaml
# Blue-green deployment configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: deployment-strategy
data:
  blue-green.yaml: |
    strategy: "blue-green"
    traffic_split:
      validation_period: "10m"
      error_threshold: 0.05
      latency_threshold: "2s"
    
    rollback:
      automatic: true
      triggers:
        - error_rate > 0.05
        - p95_latency > 2000ms
        - health_check_failures > 3
```

### Canary Deployment

```python
# canary_deployment.py - Progressive traffic shifting
from kfp import dsl

@dsl.component(
    base_image="istio/pilot:latest"
)
def configure_traffic_split(
    service_name: str,
    namespace: str,
    canary_version: str,
    traffic_percentage: int
) -> str:
    """Configure Istio traffic splitting for canary deployment"""
    import yaml
    import subprocess
    import json
    
    # Create Istio VirtualService for traffic splitting
    virtual_service = {
        "apiVersion": "networking.istio.io/v1beta1",
        "kind": "VirtualService",
        "metadata": {
            "name": f"{service_name}-traffic-split",
            "namespace": namespace
        },
        "spec": {
            "hosts": [service_name],
            "http": [{
                "match": [{"headers": {"canary": {"exact": "true"}}}],
                "route": [{
                    "destination": {
                        "host": service_name,
                        "subset": "canary"
                    }
                }]
            }, {
                "route": [{
                    "destination": {
                        "host": service_name,
                        "subset": "stable"
                    },
                    "weight": 100 - traffic_percentage
                }, {
                    "destination": {
                        "host": service_name,
                        "subset": "canary"
                    },
                    "weight": traffic_percentage
                }]
            }]
        }
    }
    
    # Apply configuration
    with open("/tmp/virtual-service.yaml", "w") as f:
        yaml.dump(virtual_service, f)
    
    try:
        result = subprocess.run(
            ["kubectl", "apply", "-f", "/tmp/virtual-service.yaml"],
            capture_output=True,
            text=True,
            check=True
        )
        
        return json.dumps({
            "status": "configured",
            "traffic_percentage": traffic_percentage,
            "output": result.stdout
        })
        
    except subprocess.CalledProcessError as e:
        return json.dumps({
            "status": "failed",
            "error": e.stderr
        })
```

## Best Practices

### Model Lifecycle Checklist

- [ ] **Model Registration**: All models registered with complete metadata
- [ ] **Validation Pipeline**: Automated performance and compatibility testing
- [ ] **Environment Progression**: Clear staging → production promotion criteria
- [ ] **Version Control**: Semantic versioning with clear changelog
- [ ] **Rollback Strategy**: Tested rollback procedures for each environment
- [ ] **Monitoring Integration**: SLOs and alerts configured for each deployment
- [ ] **Security Scanning**: Container and model vulnerability scanning
- [ ] **Resource Planning**: Capacity planning for each model version

### Common Pitfalls

1. **Insufficient Testing**: Deploy comprehensive validation before production
2. **Missing Rollback Plans**: Always have a tested rollback strategy
3. **Resource Underestimation**: Plan for peak load, not average
4. **Version Confusion**: Maintain clear version tracking and dependencies
5. **Manual Processes**: Automate everything possible to reduce errors

## Next Steps

With model lifecycle management in place, the next step is building robust [CI/CD Pipelines](./02-cicd-pipelines.md) that automate the entire process from code commit to production deployment.

## References

- [Kubeflow Pipelines Documentation](https://www.kubeflow.org/docs/components/pipelines/)
- [Argo Workflows Guide](https://argoproj.github.io/argo-workflows/)
- [Chapter 5: SRE Operations](../05-sre-operations.md) - SLO and monitoring foundations
- [Chapter 8: Troubleshooting](../08-troubleshooting/index.md) - Deployment troubleshooting procedures
