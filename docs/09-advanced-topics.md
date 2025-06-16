---
title: Advanced Topics
description: Comprehensive guide to advanced llm-d deployment patterns, covering multi-cluster federation, custom operators, and enterprise integrations
sidebar_position: 9
---

# Advanced Topics

:::info Chapter Overview
This chapter focuses on advanced llm-d deployment patterns for enterprise environments, covering multi-cluster federation, advanced networking, custom operator development, and integration with external ML platforms. You'll learn to implement sophisticated deployment architectures that scale beyond single-cluster scenarios.
:::

## Multi-Cluster Deployments and Federation

### Cross-Cluster Model Federation

Enterprise LLM deployments often require distribution across multiple Kubernetes clusters for geographical distribution, regulatory compliance, or resource optimization.

```yaml
# Multi-cluster federation configuration
apiVersion: federation.llm-d.io/v1alpha1
kind: FederatedLLMDeployment
metadata:
  name: llama-3.1-8b-global
  namespace: production
spec:
  template:
    metadata:
      labels:
        app.kubernetes.io/name: llm-d
        llm-d.ai/model: "llama-3.1"
        llm-d.ai/size: "8b"
    spec:
      model:
        name: "llama-3.1-8b"
      resources:
        requests:
          memory: "16Gi"
          cpu: "4"
          nvidia.com/gpu: "1"
        limits:
          memory: "24Gi"
          cpu: "8"
          nvidia.com/gpu: "1"
  
  placement:
    clusters:
    - name: "us-west-cluster"
      weight: 40
      constraints:
        - key: "region"
          operator: "In"
          values: ["us-west-2"]
    - name: "eu-west-cluster"
      weight: 35
      constraints:
        - key: "region"
          operator: "In"  
          values: ["eu-west-1"]
    - name: "ap-southeast-cluster"
      weight: 25
      constraints:
        - key: "region"
          operator: "In"
          values: ["ap-southeast-1"]
  
  scheduling:
    type: "LatencyBased"
    preferences:
    - weight: 80
      preference:
        matchExpressions:
        - key: "llm-d.ai/tier"
          operator: In
          values: ["premium"]
```

### Cross-Cluster Service Discovery

```python title="federation-controller.py" showLineNumbers
#!/usr/bin/env python3
"""
Multi-cluster service discovery and load balancing for federated LLM deployments.
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional
from dataclasses import dataclass
from kubernetes import client, config
import time

@dataclass
class ClusterEndpoint:
    cluster_name: str
    endpoint_url: str
    region: str
    latency_ms: float
    health_status: str
    current_load: float

class FederatedServiceDiscovery:
    def __init__(self, clusters: Dict[str, str]):
        """
        Initialize federated service discovery.
        
        Args:
            clusters: Dict mapping cluster names to kubeconfig contexts
        """
        self.clusters = clusters
        self.endpoints: Dict[str, ClusterEndpoint] = {}
        self.health_check_interval = 30
        
    async def discover_endpoints(self) -> List[ClusterEndpoint]:
        """Discover all available LLM service endpoints across clusters."""
        endpoints = []
        
        for cluster_name, context in self.clusters.items():
            try:
                # Load cluster-specific kubeconfig
                config.load_kube_config(context=context)
                v1 = client.CoreV1Api()
                
                # Find LLM services
                services = v1.list_namespaced_service(
                    namespace="production",
                    label_selector="app.kubernetes.io/name=llm-d"
                )
                
                for service in services.items:
                    endpoint_url = f"http://{service.spec.cluster_ip}:8080"
                    
                    # Measure latency
                    latency = await self._measure_latency(endpoint_url)
                    
                    endpoint = ClusterEndpoint(
                        cluster_name=cluster_name,
                        endpoint_url=endpoint_url,
                        region=service.metadata.labels.get("region", "unknown"),
                        latency_ms=latency,
                        health_status="healthy",
                        current_load=0.0
                    )
                    endpoints.append(endpoint)
                    
            except Exception as e:
                print(f"Failed to discover endpoints in {cluster_name}: {e}")
                
        return endpoints
    
    async def _measure_latency(self, endpoint_url: str) -> float:
        """Measure endpoint latency."""
        start_time = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{endpoint_url}/health", timeout=5) as response:
                    if response.status == 200:
                        return (time.time() - start_time) * 1000
        except:
            return float('inf')
        return float('inf')
    
    async def route_request(self, user_region: str) -> Optional[ClusterEndpoint]:
        """Route request to optimal endpoint based on user location and load."""
        available_endpoints = [ep for ep in self.endpoints.values() 
                             if ep.health_status == "healthy"]
        
        if not available_endpoints:
            return None
        
        # Prefer same region, then lowest latency + load
        same_region = [ep for ep in available_endpoints if ep.region == user_region]
        if same_region:
            return min(same_region, key=lambda ep: ep.current_load)
        
        # Fallback to lowest latency + load score
        return min(available_endpoints, 
                  key=lambda ep: ep.latency_ms + (ep.current_load * 100))

async def main():
    """Example usage of federated service discovery."""
    clusters = {
        "us-west": "us-west-context",
        "eu-west": "eu-west-context", 
        "ap-southeast": "ap-southeast-context"
    }
    
    discovery = FederatedServiceDiscovery(clusters)
    endpoints = await discovery.discover_endpoints()
    
    print(f"Discovered {len(endpoints)} endpoints across {len(clusters)} clusters")
    for endpoint in endpoints:
        print(f"  {endpoint.cluster_name}: {endpoint.latency_ms:.1f}ms latency")

if __name__ == "__main__":
    asyncio.run(main())
```

### Cluster Failover and Recovery

```bash
# Cross-cluster failover automation
#!/bin/bash
# cluster-failover.sh - Automated failover between clusters

PRIMARY_CLUSTER="us-west"
SECONDARY_CLUSTER="eu-west"
HEALTH_ENDPOINT="/health"

check_cluster_health() {
    local cluster=$1
    local context="${cluster}-context"
    
    # Switch to cluster context
    kubectl config use-context "$context"
    
    # Check if llm-d services are healthy
    local healthy_pods=$(kubectl get pods -n production -l app.kubernetes.io/name=llm-d \
                        --field-selector=status.phase=Running --no-headers | wc -l)
    
    echo "$healthy_pods"
}

initiate_failover() {
    local target_cluster=$1
    echo "🔄 Initiating failover to $target_cluster cluster..."
    
    # Update DNS or load balancer to point to secondary cluster
    kubectl config use-context "${target_cluster}-context"
    
    # Scale up secondary cluster capacity if needed
    kubectl patch llmdeployment llama-3.1-8b -n production \
        --type='merge' -p='{"spec":{"replicas":5}}'
    
    # Update ingress traffic routing
    kubectl patch ingress llm-api-ingress -n production \
        --type='merge' -p='{
            "metadata": {
                "annotations": {
                    "nginx.ingress.kubernetes.io/upstream-vhost": "'$target_cluster'.llm-api.example.com"
                }
            }
        }'
    
    echo "✅ Failover to $target_cluster completed"
}

# Main failover logic
while true; do
    primary_health=$(check_cluster_health "$PRIMARY_CLUSTER")
    
    if [ "$primary_health" -lt 1 ]; then
        echo "🚨 Primary cluster unhealthy ($primary_health healthy pods)"
        
        secondary_health=$(check_cluster_health "$SECONDARY_CLUSTER")
        if [ "$secondary_health" -gt 0 ]; then
            initiate_failover "$SECONDARY_CLUSTER"
        else
            echo "❌ Both clusters unhealthy - manual intervention required"
        fi
    else
        echo "✅ Primary cluster healthy ($primary_health pods)"
    fi
    
    sleep 30
done
```

## Advanced Networking and Service Mesh

### Advanced Istio Integration

```yaml
# Advanced service mesh configuration for LLM traffic management
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: llama-3.1-8b-advanced-routing
  namespace: production
spec:
  hosts:
  - llama-api.example.com
  http:
  - match:
    - headers:
        user-tier:
          exact: "premium"
    route:
    - destination:
        host: llama-3.1-8b-svc
        subset: premium
      weight: 100
    timeout: 30s
    retries:
      attempts: 3
      perTryTimeout: 10s
      retryOn: 5xx,reset,connect-failure,refused-stream
  
  - match:
    - headers:
        content-length:
          regex: "^[0-9]{1,4}$"  # Small requests < 10KB
    route:
    - destination:
        host: llama-3.1-8b-svc
        subset: fast-inference
      weight: 100
    timeout: 5s
  
  - route:
    - destination:
        host: llama-3.1-8b-svc
        subset: standard
      weight: 100
    timeout: 60s
    fault:
      delay:
        percentage:
          value: 0.1
        fixedDelay: 1s

---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: llama-3.1-8b-destination-rule
  namespace: production
spec:
  host: llama-3.1-8b-svc
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 50
        maxRequestsPerConnection: 2
    loadBalancer:
      simple: LEAST_REQUEST
    outlierDetection:
      consecutiveErrors: 3
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
  
  subsets:
  - name: premium
    labels:
      tier: premium
    trafficPolicy:
      connectionPool:
        tcp:
          maxConnections: 200
        http:
          http1MaxPendingRequests: 100
  
  - name: fast-inference
    labels:
      inference-type: fast
    trafficPolicy:
      connectionPool:
        tcp:
          maxConnections: 150
          
  - name: standard
    labels:
      tier: standard
```

### Network Security Policies

```yaml
# Advanced network security for LLM workloads
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: llm-strict-isolation
  namespace: production
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: llm-d
  policyTypes:
  - Ingress
  - Egress
  
  ingress:
  # Allow traffic from ingress controllers
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-system
    ports:
    - protocol: TCP
      port: 8080
  
  # Allow traffic from monitoring systems
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 8081  # Metrics port
  
  # Allow inter-pod communication for multi-GPU deployments
  - from:
    - podSelector:
        matchLabels:
          app.kubernetes.io/name: llm-d
    ports:
    - protocol: TCP
      port: 29500  # NCCL communication
    - protocol: TCP
      port: 29501
  
  egress:
  # Allow DNS resolution
  - to: []
    ports:
    - protocol: UDP
      port: 53
  
  # Allow model downloads from trusted registries
  - to:
    - namespaceSelector:
        matchLabels:
          name: model-registry
    ports:
    - protocol: TCP
      port: 443
  
  # Allow logging to centralized systems
  - to:
    - namespaceSelector:
        matchLabels:
          name: logging
    ports:
    - protocol: TCP
      port: 24224  # Fluentd

---
apiVersion: security.istio.io/v1beta1
kind: AuthorizationPolicy
metadata:
  name: llm-api-authz
  namespace: production
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: llm-d
  
  rules:
  # Allow authenticated users with valid tokens
  - from:
    - source:
        requestPrincipals: ["cluster.local/ns/default/sa/llm-client"]
    to:
    - operation:
        methods: ["POST"]
        paths: ["/v1/completions", "/v1/chat/completions"]
    when:
    - key: request.headers[authorization]
      values: ["Bearer *"]
  
  # Allow health checks from load balancers
  - from:
    - source:
        principals: ["cluster.local/ns/ingress-system/sa/nginx-ingress"]
    to:
    - operation:
        methods: ["GET"]
        paths: ["/health", "/metrics"]
  
  # Deny all other traffic
  - from:
    - source:
        notPrincipals: ["*"]
    to:
    - operation:
        methods: ["*"]
    action: DENY
```

## Custom Operators and CRD Development

### Extended LLM Deployment CRD

```yaml
# Extended CRD for advanced LLM deployment scenarios
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: advancedllmdeployments.inference.llm-d.io
spec:
  group: inference.llm-d.io
  versions:
  - name: v1alpha1
    served: true
    storage: true
    schema:
      openAPIV3Schema:
        type: object
        properties:
          spec:
            type: object
            properties:
              model:
                type: object
                properties:
                  name:
                    type: string
                  quantization:
                    type: object
                    properties:
                      enabled:
                        type: boolean
                      method:
                        type: string
                        enum: ["gptq", "awq", "bnb"]
                  optimization:
                    type: object
                    properties:
                      compilation:
                        type: boolean
                      tensorrt:
                        type: boolean
                      speculative_decoding:
                        type: boolean
              
              deployment:
                type: object
                properties:
                  strategy:
                    type: string
                    enum: ["RollingUpdate", "BlueGreen", "Canary"]
                  canary:
                    type: object
                    properties:
                      steps:
                        type: array
                        items:
                          type: object
                          properties:
                            weight:
                              type: integer
                            duration:
                              type: string
              
              scaling:
                type: object
                properties:
                  hpa:
                    type: object
                    properties:
                      enabled:
                        type: boolean
                      minReplicas:
                        type: integer
                      maxReplicas:
                        type: integer
                      metrics:
                        type: array
                        items:
                          type: object
                  
              federation:
                type: object
                properties:
                  enabled:
                    type: boolean
                  clusters:
                    type: array
                    items:
                      type: object
                      properties:
                        name:
                          type: string
                        weight:
                          type: integer
                        region:
                          type: string
  
  scope: Namespaced
  names:
    plural: advancedllmdeployments
    singular: advancedllmdeployment
    kind: AdvancedLLMDeployment
    shortNames:
    - allmd
```

### Custom Operator Implementation

```python title="advanced-llm-operator.py" showLineNumbers
#!/usr/bin/env python3
"""
Advanced LLM Operator with support for multi-cluster federation,
advanced scaling, and deployment strategies.
"""

import asyncio
import logging
from typing import Dict, List, Optional
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException
import yaml
import kopf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@kopf.on.create('inference.llm-d.io', 'v1alpha1', 'advancedllmdeployments')
async def create_advanced_llm_deployment(spec, name, namespace, **kwargs):
    """Handle creation of AdvancedLLMDeployment resources."""
    logger.info(f"Creating AdvancedLLMDeployment: {name} in {namespace}")
    
    # Extract configuration
    model_config = spec.get('model', {})
    deployment_config = spec.get('deployment', {})
    scaling_config = spec.get('scaling', {})
    federation_config = spec.get('federation', {})
    
    try:
        # Create base LLMDeployment
        base_deployment = await create_base_deployment(
            name, namespace, model_config, scaling_config
        )
        
        # Apply deployment strategy
        if deployment_config.get('strategy') == 'Canary':
            await setup_canary_deployment(name, namespace, deployment_config)
        elif deployment_config.get('strategy') == 'BlueGreen':
            await setup_blue_green_deployment(name, namespace)
        
        # Setup federation if enabled
        if federation_config.get('enabled', False):
            await setup_federation(name, namespace, federation_config)
        
        # Configure advanced scaling
        if scaling_config.get('hpa', {}).get('enabled', False):
            await setup_advanced_hpa(name, namespace, scaling_config['hpa'])
        
        logger.info(f"Successfully created AdvancedLLMDeployment: {name}")
        return {"status": "created", "deployment": base_deployment}
        
    except Exception as e:
        logger.error(f"Failed to create AdvancedLLMDeployment {name}: {e}")
        raise kopf.PermanentError(f"Creation failed: {e}")

async def create_base_deployment(name: str, namespace: str, 
                               model_config: Dict, scaling_config: Dict) -> Dict:
    """Create the base LLMDeployment resource."""
    
    # Build LLMDeployment spec with standard naming
    deployment_spec = {
        "apiVersion": "inference.llm-d.io/v1alpha1",
        "kind": "LLMDeployment", 
        "metadata": {
            "name": f"{name}-base",
            "namespace": namespace,
            "labels": {
                "app.kubernetes.io/name": "llm-d",
                "llm-d.ai/deployment": name,
                "llm-d.ai/model": model_config.get('name', '').split('/')[-1].lower()
            }
        },
        "spec": {
            "model": {
                "name": model_config.get('name', 'meta-llama/Llama-3.1-8B-Instruct')
            },
            "resources": {
                "requests": {
                    "memory": "16Gi",
                    "cpu": "4", 
                    "nvidia.com/gpu": "1"
                },
                "limits": {
                    "memory": "24Gi",
                    "cpu": "8",
                    "nvidia.com/gpu": "1"
                }
            }
        }
    }
    
    # Apply quantization if specified
    if model_config.get('quantization', {}).get('enabled', False):
        deployment_spec['spec']['model']['quantization'] = {
            "enabled": True,
            "method": model_config['quantization'].get('method', 'gptq')
        }
    
    # Apply optimizations
    optimization = model_config.get('optimization', {})
    if optimization:
        deployment_spec['spec']['model']['optimization'] = optimization
    
    return deployment_spec

async def setup_canary_deployment(name: str, namespace: str, config: Dict):
    """Setup canary deployment with Istio traffic splitting."""
    
    canary_config = config.get('canary', {})
    steps = canary_config.get('steps', [{"weight": 10, "duration": "5m"}])
    
    # Create VirtualService for traffic splitting
    virtual_service = {
        "apiVersion": "networking.istio.io/v1beta1",
        "kind": "VirtualService",
        "metadata": {
            "name": f"{name}-canary",
            "namespace": namespace
        },
        "spec": {
            "hosts": [f"{name}-svc"],
            "http": [{
                "match": [{"headers": {"canary": {"exact": "true"}}}],
                "route": [{
                    "destination": {
                        "host": f"{name}-svc",
                        "subset": "canary"
                    },
                    "weight": steps[0].get('weight', 10)
                }]
            }, {
                "route": [{
                    "destination": {
                        "host": f"{name}-svc", 
                        "subset": "stable"
                    },
                    "weight": 100 - steps[0].get('weight', 10)
                }]
            }]
        }
    }
    
    logger.info(f"Created canary VirtualService for {name}")
    return virtual_service

async def setup_federation(name: str, namespace: str, config: Dict):
    """Setup multi-cluster federation."""
    
    clusters = config.get('clusters', [])
    
    for cluster in clusters:
        cluster_name = cluster.get('name')
        weight = cluster.get('weight', 100)
        region = cluster.get('region', 'unknown')
        
        # Create cluster-specific deployment
        federated_deployment = {
            "apiVersion": "inference.llm-d.io/v1alpha1", 
            "kind": "LLMDeployment",
            "metadata": {
                "name": f"{name}-{cluster_name}",
                "namespace": namespace,
                "labels": {
                    "llm-d.ai/cluster": cluster_name,
                    "llm-d.ai/region": region,
                    "llm-d.ai/weight": str(weight)
                }
            }
        }
        
        logger.info(f"Created federated deployment for cluster: {cluster_name}")

async def setup_advanced_hpa(name: str, namespace: str, hpa_config: Dict):
    """Setup advanced HPA with custom metrics."""
    
    hpa_spec = {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": {
            "name": f"{name}-hpa",
            "namespace": namespace
        },
        "spec": {
            "scaleTargetRef": {
                "apiVersion": "apps/v1",
                "kind": "Deployment", 
                "name": f"{name}-base"
            },
            "minReplicas": hpa_config.get('minReplicas', 1),
            "maxReplicas": hpa_config.get('maxReplicas', 10),
            "metrics": [
                {
                    "type": "Resource",
                    "resource": {
                        "name": "memory",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": 80
                        }
                    }
                },
                {
                    "type": "Pods",
                    "pods": {
                        "metric": {
                            "name": "llm_requests_per_second"
                        },
                        "target": {
                            "type": "AverageValue",
                            "averageValue": "10"
                        }
                    }
                }
            ]
        }
    }
    
    logger.info(f"Created advanced HPA for {name}")
    return hpa_spec

@kopf.on.update('inference.llm-d.io', 'v1alpha1', 'advancedllmdeployments')
async def update_advanced_llm_deployment(spec, name, namespace, **kwargs):
    """Handle updates to AdvancedLLMDeployment resources."""
    logger.info(f"Updating AdvancedLLMDeployment: {name}")
    
    # Implement rolling updates, canary progression, etc.
    deployment_config = spec.get('deployment', {})
    
    if deployment_config.get('strategy') == 'Canary':
        await progress_canary_deployment(name, namespace, deployment_config)

async def progress_canary_deployment(name: str, namespace: str, config: Dict):
    """Progress canary deployment through defined steps."""
    
    canary_config = config.get('canary', {})
    steps = canary_config.get('steps', [])
    
    # Logic to progress through canary steps based on metrics/time
    logger.info(f"Progressing canary deployment for {name}")

if __name__ == "__main__":
    # Run the operator
    asyncio.run(kopf.run())
```

## Integration Patterns with ML Platforms

### Kubeflow Integration

```yaml
# Integration with Kubeflow Pipelines for end-to-end ML workflows
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  name: llm-training-to-deployment
  namespace: production
spec:
  entrypoint: llm-pipeline
  
  templates:
  - name: llm-pipeline
    dag:
      tasks:
      - name: prepare-data
        template: data-preparation
      
      - name: fine-tune-model
        template: fine-tuning
        dependencies: [prepare-data]
        arguments:
          parameters:
          - name: data-path
            value: "{{tasks.prepare-data.outputs.parameters.output-path}}"
      
      - name: validate-model
        template: model-validation
        dependencies: [fine-tune-model]
        arguments:
          parameters:
          - name: model-path
            value: "{{tasks.fine-tune-model.outputs.parameters.model-path}}"
      
      - name: deploy-model
        template: llm-deployment
        dependencies: [validate-model]
        when: "{{tasks.validate-model.outputs.parameters.quality-score}} > 0.85"
        arguments:
          parameters:
          - name: model-path
            value: "{{tasks.fine-tune-model.outputs.parameters.model-path}}"
  
  - name: data-preparation
    container:
      image: python:3.9
      command: [python]
      args: ["/scripts/prepare_data.py"]
      volumeMounts:
      - name: data-volume
        mountPath: /data
    outputs:
      parameters:
      - name: output-path
        value: "/data/processed"
  
  - name: fine-tuning
    inputs:
      parameters:
      - name: data-path
    container:
      image: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
      command: [python]
      args: 
      - "/scripts/fine_tune.py"
      - "--data-path={{inputs.parameters.data-path}}"
      - "--model-name=meta-llama/Llama-3.1-8B-Instruct"
      - "--output-dir=/models/fine-tuned"
      resources:
        requests:
          nvidia.com/gpu: "4"
          memory: "64Gi"
        limits:
          nvidia.com/gpu: "4"
          memory: "128Gi"
    outputs:
      parameters:
      - name: model-path
        value: "/models/fine-tuned"
  
  - name: model-validation
    inputs:
      parameters:
      - name: model-path
    container:
      image: llm-validation:latest
      command: [python]
      args:
      - "/scripts/validate_model.py"
      - "--model-path={{inputs.parameters.model-path}}"
      - "--benchmark-suite=comprehensive"
    outputs:
      parameters:
      - name: quality-score
        valueFrom:
          path: /tmp/quality_score.txt
  
  - name: llm-deployment
    inputs:
      parameters:
      - name: model-path
    resource:
      action: create
      manifest: |
        apiVersion: inference.llm-d.io/v1alpha1
        kind: LLMDeployment
        metadata:
          name: fine-tuned-llama-3.1-8b
          namespace: production
          labels:
            app.kubernetes.io/name: llm-d
            llm-d.ai/model: "llama-3.1"
            llm-d.ai/size: "8b"
            llm-d.ai/type: "fine-tuned"
        spec:
          model:
            name: "{{inputs.parameters.model-path}}"
            source: "local"
          resources:
            requests:
              memory: "16Gi"
              cpu: "4"
              nvidia.com/gpu: "1"
            limits:
              memory: "24Gi"
              cpu: "8"
              nvidia.com/gpu: "1"
          scaling:
            minReplicas: 2
            maxReplicas: 10
            targetCPUUtilizationPercentage: 70
```

### MLflow Integration

```python title="mlflow-llm-integration.py" showLineNumbers
#!/usr/bin/env python3
"""
Integration between MLflow and llm-d for model lifecycle management.
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import yaml
import subprocess
from typing import Dict, Optional
import os

class LLMModelDeployment:
    def __init__(self, mlflow_tracking_uri: str, k8s_namespace: str = "production"):
        """Initialize MLflow integration for LLM deployment."""
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.client = MlflowClient()
        self.namespace = k8s_namespace
    
    def register_model_version(self, model_name: str, model_path: str, 
                             metrics: Dict, tags: Dict = None) -> str:
        """Register a new model version in MLflow."""
        
        with mlflow.start_run() as run:
            # Log model artifacts
            mlflow.log_artifacts(model_path, "model")
            
            # Log metrics
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
            
            # Log tags
            if tags:
                mlflow.set_tags(tags)
            
            # Register model
            model_uri = f"runs:/{run.info.run_id}/model"
            model_version = mlflow.register_model(model_uri, model_name)
            
            return model_version.version
    
    def deploy_model_version(self, model_name: str, version: str, 
                           deployment_config: Dict = None) -> bool:
        """Deploy a specific model version using llm-d."""
        
        try:
            # Get model version details
            model_version = self.client.get_model_version(model_name, version)
            model_uri = model_version.source
            
            # Extract model path from MLflow
            model_path = self._download_model(model_uri)
            
            # Create LLMDeployment manifest
            deployment_manifest = self._create_deployment_manifest(
                model_name, version, model_path, deployment_config
            )
            
            # Apply to Kubernetes
            return self._apply_deployment(deployment_manifest)
            
        except Exception as e:
            print(f"Failed to deploy model {model_name} version {version}: {e}")
            return False
    
    def _download_model(self, model_uri: str) -> str:
        """Download model from MLflow to local storage."""
        local_path = f"/tmp/models/{model_uri.split('/')[-1]}"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download using MLflow
        mlflow.artifacts.download_artifacts(model_uri, dst_path=local_path)
        return local_path
    
    def _create_deployment_manifest(self, model_name: str, version: str,
                                  model_path: str, config: Dict = None) -> Dict:
        """Create Kubernetes deployment manifest."""
        
        config = config or {}
        
        # Standard naming following shared config
        deployment_name = f"{model_name.lower().replace('_', '-')}-v{version}"
        
        manifest = {
            "apiVersion": "inference.llm-d.io/v1alpha1",
            "kind": "LLMDeployment",
            "metadata": {
                "name": deployment_name,
                "namespace": self.namespace,
                "labels": {
                    "app.kubernetes.io/name": "llm-d",
                    "llm-d.ai/model": model_name.lower(),
                    "llm-d.ai/version": version,
                    "mlflow.org/model-name": model_name
                },
                "annotations": {
                    "mlflow.org/model-uri": f"models:/{model_name}/{version}",
                    "mlflow.org/run-id": self._get_run_id_for_version(model_name, version)
                }
            },
            "spec": {
                "model": {
                    "name": model_path,
                    "source": "local"
                },
                "resources": {
                    "requests": {
                        "memory": "16Gi",
                        "cpu": "4",
                        "nvidia.com/gpu": "1"
                    },
                    "limits": {
                        "memory": "24Gi", 
                        "cpu": "8",
                        "nvidia.com/gpu": "1"
                    }
                }
            }
        }
        
        # Apply configuration overrides
        if "resources" in config:
            manifest["spec"]["resources"].update(config["resources"])
        
        if "scaling" in config:
            manifest["spec"]["scaling"] = config["scaling"]
        
        return manifest
    
    def _get_run_id_for_version(self, model_name: str, version: str) -> str:
        """Get MLflow run ID for a model version."""
        model_version = self.client.get_model_version(model_name, version)
        return model_version.run_id
    
    def _apply_deployment(self, manifest: Dict) -> bool:
        """Apply deployment manifest to Kubernetes."""
        
        # Write manifest to temporary file
        manifest_file = "/tmp/deployment.yaml"
        with open(manifest_file, 'w') as f:
            yaml.dump(manifest, f)
        
        # Apply using kubectl
        result = subprocess.run(
            ["kubectl", "apply", "-f", manifest_file],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            print(f"Successfully deployed {manifest['metadata']['name']}")
            return True
        else:
            print(f"Failed to deploy: {result.stderr}")
            return False
    
    def promote_model_to_production(self, model_name: str, version: str) -> bool:
        """Promote a model version to production stage."""
        
        try:
            # Transition model version to Production stage
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production"
            )
            
            # Deploy to production namespace
            deployment_config = {
                "scaling": {
                    "minReplicas": 3,
                    "maxReplicas": 20
                }
            }
            
            return self.deploy_model_version(model_name, version, deployment_config)
            
        except Exception as e:
            print(f"Failed to promote model to production: {e}")
            return False

# Example usage
async def main():
    """Example MLflow integration workflow."""
    
    # Initialize deployment manager
    deployment = LLMModelDeployment(
        mlflow_tracking_uri="http://mlflow.example.com",
        k8s_namespace="production"
    )
    
    # Register new model version
    metrics = {
        "perplexity": 2.1,
        "bleu_score": 0.85,
        "inference_latency_p95": 150.0
    }
    
    tags = {
        "model_type": "fine-tuned",
        "base_model": "llama-3.1-8b",
        "training_dataset": "custom_domain_v2"
    }
    
    version = deployment.register_model_version(
        model_name="domain_specific_llama",
        model_path="/models/fine-tuned/",
        metrics=metrics,
        tags=tags
    )
    
    print(f"Registered model version: {version}")
    
    # Deploy to staging for testing
    staging_success = deployment.deploy_model_version(
        model_name="domain_specific_llama",
        version=version,
        deployment_config={"scaling": {"minReplicas": 1, "maxReplicas": 3}}
    )
    
    if staging_success:
        print("Model deployed to staging successfully")
        
        # After validation, promote to production
        production_success = deployment.promote_model_to_production(
            model_name="domain_specific_llama",
            version=version
        )
        
        if production_success:
            print("Model promoted to production successfully")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Best Practices

### Advanced Deployment Practices

- **Infrastructure as Code**: Use GitOps patterns for all advanced configurations
- **Progressive Deployment**: Always use canary or blue-green strategies for production
- **Cross-Cluster Consistency**: Maintain identical configurations across federated clusters
- **Resource Isolation**: Use dedicated node pools for different model tiers

### Integration Guidelines

- **Loose Coupling**: Integrate with external systems through well-defined APIs
- **Observability**: Implement comprehensive tracing across all integration points
- **Error Handling**: Design for graceful degradation when external systems fail
- **Security**: Apply principle of least privilege for all service-to-service communication

### Common Pitfalls

1. **Over-Engineering**: Start simple and add complexity only when needed
2. **Network Complexity**: Avoid unnecessary service mesh features that add latency
3. **Resource Contention**: Plan for peak load scenarios in multi-cluster setups
4. **Version Skew**: Maintain compatibility matrices for all integrated components

## Summary and Next Steps

### Key Takeaways

- ✅ **Multi-Cluster Federation**: Implemented global LLM deployment patterns with automated failover
- ✅ **Advanced Networking**: Configured sophisticated traffic management and security policies
- ✅ **Custom Operators**: Extended llm-d with enterprise-specific automation capabilities
- ✅ **ML Platform Integration**: Established seamless workflows with Kubeflow and MLflow
- ✅ **Production Readiness**: Applied enterprise-grade patterns for scalability and reliability

### Advanced Topics Integration

This chapter builds on the operational foundations from Chapters 5-8 and the MLOps workflows from Chapter 10. The advanced patterns established here enable enterprise-scale LLM deployments that can adapt to complex organizational requirements while maintaining the reliability and performance standards established in earlier chapters.

**Next Steps:**

- Apply these patterns to your specific enterprise requirements
- Implement monitoring and alerting for advanced deployment scenarios
- Review [MLOps Workflows](./10-mlops-workflows/index.md) for integration with CI/CD pipelines
- Reference [Shared Configuration](./appendix/shared-config.md) for standard specifications

---

:::info References

- [Kubernetes Multi-Cluster Management](https://kubernetes.io/docs/concepts/cluster-administration/cluster-administration-overview/)
- [Istio Advanced Traffic Management](https://istio.io/latest/docs/concepts/traffic-management/)
- [Kubeflow Pipelines Documentation](https://www.kubeflow.org/docs/components/pipelines/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Kopf Kubernetes Operator Framework](https://kopf.readthedocs.io/)
- [Shared Configuration Reference](./appendix/shared-config.md)

:::