---
title: CRD Reference
description: Complete API reference for llm-d Custom Resource Definitions with practical examples
sidebar_position: 1
---

# Appendix A: CRD Reference

This appendix provides a comprehensive reference for the core llm-d Custom Resource Definitions (CRDs). Each CRD includes the most commonly used fields, practical examples, and cross-references to relevant chapters.

## LLMDeployment

The primary CRD for deploying and managing LLM inference services.

### Core Fields Reference

| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `spec.model.modelUri` | string | Yes | Hugging Face model identifier or custom URI | `"hf://meta-llama/Llama-3.1-8B-Instruct"` |
| `spec.model.version` | string | No | Model version tag for reproducibility | `"v1.2.0"` |
| `spec.model.quantization.type` | string | No | Quantization format | `"int8"`, `"fp16"`, `"int4"` |
| `spec.resources.requests` | object | Yes | Minimum resource requirements | See examples below |
| `spec.resources.limits` | object | Yes | Maximum resource limits | See examples below |
| `spec.autoscaling.enabled` | boolean | No | Enable horizontal pod autoscaling | `true` |
| `spec.autoscaling.minReplicas` | integer | No | Minimum number of replicas | `1` |
| `spec.autoscaling.maxReplicas` | integer | No | Maximum number of replicas | `10` |
| `spec.scheduling.scheduler` | string | No | Scheduler to use | `"llm-d-inference-scheduler"` |

### Basic Example

```yaml
apiVersion: inference.llm-d.io/v1alpha1
kind: LLMDeployment
metadata:
  name: llama-3.1-8b
  namespace: production
spec:
  model:
    modelUri: "hf://meta-llama/Llama-3.1-8B-Instruct"
    quantization:
      type: "int8"
  resources:
    requests:
      memory: "16Gi"
      cpu: "4"
      nvidia.com/gpu: "1"
    limits:
      memory: "24Gi"
      cpu: "8"
      nvidia.com/gpu: "1"
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 8
    targetGPUUtilization: 70
```

### Production Example with Monitoring

```yaml
apiVersion: inference.llm-d.io/v1alpha1
kind: LLMDeployment
metadata:
  name: llama-3.1-8b-prod
  namespace: production
  labels:
    app.kubernetes.io/name: llm-d
    llm-d.ai/model: "llama-3.1"
    llm-d.ai/size: "8b"
    llm-d.ai/environment: "production"
spec:
  model:
    modelUri: "hf://meta-llama/Llama-3.1-8B-Instruct"
    version: "v1.2.0"
    quantization:
      type: "int8"
  resources:
    requests:
      memory: "32Gi"
      cpu: "8"
      nvidia.com/gpu: "2"
    limits:
      memory: "48Gi"
      cpu: "16"
      nvidia.com/gpu: "2"
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetGPUUtilization: 60
  scheduling:
    scheduler: "llm-d-inference-scheduler"
    sloPolicy:
      enabled: true
      objectives:
        - name: "request_latency_p95"
          target: "500ms"
          weight: 0.6
        - name: "cost_per_request"
          target: "$0.001"
          weight: 0.4
  monitoring:
    enabled: true
    metrics:
      port: 8081
      path: "/metrics"
    alerting:
      enabled: true
      channels: ["#sre-alerts", "#llm-team"]
```

### Validation Rules

Common validation errors and solutions:

| Error | Cause | Solution |
|-------|-------|----------|
| `Invalid quantization type` | Unsupported quantization format | Use `"fp16"`, `"int8"`, or `"int4"` |
| `Insufficient GPU memory` | Model size exceeds GPU capacity | Increase GPU memory or use quantization |
| `Invalid model URI` | Malformed model identifier | Use format `"hf://namespace/model-name"` |

**Cross-references:**

- Chapter 2: [Installation and Setup](../02-installation-setup.md#llmdeployment-basics)
- Chapter 4: [Data Scientist Workflows](../04-data-scientist-workflows.md#model-deployment)
- Chapter 11: [Cost Optimization](../11-cost-optimization.md#quantization-configuration)
- Chapter 12: [MLOps for SREs](../12-mlops-for-sres.md#production-deployment)

## InferenceScheduler

Advanced scheduler for SLO-driven optimization and cost management.

### Core Fields Reference

| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `spec.sloPolicy.enabled` | boolean | No | Enable SLO-based scheduling | `true` |
| `spec.sloPolicy.objectives` | array | No | List of SLO objectives | See examples below |
| `spec.costPolicy.enabled` | boolean | No | Enable cost optimization | `true` |
| `spec.costPolicy.budget` | string | No | Cost budget constraints | `"$1000/month"` |
| `spec.nodeSelector` | object | No | Node selection criteria | `{"node-type": "gpu"}` |
| `spec.tolerations` | array | No | Node tolerations | See examples below |

### SLO-Driven Example

```yaml
apiVersion: inference.llm-d.io/v1alpha1
kind: InferenceScheduler
metadata:
  name: cost-aware-scheduler
  namespace: llm-d-system
spec:
  sloPolicy:
    enabled: true
    objectives:
      - name: "request_latency_p95"
        target: "500ms"
        weight: 0.4
        window: "5m"
      - name: "availability"
        target: "99.9%"
        weight: 0.3
        window: "30m"
      - name: "cost_per_request"
        target: "$0.001"
        weight: 0.3
        window: "1h"
  costPolicy:
    enabled: true
    budget: "$5000/month"
    spotInstancePreference: 0.7  # Prefer spot instances
    scaleDownDelay: "5m"
  nodeSelector:
    node-type: "gpu"
    instance-family: "a100"
  tolerations:
  - key: "spot-instance"
    operator: "Equal"
    value: "true"
    effect: "NoSchedule"
```

### Cost Optimization Example

```yaml
apiVersion: inference.llm-d.io/v1alpha1
kind: InferenceScheduler
metadata:
  name: cost-optimizer
  namespace: llm-d-system
spec:
  costPolicy:
    enabled: true
    budget: "$10000/month"
    spotInstancePreference: 0.8
    scaleDownDelay: "10m"
    costThresholds:
      warning: "$8000/month"
      critical: "$9500/month"
  scheduling:
    strategy: "cost-aware"
    nodePreferences:
      - nodeType: "spot"
        weight: 80
      - nodeType: "on-demand"
        weight: 20
  monitoring:
    costTracking:
      enabled: true
      interval: "1m"
      alertThresholds:
        - type: "budget_exceeded"
          value: "90%"
```

**Cross-references:**
- Chapter 6: [Performance Optimization](../06-performance-optimization.md#scheduler-configuration)
- Chapter 11: [Cost Optimization](../11-cost-optimization.md#slo-driven-scaling)
- Chapter 12: [MLOps for SREs](../12-mlops-for-sres.md#scheduling-integration)

## ModelService

Service definition for model inference endpoints with load balancing and routing.

### Core Fields Reference

| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `spec.model.reference` | string | Yes | Reference to LLMDeployment | `"llama-3.1-8b"` |
| `spec.routing.strategy` | string | No | Traffic routing strategy | `"round-robin"`, `"least-connections"` |
| `spec.loadBalancing.enabled` | boolean | No | Enable load balancing | `true` |
| `spec.endpoints.http.port` | integer | No | HTTP service port | `8080` |
| `spec.endpoints.grpc.port` | integer | No | gRPC service port | `9090` |
| `spec.security.tls.enabled` | boolean | No | Enable TLS termination | `true` |

### Load Balancing Example

```yaml
apiVersion: inference.llm-d.io/v1alpha1
kind: ModelService
metadata:
  name: llama-3.1-8b-service
  namespace: production
spec:
  model:
    reference: "llama-3.1-8b-prod"
  routing:
    strategy: "least-connections"
    healthCheck:
      enabled: true
      path: "/health"
      interval: "30s"
      timeout: "5s"
  loadBalancing:
    enabled: true
    algorithm: "weighted-round-robin"
    sessionAffinity: false
  endpoints:
    http:
      port: 8080
      path: "/v1"
    grpc:
      port: 9090
    metrics:
      port: 8081
      path: "/metrics"
  security:
    tls:
      enabled: true
      secretName: "llm-service-tls"
    authentication:
      enabled: true
      type: "bearer-token"
```

### Multi-Model Routing Example

```yaml
apiVersion: inference.llm-d.io/v1alpha1
kind: ModelService
metadata:
  name: multi-model-router
  namespace: production
spec:
  routing:
    strategy: "content-based"
    rules:
      - condition: "request.complexity == 'simple'"
        target: "llama-3.1-8b"
        weight: 100
      - condition: "request.complexity == 'complex'"
        target: "llama-3.1-70b"
        weight: 100
    fallback:
      target: "llama-3.1-8b"
  loadBalancing:
    enabled: true
    healthCheck:
      enabled: true
      successThreshold: 2
      failureThreshold: 3
  monitoring:
    requestTracing:
      enabled: true
      samplingRate: 0.1
    metrics:
      detailed: true
```

**Cross-references:**
- Chapter 3: [Understanding the Architecture](../03-understanding-architecture.md#modelservice-components)
- Chapter 5: [SRE Operations](../05-sre-operations.md#service-configuration)
- Chapter 7: [Security and Compliance](../07-security-compliance.md#service-security)

## ResourceProfile

Resource allocation profiles for standardized deployment configurations.

### Core Fields Reference

| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `spec.profile.name` | string | Yes | Profile identifier | `"small"`, `"medium"`, `"large"` |
| `spec.resources.gpu` | object | Yes | GPU resource specifications | See examples below |
| `spec.resources.memory` | string | Yes | Memory allocation | `"16Gi"` |
| `spec.resources.cpu` | string | Yes | CPU allocation | `"4"` |
| `spec.nodeConstraints` | object | No | Node selection constraints | See examples below |

### Standard Profiles Example

```yaml
apiVersion: inference.llm-d.io/v1alpha1
kind: ResourceProfile
metadata:
  name: small-model-profile
  namespace: llm-d-system
spec:
  profile:
    name: "small"
    description: "For 7B-8B parameter models"
    modelSizeRange: "7B-8B"
  resources:
    gpu:
      count: 1
      memory: "24GB"
      type: "nvidia-a100-40gb"
    memory: "16Gi"
    cpu: "4"
    storage: "100Gi"
  nodeConstraints:
    nodeType: "gpu"
    minimumNodes: 1
    preferredZones: ["us-west-2a", "us-west-2b"]
  costEstimate:
    hourly: "$1.80"
    monthly: "$1314"
---
apiVersion: inference.llm-d.io/v1alpha1
kind: ResourceProfile
metadata:
  name: large-model-profile
  namespace: llm-d-system
spec:
  profile:
    name: "large"
    description: "For 70B+ parameter models"
    modelSizeRange: "70B+"
  resources:
    gpu:
      count: 4
      memory: "80GB"
      type: "nvidia-a100-80gb"
    memory: "160Gi"
    cpu: "32"
    storage: "500Gi"
  nodeConstraints:
    nodeType: "gpu"
    minimumNodes: 1
    preferredZones: ["us-west-2a"]
  costEstimate:
    hourly: "$16.80"
    monthly: "$12264"
```

### Development Profile Example

```yaml
apiVersion: inference.llm-d.io/v1alpha1
kind: ResourceProfile
metadata:
  name: development-profile
  namespace: llm-d-system
spec:
  profile:
    name: "development"
    description: "Cost-optimized for development and testing"
    environment: "non-production"
  resources:
    gpu:
      count: 1
      memory: "16GB"
      type: "nvidia-t4"
    memory: "8Gi"
    cpu: "2"
    storage: "50Gi"
  nodeConstraints:
    nodeType: "gpu"
    spotInstancesAllowed: true
    preemptible: true
  autoscaling:
    scaleToZero: true
    scaleUpDelay: "2m"
    scaleDownDelay: "30s"
  costEstimate:
    hourly: "$0.35"
    monthly: "$255"
```

**Cross-references:**
- Chapter 4: [Data Scientist Workflows](../04-data-scientist-workflows.md#resource-profiles)
- Chapter 6: [Performance Optimization](../06-performance-optimization.md#resource-optimization)
- Chapter 11: [Cost Optimization](../11-cost-optimization.md#resource-planning)

## Common Configuration Patterns

### Environment-Specific Labeling

Apply consistent labels across all resources:

```yaml
metadata:
  labels:
    app.kubernetes.io/name: llm-d
    app.kubernetes.io/component: inference-server
    app.kubernetes.io/version: "v1.0.0"
    llm-d.ai/model: "llama-3.1"
    llm-d.ai/size: "8b"
    llm-d.ai/environment: "production"
    llm-d.ai/team: "ml-platform"
```

### Resource Specifications by Model Size

| Model Size | GPU Count | GPU Memory | System Memory | CPU Cores |
|------------|-----------|------------|---------------|-----------|
| 7B-8B | 1 | 24GB | 16-24Gi | 4-8 |
| 13B | 1-2 | 32GB | 24-32Gi | 8-12 |
| 70B | 4 | 80GB each | 160-200Gi | 16-32 |

### Security Context Template

Standard security context for all llm-d resources:

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 2000
  seccompProfile:
    type: RuntimeDefault
  capabilities:
    drop:
    - ALL
```

## Validation and Troubleshooting

### Common CRD Validation Issues

| Issue | Error Message | Solution |
|-------|---------------|----------|
| Invalid API version | `no matches for kind "LLMDeployment"` | Ensure CRDs are installed |
| Missing required fields | `spec.model.modelUri: Required value` | Add required model URI |
| Resource conflicts | `insufficient quota` | Check resource quotas and limits |
| Scheduler not found | `scheduler "llm-d-inference-scheduler" not found` | Install inference scheduler |

### Debugging Commands

```bash
# Check CRD installation
kubectl get crds | grep llm-d

# Validate resource configuration
kubectl apply --dry-run=client -f deployment.yaml

# Check resource status
kubectl describe llmdeployment llama-3.1-8b -n production

# View resource events
kubectl get events -n production --field-selector involvedObject.name=llama-3.1-8b
```

**Cross-references:**
- Chapter 8: [Troubleshooting Guide](../08-troubleshooting/02-common-issues.md)
- Chapter 12: [MLOps for SREs](../12-mlops-for-sres.md#debugging-deployments)

---

:::info Next Steps
- Review [Command Reference](./command-reference.md) for practical kubectl operations
- Check [Configuration Templates](./configuration-templates.md) for ready-to-use examples
:::
