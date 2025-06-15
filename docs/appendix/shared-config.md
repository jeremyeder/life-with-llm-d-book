---
title: Shared Configuration Reference
description: Standardized specifications and conventions used throughout the book
sidebar_position: 1
---

# Shared Configuration Reference

This appendix provides standardized specifications and naming conventions used consistently throughout all chapters. Reference this document when implementing examples or adapting configurations.

## Model Catalog

### Standard Model Specifications

| Model | Parameters | Memory (GB) | GPU Memory | Tensor Parallel | Use Case |
|-------|------------|-------------|------------|-----------------|----------|
| **Llama 3.1 8B** | 8B | 16 | 24GB | 1 | Development, testing |
| **Llama 3.1 70B** | 70B | 140 | 160GB (4x40GB) | 4 | Production inference |
| **Mistral 7B** | 7B | 14 | 16GB | 1 | Lightweight production |
| **CodeLlama 13B** | 13B | 26 | 32GB | 1-2 | Code generation |

### Resource Templates

#### Small Models (7B-8B)

```yaml
resources:
  requests:
    memory: "16Gi"
    nvidia.com/gpu: "1"
    cpu: "4"
  limits:
    memory: "24Gi"
    nvidia.com/gpu: "1"
    cpu: "8"
```

#### Large Models (70B+)

```yaml
resources:
  requests:
    memory: "160Gi"
    nvidia.com/gpu: "4"
    cpu: "16"
  limits:
    memory: "200Gi"
    nvidia.com/gpu: "4"
    cpu: "32"
spec:
  parallelism:
    tensor: 4
```

## Namespace Conventions

### Standard Namespaces

| Namespace | Purpose | Examples |
|-----------|---------|----------|
| `llm-d-system` | Core operator and system components | Operator, webhooks, CRDs |
| `production` | Production model deployments | Customer-facing services |
| `staging` | Pre-production testing | Integration testing, validation |
| `development` | Development and experimentation | Model testing, research |

### Environment-Specific Suffixes

- Development: `dev`
- Staging: `staging`
- Production: `production`

Example: `llama-8b-dev`, `llama-8b-staging`, `llama-8b-production`

## Service Naming Conventions

### LLMDeployment Resources

- Format: `{model}-{size}-{environment}`
- Examples: `llama-8b-dev`, `mistral-7b-production`

### Services and Endpoints

- Internal: `{model}-{size}-svc`
- External: `{model}-{size}-api`
- Examples: `llama-8b-svc`, `llama-8b-api`

## Port Standards

| Service Type | Port | Purpose |
|--------------|------|---------|
| HTTP API | 8080 | Primary inference endpoint |
| gRPC API | 9090 | High-performance inference |
| Metrics | 8081 | Prometheus metrics |
| Health | 8082 | Health and readiness checks |

## Configuration Labels

### Standard Labels

```yaml
metadata:
  labels:
    app.kubernetes.io/name: llm-d
    app.kubernetes.io/component: inference-server
    app.kubernetes.io/version: "v1.0.0"
    llm-d.ai/model: "llama-3.1"
    llm-d.ai/size: "8b"
    llm-d.ai/environment: "production"
```

### Selector Labels

```yaml
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: llm-d
      llm-d.ai/model: "llama-3.1"
      llm-d.ai/size: "8b"
```

## Storage Configurations

### Model Storage (S3-Compatible)

```yaml
modelRepository:
  storageUri: "s3://llm-models/llama-3.1-8b/"
  storageConfig:
    region: "us-west-2"
    endpoint: "https://s3.amazonaws.com"
```

### Persistent Volumes

```yaml
spec:
  storage:
    size: "100Gi"
    storageClass: "fast-ssd"
    accessMode: "ReadOnlyMany"
```

## Network Configuration

### Ingress Patterns

```yaml
spec:
  rules:
  - host: "llama-8b-api.example.com"
    http:
      paths:
      - path: "/v1"
        pathType: Prefix
        backend:
          service:
            name: "llama-8b-svc"
            port:
              number: 8080
```

### Service Mesh (Istio)

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: llama-8b-vs
spec:
  hosts:
  - "llama-8b-api.example.com"
  http:
  - route:
    - destination:
        host: llama-8b-svc
        port:
          number: 8080
```

## Monitoring Standards

### SLO Definitions

```yaml
slos:
  availability:
    target: 0.999
    window: "30d"
  latency_p95:
    target: "500ms"
    window: "5m"
  error_rate:
    target: 0.001
    window: "5m"
```

### Prometheus Labels

```yaml
labels:
  model: "llama-3.1"
  model_size: "8b"
  environment: "production"
  namespace: "production"
  version: "v1.0.0"
```

## Security Standards

### Pod Security Context

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

### Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: llm-inference-policy
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: llm-d
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-system
    ports:
    - protocol: TCP
      port: 8080
```

## Cross-Chapter References

### Chapter Dependencies

- **Chapter 2 (Architecture)** → Model specifications, resource templates
- **Chapter 3 (Installation)** → Namespace conventions, security standards  
- **Chapter 4 (Data Scientists)** → Development workflows, naming conventions
- **Chapter 5 (SRE Operations)** → Monitoring standards, SLO definitions
- **Chapter 6 (Performance)** → Resource templates, optimization baselines
- **Chapter 7 (Security)** → Security standards, network policies
- **Chapter 8 (Troubleshooting)** → All standards for consistent debugging
- **Chapter 10 (MLOps)** → CI/CD patterns, deployment automation

### Configuration File Locations

- Base configurations: `/docs/examples/base/`
- Environment overlays: `/docs/examples/overlays/{env}/`
- Monitoring configs: `/docs/examples/monitoring/`
- Security policies: `/docs/examples/security/`

## Version Compatibility

### llm-d Operator Versions

- **v1.0.x**: Initial release, basic inference
- **v1.1.x**: Multi-GPU support, advanced routing
- **v1.2.x**: A/B testing, canary deployments

### Kubernetes Compatibility

- **Minimum**: Kubernetes 1.25+
- **Recommended**: Kubernetes 1.28+
- **GPU Support**: NVIDIA GPU Operator 23.9.0+

## Usage Guidelines

1. **Always reference this appendix** when creating new configurations
2. **Update this document** when introducing new patterns
3. **Validate consistency** against these standards before commits
4. **Use standard labels** for all resources to enable proper monitoring and troubleshooting

This reference ensures consistency across all book examples and real-world implementations.
