---
title: Installation and Setup
description: Complete guide to installing and configuring llm-d on Kubernetes and OpenShift clusters
sidebar_position: 2
---

# Installation and Setup

:::info Chapter Overview
This chapter provides a comprehensive guide to installing llm-d on existing Kubernetes and OpenShift clusters. You'll learn the prerequisites, installation methods, initial configuration, and verification procedures to get your first LLM serving deployment running.
:::

## Prerequisites

Before installing llm-d, ensure your environment meets these requirements:

### Cluster Requirements

- ✅ **Kubernetes** 1.24+ or **OpenShift** 4.12+
- ✅ **GPU-enabled nodes** with NVIDIA Container Toolkit installed
- ✅ **Storage class** supporting persistent volumes (minimum 100GB available)
- ✅ **Ingress controller** or LoadBalancer capability
- ✅ **Cluster admin privileges** for CRD installation

### Hardware Specifications

**Minimum Requirements (Development/Testing):**

- 1x NVIDIA GPU (L4, T4, or similar)
- 16GB GPU memory
- 64GB system RAM
- 500GB persistent storage

**Recommended Production Setup:**

- 4x NVIDIA L40S GPUs (or equivalent)
- 48GB+ GPU memory per GPU
- 256GB+ system RAM
- 2TB+ persistent storage

### Required Tools

Install these tools on your local machine:

```bash
# Essential tools
kubectl       # Kubernetes CLI
helm          # Package manager
yq            # YAML processor
jq            # JSON processor
git           # Version control
kustomize     # Configuration management

# Tool installation (Ubuntu/Debian)
sudo apt update
sudo apt install -y kubectl helm git

# Install yq
sudo wget -qO /usr/local/bin/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64
sudo chmod +x /usr/local/bin/yq

# Install jq
sudo apt install -y jq

# Install kustomize
curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
sudo mv kustomize /usr/local/bin/
```

### Credentials and Access

**Hugging Face Token:**

1. Create account at [huggingface.co](https://huggingface.co)
2. Generate access token at [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. Ensure token has model download permissions

**Cluster Access:**

```bash
# Verify kubectl access
kubectl cluster-info
kubectl get nodes

# Verify GPU availability
kubectl describe nodes | grep -A 5 "nvidia.com/gpu"
```

## Installation Methods

llm-d supports multiple installation approaches. Choose the method that best fits your environment and experience level.

### Method 1: Automated Installation (Recommended)

The quickstart installer provides the fastest path to a working llm-d deployment.

#### Step 1: Clone the Repository

```bash
git clone https://github.com/llm-d/llm-d-deployer.git
cd llm-d-deployer/quickstart
```

#### Step 2: Install Dependencies

```bash
./install-deps.sh
```

This script installs required tools and validates your environment.

#### Step 3: Configure Environment

```bash
# Set your Hugging Face token
export HF_TOKEN="hf_your_token_here"

# Optional: Set custom namespace
export LLMD_NAMESPACE="llm-d"

# Optional: Set storage class
export STORAGE_CLASS="fast-ssd"
```

#### Step 4: Run Installation

```bash
# Basic installation with defaults
./llmd-installer.sh

# Advanced installation with custom options
./llmd-installer.sh \
  --namespace production \
  --storage-size 1Ti \
  --storage-class fast-ssd \
  --values-file custom-values.yaml
```

**Installation Flags:**

| Flag | Description | Default |
|------|-------------|---------|
| `-n, --namespace` | Kubernetes namespace | `llm-d` |
| `-z, --storage-size` | PVC storage size | `100Gi` |
| `-c, --storage-class` | Storage class name | Default cluster storage class |
| `-f, --values-file` | Custom Helm values | None |
| `-D, --download-model` | Pre-download model to PVC | False |
| `-m, --disable-metrics` | Disable Prometheus metrics | False |

### Method 2: Helm Chart Installation

For more control over the installation process, use Helm directly.

#### Step 1: Add Helm Repository

```bash
helm repo add llm-d https://helm.llm-d.ai
helm repo update
```

#### Step 2: Create Namespace

```bash
kubectl create namespace llm-d
```

#### Step 3: Install Helm Chart

```bash
# Basic installation
helm install llm-d llm-d/llm-d \
  --namespace llm-d \
  --set global.hfToken="${HF_TOKEN}"

# Production installation with custom values
helm install llm-d llm-d/llm-d \
  --namespace llm-d \
  --values production-values.yaml
```

### Method 3: GitOps/ArgoCD Installation

For GitOps workflows, use the provided Kustomize manifests.

#### Step 1: Fork Repository

```bash
git clone https://github.com/llm-d/llm-d-deployer.git
cd llm-d-deployer/gitops
```

#### Step 2: Customize Configuration

```bash
# Edit kustomization.yaml
vim kustomization.yaml

# Add custom patches
vim patches/custom-config.yaml
```

#### Step 3: Apply with ArgoCD

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: llm-d
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/your-org/llm-d-deployer.git
    targetRevision: main
    path: gitops
  destination:
    server: https://kubernetes.default.svc
    namespace: llm-d
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

## Configuration

### Basic Configuration

The default configuration provides a working setup for most environments:

```yaml
# basic-config.yaml
global:
  namespace: llm-d
  storageClass: "fast-ssd"
  
sampleApplication:
  model:
    modelArtifactURI: "hf://meta-llama/Llama-3.2-1B-Instruct"
    modelName: "llama3-1b"
    baseConfigMapRefName: "basic-gpu-with-nixl-and-redis-lookup-preset"
  
  prefill:
    replicas: 1
    resources:
      limits:
        nvidia.com/gpu: 1
        memory: "24Gi"
        cpu: "8"
      requests:
        nvidia.com/gpu: 1
        memory: "16Gi"
        cpu: "4"
        memory: "16Gi"
  
  decode:
    replicas: 1
    resources:
      limits:
        nvidia.com/gpu: 1
        memory: "24Gi"
        cpu: "8"
      requests:
        nvidia.com/gpu: 1
        memory: "16Gi"
        cpu: "4"
        memory: "16Gi"

monitoring:
  enabled: true
  prometheus:
    enabled: true
  grafana:
    enabled: true
```

### Advanced Configuration

For production deployments, customize resource allocation, scaling, and monitoring:

```yaml
# production-config.yaml
global:
  namespace: production
  storageClass: "nvme-ssd"
  
sampleApplication:
  model:
    modelArtifactURI: "hf://meta-llama/Llama-3.1-70B-Instruct"
    modelName: "llama3-70b"
    baseConfigMapRefName: "production-gpu-with-advanced-caching-preset"
  
  prefill:
    replicas: 2
    resources:
      limits:
        nvidia.com/gpu: 2
        memory: "64Gi"
      requests:
        nvidia.com/gpu: 2
        memory: "64Gi"
    
    autoscaling:
      enabled: true
      minReplicas: 2
      maxReplicas: 8
      targetCPUUtilizationPercentage: 70
  
  decode:
    replicas: 4
    resources:
      limits:
        nvidia.com/gpu: 1
        memory: "32Gi"
      requests:
        nvidia.com/gpu: 1
        memory: "32Gi"
    
    autoscaling:
      enabled: true
      minReplicas: 4
      maxReplicas: 16
      targetCPUUtilizationPercentage: 80

kvCache:
  redis:
    enabled: true
    replicas: 3
    persistence:
      enabled: true
      size: "100Gi"
      storageClass: "fast-ssd"

monitoring:
  enabled: true
  prometheus:
    enabled: true
    retention: "30d"
    storageSize: "50Gi"
  grafana:
    enabled: true
    persistence:
      enabled: true
      size: "10Gi"
  alerts:
    enabled: true
    webhookUrl: "https://hooks.slack.com/your-webhook"

networking:
  ingress:
    enabled: true
    className: "nginx"
    annotations:
      cert-manager.io/cluster-issuer: "letsencrypt-prod"
    tls:
      enabled: true
      secretName: "llm-d-tls"
```

## Verification and Testing

### Installation Verification

After installation completes, verify all components are running:

```bash
# Check namespace creation
kubectl get namespace llm-d

# Verify all pods are running
kubectl get pods -n llm-d

# Check persistent volume claims
kubectl get pvc -n llm-d

# Verify services are exposed
kubectl get services -n llm-d

# Check ingress (if enabled)
kubectl get ingress -n llm-d
```

Expected output:

```bash
$ kubectl get pods -n llm-d
NAME                                    READY   STATUS    RESTARTS   AGE
llm-d-prefill-0                        1/1     Running   0          5m
llm-d-decode-0                         1/1     Running   0          5m
llm-d-inference-scheduler-xxx          1/1     Running   0          5m
llm-d-kv-cache-manager-xxx             1/1     Running   0          5m
prometheus-server-xxx                  1/1     Running   0          5m
grafana-xxx                           1/1     Running   0          5m
```

### Health Checks

Verify component health using built-in health endpoints:

```bash
# Check inference scheduler health
kubectl port-forward svc/llm-d-inference-scheduler 8080:8080 -n llm-d
curl http://localhost:8080/health

# Check model service health
kubectl port-forward svc/llm-d-model-service 8000:8000 -n llm-d
curl http://localhost:8000/v1/models
```

### First Inference Request

Test the deployment with a simple inference request:

```bash
# Port-forward to model service
kubectl port-forward svc/llm-d-model-service 8000:8000 -n llm-d &

# Test inference endpoint
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3-1b",
    "messages": [
      {
        "role": "user", 
        "content": "Hello! How are you today?"
      }
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

Expected response:

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1703123456,
  "model": "llama3-1b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm doing well, thank you for asking. I'm here and ready to help you with any questions or tasks you might have. How are you doing today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 35,
    "total_tokens": 47
  }
}
```

## Monitoring and Observability

### Prometheus Metrics

Access Prometheus to monitor system performance:

```bash
# Port-forward to Prometheus
kubectl port-forward svc/prometheus-server 9090:80 -n llm-d

# Open http://localhost:9090 in browser
```

Key metrics to monitor:

- `llm_d_inference_requests_total` - Total inference requests
- `llm_d_inference_duration_seconds` - Request latency
- `llm_d_gpu_memory_usage_bytes` - GPU memory utilization
- `llm_d_cache_hit_ratio` - KV cache efficiency

### Grafana Dashboards

Access pre-configured Grafana dashboards:

```bash
# Port-forward to Grafana
kubectl port-forward svc/grafana 3000:80 -n llm-d

# Open http://localhost:3000 in browser
# Default credentials: admin/admin
```

Pre-configured dashboards include:

- **LLM-D Overview**: High-level system metrics
- **Model Performance**: Per-model latency and throughput
- **Resource Utilization**: GPU, CPU, and memory usage
- **Cache Performance**: KV cache hit rates and efficiency

## Troubleshooting

### Common Installation Issues

**Issue**: Pods stuck in `Pending` state

```bash
# Diagnosis
kubectl describe pod <pod-name> -n llm-d

# Common causes:
# - Insufficient GPU resources
# - Storage class not available
# - Node selectors not matching
```

**Solution**: Verify node resources and labels

```bash
# Check GPU availability
kubectl describe nodes | grep -A 10 "nvidia.com/gpu"

# Check storage classes
kubectl get storageclass

# Verify node labels
kubectl get nodes --show-labels
```

**Issue**: `ImagePullBackOff` errors

```bash
# Diagnosis
kubectl describe pod <pod-name> -n llm-d

# Common causes:
# - Invalid Hugging Face token
# - Network connectivity issues
# - Model access permissions
```

**Solution**: Verify credentials and connectivity

```bash
# Test Hugging Face token
curl -H "Authorization: Bearer ${HF_TOKEN}" \
  https://huggingface.co/api/whoami

# Check network policies
kubectl get networkpolicies -n llm-d
```

### Log Analysis

Collect logs for troubleshooting:

```bash
# Get all logs from llm-d namespace
kubectl logs -l app.kubernetes.io/name=llm-d --all-containers=true -n llm-d

# Get specific component logs
kubectl logs -l llm-d.ai/role=prefill --all-containers=true -n llm-d
kubectl logs -l llm-d.ai/role=decode --all-containers=true -n llm-d

# Stream logs in real-time
kubectl logs -f deployment/llm-d-inference-scheduler -n llm-d
```

### Performance Troubleshooting

**High Latency Issues:**

```bash
# Check GPU utilization
kubectl exec -it <prefill-pod> -n llm-d -- nvidia-smi

# Monitor cache hit rates
kubectl port-forward svc/prometheus-server 9090:80 -n llm-d
# Query: rate(llm_d_cache_hits_total[5m]) / rate(llm_d_cache_requests_total[5m])
```

**Resource Exhaustion:**

```bash
# Check resource usage
kubectl top pods -n llm-d
kubectl top nodes

# Review resource requests vs limits
kubectl describe pods -n llm-d | grep -A 5 "Requests:\|Limits:"
```

## Next Steps

With llm-d successfully installed and verified, you're ready to:

1. **Deploy your first production model** (Chapter 3)
2. **Configure advanced features** like autoscaling and caching
3. **Set up monitoring and alerting** for production use
4. **Explore the Data Scientist workflow** (Chapter 4)
5. **Learn SRE operational procedures** (Chapter 5)

:::tip Pro Tip
Save your working configuration as a template for future deployments. Store custom values files in version control for reproducible installations.
:::

## Summary

This chapter covered:

- Complete prerequisites and environment preparation
- Multiple installation methods (automated, Helm, GitOps)
- Basic and advanced configuration options
- Comprehensive verification and testing procedures
- Monitoring setup with Prometheus and Grafana
- Common troubleshooting scenarios and solutions

You now have a fully functional llm-d installation ready for production LLM workloads.

---

:::info References

- [llm-d-deployer Repository](https://github.com/llm-d/llm-d-deployer)
- [llm-d Main Repository](https://github.com/llm-d/llm-d)
- [Kubernetes GPU Support](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
- [Helm Documentation](https://helm.sh/docs/)
- [Shared Configuration Reference](./appendix/shared-config.md)

:::
