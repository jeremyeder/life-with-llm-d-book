---
title: Error Pattern Reference
description: Comprehensive reference for error patterns and their solutions
sidebar_position: 5
---

# Error Pattern Reference

This reference provides a comprehensive catalog of error patterns commonly encountered in llm-d deployments, their meanings, and solutions.

## Error Categories

### Deployment Errors

#### DEPLOYMENT001: CRD Not Found

```bash
error validating data: ValidationError(LLMDeployment): unknown field "spec.model.name"
```

**Cause**: Outdated or missing CRD definitions

**Solution**:

```bash
# Update CRDs
kubectl apply -f https://github.com/llm-d/llm-d/releases/latest/download/crds.yaml

# Verify installation
kubectl get crd llmdeployments.inference.llm-d.io
```

#### DEPLOYMENT002: Resource Quota Exceeded

```bash
Error creating: pods "llm-model-5d4b8c" is forbidden: exceeded quota: gpu-quota, requested: nvidia.com/gpu=1, used: nvidia.com/gpu=8, limited: nvidia.com/gpu=8
```

**Cause**: Namespace resource quota limits reached

**Solution**:

```bash
# Check quota
kubectl describe resourcequota -n <namespace>

# Increase quota or scale down other deployments
kubectl edit resourcequota gpu-quota -n <namespace>
```

#### DEPLOYMENT003: Image Pull Failed

```bash
Failed to pull image "registry.example.com/models/llama2:latest": rpc error: code = Unknown desc = failed to pull and unpack image: failed to resolve reference: pulling from host registry.example.com failed with status code [manifests latest]: 401 Unauthorized
```

**Cause**: Missing or invalid image pull credentials

**Solution**:

```bash
# Create pull secret
kubectl create secret docker-registry regcred \
  --docker-server=registry.example.com \
  --docker-username=<user> \
  --docker-password=<pass> \
  -n <namespace>

# Update deployment
kubectl patch llmdeployment <name> -n <namespace> \
  --type='json' -p='[{"op": "add", "path": "/spec/imagePullSecrets", "value": [{"name": "regcred"}]}]'
```

### Runtime Errors

#### RUNTIME001: CUDA Out of Memory

```python
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 15.78 GiB total capacity; 14.50 GiB already allocated; 256.00 MiB free; 14.75 GiB reserved in total by PyTorch)
```

**Cause**: Model size exceeds GPU memory

**Solutions**:

```yaml
# 1. Enable quantization
spec:
  model:
    quantization:
      enabled: true
      precision: "int8"

# 2. Reduce batch size
spec:
  serving:
    batchSize: 1
    maxConcurrency: 2

# 3. Use larger GPU
spec:
  nodeSelector:
    nvidia.com/gpu.product: "NVIDIA-A100-SXM4-80GB"
```

#### RUNTIME002: Model Loading Timeout

```bash
Error: Model loading exceeded timeout of 300s
Readiness probe failed: Get "http://10.244.2.45:8080/health": context deadline exceeded
```

**Cause**: Large model taking too long to load

**Solution**:

```yaml
spec:
  # Increase timeouts
  readinessProbe:
    initialDelaySeconds: 600  # 10 minutes
    timeoutSeconds: 10
    periodSeconds: 30
  
  # Enable faster loading
  model:
    loading:
      useSharedMemory: true
      numWorkers: 8
```

#### RUNTIME003: Tensor Dimension Mismatch

```python
RuntimeError: Expected tensor for argument #1 'indices' to have scalar type Long; but got CUDAType instead
```

**Cause**: Data type mismatch in model inputs

**Solution**:

```python
# Ensure correct data types
input_ids = input_ids.long()  # Convert to Long tensor
attention_mask = attention_mask.long()

# Or in preprocessing
def preprocess_inputs(inputs):
    return {
        'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
        'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long)
    }
```

### Network Errors

#### NETWORK001: Connection Refused

```bash
curl: (7) Failed to connect to model-service port 8080: Connection refused
```

**Cause**: Service not running or wrong port

**Solution**:

```bash
# Check service and endpoints
kubectl get svc,endpoints model-service -n <namespace>

# Verify pod is running and ready
kubectl get pods -n <namespace> -l app=model

# Check container port
kubectl get pod <pod> -n <namespace> -o jsonpath='{.spec.containers[*].ports[*].containerPort}'
```

#### NETWORK002: DNS Resolution Failed

```bash
nslookup: can't resolve 'model-service.default.svc.cluster.local'
```

**Cause**: CoreDNS issues or service doesn't exist

**Solution**:

```bash
# Check CoreDNS
kubectl get pods -n kube-system -l k8s-app=kube-dns

# Restart CoreDNS if needed
kubectl rollout restart deployment coredns -n kube-system

# Verify service exists
kubectl get svc -A | grep model-service
```

#### NETWORK003: SSL/TLS Certificate Error

```bash
x509: certificate signed by unknown authority
```

**Cause**: Self-signed certificate or missing CA

**Solution**:

```yaml
# Add CA certificate to deployment
spec:
  volumes:
  - name: ca-certs
    configMap:
      name: ca-bundle
  containers:
  - name: inference
    volumeMounts:
    - name: ca-certs
      mountPath: /etc/ssl/certs/ca-certificates.crt
      subPath: ca-bundle.crt
```

### GPU Errors

#### GPU001: Driver Version Mismatch

```bash
CUDA driver version is insufficient for CUDA runtime version
```

**Cause**: Incompatible CUDA/driver versions

**Solution**:

```bash
# Check versions
nvidia-smi  # Driver version
nvcc --version  # CUDA version

# Update driver or use compatible image
spec:
  containers:
  - name: inference
    image: nvcr.io/nvidia/pytorch:23.10-py3  # Specific CUDA version
```

#### GPU002: Device Not Found

```bash
RuntimeError: No CUDA GPUs are available
```

**Cause**: GPU not allocated or not visible

**Solution**:

```yaml
# Ensure GPU request
spec:
  resources:
    limits:
      nvidia.com/gpu: "1"
    requests:
      nvidia.com/gpu: "1"
  
  # Check GPU device plugin
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```

#### GPU003: NCCL Communication Error

```bash
NCCL WARN Cuda failure 'out of memory'
NCCL WARN transport/net.cc:102 NCCL call failed ret -2 (Internal error)
```

**Cause**: Multi-GPU communication issues

**Solution**:

```bash
# Set NCCL environment variables
env:
- name: NCCL_DEBUG
  value: "INFO"
- name: NCCL_SOCKET_IFNAME
  value: "eth0"
- name: NCCL_IB_DISABLE
  value: "1"  # Disable InfiniBand if not available
```

### Memory Errors

#### MEMORY001: OOM Killed

```bash
Last State: Terminated
  Reason: OOMKilled
  Exit Code: 137
```

**Cause**: Container exceeded memory limit

**Solution**:

```yaml
# Increase memory limits
spec:
  resources:
    requests:
      memory: "32Gi"
    limits:
      memory: "48Gi"  # 50% buffer
  
  # Add memory monitoring
  env:
  - name: PYTORCH_CUDA_ALLOC_CONF
    value: "max_split_size_mb:512"
```

#### MEMORY002: Shared Memory Full

```bash
OSError: [Errno 28] No space left on device: '/dev/shm/...'
```

**Cause**: Insufficient shared memory for data loading

**Solution**:

```yaml
spec:
  volumes:
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 8Gi
  containers:
  - name: inference
    volumeMounts:
    - name: dshm
      mountPath: /dev/shm
```

### Storage Errors

#### STORAGE001: PVC Pending

```bash
persistentvolumeclaim "model-storage" is pending
Warning  ProvisioningFailed  Failed to provision volume: "StorageClass not found"
```

**Cause**: StorageClass doesn't exist or can't provision

**Solution**:

```bash
# Check available storage classes
kubectl get storageclass

# Create PVC with valid class
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-storage
spec:
  accessModes:
  - ReadWriteOnce
  storageClassName: gp3  # Use available class
  resources:
    requests:
      storage: 100Gi
```

#### STORAGE002: Mount Failed

```bash
MountVolume.MountDevice failed for volume "pvc-123": rpc error: code = Internal desc = mount failed: exit status 32
```

**Cause**: Volume already mounted elsewhere or filesystem issues

**Solution**:

```bash
# Check volume attachments
kubectl get volumeattachments

# Force detach if stuck
kubectl delete volumeattachment <name>

# Use ReadWriteMany if needed
accessModes:
- ReadWriteMany
```

### Authentication/Authorization Errors

#### AUTH001: RBAC Denied

```bash
Error from server (Forbidden): llmdeployments.inference.llm-d.io is forbidden: User "system:serviceaccount:default:default" cannot get resource "llmdeployments"
```

**Cause**: Missing RBAC permissions

**Solution**:

```yaml
# Create RBAC rules
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: llm-operator
rules:
- apiGroups: ["inference.llm-d.io"]
  resources: ["llmdeployments"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: llm-operator
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: llm-operator
subjects:
- kind: ServiceAccount
  name: default
  namespace: default
```

#### AUTH002: Registry Authentication Failed

```bash
Error response from daemon: Get https://registry.example.com/v2/: unauthorized: authentication required
```

**Cause**: Invalid registry credentials

**Solution**:

```bash
# Test credentials
docker login registry.example.com

# Recreate secret
kubectl delete secret regcred -n <namespace>
kubectl create secret docker-registry regcred \
  --docker-server=registry.example.com \
  --docker-username=<user> \
  --docker-password=<token> \
  -n <namespace>
```

## Error Handling Best Practices

### Structured Error Logging

```python
import logging
import json
from datetime import datetime

class ErrorLogger:
    def __init__(self):
        self.logger = logging.getLogger('llm-error')
        
    def log_error(self, error_code: str, error_type: str, 
                  message: str, context: dict = None):
        """Log structured error information"""
        error_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'error_code': error_code,
            'error_type': error_type,
            'message': message,
            'context': context or {}
        }
        
        self.logger.error(json.dumps(error_record))
        
        # Send to monitoring
        self.send_to_monitoring(error_record)
    
    def send_to_monitoring(self, error_record):
        """Send error to monitoring system"""
        # Implementation depends on monitoring stack
        pass
```

### Error Recovery Strategies

```python
import asyncio
from typing import Callable, Any
import backoff

class ErrorRecovery:
    @staticmethod
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=5,
        max_time=300
    )
    async def with_retry(func: Callable, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry"""
        return await func(*args, **kwargs)
    
    @staticmethod
    async def with_fallback(primary: Callable, fallback: Callable, 
                           *args, **kwargs) -> Any:
        """Execute with fallback on error"""
        try:
            return await primary(*args, **kwargs)
        except Exception as e:
            logging.warning(f"Primary failed: {e}, using fallback")
            return await fallback(*args, **kwargs)
    
    @staticmethod
    async def with_circuit_breaker(func: Callable, threshold: int = 5,
                                  timeout: int = 60, *args, **kwargs) -> Any:
        """Circuit breaker pattern implementation"""
        # Implementation of circuit breaker
        pass
```

### Error Monitoring Dashboard

```yaml
# Prometheus rules for error monitoring
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: llm-error-alerts
spec:
  groups:
  - name: llm-errors
    interval: 30s
    rules:
    - alert: HighErrorRate
      expr: |
        rate(llm_requests_failed_total[5m]) > 0.1
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "High error rate detected"
        description: "Error rate is {{ $value }} errors/sec"
    
    - alert: GPUErrors
      expr: |
        increase(llm_gpu_errors_total[5m]) > 0
      for: 1m
      labels:
        severity: warning
      annotations:
        summary: "GPU errors detected"
        description: "{{ $value }} GPU errors in last 5 minutes"
```

## Error Code Reference Table

| Code | Category | Description | Quick Fix |
|------|----------|-------------|-----------|
| DEPLOYMENT001 | Deployment | CRD not found | Update CRDs |
| DEPLOYMENT002 | Deployment | Resource quota exceeded | Increase quota |
| DEPLOYMENT003 | Deployment | Image pull failed | Add pull secret |
| RUNTIME001 | Runtime | CUDA OOM | Reduce model size |
| RUNTIME002 | Runtime | Model load timeout | Increase timeout |
| RUNTIME003 | Runtime | Tensor mismatch | Fix data types |
| NETWORK001 | Network | Connection refused | Check service |
| NETWORK002 | Network | DNS failed | Fix CoreDNS |
| NETWORK003 | Network | TLS error | Add CA cert |
| GPU001 | GPU | Driver mismatch | Update driver |
| GPU002 | GPU | Device not found | Check allocation |
| GPU003 | GPU | NCCL error | Configure NCCL |
| MEMORY001 | Memory | OOM killed | Increase limits |
| MEMORY002 | Memory | Shared memory full | Mount larger /dev/shm |
| STORAGE001 | Storage | PVC pending | Check StorageClass |
| STORAGE002 | Storage | Mount failed | Check attachments |
| AUTH001 | Auth | RBAC denied | Add permissions |
| AUTH002 | Auth | Registry auth failed | Update credentials |

## Next Steps

- Continue to [Emergency Procedures](./06-emergency-procedures.md) for critical incident response
- Review [Case Studies](./07-case-studies.md) for real-world error scenarios
