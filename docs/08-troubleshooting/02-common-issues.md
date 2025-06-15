---
title: Common Issues and Solutions
description: Solutions to frequently encountered problems in llm-d deployments
sidebar_position: 2
---

# Common Issues and Solutions

This section covers the most frequently encountered issues in llm-d deployments, their root causes, and proven solutions.

## Deployment Issues

### Issue: Model Deployment Stuck in Pending State

**Symptoms:**
- LLMDeployment remains in `Pending` status
- Pods not being created
- No error messages in deployment status

**Root Causes:**
1. Insufficient GPU resources
2. Node selector constraints
3. Taints and tolerations mismatch
4. Resource quotas exceeded

**Solutions:**

```bash
# 1. Check resource availability
kubectl describe nodes | grep -A5 "Allocated resources"
kubectl get nodes -l nvidia.com/gpu=true -o custom-columns=NAME:.metadata.name,GPUs:.status.allocatable."nvidia\.com/gpu"

# 2. Review node selectors
kubectl get llmdeployment <name> -n <namespace> -o yaml | grep -A5 nodeSelector

# 3. Check taints
kubectl get nodes -o custom-columns=NAME:.metadata.name,TAINTS:.spec.taints

# 4. Verify resource quotas
kubectl get resourcequota -n <namespace>
kubectl describe resourcequota -n <namespace>
```

**Fix Implementation:**

```yaml
# Add tolerations for GPU nodes
apiVersion: inference.llm-d.io/v1alpha1
kind: LLMDeployment
metadata:
  name: gpt-model
spec:
  model:
    name: gpt2
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
  # Remove strict node selectors
  nodeSelector: {}
  # Or use affinity for preference
  affinity:
    nodeAffinity:
      preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        preference:
          matchExpressions:
          - key: nvidia.com/gpu
            operator: Exists
```

### Issue: Image Pull Errors

**Symptoms:**
- Pod status shows `ErrImagePull` or `ImagePullBackOff`
- Events show authentication failures
- Registry unreachable errors

**Root Causes:**
1. Missing image pull secrets
2. Incorrect registry credentials
3. Network connectivity to registry
4. Image doesn't exist or wrong tag

**Solutions:**

```bash
# 1. Check current pull secrets
kubectl get secrets -n <namespace> | grep docker

# 2. Create pull secret
kubectl create secret docker-registry model-registry \
  --docker-server=registry.example.com \
  --docker-username=<username> \
  --docker-password=<password> \
  --docker-email=<email> \
  -n <namespace>

# 3. Test registry access
docker login registry.example.com
docker pull registry.example.com/models/llama2:latest

# 4. Add secret to deployment
kubectl patch llmdeployment <name> -n <namespace> --type='json' \
  -p='[{"op": "add", "path": "/spec/imagePullSecrets", "value": [{"name": "model-registry"}]}]'
```

**Permanent Fix:**

```yaml
# In LLMDeployment spec
spec:
  imagePullSecrets:
  - name: model-registry
  model:
    image: registry.example.com/models/llama2:latest
```

### Issue: CRD Version Conflicts

**Symptoms:**
- API version errors when applying manifests
- `no matches for kind "LLMDeployment"`
- Schema validation failures

**Root Causes:**
1. Outdated CRD definitions
2. Multiple CRD versions installed
3. Client/server version mismatch

**Solutions:**

```bash
# 1. Check installed CRDs
kubectl get crd | grep llm-d
kubectl get crd llmdeployments.inference.llm-d.io -o yaml | grep -A5 "versions:"

# 2. Update CRDs
kubectl apply -f https://github.com/llm-d/llm-d/releases/latest/download/crds.yaml

# 3. Verify API resources
kubectl api-resources | grep llm-d

# 4. Convert existing resources
kubectl convert -f old-deployment.yaml --output-version=inference.llm-d.io/v1alpha1
```

## Performance Issues

### Issue: High Inference Latency

**Symptoms:**
- P95 latency > 5 seconds
- Inconsistent response times
- Timeouts under load

**Root Causes:**
1. Cold starts
2. Insufficient GPU memory
3. CPU throttling
4. Network bottlenecks

**Diagnostic Steps:**

```bash
# 1. Check model loading time
kubectl logs -n <namespace> <pod> | grep -i "model loaded"

# 2. Monitor GPU utilization
kubectl exec -n <namespace> <pod> -- nvidia-smi -l 1

# 3. Check CPU throttling
kubectl exec -n <namespace> <pod> -- cat /sys/fs/cgroup/cpu/cpu.stat | grep throttled

# 4. Measure network latency
kubectl exec -n <namespace> <pod> -- ping -c 10 <service>
```

**Solutions:**

```yaml
# 1. Enable model preloading
apiVersion: inference.llm-d.io/v1alpha1
kind: LLMDeployment
spec:
  model:
    preload: true
    warmupRequests: 10
  
  # 2. Increase resources
  resources:
    requests:
      memory: "32Gi"
      nvidia.com/gpu: "1"
    limits:
      memory: "48Gi"
      nvidia.com/gpu: "1"
  
  # 3. Configure autoscaling
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetGPUUtilization: 70
```

### Issue: Memory Leaks

**Symptoms:**
- Gradual memory increase
- OOMKilled pods after hours/days
- Degrading performance over time

**Root Causes:**
1. Model caching issues
2. Request queue buildup
3. Tensor memory not released
4. Logging verbosity

**Detection:**

```bash
# Monitor memory over time
while true; do
  kubectl top pod -n <namespace> <pod> | tee -a memory.log
  sleep 60
done

# Check for OOM events
kubectl get events -n <namespace> --field-selector reason=OOMKilling

# Analyze memory allocations
kubectl exec -n <namespace> <pod> -- cat /proc/<pid>/status | grep -i vm
```

**Solutions:**

```yaml
# 1. Configure garbage collection
apiVersion: inference.llm-d.io/v1alpha1
kind: LLMDeployment
spec:
  runtime:
    env:
    - name: PYTORCH_CUDA_ALLOC_CONF
      value: "max_split_size_mb:512,garbage_collection_threshold:0.6"
    - name: OMP_NUM_THREADS
      value: "4"
  
  # 2. Set memory limits with buffer
  resources:
    requests:
      memory: "16Gi"
    limits:
      memory: "20Gi"  # 25% buffer
  
  # 3. Configure liveness probe
  livenessProbe:
    httpGet:
      path: /health
      port: 8080
    initialDelaySeconds: 300
    periodSeconds: 30
    failureThreshold: 3
```

## GPU Issues

### Issue: CUDA Out of Memory

**Symptoms:**
- `CUDA out of memory` errors
- Model loading failures
- Inference requests rejected

**Root Causes:**
1. Model too large for GPU
2. Batch size too high
3. Memory fragmentation
4. Multiple models on same GPU

**Solutions:**

```bash
# 1. Check GPU memory usage
nvidia-smi -q -d MEMORY | grep -A4 "FB Memory Usage"

# 2. Clear GPU memory
nvidia-smi --gpu-reset

# 3. Reduce batch size
kubectl patch llmdeployment <name> -n <namespace> \
  --type merge -p '{"spec":{"serving":{"batchSize":1}}}'
```

**Configuration Adjustments:**

```yaml
apiVersion: inference.llm-d.io/v1alpha1
kind: LLMDeployment
spec:
  model:
    quantization:
      enabled: true
      precision: "int8"  # Reduce memory usage
  
  serving:
    batchSize: 1
    maxConcurrency: 4
  
  # Use specific GPU
  resources:
    limits:
      nvidia.com/gpu: "1"
  nodeSelector:
    nvidia.com/gpu.product: "Tesla-V100-SXM2-32GB"
```

### Issue: GPU Not Detected

**Symptoms:**
- `nvidia-smi: command not found`
- No GPU resources in node capacity
- CUDA library errors

**Root Causes:**
1. Missing NVIDIA drivers
2. Device plugin not running
3. Container runtime misconfiguration
4. Incompatible CUDA versions

**Diagnosis and Fix:**

```bash
# 1. Check node GPU capability
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.capacity."nvidia\.com/gpu"

# 2. Verify device plugin
kubectl get ds -n kube-system nvidia-device-plugin-daemonset

# 3. Check runtime configuration
kubectl exec -n <namespace> <pod> -- cat /etc/docker/daemon.json | grep nvidia

# 4. Install/update device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.13.0/nvidia-device-plugin.yml
```

## Network Issues

### Issue: Service Discovery Failures

**Symptoms:**
- DNS resolution failures
- Service endpoints empty
- Intermittent connectivity

**Root Causes:**
1. CoreDNS issues
2. Service selector mismatch
3. Network policies blocking traffic
4. Service mesh misconfiguration

**Troubleshooting:**

```bash
# 1. Test DNS resolution
kubectl run -it --rm debug --image=busybox --restart=Never -- \
  nslookup model-service.default.svc.cluster.local

# 2. Check service endpoints
kubectl get endpoints model-service -n default

# 3. Verify pod labels match service selector
kubectl get pods -n default --show-labels
kubectl get svc model-service -n default -o yaml | grep -A5 selector

# 4. Test connectivity
kubectl run -it --rm debug --image=nicolaka/netshoot --restart=Never -- \
  curl -v http://model-service.default.svc.cluster.local:8080/health
```

**Solutions:**

```yaml
# Fix service selector
apiVersion: v1
kind: Service
metadata:
  name: model-service
spec:
  selector:
    app: llm-model
    component: inference
  ports:
  - port: 8080
    targetPort: 8080
---
# Ensure pods have matching labels
apiVersion: inference.llm-d.io/v1alpha1
kind: LLMDeployment
spec:
  template:
    metadata:
      labels:
        app: llm-model
        component: inference
```

### Issue: Ingress Configuration Problems

**Symptoms:**
- 502/503 errors from load balancer
- SSL/TLS certificate errors
- Routing to wrong backend

**Root Causes:**
1. Ingress controller not running
2. Incorrect backend service
3. Missing annotations
4. Certificate issues

**Solutions:**

```yaml
# Correct ingress configuration
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: llm-ingress
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.example.com
    secretName: llm-tls
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /v1/models
        pathType: Prefix
        backend:
          service:
            name: model-service
            port:
              number: 8080
```

## Storage Issues

### Issue: Model Loading Failures

**Symptoms:**
- "Model file not found" errors
- Slow model initialization
- Partial model loads

**Root Causes:**
1. PVC not mounted correctly
2. Insufficient storage space
3. Wrong file permissions
4. Corrupted model files

**Diagnosis:**

```bash
# 1. Check PVC status
kubectl get pvc -n <namespace>
kubectl describe pvc model-storage -n <namespace>

# 2. Verify mount inside pod
kubectl exec -n <namespace> <pod> -- ls -la /models
kubectl exec -n <namespace> <pod> -- df -h /models

# 3. Check file integrity
kubectl exec -n <namespace> <pod> -- md5sum /models/*.bin
```

**Solutions:**

```yaml
# Ensure proper PVC configuration
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-storage
spec:
  accessModes:
  - ReadWriteMany  # For multi-pod access
  storageClassName: fast-ssd
  resources:
    requests:
      storage: 100Gi
---
apiVersion: inference.llm-d.io/v1alpha1
kind: LLMDeployment
spec:
  volumes:
  - name: model-cache
    persistentVolumeClaim:
      claimName: model-storage
  containers:
  - name: inference
    volumeMounts:
    - name: model-cache
      mountPath: /models
      readOnly: true  # Prevent accidental modifications
```

## Monitoring and Observability Issues

### Issue: Missing Metrics

**Symptoms:**
- Prometheus targets down
- No metrics in Grafana
- Incomplete dashboards

**Root Causes:**
1. Metrics port not exposed
2. Service monitor misconfigured
3. Prometheus scrape config wrong
4. Network policies blocking

**Solutions:**

```yaml
# 1. Expose metrics port
apiVersion: inference.llm-d.io/v1alpha1
kind: LLMDeployment
spec:
  monitoring:
    enabled: true
    port: 9090
    path: /metrics
---
# 2. Create ServiceMonitor
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: llm-metrics
spec:
  selector:
    matchLabels:
      app: llm-model
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

## Best Practices Summary

### Prevention Strategies

1. **Resource Planning**
   - Always specify resource requests and limits
   - Use horizontal pod autoscaling
   - Monitor resource utilization trends

2. **Configuration Management**
   - Version control all configurations
   - Use GitOps for deployments
   - Implement proper RBAC

3. **Monitoring Setup**
   - Deploy comprehensive monitoring stack
   - Set up alerting rules
   - Create runbooks for common issues

4. **Testing Procedures**
   - Regular chaos engineering tests
   - Load testing before production
   - Canary deployments for updates

### Quick Reference Card

Keep this handy for rapid troubleshooting:

```bash
# Quick diagnostic commands
alias llm-status='kubectl get llmdeployments -A'
alias llm-logs='kubectl logs -n llm-d-system deployment/llm-d-operator'
alias gpu-check='kubectl get nodes -l nvidia.com/gpu=true'
alias pod-issues='kubectl get pods -A | grep -v Running'
alias recent-events='kubectl get events -A --sort-by=.lastTimestamp | head -20'

# Emergency restart procedures
kubectl rollout restart deployment/llm-d-operator -n llm-d-system
kubectl delete pod -n <namespace> -l app=<model-name>
```

## Next Steps

- Continue to [Diagnostic Tools](./03-diagnostic-tools.md) for advanced debugging procedures
- Review [Performance Troubleshooting](./04-performance-troubleshooting.md) for optimization techniques
- Check [Error Patterns](./05-error-patterns.md) for comprehensive error reference