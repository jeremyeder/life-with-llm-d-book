---
title: Troubleshooting Decision Trees
description: Visual decision trees for rapid problem identification
sidebar_position: 1
---

# Troubleshooting Decision Trees

Decision trees provide a systematic approach to identifying and resolving issues quickly. Start at the symptom and follow the tree to find the most likely cause and solution.

## Model Deployment Failures

```mermaid
graph TD
    A[Model fails to deploy] --> B{Check deployment status}
    B -->|Pending| C[Resource issues]
    B -->|Failed| D[Check error logs]
    B -->|Unknown| E[CRD issues]
    
    C --> F{GPU available?}
    F -->|No| G[Scale GPU nodes]
    F -->|Yes| H{Memory sufficient?}
    H -->|No| I[Increase limits]
    H -->|Yes| J[Check node selectors]
    
    D --> K{Error type?}
    K -->|Image pull| L[Registry access]
    K -->|OOM killed| M[Increase memory]
    K -->|Config error| N[Validate YAML]
    K -->|Permission| O[Check RBAC]
    
    E --> P[Verify CRD installation]
    P --> Q[Check operator logs]
```

### Resolution Steps

#### Resource Issues

```bash
# Check available resources
kubectl describe nodes | grep -E "Allocatable|Allocated" -A5

# Check pending pods
kubectl get pods -n <namespace> -o wide | grep Pending

# View resource requests
kubectl describe pod <pod-name> -n <namespace> | grep -A5 "Requests"
```

#### Image Pull Errors

```bash
# Check image pull secrets
kubectl get secrets -n <namespace>

# Test registry access
docker pull <registry>/<image>:<tag>

# Create pull secret if needed
kubectl create secret docker-registry regcred \
  --docker-server=<registry> \
  --docker-username=<username> \
  --docker-password=<password> \
  -n <namespace>
```

## Performance Degradation

```mermaid
graph TD
    A[Slow inference] --> B{Latency type?}
    B -->|Cold start| C[Check cache warming]
    B -->|Consistent| D[Resource saturation]
    B -->|Intermittent| E[Network issues]
    
    C --> F[Enable model preloading]
    F --> G[Configure cache policy]
    
    D --> H{Resource type?}
    H -->|CPU| I[Check CPU throttling]
    H -->|GPU| J[Check GPU utilization]
    H -->|Memory| K[Check swap usage]
    H -->|Network| L[Check bandwidth]
    
    E --> M[Trace network path]
    M --> N[Check DNS resolution]
    N --> O[Verify service mesh]
```

### Diagnostic Commands

#### Performance Metrics

```bash
# GPU utilization
nvidia-smi -l 1

# CPU and memory
kubectl top pods -n <namespace>

# Network latency
kubectl exec -n <namespace> <pod> -- curl -w "@curl-format.txt" -o /dev/null -s http://service:8080/health

# Where curl-format.txt contains:
time_namelookup:  %{time_namelookup}s\n
time_connect:  %{time_connect}s\n
time_appconnect:  %{time_appconnect}s\n
time_pretransfer:  %{time_pretransfer}s\n
time_redirect:  %{time_redirect}s\n
time_starttransfer:  %{time_starttransfer}s\n
time_total:  %{time_total}s\n
```

## Service Unavailability

```mermaid
graph TD
    A[Service unavailable] --> B{HTTP status?}
    B -->|502/503| C[Backend issues]
    B -->|504| D[Timeout issues]
    B -->|Connection refused| E[Service down]
    
    C --> F{Check backends}
    F -->|No healthy| G[Pod health checks]
    F -->|Misconfigured| H[Service selector]
    
    D --> I[Increase timeouts]
    I --> J[Check model size]
    J --> K[Optimize batch size]
    
    E --> L{Service exists?}
    L -->|No| M[Create service]
    L -->|Yes| N{Endpoints exist?}
    N -->|No| O[Check pod labels]
    N -->|Yes| P[Check network policy]
```

### Quick Checks

```bash
# Service and endpoints
kubectl get svc,endpoints -n <namespace>

# Pod readiness
kubectl get pods -n <namespace> -o custom-columns=NAME:.metadata.name,READY:.status.conditions[?(@.type=="Ready")].status

# Network policies
kubectl get networkpolicies -n <namespace>

# Service mesh status (if using Istio)
istioctl proxy-status
```

## GPU Issues

```mermaid
graph TD
    A[GPU not available] --> B{GPU nodes exist?}
    B -->|No| C[Add GPU nodes]
    B -->|Yes| D{Driver installed?}
    
    D -->|No| E[Install NVIDIA drivers]
    D -->|Yes| F{Device plugin running?}
    
    F -->|No| G[Deploy device plugin]
    F -->|Yes| H{GPU allocated?}
    
    H -->|No| I[Check resource requests]
    H -->|Yes| J{GPU visible in pod?}
    
    J -->|No| K[Check security context]
    J -->|Yes| L[Check CUDA compatibility]
```

### GPU Diagnostics

```bash
# Check GPU nodes
kubectl get nodes -l nvidia.com/gpu=true

# GPU device plugin
kubectl get pods -n kube-system | grep nvidia-device-plugin

# GPU allocation
kubectl describe nodes | grep -A5 "nvidia.com/gpu"

# Inside pod GPU check
kubectl exec -n <namespace> <pod> -- nvidia-smi
```

## Memory Issues

```mermaid
graph TD
    A[Out of memory] --> B{OOM killer triggered?}
    B -->|Yes| C[Increase limits]
    B -->|No| D{Memory leak?}
    
    C --> E[Analyze memory usage]
    E --> F[Right-size requests]
    
    D -->|Yes| G[Profile application]
    D -->|No| H{Batch size too large?}
    
    H -->|Yes| I[Reduce batch size]
    H -->|No| J[Check model size]
    
    G --> K[Fix memory leak]
    J --> L[Use model quantization]
```

### Memory Analysis

```bash
# Pod memory usage over time
kubectl top pod <pod-name> -n <namespace> --use-protocol-buffers | \
  while read line; do echo "$(date): $line"; sleep 5; done

# Check for OOM kills
kubectl describe pod <pod-name> -n <namespace> | grep -i "OOMKilled"

# System memory pressure
kubectl exec -n <namespace> <pod> -- cat /proc/meminfo | grep -E "MemTotal|MemFree|MemAvailable"
```

## Networking Issues

```mermaid
graph TD
    A[Network connectivity issue] --> B{Internal or external?}
    B -->|Internal| C[Cluster networking]
    B -->|External| D[Ingress/LoadBalancer]
    
    C --> E{DNS working?}
    E -->|No| F[Check CoreDNS]
    E -->|Yes| G{Service discovery?}
    
    G -->|Failed| H[Check service endpoints]
    G -->|Works| I[Network policies]
    
    D --> J{Ingress controller?}
    J -->|Down| K[Check controller pods]
    J -->|Running| L[Check ingress rules]
    
    F --> M[Restart CoreDNS]
    I --> N[Review network policies]
    L --> O[Validate certificates]
```

### Network Debugging

```bash
# DNS resolution test
kubectl run -it --rm debug --image=busybox --restart=Never -- nslookup <service>.<namespace>

# Service connectivity
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- curl -v http://<service>.<namespace>:8080

# Network policy test
kubectl exec -n <namespace> <pod> -- nc -zv <target-service> <port>

# Ingress debugging
kubectl logs -n ingress-nginx deployment/ingress-nginx-controller | grep <host>
```

## Storage Issues

```mermaid
graph TD
    A[Storage issue] --> B{Mount failed?}
    B -->|Yes| C[Check PVC status]
    B -->|No| D{Disk full?}
    
    C --> E{PVC bound?}
    E -->|No| F[Check storage class]
    E -->|Yes| G[Check node affinity]
    
    D -->|Yes| H[Clean up space]
    D -->|No| I{I/O errors?}
    
    I -->|Yes| J[Check disk health]
    I -->|No| K[Check permissions]
    
    F --> L[Verify provisioner]
    H --> M[Implement retention policy]
    J --> N[Replace disk]
```

### Storage Diagnostics

```bash
# PVC status
kubectl get pvc -A

# Storage usage
kubectl exec -n <namespace> <pod> -- df -h

# Disk I/O stats
kubectl exec -n <namespace> <pod> -- iostat -x 1

# Mount points
kubectl exec -n <namespace> <pod> -- mount | grep <volume>
```

## Best Practices

### Systematic Approach

1. **Gather symptoms** - What exactly is failing?
2. **Check recent changes** - What was modified?
3. **Follow the tree** - Use decision trees to narrow down
4. **Collect evidence** - Logs, metrics, events
5. **Test hypothesis** - Verify your theory
6. **Implement fix** - Apply the solution
7. **Verify resolution** - Confirm it's working
8. **Document findings** - Update runbooks

### Preventive Measures

- **Monitoring** - Set up comprehensive alerting
- **Logging** - Centralize and index all logs
- **Testing** - Regular chaos engineering
- **Documentation** - Keep runbooks updated
- **Training** - Regular incident response drills

### Emergency Kit

Keep these ready for quick access:

```bash
# Emergency diagnostic script
#!/bin/bash
echo "=== llm-d Emergency Diagnostic ==="
echo "Timestamp: $(date)"
echo ""
echo "=== Cluster Status ==="
kubectl cluster-info
echo ""
echo "=== Node Status ==="
kubectl get nodes
echo ""
echo "=== llm-d Components ==="
kubectl get all -n llm-d-system
echo ""
echo "=== LLM Deployments ==="
kubectl get llmdeployments -A
echo ""
echo "=== Recent Events ==="
kubectl get events -A --sort-by='.lastTimestamp' | head -20
```

## Next Steps

- Review [Common Issues](./02-common-issues.md) for specific problem solutions
- Check [Diagnostic Tools](./03-diagnostic-tools.md) for detailed analysis procedures
- See [Performance Troubleshooting](./04-performance-troubleshooting.md) for optimization techniques
