---
title: Diagnostic Tools and Procedures
description: Comprehensive tools and procedures for diagnosing llm-d issues
sidebar_position: 3
---

# Diagnostic Tools and Procedures

This section provides detailed procedures for using diagnostic tools to investigate and resolve issues in llm-d deployments.

## Built-in Diagnostic Tools

### llm-d CLI Diagnostics

The llm-d CLI provides several diagnostic commands:

```bash
# System health check
llm-d diagnose system

# Model deployment diagnostics
llm-d diagnose deployment <name> -n <namespace>

# Performance analysis
llm-d diagnose performance <name> -n <namespace>

# Network connectivity test
llm-d diagnose network <name> -n <namespace>
```

#### Example Output Analysis

```bash
$ llm-d diagnose deployment gpt-model -n production

=== LLMDeployment Diagnostics ===
Name: gpt-model
Namespace: production
Status: Degraded

Issues Found:
1. WARNING: GPU utilization at 95% (threshold: 80%)
2. ERROR: 2/5 pods in CrashLoopBackOff
3. WARNING: High memory pressure detected

Recommendations:
- Scale up GPU nodes or reduce model size
- Check pod logs for crash reasons
- Consider enabling model quantization

Detailed Report saved to: /tmp/llm-d-diag-20240115-143022.json
```

### Operator Diagnostics

Access operator diagnostic endpoints:

```bash
# Port-forward to operator
kubectl port-forward -n llm-d-system deployment/llm-d-operator 8080:8080

# Health endpoint
curl http://localhost:8080/healthz

# Metrics endpoint
curl http://localhost:8080/metrics | grep llm_d_

# Debug endpoint (detailed state)
curl http://localhost:8080/debug/state
```

## Kubernetes Native Tools

### kubectl Debugging

Essential kubectl commands for diagnostics:

```bash
# Comprehensive resource inspection
kubectl describe llmdeployment <name> -n <namespace>

# Event timeline
kubectl get events -n <namespace> --sort-by='.lastTimestamp' \
  --field-selector involvedObject.name=<pod-name>

# Resource usage
kubectl top pods -n <namespace> --containers

# Pod logs with timestamps
kubectl logs -n <namespace> <pod> -c <container> --timestamps=true --tail=1000

# Previous container logs (after crash)
kubectl logs -n <namespace> <pod> -c <container> --previous
```

### Debug Containers

Use ephemeral debug containers for live troubleshooting:

```bash
# Add debug container to running pod
kubectl debug -n <namespace> <pod> -it \
  --image=nicolaka/netshoot \
  --target=<container> \
  -- /bin/bash

# Common debug operations inside container
## Network diagnostics
ss -tulpn
netstat -an
tcpdump -i any -w capture.pcap

## Process inspection
ps aux
strace -p <pid>
lsof -p <pid>

## Memory analysis
pmap -x <pid>
cat /proc/<pid>/status
```

### Resource Analysis

Deep dive into resource usage:

```bash
# CPU throttling analysis
kubectl exec -n <namespace> <pod> -- \
  cat /sys/fs/cgroup/cpu/cpu.stat

# Memory pressure
kubectl exec -n <namespace> <pod> -- \
  cat /proc/pressure/memory

# I/O statistics
kubectl exec -n <namespace> <pod> -- \
  iostat -x 1 10
```

## GPU Diagnostics

### NVIDIA Tools

Comprehensive GPU diagnostics:

```bash
# Basic GPU status
kubectl exec -n <namespace> <pod> -- nvidia-smi

# Detailed GPU metrics
kubectl exec -n <namespace> <pod> -- \
  nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv

# GPU process monitoring
kubectl exec -n <namespace> <pod> -- \
  nvidia-smi pmon -i 0 -s u -c 10

# CUDA debugging
kubectl exec -n <namespace> <pod> -- \
  cuda-memcheck python inference.py
```

### GPU Memory Analysis

```python
# gpu_memory_profile.py - Run inside pod
import torch
import psutil
import GPUtil

def analyze_gpu_memory():
    # System memory
    print(f"System RAM: {psutil.virtual_memory().percent}% used")
    
    # GPU memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"Cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
            
            # Detailed GPU stats
            gpu = GPUtil.getGPUs()[i]
            print(f"GPU Load: {gpu.load * 100:.1f}%")
            print(f"GPU Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
            print(f"GPU Temp: {gpu.temperature}°C")

if __name__ == "__main__":
    analyze_gpu_memory()
```

## Network Diagnostics

### Service Mesh Debugging

For Istio-based deployments:

```bash
# Check proxy configuration
istioctl proxy-config cluster <pod> -n <namespace>
istioctl proxy-config listeners <pod> -n <namespace>
istioctl proxy-config routes <pod> -n <namespace>

# Analyze traffic
istioctl x describe pod <pod> -n <namespace>

# Check mTLS status
istioctl authn tls-check <pod>.<namespace>
```

### Network Latency Testing

```bash
# Create network test pod
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: network-test
  namespace: <namespace>
spec:
  containers:
  - name: netshoot
    image: nicolaka/netshoot
    command: ["sleep", "3600"]
EOF

# Run latency tests
kubectl exec -n <namespace> network-test -- \
  curl -o /dev/null -s -w "Connect: %{time_connect}s\nTTFB: %{time_starttransfer}s\nTotal: %{time_total}s\n" \
  http://model-service:8080/health

# MTU discovery
kubectl exec -n <namespace> network-test -- \
  ping -M do -s 1472 -c 4 model-service

# Bandwidth test
kubectl exec -n <namespace> network-test -- \
  iperf3 -c model-service -p 5201
```

## Performance Profiling

### CPU Profiling

```bash
# Install profiling tools in pod
kubectl exec -n <namespace> <pod> -- \
  apt-get update && apt-get install -y linux-tools-common

# CPU flame graph generation
kubectl exec -n <namespace> <pod> -- \
  perf record -F 99 -a -g -- sleep 30
kubectl exec -n <namespace> <pod> -- \
  perf script > perf.script

# Copy and analyze locally
kubectl cp <namespace>/<pod>:perf.script ./perf.script
flamegraph.pl perf.script > flame.svg
```

### Memory Profiling

```python
# memory_profiler.py - Memory usage analysis
import tracemalloc
import torch
import gc

def profile_model_memory():
    tracemalloc.start()
    
    # Your model loading code here
    model = load_model()
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
    
    # Detailed snapshot
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    print("\nTop 10 memory allocations:")
    for stat in top_stats[:10]:
        print(stat)
    
    tracemalloc.stop()

# GPU memory tracking
def track_gpu_memory():
    torch.cuda.reset_peak_memory_stats()
    
    # Your inference code here
    output = model(input)
    
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
```

## Log Analysis

### Structured Log Parsing

```bash
# Extract and analyze error patterns
kubectl logs -n <namespace> <pod> --tail=10000 | \
  jq -r 'select(.level == "error") | "\(.timestamp) \(.message)"' | \
  sort | uniq -c | sort -rn

# Performance log analysis
kubectl logs -n <namespace> <pod> --tail=10000 | \
  jq -r 'select(.inference_time != null) | .inference_time' | \
  awk '{sum+=$1; count++} END {print "Avg:", sum/count, "ms"}'

# Request pattern analysis
kubectl logs -n <namespace> <pod> --tail=10000 | \
  grep "POST /v1/completions" | \
  awk '{print $1}' | \
  cut -d'T' -f2 | \
  cut -d':' -f1 | \
  sort | uniq -c
```

### Log Aggregation Query

For centralized logging (Elasticsearch/Loki):

```json
// Elasticsearch query for error spike detection
{
  "query": {
    "bool": {
      "must": [
        {"term": {"kubernetes.namespace": "production"}},
        {"term": {"level": "error"}},
        {"range": {"@timestamp": {"gte": "now-1h"}}}
      ]
    }
  },
  "aggs": {
    "errors_over_time": {
      "date_histogram": {
        "field": "@timestamp",
        "interval": "1m"
      }
    }
  }
}
```

## Distributed Tracing

### OpenTelemetry Integration

```yaml
# Add tracing to LLMDeployment
apiVersion: inference.llm-d.io/v1alpha1
kind: LLMDeployment
spec:
  observability:
    tracing:
      enabled: true
      endpoint: "jaeger-collector:4317"
      samplingRate: 0.1
```

### Trace Analysis

```bash
# Port-forward to Jaeger UI
kubectl port-forward -n observability svc/jaeger-query 16686:16686

# Analyze traces programmatically
curl "http://localhost:16686/api/traces?service=llm-model&operation=inference&limit=20" | \
  jq '.data[].spans[] | select(.operationName == "model.forward") | .duration'
```

## Custom Diagnostic Scripts

### Comprehensive Health Check

```bash
#!/bin/bash
# llm-d-health-check.sh

NAMESPACE=$1
DEPLOYMENT=$2

echo "=== llm-d Health Check Report ==="
echo "Timestamp: $(date)"
echo "Namespace: $NAMESPACE"
echo "Deployment: $DEPLOYMENT"
echo

# Check deployment status
echo "1. Deployment Status:"
kubectl get llmdeployment $DEPLOYMENT -n $NAMESPACE

# Check pods
echo -e "\n2. Pod Status:"
kubectl get pods -n $NAMESPACE -l app=$DEPLOYMENT

# Check GPU allocation
echo -e "\n3. GPU Resources:"
kubectl exec -n $NAMESPACE -l app=$DEPLOYMENT -- nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv

# Check recent errors
echo -e "\n4. Recent Errors (last 50 lines):"
kubectl logs -n $NAMESPACE -l app=$DEPLOYMENT --tail=50 | grep -i error

# Check metrics
echo -e "\n5. Current Metrics:"
kubectl exec -n $NAMESPACE -l app=$DEPLOYMENT -- curl -s localhost:9090/metrics | grep -E "requests_total|request_duration|gpu_utilization"

# Network connectivity
echo -e "\n6. Network Test:"
kubectl exec -n $NAMESPACE -l app=$DEPLOYMENT -- curl -s -o /dev/null -w "HTTP Code: %{http_code}, Time: %{time_total}s\n" http://localhost:8080/health
```

### Automated Diagnostics Collection

```python
#!/usr/bin/env python3
# collect_diagnostics.py

import subprocess
import json
import datetime
import os

class LLMDiagnostics:
    def __init__(self, namespace, deployment):
        self.namespace = namespace
        self.deployment = deployment
        self.timestamp = datetime.datetime.now().isoformat()
        self.report = {
            "timestamp": self.timestamp,
            "namespace": namespace,
            "deployment": deployment,
            "diagnostics": {}
        }
    
    def run_command(self, cmd):
        """Execute command and return output"""
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def collect_deployment_info(self):
        """Collect deployment information"""
        cmd = f"kubectl get llmdeployment {self.deployment} -n {self.namespace} -o json"
        result = self.run_command(cmd)
        if result["success"]:
            self.report["diagnostics"]["deployment"] = json.loads(result["stdout"])
    
    def collect_pod_info(self):
        """Collect pod information"""
        cmd = f"kubectl get pods -n {self.namespace} -l app={self.deployment} -o json"
        result = self.run_command(cmd)
        if result["success"]:
            self.report["diagnostics"]["pods"] = json.loads(result["stdout"])
    
    def collect_logs(self, lines=100):
        """Collect recent logs"""
        cmd = f"kubectl logs -n {self.namespace} -l app={self.deployment} --tail={lines}"
        result = self.run_command(cmd)
        if result["success"]:
            self.report["diagnostics"]["logs"] = result["stdout"].split('\n')
    
    def collect_metrics(self):
        """Collect current metrics"""
        cmd = f"kubectl exec -n {self.namespace} -l app={self.deployment} -- curl -s localhost:9090/metrics"
        result = self.run_command(cmd)
        if result["success"]:
            self.report["diagnostics"]["metrics"] = result["stdout"]
    
    def save_report(self):
        """Save diagnostic report"""
        filename = f"llm-d-diagnostics-{self.deployment}-{self.timestamp.replace(':', '-')}.json"
        with open(filename, 'w') as f:
            json.dump(self.report, f, indent=2)
        print(f"Diagnostic report saved to: {filename}")
        return filename

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python collect_diagnostics.py <namespace> <deployment>")
        sys.exit(1)
    
    diag = LLMDiagnostics(sys.argv[1], sys.argv[2])
    diag.collect_deployment_info()
    diag.collect_pod_info()
    diag.collect_logs()
    diag.collect_metrics()
    diag.save_report()
```

## Best Practices

### Diagnostic Workflow

1. **Initial Assessment**
   - Check deployment status
   - Review recent events
   - Examine pod states

2. **Deep Dive**
   - Analyze logs
   - Check resource usage
   - Review metrics

3. **Root Cause Analysis**
   - Correlate symptoms
   - Test hypotheses
   - Isolate variables

4. **Documentation**
   - Record findings
   - Update runbooks
   - Share knowledge

### Tool Selection Guide

| Symptom | Primary Tool | Secondary Tool |
|---------|-------------|----------------|
| High latency | Tracing | Profiling |
| Memory issues | Memory profiler | GPU diagnostics |
| Network problems | Network diagnostics | Service mesh tools |
| Crashes | Log analysis | Debug containers |
| Performance | Metrics | Flame graphs |

## Next Steps

- Continue to [Performance Troubleshooting](./04-performance-troubleshooting.md) for optimization techniques
- Review [Error Patterns](./05-error-patterns.md) for specific error analysis
- Check [Emergency Procedures](./06-emergency-procedures.md) for critical situation handling
