#!/bin/bash
"""
P1 Incident: Memory Pressure Emergency Response

This script handles memory pressure incidents by identifying memory-hungry pods,
implementing emergency cleanup, and reducing resource consumption.

Usage:
    ./memory-pressure.sh

Execute when memory pressure alerts are triggered.
"""

echo "=== P1 INCIDENT: MEMORY PRESSURE ==="

# 1. Identify memory-hungry pods
echo "1. Finding high-memory pods..."
kubectl top pods -A --sort-by=memory | head -20

# 2. Check for memory leaks
echo "2. Checking for memory leaks..."
for pod in $(kubectl get pods -A -l app.kubernetes.io/name=llm-d -o jsonpath='{.items[*].metadata.name}'); do
  namespace=$(kubectl get pod $pod -A -o jsonpath='{.items[0].metadata.namespace}')
  echo "Pod: $pod"
  kubectl exec -n $namespace $pod -- cat /proc/meminfo | grep -E "MemTotal|MemFree|MemAvailable"
done

# 3. Emergency memory cleanup
echo "3. Emergency memory cleanup..."
# Clear caches
kubectl exec -A -l app.kubernetes.io/name=llm-d -- sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true

# 4. Reduce batch sizes
echo "4. Reducing batch sizes..."
for deployment in $(kubectl get llmdeployments -A -o jsonpath='{.items[*].metadata.name}'); do
  namespace=$(kubectl get llmdeployment $deployment -A -o jsonpath='{.items[0].metadata.namespace}')
  kubectl patch llmdeployment $deployment -n $namespace --type='merge' -p='{
    "spec": {
      "serving": {
        "batchSize": 1,
        "maxConcurrency": 2
      }
    }
  }'
done

# 5. Restart highest memory consumers
echo "5. Restarting high memory pods..."
kubectl delete pods -A -l app.kubernetes.io/name=llm-d --field-selector 'status.phase=Running'