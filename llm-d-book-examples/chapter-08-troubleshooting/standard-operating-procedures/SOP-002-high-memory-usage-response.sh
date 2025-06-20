#!/bin/bash
"""
SOP-002: High Memory Usage Response

Standardized procedure for resolving memory pressure before OOM kills occur.
This SOP provides systematic steps to identify and mitigate memory issues.

Usage:
    ./SOP-002-high-memory-usage-response.sh

Execute when memory pressure alerts are triggered.
"""

# SOP-002: High Memory Usage Response
# Objective: Resolve memory pressure before OOM kills occur

echo "=== SOP-002: HIGH MEMORY USAGE RESPONSE ==="
echo "Operator: ${USER}"
echo "Start Time: $(date)"

# Step 1: Identify High Memory Pods (0-3 minutes)
echo "Step 1: Identify High Memory Pods"

# Find top memory consumers
echo "Top memory consuming pods:"
kubectl top pods -A --sort-by=memory | head -10

# Check for memory warnings
echo "Recent memory warnings:"
kubectl get events -A --field-selector type=Warning | grep -i memory

# Step 2: Check Memory Limits (3-5 minutes)
echo "Step 2: Check Memory Limits"

# For each high-memory pod, check limits
HIGH_MEMORY_PODS=$(kubectl top pods -A --sort-by=memory --no-headers | head -5 | awk '{print $2 " " $1}')
echo "$HIGH_MEMORY_PODS" | while read pod namespace; do
  echo "=== $pod in $namespace ==="
  kubectl get pod $pod -n $namespace -o jsonpath='{.spec.containers[0].resources.limits.memory}'
  echo ""
done

# Step 3: Apply Immediate Relief (5-10 minutes)
echo "Step 3: Apply Immediate Relief"

# Reduce batch sizes
echo "Reducing batch sizes..."
kubectl get llmdeployments -A -o json | jq -r '.items[] | "\(.metadata.namespace) \(.metadata.name)"' | while read ns name; do
  kubectl patch llmdeployment $name -n $ns --type='merge' -p='{"spec":{"serving":{"batchSize":1,"maxConcurrency":2}}}'
done

# Restart highest memory consumer
echo "Restarting highest memory consumer..."
HIGHEST_POD=$(kubectl top pods -A --sort-by=memory --no-headers | head -1 | awk '{print $2 " " $1}')
echo "$HIGHEST_POD" | while read pod namespace; do
  kubectl delete pod $pod -n $namespace
done

# Step 4: Scale if Necessary (10-15 minutes)
echo "Step 4: Scale if Necessary"

# Scale down replicas temporarily
echo "Temporarily scaling down replicas..."
kubectl get llmdeployments -A -o json | jq -r '.items[] | "\(.metadata.namespace) \(.metadata.name) \(.spec.replicas)"' | while read ns name replicas; do
  if [ $replicas -gt 2 ]; then
    new_replicas=$((replicas / 2))
    kubectl patch llmdeployment $name -n $ns --type='merge' -p="{\"spec\":{\"replicas\":$new_replicas}}"
  fi
done

echo "=== SOP-002 COMPLETED ==="
echo "End Time: $(date)"

# Verification steps
echo "=== VERIFICATION ==="
echo "Check memory usage dropped:"
kubectl top pods -A --sort-by=memory | head -5

echo "Check for recent OOM events:"
kubectl get events -A --field-selector type=Warning | grep -i oom | grep "$(date -d '10 minutes ago' '+%H:%M')" || echo "No recent OOM events"