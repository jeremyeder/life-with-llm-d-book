#!/bin/bash
"""
Comprehensive Health Check for LLM-D Deployments

This script performs a thorough health check of llm-d deployments,
checking deployment status, pods, GPU resources, errors, and network connectivity.

Usage:
    ./llm-d-health-check.sh <namespace> <deployment>

Example:
    ./llm-d-health-check.sh production gpt-model
"""

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