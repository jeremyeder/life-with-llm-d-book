#!/bin/bash
"""
Fix CUDA Out of Memory Errors

This script provides comprehensive solutions for CUDA out of memory errors
including model quantization, batch size reduction, and GPU memory optimization.

Usage:
    ./fix-cuda-out-of-memory.sh <namespace> <deployment>

Example:
    ./fix-cuda-out-of-memory.sh production gpt-model
"""

NAMESPACE=$1
DEPLOYMENT=$2

if [ -z "$NAMESPACE" ] || [ -z "$DEPLOYMENT" ]; then
  echo "Usage: $0 <namespace> <deployment>"
  exit 1
fi

echo "=== FIXING CUDA OUT OF MEMORY ERRORS ==="
echo "Namespace: $NAMESPACE"
echo "Deployment: $DEPLOYMENT"

# 1. Check GPU memory usage
echo "1. Checking GPU memory usage..."
kubectl exec -n $NAMESPACE -l app=$DEPLOYMENT -- nvidia-smi -q -d MEMORY | grep -A4 "FB Memory Usage" || echo "Could not check GPU memory"

# 2. Clear GPU memory
echo "2. Clearing GPU memory..."
kubectl exec -n $NAMESPACE -l app=$DEPLOYMENT -- nvidia-smi --gpu-reset || echo "Could not reset GPU"

# 3. Reduce batch size
echo "3. Reducing batch size..."
kubectl patch llmdeployment $DEPLOYMENT -n $NAMESPACE \
  --type merge -p '{"spec":{"serving":{"batchSize":1}}}'

# 4. Enable quantization and optimize configuration
echo "4. Applying memory optimizations..."
kubectl patch llmdeployment $DEPLOYMENT -n $NAMESPACE --type='merge' -p='{
  "spec": {
    "model": {
      "quantization": {
        "enabled": true,
        "precision": "int8"
      }
    },
    "serving": {
      "batchSize": 1,
      "maxConcurrency": 4
    }
  }
}'

# 5. Restart deployment to apply changes
echo "5. Restarting deployment..."
kubectl rollout restart deployment/$DEPLOYMENT -n $NAMESPACE
kubectl rollout status deployment/$DEPLOYMENT -n $NAMESPACE --timeout=300s

echo "=== CUDA OUT OF MEMORY FIX COMPLETED ==="

# Verification
echo "=== VERIFICATION ==="
echo "Checking deployment status:"
kubectl get llmdeployment $DEPLOYMENT -n $NAMESPACE -o jsonpath='{.status.phase}'
echo ""

echo "Checking GPU memory usage after fix:"
kubectl exec -n $NAMESPACE -l app=$DEPLOYMENT -- nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv || echo "Could not check GPU status"