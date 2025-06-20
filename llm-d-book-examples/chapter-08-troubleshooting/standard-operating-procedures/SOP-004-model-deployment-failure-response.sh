#!/bin/bash
"""
SOP-004: Model Deployment Failure Response

Standardized procedure for getting failed model deployments back to healthy state.
This SOP provides systematic troubleshooting for deployment failures.

Usage:
    ./SOP-004-model-deployment-failure-response.sh <deployment_name> <namespace>

Example:
    ./SOP-004-model-deployment-failure-response.sh gpt-model production
"""

# SOP-004: Model Deployment Failure Response
# Objective: Get failed model deployment back to healthy state

DEPLOYMENT_NAME=$1
NAMESPACE=$2

if [ -z "$DEPLOYMENT_NAME" ] || [ -z "$NAMESPACE" ]; then
  echo "Usage: $0 <deployment_name> <namespace>"
  exit 1
fi

echo "=== SOP-004: MODEL DEPLOYMENT FAILURE RESPONSE ==="
echo "Operator: ${USER}"
echo "Deployment: $DEPLOYMENT_NAME"
echo "Namespace: $NAMESPACE"
echo "Start Time: $(date)"

# Step 1: Identify Failure Reason (0-3 minutes)
echo "Step 1: Identify Failure Reason"

# Check deployment status
echo "Checking deployment status:"
kubectl describe llmdeployment $DEPLOYMENT_NAME -n $NAMESPACE

# Check pod status
echo "Checking pod status:"
kubectl get pods -n $NAMESPACE -l app=$DEPLOYMENT_NAME
kubectl describe pod -n $NAMESPACE -l app=$DEPLOYMENT_NAME

# Step 2: Check Common Issues (3-6 minutes)
echo "Step 2: Check Common Issues"

# Check image pull
echo "Checking image pull issues:"
kubectl get events -n $NAMESPACE --field-selector involvedObject.name=$DEPLOYMENT_NAME | grep -i "pull"

# Check resource availability
echo "Checking resource availability:"
kubectl describe nodes | grep -A5 "Allocated resources"

# Check storage
echo "Checking storage:"
kubectl get pvc -n $NAMESPACE

# Step 3: Apply Standard Fixes (6-10 minutes)
echo "Step 3: Apply Standard Fixes"

# Delete failed pods
echo "Deleting failed pods:"
kubectl delete pods -n $NAMESPACE -l app=$DEPLOYMENT_NAME --field-selector 'status.phase!=Running'

# Check if deployment needs resource adjustment
CURRENT_REQUESTS=$(kubectl get llmdeployment $DEPLOYMENT_NAME -n $NAMESPACE -o jsonpath='{.spec.resources.requests}')
echo "Current requests: $CURRENT_REQUESTS"

# Reduce resources if needed
echo "Adjusting resources if needed:"
kubectl patch llmdeployment $DEPLOYMENT_NAME -n $NAMESPACE --type='merge' -p='{
  "spec": {
    "resources": {
      "requests": {"memory": "8Gi", "nvidia.com/gpu": "1"},
      "limits": {"memory": "16Gi", "nvidia.com/gpu": "1"}
    }
  }
}'

# Step 4: Monitor Recovery (10-15 minutes)
echo "Step 4: Monitor Recovery"

# Watch pod startup
echo "Watching pod startup:"
kubectl get pods -n $NAMESPACE -l app=$DEPLOYMENT_NAME -w &
WATCH_PID=$!

# Check readiness
echo "Waiting for readiness:"
kubectl wait --for=condition=ready pod -l app=$DEPLOYMENT_NAME -n $NAMESPACE --timeout=300s

# Stop watching
kill $WATCH_PID 2>/dev/null

echo "=== SOP-004 COMPLETED ==="
echo "End Time: $(date)"

# Verification steps
echo "=== VERIFICATION ==="
echo "Deployment status:"
kubectl get llmdeployment $DEPLOYMENT_NAME -n $NAMESPACE -o jsonpath='{.status.phase}'

echo "Health check:"
kubectl exec -n $NAMESPACE deployment/$DEPLOYMENT_NAME -- curl -f localhost:8080/health