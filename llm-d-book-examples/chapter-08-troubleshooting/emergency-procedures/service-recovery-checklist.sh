#!/bin/bash
"""
Service Recovery Checklist

This script provides a comprehensive checklist for verifying service recovery
after incident resolution, including health checks and performance validation.

Usage:
    ./service-recovery-checklist.sh

Execute after incident resolution to verify complete recovery.
"""

echo "=== SERVICE RECOVERY CHECKLIST ==="

# Health checks
checks=(
  "kubectl get llmdeployments -A"
  "kubectl get pods -A -l app.kubernetes.io/name=llm-d"
  "kubectl get nodes -l nvidia.com/gpu=true"
  "kubectl get hpa -A"
)

for check in "${checks[@]}"; do
  echo "Running: $check"
  if $check; then
    echo "✅ PASS"
  else
    echo "❌ FAIL"
  fi
  echo ""
done

# Performance validation
echo "=== PERFORMANCE VALIDATION ==="
kubectl port-forward -n default svc/llm-model-service 8080:8080 &
PF_PID=$!
sleep 5

# Test latency
response_time=$(curl -w "%{time_total}" -o /dev/null -s http://localhost:8080/health)
echo "Health check response time: ${response_time}s"

if (( $(echo "$response_time < 1.0" | bc -l) )); then
  echo "✅ Response time acceptable"
else
  echo "❌ Response time too high"
fi

kill $PF_PID