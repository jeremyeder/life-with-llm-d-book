#!/bin/bash
"""
P1 Incident: High Error Rate Response

This script handles high error rate incidents by analyzing error patterns,
scaling resources, and implementing circuit breakers for protection.

Usage:
    ./high-error-rate.sh

Execute when error rates exceed acceptable thresholds (>10%).
"""

echo "=== P1 INCIDENT: HIGH ERROR RATE ==="

# 1. Identify error patterns
echo "1. Analyzing error patterns..."
kubectl logs -A -l app.kubernetes.io/name=llm-d --tail=1000 | \
  grep -i error | \
  awk '{print $NF}' | \
  sort | \
  uniq -c | \
  sort -rn | \
  head -10

# 2. Check resource utilization
echo "2. Checking resource utilization..."
kubectl top nodes
kubectl top pods -A --containers | grep llm-d

# 3. Scale up if resource constrained
echo "3. Emergency scaling..."
for hpa in $(kubectl get hpa -A -o jsonpath='{.items[*].metadata.name}'); do
  namespace=$(kubectl get hpa $hpa -A -o jsonpath='{.items[0].metadata.namespace}')
  current_max=$(kubectl get hpa $hpa -n $namespace -o jsonpath='{.spec.maxReplicas}')
  new_max=$((current_max * 2))
  kubectl patch hpa $hpa -n $namespace --type='merge' -p="{\"spec\":{\"maxReplicas\":$new_max}}"
done

# 4. Implement circuit breaker
kubectl apply -f - <<EOF
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: llm-circuit-breaker
spec:
  host: "*.local"
  trafficPolicy:
    outlierDetection:
      consecutiveGatewayErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
EOF