#!/bin/bash
"""
P0 Critical Incident: Complete Service Outage Response

This script provides immediate response procedures for complete LLM service outages.
Execute this script as soon as a P0 service outage is detected.

Usage:
    ./P0-service-outage.sh

Execute immediately upon detecting complete service outage.
"""

echo "=== P0 INCIDENT: COMPLETE SERVICE OUTAGE ==="
echo "Timestamp: $(date)"
echo "Incident ID: INC-$(date +%Y%m%d-%H%M%S)"

# 1. Verify outage scope
echo "1. Checking service status..."
kubectl get llmdeployments -A
kubectl get pods -A -l app.kubernetes.io/name=llm-d

# 2. Check cluster health
echo "2. Checking cluster health..."
kubectl get nodes
kubectl cluster-info

# 3. Check operator status
echo "3. Checking operator status..."
kubectl get pods -n llm-d-system
kubectl logs -n llm-d-system deployment/llm-d-operator --tail=50

# 4. Enable maintenance mode
echo "4. Enabling maintenance mode..."
kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: maintenance-mode
  namespace: default
spec:
  selector:
    app: maintenance-page
  ports:
  - port: 80
    targetPort: 8080
EOF

# 5. Notify stakeholders
echo "5. Sending notifications..."
curl -X POST "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK" \
  -H 'Content-type: application/json' \
  --data '{
    "text": "🚨 P0 INCIDENT: Complete LLM service outage detected",
    "channel": "#incident-response",
    "username": "llm-d-monitor"
  }'

# Recovery Actions (5-15 minutes)
echo "6. Starting recovery procedures..."

# Check if it's an operator issue
if kubectl get pods -n llm-d-system | grep -q "0/1.*Running"; then
  echo "Operator unhealthy - restarting..."
  kubectl rollout restart deployment/llm-d-operator -n llm-d-system
  kubectl rollout status deployment/llm-d-operator -n llm-d-system --timeout=300s
fi

# Check for node issues
unhealthy_nodes=$(kubectl get nodes | grep -v Ready | grep -v NAME | wc -l)
if [ $unhealthy_nodes -gt 0 ]; then
  echo "Node issues detected - checking GPU nodes..."
  kubectl get nodes -l nvidia.com/gpu=true
  # May need manual intervention
fi

# Restart all deployments as last resort
echo "Performing full restart of all LLM deployments..."
kubectl rollout restart deployment -l app.kubernetes.io/name=llm-d -A