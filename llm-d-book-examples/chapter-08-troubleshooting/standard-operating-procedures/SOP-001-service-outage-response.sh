#!/bin/bash
"""
SOP-001: Service Outage Response

Standardized procedure for responding to complete llm-d service outages.
This SOP provides step-by-step instructions for consistent incident response.

Usage:
    ./SOP-001-service-outage-response.sh

Execute during P0 service outage incidents.
"""

# SOP-001: Service Outage Response
# Objective: Restore service availability for complete llm-d outage

echo "=== SOP-001: SERVICE OUTAGE RESPONSE ==="
echo "Incident Commander: ${USER}"
echo "Start Time: $(date)"

# Step 1: Acknowledge and Communicate (0-2 minutes)
echo "Step 1: Acknowledge and Communicate"
echo "🚨 P0 INCIDENT: LLM service outage detected at $(date)"
echo "Incident Commander: ${USER}"
echo "Status: Investigating"

# Step 2: Quick Health Check (2-5 minutes)
echo "Step 2: Quick Health Check"

# Check operator status
echo "Checking operator status..."
kubectl get pods -n llm-d-system

# Check deployments
echo "Checking deployments..."
kubectl get llmdeployments -A

# Check nodes
echo "Checking nodes..."
kubectl get nodes

# Step 3: Identify Scope (5-7 minutes)
echo "Step 3: Identify Scope"

# Count affected deployments
FAILED_DEPLOYMENTS=$(kubectl get llmdeployments -A -o jsonpath='{.items[?(@.status.phase!="Ready")].metadata.name}' | wc -w)
echo "Failed deployments: $FAILED_DEPLOYMENTS"

# Check if operator is the issue
echo "Checking operator logs..."
kubectl logs -n llm-d-system deployment/llm-d-operator --tail=20

# Step 4: Apply Standard Fixes (7-15 minutes)
echo "Step 4: Apply Standard Fixes"

# Restart operator if unhealthy
if kubectl get pods -n llm-d-system | grep -q "0/1.*Running\|Error\|CrashLoop"; then
  echo "Restarting unhealthy operator..."
  kubectl rollout restart deployment/llm-d-operator -n llm-d-system
  kubectl rollout status deployment/llm-d-operator -n llm-d-system --timeout=180s
fi

# Restart failed deployments
echo "Restarting failed deployments..."
kubectl get llmdeployments -A -o json | jq -r '.items[] | select(.status.phase != "Ready") | "\(.metadata.namespace) \(.metadata.name)"' | while read ns name; do
  kubectl rollout restart deployment/$name -n $ns
done

# Step 5: Enable Maintenance Mode (if needed)
echo "Step 5: Enable Maintenance Mode (if restarts don't work)"
read -p "Enable maintenance mode? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: maintenance-mode
  annotations:
    nginx.ingress.kubernetes.io/default-backend: "maintenance-service"
spec:
  rules:
  - host: "*.your-domain.com"
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: maintenance-service
            port:
              number: 80
EOF
fi

echo "=== SOP-001 COMPLETED ==="
echo "End Time: $(date)"