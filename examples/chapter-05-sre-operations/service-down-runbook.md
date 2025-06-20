# Runbook: LLM-D Service Down

## Overview
Complete service outage - all inference requests failing

## Severity: P0 (Critical)
**SLO Impact:** Availability SLO breach
**Response Time:** 5 minutes

## Immediate Response

### 1. Verify Outage Scope
```bash
# Check all service endpoints
kubectl get pods -n production -o wide

# Check service status
kubectl get svc -n production

# Verify ingress configuration
kubectl get ingress -n production
```

### 2. Check Recent Changes

```bash
# Check recent deployments
kubectl rollout history deployment/llm-d-service -n production

# Check recent configuration changes
kubectl get events -n production --sort-by='.lastTimestamp'

# Review recent Helm releases
helm history llm-d -n production
```

### 3. Check Infrastructure Health

```bash
# Check node status
kubectl get nodes

# Check cluster-level events
kubectl get events --all-namespaces --sort-by='.lastTimestamp'

# Check storage health
kubectl get pv,pvc -n production
```

## Recovery Actions

### Immediate Rollback (if deployment-related)

```bash
# Rollback to previous version
kubectl rollout undo deployment/llm-d-service -n production

# Monitor rollback progress
kubectl rollout status deployment/llm-d-service -n production
```

### Pod Recovery

```bash
# Force pod restart
kubectl delete pods -l app=llm-d-service -n production

# Check pod startup progress
kubectl get pods -n production -w
```

### Service Recovery

```bash
# Recreate services if needed
kubectl delete svc llm-d-service -n production
kubectl apply -f manifests/service.yaml
```

## Validation Steps

```bash
# Test service health
curl -f http://llm-d-service:8080/health

# Test inference endpoint
curl -X POST http://llm-d-service:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3-8b", "messages": [{"role": "user", "content": "test"}]}'

# Verify metrics are flowing
kubectl exec -n monitoring prometheus-0 -- \
  promtool query instant 'up{job="llm-d-service"}'
```

## Communication Template

```
🚨 INCIDENT UPDATE - LLM-D Service Outage

Status: [INVESTIGATING/MITIGATING/RESOLVED]
Severity: P0 (Critical)
Impact: Complete service outage affecting all inference requests
Start Time: [TIME]

Current Actions:
- [Action 1]
- [Action 2]

ETA to Resolution: [TIME]
Next Update: [TIME]

Incident Commander: @[NAME]
```