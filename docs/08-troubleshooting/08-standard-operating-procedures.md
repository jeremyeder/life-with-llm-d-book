---
title: Standard Operating Procedures
description: Step-by-step SOPs for common llm-d operational tasks
sidebar_position: 8
---

# Standard Operating Procedures (SOPs)

This section provides standardized, step-by-step procedures that any on-call engineer can follow to handle common operational scenarios. These SOPs are designed for consistent execution under pressure.

## SOP Format

Each SOP follows this structure:

- **Objective**: What you're trying to achieve
- **Prerequisites**: What you need before starting
- **Steps**: Numbered, actionable steps
- **Verification**: How to confirm success
- **Rollback**: How to undo if needed
- **Escalation**: When and how to escalate

## SOP-001: Service Outage Response

### Objective

Restore service availability for complete llm-d outage

### Prerequisites

- kubectl access to affected cluster
- Slack/communication channel access
- Incident tracking system access

### Steps

1. **Acknowledge and Communicate** (0-2 minutes)

   ```bash
   # Post in incident channel
   echo "🚨 P0 INCIDENT: LLM service outage detected at $(date)"
   echo "Incident Commander: [YOUR_NAME]"
   echo "Status: Investigating"
   ```

2. **Quick Health Check** (2-5 minutes)

   ```bash
   # Check operator status
   kubectl get pods -n llm-d-system
   
   # Check deployments
   kubectl get llmdeployments -A
   
   # Check nodes
   kubectl get nodes
   ```

3. **Identify Scope** (5-7 minutes)

   ```bash
   # Count affected deployments
   FAILED_DEPLOYMENTS=$(kubectl get llmdeployments -A -o jsonpath='{.items[?(@.status.phase!="Ready")].metadata.name}' | wc -w)
   echo "Failed deployments: $FAILED_DEPLOYMENTS"
   
   # Check if operator is the issue
   kubectl logs -n llm-d-system deployment/llm-d-operator --tail=20
   ```

4. **Apply Standard Fixes** (7-15 minutes)

   ```bash
   # Restart operator if unhealthy
   if kubectl get pods -n llm-d-system | grep -q "0/1.*Running\|Error\|CrashLoop"; then
     kubectl rollout restart deployment/llm-d-operator -n llm-d-system
     kubectl rollout status deployment/llm-d-operator -n llm-d-system --timeout=180s
   fi
   
   # Restart failed deployments
   kubectl get llmdeployments -A -o json | jq -r '.items[] | select(.status.phase != "Ready") | "\(.metadata.namespace) \(.metadata.name)"' | while read ns name; do
     kubectl rollout restart deployment/$name -n $ns
   done
   ```

5. **Enable Maintenance Mode** (if needed)

   ```bash
   # If restarts don't work, enable maintenance page
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
   ```

### Verification

```bash
# Check service health
curl -f http://your-llm-service/health || echo "Still down"

# Check all deployments ready
kubectl get llmdeployments -A --no-headers | awk '$4 != "Ready" {print $0}' | wc -l
```

### Rollback

```bash
# Remove maintenance mode
kubectl delete ingress maintenance-mode

# Revert to previous operator version if restart failed
kubectl set image deployment/llm-d-operator llm-d-operator=llm-d/operator:v0.4.0 -n llm-d-system
```

### Escalation

- **15 minutes**: Escalate to senior SRE if no progress
- **30 minutes**: Engage engineering team
- **45 minutes**: Consider rolling back recent changes

---

## SOP-002: High Memory Usage Response

### Objective

Resolve memory pressure before OOM kills occur

### Prerequisites

- kubectl access
- Understanding of current resource limits

### Steps

1. **Identify High Memory Pods** (0-3 minutes)

   ```bash
   # Find top memory consumers
   kubectl top pods -A --sort-by=memory | head -10
   
   # Check for memory warnings
   kubectl get events -A --field-selector type=Warning | grep -i memory
   ```

2. **Check Memory Limits** (3-5 minutes)

   ```bash
   # For each high-memory pod, check limits
   HIGH_MEMORY_PODS=$(kubectl top pods -A --sort-by=memory --no-headers | head -5 | awk '{print $2 " " $1}')
   echo "$HIGH_MEMORY_PODS" | while read pod namespace; do
     echo "=== $pod in $namespace ==="
     kubectl get pod $pod -n $namespace -o jsonpath='{.spec.containers[0].resources.limits.memory}'
     echo ""
   done
   ```

3. **Apply Immediate Relief** (5-10 minutes)

   ```bash
   # Reduce batch sizes
   kubectl get llmdeployments -A -o json | jq -r '.items[] | "\(.metadata.namespace) \(.metadata.name)"' | while read ns name; do
     kubectl patch llmdeployment $name -n $ns --type='merge' -p='{"spec":{"serving":{"batchSize":1,"maxConcurrency":2}}}'
   done
   
   # Restart highest memory consumer
   HIGHEST_POD=$(kubectl top pods -A --sort-by=memory --no-headers | head -1 | awk '{print $2 " " $1}')
   echo "$HIGHEST_POD" | while read pod namespace; do
     kubectl delete pod $pod -n $namespace
   done
   ```

4. **Scale if Necessary** (10-15 minutes)

   ```bash
   # Scale down replicas temporarily
   kubectl get llmdeployments -A -o json | jq -r '.items[] | "\(.metadata.namespace) \(.metadata.name) \(.spec.replicas)"' | while read ns name replicas; do
     if [ $replicas -gt 2 ]; then
       new_replicas=$((replicas / 2))
       kubectl patch llmdeployment $name -n $ns --type='merge' -p="{\"spec\":{\"replicas\":$new_replicas}}"
     fi
   done
   ```

### Verification

```bash
# Check memory usage dropped
kubectl top pods -A --sort-by=memory | head -5

# No OOM events in last 10 minutes
kubectl get events -A --field-selector type=Warning | grep -i oom | grep "$(date -d '10 minutes ago' '+%H:%M')"
```

### Rollback

```bash
# Restore original batch sizes and replicas
# (This would require storing original values)
```

### Escalation

- **10 minutes**: If memory usage doesn't decrease
- **20 minutes**: If OOM kills continue

---

## SOP-003: GPU Not Available Response

### Objective

Restore GPU availability for model deployments

### Prerequisites

- Node access (via kubectl debug or ssh)
- Understanding of GPU node architecture

### Steps

1. **Verify Problem Scope** (0-3 minutes)

   ```bash
   # Check GPU resource availability
   kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.capacity."nvidia\.com/gpu" | grep -v '<none>'
   
   # Check device plugin status
   kubectl get ds -n kube-system nvidia-device-plugin-daemonset
   kubectl get pods -n kube-system -l name=nvidia-device-plugin-ds
   ```

2. **Check Driver Status** (3-6 minutes)

   ```bash
   # For each GPU node
   GPU_NODES=$(kubectl get nodes -l nvidia.com/gpu=true -o jsonpath='{.items[*].metadata.name}')
   for node in $GPU_NODES; do
     echo "=== Checking $node ==="
     kubectl debug node/$node -it --image=ubuntu -- chroot /host nvidia-smi
   done
   ```

3. **Restart Device Plugin** (6-8 minutes)

   ```bash
   # Restart device plugin daemonset
   kubectl rollout restart ds nvidia-device-plugin-daemonset -n kube-system
   kubectl rollout status ds nvidia-device-plugin-daemonset -n kube-system --timeout=180s
   ```

4. **Verify GPU Visibility** (8-10 minutes)

   ```bash
   # Check GPU resources are back
   kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.capacity."nvidia\.com/gpu"
   
   # Test GPU allocation
   kubectl run gpu-test --image=nvidia/cuda:11.0-base --rm -it --restart=Never \
     --limits='nvidia.com/gpu=1' -- nvidia-smi
   ```

### Verification

```bash
# All GPU nodes showing capacity
kubectl get nodes -l nvidia.com/gpu=true -o custom-columns=NAME:.metadata.name,GPU:.status.capacity."nvidia\.com/gpu" | grep -v "0\|<none>"

# Device plugin pods running
kubectl get pods -n kube-system -l name=nvidia-device-plugin-ds | grep Running
```

### Rollback

```bash
# If driver restart fails, may need to reboot nodes
# This requires approval from incident commander
```

### Escalation

- **10 minutes**: If device plugin restart doesn't work
- **15 minutes**: If driver issues detected

---

## SOP-004: Model Deployment Failure Response

### Objective

Get failed model deployment back to healthy state

### Prerequisites

- Access to model storage
- Knowledge of deployment configuration

### Steps

1. **Identify Failure Reason** (0-3 minutes)

   ```bash
   DEPLOYMENT_NAME=$1
   NAMESPACE=$2
   
   # Check deployment status
   kubectl describe llmdeployment $DEPLOYMENT_NAME -n $NAMESPACE
   
   # Check pod status
   kubectl get pods -n $NAMESPACE -l app=$DEPLOYMENT_NAME
   kubectl describe pod -n $NAMESPACE -l app=$DEPLOYMENT_NAME
   ```

2. **Check Common Issues** (3-6 minutes)

   ```bash
   # Check image pull
   kubectl get events -n $NAMESPACE --field-selector involvedObject.name=$DEPLOYMENT_NAME | grep -i "pull"
   
   # Check resource availability
   kubectl describe nodes | grep -A5 "Allocated resources"
   
   # Check storage
   kubectl get pvc -n $NAMESPACE
   ```

3. **Apply Standard Fixes** (6-10 minutes)

   ```bash
   # Delete failed pods
   kubectl delete pods -n $NAMESPACE -l app=$DEPLOYMENT_NAME --field-selector 'status.phase!=Running'
   
   # Check if deployment needs resource adjustment
   CURRENT_REQUESTS=$(kubectl get llmdeployment $DEPLOYMENT_NAME -n $NAMESPACE -o jsonpath='{.spec.resources.requests}')
   echo "Current requests: $CURRENT_REQUESTS"
   
   # Reduce resources if needed
   kubectl patch llmdeployment $DEPLOYMENT_NAME -n $NAMESPACE --type='merge' -p='{
     "spec": {
       "resources": {
         "requests": {"memory": "8Gi", "nvidia.com/gpu": "1"},
         "limits": {"memory": "16Gi", "nvidia.com/gpu": "1"}
       }
     }
   }'
   ```

4. **Monitor Recovery** (10-15 minutes)

   ```bash
   # Watch pod startup
   kubectl get pods -n $NAMESPACE -l app=$DEPLOYMENT_NAME -w
   
   # Check readiness
   kubectl wait --for=condition=ready pod -l app=$DEPLOYMENT_NAME -n $NAMESPACE --timeout=300s
   ```

### Verification

```bash
# Deployment shows Ready
kubectl get llmdeployment $DEPLOYMENT_NAME -n $NAMESPACE -o jsonpath='{.status.phase}'

# Health check passes
kubectl exec -n $NAMESPACE deployment/$DEPLOYMENT_NAME -- curl -f localhost:8080/health
```

### Rollback

```bash
# Restore original resource configuration
kubectl patch llmdeployment $DEPLOYMENT_NAME -n $NAMESPACE --type='merge' -p='{"spec":{"resources":{"requests":{"memory":"16Gi"}}}}'
```

### Escalation

- **15 minutes**: If deployment still failing after resource adjustment
- **30 minutes**: If multiple deployments affected

---

## Quick Reference Cards

### 🚨 P0 Incident Checklist

```
□ Acknowledge in #incident-response
□ kubectl get llmdeployments -A
□ kubectl get pods -n llm-d-system  
□ kubectl logs -n llm-d-system deployment/llm-d-operator
□ Restart operator if needed
□ Enable maintenance mode if required
□ Update stakeholders every 15 minutes
□ Escalate at 15/30/45 minute marks
```

### 🔧 Common Commands

```bash
# Health checks
kubectl get llmdeployments -A
kubectl get pods -n llm-d-system
kubectl get nodes -l nvidia.com/gpu=true

# Quick restarts  
kubectl rollout restart deployment/llm-d-operator -n llm-d-system
kubectl delete pods -l app=<model> -n <namespace>

# Resource checks
kubectl top nodes
kubectl top pods -A --sort-by=memory
kubectl describe nodes | grep -A5 "Allocated resources"

# GPU debugging
kubectl debug node/<node> -it --image=ubuntu -- chroot /host nvidia-smi
kubectl get ds -n kube-system nvidia-device-plugin-daemonset
```

### 📞 Escalation Path

```
0-15 min:  On-call SRE
15-30 min: Senior SRE  
30-45 min: Engineering team
45+ min:   Incident Commander + Engineering Manager
```

### 🎯 Key Metrics to Watch

```
- llm_deployments_ready vs llm_deployments_total
- GPU utilization < 90%
- Memory usage < 85% of limits
- Request error rate < 5%
- P95 latency < 2x baseline
```

## SOP Maintenance

### When to Update SOPs

- After any P0/P1 incident
- When infrastructure changes
- When new failure modes discovered
- Quarterly review cycles

### SOP Testing

- Monthly SOP walkthroughs
- Quarterly disaster recovery drills
- New team member training
- Post-incident validation

---

**Remember**: SOPs are living documents. If you find a step that doesn't work or could be improved, create a follow-up task to update it.
