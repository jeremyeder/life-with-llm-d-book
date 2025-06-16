---
title: Command Reference
description: Complete kubectl command reference for llm-d operations organized by workflow
sidebar_position: 2
---

# Appendix B: Command Reference

This appendix provides a comprehensive reference of kubectl commands for llm-d operations, organized by workflow. All commands are cohesive throughout the book and include practical examples with expected outputs.

## Deployment Workflows

### Creating and Managing LLM Deployments

#### Deploy a New Model

```bash
# Create a basic LLM deployment
kubectl apply -f - <<EOF
apiVersion: inference.llm-d.io/v1alpha1
kind: LLMDeployment
metadata:
  name: llama-3.1-8b
  namespace: production
spec:
  model:
    modelUri: "hf://meta-llama/Llama-3.1-8B-Instruct"
    quantization:
      type: "int8"
  resources:
    requests:
      memory: "16Gi"
      cpu: "4"
      nvidia.com/gpu: "1"
    limits:
      memory: "24Gi"
      cpu: "8"
      nvidia.com/gpu: "1"
EOF

# Expected output:
# llmdeployment.inference.llm-d.io/llama-3.1-8b created
```

#### List LLM Deployments

```bash
# List all LLM deployments
kubectl get llmdeployments -A

# List deployments in specific namespace
kubectl get llmdeployments -n production

# Get detailed view with custom columns
kubectl get llmdeployments -n production -o custom-columns="NAME:.metadata.name,MODEL:.spec.model.modelUri,REPLICAS:.status.replicas,STATUS:.status.phase"
```

**Expected output:**
```
NAME             MODEL                                          REPLICAS   STATUS
llama-3.1-8b     hf://meta-llama/Llama-3.1-8B-Instruct        2          Running
llama-3.1-70b    hf://meta-llama/Llama-3.1-70B-Instruct       4          Running
```

#### Check Deployment Status

```bash
# Get detailed deployment information
kubectl describe llmdeployment llama-3.1-8b -n production

# Check deployment readiness
kubectl get llmdeployment llama-3.1-8b -n production -o jsonpath='{.status.phase}'

# Watch deployment status changes
kubectl get llmdeployments -n production -w
```

#### Update Model Configuration

```bash
# Update quantization settings
kubectl patch llmdeployment llama-3.1-8b -n production --type='merge' -p='
{
  "spec": {
    "model": {
      "quantization": {
        "type": "int4"
      }
    }
  }
}'

# Scale replicas
kubectl patch llmdeployment llama-3.1-8b -n production --type='merge' -p='
{
  "spec": {
    "autoscaling": {
      "minReplicas": 3,
      "maxReplicas": 12
    }
  }
}'
```

**Cross-references:**

- Chapter 4: [Data Scientist Workflows](../04-data-scientist-workflows.md#deploying-models)
- Chapter 12: [MLOps for SREs](../12-mlops-for-sres.md#deployment-procedures)

## Monitoring and Observability

### Resource Monitoring

#### Check GPU Utilization

```bash
# View GPU allocation across nodes
kubectl get nodes -l node-type=gpu -o custom-columns="NAME:.metadata.name,GPU_ALLOCATABLE:.status.allocatable.nvidia\.com/gpu,GPU_CAPACITY:.status.capacity.nvidia\.com/gpu"

# Check GPU usage by pods
kubectl top pods -n production --containers | grep nvidia

# Detailed GPU information
kubectl describe nodes -l node-type=gpu | grep -A 5 "nvidia.com/gpu"
```

**Expected output:**
```
NAME                     GPU_ALLOCATABLE   GPU_CAPACITY
gpu-node-1              8                  8
gpu-node-2              8                  8
gpu-node-3              4                  8
```

#### Monitor Model Performance

```bash
# Get pod metrics
kubectl top pods -n production -l app.kubernetes.io/name=llm-d

# Check resource consumption
kubectl get pods -n production -o custom-columns="NAME:.metadata.name,CPU_REQ:.spec.containers[*].resources.requests.cpu,MEM_REQ:.spec.containers[*].resources.requests.memory,GPU_REQ:.spec.containers[*].resources.requests.nvidia\.com/gpu"

# Monitor autoscaling status
kubectl get hpa -n production
```

#### View Metrics and Logs

```bash
# Get model serving logs
kubectl logs -n production -l app.kubernetes.io/name=llm-d --tail=100

# Follow logs in real-time
kubectl logs -n production -f deployment/llama-3.1-8b

# Get logs from specific replica
kubectl logs -n production llama-3.1-8b-7c8f9d5b6-xyz12

# Check previous container logs (useful for crashed containers)
kubectl logs -n production llama-3.1-8b-7c8f9d5b6-xyz12 --previous
```

**Cross-references:**

- Chapter 5: [SRE Operations](../05-sre-operations.md#monitoring-commands)
- Chapter 6: [Performance Optimization](../06-performance-optimization.md#performance-monitoring)

## Troubleshooting Workflows

### Diagnostic Commands

#### Pod and Container Issues

```bash
# Check pod status and events
kubectl get pods -n production -l app.kubernetes.io/name=llm-d
kubectl describe pod <pod-name> -n production

# Get events for troubleshooting
kubectl get events -n production --sort-by='.lastTimestamp' | tail -20

# Check resource constraints
kubectl describe nodes | grep -A 5 "Allocated resources"

# Verify image pull status
kubectl get pods -n production -o jsonpath='{.items[*].status.containerStatuses[*].state}'
```

#### Network and Service Issues

```bash
# Check service endpoints
kubectl get endpoints -n production
kubectl describe service llama-3.1-8b-service -n production

# Test service connectivity
kubectl run debug-pod --image=curlimages/curl -it --rm -- sh
# From within the pod:
# curl http://llama-3.1-8b-service.production:8080/health

# Check network policies
kubectl get networkpolicies -n production
kubectl describe networkpolicy <policy-name> -n production
```

#### Configuration and Validation Issues

```bash
# Validate YAML configuration
kubectl apply --dry-run=client -f deployment.yaml

# Check CRD installation
kubectl get crds | grep llm-d
kubectl describe crd llmdeployments.inference.llm-d.io

# Verify RBAC permissions
kubectl auth can-i create llmdeployments --as=system:serviceaccount:production:default -n production

# Check resource quotas
kubectl describe quota -n production
kubectl describe limitrange -n production
```

**Expected diagnostic output:**
```bash
# Pod status check
NAME                            READY   STATUS    RESTARTS   AGE
llama-3.1-8b-7c8f9d5b6-abc12   1/1     Running   0          5m
llama-3.1-8b-7c8f9d5b6-def34   1/1     Running   0          3m

# Service endpoints
NAME               ENDPOINTS                           AGE
llama-3.1-8b-service   10.244.1.15:8080,10.244.2.20:8080  5m
```

**Cross-references:**
- Chapter 8: [Troubleshooting Guide](../08-troubleshooting/03-diagnostic-tools.md)
- Chapter 12: [MLOps for SREs](../12-mlops-for-sres.md#incident-response)

## Security and Access Control

### RBAC Management

#### Create Service Accounts and Roles

```bash
# Create service account for model deployments
kubectl create serviceaccount llm-deployer -n production

# Create role with specific permissions
kubectl apply -f - <<EOF
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: production
  name: llm-deployment-manager
rules:
- apiGroups: ["inference.llm-d.io"]
  resources: ["llmdeployments", "modelservices"]
  verbs: ["get", "list", "create", "update", "patch", "watch"]
- apiGroups: [""]
  resources: ["pods", "services", "configmaps"]
  verbs: ["get", "list", "watch"]
EOF

# Bind role to service account
kubectl create rolebinding llm-deployer-binding \
  --role=llm-deployment-manager \
  --serviceaccount=production:llm-deployer \
  -n production
```

#### Check Permissions

```bash
# Test specific permissions
kubectl auth can-i create llmdeployments -n production
kubectl auth can-i delete pods -n production --as=system:serviceaccount:production:llm-deployer

# List available resources
kubectl api-resources | grep llm-d

# Check current user permissions
kubectl auth whoami
kubectl describe clusterrolebinding | grep $(kubectl auth whoami -o jsonpath='{.status.userInfo.username}')
```

### Security Validation

```bash
# Check pod security contexts
kubectl get pods -n production -o jsonpath='{.items[*].spec.securityContext}'

# Verify network policies
kubectl get networkpolicies -n production -o yaml

# Check for privileged containers
kubectl get pods -n production -o jsonpath='{.items[*].spec.containers[*].securityContext.privileged}'

# Scan for security issues (requires security scanner)
kubectl get pods -n production -o json | jq '.items[] | select(.spec.containers[].securityContext.runAsRoot == true)'
```

**Cross-references:**
- Chapter 7: [Security and Compliance](../07-security-compliance.md#rbac-configuration)

## Scaling and Performance Management

### Horizontal Pod Autoscaling

#### Configure Autoscaling

```bash
# Create HPA for LLM deployment
kubectl autoscale deployment llama-3.1-8b \
  --cpu-percent=70 \
  --min=2 \
  --max=10 \
  -n production

# Create custom HPA with GPU metrics
kubectl apply -f - <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llama-3.1-8b-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llama-3.1-8b
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: nvidia_gpu_utilization_percent
      target:
        type: AverageValue
        averageValue: "80"
EOF
```

#### Monitor Scaling

```bash
# Check HPA status
kubectl get hpa -n production
kubectl describe hpa llama-3.1-8b-hpa -n production

# Watch scaling events
kubectl get events -n production --field-selector reason=ScalingReplicaSet -w

# Check current replica count
kubectl get deployment llama-3.1-8b -n production -o jsonpath='{.status.replicas}'
```

**Expected HPA output:**
```
NAME                REFERENCE               TARGETS         MINPODS   MAXPODS   REPLICAS   AGE
llama-3.1-8b-hpa   Deployment/llama-3.1-8b   45%/70%, 65%/80%   2         10        4          5m
```

### Resource Management

#### Node Management

```bash
# List GPU nodes and their status
kubectl get nodes -l node-type=gpu

# Cordon a node (prevent new pods)
kubectl cordon gpu-node-1

# Drain a node safely
kubectl drain gpu-node-1 --ignore-daemonsets --force --delete-emptydir-data

# Uncordon a node
kubectl uncordon gpu-node-1

# Check node resource allocation
kubectl describe node gpu-node-1 | grep -A 10 "Allocated resources"
```

#### Resource Quotas

```bash
# Check current resource usage against quotas
kubectl describe quota -n production

# Create resource quota for namespace
kubectl apply -f - <<EOF
apiVersion: v1
kind: ResourceQuota
metadata:
  name: llm-quota
  namespace: production
spec:
  hard:
    requests.nvidia.com/gpu: "16"
    limits.nvidia.com/gpu: "16"
    requests.memory: "256Gi"
    limits.memory: "512Gi"
    pods: "50"
EOF
```

**Cross-references:**
- Chapter 6: [Performance Optimization](../06-performance-optimization.md#scaling-strategies)
- Chapter 11: [Cost Optimization](../11-cost-optimization.md#resource-management)

## Backup and Recovery

### Configuration Backup

```bash
# Backup all LLM deployments
kubectl get llmdeployments -n production -o yaml > llm-deployments-backup.yaml

# Backup specific deployment configuration
kubectl get llmdeployment llama-3.1-8b -n production -o yaml > llama-3.1-8b-backup.yaml

# Backup all resources in namespace
kubectl get all,configmaps,secrets,llmdeployments -n production -o yaml > production-backup.yaml

# Create namespace backup with labels
kubectl get all -n production -l app.kubernetes.io/name=llm-d -o yaml > llm-d-backup.yaml
```

### Recovery Operations

```bash
# Restore from backup
kubectl apply -f llm-deployments-backup.yaml

# Rollback to previous deployment version
kubectl rollout undo deployment/llama-3.1-8b -n production

# Check rollout history
kubectl rollout history deployment/llama-3.1-8b -n production

# Rollback to specific revision
kubectl rollout undo deployment/llama-3.1-8b --to-revision=2 -n production

# Verify rollback status
kubectl rollout status deployment/llama-3.1-8b -n production
```

### Disaster Recovery

```bash
# Check cluster health
kubectl get nodes
kubectl get pods -n kube-system

# Recreate namespace if needed
kubectl create namespace production

# Apply resource quotas and policies
kubectl apply -f production-policies.yaml

# Restore deployments in order
kubectl apply -f llm-deployments-backup.yaml
kubectl wait --for=condition=Ready pod -l app.kubernetes.io/name=llm-d -n production --timeout=300s
```

**Cross-references:**
- Chapter 12: [MLOps for SREs](../12-mlops-for-sres.md#backup-procedures)

## Performance Testing and Validation

### Load Testing

```bash
# Create test pod for load generation
kubectl run load-test --image=curlimages/curl -it --rm -- sh

# From within the test pod, generate load:
# while true; do curl -X POST -H "Content-Type: application/json" \
#   -d '{"prompt": "Hello world", "max_tokens": 10}' \
#   http://llama-3.1-8b-service.production:8080/v1/completions; sleep 1; done

# Monitor during load test
kubectl top pods -n production --containers
kubectl get hpa -n production -w
```

### Performance Validation

```bash
# Check model response times
kubectl exec -n production deployment/llama-3.1-8b -- curl -w "@curl-format.txt" \
  -X POST -H "Content-Type: application/json" \
  -d '{"prompt": "test", "max_tokens": 1}' \
  http://localhost:8080/v1/completions

# Validate GPU utilization
kubectl exec -n production <pod-name> -- nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits

# Check memory usage
kubectl top pods -n production -l app.kubernetes.io/name=llm-d --containers
```

**Cross-references:**
- Chapter 6: [Performance Optimization](../06-performance-optimization.md#performance-testing)

## Maintenance and Updates

### Model Updates

```bash
# Update model version
kubectl patch llmdeployment llama-3.1-8b -n production --type='merge' -p='
{
  "spec": {
    "model": {
      "version": "v1.3.0"
    }
  }
}'

# Watch rollout progress
kubectl rollout status deployment/llama-3.1-8b -n production

# Verify update
kubectl get llmdeployment llama-3.1-8b -n production -o jsonpath='{.spec.model.version}'
```

### System Maintenance

```bash
# Check for outdated images
kubectl get pods -n production -o jsonpath='{.items[*].spec.containers[*].image}' | tr ' ' '\n' | sort -u

# Update operator
kubectl get pods -n llm-d-system
kubectl delete pods -n llm-d-system -l app.kubernetes.io/name=llm-d-operator

# Clean up completed jobs
kubectl delete jobs -n production --field-selector status.successful=1

# Remove unused ConfigMaps and Secrets
kubectl get configmaps,secrets -n production --sort-by=.metadata.creationTimestamp
```

### Cluster Information

```bash
# Get cluster version information
kubectl version --short

# Check available storage classes
kubectl get storageclass

# List available node selectors
kubectl get nodes --show-labels | grep node-type

# Check cluster resource capacity
kubectl describe nodes | grep -A 4 "Capacity:\|Allocatable:"
```

**Cross-references:**
- Chapter 5: [SRE Operations](../05-sre-operations.md#maintenance-procedures)
- Chapter 12: [MLOps for SREs](../12-mlops-for-sres.md#operational-workflows)

## Quick Reference Commands

### Daily Operations Checklist

```bash
#!/bin/bash
# Daily LLM infrastructure health check

echo "🔍 Daily LLM Health Check - $(date)"
echo "=================================="

# Check all deployments
kubectl get llmdeployments -A

# Check node health
kubectl get nodes -l node-type=gpu

# Check resource utilization
kubectl top nodes -l node-type=gpu

# Check for failed pods
kubectl get pods -A --field-selector=status.phase=Failed

# Check HPA status
kubectl get hpa -A

echo "✅ Health check complete"
```

### Emergency Response Commands

```bash
# Quick deployment restart
kubectl rollout restart deployment/llama-3.1-8b -n production

# Scale down immediately (emergency)
kubectl scale deployment llama-3.1-8b --replicas=0 -n production

# Check recent errors
kubectl get events -n production --sort-by='.lastTimestamp' | tail -10

# Get pod status for all deployments
kubectl get pods -n production -l app.kubernetes.io/name=llm-d -o wide
```

### Common Aliases

Add these to your `~/.bashrc` or `~/.zshrc`:

```bash
# kubectl aliases for llm-d operations
alias kgld='kubectl get llmdeployments'
alias kdld='kubectl describe llmdeployment'
alias kgp='kubectl get pods'
alias kgn='kubectl get nodes'
alias ktn='kubectl top nodes'
alias ktp='kubectl top pods'
alias kgpu='kubectl get nodes -l node-type=gpu'
```

---

:::info Command Best Practices
- Always specify namespace with `-n` flag for production commands
- Use `--dry-run=client` to validate configurations before applying
- Include timeouts in wait commands: `--timeout=300s`
- Use labels for bulk operations: `-l app.kubernetes.io/name=llm-d`
- Check resource quotas before scaling operations
:::

:::tip Next Steps
- Review [CRD Reference](./crd-reference.md) for resource specifications
- Check [Configuration Templates](./configuration-templates.md) for ready-to-use configurations
- Reference [Shared Configuration](./shared-config.md) for naming conventions and standards
- Reference main chapters for context-specific command usage
:::
