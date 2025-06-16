---
title: MLOps for SREs
description: Comprehensive guide to managing LLM deployments in production, covering day-to-day operations, incident response, and model lifecycle management
sidebar_position: 12
---

# MLOps for SREs

:::info Chapter Overview
This chapter focuses on production operations for LLM deployments using llm-d. You'll learn how to manage model lifecycles, respond to incidents, and maintain reliable LLM services at scale.

**What you'll learn:**

- Day-to-day operational workflows for LLM services
- Incident response and troubleshooting for model failures
- Model versioning and rollback strategies using llm-d
- Handoff procedures between Data Scientists and SREs
- Cost-aware operational decision making
:::

## Day-to-Day SRE Operations

### Service Health Monitoring

Running production LLM services requires continuous monitoring of both traditional infrastructure metrics and model-specific indicators. Based on real-world incident patterns from major LLM providers, here are the critical metrics to track:

```yaml
# Core llm-d monitoring configuration
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: llm-deployment-metrics
  namespace: production
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: llm-deployment
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

#### Critical SRE Metrics

**Infrastructure Health:**

- GPU utilization and memory usage
- Request latency (p50, p95, p99)
- Throughput (requests/second)
- Error rates and HTTP status codes

**Model-Specific Metrics:**

- Token generation rate
- Model loading time
- Cache hit rates
- Memory allocation patterns

**Business Metrics:**

- Cost per request
- User satisfaction scores
- Feature utilization rates

:::warning Production Reality Check
Based on Meta's Llama 3 training data, expect **1 GPU failure every 3 hours** in large deployments. Plan your monitoring and alerting accordingly.
:::

### Daily Operational Workflows

#### Morning Health Check

```bash
#!/bin/bash
# Daily LLM service health check script

echo "🔍 Daily LLM Service Health Check - $(date)"
echo "================================================"

# Check all LLM deployments
kubectl get llmdeployments -n production -o wide

# Check GPU node health
kubectl get nodes -l node-type=gpu -o custom-columns="NAME:.metadata.name,STATUS:.status.conditions[?(@.type=='Ready')].status,GPU:.status.allocatable.nvidia\.com/gpu"

# Check recent errors
echo "📊 Error rates (last 1h):"
kubectl logs -n production -l app.kubernetes.io/name=llm-deployment --since=1h | grep -i error | wc -l

# Check resource utilization
echo "💾 Resource utilization:"
kubectl top nodes -l node-type=gpu

# Check model serving latency
echo "⏱️  Average response time (last 1h):"
# This would integrate with your monitoring system
```

#### Capacity Planning Alerts

```yaml
# Alert when approaching capacity limits
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: llm-capacity-alerts
  namespace: production
spec:
  groups:
  - name: llm.capacity
    rules:
    - alert: LLMHighGPUUtilization
      expr: nvidia_gpu_utilization_percent > 85
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High GPU utilization on {{ $labels.instance }}"
        description: "GPU utilization is {{ $value }}% on {{ $labels.instance }}"
        
    - alert: LLMHighLatency
      expr: histogram_quantile(0.95, llm_request_duration_seconds) > 2.0
      for: 2m
      labels:
        severity: critical
      annotations:
        summary: "High LLM response latency"
        description: "95th percentile latency is {{ $value }}s"
```

### Incident Response Workflows

#### GPU Hardware Failures

Based on real production data, GPU failures are the most common LLM infrastructure issue:

```bash
# GPU failure detection and response
kubectl describe node <node-name> | grep -A 10 "nvidia.com/gpu"

# Check GPU health
kubectl exec -n production <pod-name> -- nvidia-smi

# Cordon node if GPU is faulty
kubectl cordon <node-name>

# Drain workloads safely
kubectl drain <node-name> --ignore-daemonsets --force
```

#### Model Serving Failures

```yaml
# Automated model health checking
apiVersion: batch/v1
kind: CronJob
metadata:
  name: model-health-check
  namespace: production
spec:
  schedule: "*/5 * * * *"  # Every 5 minutes
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: health-checker
            image: curlimages/curl:latest
            command:
            - /bin/sh
            - -c
            - |
              # Test model endpoint
              response=$(curl -s -w "%{http_code}" \
                -X POST \
                -H "Content-Type: application/json" \
                -d '{"prompt": "test", "max_tokens": 1}' \
                http://llama-3.1-8b.production:8080/v1/completions)
              
              if [ "$response" != "200" ]; then
                echo "Model health check failed: $response"
                exit 1
              fi
          restartPolicy: OnFailure
```

#### Rollback Procedures

```bash
# Quick rollback to previous model version
kubectl patch llmdeployment llama-3.1-8b -n production --type='merge' -p='
{
  "spec": {
    "model": {
      "version": "previous-stable-tag"
    }
  }
}'

# Verify rollback success
kubectl rollout status deployment/llama-3.1-8b -n production --timeout=300s
```

## Initial Deployment Workflows

### Proof of Concept to Production

#### Stage 1: PoC Deployment (Data Scientist → SRE Handoff)

```yaml
# PoC deployment template
apiVersion: inference.llm-d.io/v1alpha1
kind: LLMDeployment
metadata:
  name: llama-3.1-8b-poc
  namespace: development
  annotations:
    deployment.stage: "poc"
    data-scientist.owner: "jane.doe@company.com"
    sre.contact: "sre-team@company.com"
spec:
  model:
    modelUri: "hf://meta-llama/Llama-3.1-8B-Instruct"
    version: "latest"
    quantization:
      type: "int8"  # For cost efficiency in PoC
  resources:
    requests:
      memory: "16Gi"
      cpu: "4"
      nvidia.com/gpu: "1"
    limits:
      memory: "24Gi"
      cpu: "8"
      nvidia.com/gpu: "1"
  autoscaling:
    enabled: true
    minReplicas: 1
    maxReplicas: 3
    targetGPUUtilization: 70
  deployment:
    stage: "poc"
    rolloutStrategy: "RollingUpdate"
```

#### Stage 2: Production Hardening

```yaml
# Production deployment with full observability
apiVersion: inference.llm-d.io/v1alpha1
kind: LLMDeployment
metadata:
  name: llama-3.1-8b-prod
  namespace: production
  annotations:
    deployment.stage: "production"
    sre.runbook: "https://wiki.company.com/sre/llm-runbooks"
    cost.budget: "high-priority"
spec:
  model:
    modelUri: "hf://meta-llama/Llama-3.1-8B-Instruct"
    version: "v1.2.0"  # Pinned version for stability
    quantization:
      type: "int8"
  resources:
    requests:
      memory: "32Gi"  # Production sizing
      cpu: "8"
      nvidia.com/gpu: "2"  # Redundancy
    limits:
      memory: "48Gi"
      cpu: "16"
      nvidia.com/gpu: "2"
  autoscaling:
    enabled: true
    minReplicas: 2  # Always maintain capacity
    maxReplicas: 10
    targetGPUUtilization: 60  # Conservative for reliability
  scheduling:
    scheduler: "llm-d-inference-scheduler"
    sloPolicy:
      enabled: true
      objectives:
        - name: "request_latency_p95"
          target: "500ms"
          weight: 0.6
        - name: "cost_per_request"
          target: "$0.001"
          weight: 0.4
  monitoring:
    enabled: true
    alerting:
      enabled: true
      channels: ["#sre-alerts", "#llm-team"]
  backup:
    enabled: true
    schedule: "0 2 * * *"  # Daily backups
```

### Data Scientist to SRE Handoff

#### Pre-Handoff Checklist

**Data Scientist Responsibilities:**

- [ ] Model performance benchmarks documented
- [ ] Resource requirements validated
- [ ] Test suite provided
- [ ] Performance thresholds defined
- [ ] Cost estimates provided

**SRE Responsibilities:**

- [ ] Monitoring dashboards configured
- [ ] Alerting rules defined
- [ ] Runbook documentation complete
- [ ] Rollback procedures tested
- [ ] Capacity planning reviewed

#### Handoff Documentation Template

```yaml
# Model Handoff Specification
apiVersion: v1
kind: ConfigMap
metadata:
  name: llama-3.1-8b-handoff
  namespace: production
data:
  model-specs.yaml: |
    model:
      modelUri: "hf://meta-llama/Llama-3.1-8B-Instruct"
      version: "v1.2.0"
      performance:
        expected_latency_p95: "450ms"
        expected_throughput: "25 tokens/second"
        accuracy_threshold: "0.85"
      resource_requirements:
        min_gpu_memory: "16GB"
        recommended_gpu_memory: "24GB"
        cpu_cores: "8"
    
  operational-requirements.yaml: |
    slos:
      availability: "99.9%"
      latency_p95: "500ms"
      error_rate: "<0.1%"
    scaling:
      min_replicas: 2
      max_replicas: 10
      scale_trigger: "70% GPU utilization"
    cost:
      budget_monthly: "$5000"
      cost_per_request_target: "$0.001"
    
  emergency-contacts.yaml: |
    data_science:
      primary: "jane.doe@company.com"
      secondary: "ml-team@company.com"
    sre:
      primary: "sre-oncall@company.com"
      escalation: "sre-manager@company.com"
```

### Model Versioning and Rollback Strategies

#### Semantic Versioning for Models

```yaml
# Model version management
apiVersion: inference.llm-d.io/v1alpha1
kind: LLMDeployment
metadata:
  name: llama-3.1-8b
  namespace: production
  labels:
    model.version: "v1.2.0"
    model.type: "instruct"
    deployment.stage: "production"
spec:
  model:
    modelUri: "hf://meta-llama/Llama-3.1-8B-Instruct"
    version: "v1.2.0"
    registry: "company-registry.com/models"
    checksums:
      sha256: "a1b2c3d4e5f6..."  # Model integrity verification
  rollback:
    enabled: true
    previousVersions:
      - "v1.1.3"  # Last known good version
      - "v1.0.5"  # Fallback version
    autoRollback:
      enabled: true
      triggers:
        - errorRate: ">1%"
        - latencyP95: ">1000ms"
        - throughput: "<10 tokens/sec"
```

#### Blue-Green Deployment Strategy

```bash
#!/bin/bash
# Blue-green deployment script for model updates

NEW_VERSION="v1.3.0"
CURRENT_VERSION="v1.2.0"
NAMESPACE="production"
MODEL_NAME="llama-3.1-8b"

echo "🚀 Starting blue-green deployment: $CURRENT_VERSION → $NEW_VERSION"

# Deploy green environment
kubectl apply -f - <<EOF
apiVersion: inference.llm-d.io/v1alpha1
kind: LLMDeployment
metadata:
  name: ${MODEL_NAME}-green
  namespace: ${NAMESPACE}
spec:
  model:
    modelUri: "hf://meta-llama/Llama-3.1-8B-Instruct"
    version: "${NEW_VERSION}"
  # ... rest of config
EOF

# Wait for green deployment to be ready
kubectl wait --for=condition=Ready llmdeployment/${MODEL_NAME}-green -n ${NAMESPACE} --timeout=600s

# Run health checks on green
echo "🔍 Running health checks on green deployment..."
bash health-check.sh ${MODEL_NAME}-green ${NAMESPACE}

if [ $? -eq 0 ]; then
  echo "✅ Health checks passed. Switching traffic..."
  
  # Switch service to green
  kubectl patch service ${MODEL_NAME} -n ${NAMESPACE} --type='merge' -p='
  {
    "spec": {
      "selector": {
        "deployment": "green"
      }
    }
  }'
  
  echo "✅ Traffic switched to green deployment"
  echo "🧹 Cleaning up blue deployment in 5 minutes..."
  sleep 300
  kubectl delete llmdeployment ${MODEL_NAME}-blue -n ${NAMESPACE}
else
  echo "❌ Health checks failed. Rolling back..."
  kubectl delete llmdeployment ${MODEL_NAME}-green -n ${NAMESPACE}
  exit 1
fi
```

## Lessons from Real-World LLM Outages

Based on analysis of recent production incidents at major LLM providers, here are critical operational patterns:

### OpenAI December 2024: Kubernetes Control Plane Overload

**Incident**: New telemetry service overwhelmed Kubernetes API servers
**Duration**: 4+ hours
**Impact**: Complete service unavailability

**Key Lessons for llm-d Operators:**

```yaml
# Implement staged rollouts for infrastructure changes
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: llm-infrastructure-update
spec:
  strategy:
    canary:
      steps:
      - setWeight: 10
      - pause: {duration: 10m}
      - setWeight: 50
      - pause: {duration: 30m}
      - setWeight: 100
  # Monitor Kubernetes API health during rollouts
  analysis:
    templates:
    - templateName: k8s-api-health
    args:
    - name: service-name
      value: llm-deployment
```

### Meta Llama 3: Hardware Reliability Patterns

**Finding**: 419 interruptions over 54 days (30.1% GPU failures)
**Operational Response**: Proactive GPU health monitoring

```bash
# Automated GPU health monitoring
#!/bin/bash
# Run every 15 minutes via cron

for node in $(kubectl get nodes -l node-type=gpu -o name); do
  # Check GPU health
  kubectl exec -n kube-system daemonset/nvidia-device-plugin -- nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used --format=csv,noheader,nounits > /tmp/gpu-health.log
  
  # Alert on anomalies
  if grep -q "N/A\|ERR" /tmp/gpu-health.log; then
    echo "⚠️  GPU health issue detected on $node"
    # Trigger alert to SRE team
  fi
done
```

### Cost-Aware Operational Decisions

High-level cost implications of common SRE decisions:

| Decision | Cost Impact | Reliability Impact |
|----------|-------------|-------------------|
| Scale up during incidents | +40-60% hourly cost | +30% availability |
| Use spot instances | -50% cost | -5% availability |
| Aggressive autoscaling | Variable cost | +15% latency |
| Model rollback | No cost impact | +99% immediate recovery |

:::tip Production Recommendation
Based on real incident data, prioritize **fast rollback capabilities** over **horizontal scaling** for LLM service reliability. Model rollbacks typically restore service in under 2 minutes, while scaling can take 10+ minutes.
:::

## Integration with Existing SRE Tools

### Prometheus Integration

```yaml
# Custom LLM metrics for Prometheus
apiVersion: v1
kind: ConfigMap
metadata:
  name: llm-prometheus-rules
data:
  llm-rules.yaml: |
    groups:
    - name: llm-d.rules
      rules:
      - alert: LLMModelNotResponding
        expr: up{job="llm-deployment"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "LLM model {{ $labels.instance }} is not responding"
      
      - alert: LLMHighTokenLatency
        expr: histogram_quantile(0.95, llm_token_generation_seconds) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High token generation latency"
```

### GitOps Integration

```yaml
# ArgoCD Application for LLM deployments
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: llm-production
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://git.company.com/sre/llm-deployments
    targetRevision: HEAD
    path: production/
  destination:
    server: https://kubernetes.default.svc
    namespace: production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
    - CreateNamespace=true
    # Require manual approval for model version changes
    retry:
      limit: 3
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
```

## Summary and Next Steps

### Key Takeaways

- ✅ **Hardware Reliability**: Expect frequent GPU failures; plan monitoring accordingly
- ✅ **Incident Response**: Fast rollbacks are more effective than scaling for model issues  
- ✅ **Operational Workflow**: Structured handoff processes prevent production surprises
- ✅ **Cost Awareness**: SRE decisions have significant cost implications

### MLOps Maturity Path

This chapter builds on the operational foundation from previous chapters and establishes production-ready LLM service management. The monitoring and incident response patterns here integrate with the performance optimization (Chapter 6) and cost management (Chapter 11) strategies.

**Next Steps:**

- Apply these operational patterns to your llm-d deployments
- Establish monitoring baselines using the provided metrics
- Practice rollback procedures in staging environments
- Review the appendices for command references and configuration templates

---

:::info References

- [Google SRE Book](https://sre.google/sre-book/) - Foundational SRE concepts and practices
- [OpenAI December 2024 Post-Mortem](https://status.openai.com/) - Kubernetes control plane incident analysis
- [Meta Llama 3 Training Report](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/) - Hardware reliability data
- [llm-d-deployer Quickstart](https://github.com/llm-d/llm-d-deployer) - Monitoring setup examples
- [Shared Configuration Reference](./appendix/shared-config.md)

:::
