---
title: SRE Operations
description: Comprehensive guide to Site Reliability Engineering practices for llm-d production environments
sidebar_position: 5
---

# SRE Operations

:::info Chapter Overview
This chapter focuses on Site Reliability Engineering (SRE) practices for llm-d, covering production operations, incident response, monitoring strategies, capacity planning, and reliability engineering. You'll learn how to maintain high availability, respond to incidents, and continuously improve system reliability.
:::

## SRE Fundamentals for LLM Infrastructure

### Service Level Objectives (SLOs)

Define measurable reliability targets for llm-d services:

**Core SLOs for LLM Services:**

```yaml
# SLO Configuration
apiVersion: monitoring.llm-d.ai/v1alpha1
kind: ServiceLevelObjective
metadata:
  name: llm-d-production-slos
  namespace: sre-monitoring
spec:
  service: "llm-d-production"
  
  objectives:
    - name: "availability"
      description: "Service availability excluding planned maintenance"
      target: 99.9  # 99.9% uptime (43.8 minutes downtime/month)
      metric:
        type: "availability"
        good_query: "up{job='llm-d-service'} == 1"
        total_query: "up{job='llm-d-service'}"
      window: "30d"
    
    - name: "latency"
      description: "95th percentile inference latency"
      target: 95.0  # 95% of requests under threshold
      metric:
        type: "latency"
        query: "histogram_quantile(0.95, rate(llm_d_request_duration_seconds_bucket[5m]))"
        threshold: "2s"
      window: "30d"
    
    - name: "error_rate"
      description: "Error rate for inference requests"
      target: 99.5  # 99.5% success rate (0.5% error budget)
      metric:
        type: "ratio"
        good_query: "rate(llm_d_requests_total{status!~'5..'}[5m])"
        total_query: "rate(llm_d_requests_total[5m])"
      window: "30d"
    
    - name: "throughput"
      description: "Minimum sustained throughput"
      target: 100.0  # Minimum 100 requests/second
      metric:
        type: "throughput" 
        query: "rate(llm_d_requests_total[5m])"
        threshold: "100"
      window: "30d"

  # Error budget configuration
  error_budget:
    policy: "burn_rate"
    alerts:
      - severity: "page"
        burn_rate: 14.4  # Page if burning budget 14.4x faster
        duration: "2m"
      
      - severity: "ticket"
        burn_rate: 6.0   # Create ticket if burning 6x faster
        duration: "15m"
```

### Monitoring and Alerting Strategy

Implement comprehensive monitoring aligned with SRE principles:

**Golden Signals Dashboard:**

```yaml
# Grafana Dashboard Configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: llm-d-golden-signals-dashboard
  namespace: monitoring
data:
  dashboard.json: |
    {
      "dashboard": {
        "title": "LLM-D Golden Signals",
        "panels": [
          {
            "title": "Request Rate (Traffic)",
            "type": "stat",
            "targets": [{
              "expr": "sum(rate(llm_d_requests_total[5m]))",
              "legendFormat": "Requests/sec"
            }],
            "thresholds": [
              {"value": 50, "color": "red"},
              {"value": 100, "color": "yellow"},
              {"value": 200, "color": "green"}
            ]
          },
          {
            "title": "Error Rate",
            "type": "stat",
            "targets": [{
              "expr": "sum(rate(llm_d_requests_total{status=~'5..'}[5m])) / sum(rate(llm_d_requests_total[5m])) * 100",
              "legendFormat": "Error %"
            }],
            "thresholds": [
              {"value": 1, "color": "green"},
              {"value": 5, "color": "yellow"},
              {"value": 10, "color": "red"}
            ]
          },
          {
            "title": "Latency Distribution",
            "type": "heatmap",
            "targets": [{
              "expr": "histogram_quantile(0.50, rate(llm_d_request_duration_seconds_bucket[5m]))",
              "legendFormat": "p50"
            }, {
              "expr": "histogram_quantile(0.95, rate(llm_d_request_duration_seconds_bucket[5m]))",
              "legendFormat": "p95"
            }, {
              "expr": "histogram_quantile(0.99, rate(llm_d_request_duration_seconds_bucket[5m]))",
              "legendFormat": "p99"
            }]
          },
          {
            "title": "Resource Saturation",
            "type": "graph",
            "targets": [{
              "expr": "avg(llm_d_gpu_utilization) by (instance)",
              "legendFormat": "GPU Utilization - {{instance}}"
            }, {
              "expr": "avg(llm_d_memory_utilization) by (instance)",
              "legendFormat": "Memory Utilization - {{instance}}"
            }]
          }
        ]
      }
    }
```

**Alert Rules Configuration:**

```yaml
# Prometheus Alerting Rules
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: llm-d-sre-alerts
  namespace: monitoring
spec:
  groups:
  - name: llm-d.critical
    interval: 30s
    rules:
    - alert: LLMDServiceDown
      expr: up{job="llm-d-service"} == 0
      for: 1m
      labels:
        severity: critical
        service: llm-d
        team: sre
      annotations:
        summary: "LLM-D service is down"
        description: "LLM-D service {{ $labels.instance }} has been down for more than 1 minute"
        runbook_url: "https://runbooks.company.com/llm-d/service-down"
        
    - alert: LLMDHighErrorRate
      expr: |
        (
          sum(rate(llm_d_requests_total{status=~"5.."}[5m])) /
          sum(rate(llm_d_requests_total[5m]))
        ) * 100 > 5
      for: 2m
      labels:
        severity: critical
        service: llm-d
        team: sre
      annotations:
        summary: "High error rate detected"
        description: "Error rate is {{ $value }}% over the last 5 minutes"
        runbook_url: "https://runbooks.company.com/llm-d/high-error-rate"
        
    - alert: LLMDHighLatency
      expr: |
        histogram_quantile(0.95, 
          rate(llm_d_request_duration_seconds_bucket[5m])
        ) > 3
      for: 5m
      labels:
        severity: warning
        service: llm-d
        team: sre
      annotations:
        summary: "High inference latency detected"
        description: "95th percentile latency is {{ $value }}s"
        runbook_url: "https://runbooks.company.com/llm-d/high-latency"

  - name: llm-d.capacity
    interval: 60s
    rules:
    - alert: LLMDGPUHighUtilization
      expr: avg(llm_d_gpu_utilization) > 85
      for: 10m
      labels:
        severity: warning
        service: llm-d
        team: sre
      annotations:
        summary: "High GPU utilization detected"
        description: "Average GPU utilization is {{ $value }}%"
        runbook_url: "https://runbooks.company.com/llm-d/gpu-utilization"
        
    - alert: LLMDMemoryPressure
      expr: avg(llm_d_memory_utilization) > 90
      for: 5m
      labels:
        severity: critical
        service: llm-d
        team: sre
      annotations:
        summary: "Memory pressure detected"
        description: "Memory utilization is {{ $value }}%"
        runbook_url: "https://runbooks.company.com/llm-d/memory-pressure"
        
    - alert: LLMDKVCacheOverflow
      expr: llm_d_kv_cache_evictions_total > 100
      for: 5m
      labels:
        severity: warning
        service: llm-d
        team: sre
      annotations:
        summary: "High KV cache eviction rate"
        description: "KV cache evictions: {{ $value }} in last 5 minutes"
        runbook_url: "https://runbooks.company.com/llm-d/cache-evictions"
```

## Production Operations

### Deployment Management

Implement zero-downtime deployment strategies:

**Blue-Green Deployment:**

```yaml
# Blue-Green Deployment Configuration
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: llm-d-blue-green
  namespace: production
spec:
  replicas: 8
  strategy:
    blueGreen:
      # Reference to service that the rollout modifies as the active service
      activeService: llm-d-active
      # Pre-promotion analysis
      prePromotionAnalysis:
        templates:
        - templateName: success-rate
        args:
        - name: service-name
          value: llm-d-preview
      # Post-promotion analysis  
      postPromotionAnalysis:
        templates:
        - templateName: success-rate
        args:
        - name: service-name
          value: llm-d-active
      
      # Automatic promotion after 10 minutes if analysis passes
      autoPromotionEnabled: false
      scaleDownDelaySeconds: 30
      previewReplicaCount: 2
      
  selector:
    matchLabels:
      app: llm-d
  template:
    metadata:
      labels:
        app: llm-d
    spec:
      containers:
      - name: llm-d-service
        image: llm-d/service:v1.2.0
        ports:
        - containerPort: 8080
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
        
        # Health checks
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3

---
# Analysis Template for Success Rate
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: success-rate
  namespace: production
spec:
  args:
  - name: service-name
  metrics:
  - name: success-rate
    interval: 60s
    count: 5
    successCondition: result[0] >= 0.95
    failureLimit: 3
    provider:
      prometheus:
        address: http://prometheus.monitoring.svc.cluster.local:9090
        query: |
          sum(
            rate(llm_d_requests_total{service="{{args.service-name}}",status!~"5.."}[2m])
          ) /
          sum(
            rate(llm_d_requests_total{service="{{args.service-name}}"}[2m])
          )
```

**Canary Deployment:**

```yaml
# Canary Deployment with Traffic Splitting
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: llm-d-canary
  namespace: production
spec:
  replicas: 10
  strategy:
    canary:
      # Canary service and ingress
      canaryService: llm-d-canary
      stableService: llm-d-stable
      
      # Traffic splitting via Istio
      trafficRouting:
        istio:
          virtualService:
            name: llm-d-vs
            routes:
            - primary
      
      # Progressive traffic increase
      steps:
      - setWeight: 5    # 5% traffic to canary
      - pause: {duration: 5m}
      - analysis:
          templates:
          - templateName: canary-analysis
          args:
          - name: canary-service
            value: llm-d-canary
          
      - setWeight: 20   # 20% traffic
      - pause: {duration: 10m}
      - analysis:
          templates:
          - templateName: canary-analysis
          args:
          - name: canary-service
            value: llm-d-canary
          
      - setWeight: 50   # 50% traffic
      - pause: {duration: 10m}
      - analysis:
          templates:
          - templateName: canary-analysis
          args:
          - name: canary-service
            value: llm-d-canary
      
      # Automatic rollback conditions
      analysis:
        templates:
        - templateName: canary-analysis
        args:
        - name: canary-service
          value: llm-d-canary
```

### Capacity Planning

Implement data-driven capacity planning:

**Resource Utilization Analysis:**

```python
class CapacityPlanner:
    def __init__(self, prometheus_client, metrics_retention="30d"):
        self.prometheus = prometheus_client
        self.retention = metrics_retention
        
    def analyze_historical_usage(self, service_name):
        """Analyze historical resource usage patterns"""
        queries = {
            "request_rate": f"rate(llm_d_requests_total{{service='{service_name}'}}[5m])",
            "gpu_utilization": f"avg(llm_d_gpu_utilization{{service='{service_name}'}}) by (instance)",
            "memory_utilization": f"avg(llm_d_memory_utilization{{service='{service_name}'}}) by (instance)",
            "latency_p95": f"histogram_quantile(0.95, rate(llm_d_request_duration_seconds_bucket{{service='{service_name}'}}[5m]))",
            "queue_depth": f"avg(llm_d_queue_depth{{service='{service_name}'}})"
        }
        
        historical_data = {}
        for metric, query in queries.items():
            data = self.prometheus.query_range(
                query=query,
                start_time=f"-{self.retention}",
                end_time="now",
                step="1h"
            )
            historical_data[metric] = self._process_time_series(data)
        
        return self._analyze_patterns(historical_data)
    
    def _analyze_patterns(self, data):
        """Analyze usage patterns and identify trends"""
        analysis = {
            "peak_hours": self._identify_peak_hours(data["request_rate"]),
            "growth_trend": self._calculate_growth_trend(data["request_rate"]),
            "resource_correlation": self._analyze_resource_correlation(data),
            "capacity_utilization": self._calculate_capacity_utilization(data),
            "forecasting": self._forecast_requirements(data)
        }
        
        return analysis
    
    def _forecast_requirements(self, data, forecast_period="90d"):
        """Forecast future resource requirements"""
        import numpy as np
        from sklearn.linear_model import LinearRegression
        
        # Prepare time series data
        request_rates = data["request_rate"]
        timestamps = [point["timestamp"] for point in request_rates]
        values = [point["value"] for point in request_rates]
        
        # Simple linear regression for trend
        X = np.array(range(len(values))).reshape(-1, 1)
        y = np.array(values)
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Forecast future points
        future_points = 90 * 24  # 90 days of hourly data
        future_X = np.array(range(len(values), len(values) + future_points)).reshape(-1, 1)
        forecast = model.predict(future_X)
        
        # Calculate resource requirements based on forecast
        peak_forecast = np.max(forecast) * 1.2  # 20% safety margin
        
        return {
            "forecast_peak_rps": peak_forecast,
            "recommended_replicas": self._calculate_required_replicas(peak_forecast),
            "recommended_gpu_count": self._calculate_required_gpus(peak_forecast),
            "growth_rate_per_day": model.coef_[0] * 24
        }
    
    def _calculate_required_replicas(self, peak_rps):
        """Calculate required replicas based on peak RPS"""
        # Assume each replica can handle 50 RPS at 80% utilization
        replicas_needed = peak_rps / (50 * 0.8)
        return max(int(np.ceil(replicas_needed)), 2)  # Minimum 2 replicas
    
    def _calculate_required_gpus(self, peak_rps):
        """Calculate required GPU count based on peak RPS"""
        # Assume each GPU can handle 25 RPS for LLM inference
        gpus_needed = peak_rps / 25
        return max(int(np.ceil(gpus_needed)), 1)
    
    def generate_capacity_plan(self, analysis_results):
        """Generate comprehensive capacity plan"""
        plan = {
            "current_state": {
                "avg_utilization": analysis_results["capacity_utilization"]["avg"],
                "peak_utilization": analysis_results["capacity_utilization"]["peak"],
                "current_replicas": self._get_current_replicas(),
                "current_gpu_count": self._get_current_gpu_count()
            },
            
            "recommendations": {
                "immediate": self._immediate_recommendations(analysis_results),
                "short_term": self._short_term_recommendations(analysis_results),
                "long_term": self._long_term_recommendations(analysis_results)
            },
            
            "scaling_triggers": {
                "scale_up": {
                    "cpu_threshold": "80%",
                    "gpu_threshold": "85%",
                    "memory_threshold": "85%",
                    "latency_threshold": "2s",
                    "queue_depth_threshold": "10"
                },
                "scale_down": {
                    "cpu_threshold": "30%",
                    "gpu_threshold": "40%", 
                    "memory_threshold": "40%",
                    "sustained_duration": "15m"
                }
            },
            
            "cost_analysis": self._analyze_costs(analysis_results)
        }
        
        return plan

# Example usage
planner = CapacityPlanner(prometheus_client)
usage_analysis = planner.analyze_historical_usage("llm-d-production")
capacity_plan = planner.generate_capacity_plan(usage_analysis)

print("Capacity Planning Results:")
print(f"Current utilization: {capacity_plan['current_state']['avg_utilization']:.1%}")
print(f"Recommended replicas: {capacity_plan['recommendations']['immediate']['replicas']}")
print(f"Forecasted peak RPS: {usage_analysis['forecasting']['forecast_peak_rps']:.1f}")
```

### Configuration Management

Implement GitOps-based configuration management:

```yaml
# Configuration Management Structure
apiVersion: v1
kind: ConfigMap
metadata:
  name: llm-d-config-template
  namespace: config-management
data:
  # Environment-specific configurations
  environments.yaml: |
    environments:
      development:
        namespace: "development"
        replicas:
          prefill: 1
          decode: 1
        resources:
          gpu_per_pod: 1
          memory_per_pod: "16Gi"
        monitoring:
          sample_rate: 1.0
          detailed_logging: true
        
      staging:
        namespace: "staging"
        replicas:
          prefill: 2
          decode: 4
        resources:
          gpu_per_pod: 1
          memory_per_pod: "24Gi"
        monitoring:
          sample_rate: 0.1
          detailed_logging: false
        
      production:
        namespace: "production"
        replicas:
          prefill: 4
          decode: 8
        resources:
          gpu_per_pod: 2
          memory_per_pod: "32Gi"
        monitoring:
          sample_rate: 0.01
          detailed_logging: false
          
  # Model configurations
  models.yaml: |
    models:
      llama3-8b:
        model_uri: "hf://meta-llama/Llama-3.1-8B-Instruct"
        quantization: "fp8"
        tensor_parallel_size: 1
        context_length: 8192
        
      llama3-70b:
        model_uri: "hf://meta-llama/Llama-3.1-70B-Instruct"
        quantization: "fp8"
        tensor_parallel_size: 4
        context_length: 8192
        
  # Operational configurations
  operations.yaml: |
    operations:
      health_checks:
        liveness_probe:
          path: "/health"
          initial_delay: 30
          period: 10
          timeout: 5
          failure_threshold: 3
          
        readiness_probe:
          path: "/ready"
          initial_delay: 10
          period: 5
          timeout: 3
          failure_threshold: 3
      
      scaling:
        hpa:
          min_replicas: 2
          max_replicas: 20
          target_cpu: 70
          target_memory: 80
          scale_up_policies:
            - type: "Percent"
              value: 100
              period_seconds: 15
          scale_down_policies:
            - type: "Percent"
              value: 10
              period_seconds: 60
      
      security:
        pod_security_context:
          run_as_non_root: true
          run_as_user: 1000
          fs_group: 2000
        
        network_policies:
          enabled: true
          allow_namespaces: ["monitoring", "ingress-nginx"]
          
        rbac:
          create_service_account: true
          cluster_role_rules:
            - api_groups: [""]
              resources: ["configmaps", "secrets"]
              verbs: ["get", "list", "watch"]
```

## Incident Response

### Incident Classification and Response

Implement structured incident response procedures:

**Incident Severity Levels:**

```yaml
# Incident Response Configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: incident-response-procedures
  namespace: sre
data:
  severity_levels.yaml: |
    severity_levels:
      P0_CRITICAL:
        description: "Complete service outage affecting all users"
        response_time: "5 minutes"
        escalation_time: "15 minutes"
        team_members: ["on-call-sre", "service-owner", "incident-commander"]
        communication_channels: ["#incidents-critical", "pager-duty"]
        examples:
          - "All inference requests failing (>95% error rate)"
          - "Complete cluster outage"
          - "Data corruption or security breach"
        
      P1_HIGH:
        description: "Significant service degradation affecting majority of users"
        response_time: "15 minutes"
        escalation_time: "30 minutes"
        team_members: ["on-call-sre", "service-owner"]
        communication_channels: ["#incidents-high", "slack"]
        examples:
          - "High error rate (>10%)"
          - "Severe latency degradation (>5s p95)"
          - "GPU resource exhaustion"
        
      P2_MEDIUM:
        description: "Partial service degradation with workarounds available"
        response_time: "30 minutes"
        escalation_time: "2 hours"
        team_members: ["on-call-sre"]
        communication_channels: ["#incidents-medium"]
        examples:
          - "Elevated error rate (2-10%)"
          - "Moderate latency increase (2-5s p95)"
          - "Single node failures"
        
      P3_LOW:
        description: "Minor issues with minimal user impact"
        response_time: "2 hours"
        escalation_time: "8 hours"
        team_members: ["on-call-sre"]
        communication_channels: ["#incidents-low"]
        examples:
          - "Non-critical monitoring alerts"
          - "Minor performance degradation"
          - "Capacity warnings"

  response_procedures.yaml: |
    response_procedures:
      initial_response:
        - "Acknowledge the incident in PagerDuty"
        - "Join the incident response channel"
        - "Assess severity using established criteria"
        - "Notify stakeholders based on severity level"
        - "Begin investigation using runbooks"
        
      investigation_steps:
        - "Check service status dashboard"
        - "Review recent deployments and changes"
        - "Analyze logs and metrics"
        - "Identify root cause"
        - "Implement immediate mitigation"
        
      communication_protocol:
        - "Post initial status within 15 minutes"
        - "Provide updates every 30 minutes for P0/P1"
        - "Maintain clear, factual communication"
        - "Document all actions taken"
        - "Conduct post-incident review"
```

### Runbooks and Troubleshooting

Create comprehensive runbooks for common scenarios:

**High Latency Runbook:**

```markdown
# Runbook: High Inference Latency

## Overview
This runbook covers troubleshooting steps for when LLM-D inference latency exceeds acceptable thresholds.

## Severity: P2 (Medium)
**SLO Impact:** Latency SLO breach (>2s p95)
**Response Time:** 30 minutes

## Investigation Steps

### 1. Verify the Alert
```bash
# Check current latency metrics
kubectl exec -n monitoring prometheus-0 -- \
  promtool query instant \
  'histogram_quantile(0.95, rate(llm_d_request_duration_seconds_bucket[5m]))'

# Check if latency is consistently high
kubectl exec -n monitoring prometheus-0 -- \
  promtool query range \
  'histogram_quantile(0.95, rate(llm_d_request_duration_seconds_bucket[5m]))' \
  --start $(date -d '1 hour ago' --iso-8601) \
  --end $(date --iso-8601) \
  --step 1m
```

### 2. Check System Resources

```bash
# Check GPU utilization
kubectl top pods -n production --sort-by='gpu'

# Check memory usage
kubectl describe nodes | grep -A 5 "Allocated resources"

# Check queue depths
kubectl exec -n monitoring prometheus-0 -- \
  promtool query instant \
  'llm_d_queue_depth'
```

### 3. Analyze Request Patterns

```bash
# Check request rate
kubectl exec -n monitoring prometheus-0 -- \
  promtool query instant \
  'rate(llm_d_requests_total[5m])'

# Check request size distribution
kubectl logs -n production deployment/llm-d-prefill \
  | grep "request_tokens" \
  | tail -50
```

### 4. Check Cache Performance

```bash
# Check cache hit ratio
kubectl exec -n monitoring prometheus-0 -- \
  promtool query instant \
  'rate(llm_d_cache_hits_total[5m]) / rate(llm_d_cache_requests_total[5m])'

# Check cache memory usage
kubectl exec -n production deployment/redis-cluster -- \
  redis-cli info memory
```

## Common Root Causes and Solutions

### High Queue Depth

**Symptoms:** Queue depth > 10, increased latency
**Solution:** Scale up decode pods

```bash
kubectl scale deployment llm-d-decode \
  --replicas=16 \
  -n production
```

### GPU Resource Exhaustion

**Symptoms:** GPU utilization > 95%, OOM errors
**Solution:** Optimize batch size or scale prefill pods

```bash
# Check for OOM events
kubectl get events -n production \
  --field-selector reason=OOMKilled

# Scale prefill pods if needed
kubectl scale deployment llm-d-prefill \
  --replicas=6 \
  -n production
```

### Cache Misses

**Symptoms:** Low cache hit ratio < 60%
**Solution:** Investigate cache configuration

```bash
# Check cache eviction patterns
kubectl exec -n production deployment/redis-cluster -- \
  redis-cli info stats | grep evicted

# Review cache sizing
kubectl exec -n production deployment/redis-cluster -- \
  redis-cli config get maxmemory
```

### Model Loading Issues

**Symptoms:** Cold start latency spikes
**Solution:** Ensure model warmup

```bash
# Check model loading times
kubectl logs -n production deployment/llm-d-prefill \
  | grep "model_load_time"

# Trigger model warmup
curl -X POST http://llm-d-service:8080/v1/warmup \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3-8b"}'
```

## Escalation Criteria

- Latency remains >5s p95 after 30 minutes
- GPU utilization cannot be reduced below 90%
- Queue depth continues growing despite scaling
- Multiple service dependencies affected

## Post-Incident Actions

1. Update capacity planning based on findings
2. Review and update autoscaling policies
3. Consider infrastructure improvements
4. Update monitoring thresholds if needed

```

**Service Down Runbook:**

```markdown
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

```

### Disaster Recovery

Implement comprehensive disaster recovery procedures:

```yaml
# Disaster Recovery Configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: disaster-recovery-plan
  namespace: sre
data:
  recovery_procedures.yaml: |
    disaster_recovery:
      scenarios:
        cluster_failure:
          description: "Complete cluster failure or corruption"
          rto: "2 hours"  # Recovery Time Objective
          rpo: "15 minutes"  # Recovery Point Objective
          
          procedures:
            - "Activate backup cluster"
            - "Restore configuration from Git"
            - "Restore model artifacts from backup"
            - "Redirect traffic via DNS"
            - "Validate service functionality"
            
        data_center_outage:
          description: "Complete data center or region failure"
          rto: "4 hours"
          rpo: "30 minutes"
          
          procedures:
            - "Failover to secondary region"
            - "Restore from cross-region backups"
            - "Update load balancer configuration"
            - "Validate cross-region connectivity"
            
        persistent_storage_failure:
          description: "Loss of persistent storage volumes"
          rto: "1 hour"
          rpo: "15 minutes"
          
          procedures:
            - "Restore from volume snapshots"
            - "Recreate PVCs with restored data"
            - "Restart affected pods"
            - "Verify model artifacts integrity"

      backup_strategy:
        configuration:
          frequency: "continuous"  # GitOps-based
          location: "git-repository"
          retention: "90 days"
          
        model_artifacts:
          frequency: "daily"
          location: "s3://backups/llm-models"
          retention: "30 days"
          
        persistent_data:
          frequency: "every 6 hours"
          location: "volume-snapshots"
          retention: "7 days"
          
        metrics_and_logs:
          frequency: "real-time"
          location: "centralized-logging"
          retention: "30 days"

  backup_scripts.sh: |
    #!/bin/bash
    # Automated backup script for LLM-D infrastructure
    
    set -e
    
    NAMESPACE="llm-d-production"
    BACKUP_LOCATION="s3://llm-d-backups/$(date +%Y-%m-%d)"
    
    echo "Starting backup process..."
    
    # Backup Kubernetes configurations
    echo "Backing up Kubernetes configurations..."
    kubectl get all -n $NAMESPACE -o yaml > k8s-configs-backup.yaml
    kubectl get configmaps -n $NAMESPACE -o yaml > configmaps-backup.yaml
    kubectl get secrets -n $NAMESPACE -o yaml > secrets-backup.yaml
    
    # Create volume snapshots
    echo "Creating volume snapshots..."
    kubectl get pvc -n $NAMESPACE -o json | \
      jq -r '.items[].metadata.name' | \
      xargs -I {} kubectl patch pvc {} -n $NAMESPACE \
      --type merge -p '{"metadata":{"annotations":{"backup.kubernetes.io/snapshot":"true"}}}'
    
    # Backup model artifacts
    echo "Backing up model artifacts..."
    kubectl exec -n $NAMESPACE deployment/llm-d-model-service -- \
      tar czf - /models | aws s3 cp - $BACKUP_LOCATION/models.tar.gz
    
    # Upload configurations to S3
    aws s3 cp k8s-configs-backup.yaml $BACKUP_LOCATION/
    aws s3 cp configmaps-backup.yaml $BACKUP_LOCATION/
    aws s3 cp secrets-backup.yaml $BACKUP_LOCATION/
    
    echo "Backup completed successfully!"
    
    # Cleanup local files
    rm -f *-backup.yaml
```

## Performance Optimization

### Resource Right-sizing

Continuously optimize resource allocation:

```python
class ResourceOptimizer:
    def __init__(self, metrics_client):
        self.metrics = metrics_client
        
    def analyze_resource_efficiency(self, service_name, timeframe="7d"):
        """Analyze resource efficiency and identify optimization opportunities"""
        
        # Collect resource utilization data
        utilization_data = self._collect_utilization_metrics(service_name, timeframe)
        
        # Analyze efficiency
        efficiency_analysis = {
            "cpu": self._analyze_cpu_efficiency(utilization_data["cpu"]),
            "memory": self._analyze_memory_efficiency(utilization_data["memory"]),
            "gpu": self._analyze_gpu_efficiency(utilization_data["gpu"]),
            "network": self._analyze_network_efficiency(utilization_data["network"])
        }
        
        # Generate recommendations
        recommendations = self._generate_resource_recommendations(efficiency_analysis)
        
        return {
            "current_efficiency": efficiency_analysis,
            "recommendations": recommendations,
            "potential_savings": self._calculate_cost_savings(recommendations)
        }
    
    def _analyze_cpu_efficiency(self, cpu_data):
        """Analyze CPU utilization patterns"""
        avg_utilization = np.mean(cpu_data["utilization"])
        peak_utilization = np.max(cpu_data["utilization"])
        
        return {
            "average_utilization": avg_utilization,
            "peak_utilization": peak_utilization,
            "efficiency_score": min(avg_utilization / 0.7, 1.0),  # Target 70% utilization
            "recommendation": self._cpu_recommendation(avg_utilization, peak_utilization)
        }
    
    def _analyze_gpu_efficiency(self, gpu_data):
        """Analyze GPU utilization patterns"""
        avg_utilization = np.mean(gpu_data["utilization"])
        memory_utilization = np.mean(gpu_data["memory_utilization"])
        
        return {
            "average_utilization": avg_utilization,
            "memory_utilization": memory_utilization,
            "efficiency_score": min(avg_utilization / 0.8, 1.0),  # Target 80% GPU utilization
            "recommendation": self._gpu_recommendation(avg_utilization, memory_utilization)
        }
    
    def _generate_resource_recommendations(self, efficiency_analysis):
        """Generate specific resource optimization recommendations"""
        recommendations = []
        
        # CPU recommendations
        if efficiency_analysis["cpu"]["efficiency_score"] < 0.5:
            recommendations.append({
                "type": "cpu_reduction",
                "current": "2000m",
                "recommended": "1000m", 
                "savings": "50%",
                "impact": "minimal"
            })
        
        # GPU recommendations
        if efficiency_analysis["gpu"]["efficiency_score"] < 0.6:
            recommendations.append({
                "type": "gpu_sharing",
                "current": "1 GPU per pod",
                "recommended": "GPU sharing with MIG",
                "savings": "40%",
                "impact": "requires testing"
            })
        
        # Memory recommendations
        if efficiency_analysis["memory"]["efficiency_score"] < 0.7:
            recommendations.append({
                "type": "memory_optimization",
                "current": "32Gi",
                "recommended": "24Gi",
                "savings": "25%",
                "impact": "monitor for OOM"
            })
        
        return recommendations

# Example usage
optimizer = ResourceOptimizer(metrics_client)
efficiency_report = optimizer.analyze_resource_efficiency("llm-d-production")

print("Resource Efficiency Analysis:")
print(f"CPU Efficiency: {efficiency_report['current_efficiency']['cpu']['efficiency_score']:.2%}")
print(f"GPU Efficiency: {efficiency_report['current_efficiency']['gpu']['efficiency_score']:.2%}")
print(f"Potential Monthly Savings: ${efficiency_report['potential_savings']['monthly_usd']:.2f}")
```

### Autoscaling Optimization

Implement intelligent autoscaling based on multiple metrics:

```yaml
# Multi-metric HPA Configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-d-intelligent-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-d-decode
  
  minReplicas: 4
  maxReplicas: 32
  
  # Multi-metric scaling
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
        
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
        
  - type: Custom
    custom:
      metric:
        name: llm_d_queue_depth
      target:
        type: AverageValue
        averageValue: "5"
        
  - type: Custom
    custom:
      metric:
        name: llm_d_request_latency_p95
      target:
        type: AverageValue
        averageValue: "1500m"  # 1.5 seconds
  
  # Scaling behavior
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Max
      
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
      selectPolicy: Min

---
# Vertical Pod Autoscaler for Resource Optimization
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: llm-d-vpa
  namespace: production
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-d-prefill
  
  updatePolicy:
    updateMode: "Auto"  # Automatically apply recommendations
    
  resourcePolicy:
    containerPolicies:
    - containerName: llm-d-service
      minAllowed:
        cpu: 500m
        memory: 8Gi
      maxAllowed:
        cpu: 4000m
        memory: 64Gi
      controlledResources: ["cpu", "memory"]
```

## Continuous Improvement

### SRE Metrics and KPIs

Track SRE effectiveness with key metrics:

```python
class SREMetricsCollector:
    def __init__(self):
        self.metrics = {
            "reliability": self._collect_reliability_metrics,
            "performance": self._collect_performance_metrics,
            "operational": self._collect_operational_metrics,
            "cost": self._collect_cost_metrics
        }
    
    def generate_sre_report(self, period="monthly"):
        """Generate comprehensive SRE metrics report"""
        report = {}
        
        for category, collector in self.metrics.items():
            report[category] = collector(period)
        
        # Calculate overall SRE score
        report["overall_score"] = self._calculate_sre_score(report)
        
        # Generate insights and recommendations
        report["insights"] = self._generate_insights(report)
        
        return report
    
    def _collect_reliability_metrics(self, period):
        """Collect reliability-focused metrics"""
        return {
            "availability": {
                "slo_target": 99.9,
                "actual": 99.95,
                "error_budget_remaining": 80.0
            },
            "mttr": {
                "target_minutes": 30,
                "actual_minutes": 22,
                "trend": "improving"
            },
            "mtbf": {
                "target_hours": 720,  # 30 days
                "actual_hours": 896,
                "trend": "stable"
            },
            "incident_count": {
                "p0": 0,
                "p1": 2,
                "p2": 8,
                "p3": 15
            }
        }
    
    def _collect_performance_metrics(self, period):
        """Collect performance-focused metrics"""
        return {
            "latency": {
                "p50_ms": 850,
                "p95_ms": 1800,
                "p99_ms": 3200,
                "slo_target_ms": 2000
            },
            "throughput": {
                "rps_avg": 250,
                "rps_peak": 890,
                "capacity_utilization": 0.68
            },
            "resource_efficiency": {
                "cpu_utilization": 0.72,
                "gpu_utilization": 0.85,
                "memory_utilization": 0.78
            }
        }
    
    def _collect_operational_metrics(self, period):
        """Collect operational efficiency metrics"""
        return {
            "deployment_metrics": {
                "deployment_frequency": "2.3/week",
                "deployment_success_rate": 0.96,
                "rollback_rate": 0.04,
                "lead_time_hours": 4.2
            },
            "automation": {
                "automated_responses": 0.85,
                "manual_interventions": 12,
                "runbook_coverage": 0.92
            },
            "monitoring": {
                "alert_noise_ratio": 0.15,
                "false_positive_rate": 0.08,
                "monitoring_coverage": 0.94
            }
        }
    
    def _calculate_sre_score(self, report):
        """Calculate overall SRE effectiveness score"""
        weights = {
            "availability": 0.4,
            "performance": 0.3,
            "automation": 0.2,
            "cost_efficiency": 0.1
        }
        
        scores = {
            "availability": min(report["reliability"]["availability"]["actual"] / 99.9, 1.0),
            "performance": min(2000 / report["performance"]["latency"]["p95_ms"], 1.0),
            "automation": report["operational"]["automation"]["automated_responses"],
            "cost_efficiency": min(report["cost"]["efficiency_score"], 1.0)
        }
        
        overall_score = sum(score * weights[metric] for metric, score in scores.items())
        return round(overall_score * 100, 1)  # Convert to percentage

# Example usage
metrics_collector = SREMetricsCollector()
monthly_report = metrics_collector.generate_sre_report("monthly")

print(f"Overall SRE Score: {monthly_report['overall_score']}%")
print(f"Availability: {monthly_report['reliability']['availability']['actual']}%")
print(f"P95 Latency: {monthly_report['performance']['latency']['p95_ms']}ms")
```

## Summary

This chapter covered comprehensive SRE practices for llm-d:

**Key Takeaways:**

- **SLO-Driven Reliability**: Define and monitor meaningful service level objectives
- **Proactive Monitoring**: Implement comprehensive monitoring and alerting strategies
- **Incident Response**: Establish clear procedures and runbooks for rapid resolution
- **Capacity Planning**: Use data-driven approaches for resource planning and optimization
- **Disaster Recovery**: Implement robust backup and recovery procedures
- **Continuous Improvement**: Track SRE metrics and continuously optimize operations

## Next Steps

Continue your SRE journey with:

- **Chapter 6**: Performance optimization and tuning strategies
- **Chapter 7**: Troubleshooting guide with decision trees and solutions
- **Chapter 8**: Security and compliance considerations

---

:::info References

- [Site Reliability Engineering Book](https://sre.google/sre-book/table-of-contents/)
- [SLI/SLO Best Practices](https://cloud.google.com/blog/products/management-tools/practical-guide-to-setting-slos)
- [Prometheus Monitoring](https://prometheus.io/docs/practices/rules/)
- [Kubernetes HPA Documentation](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)
- [Shared Configuration Reference](./appendix/shared-config.md)

:::
