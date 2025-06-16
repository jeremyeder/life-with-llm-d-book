---
title: Configuration Templates
description: Ready-to-use YAML and Helm templates for all deployment scenarios
sidebar_position: 3
---

# Appendix C: Configuration Templates

This appendix provides ready-to-use configuration templates for all deployment scenarios. Templates are organized by environment and use case, with both minimal starting points and comprehensive production-ready configurations.

## Template Organization

### Raw YAML Templates

- **Development**: Cost-optimized, single-replica configurations
- **Staging**: Production-like with reduced resources  
- **Production**: Full production with monitoring, security, and scaling
- **Multi-tenant**: Shared infrastructure with isolation

### Helm Chart Templates

- **llm-d-chart**: Complete Helm chart following llm-d-deployer patterns
- **Environment-specific values**: Overlay files for different environments
- **Component templates**: Modular configurations for different use cases

## Development Environment Templates

### Minimal Development Configuration

Perfect for experimentation and testing with cost optimization:

```yaml title="development/minimal-llm-deployment.yaml"
apiVersion: inference.llm-d.io/v1alpha1
kind: LLMDeployment
metadata:
  name: llama-3.1-8b-dev
  namespace: development
  labels:
    app.kubernetes.io/name: llm-d
    llm-d.ai/environment: development
    llm-d.ai/model: llama-3.1
    llm-d.ai/size: 8b
spec:
  model:
    modelUri: "hf://meta-llama/Llama-3.1-8B-Instruct"
    quantization:
      type: int8  # 50% cost reduction
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
    minReplicas: 1  # Scale to zero when idle
    maxReplicas: 3
    targetGPUUtilization: 80  # Higher utilization for cost efficiency
  nodeSelector:
    node-type: gpu
  tolerations:
  - key: "spot-instance"
    operator: "Equal"
    value: "true"
    effect: "NoSchedule"  # Use spot instances for cost savings
---
apiVersion: v1
kind: Service
metadata:
  name: llama-3.1-8b-dev-service
  namespace: development
spec:
  selector:
    app.kubernetes.io/name: llm-d
    llm-d.ai/model: llama-3.1
    llm-d.ai/size: 8b
  ports:
  - name: http
    port: 8080
    targetPort: 8080
  - name: metrics
    port: 8081
    targetPort: 8081
  type: ClusterIP
```

### Development with Debugging

Enhanced development setup with debugging capabilities:

```yaml title="development/debug-llm-deployment.yaml"
apiVersion: inference.llm-d.io/v1alpha1
kind: LLMDeployment
metadata:
  name: llama-3.1-8b-debug
  namespace: development
  annotations:
    debug.llm-d.ai/enabled: "true"
    debug.llm-d.ai/log-level: "debug"
spec:
  model:
    modelUri: "hf://meta-llama/Llama-3.1-8B-Instruct"
    quantization:
      type: int8
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
    enabled: false  # Disable for consistent debugging
  replicas: 1
  debugging:
    enabled: true
    logLevel: debug
    profiling:
      enabled: true
      port: 6060
    healthChecks:
      verbose: true
  monitoring:
    enabled: true
    metrics:
      detailed: true
      port: 8081
    logging:
      level: debug
      format: json
---
# Debug service with profiling endpoint
apiVersion: v1
kind: Service
metadata:
  name: llama-3.1-8b-debug-service
  namespace: development
spec:
  selector:
    app.kubernetes.io/name: llm-d
  ports:
  - name: http
    port: 8080
    targetPort: 8080
  - name: metrics
    port: 8081
    targetPort: 8081
  - name: profiling
    port: 6060
    targetPort: 6060
  type: ClusterIP
```

**Key customization points for development:**

- `spec.model.quantization.type`: Change to `fp16` for accuracy, `int4` for maximum cost savings
- `spec.autoscaling.minReplicas`: Set to 0 for scale-to-zero in extended idle periods
- `tolerations`: Add spot instance tolerations for 40-60% cost reduction
- `debugging.enabled`: Enable for development, disable for performance testing

## Staging Environment Templates

### Pre-Production Staging

Production-like configuration with reduced resources:

```yaml title="staging/staging-llm-deployment.yaml"
apiVersion: inference.llm-d.io/v1alpha1
kind: LLMDeployment
metadata:
  name: llama-3.1-8b-staging
  namespace: staging
  labels:
    app.kubernetes.io/name: llm-d
    llm-d.ai/environment: staging
    llm-d.ai/model: llama-3.1
    llm-d.ai/size: 8b
  annotations:
    deployment.llm-d.ai/validation: "pre-production"
spec:
  model:
    modelUri: "hf://meta-llama/Llama-3.1-8B-Instruct"
    version: "v1.2.0"  # Pin version for consistency
    quantization:
      type: int8
  resources:
    requests:
      memory: "24Gi"  # Slightly higher than dev
      cpu: "6"
      nvidia.com/gpu: "1"
    limits:
      memory: "32Gi"
      cpu: "12"
      nvidia.com/gpu: "1"
  autoscaling:
    enabled: true
    minReplicas: 1
    maxReplicas: 5  # Lower than production
    targetGPUUtilization: 70
    behavior:
      scaleUp:
        stabilizationWindowSeconds: 60
      scaleDown:
        stabilizationWindowSeconds: 300
  scheduling:
    nodeSelector:
      node-type: gpu
      environment: staging
  monitoring:
    enabled: true
    metrics:
      port: 8081
      path: "/metrics"
    alerting:
      enabled: true
      channels: ["#staging-alerts"]
  security:
    enabled: true
    runAsNonRoot: true
    readOnlyRootFilesystem: true
---
apiVersion: v1
kind: Service
metadata:
  name: llama-3.1-8b-staging-service
  namespace: staging
  annotations:
    service.llm-d.ai/load-balancer: "internal"
spec:
  selector:
    app.kubernetes.io/name: llm-d
    llm-d.ai/model: llama-3.1
    llm-d.ai/size: 8b
  ports:
  - name: http
    port: 8080
    targetPort: 8080
  - name: metrics
    port: 8081
    targetPort: 8081
  type: ClusterIP
---
# Horizontal Pod Autoscaler for staging
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llama-3.1-8b-staging-hpa
  namespace: staging
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llama-3.1-8b-staging
  minReplicas: 1
  maxReplicas: 5
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
        averageValue: "70"
```

## Production Environment Templates

### Comprehensive Production Configuration

Full production setup with all features enabled:

```yaml title="production/production-llm-deployment.yaml"
apiVersion: inference.llm-d.io/v1alpha1
kind: LLMDeployment
metadata:
  name: llama-3.1-8b-prod
  namespace: production
  labels:
    app.kubernetes.io/name: llm-d
    app.kubernetes.io/component: inference-server
    app.kubernetes.io/version: "v1.2.0"
    llm-d.ai/environment: production
    llm-d.ai/model: llama-3.1
    llm-d.ai/size: 8b
    llm-d.ai/team: ml-platform
  annotations:
    deployment.llm-d.ai/sla: "99.9%"
    deployment.llm-d.ai/budget: "high-priority"
    deployment.llm-d.ai/owner: "ml-platform-team@company.com"
spec:
  model:
    modelUri: "hf://meta-llama/Llama-3.1-8B-Instruct"
    version: "v1.2.0"  # Always pin production versions
    quantization:
      type: int8
    registry:
      endpoint: "company-registry.com/models"
      credentials:
        secretName: "model-registry-secret"
  resources:
    requests:
      memory: "32Gi"
      cpu: "8"
      nvidia.com/gpu: "2"  # Redundancy for reliability
    limits:
      memory: "48Gi"
      cpu: "16"
      nvidia.com/gpu: "2"
  autoscaling:
    enabled: true
    minReplicas: 3  # Always maintain capacity
    maxReplicas: 15
    targetGPUUtilization: 60  # Conservative for reliability
    behavior:
      scaleUp:
        stabilizationWindowSeconds: 30
        policies:
        - type: Percent
          value: 100
          periodSeconds: 15
      scaleDown:
        stabilizationWindowSeconds: 600  # Slow scale-down
        policies:
        - type: Percent
          value: 10
          periodSeconds: 60
  scheduling:
    scheduler: "llm-d-inference-scheduler"
    sloPolicy:
      enabled: true
      objectives:
        - name: "request_latency_p95"
          target: "500ms"
          weight: 0.5
        - name: "availability"
          target: "99.9%"
          weight: 0.3
        - name: "cost_per_request"
          target: "$0.001"
          weight: 0.2
    nodeSelector:
      node-type: gpu
      instance-family: a100
      environment: production
    tolerations:
    - key: "dedicated"
      operator: "Equal"
      value: "ml-workloads"
      effect: "NoSchedule"
    affinity:
      podAntiAffinity:
        preferredDuringSchedulingIgnoredDuringExecution:
        - weight: 100
          podAffinityTerm:
            labelSelector:
              matchExpressions:
              - key: llm-d.ai/model
                operator: In
                values: ["llama-3.1"]
            topologyKey: kubernetes.io/hostname
  monitoring:
    enabled: true
    metrics:
      enabled: true
      port: 8081
      path: "/metrics"
      detailed: true
    alerting:
      enabled: true
      channels: ["#sre-alerts", "#ml-platform-alerts"]
      rules:
        - name: "high_latency"
          condition: "p95_latency > 1000ms"
          severity: "critical"
        - name: "low_availability"
          condition: "availability < 99.5%"
          severity: "critical"
    tracing:
      enabled: true
      samplingRate: 0.1
  security:
    enabled: true
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 2000
    readOnlyRootFilesystem: true
    capabilities:
      drop:
      - ALL
    seccompProfile:
      type: RuntimeDefault
  backup:
    enabled: true
    schedule: "0 2 * * *"  # Daily at 2 AM
    retention: "30d"
  rollback:
    enabled: true
    previousVersions:
      - "v1.1.3"
      - "v1.0.5"
    autoRollback:
      enabled: true
      triggers:
        - errorRate: ">1%"
        - latencyP95: ">1000ms"
        - throughput: "<10 tokens/sec"
---
# Production service with load balancing
apiVersion: v1
kind: Service
metadata:
  name: llama-3.1-8b-prod-service
  namespace: production
  labels:
    app.kubernetes.io/name: llm-d
    llm-d.ai/model: llama-3.1
  annotations:
    service.llm-d.ai/load-balancer: "external"
    service.llm-d.ai/ssl-redirect: "true"
spec:
  selector:
    app.kubernetes.io/name: llm-d
    llm-d.ai/model: llama-3.1
    llm-d.ai/size: 8b
  ports:
  - name: http
    port: 8080
    targetPort: 8080
    protocol: TCP
  - name: metrics
    port: 8081
    targetPort: 8081
    protocol: TCP
  sessionAffinity: None
  type: ClusterIP
---
# Production HPA with multiple metrics
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llama-3.1-8b-prod-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llama-3.1-8b-prod
  minReplicas: 3
  maxReplicas: 15
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60
  - type: Pods
    pods:
      metric:
        name: nvidia_gpu_utilization_percent
      target:
        type: AverageValue
        averageValue: "60"
  - type: Pods
    pods:
      metric:
        name: request_latency_p95_milliseconds
      target:
        type: AverageValue
        averageValue: "400"
---
# Production network policy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: llama-3.1-8b-prod-netpol
  namespace: production
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: llm-d
      llm-d.ai/model: llama-3.1
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: production
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 8080
    - protocol: TCP
      port: 8081
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS for model downloads
    - protocol: TCP
      port: 53   # DNS
    - protocol: UDP
      port: 53   # DNS
```

### High-Availability Production Setup

```yaml title="production/ha-llm-deployment.yaml"
apiVersion: inference.llm-d.io/v1alpha1
kind: LLMDeployment
metadata:
  name: llama-3.1-8b-ha
  namespace: production
  labels:
    app.kubernetes.io/name: llm-d
    llm-d.ai/availability: high
spec:
  model:
    modelUri: "hf://meta-llama/Llama-3.1-8B-Instruct"
    version: "v1.2.0"
    quantization:
      type: int8
  resources:
    requests:
      memory: "32Gi"
      cpu: "8"
      nvidia.com/gpu: "2"
    limits:
      memory: "48Gi"
      cpu: "16"
      nvidia.com/gpu: "2"
  autoscaling:
    enabled: true
    minReplicas: 5  # High minimum for availability
    maxReplicas: 20
    targetGPUUtilization: 50  # Very conservative
  scheduling:
    affinity:
      podAntiAffinity:
        requiredDuringSchedulingIgnoredDuringExecution:
        - labelSelector:
            matchExpressions:
            - key: llm-d.ai/model
              operator: In
              values: ["llama-3.1"]
          topologyKey: kubernetes.io/hostname
        - labelSelector:
            matchExpressions:
            - key: llm-d.ai/model
              operator: In
              values: ["llama-3.1"]
          topologyKey: topology.kubernetes.io/zone
    nodeSelector:
      node-type: gpu
    tolerations:
    - key: "dedicated"
      operator: "Equal"
      value: "ml-workloads"
      effect: "NoSchedule"
  monitoring:
    enabled: true
    alerting:
      enabled: true
      channels: ["#sre-critical", "#ml-platform-critical"]
  disruption:
    maxUnavailable: 1  # Only one pod down at a time
    minAvailable: 4    # Always keep 4 pods running
```

**Key customization points for production:**
- `spec.model.version`: Always pin to specific versions
- `spec.autoscaling.minReplicas`: Adjust based on traffic patterns
- `spec.scheduling.sloPolicy.objectives`: Customize SLO targets
- `spec.monitoring.alerting.channels`: Update alert destinations
- `spec.security`: Enable all security features for compliance

## Multi-Tenant Environment Templates

### Shared Infrastructure with Isolation

```yaml title="multi-tenant/tenant-a-deployment.yaml"
apiVersion: v1
kind: Namespace
metadata:
  name: tenant-a
  labels:
    tenant.llm-d.ai/name: tenant-a
    tenant.llm-d.ai/tier: premium
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: tenant-a-quota
  namespace: tenant-a
spec:
  hard:
    requests.nvidia.com/gpu: "8"
    limits.nvidia.com/gpu: "8"
    requests.memory: "128Gi"
    limits.memory: "256Gi"
    pods: "20"
    services: "10"
---
apiVersion: inference.llm-d.io/v1alpha1
kind: LLMDeployment
metadata:
  name: llama-3.1-8b-tenant-a
  namespace: tenant-a
  labels:
    app.kubernetes.io/name: llm-d
    tenant.llm-d.ai/name: tenant-a
    tenant.llm-d.ai/tier: premium
spec:
  model:
    modelUri: "hf://meta-llama/Llama-3.1-8B-Instruct"
    quantization:
      type: int8
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
    maxReplicas: 6  # Limited by quota
    targetGPUUtilization: 70
  scheduling:
    nodeSelector:
      tenant.llm-d.ai/allowed: "tenant-a"
    tolerations:
    - key: "tenant"
      operator: "Equal"
      value: "tenant-a"
      effect: "NoSchedule"
  monitoring:
    enabled: true
    labels:
      tenant: "tenant-a"
  security:
    enabled: true
    networkPolicy:
      enabled: true
      isolateTenant: true
---
# Tenant-specific network policy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: tenant-a-isolation
  namespace: tenant-a
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          tenant.llm-d.ai/name: tenant-a
    - namespaceSelector:
        matchLabels:
          name: monitoring
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
```

## Helm Chart Templates

### Chart Structure

Following llm-d-deployer patterns, here's the complete Helm chart structure:

```yaml title="helm/llm-d-chart/Chart.yaml"
apiVersion: v2
name: llm-d-deployment
version: 1.0.0
appVersion: "v1.2.0"
type: application
description: Helm chart for llm-d LLM deployments
kubeVersion: ">= 1.25.0-0"
keywords:
  - llm-d
  - llm
  - inference
  - ai
  - machine-learning
annotations:
  category: ai-machine-learning
  licenses: Apache-2.0
dependencies:
  - name: common
    repository: https://charts.bitnami.com/bitnami
    version: "2.27.0"
```

### Values Configuration

```yaml title="helm/llm-d-chart/values.yaml"
# Global configuration
global:
  imageRegistry: ""
  imagePullSecrets: []

# Model configuration
model:
  modelUri: "hf://meta-llama/Llama-3.1-8B-Instruct"
  version: "latest"
  quantization:
    enabled: true
    type: "int8"
  registry:
    enabled: false
    endpoint: ""
    credentials:
      secretName: ""

# Resource configuration
resources:
  requests:
    memory: "16Gi"
    cpu: "4"
    gpu: "1"
  limits:
    memory: "24Gi"
    cpu: "8"
    gpu: "1"

# Autoscaling configuration
autoscaling:
  enabled: true
  minReplicas: 1
  maxReplicas: 10
  targetGPUUtilization: 70
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
    scaleDown:
      stabilizationWindowSeconds: 300

# Scheduling configuration
scheduling:
  scheduler: ""
  nodeSelector:
    node-type: gpu
  tolerations: []
  affinity: {}
  sloPolicy:
    enabled: false
    objectives: []

# Service configuration
service:
  type: ClusterIP
  ports:
    http: 8080
    metrics: 8081
  annotations: {}

# Monitoring configuration
monitoring:
  enabled: true
  metrics:
    enabled: true
    port: 8081
    path: "/metrics"
  alerting:
    enabled: false
    channels: []

# Security configuration
security:
  enabled: true
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 2000
  readOnlyRootFilesystem: true
  capabilities:
    drop:
    - ALL
  seccompProfile:
    type: RuntimeDefault

# Network policy
networkPolicy:
  enabled: false
  policyTypes:
    - Ingress
    - Egress

# Environment-specific overrides
environment: "development"

# Extra resources
extraDeploy: []
```

### Deployment Template

```yaml title="helm/llm-d-chart/templates/llmdeployment.yaml"
{{- if .Values.enabled | default true }}
apiVersion: inference.llm-d.io/v1alpha1
kind: LLMDeployment
metadata:
  name: {{ include "llm-d-chart.fullname" . }}
  namespace: {{ .Release.Namespace | quote }}
  labels:
    {{- include "llm-d-chart.labels" . | nindent 4 }}
  {{- if .Values.annotations }}
  annotations:
    {{- include "common.tplvalues.render" (dict "value" .Values.annotations "context" $) | nindent 4 }}
  {{- end }}
spec:
  model:
    modelUri: {{ .Values.model.modelUri | quote }}
    {{- if .Values.model.version }}
    version: {{ .Values.model.version | quote }}
    {{- end }}
    {{- if .Values.model.quantization.enabled }}
    quantization:
      type: {{ .Values.model.quantization.type | quote }}
    {{- end }}
    {{- if .Values.model.registry.enabled }}
    registry:
      endpoint: {{ .Values.model.registry.endpoint | quote }}
      {{- if .Values.model.registry.credentials.secretName }}
      credentials:
        secretName: {{ .Values.model.registry.credentials.secretName | quote }}
      {{- end }}
    {{- end }}
  resources:
    requests:
      memory: {{ .Values.resources.requests.memory | quote }}
      cpu: {{ .Values.resources.requests.cpu | quote }}
      nvidia.com/gpu: {{ .Values.resources.requests.gpu | quote }}
    limits:
      memory: {{ .Values.resources.limits.memory | quote }}
      cpu: {{ .Values.resources.limits.cpu | quote }}
      nvidia.com/gpu: {{ .Values.resources.limits.gpu | quote }}
  {{- if .Values.autoscaling.enabled }}
  autoscaling:
    enabled: true
    minReplicas: {{ .Values.autoscaling.minReplicas }}
    maxReplicas: {{ .Values.autoscaling.maxReplicas }}
    targetGPUUtilization: {{ .Values.autoscaling.targetGPUUtilization }}
    {{- if .Values.autoscaling.behavior }}
    behavior:
      {{- include "common.tplvalues.render" (dict "value" .Values.autoscaling.behavior "context" $) | nindent 6 }}
    {{- end }}
  {{- end }}
  {{- if or .Values.scheduling.nodeSelector .Values.scheduling.tolerations .Values.scheduling.affinity .Values.scheduling.sloPolicy.enabled }}
  scheduling:
    {{- if .Values.scheduling.scheduler }}
    scheduler: {{ .Values.scheduling.scheduler | quote }}
    {{- end }}
    {{- if .Values.scheduling.nodeSelector }}
    nodeSelector:
      {{- include "common.tplvalues.render" (dict "value" .Values.scheduling.nodeSelector "context" $) | nindent 6 }}
    {{- end }}
    {{- if .Values.scheduling.tolerations }}
    tolerations:
      {{- include "common.tplvalues.render" (dict "value" .Values.scheduling.tolerations "context" $) | nindent 6 }}
    {{- end }}
    {{- if .Values.scheduling.affinity }}
    affinity:
      {{- include "common.tplvalues.render" (dict "value" .Values.scheduling.affinity "context" $) | nindent 6 }}
    {{- end }}
    {{- if .Values.scheduling.sloPolicy.enabled }}
    sloPolicy:
      enabled: true
      {{- if .Values.scheduling.sloPolicy.objectives }}
      objectives:
        {{- include "common.tplvalues.render" (dict "value" .Values.scheduling.sloPolicy.objectives "context" $) | nindent 8 }}
      {{- end }}
    {{- end }}
  {{- end }}
  {{- if .Values.monitoring.enabled }}
  monitoring:
    enabled: true
    {{- if .Values.monitoring.metrics.enabled }}
    metrics:
      enabled: true
      port: {{ .Values.monitoring.metrics.port }}
      path: {{ .Values.monitoring.metrics.path | quote }}
    {{- end }}
    {{- if .Values.monitoring.alerting.enabled }}
    alerting:
      enabled: true
      {{- if .Values.monitoring.alerting.channels }}
      channels:
        {{- range .Values.monitoring.alerting.channels }}
        - {{ . | quote }}
        {{- end }}
      {{- end }}
    {{- end }}
  {{- end }}
  {{- if .Values.security.enabled }}
  security:
    enabled: true
    runAsNonRoot: {{ .Values.security.runAsNonRoot }}
    runAsUser: {{ .Values.security.runAsUser }}
    fsGroup: {{ .Values.security.fsGroup }}
    readOnlyRootFilesystem: {{ .Values.security.readOnlyRootFilesystem }}
    capabilities:
      drop:
        {{- range .Values.security.capabilities.drop }}
        - {{ . }}
        {{- end }}
    seccompProfile:
      type: {{ .Values.security.seccompProfile.type }}
  {{- end }}
{{- end }}
```

### Environment-Specific Values

```yaml title="helm/llm-d-chart/values-development.yaml"
# Development environment overrides
environment: "development"

resources:
  requests:
    memory: "8Gi"
    cpu: "2"
    gpu: "1"
  limits:
    memory: "16Gi"
    cpu: "4"
    gpu: "1"

autoscaling:
  minReplicas: 0  # Scale to zero
  maxReplicas: 3
  targetGPUUtilization: 80

scheduling:
  nodeSelector:
    node-type: gpu
    environment: development
  tolerations:
  - key: "spot-instance"
    operator: "Equal"
    value: "true"
    effect: "NoSchedule"

monitoring:
  alerting:
    enabled: false

security:
  enabled: false  # Relaxed for development
```

```yaml title="helm/llm-d-chart/values-production.yaml"
# Production environment overrides
environment: "production"

model:
  version: "v1.2.0"  # Pin version
  registry:
    enabled: true
    endpoint: "company-registry.com/models"
    credentials:
      secretName: "model-registry-secret"

resources:
  requests:
    memory: "32Gi"
    cpu: "8"
    gpu: "2"
  limits:
    memory: "48Gi"
    cpu: "16"
    gpu: "2"

autoscaling:
  minReplicas: 3
  maxReplicas: 15
  targetGPUUtilization: 60
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
    scaleDown:
      stabilizationWindowSeconds: 600

scheduling:
  scheduler: "llm-d-inference-scheduler"
  nodeSelector:
    node-type: gpu
    instance-family: a100
    environment: production
  sloPolicy:
    enabled: true
    objectives:
    - name: "request_latency_p95"
      target: "500ms"
      weight: 0.5
    - name: "availability"
      target: "99.9%"
      weight: 0.3
    - name: "cost_per_request"
      target: "$0.001"
      weight: 0.2

monitoring:
  alerting:
    enabled: true
    channels:
    - "#sre-alerts"
    - "#ml-platform-alerts"

networkPolicy:
  enabled: true
```

### Deployment Commands

```bash
# Install development environment
helm install llama-3.1-8b-dev ./helm/llm-d-chart \
  --namespace development \
  --create-namespace \
  --values ./helm/llm-d-chart/values-development.yaml

# Install production environment
helm install llama-3.1-8b-prod ./helm/llm-d-chart \
  --namespace production \
  --create-namespace \
  --values ./helm/llm-d-chart/values-production.yaml

# Upgrade with new values
helm upgrade llama-3.1-8b-prod ./helm/llm-d-chart \
  --namespace production \
  --values ./helm/llm-d-chart/values-production.yaml

# Template and validate without installing
helm template llama-3.1-8b-test ./helm/llm-d-chart \
  --values ./helm/llm-d-chart/values-development.yaml \
  --dry-run
```

## Usage Guidelines

### Template Selection Guide

| Scenario | Template | Key Features |
|----------|----------|--------------|
| Local development | `development/minimal-llm-deployment.yaml` | Cost-optimized, spot instances |
| Debugging/testing | `development/debug-llm-deployment.yaml` | Verbose logging, profiling |
| Integration testing | `staging/staging-llm-deployment.yaml` | Production-like, reduced resources |
| Production service | `production/production-llm-deployment.yaml` | Full monitoring, security, scaling |
| High availability | `production/ha-llm-deployment.yaml` | Multi-zone, high replica count |
| Multi-tenancy | `multi-tenant/tenant-a-deployment.yaml` | Resource quotas, network isolation |
| Flexible deployment | Helm chart | Environment-specific values |

### Customization Checklist

Before deploying any template:

- [ ] Update `namespace` to match your environment
- [ ] Adjust `spec.model.modelUri` for your model
- [ ] Set appropriate `resources.requests` and `limits`
- [ ] Configure `nodeSelector` and `tolerations` for your nodes
- [ ] Update `monitoring.alerting.channels` for your team
- [ ] Review `security` settings for compliance requirements
- [ ] Set `autoscaling` parameters based on expected load

---

:::tip Template Best Practices
- Always pin model versions in production (`spec.model.version`)
- Use resource quotas in multi-tenant environments
- Enable monitoring and alerting for all non-development deployments
- Test scaling behavior in staging before production deployment
- Keep environment-specific values in separate files
:::

:::info Next Steps
- Review [CRD Reference](./crd-reference.md) for complete field specifications
- Check [Command Reference](./command-reference.md) for deployment and management commands
- Reference [Shared Configuration](./shared-config.md) for naming conventions and standards
- Reference main chapters for detailed explanations of configuration options
:::
