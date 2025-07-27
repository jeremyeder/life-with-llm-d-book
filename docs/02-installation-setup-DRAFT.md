---
title: Installation and Setup
description: Strategic guide to installing and configuring llm-d on Kubernetes and OpenShift clusters
sidebar_position: 2
---

# Installation and Setup

:::info Chapter Overview
This chapter provides a strategic approach to installing llm-d that aligns with your team structure and organizational goals. You'll learn how to choose the right installation method, apply proven deployment patterns, and build sustainable operational practices from day one.

**Strategic Foundation**: Infrastructure decisions impact team effectiveness for years. This chapter helps you make informed choices that scale with your organization.
:::

## Strategic Foundation: Why LLM Infrastructure Matters

Before diving into installation mechanics, let's establish the strategic context using Lencioni's Six Questions framework:

### Why do we exist?

Your LLM infrastructure exists to **enable rapid experimentation and reliable production deployment** of language models. This infrastructure should be:

- **Boring**: Predictable, reliable, well-understood operations that fade into the background
- **Enabling**: Reduces time from idea to production deployment by 10x or more  
- **Scalable**: Grows with your team and usage patterns without architectural rewrites
- **Cost-effective**: Optimizes for total cost of ownership, not just initial deployment speed

The strategic value is **platform leverage** - investing infrastructure effort once to enable exponential team productivity gains.

### How do we behave?

Installation and configuration decisions should reflect these operational principles:

- **Upstream-first**: Contribute improvements back to the open source community
- **Team-topology-aware**: Match infrastructure complexity to your organizational structure
- **Measurement-driven**: Instrument everything from day one for continuous improvement
- **Learning-oriented**: Build in feedback loops and retrospective practices
- **Conway's Law conscious**: Your installation method shapes your system architecture

### What do we do?

This chapter provides **three proven installation paths** optimized for different team structures:

- **Automated Installation**: For teams prioritizing speed and standardization
- **Helm-based Installation**: For teams needing customization and upstream contribution
- **GitOps Installation**: For teams with mature operational practices and multiple stakeholders

Each path includes decision criteria, team collaboration patterns, and upstream contribution opportunities.

### How will we succeed?

Success metrics for your installation (measure from day one):

- **Time to first inference**: < 30 minutes from start to working model
- **Team onboarding time**: New team members productive in < 1 day  
- **Configuration drift**: Zero unplanned configuration changes
- **Learning velocity**: Regular improvements identified and implemented
- **Upstream contributions**: Consistent contributions back to the community

### What is most important right now?

**Priority 1**: Choose the installation method that matches your current team topology
**Priority 2**: Establish measurement and learning practices from the start
**Priority 3**: Plan your contribution strategy to the upstream community

### Who must do what?

**Platform Engineer/SRE**: Infrastructure setup, monitoring configuration, operational procedures
**Data Scientist/ML Engineer**: Model deployment testing, performance validation, workflow integration  
**Team Lead**: Installation method selection, team coordination, success criteria definition

Clear role boundaries prevent overlap and ensure all critical areas are covered.

## Team-Topology-Based Installation Strategy

Your installation method should align with your team structure and organizational maturity. This prevents Conway's Law violations that create long-term architectural debt.

### Decision Framework

| Team Type | Recommended Method | Why | Tradeoffs |
|-----------|-------------------|-----|-----------|
| **Stream-Aligned** (2-8 people, product-focused) | Automated | Minimizes cognitive load, maximizes delivery focus | Less customization flexibility |
| **Platform** (Serving multiple teams) | GitOps | Enables self-service, scales team leverage | Higher initial complexity |
| **Enabling** (Providing capability) | Helm | Maximum flexibility for teaching/customization | Requires deeper expertise |
| **Individual/Research** | Automated | Fastest path to experimentation | Limited production scalability |

### Team Topology Considerations

**Stream-Aligned Teams**:

- Need **fast feedback loops** and minimal operational overhead
- Benefit from **standardized, opinionated setups** that reduce decision fatigue
- Should focus cognitive capacity on **business logic, not infrastructure**

**Platform Teams**:

- Must **serve multiple downstream teams** with different needs
- Require **self-service capabilities** and clear boundaries
- Need **scalable, repeatable deployment patterns**

**Enabling Teams**:

- Focus on **knowledge transfer** and capability building
- Need **maximum flexibility** for experimentation and teaching
- Should **contribute learnings back** to the broader community

### Conway's Law Implications

Your installation method influences your system architecture:

- **Automated** → Simple, monolithic deployments → Good for small, focused teams
- **Helm** → Modular, configurable systems → Good for teams that need customization
- **GitOps** → Distributed, declarative architectures → Good for complex organizations

Choose thoughtfully - changing later requires both technical and organizational changes.

## Installation Method Selection

### Method 1: Automated Installation (Stream-Aligned Teams)

**When to choose**: Teams prioritizing speed, standardization, and minimal operational overhead.

**Strategic value**: Enables rapid experimentation with minimal infrastructure investment.

**Team requirements**: Basic Kubernetes knowledge, willingness to accept opinionated defaults.

#### Prerequisites

**Cluster Requirements:**

- Kubernetes 1.24+ or OpenShift 4.12+  
- GPU-enabled nodes with container runtime support
- Storage provisioning capability (minimum 100GB available)
- Network connectivity for model downloads
- Cluster admin privileges for initial setup

**Hardware Specifications (Capability-Based):**

*Development/Learning Setup:*

- GPU memory bandwidth: > 500 GB/s
- System memory: 64GB+ for model loading and caching
- Storage throughput: > 1 GB/s for model loading
- Network: 1 Gbps+ for model downloads

*Production Setup:*  

- GPU memory bandwidth: > 1 TB/s for sustained workloads
- System memory: 256GB+ for multiple model hosting
- Storage throughput: > 5 GB/s for concurrent model loading
- Network: 10 Gbps+ with redundancy

**Required Tools:**

```bash
# Essential toolchain
kubectl       # Kubernetes CLI
helm          # Package manager  
yq            # YAML processor
jq            # JSON processor
git           # Version control
kustomize     # Configuration management

# Installation for RHEL/CentOS/Fedora
sudo dnf install -y git jq

# For RHEL/CentOS 8+
sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo dnf install -y kubectl

# For Fedora  
sudo dnf install -y kubectl helm

# Install yq (universal)
sudo wget -qO /usr/local/bin/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64
sudo chmod +x /usr/local/bin/yq

# Install kustomize (universal)
curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
sudo mv kustomize /usr/local/bin/
```

**Access Requirements:**

```bash
# Verify cluster connectivity
kubectl cluster-info
kubectl get nodes

# Verify GPU availability
kubectl describe nodes | grep -A 5 "nvidia.com/gpu"

# Test cluster permissions
kubectl auth can-i create customresourcedefinitions
```

#### Installation Procedure

**Step 1: Environment Preparation**

```bash
git clone https://github.com/llm-d/llm-d-deployer.git
cd llm-d-deployer/quickstart

# Install dependencies and validate environment
./install-deps.sh
```

**Step 2: Configuration**

```bash
# Set model access credentials
export HF_TOKEN="hf_your_token_here"

# Optional customizations
export LLMD_NAMESPACE="llm-d"
export STORAGE_CLASS="fast-ssd"  # Use your cluster's fast storage class
```

**Step 3: Installation**

```bash
# Basic installation with sensible defaults
./llmd-installer.sh

# Advanced installation with customization
./llmd-installer.sh \
  --namespace production \
  --storage-size 1Ti \
  --storage-class nvme-ssd \
  --values-file custom-values.yaml
```

**Installation Flags:**

| Flag | Purpose | Default | Strategic Consideration |
|------|---------|---------|------------------------|
| `-n, --namespace` | Kubernetes namespace | `llm-d` | Align with team boundaries |
| `-z, --storage-size` | Persistent storage size | `100Gi` | Size for your model portfolio |
| `-c, --storage-class` | Storage performance tier | Default cluster class | Match performance needs |
| `-f, --values-file` | Custom configuration | None | Standardize team configs |
| `-D, --download-model` | Pre-populate model cache | False | Reduce first-request latency |
| `-m, --disable-metrics` | Skip monitoring setup | False | Always enable for learning |

### Method 2: Helm Installation (Enabling Teams)

**When to choose**: Teams needing customization flexibility and planning to contribute upstream.

**Strategic value**: Maximum configurability for experimentation and knowledge transfer.

**Team requirements**: Intermediate Kubernetes expertise, Helm experience, commitment to documentation.

#### Installation Procedure

**Step 1: Repository Setup**

```bash
# Add official Helm repository
helm repo add llm-d https://helm.llm-d.ai
helm repo update

# Verify repository access
helm search repo llm-d
```

**Step 2: Namespace Preparation**

```bash
# Create dedicated namespace
kubectl create namespace llm-d

# Set up RBAC (if required)
kubectl apply -f rbac-setup.yaml
```

**Step 3: Custom Configuration**

```bash
# Create custom values file
cat > production-values.yaml <<EOF
global:
  namespace: llm-d
  monitoring:
    enabled: true
    
sampleApplication:
  model:
    # Use capability-based model selection
    modelArtifactURI: "hf://meta-llama/Llama-3.1-8B-Instruct"
    modelName: "llama3-8b"
    
  resources:
    # Right-size based on actual requirements
    prefill:
      gpu_memory: "24Gi"
      system_memory: "64Gi"
    decode:  
      gpu_memory: "16Gi"
      system_memory: "32Gi"
      
  autoscaling:
    enabled: true
    metrics:
      - type: "gpu_utilization"
        target: 70
      - type: "request_latency_p95"  
        target: "500ms"

monitoring:
  prometheus:
    enabled: true
    retention: "30d"
  grafana:
    enabled: true
    persistence:
      enabled: true
      
kvCache:
  redis:
    enabled: true
    persistence:
      enabled: true
      storageClass: "fast-ssd"
EOF
```

**Step 4: Installation with Monitoring**

```bash
# Install with custom configuration
helm install llm-d llm-d/llm-d \
  --namespace llm-d \
  --values production-values.yaml \
  --set global.hfToken="${HF_TOKEN}"

# Verify installation
helm status llm-d -n llm-d
```

### Method 3: GitOps Installation (Platform Teams)

**When to choose**: Multiple teams, mature operational practices, compliance requirements.

**Strategic value**: Enables self-service while maintaining governance and consistency.

**Team requirements**: Advanced Kubernetes expertise, GitOps tooling, CI/CD infrastructure.

#### Installation Procedure

**Step 1: Repository Preparation**

```bash
# Fork the official repository for customization
git clone https://github.com/llm-d/llm-d-deployer.git
cd llm-d-deployer/gitops

# Create environment-specific branches
git checkout -b production
git checkout -b staging  
git checkout -b development
```

**Step 2: Configuration Management**

```bash
# Customize base configuration
cat > kustomization.yaml <<EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: llm-d

resources:
  - base/

patches:
  - patch: |
      - op: replace
        path: /spec/model/modelArtifactURI
        value: "hf://meta-llama/Llama-3.1-8B-Instruct"
    target:
      kind: LLMDeployment
      name: sample-application

configMapGenerator:
  - name: environment-config
    literals:
      - environment=production
      - monitoring.enabled=true
      - autoscaling.enabled=true
EOF
```

**Step 3: ArgoCD Application**

```yaml
# argocd-application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: llm-d-production
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/your-org/llm-d-deployer.git
    targetRevision: production
    path: gitops
  destination:
    server: https://kubernetes.default.svc
    namespace: llm-d
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
    retry:
      limit: 3
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
```

**Step 4: Multi-Environment Management**

```bash
# Apply ArgoCD application
kubectl apply -f argocd-application.yaml

# Monitor synchronization
argocd app sync llm-d-production
argocd app wait llm-d-production
```

## Verification and Learning Framework

### Installation Verification

**System Health Checks:**

```bash
# Verify namespace and resources
kubectl get namespace llm-d
kubectl get pods -n llm-d -o wide
kubectl get pvc -n llm-d
kubectl get services -n llm-d

# Check resource allocation
kubectl describe nodes | grep -A 10 "Allocated resources"
kubectl top pods -n llm-d
```

**Expected Healthy State:**

```bash
$ kubectl get pods -n llm-d
NAME                                    READY   STATUS    RESTARTS   AGE
llm-d-prefill-0                        1/1     Running   0          5m
llm-d-decode-0                         1/1     Running   0          5m  
llm-d-inference-scheduler-xxx          1/1     Running   0          5m
llm-d-kv-cache-manager-xxx             1/1     Running   0          5m
prometheus-server-xxx                  1/1     Running   0          5m
grafana-xxx                           1/1     Running   0          5m
```

### Functional Testing

**Health Endpoint Validation:**

```bash
# Test inference scheduler
kubectl port-forward svc/llm-d-inference-scheduler 8080:8080 -n llm-d
curl http://localhost:8080/health

# Test model service  
kubectl port-forward svc/llm-d-model-service 8000:8000 -n llm-d
curl http://localhost:8000/v1/models
```

**First Inference Request:**

```bash
# Port-forward to model service
kubectl port-forward svc/llm-d-model-service 8000:8000 -n llm-d &

# Test inference capability
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3-8b",
    "messages": [
      {
        "role": "user",
        "content": "Explain the concept of infrastructure as code in one sentence."
      }
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

**Expected Response Pattern:**

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion", 
  "created": 1703123456,
  "model": "llama3-8b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Infrastructure as code treats infrastructure configuration as software, enabling version control, automated deployment, and consistent, repeatable infrastructure management."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 28,
    "total_tokens": 43
  }
}
```

### Performance Baseline Establishment

**Key Metrics to Capture:**

```bash
# Latency measurement
time curl -X POST http://localhost:8000/v1/chat/completions [request]

# Throughput testing (install hey first)
hey -n 100 -c 5 -m POST -H "Content-Type: application/json" \
  -d '{"model":"llama3-8b","messages":[{"role":"user","content":"test"}],"max_tokens":10}' \
  http://localhost:8000/v1/chat/completions

# Resource utilization
kubectl top pods -n llm-d
kubectl exec -it llm-d-prefill-0 -n llm-d -- nvidia-smi
```

## Monitoring and Observability Setup

### Prometheus Integration

**Access Prometheus Dashboard:**

```bash
# Port-forward to Prometheus
kubectl port-forward svc/prometheus-server 9090:80 -n llm-d
# Open http://localhost:9090
```

**Critical Metrics to Monitor:**

- `llm_d_inference_requests_total` - Request volume and patterns
- `llm_d_inference_duration_seconds` - Response latency distribution  
- `llm_d_gpu_memory_usage_bytes` - Resource utilization efficiency
- `llm_d_cache_hit_ratio` - Cache effectiveness
- `llm_d_queue_depth` - System load and capacity planning

### Grafana Dashboard Configuration

**Access Grafana:**

```bash
# Port-forward to Grafana
kubectl port-forward svc/grafana 3000:80 -n llm-d
# Open http://localhost:3000 (admin/admin)
```

**Pre-configured Strategic Dashboards:**

- **Executive Summary**: High-level system health and business metrics
- **Engineering Metrics**: Detailed performance and reliability indicators
- **Capacity Planning**: Resource utilization trends and forecasting
- **Cost Analysis**: Resource efficiency and optimization opportunities

## Troubleshooting Framework

### Systematic Debugging Approach

**Step 1: Gather Information**

```bash
# Collect system state snapshot
kubectl get all -n llm-d -o wide > system-snapshot.yaml
kubectl describe pods -n llm-d > pod-details.txt
kubectl logs -l app.kubernetes.io/name=llm-d --all-containers=true -n llm-d > all-logs.txt
```

**Step 2: Identify Problem Category**

| Symptom | Likely Cause | Investigation Command |
|---------|--------------|----------------------|
| Pods Pending | Resource constraints | `kubectl describe pod <name> -n llm-d` |
| ImagePullBackOff | Credential/network issues | `kubectl describe pod <name> -n llm-d` |
| High Latency | Resource saturation | `kubectl top pods -n llm-d; nvidia-smi` |
| Connection Refused | Service configuration | `kubectl get svc,ep -n llm-d` |

**Step 3: Apply Systematic Solutions**

*Resource Constraint Resolution:*

```bash
# Check GPU availability
kubectl describe nodes | grep -A 10 "nvidia.com/gpu"

# Verify storage provisioning
kubectl get storageclass
kubectl describe pvc -n llm-d

# Review resource requests vs. limits
kubectl describe pods -n llm-d | grep -A 5 "Requests:\|Limits:"
```

*Network Configuration Issues:*

```bash
# Test service connectivity
kubectl run debug --image=curlimages/curl -it --rm -- sh
# From debug pod: curl http://llm-d-model-service.llm-d:8000/health

# Check ingress configuration
kubectl get ingress -n llm-d -o yaml
kubectl describe ingress -n llm-d
```

### Learning from Failures

**Post-Incident Analysis Framework:**

1. **Timeline reconstruction**: What happened when?
2. **Root cause analysis**: Why did it happen?
3. **Contributing factors**: What made it worse?
4. **Prevention measures**: How do we prevent recurrence?
5. **Upstream contributions**: What can we share with the community?

## Continuous Improvement Patterns

### Post-Installation Retrospective

Conduct this retrospective 1 week after installation:

**What worked well?**

- Which installation steps were smooth and well-documented?
- What automation saved the most time?
- Which monitoring provided immediate value?
- Where did team collaboration excel?

**What could be improved?**

- Which steps required additional research or iteration?
- Where did team knowledge gaps create delays?
- What customizations were immediately needed?
- Which documentation needs enhancement?

**Action Items:**

- **Upstream contributions**: Documentation improvements, bug reports, feature requests
- **Internal improvements**: Runbook updates, automation enhancements, training needs
- **Community engagement**: Share learnings, participate in discussions, mentor others

### Learning Acceleration Patterns

**Weekly Infrastructure Reviews:**

- Monitor installation success rates for new team members
- Track time-to-productivity metrics for the team
- Identify configuration drift and technical debt
- Plan contributions to upstream projects

**Monthly Architecture Reviews:**

- Assess team topology alignment with current infrastructure
- Review cost/benefit of current installation approach  
- Plan evolution toward more sophisticated patterns
- Evaluate new upstream capabilities and community developments

**Quarterly Strategic Reviews:**

- Measure against original success criteria
- Assess team scaling and infrastructure scaling
- Plan major architectural improvements
- Evaluate contribution strategy effectiveness

### Upstream Contribution Strategy

**Contribution Categories:**

*Documentation Improvements:*

- Installation guides for specific environments
- Troubleshooting procedures for common issues
- Configuration examples for different team structures
- Performance tuning guides

*Code Contributions:*

- Bug fixes discovered during installation
- Feature enhancements that benefit the community
- New installation automation or tools
- Monitoring and observability improvements

*Community Engagement:*

- Share installation experiences and lessons learned
- Mentor new community members
- Participate in architecture discussions
- Provide feedback on roadmap priorities

## Summary and Strategic Next Steps

### Key Strategic Takeaways

✅ **Infrastructure choice impacts team effectiveness** - Choose installation method based on team topology, not just technical preferences

✅ **Measure from day one** - Establish baseline metrics immediately for continuous improvement

✅ **Build learning into operations** - Regular retrospectives and improvement cycles prevent technical debt accumulation

✅ **Contribute upstream** - Shared improvements benefit the entire community and improve your own capabilities

### Installation Method Decision Summary

| Team Structure | Method | Why | Next Chapter Focus |
|----------------|---------|-----|-------------------|
| **Stream-Aligned** | Automated | Minimize cognitive load | Chapter 4: Data Scientist Workflows |
| **Platform** | GitOps | Enable self-service | Chapter 5: SRE Operations |
| **Enabling** | Helm | Maximum flexibility | Chapter 6: Performance Optimization |
| **Research/Individual** | Automated | Fastest experimentation | Chapter 3: Understanding Architecture |

### Immediate Next Steps

**Week 1**: Complete installation and verify all health checks pass
**Week 2**: Establish monitoring baselines and conduct retrospective  
**Week 3**: Plan first upstream contribution based on installation experience
**Month 1**: Evaluate team productivity improvements and plan scaling

### Preparation for Next Chapters

With llm-d successfully installed and strategic practices established, you're ready for:

1. **Chapter 3: Understanding Architecture** - Deep dive into system design and optimization opportunities
2. **Chapter 4: Data Scientist Workflows** - Productive patterns for model development and deployment  
3. **Chapter 5: SRE Operations** - Production reliability and operational excellence
4. **Appendices** - Reference materials for ongoing operations

:::tip Strategic Success Pattern
The most successful LLM infrastructure deployments start simple, measure everything, improve continuously, and contribute back to the community. This creates a virtuous cycle of learning and capability building.
:::

---

:::info References

- [llm-d-deployer Repository](https://github.com/llm-d/llm-d-deployer) - Official installation tools and documentation
- [Team Topologies](https://teamtopologies.com/) - Organizational patterns for technology effectiveness  
- [Site Reliability Engineering](https://sre.google/sre-book/) - Infrastructure excellence and measurement practices
- [The Phoenix Project](https://itrevolution.com/the-phoenix-project/) - DevOps transformation patterns
- [Shared Configuration Reference](./appendix/shared-config.md) - Standardized configurations and best practices

:::
