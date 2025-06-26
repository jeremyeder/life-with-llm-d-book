---
title: Introduction to llm-d
description: Learn about llm-d, the open-source platform for deploying large language models on Kubernetes and OpenShift
sidebar_position: 1
---

# Introduction to llm-d

:::info Chapter Overview
This chapter introduces llm-d, the open-source platform for deploying large language models on Kubernetes and OpenShift. You'll learn the core concepts, understand why llm-d exists, and deploy your first LLM using the platform.
:::

## Prerequisites

- [ ] Basic understanding of Kubernetes concepts (pods, deployments, services)
- [ ] Access to a Kubernetes or OpenShift cluster
- [ ] kubectl CLI installed and configured
- [ ] GPU-enabled nodes in your cluster

:::warning Important
This book assumes familiarity with Kubernetes fundamentals. If you're new to Kubernetes, review the official Kubernetes documentation before proceeding.
:::

## What is llm-d?

llm-d (Large Language Model Deployment) is an open-source platform designed specifically for deploying, scaling, and managing large language models on Kubernetes and OpenShift. llm-d serves as the bridge between your model development environment and production-grade infrastructure.

### Why llm-d Matters

Traditional deployment tools weren't built with LLMs in mind. llm-d addresses unique challenges:

- **Model Size**: LLMs can be hundreds of gigabytes
- **GPU Management**: Efficient scheduling and sharing of expensive GPU resources
- **Memory Optimization**: Techniques like KV-cache management
- **Inference Optimization**: Support for various backends like vLLM
- **Operational Excellence**: Built-in monitoring, scaling, and reliability

### Strategic Value Proposition

**Open Source Advantage**: Unlike proprietary cloud providers, llm-d gives you complete control over your AI infrastructure. You're not locked into a single vendor's ecosystem or pricing model.

**Enterprise-Grade Reliability**: Built on proven enterprise Kubernetes platforms, llm-d delivers the "boring reliability" that enterprises require for mission-critical AI workloads.

**Total Cost of Ownership (TCO) Benefits**:

| Approach | Initial Setup | Annual Costs | Vendor Lock-in | Customization |
|----------|--------------|--------------|----------------|---------------|
| **Managed Cloud AI Services** | Low | High ($$$$) | High | Limited |
| **Cloud Provider APIs** | Low | High ($$$$) | High | Limited |
| **Proprietary Platforms** | Medium | Very High ($$$$$) | High | Limited |
| **llm-d on Kubernetes** | Medium | Lower ($$) | None | Full |

**Risk Mitigation**: 
- **No vendor lock-in**: Move between cloud providers or on-premises freely
- **Regulatory compliance**: Keep sensitive data in your own infrastructure
- **Open source transparency**: No black box dependencies
- **Community-driven innovation**: Benefit from ecosystem contributions

## Your First llm-d Deployment

Before deploying any LLM workload, verify your cluster's GPU availability:

```bash
# Check cluster nodes
kubectl get nodes
kubectl describe node | grep -E "nvidia.com/gpu|Capacity:" -A 5
```

:::tip
Ensure GPU availability before deploying LLM workloads. LLMs require significant computational resources.
:::

### Creating a Simple Deployment

Start with a minimal deployment to understand the basics:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: llm-experiments
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-first-llm
  namespace: llm-experiments
spec:
  replicas: 1
  selector:
    matchLabels:
      app: my-first-llm
  template:
    metadata:
      labels:
        app: my-first-llm
    spec:
      containers:
      - name: llm-server
        image: vllm/vllm-openai:latest
        args:
          - "--model"
          - "facebook/opt-125m"  # Starting with a small model
          - "--dtype"
          - "float16"
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
        ports:
        - containerPort: 8000
          name: http
```

### Deployment Instructions

```bash
# Deploy our first LLM
kubectl apply -f my-first-llm.yaml

# Watch the deployment
kubectl get pods -n llm-experiments -w

# Check the logs
kubectl logs -n llm-experiments -l app=my-first-llm -f
```

## Verifying Your Deployment

Verify the deployment is functioning correctly:

```bash
# Port-forward to access the model
kubectl port-forward -n llm-experiments svc/my-first-llm 8000:8000 &

# Test the model
curl http://localhost:8000/v1/models
```

Expected output:

```json
{
  "data": [
    {
      "id": "facebook/opt-125m",
      "object": "model",
      "owned_by": "vllm",
      "permission": []
    }
  ]
}
```

## Common First-Timer Mistakes

:::note Troubleshooting
Common issues when getting started with llm-d deployments:
:::

| Issue | Cause | Solution |
|-------|-------|----------|
| Pod stuck in Pending | No GPU nodes available | Check node labels and GPU operator |
| OOMKilled | Model too large for memory | Start with smaller model or increase limits |
| ImagePullBackOff | Wrong image or registry | Verify image name and pull secrets |

## The Architecture Ahead

llm-d provides a comprehensive platform with advanced features:

- **Custom Resource Definitions (CRDs)** for model management
- **Intelligent scheduling** for GPU utilization
- **Model registry integration** for versioning
- **Autoscaling** based on inference load
- **Multi-model serving** on single GPU
- **Observability** with Prometheus and Grafana

Each of these components will be explored in detail throughout this book.

## Best Practices from Day One

- ✅ **Do**: Start with small models to learn the platform
- ✅ **Do**: Monitor GPU memory usage from the beginning
- ✅ **Do**: Use namespaces to organize your experiments
- ❌ **Don't**: Deploy large models without understanding resource requirements
- ❌ **Don't**: Forget to set resource limits
- 💡 **Consider**: Cost implications of GPU usage

## Summary

- llm-d simplifies LLM deployment on Kubernetes while delivering strategic competitive advantages
- **Strategic Benefits**: Significant cost reduction vs cloud providers, zero vendor lock-in, full customization control
- **Technical Benefits**: Start small and gradually increase complexity, efficient GPU resource management
- **Operational Benefits**: Enterprise-grade reliability with open source transparency
- **Success Path**: Production deployment requires careful planning and execution, but the ROI justifies the investment

## Next Steps

Chapter 2 covers the complete llm-d installation process using llm-d-deployer and explores various deployment options.

---

:::info References

- [llm-d GitHub Organization](https://github.com/llm-d)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Kubernetes GPU Support](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)

:::
