---
title: Troubleshooting Guide
description: Comprehensive troubleshooting guide for llm-d deployments
sidebar_position: 8
---

# Troubleshooting Guide

This chapter provides a comprehensive guide to troubleshooting llm-d deployments, including decision trees, diagnostic procedures, and solutions to common issues.

## Overview

Operating LLM workloads at scale presents unique challenges that require systematic approaches to problem-solving. This guide equips you with:

- **Decision trees** for rapid problem identification
- **Diagnostic tools** and procedures
- **Common issue patterns** and their solutions
- **Performance troubleshooting** techniques
- **Emergency response** procedures

### Using This Guide

This guide is organized by symptom rather than cause, allowing you to quickly find relevant troubleshooting procedures:

1. **Start with symptoms** - What are you observing?
2. **Follow decision trees** - Navigate to the root cause
3. **Apply diagnostics** - Gather detailed information
4. **Implement solutions** - Follow tested procedures
5. **Verify resolution** - Confirm the fix worked

### Key Principles

When troubleshooting llm-d deployments:

- **Document everything** - Keep detailed notes of symptoms and actions
- **Change one thing at a time** - Isolate variables for accurate diagnosis
- **Use the scientific method** - Form hypotheses and test them
- **Check the basics first** - Often the simplest explanation is correct
- **Preserve evidence** - Collect logs before making changes

## Quick Reference

### Emergency Contacts

```yaml
# Critical Issues
on-call-sre: "+1-xxx-xxx-xxxx"
escalation-manager: "escalate@company.com"

# Support Channels
slack: "#llm-d-support"
pagerduty: "llm-d-incidents"
```

### Common Commands

```bash
# Check llm-d status
kubectl get llmdeployments -A
kubectl describe llmdeployment <name> -n <namespace>

# View logs
kubectl logs -n llm-d-system deployment/llm-d-operator
kubectl logs -n <namespace> <pod-name> -c inference-server

# Resource status
kubectl top nodes
kubectl top pods -n <namespace>

# GPU status
nvidia-smi
kubectl describe nodes | grep -A5 "nvidia.com/gpu"
```

### Health Check URLs

```bash
# Operator health
curl http://llm-d-operator.llm-d-system:8080/healthz

# Model endpoint health
curl http://<model-service>.<namespace>:8080/health

# Metrics endpoint
curl http://llm-d-operator.llm-d-system:8080/metrics
```

## Chapter Contents

```mdx-code-block
import DocCardList from '@theme/DocCardList';

<DocCardList />
```
