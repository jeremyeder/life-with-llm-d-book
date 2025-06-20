# Chapter 10: MLOps Workflows Examples

Complete examples for implementing MLOps workflows with llm-d.

## Directory Structure

```
chapter-10-mlops/
├── github-actions/      # GitHub Actions workflows
├── pipelines/          # Kubeflow pipeline definitions
├── tekton/             # Tekton pipeline examples
├── argo/               # Argo Workflows
└── testing/            # Testing frameworks and examples
```

## Quick Start

### GitHub Actions CI/CD

```bash
# Copy workflow to your repository
cp github-actions/model-validation.yml .github/workflows/

# Trigger pipeline on push
git push origin main
```

### Kubeflow Pipelines

```bash
# Deploy pipeline
python pipelines/model-registration.py

# Submit pipeline run
kfp run submit -e experiment-name -r run-name -p pipeline.yaml
```

## Examples Overview

### CI/CD Pipelines
- [Model Validation Workflow](./github-actions/model-validation.yml)
- [Staging Deployment](./github-actions/deploy-staging.yml)
- [Production Deployment](./github-actions/deploy-production.yml)

### Pipeline Components
- [Model Registration](./pipelines/model-registration.py)
- [Deployment Pipeline](./pipelines/deployment.py)
- [Validation Pipeline](./pipelines/validation.py)

### Testing Frameworks
- [Unit Tests](./testing/unit/)
- [Integration Tests](./testing/integration/)
- [Performance Tests](./testing/performance/)

## Related Book Sections

These examples support Chapter 10 of "Life with llm-d":
- Section 10.2: CI/CD Pipeline Setup
- Section 10.3: Automated Testing
- Section 10.4: Deployment Automation