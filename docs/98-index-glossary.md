---
title: Index & Glossary
description: Comprehensive index and glossary of terms for Life with llm-d
sidebar_position: 98
---

# Index & Glossary

## Key Terms and Concepts

### A

- **Autoscaling**: Automatic scaling of LLM deployments based on metrics like GPU utilization
- **API Gateway**: Service routing and load balancing for model inference endpoints

### B

- **Blue-Green Deployment**: Strategy for zero-downtime model updates using parallel environments

### C

- **CRD (Custom Resource Definition)**: Kubernetes extension defining llm-d resources
- **Cost Optimization**: Strategies to reduce LLM deployment expenses through quantization, scheduling, and resource management

### D

- **Data Scientist Workflows**: Development and deployment patterns for ML practitioners
- **Disaggregation**: Separating prefill and decode phases for improved efficiency

### F

- **FP16**: 16-bit floating point quantization format

### G

- **GPU Utilization**: Measurement of graphics processing unit usage efficiency

### H

- **HPA (Horizontal Pod Autoscaler)**: Kubernetes component for automatic pod scaling
- **Helm**: Package manager for Kubernetes applications

### I

- **InferenceScheduler**: llm-d CRD for SLO-driven optimization and cost management
- **INT4/INT8**: Integer quantization formats for reduced memory usage

### K

- **kubectl**: Command-line tool for interacting with Kubernetes clusters

### L

- **LLMDeployment**: Primary llm-d CRD for deploying LLM inference services
- **llm-d**: LLM deployment operator for Kubernetes

### M

- **ModelService**: Service definition for model inference endpoints with load balancing
- **MLOps**: Machine Learning Operations practices and workflows

### N

- **Namespace**: Kubernetes mechanism for resource isolation and organization
- **Network Policy**: Kubernetes resource for controlling network traffic

### P

- **Pod Security Context**: Security configuration for Kubernetes pods
- **Prefill/Decode**: Phases of LLM inference processing

### Q

- **Quantization**: Technique to reduce model memory requirements and inference costs

### R

- **RBAC (Role-Based Access Control)**: Kubernetes security model for authorization
- **ResourceProfile**: llm-d resource allocation profiles for standardized deployments
- **Rollback**: Process of reverting to a previous model version

### S

- **SLO (Service Level Objective)**: Target metrics for service performance
- **SRE (Site Reliability Engineering)**: Practices for maintaining reliable services
- **Spot Instances**: Cost-effective cloud compute with potential interruption

### T

- **Tensor Parallel**: Distribution of model computation across multiple GPUs
- **Tolerations**: Kubernetes mechanism allowing pods to schedule on specific nodes

## Command Index

### Deployment Commands

- `kubectl apply`: Deploy llm-d resources
- `kubectl get llmdeployments`: List LLM deployments
- `kubectl describe llmdeployment`: Get detailed deployment information

### Monitoring Commands

- `kubectl top nodes`: View node resource usage
- `kubectl top pods`: View pod resource consumption
- `kubectl logs`: Access container logs

### Troubleshooting Commands

- `kubectl describe`: Get detailed resource information
- `kubectl get events`: View cluster events
- `kubectl rollout status`: Check deployment status

### Scaling Commands

- `kubectl scale`: Manually scale deployments
- `kubectl get hpa`: View autoscaler status
- `kubectl patch`: Update resource configurations

## Configuration Index

### CRD Resources

- **LLMDeployment**: Primary deployment resource
- **InferenceScheduler**: SLO-driven scheduler
- **ModelService**: Load balancing service
- **ResourceProfile**: Standardized resource allocation

### Environment Types

- **Development**: Cost-optimized testing environments
- **Staging**: Production-like pre-deployment testing
- **Production**: Full-scale production deployments
- **Multi-tenant**: Shared infrastructure with isolation

### Quantization Options

- **FP16**: Standard 16-bit floating point
- **INT8**: 8-bit integer (50% memory reduction)
- **INT4**: 4-bit integer (75% memory reduction)

## Chapter Cross-Reference

### Installation and Setup

- Chapters 2-3: Initial setup and architecture
- Appendix C: Configuration templates

### Model Deployment

- Chapter 4: Data scientist workflows
- Chapter 10: MLOps workflows
- Appendix A: CRD reference

### Operations and Monitoring

- Chapter 5: SRE operations
- Chapter 12: MLOps for SREs
- Appendix B: Command reference

### Performance and Cost

- Chapter 6: Performance optimization
- Chapter 11: Cost optimization

### Security and Compliance

- Chapter 7: Security and compliance
- Appendix A: Security configurations

### Troubleshooting

- Chapter 8: Troubleshooting guide
- Appendix B: Diagnostic commands

## Acronyms and Abbreviations

- **API**: Application Programming Interface
- **CD**: Continuous Deployment
- **CI**: Continuous Integration
- **CPU**: Central Processing Unit
- **CRD**: Custom Resource Definition
- **GPU**: Graphics Processing Unit
- **HPA**: Horizontal Pod Autoscaler
- **HTTP**: Hypertext Transfer Protocol
- **JSON**: JavaScript Object Notation
- **LLM**: Large Language Model
- **MLOps**: Machine Learning Operations
- **RBAC**: Role-Based Access Control
- **SLA**: Service Level Agreement
- **SLO**: Service Level Objective
- **SRE**: Site Reliability Engineering
- **TLS**: Transport Layer Security
- **YAML**: Yet Another Markup Language

---

**Note**: This index provides quick reference to key concepts, commands, and configurations throughout the book. For detailed explanations, refer to the specific chapters and appendices.
