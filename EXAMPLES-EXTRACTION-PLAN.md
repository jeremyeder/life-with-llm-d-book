# Example Extraction Plan: llm-d-book-examples Repository

## Overview
Extract 43,182 words of code examples from the book to a separate repository, reducing book size by ~50% while improving maintainability and usability.

**Target Repository**: `github.com/jeremyeder/llm-d-book-examples`

---

## Repository Structure

```
llm-d-book-examples/
├── README.md                    # Overview and navigation guide
├── LICENSE                      # Matching book license
├── .github/
│   └── workflows/
│       ├── validate-yaml.yml    # YAML syntax validation
│       └── test-scripts.yml     # Script testing
├── chapter-02-installation/
│   ├── README.md               # Chapter context and examples list
│   ├── basic-install.yaml      # Basic installation config
│   ├── gpu-node-setup.yaml     # GPU node configuration
│   └── scripts/
│       └── verify-install.sh   # Installation verification
├── chapter-03-architecture/
│   ├── README.md
│   ├── crd-examples/           # CRD examples
│   └── diagrams/               # Architecture diagrams
├── chapter-04-data-scientist/
│   ├── README.md
│   ├── model-deployment/       # Model deployment YAMLs
│   ├── notebooks/              # Jupyter notebooks
│   └── scripts/                # Data science scripts
├── chapter-05-sre-operations/
│   ├── README.md
│   ├── monitoring/             # Prometheus/Grafana configs
│   ├── alerts/                 # Alert rules
│   └── runbooks/              # Operational runbooks
├── chapter-06-performance/
│   ├── README.md
│   ├── benchmarks/            # Benchmark scripts
│   ├── rdma-configs/          # RDMA configurations
│   └── optimization/          # Performance tuning configs
├── chapter-07-security/
│   ├── README.md
│   ├── rbac/                  # RBAC configurations
│   ├── policies/              # Security policies
│   └── compliance/            # Compliance templates
├── chapter-08-troubleshooting/
│   ├── README.md
│   ├── diagnostic-scripts/    # Diagnostic tools
│   ├── decision-trees/        # Troubleshooting flowcharts
│   └── common-fixes/          # Common issue resolutions
├── chapter-10-mlops/
│   ├── README.md
│   ├── pipelines/             # CI/CD pipeline definitions
│   ├── tekton/                # Tekton pipelines
│   ├── argo/                  # Argo workflows
│   └── testing/               # Test frameworks
├── chapter-11-cost/
│   ├── README.md
│   ├── calculators/           # Cost calculation scripts
│   └── optimization/          # Cost optimization configs
└── reference/
    ├── commands.md            # Complete command reference
    ├── configurations/        # Template library
    └── scripts/              # Utility scripts
```

---

## Extraction Rules

### Keep Inline (in book)
1. **Commands** (1-3 lines)
   ```bash
   kubectl get pods -n llm-d-system
   ```

2. **Key Configuration Snippets** (<10 lines)
   ```yaml
   spec:
     model: llama-3.1-8b
     replicas: 2
   ```

3. **Essential Error Messages**
   ```
   Error: GPU memory insufficient
   ```

4. **Quick Examples** that illustrate concepts

5. **Mermaid Diagrams** - Visual explanations that need context
   ```mermaid
   graph LR
     A[User] --> B[API Gateway]
     B --> C[Model Server]
   ```

6. **Key Benchmark Results** - Summary tables and performance graphs
   - Performance comparison charts
   - Critical benchmark images/graphs
   - Summary tables (tokens/sec, latency)
   - "Before/After" optimization results

7. **Inline Code** demonstrating key concepts

### Move to External Repo
1. **Complete YAML Files** (>10 lines)
2. **Full Scripts** (Python, Bash)
3. **Multi-file Examples**
4. **Extensive Configuration Templates**
5. **Raw Benchmark Data** and detailed logs
6. **Benchmark Scripts** and testing harnesses
7. **Complete Troubleshooting Procedures**
8. **Long Code Blocks** (>20 lines)
9. **Sample Data Files**
10. **Complete Working Examples**
11. **Detailed Performance Tuning Scripts**

---

## Migration Strategy

### Phase 1: Repository Setup (Day 1)
1. Create `llm-d-book-examples` repository
2. Set up directory structure
3. Create README with navigation
4. Set up CI/CD for validation

### Phase 2: Example Extraction (Days 2-5)
**Priority Order** (by impact):

1. **Chapter 10: MLOps** (15,874 words of code)
   - Extract complete pipeline definitions
   - Move Tekton/Argo workflow examples
   - Create working CI/CD templates

2. **Chapter 8: Troubleshooting** (11,292 words of code)
   - Extract diagnostic scripts
   - Move complete runbooks
   - Create executable troubleshooting tools

3. **Chapter 6: Performance** (8,350 words of code)
   - Extract benchmark scripts
   - Move RDMA configurations
   - Create performance testing suite

4. **Chapter 5: SRE Operations** (3,105 words of code)
   - Extract monitoring configs
   - Move alert definitions
   - Create operational templates

### Phase 3: Book Updates (Days 6-7)
1. Replace extracted code with references:
   ```markdown
   For the complete configuration, see:
   [Model Deployment Example](https://github.com/jeremyeder/llm-d-book-examples/tree/main/chapter-04-data-scientist/model-deployment/basic-deployment.yaml)
   ```

2. Add "Example Repository" section to each chapter
3. Update README with examples repo link

---

## Example Reference Format

### Before (in book):
```yaml
# 50+ lines of YAML configuration
apiVersion: llm.d.ai/v1alpha1
kind: Model
metadata:
  name: llama-3-1-8b
  namespace: production
spec:
  model: llama-3.1-8b
  # ... extensive configuration ...
```

### After (in book):
```markdown
Deploy the model using our production-ready configuration:

```yaml
# Basic structure
apiVersion: llm.d.ai/v1alpha1
kind: Model
metadata:
  name: llama-3-1-8b
spec:
  model: llama-3.1-8b
  replicas: 2
```

📎 **Full Example**: [production-deployment.yaml](https://github.com/jeremyeder/llm-d-book-examples/tree/main/chapter-04-data-scientist/model-deployment/production-deployment.yaml)

Key configuration points:
- GPU resource allocation
- Health check configuration
- Autoscaling parameters
```

### Link Formats to Use

1. **Inline Reference**:
   ```markdown
   For complete configuration, see the [production deployment example](https://github.com/jeremyeder/llm-d-book-examples/tree/main/chapter-04-data-scientist/model-deployment/production-deployment.yaml).
   ```

2. **Callout Box**:
   ```markdown
   > 📦 **Example Files**: Find all monitoring configurations in the [examples repository](https://github.com/jeremyeder/llm-d-book-examples/tree/main/chapter-05-sre-operations/monitoring/)
   ```

3. **Chapter Footer**:
   ```markdown
   ## Example Code
   
   All examples from this chapter are available in the [llm-d-book-examples repository](https://github.com/jeremyeder/llm-d-book-examples/tree/main/chapter-05-sre-operations/):
   - [Monitoring Setup](./monitoring/)
   - [Alert Configurations](./alerts/)
   - [Operational Runbooks](./runbooks/)
   ```

4. **Try It Yourself Sections**:
   ```markdown
   ### Try It Yourself
   
   1. Clone the examples:
      ```bash
      git clone https://github.com/jeremyeder/llm-d-book-examples
      cd llm-d-book-examples/chapter-10-mlops/pipelines
      ```
   
   2. Deploy the CI/CD pipeline:
      ```bash
      kubectl apply -f tekton-pipeline.yaml
      ```
   ```

---

## Benefits Analysis

### Immediate Benefits
1. **Book Size Reduction**: ~43,000 words (55% reduction)
2. **Improved Readability**: Focus on concepts, not implementation
3. **Live Examples**: Users can clone and run immediately
4. **Version Control**: Examples can evolve independently

### Long-term Benefits
1. **Community Contributions**: Users can submit example improvements
2. **Testing**: Examples can be automatically tested
3. **Multiple Versions**: Support different llm-d versions
4. **Real-world Examples**: Accumulate production patterns

---

## Success Metrics

### Quantitative
- [ ] Book reduced by 40,000+ words
- [ ] All examples executable
- [ ] CI/CD validates all YAML
- [ ] Scripts have tests

### Qualitative
- [ ] Improved book flow and readability
- [ ] Examples are self-contained and runnable
- [ ] Clear navigation between book and examples
- [ ] Community engagement with examples repo

---

## Implementation Checklist

### Week 1
- [ ] Create llm-d-book-examples repository
- [ ] Set up directory structure
- [ ] Extract Chapter 10 (MLOps) examples
- [ ] Extract Chapter 8 (Troubleshooting) examples
- [ ] Update book chapters with references

### Week 2
- [ ] Extract remaining chapter examples
- [ ] Set up CI/CD validation
- [ ] Create navigation documentation
- [ ] Test all example links
- [ ] Final book word count validation

---

## Example README for External Repo

```markdown
# llm-d Book Examples

Complete, runnable examples from "Life with llm-d" book.

## Quick Start

```bash
git clone https://github.com/jeremyeder/llm-d-book-examples
cd llm-d-book-examples/chapter-04-data-scientist/model-deployment
kubectl apply -f basic-deployment.yaml
```

## Navigation

| Chapter | Topic | Examples |
|---------|-------|----------|
| [Chapter 2](./chapter-02-installation) | Installation | Basic setup, GPU nodes |
| [Chapter 4](./chapter-04-data-scientist) | Data Science | Model deployment, notebooks |
| [Chapter 5](./chapter-05-sre-operations) | SRE Ops | Monitoring, alerts, runbooks |
| [Chapter 8](./chapter-08-troubleshooting) | Troubleshooting | Diagnostic scripts, fixes |
| [Chapter 10](./chapter-10-mlops) | MLOps | CI/CD pipelines, automation |

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](./CONTRIBUTING.md).
```

---

**Next Step**: Create the external repository and begin extraction with highest-impact chapters (10, 8, 6).