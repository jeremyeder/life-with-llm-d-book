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
| [Chapter 3](./chapter-03-architecture) | Architecture | CRD examples, diagrams |
| [Chapter 4](./chapter-04-data-scientist) | Data Science | Model deployment, notebooks |
| [Chapter 5](./chapter-05-sre-operations) | SRE Ops | Monitoring, alerts, runbooks |
| [Chapter 6](./chapter-06-performance) | Performance | Benchmarks, RDMA, optimization |
| [Chapter 7](./chapter-07-security) | Security | RBAC, policies, compliance |
| [Chapter 8](./chapter-08-troubleshooting) | Troubleshooting | Diagnostic scripts, fixes |
| [Chapter 10](./chapter-10-mlops) | MLOps | CI/CD pipelines, automation |
| [Chapter 11](./chapter-11-cost) | Cost | Calculators, optimization configs |
| [Reference](./reference) | Reference | Commands, configurations, scripts |

## Repository Structure

```
llm-d-book-examples/
├── chapter-02-installation/     # Installation configs and scripts
├── chapter-03-architecture/     # Architecture examples
├── chapter-04-data-scientist/   # Data science workflows
├── chapter-05-sre-operations/   # SRE configurations
├── chapter-06-performance/      # Performance optimization
├── chapter-07-security/         # Security configurations  
├── chapter-08-troubleshooting/  # Troubleshooting tools
├── chapter-10-mlops/           # MLOps pipelines
├── chapter-11-cost/            # Cost optimization
└── reference/                  # Complete reference materials
```

## Contributing

We welcome contributions! Please:
1. Follow the existing directory structure
2. Include README.md in each example directory
3. Test all examples before submitting
4. Add comments to complex configurations

## License

Same as the main book repository.

## Support

For issues or questions:
- Book repository: https://github.com/jeremyeder/life-with-llm-d-book
- Examples repository: https://github.com/jeremyeder/llm-d-book-examples