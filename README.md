# Life with llm-d Book

A comprehensive guide to deploying, operating, and optimizing Large Language Model workloads using llm-d on Kubernetes and OpenShift.

## 📚 Book Structure

### Core Content
- **12 Chapters**: From installation through advanced production operations
- **3 Appendices**: Practical reference materials for daily operations
- **Complete Coverage**: Installation → Development → Production → Optimization

### Target Audiences
- **Data Scientists**: Model deployment and experimentation workflows
- **Site Reliability Engineers**: Production operations and incident response  
- **Platform Engineers**: Infrastructure design and optimization

## 🏗️ Repository Structure

```
life-with-llm-d-book/
├── docs/                           # Book content (Docusaurus)
│   ├── 00-forward.md              # Book introduction
│   ├── 01-introduction.md         # Chapter 1: Introduction
│   ├── 02-installation-setup.md   # Chapter 2: Installation and Setup
│   ├── ...                       # Chapters 3-12
│   ├── 98-index-glossary.md       # Index and glossary
│   ├── 99-table-of-contents.md    # Table of contents
│   └── appendix/                  # Reference appendices
│       ├── crd-reference.md       # Appendix A: CRD Reference
│       ├── command-reference.md   # Appendix B: Command Reference
│       ├── configuration-templates.md # Appendix C: Configuration Templates
│       └── shared-config.md       # Shared standards and conventions
├── scripts/                       # Validation and quality assurance
├── examples/                      # Code samples and configurations
├── templates/                     # Reusable templates
└── static/                        # Images and assets
```

## 📖 Chapter Overview

### Getting Started
1. **Introduction** - Project overview and value proposition
2. **Installation and Setup** - Getting started with llm-d
3. **Understanding the Architecture** - Core concepts and components

### User Workflows  
4. **Data Scientist Workflows** - Model deployment and experimentation
5. **SRE Operations** - Infrastructure management and monitoring

### Advanced Operations
6. **Performance Optimization** - Hardware, networking, and efficiency
7. **Security and Compliance** - RBAC, Pod Security, compliance frameworks
8. **Troubleshooting Guide** - Decision trees and systematic problem-solving

### Production Excellence
9. **Advanced Topics** - Scaling, high availability, and edge cases
10. **MLOps Workflows** - Model lifecycle and automation
11. **Cost Optimization** - Quantization, scheduling, and financial optimization
12. **MLOps for SREs** - Production operations and incident response

### Reference Materials
- **Appendix A**: Complete API reference for llm-d CRDs
- **Appendix B**: Comprehensive kubectl command workflows  
- **Appendix C**: Ready-to-deploy YAML and Helm templates

## 🛠️ Development

### Local Development
```bash
# Install dependencies
npm install

# Install pre-commit hooks (required for all contributors)
pip install pre-commit
pre-commit install

# Start development server
npm start

# Build for production
npm run build

# Run validation tests
make validate-chapter docs/[chapter-name].md
```

### Quality Standards
- **Pre-commit Hooks**: Automated validation before every commit (prevents CI failures)
- **Validation Scripts**: Automated checks for naming conventions, resource specs, and consistency
- **Spell Checking**: Technical term validation with custom dictionary
- **Cross-References**: Verified links between chapters and appendices
- **Code Examples**: All YAML and commands tested against standards

### Content Standards
- Consistent naming conventions throughout (e.g., `llama-3.1-8b`, not abbreviated forms)
- Standard namespaces: `production`, `staging`, `development`, `llm-d-system`
- Real-world examples with practical guidance
- Cross-references connecting related concepts across chapters

## 📊 Book Statistics

- **Total Content**: 12 chapters + 3 appendices + reference materials
- **Cross-References**: 28+ chapter references across appendices
- **Code Examples**: 100+ YAML configurations and kubectl commands
- **Coverage**: Complete llm-d operational lifecycle

## 🚀 Getting Started

1. **Read the Forward**: Overview and audience guidance
2. **Follow Installation**: Chapters 2-3 for setup
3. **Choose Your Path**: 
   - Data Scientists → Chapters 4, 10 + Appendix A
   - SREs → Chapters 5, 8, 12 + Appendices B-C
   - Platform Engineers → All chapters with focus on 6-9, 11

## 🤝 Contributing

For contribution guidelines and next steps, visit [llm-d.ai](https://llm-d.ai) contribution webpage.

## 📄 License

This project follows open source principles with proper attribution for all sources and references documented for publication compliance.

---

**Status**: Core content complete with comprehensive appendices  
**Next Phase**: Review, polish, and production readiness  
**Target**: Professional technical book publication

*Project initiated: 2025-06-15*  
*Book completed: 2025-06-16*