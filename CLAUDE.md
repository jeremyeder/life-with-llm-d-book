# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive technical book project "Life with llm-d" built with Docusaurus. It combines documentation, executable code examples, comprehensive testing, and validation frameworks for Large Language Model deployment on Kubernetes.

## Development Commands

### Docusaurus (Primary Development)
```bash
# Start development server
npm start

# Build production site
npm run build

# Serve built site locally
npm run serve

# Clear Docusaurus cache
npm run clear
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file (preferred for performance)
pytest tests/examples/chapter-04-data-scientist/test_llm_client.py

# Run tests with coverage
pytest --cov=examples --cov=docs/cost-optimization

# Run only unit tests
pytest -m unit

# Run GPU-related tests (mocked)
pytest -m gpu
```

### Content Validation
```bash
# Run comprehensive validation
./scripts/run-all-validations.sh

# Validate specific chapter
make validate-chapter docs/11-cost-optimization.md

# Run individual validations
./scripts/validate-model-names.sh
./scripts/validate-namespaces.sh
./scripts/validate-resource-specs.sh

# Lint markdown
npm run lint
npm run lint:fix

# Spell check
npm run spell
```

### Cost Optimization Workflow
```bash
# Update pricing and validate costs
make all

# Generate cost forecasts
make generate-forecasts

# Run cost calculation tests
make test-costs

# Generate cost comparison report
make cost-report
```

## Architecture Overview

### Multi-Language Structure
- **Frontend**: Docusaurus (React/JavaScript) for book website
- **Examples**: Python scripts with comprehensive testing
- **Configs**: YAML configurations for Kubernetes/OpenShift
- **Scripts**: Bash validation and automation tools

### Key Directories
- `docs/` - Book chapters and content (Markdown)
- `examples/` - Executable code examples organized by chapter
- `tests/` - Comprehensive test suite mirroring examples structure
- `scripts/` - Validation and automation scripts
- `src/` - Docusaurus React components and theme customizations

### Testing Architecture
- **pytest** with coverage reporting and HTML output
- **Test Structure**: Mirrors examples/ directory structure exactly
- **Mocking**: GPU operations, Kubernetes APIs, cost calculations
- **Fixtures**: Shared test data in tests/fixtures/
- **Markers**: unit, integration, gpu, slow, k8s for test categorization

## Content Standards

### Model Naming (from docs/appendix/shared-config.md)
- Use exact names: `llama-3.1-8b`, `llama-3.1-70b`, `mistral-7b`
- Never abbreviate: avoid `llama-8b` or `llama-70b`

### Namespace Conventions
- `production` - Production workloads
- `staging` - Staging environment
- `development` - Development and testing
- `llm-d-system` - System components

### Resource Specifications
- Follow templates in docs/appendix/shared-config.md
- Small models (7B-8B): 16-24Gi memory, 1 GPU
- Large models (70B+): 160-200Gi memory, 4 GPUs

## Quality Assurance Framework

### Automated Validation
- **Model Names**: Consistency across all chapters
- **Namespaces**: Standard namespace usage
- **Resource Specs**: Adherence to shared templates
- **Cross-References**: Links between chapters and appendices
- **Shared Config**: References to standardized configurations

### Content Validation Flow
1. Run `./scripts/run-all-validations.sh` before commits
2. Individual validation scripts provide specific feedback
3. Markdown linting ensures consistent formatting
4. Spell checking with technical term dictionary

### Cost Calculation Framework
- **Dynamic Pricing**: Scripts update GPU pricing data
- **Validation**: All cost calculations verified against current market
- **Testing**: Comprehensive test coverage for cost prediction logic
- **Forecasting**: Multi-month cost projection capabilities

## Development Workflow

### Content Development
1. Write/edit markdown in docs/
2. Add corresponding code examples in examples/
3. Create tests in tests/ following same structure
4. Run validation scripts before commit
5. Test single files for performance: `pytest tests/specific_test.py`

### Code Example Development
1. Follow existing patterns in examples/chapter-XX/
2. Add comprehensive tests with realistic mocking
3. Include fixtures for complex test data
4. Use appropriate pytest markers (unit, integration, gpu, etc.)
5. Ensure 100% test coverage for new utilities

### Quality Checks
- All validation scripts must pass
- Markdown linting must be clean
- Test coverage maintained
- Cost calculations must validate against current pricing
- Model names and namespaces must follow standards

## Project-Specific Considerations

### Book Structure
- 12 chapters + 3 appendices covering complete llm-d lifecycle
- Target audiences: Data Scientists, SREs, Platform Engineers
- Cross-references link related concepts across chapters

### Technical Validation
- Executable code examples with comprehensive test coverage
- Real-world configurations validated against best practices
- Cost optimization with dynamic pricing integration
- Standards enforcement through automated validation

### Multi-Audience Content
- Chapter 4: Data Scientist workflows and experimentation
- Chapter 5: SRE operations and monitoring
- Chapter 8: Systematic troubleshooting procedures
- Chapter 10: MLOps and automation workflows

## Content Management Guidelines

### Version Control and Content Refresh
- We will attempt to refresh content monthly or as major decisions are reached
- MUST maintain comprehensive works cited for this book in the book appendix

## Code Generation Guidelines

### Validation Practices
- When referencing dates, always validate them. For example:
  - Code copyright should always be the current year
  - Release dates should match reality
  - If uncertain and unable to reliably find out, ask for clarification

## Development Best Practices

### Source Control
- When using git, ALWAYS work in feature branches unless told explicitly otherwise.