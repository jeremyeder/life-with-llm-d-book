# LLM-D Book Examples - Test Suite

This directory contains comprehensive tests for all Python examples in the LLM-D book.

## Overview

The test suite provides:
- Unit tests for all Python examples
- Integration tests for API interactions
- Mock implementations for external dependencies (Kubernetes, GPUs, LLM APIs)
- Code coverage tracking with a minimum threshold of 80%
- CI/CD integration via GitHub Actions

## Structure

```
tests/
├── conftest.py              # Global pytest configuration and fixtures
├── utils/                   # Test utilities
│   └── test_helpers.py      # Common test helpers and validators
├── fixtures/                # Mock data and responses
│   └── mock_responses.py    # Pre-defined mock API responses
├── examples/                # Tests for examples/ directory
│   ├── chapter-04-data-scientist/
│   ├── chapter-05-sre-operations/
│   ├── chapter-06-performance/
│   ├── chapter-08-troubleshooting/
│   └── chapter-10-mlops/
└── docs/                    # Tests for docs/ examples
    ├── cost-optimization/
    └── security-configs/
```

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install -r requirements-test.txt
```

### Run All Tests

```bash
# Run all tests with coverage
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/examples/chapter-04-data-scientist/test_llm_client.py

# Run tests matching a pattern
pytest -k "test_cost"
```

### Coverage Reports

```bash
# Generate coverage report in terminal
pytest --cov --cov-report=term-missing

# Generate HTML coverage report
pytest --cov --cov-report=html
# Open htmlcov/index.html in browser

# Check if coverage meets threshold
coverage report --fail-under=80
```

### Test Markers

Tests are marked for easy filtering:

```bash
# Run only unit tests (fast)
pytest -m unit

# Run integration tests
pytest -m integration

# Run tests requiring GPU mocking
pytest -m gpu

# Run tests requiring K8s API mocking
pytest -m k8s

# Skip slow tests
pytest -m "not slow"
```

## Writing Tests

### Example Test Structure

```python
import pytest
from unittest.mock import Mock, patch

class TestMyModule:
    @pytest.fixture
    def client(self):
        """Create test client instance."""
        return MyClient("test-endpoint")
    
    def test_basic_functionality(self, client):
        """Test basic client functionality."""
        response = client.do_something()
        assert response.status == "success"
    
    @pytest.mark.gpu
    def test_gpu_operations(self, mock_gpu_environment):
        """Test operations requiring GPU mocking."""
        # mock_gpu_environment fixture provides GPU env
        result = gpu_operation()
        assert result.device_count == 4
```

### Available Fixtures

Global fixtures in `conftest.py`:
- `temp_dir`: Temporary directory for test files
- `mock_kubernetes_client`: Mocked Kubernetes client
- `mock_gpu_environment`: Mocked GPU/CUDA environment
- `mock_llm_endpoint`: Mocked LLM service responses
- `sample_model_config`: Sample model configuration
- `sample_metrics_data`: Sample monitoring metrics
- `mock_prometheus_client`: Mocked Prometheus client
- `mock_mlflow_client`: Mocked MLflow tracking

### Mock Helpers

Use pre-defined mock responses from `fixtures/mock_responses.py`:
- `MockLLMResponses`: LLM API responses
- `MockKubernetesResponses`: K8s API responses
- `MockPrometheusResponses`: Prometheus query responses
- `MockMLFlowResponses`: MLflow tracking responses
- `MockCostServiceResponses`: Cost tracking responses

## CI/CD Integration

Tests run automatically on:
- Push to `main` or `develop` branches
- Pull requests to `main`
- When Python files or test configurations change

GitHub Actions workflow features:
- Multi-version Python testing (3.12, 3.13)
- Code coverage reporting to Codecov
- Coverage comments on PRs
- Security scanning with Bandit
- Linting with ruff
- Type checking with mypy

## Coverage Requirements

- Minimum coverage threshold: 80%
- Target coverage: 90%+
- Coverage tracked for:
  - `examples/`
  - `docs/cost-optimization/`
  - `docs/security-configs/`
  - `llm-d-book-examples/`

## Future Enhancements (Roadmap)

1. **Performance Testing Suite**
   - Benchmark tests for critical paths
   - Load testing for API endpoints
   - Memory profiling for optimization algorithms
   - GPU utilization benchmarks

2. **Complex Module Testing**
   - Advanced GPU optimizer testing with hardware simulation
   - ML pipeline testing with full workflow validation
   - Complex state machine testing for emergency procedures
   - End-to-end integration tests

3. **Additional Test Types**
   - Property-based testing with Hypothesis
   - Mutation testing
   - Fuzz testing for input validation
   - Contract testing for API compatibility

## Contributing

When adding new examples:
1. Write corresponding tests in the appropriate test directory
2. Ensure tests cover both success and error cases
3. Add appropriate test markers
4. Run coverage locally to ensure threshold is met
5. Update this README if adding new fixtures or patterns

## Troubleshooting

### Common Issues

**Import errors**: Ensure the examples directory is in Python path
- Tests automatically add required paths

**Mock not working**: Check fixture scope and usage
- Use `autouse=True` for automatic fixture application

**Coverage not tracked**: Verify file paths in `.coveragerc`
- Ensure new modules are included in coverage source

**Tests timing out**: Use `pytest-timeout` settings
- Default timeout is 300 seconds per test

### Debug Mode

```bash
# Run with debugging output
pytest -vv --tb=short

# Run with pdb on failure
pytest --pdb

# Run with specific log level
pytest --log-cli-level=DEBUG
```