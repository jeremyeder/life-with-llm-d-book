# Code Example Template v1

Use this template for creating consistent code examples across chapters.

## Prompt Template

```
Create a Python code example that demonstrates [concept] following these requirements:

1. **Follow Existing Patterns**: Use utilities and patterns from examples/shared-utilities if applicable
2. **Comprehensive Testing**: Include corresponding test in tests/ directory following project structure
3. **Error Handling**: Include appropriate exception handling and validation
4. **Documentation**: Add docstrings and inline comments for complex logic
5. **Realistic Context**: Use real-world scenarios relevant to llm-d deployment

Code Standards:
- Python 3.13 compatibility (versions N and N-1)
- Use virtual environments and uv package manager
- Follow existing naming conventions from codebase
- Include imports for all dependencies
- Add type hints where beneficial

Structure:
- Main implementation in examples/chapter-[X]/[descriptive-name].py
- Corresponding test in tests/examples/chapter-[X]/test_[descriptive-name].py
- Use shared fixtures from tests/fixtures/ when applicable
- Include pytest markers: unit, integration, gpu, etc.

Template Variables:
- [concept] = The technical concept being demonstrated
- [X] = Chapter number
- [descriptive-name] = Clear, hyphenated filename
```

## Example Usage

For Chapter 4 LLM client example:
- File: `examples/chapter-04-data-scientist/llm_client.py`
- Test: `tests/examples/chapter-04-data-scientist/test_llm_client.py`
- Template version: v1

## Usage Notes

- Ensure code examples are executable and tested
- Mock external dependencies (GPUs, Kubernetes APIs)
- Follow shared configuration patterns from docs/appendix/shared-config.md