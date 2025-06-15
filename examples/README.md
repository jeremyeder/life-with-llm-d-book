# Life with llm-d - Code Examples

This directory contains all code examples from the book, organized by chapter. Each example is a complete, working file that can be used directly with llm-d.

## Directory Structure

```
examples/
├── chapter-02-installation/    # Installation and setup scripts
├── chapter-03-architecture/    # Architecture configurations
├── chapter-04-data-scientist/  # Data scientist workflows and notebooks
├── chapter-05-sre-operations/  # SRE runbooks and monitoring configs
└── README.md                   # This file
```

## Usage

All examples are designed to work with the llm-d platform. To use an example:

1. Navigate to the relevant chapter directory
2. Copy or download the file you need
3. Modify any environment-specific values (namespaces, URLs, etc.)
4. Apply using kubectl, helm, or the appropriate tool

## Testing

These examples are automatically tested against the upstream llm-d codebase. See `.github/workflows/test-examples.yml` for the CI configuration.

## Contributing

When adding new examples:
1. Place files in the appropriate chapter directory
2. Use descriptive filenames
3. Include comments explaining the purpose
4. Ensure the example is complete and functional