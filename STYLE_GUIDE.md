# Documentation Style Guide

This guide ensures consistency across all chapters in the "Life with llm-d" book.

## Content Standards

### Chapter Structure

All chapters must follow this structure:

```markdown
---
title: [Descriptive Title]
description: [1-2 sentence description following pattern]
sidebar_position: [Number]
---

# [Chapter Title]

:::info Chapter Overview
This chapter focuses on [topic], covering [areas]. You'll learn [outcomes].
:::

## [Major Section 1]
### [Subsection]
## [Major Section 2]
## Best Practices
## Summary and Next Steps
---
:::info References
[Reference list with shared-config.md]
:::
```

### Model Naming Standards

**Always use these standard names from shared-config.md:**

- ✅ `llama-3.1-8b` or `Llama 3.1 8B`
- ✅ `llama-3.1-70b` or `Llama 3.1 70B`
- ✅ `mistral-7b` or `Mistral 7B`
- ✅ `codellama-13b` or `CodeLlama 13B`

**Never use:**

- ❌ `llama4`
- ❌ `Llama-3.1`
- ❌ `llama-8b` (without version)
- ❌ `llama-70b` (without version)

### Namespace Conventions

**Standard namespaces:**

- ✅ `llm-d-system` (for operator/system components only)
- ✅ `production` (for production workloads)
- ✅ `staging` (for staging/testing)
- ✅ `development` (for development/experimentation)

**Deprecated patterns:**

- ❌ `llm-d-production`
- ❌ `llm-d-staging`
- ❌ `llm-d-dev`

### Resource Specifications

**Small Models (8B):**

```yaml
resources:
  requests:
    memory: "16Gi"
    cpu: "4"
    nvidia.com/gpu: "1"
  limits:
    memory: "24Gi"
    cpu: "8"
    nvidia.com/gpu: "1"
```

**Large Models (70B):**

```yaml
resources:
  requests:
    memory: "160Gi"
    cpu: "16"
    nvidia.com/gpu: "4"
  limits:
    memory: "200Gi"
    cpu: "32"
    nvidia.com/gpu: "4"
```

## Writing Style

### Technical Writing Principles

1. **Clarity First**: Use simple, direct language
2. **Active Voice**: "Deploy the model" not "The model should be deployed"
3. **Imperative Mood**: Use commands ("Run kubectl apply") for instructions
4. **Consistent Terminology**: Use the same term throughout (e.g., "deployment" not "deployment/deploy/deploying")

### Code Examples

#### YAML Configuration

```yaml
# Always include comments explaining the configuration
apiVersion: inference.llm-d.io/v1alpha1
kind: LLMDeployment
metadata:
  name: llama-3.1-8b  # Standard naming
  namespace: production  # Standard namespace
  labels:
    app.kubernetes.io/name: llm-d  # Standard labels
    llm-d.ai/model: "llama-3.1"
    llm-d.ai/size: "8b"
spec:
  # Reference shared-config.md for resource templates
  resources:
    requests:
      memory: "16Gi"
      cpu: "4"
      nvidia.com/gpu: "1"
```

#### Bash Commands

```bash
# Always include context and error handling
if ! kubectl get namespace production > /dev/null 2>&1; then
    echo "Creating production namespace..."
    kubectl create namespace production
fi

# Use standard namespace references
kubectl get llmdeployments -n production
kubectl describe llmdeployment llama-3.1-8b -n production

# Include verification steps
kubectl wait --for=condition=ready pod -l app=llama-3.1-8b -n production --timeout=300s
```

#### Python Code

```python title="script-name.py" showLineNumbers
#!/usr/bin/env python3
"""
Brief description of what this script does.
"""

import os
import json
from typing import Dict, List, Optional

def main():
    """Main function with clear docstring."""
    # Implementation here
    pass

if __name__ == "__main__":
    main()
```

### Content Patterns

#### Info Boxes

```markdown
:::info Chapter Overview
Brief overview of chapter content and learning outcomes.
:::

:::warning Important
Critical information that could cause issues if ignored.
:::

:::tip Best Practice
Helpful recommendations and optimization tips.
:::
```

#### Section Headers

- Use descriptive headers that indicate the action or concept
- Follow logical hierarchy (H1 → H2 → H3)
- Avoid generic headers like "Configuration" (use "Configuring Model Resources")

#### Cross-References

Always reference related content:

```markdown
For detailed specifications, see [Shared Configuration](./appendix/shared-config.md).

This builds on concepts from [Chapter 5: SRE Operations](./05-sre-operations.md).
```

## Quality Assurance

### Pre-Commit Validation

Before committing, run:

```bash
# Validate consistency
scripts/validate-model-names.sh
scripts/validate-resource-specs.sh
scripts/validate-namespaces.sh
scripts/check-shared-config-refs.sh

# Check formatting
npm run lint
npm run spell
```

### Required Elements

Every chapter must include:

- [ ] Standard YAML frontmatter
- [ ] Chapter overview info box
- [ ] At least one practical example
- [ ] Reference to shared-config.md (if applicable)
- [ ] Cross-references to related chapters
- [ ] Next steps section
- [ ] References section

### Common Mistakes to Avoid

1. **Inconsistent naming**: Using different model names across examples
2. **Missing namespaces**: Not specifying namespaces in kubectl commands
3. **Incomplete resource specs**: Missing either requests or limits
4. **Broken links**: References to non-existent chapters or sections
5. **Missing context**: Commands without explanation or error handling

## File Organization

### Directory Structure

```
docs/
├── 01-introduction.md
├── 02-installation-setup.md
├── [numbered-chapters].md
├── 08-troubleshooting/
│   ├── index.md
│   ├── 01-decision-trees.md
│   └── [subsections].md
├── 10-mlops-workflows/
│   ├── index.md
│   ├── 01-model-lifecycle.md
│   └── [subsections].md
├── appendix/
│   └── shared-config.md
└── templates/
    └── chapter-template.md
```

### Naming Conventions

- Chapters: `##-chapter-name.md`
- Subsections: `##-subsection-name.md`
- Use kebab-case for all filenames
- Include sidebar_position in frontmatter

## Validation and Testing

### Automated Checks

The following scripts validate content consistency:

- `validate-model-names.sh`: Ensures standard model naming
- `validate-resource-specs.sh`: Checks resource specification consistency
- `validate-namespaces.sh`: Validates namespace conventions
- `check-shared-config-refs.sh`: Ensures proper cross-referencing

### Manual Review Checklist

Before submitting content:

- [ ] All code examples are tested and functional
- [ ] Cross-references point to existing content
- [ ] Resource specifications match shared-config.md
- [ ] Model names use standard conventions
- [ ] Namespace references are consistent
- [ ] Writing is clear and follows active voice
- [ ] Chapter includes practical examples
- [ ] References section is complete

## Tools and Resources

### Development Tools

- **Markdownlint**: Automated formatting validation
- **CSpell**: Spell checking with technical dictionary
- **Pre-commit hooks**: Automated validation before commits

### Reference Materials

- [Shared Configuration](./appendix/shared-config.md): Standard specifications
- [Chapter Template](./templates/chapter-template.md): Standardized structure
- [Pre-commit config](./.pre-commit-config.yaml): Validation automation

---

**Remember**: Consistency is key to professional documentation. When in doubt, reference existing chapters and the shared configuration guide.