# Documentation Safeguards

This document explains the automated safeguards implemented to prevent content drift and maintain consistency across the "Life with llm-d" book.

## 🛡️ Implemented Safeguards

### 1. Pre-Commit Hooks (`.pre-commit-config.yaml`)

Automated validation that runs before every commit:

```bash
# Install pre-commit hooks (one-time setup)
pip install pre-commit
pre-commit install

# Hooks will run automatically on commit, or manually:
pre-commit run --all-files
```

### 2. Validation Scripts (`scripts/`)

Four specialized validation scripts:

- **`validate-model-names.sh`**: Ensures standard model naming (llama-3.1-8b, mistral-7b)
- **`validate-resource-specs.sh`**: Checks resource specification consistency
- **`validate-namespaces.sh`**: Validates namespace conventions (production, staging, development)
- **`check-shared-config-refs.sh`**: Ensures proper cross-referencing

### 3. Comprehensive Validation (`scripts/run-all-validations.sh`)

Single command to run all checks:

```bash
./scripts/run-all-validations.sh
```

### 4. Enhanced Linting (`.markdownlint.json`)

Stricter markdown formatting rules for consistency.

### 5. Shared Configuration Reference (`docs/appendix/shared-config.md`)

Single source of truth for:
- Model specifications
- Resource templates  
- Namespace conventions
- Configuration standards

### 6. Chapter Template (`templates/chapter-template.md`)

Standardized structure for new chapters with embedded standards.

### 7. Style Guide (`STYLE_GUIDE.md`)

Comprehensive writing and formatting guidelines.

## 🚀 Usage for New Chapters

### Before Writing

1. **Copy the template**:
   ```bash
   cp templates/chapter-template.md docs/##-new-chapter.md
   ```

2. **Review the style guide**:
   ```bash
   # Read the standards
   cat STYLE_GUIDE.md
   ```

### During Writing

3. **Reference shared config**:
   - Use standard model names from `docs/appendix/shared-config.md`
   - Use standard resource templates
   - Use standard namespace conventions

4. **Validate frequently**:
   ```bash
   ./scripts/run-all-validations.sh
   ```

### Before Committing

5. **Final validation**:
   ```bash
   # Run comprehensive checks
   ./scripts/run-all-validations.sh
   
   # Fix any errors before committing
   git add .
   git commit -m "Your commit message"
   ```

## 🔧 Validation Details

### Model Name Validation

**✅ Correct:**
```yaml
metadata:
  name: llama-3.1-8b
spec:
  model:
    name: "meta-llama/Llama-3.1-8B-Instruct"
```

**❌ Incorrect:**
```yaml
metadata:
  name: llama4-7b  # Deprecated
spec:
  model:
    name: "meta-llama/Llama-4-7B"  # Non-standard
```

### Namespace Validation

**✅ Correct:**
```bash
kubectl get pods -n production
kubectl describe deployment -n staging
```

**❌ Incorrect:**
```bash
kubectl get pods -n llm-d-production  # Deprecated
kubectl describe deployment -n llm-d-staging  # Deprecated
```

### Resource Specification Validation

**✅ Correct (8B model):**
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

**❌ Incorrect:**
```yaml
resources:
  requests:
    memory: "8Gi"  # Non-standard for 8B model
    cpu: "2"       # Non-standard
```

## 🎯 Benefits

### Prevents Future Drift

- **Automated enforcement** of naming conventions
- **Consistent resource specifications** across chapters
- **Standardized cross-references** between chapters
- **Uniform writing style** and formatting

### Improves Quality

- **Early error detection** before publishing
- **Professional consistency** across all content
- **Reduced manual review** time
- **Easier maintenance** and updates

### Enables Collaboration

- **Clear standards** for contributors
- **Automated validation** reduces review burden
- **Template-based** chapter creation
- **Documentation of decisions** in shared config

## 📋 Validation Status

Current book validation status (run `./scripts/run-all-validations.sh` for live status):

- ✅ **Shared Configuration**: Established and referenced
- ⚠️  **Model Naming**: Some legacy inconsistencies remain
- ⚠️  **Namespace Conventions**: Migration from old patterns in progress
- ✅ **Resource Specifications**: Standardized templates in place
- ✅ **Cross-References**: Shared config properly referenced
- ✅ **Formatting**: Automated linting rules active

## 🔄 Continuous Improvement

### Adding New Standards

1. Update `docs/appendix/shared-config.md`
2. Update validation scripts in `scripts/`
3. Update `STYLE_GUIDE.md`
4. Update chapter template if needed

### Extending Validation

Add new validation scripts:

```bash
# Create new validation
cat > scripts/validate-new-standard.sh << 'EOF'
#!/bin/bash
# Validate new standard across documentation
# Implementation here
EOF

chmod +x scripts/validate-new-standard.sh

# Add to run-all-validations.sh
# Add to .pre-commit-config.yaml
```

## 🎉 Result

These safeguards ensure that future chapters automatically maintain consistency with existing content, preventing the drift that required our recent cohesiveness fixes. New contributors can follow the template and validation will catch any deviations from standards.