#!/bin/bash
# Validate model naming consistency across documentation

set -e

# Define standard model names from shared config
STANDARD_MODEL_NAMES=(
    "llama-3.1-8b"
    "llama-3.1-70b" 
    "mistral-7b"
    "codellama-13b"
)

# Define deprecated/inconsistent patterns
DEPRECATED_PATTERNS=(
    "llama4"
    "llama-8b"
    "llama-70b"
)

ERRORS=0
FILES_TO_CHECK="$*"

if [ -z "$FILES_TO_CHECK" ]; then
    FILES_TO_CHECK=$(find docs -name "*.md" -not -path "docs/appendix/shared-config.md")
fi

echo "🔍 Validating model naming consistency..."

for file in $FILES_TO_CHECK; do
    if [ ! -f "$file" ]; then
        continue
    fi
    
    # Check for deprecated patterns
    for pattern in "${DEPRECATED_PATTERNS[@]}"; do
        if grep -q "$pattern" "$file"; then
            echo "❌ $file: Found deprecated model name '$pattern'"
            echo "   Please use standard naming from docs/appendix/shared-config.md"
            ERRORS=$((ERRORS + 1))
        fi
    done
    
    # Check for inconsistent resource naming in YAML
    if grep -q "name:.*llama.*[0-9]" "$file"; then
        while IFS= read -r line; do
            if echo "$line" | grep -q "name:.*llama" && ! echo "$line" | grep -E "(llama-3\.1-8b|llama-3\.1-70b)" > /dev/null; then
                echo "❌ $file: Inconsistent resource name in line: $line"
                echo "   Use: llama-3.1-8b or llama-3.1-70b"
                ERRORS=$((ERRORS + 1))
            fi
        done < <(grep "name:.*llama" "$file")
    fi
done

if [ $ERRORS -eq 0 ]; then
    echo "✅ Model naming validation passed"
    exit 0
else
    echo "❌ Model naming validation failed with $ERRORS errors"
    echo ""
    echo "💡 Reference standard model names in docs/appendix/shared-config.md"
    exit 1
fi