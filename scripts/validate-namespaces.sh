#!/bin/bash
# Validate namespace naming conventions

set -e

ERRORS=0
FILES_TO_CHECK="$*"

if [ -z "$FILES_TO_CHECK" ]; then
    FILES_TO_CHECK=$(find docs -name "*.md")
fi

echo "🔍 Validating namespace conventions..."

# Standard namespaces from shared config
STANDARD_NAMESPACES=(
    "llm-d-system"
    "production" 
    "staging"
    "development"
)

# Deprecated namespace patterns
DEPRECATED_PATTERNS=(
    "llm-d-production"
    "llm-d-staging" 
    "llm-d-dev"
    "llm-d-development"
)

for file in $FILES_TO_CHECK; do
    if [ ! -f "$file" ]; then
        continue
    fi
    
    # Check for deprecated namespace patterns
    for pattern in "${DEPRECATED_PATTERNS[@]}"; do
        if grep -q "$pattern" "$file"; then
            echo "❌ $file: Found deprecated namespace '$pattern'"
            case "$pattern" in
                "llm-d-production") echo "   Use: production" ;;
                "llm-d-staging") echo "   Use: staging" ;;
                "llm-d-dev"|"llm-d-development") echo "   Use: development" ;;
            esac
            ERRORS=$((ERRORS + 1))
        fi
    done
    
    # Check for inconsistent namespace references in kubectl commands
    if grep -q "kubectl.*-n" "$file"; then
        while IFS= read -r line; do
            if echo "$line" | grep -q "kubectl.*-n.*llm-d-" && ! echo "$line" | grep -q "llm-d-system"; then
                echo "❌ $file: Inconsistent namespace in kubectl command:"
                echo "   $line"
                echo "   Use standard namespaces: production, staging, development, llm-d-system"
                ERRORS=$((ERRORS + 1))
            fi
        done < <(grep "kubectl.*-n" "$file")
    fi
    
    # Check for namespace in YAML metadata
    if grep -q "namespace:" "$file"; then
        while IFS= read -r line; do
            namespace=$(echo "$line" | grep "namespace:" | sed 's/.*namespace: *//; s/"//g; s/ .*//')
            is_standard=false
            for std_ns in "${STANDARD_NAMESPACES[@]}"; do
                if [ "$namespace" = "$std_ns" ]; then
                    is_standard=true
                    break
                fi
            done
            
            if [ "$is_standard" = false ] && [[ "$namespace" =~ llm-d- ]]; then
                echo "❌ $file: Non-standard namespace '$namespace' in YAML"
                echo "   Line: $line"
                ERRORS=$((ERRORS + 1))
            fi
        done < <(grep "namespace:" "$file")
    fi
done

if [ $ERRORS -eq 0 ]; then
    echo "✅ Namespace conventions validation passed"
    exit 0
else
    echo "❌ Namespace conventions validation failed with $ERRORS errors"
    echo ""
    echo "💡 Reference standard namespaces in docs/appendix/shared-config.md"
    exit 1
fi