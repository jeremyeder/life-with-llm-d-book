#!/bin/bash
# Validate resource specifications consistency

set -e

ERRORS=0
FILES_TO_CHECK="$*"

if [ -z "$FILES_TO_CHECK" ]; then
    FILES_TO_CHECK=$(find docs -name "*.md" -not -path "docs/appendix/shared-config.md")
fi

echo "🔍 Validating resource specifications..."

# Standard resource patterns from shared config
SMALL_MODEL_MEMORY_VALUES=("16Gi" "24Gi")
LARGE_MODEL_MEMORY_VALUES=("160Gi" "200Gi")

for file in $FILES_TO_CHECK; do
    if [ ! -f "$file" ]; then
        continue
    fi
    
    # Check for inconsistent small model (8B) resource specs
    if grep -q "llama-3\.1-8b\|mistral-7b" "$file"; then
        # Check memory specs for small models
        if grep -A 10 -B 5 "llama-3\.1-8b\|mistral-7b" "$file" | grep -q "memory:" ; then
            memory_values=$(grep -A 10 -B 5 "llama-3\.1-8b\|mistral-7b" "$file" | grep "memory:" | grep -o '"[0-9]*Gi"' | tr -d '"')
            for mem in $memory_values; do
                if [[ "$mem" != "16Gi" && "$mem" != "24Gi" ]]; then
                    echo "❌ $file: Non-standard memory spec '$mem' for small model"
                    echo "   Expected: 16Gi (requests) or 24Gi (limits)"
                    ERRORS=$((ERRORS + 1))
                fi
            done
        fi
    fi
    
    # Check for inconsistent large model (70B) resource specs  
    if grep -q "llama-3\.1-70b" "$file"; then
        if grep -A 10 -B 5 "llama-3\.1-70b" "$file" | grep -q "memory:" ; then
            memory_values=$(grep -A 10 -B 5 "llama-3\.1-70b" "$file" | grep "memory:" | grep -o '"[0-9]*Gi"' | tr -d '"')
            for mem in $memory_values; do
                if [[ "$mem" != "160Gi" && "$mem" != "200Gi" ]]; then
                    echo "❌ $file: Non-standard memory spec '$mem' for large model"  
                    echo "   Expected: 160Gi (requests) or 200Gi (limits)"
                    ERRORS=$((ERRORS + 1))
                fi
            done
        fi
    fi
    
    # Check for missing resource requests/limits
    if grep -q "nvidia.com/gpu" "$file"; then
        gpu_blocks=$(grep -A 5 -B 5 "nvidia.com/gpu" "$file")
        if ! echo "$gpu_blocks" | grep -q "requests:" || ! echo "$gpu_blocks" | grep -q "limits:"; then
            echo "⚠️  $file: GPU resource found without both requests and limits"
            echo "   Best practice: Always specify both requests and limits"
        fi
    fi
done

if [ $ERRORS -eq 0 ]; then
    echo "✅ Resource specifications validation passed"
    exit 0
else
    echo "❌ Resource specifications validation failed with $ERRORS errors"
    echo ""
    echo "💡 Reference standard resource templates in docs/appendix/shared-config.md"
    exit 1
fi