#!/bin/bash
# Check that chapters reference shared config when appropriate

set -e

WARNINGS=0
FILES_TO_CHECK="$*"

if [ -z "$FILES_TO_CHECK" ]; then
    FILES_TO_CHECK=$(find docs -name "*.md" -not -path "docs/appendix/*" -not -path "docs/08-troubleshooting/*")
fi

echo "🔍 Checking shared config references..."

# Chapters that should reference shared config
CHAPTERS_NEEDING_REFS=(
    "docs/02-installation-setup.md"
    "docs/04-data-scientist-workflows.md" 
    "docs/05-sre-operations.md"
    "docs/10-mlops-workflows/index.md"
)

for file in $FILES_TO_CHECK; do
    if [ ! -f "$file" ]; then
        continue
    fi
    
    # Check if file contains model specs or resources but no shared config reference
    has_model_specs=false
    has_shared_ref=false
    
    if grep -q "llama-3\.1\|mistral-7b\|codellama" "$file"; then
        has_model_specs=true
    fi
    
    if grep -q "resources:" "$file" && grep -q "memory:\|cpu:\|nvidia.com/gpu" "$file"; then
        has_model_specs=true
    fi
    
    if grep -q "shared-config.md\|Shared Configuration" "$file"; then
        has_shared_ref=true
    fi
    
    # Check for important chapters
    for important_chapter in "${CHAPTERS_NEEDING_REFS[@]}"; do
        if [ "$file" = "$important_chapter" ]; then
            if [ "$has_shared_ref" = false ]; then
                echo "⚠️  $file: Important chapter missing shared config reference"
                echo "   Add reference to docs/appendix/shared-config.md"
                WARNINGS=$((WARNINGS + 1))
            fi
            break
        fi
    done
    
    # Warn if file has model specs but no reference
    if [ "$has_model_specs" = true ] && [ "$has_shared_ref" = false ]; then
        echo "⚠️  $file: Contains model/resource specs but no shared config reference"
        echo "   Consider adding: [Shared Configuration](../appendix/shared-config.md)"
        WARNINGS=$((WARNINGS + 1))
    fi
done

# Check if shared config is up to date with recent changes
if [ -f "docs/appendix/shared-config.md" ]; then
    config_age=$(stat -f "%m" "docs/appendix/shared-config.md" 2>/dev/null || stat -c "%Y" "docs/appendix/shared-config.md" 2>/dev/null || echo "0")
    current_time=$(date +%s)
    age_days=$(( (current_time - config_age) / 86400 ))
    
    if [ $age_days -gt 7 ]; then
        echo "⚠️  Shared config is $age_days days old - consider reviewing for updates"
    fi
fi

if [ $WARNINGS -eq 0 ]; then
    echo "✅ Shared config references check passed"
    exit 0
else
    echo "⚠️  Found $WARNINGS potential improvements for shared config references"
    echo ""
    echo "💡 These are recommendations, not errors"
    exit 0
fi