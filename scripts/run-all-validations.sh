#!/bin/bash
# Run all validation scripts for comprehensive checking

set -e

echo "🚀 Running comprehensive documentation validation..."
echo ""

# Track overall results
TOTAL_ERRORS=0
TOTAL_WARNINGS=0

# Function to run validation and track results
run_validation() {
    local script_name="$1"
    local description="$2"
    
    echo "📋 $description"
    echo "   Running: $script_name"
    
    if ./"$script_name"; then
        echo "   ✅ PASSED"
    else
        exit_code=$?
        if [ $exit_code -eq 1 ]; then
            echo "   ❌ FAILED"
            TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
        else
            echo "   ⚠️  WARNINGS"
            TOTAL_WARNINGS=$((TOTAL_WARNINGS + 1))
        fi
    fi
    echo ""
}

# Change to repo root
cd "$(dirname "$0")/.."

# Run all validation scripts
run_validation "scripts/validate-model-names.sh" "Model naming consistency"
run_validation "scripts/validate-resource-specs.sh" "Resource specifications"  
run_validation "scripts/validate-namespaces.sh" "Namespace conventions"
run_validation "scripts/check-shared-config-refs.sh" "Shared config references"

# Run markdown linting
echo "📋 Markdown formatting and style"
echo "   Running: npm run lint"
if npm run lint > /dev/null 2>&1; then
    echo "   ✅ PASSED"
else
    echo "   ❌ FAILED"
    TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
fi
echo ""

# Run spell checking
echo "📋 Spell checking"
echo "   Running: npm run spell"
if npm run spell > /dev/null 2>&1; then
    echo "   ✅ PASSED"
else
    echo "   ⚠️  WARNINGS (check for legitimate technical terms)"
    TOTAL_WARNINGS=$((TOTAL_WARNINGS + 1))
fi
echo ""

# Summary
echo "========================================="
echo "📊 VALIDATION SUMMARY"
echo "========================================="

if [ $TOTAL_ERRORS -eq 0 ] && [ $TOTAL_WARNINGS -eq 0 ]; then
    echo "🎉 ALL VALIDATIONS PASSED!"
    echo "✅ Documentation is ready for commit"
    exit 0
elif [ $TOTAL_ERRORS -eq 0 ]; then
    echo "⚠️  $TOTAL_WARNINGS warning(s) found"
    echo "💡 Review warnings but safe to proceed"
    exit 0
else
    echo "❌ $TOTAL_ERRORS error(s) and $TOTAL_WARNINGS warning(s) found"
    echo "🔧 Fix errors before committing"
    echo ""
    echo "💡 Quick fixes:"
    echo "   - Check model names against docs/appendix/shared-config.md"
    echo "   - Use standard namespaces: production, staging, development"
    echo "   - Ensure resource specs match templates"
    echo "   - Add shared config references where appropriate"
    exit 1
fi