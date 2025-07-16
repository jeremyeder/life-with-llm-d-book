#!/bin/bash
# Enhanced comprehensive validation runner for preventing documentation drift

set -e

# Color codes for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Enhanced Documentation Validation Suite${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""

# Track overall results
TOTAL_ERRORS=0
TOTAL_WARNINGS=0
VALIDATION_COUNT=0

# Function to run validation and track results
run_validation() {
    local script_name="$1"
    local description="$2"
    local category="$3"
    
    VALIDATION_COUNT=$((VALIDATION_COUNT + 1))
    
    echo -e "${BLUE}📋 [$category] $description${NC}"
    echo "   Running: $script_name"
    
    local start_time=$(date +%s)
    
    if ./"$script_name"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo -e "${GREEN}   ✅ PASSED${NC} (${duration}s)"
    else
        exit_code=$?
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        if [ $exit_code -eq 1 ]; then
            echo -e "${RED}   ❌ FAILED${NC} (${duration}s)"
            TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
        else
            echo -e "${YELLOW}   ⚠️  WARNINGS${NC} (${duration}s)"
            TOTAL_WARNINGS=$((TOTAL_WARNINGS + 1))
        fi
    fi
    echo ""
}

# Change to repo root
cd "$(dirname "$0")/.."

echo -e "${BLUE}🔍 Phase 1: Core Consistency Validation${NC}"
echo "=============================================="

# Original validation scripts (enhanced)
run_validation "scripts/validate-model-names.sh" "Model naming consistency" "CONSISTENCY"
run_validation "scripts/validate-resource-specs.sh" "Resource specifications" "CONSISTENCY"  
run_validation "scripts/validate-namespaces.sh" "Namespace conventions" "CONSISTENCY"
run_validation "scripts/check-shared-config-refs.sh" "Shared config references" "CONSISTENCY"

echo -e "${BLUE}🔍 Phase 2: Enhanced Content Validation${NC}"
echo "=============================================="

# New enhanced validation scripts
run_validation "scripts/validate-yaml-syntax.sh" "YAML syntax and structure" "SYNTAX"
run_validation "scripts/validate-cross-references.sh" "Cross-references and links" "LINKS"
run_validation "scripts/validate-mathematical-accuracy.sh" "Mathematical accuracy" "ACCURACY"
run_validation "scripts/validate-technical-claims.sh" "Technical claims validation" "ACCURACY"
run_validation "scripts/validate-consistency-matrix.sh" "Cross-chapter consistency" "CONSISTENCY"
run_validation "scripts/validate-works-cited.sh" "Works cited format and accessibility" "REFERENCES"

echo -e "${BLUE}🔍 Phase 3: Content Quality Validation${NC}"
echo "=============================================="

# Run markdown linting
echo -e "${BLUE}📋 [FORMATTING] Markdown formatting and style${NC}"
echo "   Running: npm run lint"
local start_time=$(date +%s)
if npm run lint > /dev/null 2>&1; then
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo -e "${GREEN}   ✅ PASSED${NC} (${duration}s)"
else
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo -e "${RED}   ❌ FAILED${NC} (${duration}s)"
    TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
fi
echo ""

# Run spell checking
echo -e "${BLUE}📋 [FORMATTING] Spell checking${NC}"
echo "   Running: npm run spell"
local start_time=$(date +%s)
if npm run spell > /dev/null 2>&1; then
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo -e "${GREEN}   ✅ PASSED${NC} (${duration}s)"
else
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo -e "${YELLOW}   ⚠️  WARNINGS${NC} (${duration}s) (check for legitimate technical terms)"
    TOTAL_WARNINGS=$((TOTAL_WARNINGS + 1))
fi
echo ""

VALIDATION_COUNT=$((VALIDATION_COUNT + 2))

echo -e "${BLUE}🔍 Phase 4: Code Example Validation${NC}"
echo "=============================================="

# Run pytest for code examples
echo -e "${BLUE}📋 [CODE] Python code examples${NC}"
echo "   Running: pytest tests/ -x --tb=short"
local start_time=$(date +%s)
if pytest tests/ -x --tb=short > /dev/null 2>&1; then
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo -e "${GREEN}   ✅ PASSED${NC} (${duration}s)"
else
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo -e "${RED}   ❌ FAILED${NC} (${duration}s)"
    TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
fi
VALIDATION_COUNT=$((VALIDATION_COUNT + 1))
echo ""

# Summary
echo "=============================================="
echo -e "${BLUE}📊 ENHANCED VALIDATION SUMMARY${NC}"
echo "=============================================="
echo -e "${BLUE}Total validations run: $VALIDATION_COUNT${NC}"
echo ""

if [ $TOTAL_ERRORS -eq 0 ] && [ $TOTAL_WARNINGS -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL VALIDATIONS PASSED!${NC}"
    echo -e "${GREEN}✅ Documentation is ready for commit${NC}"
    echo -e "${GREEN}🛡️  Enhanced validation prevents drift${NC}"
    exit 0
elif [ $TOTAL_ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠️  $TOTAL_WARNINGS warning(s) found across $VALIDATION_COUNT validations${NC}"
    echo -e "${YELLOW}💡 Review warnings but safe to proceed${NC}"
    echo ""
    echo -e "${BLUE}🔧 Recommended actions:${NC}"
    echo "   - Review technical claims for accuracy"
    echo "   - Check cross-references for completeness"
    echo "   - Validate version consistency across chapters"
    exit 0
else
    echo -e "${RED}❌ $TOTAL_ERRORS error(s) and $TOTAL_WARNINGS warning(s) found${NC}"
    echo -e "${RED}🔧 Fix critical errors before committing${NC}"
    echo ""
    echo -e "${BLUE}💡 Enhanced fix recommendations:${NC}"
    echo "   - Model names: Use exact shared-config.md standards"
    echo "   - Namespaces: production, staging, development, llm-d-system only"
    echo "   - YAML syntax: Fix duplicate keys and invalid formats"
    echo "   - Math accuracy: Verify SLO calculations and cost claims"
    echo "   - Cross-references: Fix broken links and chapter references"
    echo "   - Consistency: Align resource specs across chapters"
    echo ""
    echo -e "${BLUE}🚀 Run individual scripts for detailed diagnostics:${NC}"
    echo "   ./scripts/validate-yaml-syntax.sh"
    echo "   ./scripts/validate-mathematical-accuracy.sh"
    echo "   ./scripts/validate-consistency-matrix.sh"
    exit 1
fi