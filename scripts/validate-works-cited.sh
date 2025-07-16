#!/bin/bash
# Validate works-cited.md for URL accessibility and citation format consistency

set -e

# Color codes for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

WORKS_CITED_FILE="docs/appendix/works-cited.md"
ERRORS=0
WARNINGS=0

echo -e "${BLUE}🔍 Validating Works Cited Section${NC}"
echo "=================================="

# Check if works-cited.md exists
if [ ! -f "$WORKS_CITED_FILE" ]; then
    echo -e "${RED}❌ ERROR: works-cited.md not found at $WORKS_CITED_FILE${NC}"
    exit 1
fi

echo -e "${BLUE}📋 Checking URL accessibility...${NC}"

# Extract URLs from works-cited.md and test accessibility
urls=$(grep -oE 'https://[^[:space:]]+' "$WORKS_CITED_FILE" | sort -u)

if [ -z "$urls" ]; then
    echo -e "${RED}❌ ERROR: No URLs found in works-cited.md${NC}"
    ERRORS=$((ERRORS + 1))
else
    url_count=0
    accessible_count=0
    
    for url in $urls; do
        url_count=$((url_count + 1))
        
        # Test URL accessibility with timeout
        if curl -s --head --max-time 10 "$url" > /dev/null 2>&1; then
            accessible_count=$((accessible_count + 1))
            echo -e "${GREEN}   ✅ $url${NC}"
        else
            echo -e "${RED}   ❌ $url (inaccessible)${NC}"
            ERRORS=$((ERRORS + 1))
        fi
    done
    
    echo ""
    echo -e "${BLUE}📊 URL Accessibility Summary:${NC}"
    echo "   Total URLs: $url_count"
    echo "   Accessible: $accessible_count"
    echo "   Inaccessible: $((url_count - accessible_count))"
fi

echo ""
echo -e "${BLUE}📋 Checking citation format consistency...${NC}"

# Check for required sections
required_sections=(
    "Primary Project Documentation"
    "Technical Components Documentation" 
    "Official Announcements and Strategic Context"
    "Citation Standards"
)

for section in "${required_sections[@]}"; do
    if grep -q "## $section" "$WORKS_CITED_FILE"; then
        echo -e "${GREEN}   ✅ Section found: $section${NC}"
    else
        echo -e "${RED}   ❌ Missing section: $section${NC}"
        ERRORS=$((ERRORS + 1))
    fi
done

# Check for access dates
if grep -q "Accessed:" "$WORKS_CITED_FILE"; then
    access_date_count=$(grep -c "Accessed:" "$WORKS_CITED_FILE")
    echo -e "${GREEN}   ✅ Access dates found: $access_date_count entries${NC}"
else
    echo -e "${RED}   ❌ No access dates found${NC}"
    ERRORS=$((ERRORS + 1))
fi

# Check for last updated date
if grep -q "Last updated:" "$WORKS_CITED_FILE"; then
    echo -e "${GREEN}   ✅ Last updated date found${NC}"
else
    echo -e "${YELLOW}   ⚠️  No 'Last updated' date found${NC}"
    WARNINGS=$((WARNINGS + 1))
fi

# Check for next review date  
if grep -q "Next scheduled review:" "$WORKS_CITED_FILE"; then
    echo -e "${GREEN}   ✅ Next review date found${NC}"
else
    echo -e "${YELLOW}   ⚠️  No 'Next scheduled review' date found${NC}"
    WARNINGS=$((WARNINGS + 1))
fi

# Validate GitHub URLs point to llm-d organization
github_urls=$(grep -oE 'https://github\.com/[^[:space:]]+' "$WORKS_CITED_FILE")
non_llmd_count=0

for url in $github_urls; do
    if ! echo "$url" | grep -q "github.com/llm-d/"; then
        echo -e "${YELLOW}   ⚠️  Non-llm-d GitHub URL: $url${NC}"
        non_llmd_count=$((non_llmd_count + 1))
        WARNINGS=$((WARNINGS + 1))
    fi
done

if [ $non_llmd_count -eq 0 ]; then
    echo -e "${GREEN}   ✅ All GitHub URLs point to llm-d organization${NC}"
fi

echo ""
echo "=================================="
echo -e "${BLUE}📊 WORKS CITED VALIDATION SUMMARY${NC}"
echo "=================================="

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL VALIDATIONS PASSED!${NC}"
    echo -e "${GREEN}✅ Works cited section is properly formatted and accessible${NC}"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠️  $WARNINGS warning(s) found${NC}"
    echo -e "${YELLOW}💡 Consider addressing warnings for completeness${NC}"
    exit 2
else
    echo -e "${RED}❌ $ERRORS error(s) and $WARNINGS warning(s) found${NC}"
    echo -e "${RED}🔧 Fix critical errors before proceeding${NC}"
    echo ""
    echo -e "${BLUE}💡 Common fixes:${NC}"
    echo "   - Ensure all URLs are accessible"
    echo "   - Add missing required sections"
    echo "   - Include access dates for all references"
    echo "   - Add last updated and next review dates"
    exit 1
fi