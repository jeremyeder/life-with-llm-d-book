#!/bin/bash
# Refresh works-cited access dates and validate URLs

set -e

# Color codes for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

WORKS_CITED_FILE="docs/appendix/works-cited.md"

echo -e "${BLUE}📚 Refreshing works-cited access dates and validating URLs${NC}"

if [ ! -f "$WORKS_CITED_FILE" ]; then
    echo -e "${RED}❌ Works cited file not found: $WORKS_CITED_FILE${NC}" >&2
    exit 1
fi

# Get current date for access date updates
CURRENT_DATE=$(date "+%B %d, %Y")
CURRENT_DATE_SHORT=$(date "+%Y-%m-%d")

# Update all "Accessed:" dates to current date
echo -e "${BLUE}  📅 Updating access dates to ${CURRENT_DATE}${NC}"
sed -i "s/Accessed: .*/Accessed: ${CURRENT_DATE}/" "$WORKS_CITED_FILE"

# Update "Last updated" date
sed -i "s/\*Last updated\*: .*/\*Last updated\*: ${CURRENT_DATE}/" "$WORKS_CITED_FILE"

# Calculate next review date (add one month)
NEXT_REVIEW_DATE=$(python3 -c "
from datetime import datetime
import calendar
current = datetime.strptime('$CURRENT_DATE_SHORT', '%Y-%m-%d')
if current.month == 12:
    next_month = current.replace(year=current.year + 1, month=1)
else:
    next_month = current.replace(month=current.month + 1)
print(next_month.strftime('%B %d, %Y'))
" 2>/dev/null || echo "Next month")

sed -i "s/\*Next scheduled review\*: .*/\*Next scheduled review\*: ${NEXT_REVIEW_DATE}/" "$WORKS_CITED_FILE"

echo -e "${BLUE}  🔍 Validating URL accessibility...${NC}"

# Extract and validate URLs
urls=$(grep -oE 'https://[^[:space:]]+' "$WORKS_CITED_FILE" | sort -u)
total_urls=0
accessible_urls=0
failed_urls=()

for url in $urls; do
    total_urls=$((total_urls + 1))
    
    # Test URL with timeout and follow redirects
    if curl -s --head --max-time 10 --location "$url" > /dev/null 2>&1; then
        accessible_urls=$((accessible_urls + 1))
        echo -e "${GREEN}    ✅ $url${NC}"
    else
        failed_urls+=("$url")
        echo -e "${RED}    ❌ $url${NC}"
    fi
done

echo ""
echo -e "${BLUE}📊 URL Validation Summary:${NC}"
echo -e "  Total URLs: $total_urls"
echo -e "  Accessible: $accessible_urls"
echo -e "  Failed: ${#failed_urls[@]}"

if [ ${#failed_urls[@]} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}⚠️  Failed URLs that need attention:${NC}"
    for url in "${failed_urls[@]}"; do
        echo -e "  - $url"
    done
    echo ""
    echo -e "${YELLOW}💡 These URLs should be reviewed and updated if necessary${NC}"
fi

# Validate citation format consistency
echo -e "${BLUE}  📋 Validating citation format...${NC}"

# Check for required sections
required_sections=(
    "Primary Project Documentation"
    "Technical Components Documentation" 
    "Official Announcements and Strategic Context"
    "Citation Standards"
)

missing_sections=()
for section in "${required_sections[@]}"; do
    if grep -q "## $section" "$WORKS_CITED_FILE"; then
        echo -e "${GREEN}    ✅ Section found: $section${NC}"
    else
        missing_sections+=("$section")
        echo -e "${RED}    ❌ Missing section: $section${NC}"
    fi
done

# Check for GitHub organization consistency
github_urls=$(grep -oE 'https://github\.com/[^[:space:]]+' "$WORKS_CITED_FILE")
non_llmd_count=0

for url in $github_urls; do
    if ! echo "$url" | grep -q "github.com/llm-d/"; then
        non_llmd_count=$((non_llmd_count + 1))
        echo -e "${YELLOW}    ⚠️  Non-llm-d GitHub URL: $url${NC}"
    fi
done

if [ $non_llmd_count -eq 0 ]; then
    echo -e "${GREEN}    ✅ All GitHub URLs point to llm-d organization${NC}"
fi

echo ""
echo -e "${BLUE}📊 Works Cited Refresh Summary:${NC}"
echo -e "  ✅ Access dates updated to: ${CURRENT_DATE}"
echo -e "  ✅ Next review scheduled: ${NEXT_REVIEW_DATE}"
echo -e "  📊 URL validation: ${accessible_urls}/${total_urls} accessible"

if [ ${#failed_urls[@]} -eq 0 ] && [ ${#missing_sections[@]} -eq 0 ]; then
    echo -e "${GREEN}🎉 Works cited refresh completed successfully!${NC}"
    exit 0
else
    if [ ${#failed_urls[@]} -gt 0 ]; then
        echo -e "${YELLOW}⚠️  ${#failed_urls[@]} URL(s) need attention${NC}"
    fi
    if [ ${#missing_sections[@]} -gt 0 ]; then
        echo -e "${RED}❌ ${#missing_sections[@]} required section(s) missing${NC}"
    fi
    echo -e "${YELLOW}💡 Review and fix issues before finalizing update${NC}"
    exit 2
fi