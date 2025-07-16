#!/bin/bash
# Update What's New section with new release information

set -e

VERSION="$1"
RELEASE_DATE="$2"

if [ -z "$VERSION" ] || [ -z "$RELEASE_DATE" ]; then
    echo "Usage: $0 <version> <release_date>" >&2
    exit 1
fi

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

WHATS_NEW_FILE="docs/00-whats-new.md"

echo -e "${BLUE}📝 Updating What's New section for llm-d v${VERSION}${NC}"

if [ ! -f "$WHATS_NEW_FILE" ]; then
    echo -e "${RED}❌ What's New file not found: $WHATS_NEW_FILE${NC}" >&2
    exit 1
fi

# Update current release info box
sed -i "s/\*\*llm-d v[^*]*\*\* - .*/\*\*llm-d v${VERSION}\*\* - Latest stable release/" "$WHATS_NEW_FILE"
sed -i "s/Released: .*/Released: ${RELEASE_DATE}/" "$WHATS_NEW_FILE"

# Update latest release highlights
FORMATTED_DATE=$(date -d "$RELEASE_DATE" "+%B %d, %Y" 2>/dev/null || echo "$RELEASE_DATE")

# Add new release section at the top of highlights
NEW_SECTION="### llm-d v${VERSION} - Latest Release
*Released: ${FORMATTED_DATE}*

**🚀 New Features:**
- Enhanced platform stability and performance
- Improved deployment reliability
- Community feedback integration

**🔧 Improvements:**
- Bug fixes and optimizations
- Documentation updates
- Enhanced validation frameworks

**📊 Performance:**
- Continued optimization for production workloads
- Improved resource efficiency
- Enhanced monitoring capabilities

---

"

# Insert the new section after the "Latest Release Highlights" header
sed -i "/## Latest Release Highlights/a\\
\\
${NEW_SECTION}" "$WHATS_NEW_FILE"

# Update the "Next Release" date at bottom
NEXT_MAJOR=$(echo "$VERSION" | cut -d. -f1)
NEXT_VERSION="$((NEXT_MAJOR + 1)).0.0"

# Calculate next release date (add one month)
NEXT_DATE=$(python3 -c "
from datetime import datetime
import calendar
current = datetime.strptime('$RELEASE_DATE', '%Y-%m-%d')
if current.month == 12:
    next_month = current.replace(year=current.year + 1, month=1)
else:
    next_month = current.replace(month=current.month + 1)
print(next_month.strftime('%B %d, %Y'))
" 2>/dev/null || echo "Next month")

NEXT_DATE_ISO=$(python3 -c "
from datetime import datetime
import calendar
current = datetime.strptime('$RELEASE_DATE', '%Y-%m-%d')
if current.month == 12:
    next_month = current.replace(year=current.year + 1, month=1)
else:
    next_month = current.replace(month=current.month + 1)
print(next_month.strftime('%Y-%m-%d'))
" 2>/dev/null || echo "$RELEASE_DATE")

sed -i "s/\*\*Next Release\*\*: llm-d v[^*]* expected .*/\*\*Next Release\*\*: llm-d v${NEXT_VERSION} expected ${NEXT_DATE_ISO}/" "$WHATS_NEW_FILE"

# Update last updated date
CURRENT_DATE=$(date "+%B %d, %Y")
sed -i "s/\*\*Last Updated\*\*: .*/\*\*Last Updated\*\*: $(date "+%B %d, %Y")/" "$WHATS_NEW_FILE"

# Update recent announcements
ANNOUNCEMENT_MONTH=$(date -d "$RELEASE_DATE" "+%B %Y" 2>/dev/null || echo "Current month")
NEW_ANNOUNCEMENT="- **${ANNOUNCEMENT_MONTH}**: llm-d v${VERSION} released with enhanced stability and performance"

# Add new announcement to the top of the list
sed -i "/### Recent Announcements/a\\
${NEW_ANNOUNCEMENT}" "$WHATS_NEW_FILE"

echo -e "${GREEN}✅ What's New section updated successfully${NC}"
echo -e "${BLUE}  - Current release: v${VERSION} (${RELEASE_DATE})${NC}"
echo -e "${BLUE}  - Next release: v${NEXT_VERSION} (${NEXT_DATE_ISO})${NC}"