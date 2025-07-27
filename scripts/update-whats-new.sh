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

# Use Python for more reliable text processing
python3 << EOF
import re
import sys
from datetime import datetime

VERSION = "$VERSION"
RELEASE_DATE = "$RELEASE_DATE"
WHATS_NEW_FILE = "$WHATS_NEW_FILE"

try:
    # Read the file
    with open(WHATS_NEW_FILE, 'r') as f:
        content = f.read()
    
    # Update current release info box
    content = re.sub(
        r'\*\*llm-d v[0-9.]+\*\* - Latest stable release',
        f'**llm-d v{VERSION}** - Latest stable release',
        content
    )
    
    # Update release date
    content = re.sub(
        r'Released: [^\n]+',
        f'Released: {RELEASE_DATE}',
        content
    )
    
    # Format the date nicely
    try:
        date_obj = datetime.strptime(RELEASE_DATE, '%Y-%m-%d')
        formatted_date = date_obj.strftime('%B %d, %Y')
    except:
        formatted_date = RELEASE_DATE
    
    # Add new release section
    new_section = f"""
### llm-d v{VERSION} - Latest Release
*Released: {formatted_date}*

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

"""
    
    # Insert after "Latest Release Highlights"
    content = re.sub(
        r'(## Latest Release Highlights\n)',
        f'\\1{new_section}',
        content
    )
    
    # Calculate next version and date
    version_parts = VERSION.split('.')
    next_major = int(version_parts[0]) + 1
    next_version = f"{next_major}.0.0"
    
    # Calculate next month
    try:
        current_date = datetime.strptime(RELEASE_DATE, '%Y-%m-%d')
        if current_date.month == 12:
            next_month = current_date.replace(year=current_date.year + 1, month=1)
        else:
            next_month = current_date.replace(month=current_date.month + 1)
        next_date_formatted = next_month.strftime('%B %d, %Y')
        next_date_iso = next_month.strftime('%Y-%m-%d')
    except:
        next_date_formatted = "Next month"
        next_date_iso = RELEASE_DATE
    
    # Update next release info
    content = re.sub(
        r'\*\*Next Release\*\*: llm-d v[0-9.]+ expected [^\n]+',
        f'**Next Release**: llm-d v{next_version} expected {next_date_iso}',
        content
    )
    
    # Update last updated date
    current_date_str = datetime.now().strftime('%B %d, %Y')
    content = re.sub(
        r'\*\*Last Updated\*\*: [^\n]+',
        f'**Last Updated**: {current_date_str}',
        content
    )
    
    # Add new announcement
    try:
        announcement_month = datetime.strptime(RELEASE_DATE, '%Y-%m-%d').strftime('%B %Y')
    except:
        announcement_month = "Current month"
    
    new_announcement = f"- **{announcement_month}**: llm-d v{VERSION} released with enhanced stability and performance"
    
    # Add after "Recent Announcements"
    content = re.sub(
        r'(### Recent Announcements\n)',
        f'\\1{new_announcement}\n',
        content
    )
    
    # Write the updated content
    with open(WHATS_NEW_FILE, 'w') as f:
        f.write(content)
    
    print(f"✅ Successfully updated {WHATS_NEW_FILE}")
    print(f"  - Current release: v{VERSION} ({RELEASE_DATE})")
    print(f"  - Next release: v{next_version} ({next_date_iso})")
    
except Exception as e:
    print(f"❌ Error updating file: {e}", file=sys.stderr)
    sys.exit(1)
EOF

echo -e "${GREEN}✅ What's New section updated successfully${NC}"