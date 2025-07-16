#!/bin/bash
# Fetch llm-d release data from GitHub API

set -e

VERSION="$1"

if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version>" >&2
    exit 1
fi

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔍 Fetching release data for llm-d v${VERSION}${NC}" >&2

# Try to fetch release data from multiple llm-d repositories
REPOS=(
    "llm-d/llm-d"
    "llm-d/llm-d-deployer"
    "llm-d/llm-d-inference-scheduler"
    "llm-d/llm-d-kv-cache-manager"
    "llm-d/llm-d-benchmark"
)

RELEASE_DATA=""
FOUND_RELEASE=false

for repo in "${REPOS[@]}"; do
    echo -e "${BLUE}  Checking ${repo}...${NC}" >&2
    
    # Try to fetch release by tag
    RESPONSE=$(curl -s "https://api.github.com/repos/${repo}/releases/tags/v${VERSION}" 2>/dev/null || echo "")
    
    if [ -n "$RESPONSE" ] && echo "$RESPONSE" | jq -e '.tag_name' > /dev/null 2>&1; then
        echo -e "${GREEN}  ✅ Found release in ${repo}${NC}" >&2
        RELEASE_DATA="$RESPONSE"
        FOUND_RELEASE=true
        break
    fi
    
    # Try alternative tag formats
    for tag_format in "llm-d-${VERSION}" "${VERSION}" "v${VERSION}"; do
        RESPONSE=$(curl -s "https://api.github.com/repos/${repo}/releases/tags/${tag_format}" 2>/dev/null || echo "")
        
        if [ -n "$RESPONSE" ] && echo "$RESPONSE" | jq -e '.tag_name' > /dev/null 2>&1; then
            echo -e "${GREEN}  ✅ Found release in ${repo} with tag ${tag_format}${NC}" >&2
            RELEASE_DATA="$RESPONSE"
            FOUND_RELEASE=true
            break 2
        fi
    done
done

if [ "$FOUND_RELEASE" = false ]; then
    echo -e "${RED}  ❌ No release found for version ${VERSION}${NC}" >&2
    
    # Return minimal structure for manual processing
    cat << EOF
{
  "tag_name": "v${VERSION}",
  "name": "llm-d v${VERSION}",
  "published_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "body": "Release notes to be updated manually",
  "html_url": "https://github.com/llm-d/llm-d/releases",
  "draft": true,
  "prerelease": false
}
EOF
    exit 0
fi

# Process and clean the release data
echo "$RELEASE_DATA" | jq '{
  tag_name: .tag_name,
  name: .name,
  published_at: .published_at,
  body: .body,
  html_url: .html_url,
  draft: .draft,
  prerelease: .prerelease,
  assets: [.assets[]? | {name: .name, download_count: .download_count, size: .size}]
}'

echo -e "${GREEN}✅ Release data fetched successfully${NC}" >&2