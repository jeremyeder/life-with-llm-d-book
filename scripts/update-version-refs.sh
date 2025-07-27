#!/bin/bash
# Update version references throughout the book

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
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔄 Updating version references throughout the book to v${VERSION}${NC}"

# Files to update (excluding auto-generated and temporary files)
FILES_TO_UPDATE=(
    "docs/02-installation-setup.md"
    "docs/03-understanding-architecture.md"
    "docs/04-data-scientist-workflows.md"
    "docs/appendix/shared-config.md"
    "README.md"
)

UPDATES_MADE=0
TOTAL_FILES_CHECKED=0

# Function to update version references in a file
update_file_versions() {
    local file="$1"
    local updates=0
    
    if [ ! -f "$file" ]; then
        echo -e "${YELLOW}    ⚠️  File not found: $file${NC}"
        return 0
    fi
    
    TOTAL_FILES_CHECKED=$((TOTAL_FILES_CHECKED + 1))
    
    # Create backup
    cp "$file" "${file}.backup"
    
    # Update common version patterns
    
    # llm-d version references
    if sed -i "s/llm-d v[0-9]\+\.[0-9]\+\.[0-9]\+/llm-d v${VERSION}/g" "$file" 2>/dev/null; then
        local count=$(diff "${file}.backup" "$file" 2>/dev/null | grep -c "^[<>]" || echo "0")
        count=$(echo "$count" | head -1 | tr -d '\n')  # Ensure single line, no newlines
        if [ "${count:-0}" -gt 0 ] 2>/dev/null; then
            updates=$((updates + count / 2))  # Each change shows as 2 lines in diff
        fi
    fi
    
    # Container image tags (if any reference specific versions)
    if sed -i "s/:v[0-9]\+\.[0-9]\+\.[0-9]\+/:v${VERSION}/g" "$file" 2>/dev/null; then
        local count=$(diff "${file}.backup" "$file" 2>/dev/null | grep -c "image.*:v" || echo "0")
        count=$(echo "$count" | head -1 | tr -d '\n')  # Ensure single line, no newlines
        if [ "${count:-0}" -gt 0 ] 2>/dev/null; then
            updates=$((updates + count))
        fi
    fi
    
    # Helm chart version references
    if sed -i "s/version: [0-9]\+\.[0-9]\+\.[0-9]\+/version: ${VERSION}/g" "$file" 2>/dev/null; then
        local count=$(diff "${file}.backup" "$file" 2>/dev/null | grep -c "version:" || echo "0")
        count=$(echo "$count" | head -1 | tr -d '\n')  # Ensure single line, no newlines
        if [ "${count:-0}" -gt 0 ] 2>/dev/null; then
            updates=$((updates + count))
        fi
    fi
    
    # Release tag references
    if sed -i "s/releases\/tag\/v[0-9]\+\.[0-9]\+\.[0-9]\+/releases\/tag\/v${VERSION}/g" "$file" 2>/dev/null; then
        local count=$(diff "${file}.backup" "$file" 2>/dev/null | grep -c "releases/tag" || echo "0")
        count=$(echo "$count" | head -1 | tr -d '\n')  # Ensure single line, no newlines
        if [ "${count:-0}" -gt 0 ] 2>/dev/null; then
            updates=$((updates + count))
        fi
    fi
    
    if [ "$updates" -gt 0 ]; then
        echo -e "${GREEN}    ✅ $file - $updates update(s) made${NC}"
        UPDATES_MADE=$((UPDATES_MADE + updates))
        rm "${file}.backup"
    else
        echo -e "${BLUE}    📄 $file - no version references found${NC}"
        # Restore from backup if no changes
        mv "${file}.backup" "$file"
    fi
    
    return $updates
}

# Update package.json version if it exists
if [ -f "package.json" ]; then
    echo -e "${BLUE}  📦 Updating package.json version${NC}"
    if command -v jq > /dev/null 2>&1; then
        jq ".version = \"${VERSION}\"" package.json > package.json.tmp && mv package.json.tmp package.json
        echo -e "${GREEN}    ✅ package.json version updated${NC}"
        UPDATES_MADE=$((UPDATES_MADE + 1))
    else
        echo -e "${YELLOW}    ⚠️  jq not available, skipping package.json update${NC}"
    fi
fi

# Update documentation files
echo -e "${BLUE}  📚 Updating documentation files${NC}"
for file in "${FILES_TO_UPDATE[@]}"; do
    update_file_versions "$file"
done

# Look for additional files that might contain version references
echo -e "${BLUE}  🔍 Scanning for additional version references${NC}"

# Search for potential version references in markdown files
POTENTIAL_FILES=$(find docs -name "*.md" -not -path "docs/release-notes/*" -not -name "00-whats-*.md" 2>/dev/null | head -20)

for file in $POTENTIAL_FILES; do
    if grep -q "v[0-9]\+\.[0-9]\+\.[0-9]\+" "$file" 2>/dev/null; then
        # Check if this file wasn't already processed
        if [[ ! " ${FILES_TO_UPDATE[@]} " =~ " ${file} " ]]; then
            echo -e "${YELLOW}    📄 Found potential version references in: $file${NC}"
            if grep -n "v[0-9]\+\.[0-9]\+\.[0-9]\+" "$file" | head -3; then
                echo -e "${BLUE}      Consider reviewing this file manually${NC}"
            fi
        fi
    fi
done

# Update example configurations
echo -e "${BLUE}  ⚙️  Updating example configurations${NC}"
CONFIG_FILES=$(find examples -name "*.yaml" -o -name "*.yml" 2>/dev/null | head -10)

for file in $CONFIG_FILES; do
    if [ -f "$file" ]; then
        update_file_versions "$file"
    fi
done

echo ""
echo -e "${BLUE}📊 Version Update Summary:${NC}"
echo -e "  Target version: v${VERSION}"
echo -e "  Files checked: $TOTAL_FILES_CHECKED"
echo -e "  Total updates made: $UPDATES_MADE"

if [ $UPDATES_MADE -gt 0 ]; then
    echo -e "${GREEN}✅ Version references updated successfully${NC}"
    
    echo ""
    echo -e "${BLUE}🔍 Manual Review Recommended:${NC}"
    echo -e "  - Check installation commands for correct version"
    echo -e "  - Verify Helm chart references"
    echo -e "  - Review any custom configurations"
    echo -e "  - Validate container image tags"
    
    exit 0
else
    echo -e "${YELLOW}ℹ️  No version references found to update${NC}"
    echo -e "${BLUE}💡 This might be normal if version references are minimal${NC}"
    exit 0
fi