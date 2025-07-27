#!/bin/bash
# Simplified version reference updater for monthly releases

set -e

VERSION="$1"

if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version>" >&2
    exit 1
fi

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔄 Updating version references throughout the book to v${VERSION}${NC}"

# Update package.json version
echo -e "${BLUE}  📦 Updating package.json version${NC}"
if [ -f "package.json" ]; then
    sed -i 's/"version": "[0-9.]*"/"version": "'${VERSION}'"/' package.json
    echo -e "${GREEN}    ✅ package.json version updated${NC}"
fi

# Function to update a single file
update_file() {
    local file="$1"
    [ ! -f "$file" ] && return 0
    
    # Create backup
    cp "$file" "${file}.backup"
    
    # Update version patterns
    local updates=0
    
    # llm-d version references
    if sed -i "s/llm-d v[0-9]\+\.[0-9]\+\.[0-9]\+/llm-d v${VERSION}/g" "$file" 2>/dev/null; then
        local count=$(diff "${file}.backup" "$file" 2>/dev/null | grep -c "llm-d v" || echo "0")
        updates=$((updates + count))
    fi
    
    # Container image tags
    if sed -i "s/:v[0-9]\+\.[0-9]\+\.[0-9]\+/:v${VERSION}/g" "$file" 2>/dev/null; then
        local count=$(diff "${file}.backup" "$file" 2>/dev/null | grep -c ":v" || echo "0")
        updates=$((updates + count))
    fi
    
    # Clean up
    if [ "$updates" -gt 0 ]; then
        echo -e "${GREEN}    ✅ $(basename "$file") - $updates update(s) made${NC}"
        rm "${file}.backup"
    else
        mv "${file}.backup" "$file"
    fi
}

# Update key documentation files
echo -e "${BLUE}  📚 Updating documentation files${NC}"
for file in docs/*.md docs/appendix/*.md README.md; do
    [ -f "$file" ] && update_file "$file"
done

# Update example configurations  
echo -e "${BLUE}  ⚙️  Updating example configurations${NC}"
for file in examples/chapter-05-sre-operations/*.yaml; do
    [ -f "$file" ] && update_file "$file"
done

echo -e "${GREEN}✅ Version reference update complete${NC}"