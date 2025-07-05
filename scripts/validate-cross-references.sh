#!/bin/bash
# Cross-reference validation for comprehensive link and reference checking

set -e

ERRORS=0
WARNINGS=0
FILES_TO_CHECK="$*"

# Define temporary files for processing
TMP_DIR=$(mktemp -d)
LINK_TEMP="$TMP_DIR/links.txt"
REFS_TEMP="$TMP_DIR/refs.txt"

# Cleanup function
cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

# Color codes for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔗 Cross-Reference Validation${NC}"
echo "=============================================="

# Set default files to check if none provided
if [ -z "$FILES_TO_CHECK" ]; then
    FILES_TO_CHECK=$(find docs -name "*.md" -type f)
fi

echo -e "${BLUE}📋 Files to validate:${NC}"
for file in $FILES_TO_CHECK; do
    echo "   - $file"
done
echo ""

# Function to extract internal links from markdown
extract_internal_links() {
    local file="$1"
    local output="$2"
    
    # Clear output file
    > "$output"
    
    # Extract markdown links [text](./path/to/file.md)
    grep -oE '\[([^\]]+)\]\(([^)]+)\)' "$file" | while read -r link; do
        # Extract the URL part
        url=$(echo "$link" | sed 's/.*(\([^)]*\)).*/\1/')
        
        # Only process relative links (internal references)
        if [[ "$url" =~ ^\./ ]] || [[ "$url" =~ ^[^/:]+\.md ]] || [[ "$url" =~ ^../ ]]; then
            echo "$file|$url|$link" >> "$output"
        fi
    done
    
    # Extract reference-style links [text][ref]
    grep -oE '\[([^\]]+)\]\[([^\]]+)\]' "$file" | while read -r link; do
        ref=$(echo "$link" | sed 's/.*\[\([^\]]*\)\].*/\1/')
        echo "$file|ref:$ref|$link" >> "$output"
    done
}

# Function to validate file references
validate_file_references() {
    local file="$1"
    local link_file="$2"
    
    if [[ ! -s "$link_file" ]]; then
        return 0
    fi
    
    while IFS='|' read -r source_file url display_link; do
        # Skip reference-style links for now
        if [[ "$url" =~ ^ref: ]]; then
            continue
        fi
        
        # Resolve relative paths
        local base_dir=$(dirname "$source_file")
        local resolved_path
        
        if [[ "$url" =~ ^\./ ]]; then
            # Remove leading ./
            resolved_path="$base_dir/${url#./}"
        elif [[ "$url" =~ ^../ ]]; then
            # Handle ../ paths
            resolved_path="$base_dir/$url"
        else
            # Assume relative to current directory
            resolved_path="$base_dir/$url"
        fi
        
        # Normalize path (remove ./ and ../ properly)
        resolved_path=$(realpath -m "$resolved_path" 2>/dev/null || echo "$resolved_path")
        
        # Check if file exists
        if [[ ! -f "$resolved_path" ]]; then
            echo -e "${RED}❌ $source_file: Broken reference${NC}"
            echo "   Link: $display_link"
            echo "   Target: $resolved_path"
            echo "   Status: File not found"
            ERRORS=$((ERRORS + 1))
        fi
        
        # Check for fragment references (#section)
        if [[ "$url" =~ #(.+)$ ]]; then
            local fragment="${BASH_REMATCH[1]}"
            local target_file="${resolved_path%#*}"
            
            if [[ -f "$target_file" ]]; then
                # Check if the target section exists
                if ! grep -q "^#.*$fragment" "$target_file" 2>/dev/null; then
                    echo -e "${YELLOW}⚠️  $source_file: Fragment reference may be broken${NC}"
                    echo "   Link: $display_link"
                    echo "   Fragment: #$fragment"
                    echo "   Target: $target_file"
                    WARNINGS=$((WARNINGS + 1))
                fi
            fi
        fi
        
    done < "$link_file"
}

# Function to validate shared config references
validate_shared_config_refs() {
    local file="$1"
    
    # Check for references to shared config
    if grep -q "shared.config\|shared-config" "$file" 2>/dev/null; then
        # Verify the shared config file exists
        if [[ ! -f "docs/appendix/shared-config.md" ]]; then
            echo -e "${RED}❌ $file: References shared config but file doesn't exist${NC}"
            echo "   Expected: docs/appendix/shared-config.md"
            ERRORS=$((ERRORS + 1))
        fi
        
        # Check for specific shared config references
        while IFS= read -r line; do
            if [[ "$line" =~ shared-config\.md ]]; then
                local ref_line="$line"
                # Extract the link to validate
                if [[ "$ref_line" =~ \[([^\]]+)\]\(([^)]+)\) ]]; then
                    local link_url="${BASH_REMATCH[2]}"
                    local base_dir=$(dirname "$file")
                    local resolved_path="$base_dir/$link_url"
                    
                    if [[ ! -f "$resolved_path" ]]; then
                        echo -e "${RED}❌ $file: Broken shared config reference${NC}"
                        echo "   Link: $ref_line"
                        echo "   Target: $resolved_path"
                        ERRORS=$((ERRORS + 1))
                    fi
                fi
            fi
        done < <(grep -n "shared-config" "$file" 2>/dev/null || true)
    fi
}

# Function to validate chapter cross-references
validate_chapter_refs() {
    local file="$1"
    
    # Look for chapter references like "Chapter 4", "Chapter 11", etc.
    while IFS= read -r line; do
        if [[ "$line" =~ Chapter[[:space:]]+([0-9]+) ]]; then
            local chapter_num="${BASH_REMATCH[1]}"
            local chapter_file="docs/$(printf "%02d" "$chapter_num")-*.md"
            
            # Check if chapter file exists (using glob pattern)
            if ! ls $chapter_file >/dev/null 2>&1; then
                echo -e "${YELLOW}⚠️  $file: Reference to Chapter $chapter_num${NC}"
                echo "   Line: $line"
                echo "   Status: Chapter file pattern not found: $chapter_file"
                WARNINGS=$((WARNINGS + 1))
            fi
        fi
    done < <(grep -i "chapter [0-9]" "$file" 2>/dev/null || true)
}

# Function to validate appendix references
validate_appendix_refs() {
    local file="$1"
    
    # Look for appendix references
    while IFS= read -r line; do
        if [[ "$line" =~ Appendix[[:space:]]+([ABC]) ]]; then
            local appendix_letter="${BASH_REMATCH[1]}"
            local appendix_file
            
            case "$appendix_letter" in
                "A") appendix_file="docs/appendix/crd-reference.md" ;;
                "B") appendix_file="docs/appendix/command-reference.md" ;;
                "C") appendix_file="docs/appendix/configuration-templates.md" ;;
                *) continue ;;
            esac
            
            if [[ ! -f "$appendix_file" ]]; then
                echo -e "${RED}❌ $file: Broken appendix reference${NC}"
                echo "   Line: $line"
                echo "   Expected: $appendix_file"
                ERRORS=$((ERRORS + 1))
            fi
        fi
    done < <(grep -i "appendix [abc]" "$file" 2>/dev/null || true)
}

# Function to validate code example references
validate_code_refs() {
    local file="$1"
    
    # Look for references to code examples
    while IFS= read -r line; do
        if [[ "$line" =~ examples/([^[:space:]]+) ]]; then
            local example_path="${BASH_REMATCH[1]}"
            local full_path="examples/$example_path"
            
            if [[ ! -f "$full_path" ]] && [[ ! -d "$full_path" ]]; then
                echo -e "${YELLOW}⚠️  $file: Code example reference may be broken${NC}"
                echo "   Line: $line"
                echo "   Target: $full_path"
                echo "   Status: File or directory not found"
                WARNINGS=$((WARNINGS + 1))
            fi
        fi
    done < <(grep "examples/" "$file" 2>/dev/null || true)
}

# Function to validate image references
validate_image_refs() {
    local file="$1"
    
    # Look for image references
    while IFS= read -r line; do
        if [[ "$line" =~ !\[[^\]]*\]\(([^)]+)\) ]]; then
            local img_path="${BASH_REMATCH[1]}"
            local base_dir=$(dirname "$file")
            local resolved_path
            
            if [[ "$img_path" =~ ^\./ ]]; then
                resolved_path="$base_dir/${img_path#./}"
            elif [[ "$img_path" =~ ^/ ]]; then
                resolved_path="${img_path#/}"
            else
                resolved_path="$base_dir/$img_path"
            fi
            
            if [[ ! -f "$resolved_path" ]]; then
                echo -e "${YELLOW}⚠️  $file: Image reference may be broken${NC}"
                echo "   Line: $line"
                echo "   Target: $resolved_path"
                WARNINGS=$((WARNINGS + 1))
            fi
        fi
    done < <(grep "!\[" "$file" 2>/dev/null || true)
}

# Main validation loop
for file in $FILES_TO_CHECK; do
    if [[ ! -f "$file" ]]; then
        continue
    fi
    
    echo -e "${BLUE}🔍 Validating: $file${NC}"
    
    # Extract links from the file
    extract_internal_links "$file" "$LINK_TEMP"
    
    # Validate file references
    validate_file_references "$file" "$LINK_TEMP"
    
    # Validate shared config references
    validate_shared_config_refs "$file"
    
    # Validate chapter cross-references
    validate_chapter_refs "$file"
    
    # Validate appendix references
    validate_appendix_refs "$file"
    
    # Validate code example references
    validate_code_refs "$file"
    
    # Validate image references
    validate_image_refs "$file"
    
    echo ""
done

# Additional validation: Check for orphaned files
echo -e "${BLUE}🔍 Checking for orphaned files${NC}"
echo "=============================================="

# Find all .md files in docs/
all_docs=$(find docs -name "*.md" -type f)
referenced_files=()

# Extract all referenced files
for file in $all_docs; do
    while IFS= read -r line; do
        if [[ "$line" =~ \[([^\]]+)\]\(([^)]+)\) ]]; then
            local url="${BASH_REMATCH[2]}"
            if [[ "$url" =~ \.md$ ]] && [[ ! "$url" =~ ^http ]]; then
                local base_dir=$(dirname "$file")
                local resolved_path="$base_dir/$url"
                resolved_path=$(realpath -m "$resolved_path" 2>/dev/null || echo "$resolved_path")
                referenced_files+=("$resolved_path")
            fi
        fi
    done < <(grep -oE '\[([^\]]+)\]\(([^)]+)\)' "$file" 2>/dev/null || true)
done

# Check for orphaned files (files that are never referenced)
orphaned_count=0
for doc_file in $all_docs; do
    # Skip certain files that are expected to be orphaned
    if [[ "$doc_file" =~ (99-table-of-contents|00-forward|README)\.md$ ]]; then
        continue
    fi
    
    local is_referenced=false
    for ref_file in "${referenced_files[@]}"; do
        if [[ "$doc_file" == "$ref_file" ]]; then
            is_referenced=true
            break
        fi
    done
    
    if [[ "$is_referenced" == false ]]; then
        echo -e "${YELLOW}⚠️  Potentially orphaned file: $doc_file${NC}"
        echo "   This file may not be referenced by any other documentation"
        orphaned_count=$((orphaned_count + 1))
        WARNINGS=$((WARNINGS + 1))
    fi
done

if [[ $orphaned_count -eq 0 ]]; then
    echo -e "${GREEN}✅ No orphaned files found${NC}"
fi

# Final summary
echo ""
echo "=============================================="
echo -e "${BLUE}📊 Cross-Reference Validation Summary${NC}"
echo "=============================================="

if [[ $ERRORS -eq 0 ]] && [[ $WARNINGS -eq 0 ]]; then
    echo -e "${GREEN}🎉 All cross-reference validations passed!${NC}"
    echo -e "${GREEN}✅ No broken links or references found${NC}"
    exit 0
elif [[ $ERRORS -eq 0 ]]; then
    echo -e "${YELLOW}⚠️  $WARNINGS warning(s) found${NC}"
    echo -e "${YELLOW}💡 Review warnings for reference improvements${NC}"
    exit 0
else
    echo -e "${RED}❌ $ERRORS error(s) and $WARNINGS warning(s) found${NC}"
    echo -e "${RED}🔧 Fix broken references before proceeding${NC}"
    echo ""
    echo -e "${BLUE}💡 Common fixes:${NC}"
    echo "   - Update broken file paths"
    echo "   - Create missing referenced files"
    echo "   - Fix relative path references"
    echo "   - Update chapter and appendix references"
    exit 1
fi