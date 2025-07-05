#!/bin/bash
# Enhanced YAML syntax validation for comprehensive accuracy checking

set -e

ERRORS=0
WARNINGS=0
FILES_TO_CHECK="$*"

# Define temporary files for processing
TMP_DIR=$(mktemp -d)
YAML_TEMP="$TMP_DIR/extracted.yaml"
DUPLICATION_TEMP="$TMP_DIR/duplicates.txt"

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

echo -e "${BLUE}🔍 Enhanced YAML Syntax Validation${NC}"
echo "=============================================="

# Check if yamllint is available
if ! command -v yamllint &> /dev/null; then
    echo -e "${YELLOW}⚠️  yamllint not found. Installing...${NC}"
    pip install yamllint > /dev/null 2>&1 || {
        echo -e "${RED}❌ Failed to install yamllint. Please install manually: pip install yamllint${NC}"
        exit 1
    }
fi

# Set default files to check if none provided
if [ -z "$FILES_TO_CHECK" ]; then
    FILES_TO_CHECK=$(find docs -name "*.md" -type f)
fi

echo -e "${BLUE}📋 Files to validate:${NC}"
for file in $FILES_TO_CHECK; do
    echo "   - $file"
done
echo ""

# Function to extract YAML blocks from markdown
extract_yaml_blocks() {
    local file="$1"
    local output="$2"
    
    # Clear output file
    > "$output"
    
    # Extract YAML blocks between ```yaml and ```
    awk '
        /^```yaml/ { in_yaml = 1; next }
        /^```/ && in_yaml { in_yaml = 0; print "---"; next }
        in_yaml { print }
    ' "$file" >> "$output"
    
    # Extract YAML blocks between ```yml and ```
    awk '
        /^```yml/ { in_yaml = 1; next }
        /^```/ && in_yaml { in_yaml = 0; print "---"; next }
        in_yaml { print }
    ' "$file" >> "$output"
}

# Function to check for duplicate keys in YAML
check_duplicate_keys() {
    local yaml_file="$1"
    local source_file="$2"
    
    # Use yq to parse and check for duplicates
    if command -v yq &> /dev/null; then
        # Check each document separately
        local doc_count=0
        while IFS= read -r line; do
            if [[ "$line" == "---" ]]; then
                doc_count=$((doc_count + 1))
                continue
            fi
            
            # Look for duplicate keys in the current context
            if echo "$line" | grep -q "^[[:space:]]*[^[:space:]#]*:[[:space:]]*"; then
                local key=$(echo "$line" | sed 's/^[[:space:]]*//' | sed 's/:.*//')
                if [[ -n "$key" ]]; then
                    local key_count=$(grep -c "^[[:space:]]*$key[[:space:]]*:" "$yaml_file" || echo 0)
                    if [[ $key_count -gt 1 ]]; then
                        echo -e "${RED}❌ $source_file: Duplicate key '$key' found${NC}"
                        echo "   Line: $line"
                        ERRORS=$((ERRORS + 1))
                    fi
                fi
            fi
        done < "$yaml_file"
    else
        echo -e "${YELLOW}⚠️  yq not available, skipping duplicate key check${NC}"
        WARNINGS=$((WARNINGS + 1))
    fi
}

# Function to validate resource specifications
validate_resource_specs() {
    local yaml_file="$1"
    local source_file="$2"
    
    # Check for common resource specification issues
    
    # 1. Check for duplicate memory specifications
    local memory_count=$(grep -c "memory:" "$yaml_file" 2>/dev/null || echo 0)
    if [[ $memory_count -gt 0 ]]; then
        # Extract unique memory specifications per resource block
        awk '
            /resources:/ { in_resources = 1; next }
            /^[[:space:]]*[^[:space:]]/ && in_resources && !/^[[:space:]]*[[:space:]]/ { in_resources = 0 }
            in_resources && /memory:/ { 
                if (seen_memory) {
                    print "DUPLICATE_MEMORY: " $0
                }
                seen_memory = 1
            }
            !in_resources { seen_memory = 0 }
        ' "$yaml_file" | while read -r line; do
            if [[ "$line" == DUPLICATE_MEMORY:* ]]; then
                echo -e "${RED}❌ $source_file: Duplicate memory specification${NC}"
                echo "   Line: ${line#DUPLICATE_MEMORY: }"
                ERRORS=$((ERRORS + 1))
            fi
        done
    fi
    
    # 2. Check for invalid resource formats
    if grep -q "nvidia.com/gpu:[[:space:]]*[^0-9]" "$yaml_file" 2>/dev/null; then
        echo -e "${RED}❌ $source_file: Invalid GPU resource format${NC}"
        grep -n "nvidia.com/gpu:" "$yaml_file" | while read -r line; do
            echo "   Line: $line"
        done
        ERRORS=$((ERRORS + 1))
    fi
    
    # 3. Check for memory format consistency
    if grep -q "memory:[[:space:]]*[0-9]*[^GM]i" "$yaml_file" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  $source_file: Memory format may be inconsistent${NC}"
        echo "   Use standard formats: 16Gi, 32Gi, 64Gi, etc."
        WARNINGS=$((WARNINGS + 1))
    fi
}

# Function to validate port specifications
validate_port_specs() {
    local yaml_file="$1"
    local source_file="$2"
    
    # Check for non-standard port usage
    local standard_ports=(8080 8081 8082 9090)
    
    while IFS= read -r line; do
        if [[ "$line" =~ containerPort:[[:space:]]*([0-9]+) ]]; then
            local port="${BASH_REMATCH[1]}"
            local is_standard=false
            
            for std_port in "${standard_ports[@]}"; do
                if [[ "$port" == "$std_port" ]]; then
                    is_standard=true
                    break
                fi
            done
            
            if [[ "$is_standard" == false ]]; then
                echo -e "${YELLOW}⚠️  $source_file: Non-standard port $port${NC}"
                echo "   Standard ports: 8080 (HTTP), 8081 (metrics), 8082 (health), 9090 (gRPC)"
                echo "   Line: $line"
                WARNINGS=$((WARNINGS + 1))
            fi
        fi
    done < "$yaml_file"
}

# Function to validate namespace references
validate_namespace_refs() {
    local yaml_file="$1"
    local source_file="$2"
    
    # Standard namespaces from shared config
    local standard_namespaces=("production" "staging" "development" "llm-d-system")
    
    while IFS= read -r line; do
        if [[ "$line" =~ namespace:[[:space:]]*([^[:space:]]+) ]]; then
            local namespace="${BASH_REMATCH[1]}"
            local is_standard=false
            
            for std_ns in "${standard_namespaces[@]}"; do
                if [[ "$namespace" == "$std_ns" ]]; then
                    is_standard=true
                    break
                fi
            done
            
            if [[ "$is_standard" == false ]]; then
                echo -e "${YELLOW}⚠️  $source_file: Non-standard namespace '$namespace'${NC}"
                echo "   Standard namespaces: production, staging, development, llm-d-system"
                echo "   Line: $line"
                WARNINGS=$((WARNINGS + 1))
            fi
        fi
    done < "$yaml_file"
}

# Main validation loop
for file in $FILES_TO_CHECK; do
    if [[ ! -f "$file" ]]; then
        continue
    fi
    
    echo -e "${BLUE}🔍 Validating: $file${NC}"
    
    # Extract YAML blocks
    extract_yaml_blocks "$file" "$YAML_TEMP"
    
    # Skip if no YAML found
    if [[ ! -s "$YAML_TEMP" ]]; then
        echo "   No YAML blocks found"
        continue
    fi
    
    # Basic YAML syntax validation with yamllint
    if ! yamllint -c <(echo "extends: default"; echo "rules:"; echo "  line-length: disable") "$YAML_TEMP" >/dev/null 2>&1; then
        echo -e "${RED}❌ $file: YAML syntax errors detected${NC}"
        yamllint -c <(echo "extends: default"; echo "rules:"; echo "  line-length: disable") "$YAML_TEMP" | head -10
        ERRORS=$((ERRORS + 1))
    else
        echo -e "${GREEN}   ✅ Basic YAML syntax valid${NC}"
    fi
    
    # Check for duplicate keys
    check_duplicate_keys "$YAML_TEMP" "$file"
    
    # Validate resource specifications
    validate_resource_specs "$YAML_TEMP" "$file"
    
    # Validate port specifications  
    validate_port_specs "$YAML_TEMP" "$file"
    
    # Validate namespace references
    validate_namespace_refs "$YAML_TEMP" "$file"
    
    echo ""
done

# Final summary
echo "=============================================="
echo -e "${BLUE}📊 YAML Validation Summary${NC}"
echo "=============================================="

if [[ $ERRORS -eq 0 ]] && [[ $WARNINGS -eq 0 ]]; then
    echo -e "${GREEN}🎉 All YAML validations passed!${NC}"
    echo -e "${GREEN}✅ No syntax errors or consistency issues found${NC}"
    exit 0
elif [[ $ERRORS -eq 0 ]]; then
    echo -e "${YELLOW}⚠️  $WARNINGS warning(s) found${NC}"
    echo -e "${YELLOW}💡 Review warnings for consistency improvements${NC}"
    exit 0
else
    echo -e "${RED}❌ $ERRORS error(s) and $WARNINGS warning(s) found${NC}"
    echo -e "${RED}🔧 Fix errors before proceeding${NC}"
    echo ""
    echo -e "${BLUE}💡 Common fixes:${NC}"
    echo "   - Remove duplicate keys in YAML blocks"
    echo "   - Use standard resource formats (16Gi, not 16G)"
    echo "   - Use standard ports: 8080, 8081, 8082, 9090"
    echo "   - Use standard namespaces: production, staging, development, llm-d-system"
    exit 1
fi