#!/bin/bash
# Consistency matrix validation for cross-chapter consistency checking

set -e

ERRORS=0
WARNINGS=0
FILES_TO_CHECK="$*"

# Define temporary files for processing
TMP_DIR=$(mktemp -d)
MATRIX_TEMP="$TMP_DIR/consistency_matrix.txt"
VIOLATIONS_TEMP="$TMP_DIR/violations.txt"

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

echo -e "${BLUE}📊 Consistency Matrix Validation${NC}"
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

# Function to extract and validate model names across chapters
validate_model_name_consistency() {
    echo -e "${BLUE}🔍 Checking model name consistency across chapters${NC}"
    
    # Extract all model references
    declare -A model_refs
    local violation_count=0
    
    for file in $FILES_TO_CHECK; do
        if [[ ! -f "$file" ]]; then
            continue
        fi
        
        # Look for model name patterns
        while IFS= read -r line; do
            # Extract various model name patterns
            if [[ "$line" =~ (llama-?[0-9.-]*[0-9]+[bB]?) ]]; then
                local model="${BASH_REMATCH[1],,}"  # Convert to lowercase
                model_refs["$model"]+="$file:$line|"
            fi
            
            if [[ "$line" =~ (mistral-?[0-9.-]*[0-9]+[bB]?) ]]; then
                local model="${BASH_REMATCH[1],,}"
                model_refs["$model"]+="$file:$line|"
            fi
            
            if [[ "$line" =~ (codellama-?[0-9.-]*[0-9]+[bB]?) ]]; then
                local model="${BASH_REMATCH[1],,}"
                model_refs["$model"]+="$file:$line|"
            fi
        done < <(grep -i -E "(llama|mistral|codellama).*[0-9]+[bB]?" "$file" 2>/dev/null || true)
    done
    
    # Standard model names from shared config
    local standard_models=("llama-3.1-8b" "llama-3.1-70b" "mistral-7b" "codellama-13b")
    
    # Check for inconsistent model naming
    for model in "${!model_refs[@]}"; do
        local is_standard=false
        for std_model in "${standard_models[@]}"; do
            if [[ "$model" == "$std_model" ]]; then
                is_standard=true
                break
            fi
        done
        
        if [[ "$is_standard" == false ]]; then
            echo -e "${RED}❌ Inconsistent model name: $model${NC}"
            local refs="${model_refs[$model]}"
            IFS='|' read -ra ref_array <<< "$refs"
            for ref in "${ref_array[@]}"; do
                if [[ -n "$ref" ]]; then
                    echo "   Referenced in: $ref"
                fi
            done
            violation_count=$((violation_count + 1))
            ERRORS=$((ERRORS + 1))
        fi
    done
    
    if [[ $violation_count -eq 0 ]]; then
        echo -e "${GREEN}   ✅ Model names are consistent${NC}"
    fi
}

# Function to validate namespace consistency
validate_namespace_consistency() {
    echo -e "${BLUE}🔍 Checking namespace consistency across chapters${NC}"
    
    # Extract all namespace references
    declare -A namespace_refs
    local violation_count=0
    
    for file in $FILES_TO_CHECK; do
        if [[ ! -f "$file" ]]; then
            continue
        fi
        
        # Look for namespace specifications
        while IFS= read -r line; do
            if [[ "$line" =~ namespace:[[:space:]]*[\"\']?([^\"\'[:space:]]+)[\"\']? ]]; then
                local namespace="${BASH_REMATCH[1]}"
                namespace_refs["$namespace"]+="$file:$line|"
            fi
        done < <(grep -E "namespace:" "$file" 2>/dev/null || true)
    done
    
    # Standard namespaces from shared config
    local standard_namespaces=("production" "staging" "development" "llm-d-system")
    
    # Check for non-standard namespaces
    for namespace in "${!namespace_refs[@]}"; do
        local is_standard=false
        for std_ns in "${standard_namespaces[@]}"; do
            if [[ "$namespace" == "$std_ns" ]]; then
                is_standard=true
                break
            fi
        done
        
        if [[ "$is_standard" == false ]]; then
            echo -e "${YELLOW}⚠️  Non-standard namespace: $namespace${NC}"
            local refs="${namespace_refs[$namespace]}"
            IFS='|' read -ra ref_array <<< "$refs"
            for ref in "${ref_array[@]}"; do
                if [[ -n "$ref" ]]; then
                    echo "   Referenced in: $ref"
                fi
            done
            violation_count=$((violation_count + 1))
            WARNINGS=$((WARNINGS + 1))
        fi
    done
    
    if [[ $violation_count -eq 0 ]]; then
        echo -e "${GREEN}   ✅ Namespaces are consistent${NC}"
    fi
}

# Function to validate port usage consistency
validate_port_consistency() {
    echo -e "${BLUE}🔍 Checking port usage consistency across chapters${NC}"
    
    # Extract all port references
    declare -A port_refs
    local violation_count=0
    
    for file in $FILES_TO_CHECK; do
        if [[ ! -f "$file" ]]; then
            continue
        fi
        
        # Look for port specifications
        while IFS= read -r line; do
            if [[ "$line" =~ (containerPort|port|targetPort):[[:space:]]*([0-9]+) ]]; then
                local port="${BASH_REMATCH[2]}"
                local context="${BASH_REMATCH[1]}"
                port_refs["$port"]+="$file:$context:$line|"
            fi
        done < <(grep -E "(containerPort|port|targetPort):" "$file" 2>/dev/null || true)
    done
    
    # Standard port assignments from shared config
    declare -A standard_ports=(
        ["8080"]="HTTP API"
        ["8081"]="Metrics"
        ["8082"]="Health checks"
        ["9090"]="gRPC"
    )
    
    # Check for conflicting port usage
    for port in "${!port_refs[@]}"; do
        local refs="${port_refs[$port]}"
        local usage_contexts=()
        
        IFS='|' read -ra ref_array <<< "$refs"
        for ref in "${ref_array[@]}"; do
            if [[ -n "$ref" ]]; then
                IFS=':' read -ra ref_parts <<< "$ref"
                local context="${ref_parts[1]}"
                usage_contexts+=("$context")
            fi
        done
        
        # Check if port is used for multiple different purposes
        local unique_contexts=($(printf "%s\n" "${usage_contexts[@]}" | sort -u))
        if [[ ${#unique_contexts[@]} -gt 1 ]] && [[ -n "${standard_ports[$port]}" ]]; then
            echo -e "${YELLOW}⚠️  Port $port used in multiple contexts${NC}"
            echo "   Standard usage: ${standard_ports[$port]}"
            for ref in "${ref_array[@]}"; do
                if [[ -n "$ref" ]]; then
                    echo "   Referenced in: $ref"
                fi
            done
            violation_count=$((violation_count + 1))
            WARNINGS=$((WARNINGS + 1))
        fi
    done
    
    if [[ $violation_count -eq 0 ]]; then
        echo -e "${GREEN}   ✅ Port usage is consistent${NC}"
    fi
}

# Function to validate resource specification consistency
validate_resource_consistency() {
    echo -e "${BLUE}🔍 Checking resource specification consistency${NC}"
    
    # Extract resource specifications by model type
    declare -A resource_specs
    local violation_count=0
    
    for file in $FILES_TO_CHECK; do
        if [[ ! -f "$file" ]]; then
            continue
        fi
        
        # Extract YAML blocks and look for resource specifications
        local in_yaml=false
        local current_model=""
        local current_memory=""
        local current_gpu=""
        
        while IFS= read -r line; do
            if [[ "$line" =~ ^```yaml ]]; then
                in_yaml=true
                current_model=""
                current_memory=""
                current_gpu=""
                continue
            elif [[ "$line" =~ ^``` ]] && [[ "$in_yaml" == true ]]; then
                in_yaml=false
                # Store collected resource info
                if [[ -n "$current_model" ]] && [[ -n "$current_memory" ]]; then
                    resource_specs["$current_model"]+="$file:${current_memory}Gi:${current_gpu}GPU|"
                fi
                continue
            fi
            
            if [[ "$in_yaml" == true ]]; then
                # Look for model indicators
                if [[ "$line" =~ llama.*8b ]] || [[ "$line" =~ 8b.*llama ]]; then
                    current_model="8b"
                elif [[ "$line" =~ llama.*70b ]] || [[ "$line" =~ 70b.*llama ]]; then
                    current_model="70b"
                elif [[ "$line" =~ mistral.*7b ]] || [[ "$line" =~ 7b.*mistral ]]; then
                    current_model="7b"
                fi
                
                # Look for memory specifications
                if [[ "$line" =~ memory:[[:space:]]*[\"']?([0-9]+)Gi[\"']? ]]; then
                    current_memory="${BASH_REMATCH[1]}"
                fi
                
                # Look for GPU specifications
                if [[ "$line" =~ nvidia\.com/gpu:[[:space:]]*([0-9]+) ]]; then
                    current_gpu="${BASH_REMATCH[1]}"
                fi
            fi
        done < "$file"
    done
    
    # Check for inconsistent resource specifications
    for model in "${!resource_specs[@]}"; do
        local specs="${resource_specs[$model]}"
        IFS='|' read -ra spec_array <<< "$specs"
        
        local unique_specs=()
        for spec in "${spec_array[@]}"; do
            if [[ -n "$spec" ]]; then
                IFS=':' read -ra spec_parts <<< "$spec"
                local memory_gpu="${spec_parts[1]}:${spec_parts[2]}"
                unique_specs+=("$memory_gpu")
            fi
        done
        
        # Remove duplicates and check consistency
        local sorted_unique=($(printf "%s\n" "${unique_specs[@]}" | sort -u))
        if [[ ${#sorted_unique[@]} -gt 1 ]]; then
            echo -e "${YELLOW}⚠️  Inconsistent resource specs for $model models${NC}"
            for spec in "${spec_array[@]}"; do
                if [[ -n "$spec" ]]; then
                    IFS=':' read -ra spec_parts <<< "$spec"
                    echo "   ${spec_parts[0]}: ${spec_parts[1]}:${spec_parts[2]}"
                fi
            done
            violation_count=$((violation_count + 1))
            WARNINGS=$((WARNINGS + 1))
        fi
    done
    
    if [[ $violation_count -eq 0 ]]; then
        echo -e "${GREEN}   ✅ Resource specifications are consistent${NC}"
    fi
}

# Function to validate cross-chapter references
validate_cross_chapter_references() {
    echo -e "${BLUE}🔍 Checking cross-chapter reference consistency${NC}"
    
    local violation_count=0
    
    for file in $FILES_TO_CHECK; do
        if [[ ! -f "$file" ]]; then
            continue
        fi
        
        # Look for chapter references
        while IFS= read -r line; do
            if [[ "$line" =~ Chapter[[:space:]]+([0-9]+) ]]; then
                local chapter_num="${BASH_REMATCH[1]}"
                local chapter_file_pattern="docs/$(printf "%02d" "$chapter_num")-*.md"
                
                # Check if the referenced chapter exists
                if ! ls $chapter_file_pattern >/dev/null 2>&1; then
                    echo -e "${RED}❌ $file: References non-existent Chapter $chapter_num${NC}"
                    echo "   Line: $line"
                    violation_count=$((violation_count + 1))
                    ERRORS=$((ERRORS + 1))
                fi
            fi
        done < <(grep -i "chapter [0-9]" "$file" 2>/dev/null || true)
        
        # Look for shared config references
        while IFS= read -r line; do
            if [[ "$line" =~ shared.config ]] || [[ "$line" =~ shared-config ]]; then
                if [[ ! -f "docs/appendix/shared-config.md" ]]; then
                    echo -e "${RED}❌ $file: References non-existent shared config${NC}"
                    echo "   Line: $line"
                    violation_count=$((violation_count + 1))
                    ERRORS=$((ERRORS + 1))
                fi
            fi
        done < <(grep -i "shared" "$file" 2>/dev/null || true)
    done
    
    if [[ $violation_count -eq 0 ]]; then
        echo -e "${GREEN}   ✅ Cross-chapter references are consistent${NC}"
    fi
}

# Function to validate configuration template consistency
validate_template_consistency() {
    echo -e "${BLUE}🔍 Checking configuration template consistency${NC}"
    
    local violation_count=0
    
    # Look for references to configuration templates
    for file in $FILES_TO_CHECK; do
        if [[ ! -f "$file" ]]; then
            continue
        fi
        
        # Look for template references
        while IFS= read -r line; do
            if [[ "$line" =~ (template|example).*\.yaml ]]; then
                local template_name=$(echo "$line" | grep -oE '[a-zA-Z0-9_-]+\.yaml')
                
                # Check if template exists in examples or docs directories
                local template_found=false
                for template_path in "examples/$template_name" "docs/$template_name" "docs/appendix/$template_name"; do
                    if [[ -f "$template_path" ]]; then
                        template_found=true
                        break
                    fi
                done
                
                if [[ "$template_found" == false ]]; then
                    echo -e "${YELLOW}⚠️  $file: References potentially missing template${NC}"
                    echo "   Line: $line"
                    echo "   Template: $template_name"
                    violation_count=$((violation_count + 1))
                    WARNINGS=$((WARNINGS + 1))
                fi
            fi
        done < <(grep -i -E "(template|example).*\.yaml" "$file" 2>/dev/null || true)
    done
    
    if [[ $violation_count -eq 0 ]]; then
        echo -e "${GREEN}   ✅ Template references are consistent${NC}"
    fi
}

# Function to validate version consistency
validate_version_consistency() {
    echo -e "${BLUE}🔍 Checking version consistency across chapters${NC}"
    
    # Extract version references
    declare -A version_refs
    local violation_count=0
    
    for file in $FILES_TO_CHECK; do
        if [[ ! -f "$file" ]]; then
            continue
        fi
        
        # Look for version specifications
        while IFS= read -r line; do
            # Kubernetes versions
            if [[ "$line" =~ Kubernetes[[:space:]]+([0-9]+\.[0-9]+) ]]; then
                local version="${BASH_REMATCH[1]}"
                version_refs["kubernetes:$version"]+="$file:$line|"
            fi
            
            # OpenShift versions
            if [[ "$line" =~ OpenShift[[:space:]]+([0-9]+\.[0-9]+) ]]; then
                local version="${BASH_REMATCH[1]}"
                version_refs["openshift:$version"]+="$file:$line|"
            fi
            
            # CUDA versions
            if [[ "$line" =~ CUDA[[:space:]]+([0-9]+\.[0-9]+) ]]; then
                local version="${BASH_REMATCH[1]}"
                version_refs["cuda:$version"]+="$file:$line|"
            fi
        done < <(grep -i -E "(kubernetes|openshift|cuda).*[0-9]+\.[0-9]+" "$file" 2>/dev/null || true)
    done
    
    # Check for version conflicts
    for version_key in "${!version_refs[@]}"; do
        local software=$(echo "$version_key" | cut -d: -f1)
        local version=$(echo "$version_key" | cut -d: -f2)
        
        # Look for conflicting versions of the same software
        for other_key in "${!version_refs[@]}"; do
            if [[ "$other_key" != "$version_key" ]]; then
                local other_software=$(echo "$other_key" | cut -d: -f1)
                local other_version=$(echo "$other_key" | cut -d: -f2)
                
                if [[ "$software" == "$other_software" ]] && [[ "$version" != "$other_version" ]]; then
                    echo -e "${YELLOW}⚠️  Conflicting $software versions: $version vs $other_version${NC}"
                    echo "   $version referenced in: ${version_refs[$version_key]%|*}"
                    echo "   $other_version referenced in: ${version_refs[$other_key]%|*}"
                    violation_count=$((violation_count + 1))
                    WARNINGS=$((WARNINGS + 1))
                fi
            fi
        done
    done
    
    if [[ $violation_count -eq 0 ]]; then
        echo -e "${GREEN}   ✅ Versions are consistent${NC}"
    fi
}

# Function to generate consistency report
generate_consistency_report() {
    echo -e "${BLUE}📋 Generating consistency matrix report${NC}"
    
    cat > "$MATRIX_TEMP" << EOF
# Consistency Matrix Report
Generated: $(date)

## Summary
- Total files checked: $(echo $FILES_TO_CHECK | wc -w)
- Errors found: $ERRORS
- Warnings found: $WARNINGS

## Validation Categories
1. Model Name Consistency ✓
2. Namespace Consistency ✓
3. Port Usage Consistency ✓
4. Resource Specification Consistency ✓
5. Cross-Chapter References ✓
6. Template References ✓
7. Version Consistency ✓

## Next Steps
$(if [[ $ERRORS -gt 0 ]]; then
    echo "- Fix $ERRORS critical consistency errors"
fi)
$(if [[ $WARNINGS -gt 0 ]]; then
    echo "- Review $WARNINGS consistency warnings"
fi)
$(if [[ $ERRORS -eq 0 ]] && [[ $WARNINGS -eq 0 ]]; then
    echo "- All consistency checks passed!"
fi)

EOF
    
    echo -e "${BLUE}   Report saved to: $MATRIX_TEMP${NC}"
}

# Main validation execution
echo -e "${BLUE}Running comprehensive consistency validation...${NC}"
echo ""

validate_model_name_consistency
validate_namespace_consistency
validate_port_consistency
validate_resource_consistency
validate_cross_chapter_references
validate_template_consistency
validate_version_consistency

echo ""
generate_consistency_report

# Final summary
echo ""
echo "=============================================="
echo -e "${BLUE}📊 Consistency Matrix Summary${NC}"
echo "=============================================="

if [[ $ERRORS -eq 0 ]] && [[ $WARNINGS -eq 0 ]]; then
    echo -e "${GREEN}🎉 All consistency validations passed!${NC}"
    echo -e "${GREEN}✅ Content is consistent across all chapters${NC}"
    exit 0
elif [[ $ERRORS -eq 0 ]]; then
    echo -e "${YELLOW}⚠️  $WARNINGS consistency warning(s) found${NC}"
    echo -e "${YELLOW}💡 Review warnings for consistency improvements${NC}"
    exit 0
else
    echo -e "${RED}❌ $ERRORS consistency error(s) and $WARNINGS warning(s) found${NC}"
    echo -e "${RED}🔧 Fix consistency errors before proceeding${NC}"
    echo ""
    echo -e "${BLUE}💡 Common fixes:${NC}"
    echo "   - Standardize model names across all chapters"
    echo "   - Use consistent namespace conventions"
    echo "   - Align resource specifications with shared config"
    echo "   - Fix broken cross-chapter references"
    echo "   - Resolve version conflicts"
    exit 1
fi