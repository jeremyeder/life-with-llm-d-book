#!/bin/bash
# Technical claims validation for fact-checking against authoritative sources

set -e

ERRORS=0
WARNINGS=0
FILES_TO_CHECK="$*"

# Define temporary files for processing
TMP_DIR=$(mktemp -d)
CLAIMS_TEMP="$TMP_DIR/claims.txt"

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

echo -e "${BLUE}🔬 Technical Claims Validation${NC}"
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

# Function to validate Kubernetes version claims
validate_k8s_versions() {
    local file="$1"
    
    echo -e "${BLUE}   Checking Kubernetes version claims...${NC}"
    
    # Known Kubernetes release dates and lifecycle
    declare -A k8s_versions=(
        ["1.20"]="deprecated"
        ["1.21"]="deprecated"
        ["1.22"]="deprecated"
        ["1.23"]="deprecated"
        ["1.24"]="supported"
        ["1.25"]="supported"
        ["1.26"]="supported"
        ["1.27"]="supported"
        ["1.28"]="supported"
        ["1.29"]="supported"
        ["1.30"]="supported"
    )
    
    while IFS= read -r line; do
        if [[ "$line" =~ Kubernetes[[:space:]]+([0-9]+\.[0-9]+)\+ ]]; then
            local version="${BASH_REMATCH[1]}"
            local status="${k8s_versions[$version]:-unknown}"
            
            if [[ "$status" == "deprecated" ]]; then
                echo -e "${YELLOW}⚠️  $file: References deprecated Kubernetes version${NC}"
                echo "   Line: $line"
                echo "   Kubernetes $version is deprecated. Consider updating to 1.25+"
                WARNINGS=$((WARNINGS + 1))
            elif [[ "$status" == "unknown" ]]; then
                echo -e "${YELLOW}⚠️  $file: Unknown Kubernetes version${NC}"
                echo "   Line: $line"
                echo "   Kubernetes $version - verify this version exists"
                WARNINGS=$((WARNINGS + 1))
            fi
        fi
    done < <(grep -i "kubernetes.*[0-9]\+\.[0-9]\+" "$file" 2>/dev/null || true)
}

# Function to validate OpenShift version claims
validate_openshift_versions() {
    local file="$1"
    
    echo -e "${BLUE}   Checking OpenShift version claims...${NC}"
    
    # Known OpenShift versions and their support status
    declare -A ocp_versions=(
        ["4.10"]="deprecated"
        ["4.11"]="deprecated"
        ["4.12"]="supported"
        ["4.13"]="supported"
        ["4.14"]="supported"
        ["4.15"]="supported"
        ["4.16"]="supported"
    )
    
    while IFS= read -r line; do
        if [[ "$line" =~ OpenShift[[:space:]]+([0-9]+\.[0-9]+)\+ ]]; then
            local version="${BASH_REMATCH[1]}"
            local status="${ocp_versions[$version]:-unknown}"
            
            if [[ "$status" == "deprecated" ]]; then
                echo -e "${YELLOW}⚠️  $file: References deprecated OpenShift version${NC}"
                echo "   Line: $line"
                echo "   OpenShift $version may be deprecated. Consider updating to 4.12+"
                WARNINGS=$((WARNINGS + 1))
            elif [[ "$status" == "unknown" ]]; then
                echo -e "${YELLOW}⚠️  $file: Unknown OpenShift version${NC}"
                echo "   Line: $line"
                echo "   OpenShift $version - verify this version exists"
                WARNINGS=$((WARNINGS + 1))
            fi
        fi
    done < <(grep -i "openshift.*[0-9]\+\.[0-9]\+" "$file" 2>/dev/null || true)
}

# Function to validate NVIDIA software versions
validate_nvidia_versions() {
    local file="$1"
    
    echo -e "${BLUE}   Checking NVIDIA software versions...${NC}"
    
    # Check for CUDA versions
    while IFS= read -r line; do
        if [[ "$line" =~ CUDA[[:space:]]+([0-9]+\.[0-9]+) ]]; then
            local version="${BASH_REMATCH[1]}"
            
            # CUDA versions older than 11.0 are quite old
            if [[ $(echo "$version < 11.0" | bc -l 2>/dev/null || echo 0) -eq 1 ]]; then
                echo -e "${YELLOW}⚠️  $file: References old CUDA version${NC}"
                echo "   Line: $line"
                echo "   CUDA $version is quite old. Consider updating examples to CUDA 11.8+ or 12.x"
                WARNINGS=$((WARNINGS + 1))
            fi
        fi
    done < <(grep -i "cuda.*[0-9]\+\.[0-9]\+" "$file" 2>/dev/null || true)
    
    # Check for GPU Operator versions
    while IFS= read -r line; do
        if [[ "$line" =~ (GPU[[:space:]]+Operator|nvidia-operator).*([0-9]+\.[0-9]+\.[0-9]+) ]]; then
            local version="${BASH_REMATCH[2]}"
            
            # GPU Operator versions before 23.0.0 are quite old
            if [[ $(echo "$version" | cut -d. -f1) -lt 23 ]]; then
                echo -e "${YELLOW}⚠️  $file: References old GPU Operator version${NC}"
                echo "   Line: $line"
                echo "   GPU Operator $version is quite old. Consider updating to 23.9.0+"
                WARNINGS=$((WARNINGS + 1))
            fi
        fi
    done < <(grep -i "gpu.*operator\|nvidia.*operator" "$file" 2>/dev/null || true)
}

# Function to validate container runtime claims
validate_container_runtime_claims() {
    local file="$1"
    
    echo -e "${BLUE}   Checking container runtime claims...${NC}"
    
    # Check for deprecated container runtimes
    while IFS= read -r line; do
        if [[ "$line" =~ dockershim ]]; then
            echo -e "${RED}❌ $file: References deprecated dockershim${NC}"
            echo "   Line: $line"
            echo "   dockershim was removed in Kubernetes 1.24. Use containerd or CRI-O"
            ERRORS=$((ERRORS + 1))
        fi
        
        if [[ "$line" =~ Docker.*runtime ]] && [[ "$line" =~ Kubernetes ]]; then
            echo -e "${YELLOW}⚠️  $file: Docker runtime reference may need clarification${NC}"
            echo "   Line: $line"
            echo "   Clarify if this refers to Docker Engine or containerd"
            WARNINGS=$((WARNINGS + 1))
        fi
    done < <(grep -i -E "dockershim|docker.*runtime" "$file" 2>/dev/null || true)
}

# Function to validate model architecture claims
validate_model_architecture_claims() {
    local file="$1"
    
    echo -e "${BLUE}   Checking model architecture claims...${NC}"
    
    # Known model architectures and parameter counts
    declare -A model_params=(
        ["llama-3.1-8b"]="8"
        ["llama-3.1-70b"]="70"
        ["llama-3.1-405b"]="405"
        ["mistral-7b"]="7"
        ["codellama-13b"]="13"
        ["codellama-34b"]="34"
    )
    
    # Check for parameter count claims
    while IFS= read -r line; do
        for model in "${!model_params[@]}"; do
            local expected_params="${model_params[$model]}"
            
            if [[ "$line" =~ $model ]] && [[ "$line" =~ ([0-9]+)B[[:space:]]*(parameter|param) ]]; then
                local claimed_params="${BASH_REMATCH[1]}"
                
                if [[ "$claimed_params" != "$expected_params" ]]; then
                    echo -e "${RED}❌ $file: Incorrect parameter count for model${NC}"
                    echo "   Line: $line"
                    echo "   $model has ${expected_params}B parameters, not ${claimed_params}B"
                    ERRORS=$((ERRORS + 1))
                fi
            fi
        done
    done < <(grep -i -E "(llama|mistral|codellama).*[0-9]+B.*param" "$file" 2>/dev/null || true)
}

# Function to validate networking and protocol claims
validate_networking_claims() {
    local file="$1"
    
    echo -e "${BLUE}   Checking networking and protocol claims...${NC}"
    
    # Check for protocol specifications
    while IFS= read -r line; do
        # Check for deprecated API versions
        if [[ "$line" =~ apiVersion:[[:space:]]*([^[:space:]]+) ]]; then
            local api_version="${BASH_REMATCH[1]}"
            
            case "$api_version" in
                "extensions/v1beta1")
                    echo -e "${RED}❌ $file: Deprecated API version${NC}"
                    echo "   Line: $line"
                    echo "   extensions/v1beta1 is deprecated. Use apps/v1"
                    ERRORS=$((ERRORS + 1))
                    ;;
                "networking.k8s.io/v1beta1")
                    echo -e "${YELLOW}⚠️  $file: Beta API version${NC}"
                    echo "   Line: $line"
                    echo "   Consider using stable v1 version if available"
                    WARNINGS=$((WARNINGS + 1))
                    ;;
            esac
        fi
        
        # Check for HTTP/3 or QUIC references
        if [[ "$line" =~ (HTTP/3|QUIC) ]] && [[ "$line" =~ (production|stable) ]]; then
            echo -e "${YELLOW}⚠️  $file: HTTP/3/QUIC production readiness${NC}"
            echo "   Line: $line"
            echo "   Verify HTTP/3/QUIC support in your infrastructure"
            WARNINGS=$((WARNINGS + 1))
        fi
    done < <(grep -E "apiVersion:|HTTP/3|QUIC" "$file" 2>/dev/null || true)
}

# Function to validate security and compliance claims
validate_security_claims() {
    local file="$1"
    
    echo -e "${BLUE}   Checking security and compliance claims...${NC}"
    
    # Check for Pod Security Standards references
    while IFS= read -r line; do
        if [[ "$line" =~ PodSecurityPolicy ]]; then
            echo -e "${RED}❌ $file: References deprecated PodSecurityPolicy${NC}"
            echo "   Line: $line"
            echo "   PodSecurityPolicy is deprecated. Use Pod Security Standards"
            ERRORS=$((ERRORS + 1))
        fi
        
        if [[ "$line" =~ (privileged|hostNetwork|hostPID).*true ]] && [[ "$line" =~ production ]]; then
            echo -e "${YELLOW}⚠️  $file: Privileged settings in production${NC}"
            echo "   Line: $line"
            echo "   Review privileged settings for production security"
            WARNINGS=$((WARNINGS + 1))
        fi
    done < <(grep -i -E "podsecuritypolicy|privileged.*true|hostnetwork.*true|hostpid.*true" "$file" 2>/dev/null || true)
}

# Function to validate storage claims
validate_storage_claims() {
    local file="$1"
    
    echo -e "${BLUE}   Checking storage claims...${NC}"
    
    # Check for storage class references
    while IFS= read -r line; do
        if [[ "$line" =~ storageClass:[[:space:]]*[\"']?([^\"'[:space:]]+)[\"']? ]]; then
            local storage_class="${BASH_REMATCH[1]}"
            
            # Check for potentially non-standard storage classes
            case "$storage_class" in
                "fast-ssd"|"nvme-ssd"|"premium-storage")
                    echo -e "${YELLOW}⚠️  $file: Custom storage class reference${NC}"
                    echo "   Line: $line"
                    echo "   Verify storage class '$storage_class' exists in target clusters"
                    WARNINGS=$((WARNINGS + 1))
                    ;;
            esac
        fi
        
        # Check for unrealistic storage sizes
        if [[ "$line" =~ ([0-9]+)(Ti|TB)[[:space:]]*(storage|disk|volume) ]]; then
            local size="${BASH_REMATCH[1]}"
            local unit="${BASH_REMATCH[2]}"
            
            if [[ $size -gt 10 ]]; then
                echo -e "${YELLOW}⚠️  $file: Large storage claim${NC}"
                echo "   Line: $line"
                echo "   ${size}${unit} is quite large. Verify this is necessary"
                WARNINGS=$((WARNINGS + 1))
            fi
        fi
    done < <(grep -i -E "storageclass|[0-9]+(ti|tb).*storage" "$file" 2>/dev/null || true)
}

# Function to validate monitoring and observability claims
validate_monitoring_claims() {
    local file="$1"
    
    echo -e "${BLUE}   Checking monitoring and observability claims...${NC}"
    
    # Check for Prometheus/Grafana version claims
    while IFS= read -r line; do
        if [[ "$line" =~ Prometheus.*([0-9]+\.[0-9]+) ]]; then
            local version="${BASH_REMATCH[1]}"
            
            # Prometheus versions before 2.40 are quite old
            if [[ $(echo "$version < 2.40" | bc -l 2>/dev/null || echo 0) -eq 1 ]]; then
                echo -e "${YELLOW}⚠️  $file: Old Prometheus version reference${NC}"
                echo "   Line: $line"
                echo "   Prometheus $version is quite old. Consider updating to 2.45+"
                WARNINGS=$((WARNINGS + 1))
            fi
        fi
        
        # Check for metric naming conventions
        if [[ "$line" =~ prometheus.*metric.*[A-Z] ]]; then
            echo -e "${YELLOW}⚠️  $file: Prometheus metric naming${NC}"
            echo "   Line: $line"
            echo "   Prometheus metrics should use snake_case, not camelCase"
            WARNINGS=$((WARNINGS + 1))
        fi
    done < <(grep -i "prometheus\|grafana" "$file" 2>/dev/null || true)
}

# Function to validate container image references
validate_container_images() {
    local file="$1"
    
    echo -e "${BLUE}   Checking container image references...${NC}"
    
    while IFS= read -r line; do
        # Check for :latest tag usage
        if [[ "$line" =~ image:.*:latest ]] && [[ ! "$line" =~ (development|dev|test) ]]; then
            echo -e "${YELLOW}⚠️  $file: Using :latest tag in production context${NC}"
            echo "   Line: $line"
            echo "   Avoid :latest tags in production. Use specific versions"
            WARNINGS=$((WARNINGS + 1))
        fi
        
        # Check for unversioned images
        if [[ "$line" =~ image:[[:space:]]*[\"']?([^:\"'[:space:]]+)[\"']?$ ]]; then
            local image="${BASH_REMATCH[1]}"
            if [[ ! "$image" =~ @ ]]; then  # Not using digest
                echo -e "${YELLOW}⚠️  $file: Unversioned container image${NC}"
                echo "   Line: $line"
                echo "   Consider specifying image version: $image:version"
                WARNINGS=$((WARNINGS + 1))
            fi
        fi
    done < <(grep -E "image:" "$file" 2>/dev/null || true)
}

# Main validation loop
for file in $FILES_TO_CHECK; do
    if [[ ! -f "$file" ]]; then
        continue
    fi
    
    echo -e "${BLUE}🔍 Validating: $file${NC}"
    
    # Run all technical claim validations
    validate_k8s_versions "$file"
    validate_openshift_versions "$file"
    validate_nvidia_versions "$file"
    validate_container_runtime_claims "$file"
    validate_model_architecture_claims "$file"
    validate_networking_claims "$file"
    validate_security_claims "$file"
    validate_storage_claims "$file"
    validate_monitoring_claims "$file"
    validate_container_images "$file"
    
    echo ""
done

# Final summary
echo "=============================================="
echo -e "${BLUE}📊 Technical Claims Validation Summary${NC}"
echo "=============================================="

if [[ $ERRORS -eq 0 ]] && [[ $WARNINGS -eq 0 ]]; then
    echo -e "${GREEN}🎉 All technical claims validated successfully!${NC}"
    echo -e "${GREEN}✅ No factual errors or outdated information found${NC}"
    exit 0
elif [[ $ERRORS -eq 0 ]]; then
    echo -e "${YELLOW}⚠️  $WARNINGS warning(s) found${NC}"
    echo -e "${YELLOW}💡 Review warnings for technical accuracy improvements${NC}"
    exit 0
else
    echo -e "${RED}❌ $ERRORS error(s) and $WARNINGS warning(s) found${NC}"
    echo -e "${RED}🔧 Fix technical claim errors before proceeding${NC}"
    echo ""
    echo -e "${BLUE}💡 Common fixes:${NC}"
    echo "   - Update deprecated Kubernetes/OpenShift versions"
    echo "   - Replace deprecated API versions and features"
    echo "   - Verify model parameter counts and specifications"
    echo "   - Update software version references"
    echo "   - Review security and compliance claims"
    exit 1
fi