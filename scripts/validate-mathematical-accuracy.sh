#!/bin/bash
# Mathematical accuracy validation for formulas, calculations, and technical claims

set -e

ERRORS=0
WARNINGS=0
FILES_TO_CHECK="$*"

# Define temporary files for processing
TMP_DIR=$(mktemp -d)
MATH_TEMP="$TMP_DIR/math_expressions.txt"
CALC_TEMP="$TMP_DIR/calculations.py"

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

echo -e "${BLUE}🧮 Mathematical Accuracy Validation${NC}"
echo "=============================================="

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ python3 not found. Required for mathematical validation.${NC}"
    exit 1
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

# Function to validate SLO calculations
validate_slo_calculations() {
    local file="$1"
    
    echo -e "${BLUE}   Checking SLO calculations...${NC}"
    
    # Check for SLO percentage to downtime calculations
    while IFS= read -r line; do
        # Look for patterns like "99.9% = X minutes downtime"
        if [[ "$line" =~ ([0-9]+\.?[0-9]*)%.*([0-9]+\.?[0-9]*)[[:space:]]*(minutes?|hours?|days?)[[:space:]]*/[[:space:]]*(month|year) ]]; then
            local percentage="${BASH_REMATCH[1]}"
            local downtime="${BASH_REMATCH[2]}"
            local unit="${BASH_REMATCH[3]}"
            local period="${BASH_REMATCH[4]}"
            
            # Calculate expected downtime
            local expected_downtime
            expected_downtime=$(python3 -c "
uptime = $percentage / 100.0
if '$period' == 'month':
    total_minutes = 30.44 * 24 * 60  # Average month
elif '$period' == 'year':
    total_minutes = 365.25 * 24 * 60  # Average year
else:
    total_minutes = 30.44 * 24 * 60  # Default to month

downtime_minutes = total_minutes * (1 - uptime)

if '$unit' == 'hours' or '$unit' == 'hour':
    downtime_in_unit = downtime_minutes / 60
elif '$unit' == 'days' or '$unit' == 'day':
    downtime_in_unit = downtime_minutes / (60 * 24)
else:
    downtime_in_unit = downtime_minutes

print(f'{downtime_in_unit:.1f}')
")
            
            # Compare with stated downtime (allow 5% tolerance)
            local tolerance_check
            tolerance_check=$(python3 -c "
expected = $expected_downtime
actual = $downtime
tolerance = abs(expected - actual) / expected
print('PASS' if tolerance <= 0.05 else 'FAIL')
print(f'Expected: {expected:.1f}, Actual: {actual:.1f}, Tolerance: {tolerance:.3f}')
" | head -1)
            
            if [[ "$tolerance_check" == "FAIL" ]]; then
                local details
                details=$(python3 -c "
expected = $expected_downtime
actual = $downtime
tolerance = abs(expected - actual) / expected
print(f'Expected: {expected:.1f}, Actual: {actual:.1f}, Difference: {tolerance:.1%}')
")
                echo -e "${RED}❌ $file: SLO calculation error${NC}"
                echo "   Line: $line"
                echo "   $details"
                ERRORS=$((ERRORS + 1))
            fi
        fi
        
        # Look for incorrect mathematical relationships in SLOs
        if [[ "$line" =~ 99\.9%.*43\.8[[:space:]]*minutes ]]; then
            echo -e "${RED}❌ $file: Incorrect SLO calculation${NC}"
            echo "   Line: $line"
            echo "   99.9% uptime = 43.2 minutes/month, not 43.8 minutes"
            ERRORS=$((ERRORS + 1))
        fi
        
    done < <(grep -i -E "(99\.[0-9]+%|9[0-9]\.[0-9]+%).*minutes?|hours?|days?" "$file" 2>/dev/null || true)
}

# Function to validate memory calculations
validate_memory_calculations() {
    local file="$1"
    
    echo -e "${BLUE}   Checking memory calculations...${NC}"
    
    # Look for model memory calculations
    while IFS= read -r line; do
        # Check for patterns like "70B model ~140GB memory"
        if [[ "$line" =~ ([0-9]+)B[[:space:]]*model.*~?([0-9]+)GB ]]; then
            local params="${BASH_REMATCH[1]}"
            local memory="${BASH_REMATCH[2]}"
            
            # Validate memory calculation (rough estimate: 2 bytes per parameter for FP16)
            local expected_memory
            expected_memory=$(python3 -c "
params_billions = $params
bytes_per_param = 2  # FP16
model_size_gb = (params_billions * 1e9 * bytes_per_param) / (1024**3)
# Add overhead for KV cache, activations, etc. (typically 50-100% overhead)
total_memory = model_size_gb * 1.8  # 80% overhead
print(f'{total_memory:.0f}')
")
            
            # Check if stated memory is reasonable (within 50% of calculation)
            local is_reasonable
            is_reasonable=$(python3 -c "
expected = $expected_memory
actual = $memory
ratio = actual / expected
print('PASS' if 0.5 <= ratio <= 2.0 else 'FAIL')
print(f'Expected: ~{expected}GB, Actual: {actual}GB, Ratio: {ratio:.2f}')
" | head -1)
            
            if [[ "$is_reasonable" == "FAIL" ]]; then
                local details
                details=$(python3 -c "
expected = $expected_memory
actual = $memory
ratio = actual / expected
print(f'Expected: ~{expected}GB, Actual: {actual}GB, Ratio: {ratio:.2f}')
")
                echo -e "${YELLOW}⚠️  $file: Memory calculation may be inaccurate${NC}"
                echo "   Line: $line"
                echo "   $details"
                WARNINGS=$((WARNINGS + 1))
            fi
        fi
    done < <(grep -i -E "[0-9]+B[[:space:]]*model.*[0-9]+GB" "$file" 2>/dev/null || true)
}

# Function to validate cost calculations
validate_cost_calculations() {
    local file="$1"
    
    echo -e "${BLUE}   Checking cost calculations...${NC}"
    
    # Look for cost reduction claims
    while IFS= read -r line; do
        # Check for percentage claims like "70% reduction" or "saves 50%"
        if [[ "$line" =~ ([0-9]+)%[[:space:]]*(reduction|savings?|decrease) ]] || [[ "$line" =~ (saves?|reduces?)[[:space:]]*([0-9]+)% ]]; then
            local percentage
            if [[ -n "${BASH_REMATCH[1]}" ]]; then
                percentage="${BASH_REMATCH[1]}"
            else
                percentage="${BASH_REMATCH[2]}"
            fi
            
            # Warn about unrealistic cost reduction claims
            if [[ $percentage -gt 90 ]]; then
                echo -e "${YELLOW}⚠️  $file: Potentially unrealistic cost reduction claim${NC}"
                echo "   Line: $line"
                echo "   Claim: $percentage% reduction - verify this is achievable"
                WARNINGS=$((WARNINGS + 1))
            fi
        fi
        
        # Check for compound vs additive savings errors
        if [[ "$line" =~ ([0-9]+)%.*\+.*([0-9]+)%.*=.*([0-9]+)% ]]; then
            local first_pct="${BASH_REMATCH[1]}"
            local second_pct="${BASH_REMATCH[2]}"
            local claimed_total="${BASH_REMATCH[3]}"
            
            # Check if they're adding percentages incorrectly
            local simple_add=$((first_pct + second_pct))
            local compound_calc
            compound_calc=$(python3 -c "
# Compound calculation: 1 - (1-a/100) * (1-b/100)
first = $first_pct / 100.0
second = $second_pct / 100.0
compound = 1 - (1 - first) * (1 - second)
print(f'{compound * 100:.0f}')
")
            
            if [[ $claimed_total -eq $simple_add ]] && [[ $simple_add -ne $compound_calc ]]; then
                echo -e "${YELLOW}⚠️  $file: Possible additive vs compound savings error${NC}"
                echo "   Line: $line"
                echo "   Simple addition: $simple_add%, Compound: ${compound_calc}%"
                WARNINGS=$((WARNINGS + 1))
            fi
        fi
    done < <(grep -i -E "[0-9]+%.*reduction|[0-9]+%.*saving|saves.*[0-9]+%" "$file" 2>/dev/null || true)
}

# Function to validate GPU specifications
validate_gpu_specifications() {
    local file="$1"
    
    echo -e "${BLUE}   Checking GPU specifications...${NC}"
    
    # Known GPU memory specifications
    declare -A gpu_memory=(
        ["A100-40GB"]="40"
        ["A100-80GB"]="80"
        ["H100-80GB"]="80"
        ["H100-96GB"]="96"
        ["V100-32GB"]="32"
        ["V100-16GB"]="16"
        ["L4-24GB"]="24"
        ["L40S-48GB"]="48"
        ["T4-16GB"]="16"
    )
    
    # Check for GPU memory claims
    while IFS= read -r line; do
        for gpu in "${!gpu_memory[@]}"; do
            local gpu_pattern="${gpu%%-*}"  # Get base GPU name
            local expected_memory="${gpu_memory[$gpu]}"
            
            if [[ "$line" =~ $gpu_pattern.*([0-9]+)GB ]] && [[ "$line" =~ $gpu ]]; then
                local claimed_memory="${BASH_REMATCH[1]}"
                
                if [[ "$claimed_memory" != "$expected_memory" ]]; then
                    echo -e "${RED}❌ $file: Incorrect GPU memory specification${NC}"
                    echo "   Line: $line"
                    echo "   $gpu has ${expected_memory}GB memory, not ${claimed_memory}GB"
                    ERRORS=$((ERRORS + 1))
                fi
            fi
        done
    done < <(grep -i -E "(A100|H100|V100|L4|L40S|T4).*[0-9]+GB" "$file" 2>/dev/null || true)
}

# Function to validate quantization claims
validate_quantization_claims() {
    local file="$1"
    
    echo -e "${BLUE}   Checking quantization claims...${NC}"
    
    # Known quantization memory reductions
    declare -A quant_reductions=(
        ["FP16"]="50"
        ["INT8"]="75"
        ["INT4"]="87.5"
        ["4-bit"]="87.5"
        ["8-bit"]="75"
    )
    
    # Check for quantization memory reduction claims
    while IFS= read -r line; do
        for quant in "${!quant_reductions[@]}"; do
            local expected_reduction="${quant_reductions[$quant]}"
            
            if [[ "$line" =~ $quant.*([0-9]+\.?[0-9]*)%.*memory.*reduction ]] || [[ "$line" =~ $quant.*([0-9]+\.?[0-9]*)%.*reduction ]]; then
                local claimed_reduction="${BASH_REMATCH[1]}"
                
                # Allow 5% tolerance
                local tolerance_check
                tolerance_check=$(python3 -c "
expected = $expected_reduction
actual = $claimed_reduction
tolerance = abs(expected - actual) / expected
print('PASS' if tolerance <= 0.1 else 'FAIL')
")
                
                if [[ "$tolerance_check" == "FAIL" ]]; then
                    echo -e "${YELLOW}⚠️  $file: Quantization reduction claim may be inaccurate${NC}"
                    echo "   Line: $line"
                    echo "   $quant typically provides ~${expected_reduction}% memory reduction, not ${claimed_reduction}%"
                    WARNINGS=$((WARNINGS + 1))
                fi
            fi
        done
    done < <(grep -i -E "(FP16|INT8|INT4|4-bit|8-bit).*[0-9]+\.?[0-9]*%.*reduction" "$file" 2>/dev/null || true)
}

# Function to validate performance claims
validate_performance_claims() {
    local file="$1"
    
    echo -e "${BLUE}   Checking performance claims...${NC}"
    
    # Look for latency and throughput claims
    while IFS= read -r line; do
        # Check for unrealistic latency claims
        if [[ "$line" =~ ([0-9]+\.?[0-9]*)[[:space:]]*(ms|milliseconds?)[[:space:]]*(latency|response) ]]; then
            local latency="${BASH_REMATCH[1]}"
            
            # Warn about unrealistically low latency claims for LLMs
            if (( $(echo "$latency < 10" | bc -l 2>/dev/null || echo 0) )); then
                echo -e "${YELLOW}⚠️  $file: Potentially unrealistic latency claim${NC}"
                echo "   Line: $line"
                echo "   Sub-10ms latency for LLM inference is typically unrealistic"
                WARNINGS=$((WARNINGS + 1))
            fi
        fi
        
        # Check for throughput claims
        if [[ "$line" =~ ([0-9]+)[[:space:]]*(tokens?/s|tokens? per second) ]]; then
            local throughput="${BASH_REMATCH[1]}"
            
            # Warn about unrealistic throughput claims
            if [[ $throughput -gt 10000 ]]; then
                echo -e "${YELLOW}⚠️  $file: Potentially unrealistic throughput claim${NC}"
                echo "   Line: $line"
                echo "   >10,000 tokens/s is typically unrealistic for single GPU inference"
                WARNINGS=$((WARNINGS + 1))
            fi
        fi
    done < <(grep -i -E "[0-9]+\.?[0-9]*[[:space:]]*(ms|tokens?/s)" "$file" 2>/dev/null || true)
}

# Function to validate Kubernetes resource calculations
validate_k8s_resources() {
    local file="$1"
    
    echo -e "${BLUE}   Checking Kubernetes resource calculations...${NC}"
    
    # Check for resource request/limit relationships
    while IFS= read -r line; do
        # Look for memory specifications that might be inconsistent
        if [[ "$line" =~ requests:.*memory:[[:space:]]*[\"']?([0-9]+)Gi[\"']? ]] && [[ "$line" =~ limits:.*memory:[[:space:]]*[\"']?([0-9]+)Gi[\"']? ]]; then
            local request_memory="${BASH_REMATCH[1]}"
            local limit_memory="${BASH_REMATCH[2]}"
            
            if [[ $limit_memory -lt $request_memory ]]; then
                echo -e "${RED}❌ $file: Memory limit less than request${NC}"
                echo "   Line context around: $line"
                echo "   Limits ($limit_memory Gi) should be >= requests ($request_memory Gi)"
                ERRORS=$((ERRORS + 1))
            fi
        fi
    done < <(grep -A5 -B5 "memory:" "$file" 2>/dev/null || true)
}

# Main validation loop
for file in $FILES_TO_CHECK; do
    if [[ ! -f "$file" ]]; then
        continue
    fi
    
    echo -e "${BLUE}🔍 Validating: $file${NC}"
    
    # Run all mathematical validations
    validate_slo_calculations "$file"
    validate_memory_calculations "$file"
    validate_cost_calculations "$file"
    validate_gpu_specifications "$file"
    validate_quantization_claims "$file"
    validate_performance_claims "$file"
    validate_k8s_resources "$file"
    
    echo ""
done

# Final summary
echo "=============================================="
echo -e "${BLUE}📊 Mathematical Accuracy Summary${NC}"
echo "=============================================="

if [[ $ERRORS -eq 0 ]] && [[ $WARNINGS -eq 0 ]]; then
    echo -e "${GREEN}🎉 All mathematical validations passed!${NC}"
    echo -e "${GREEN}✅ No calculation errors or unrealistic claims found${NC}"
    exit 0
elif [[ $ERRORS -eq 0 ]]; then
    echo -e "${YELLOW}⚠️  $WARNINGS warning(s) found${NC}"
    echo -e "${YELLOW}💡 Review warnings for accuracy improvements${NC}"
    exit 0
else
    echo -e "${RED}❌ $ERRORS error(s) and $WARNINGS warning(s) found${NC}"
    echo -e "${RED}🔧 Fix mathematical errors before proceeding${NC}"
    echo ""
    echo -e "${BLUE}💡 Common fixes:${NC}"
    echo "   - Verify SLO percentage calculations (99.9% = 43.2 min/month)"
    echo "   - Check memory requirements for model sizes"
    echo "   - Validate cost reduction claims are realistic"
    echo "   - Ensure GPU specifications are accurate"
    echo "   - Review quantization memory reduction percentages"
    exit 1
fi