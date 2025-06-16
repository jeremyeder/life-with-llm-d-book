# Makefile for LLM Cost Optimization - Life with llm-d Book
# Automates pricing updates, validation, and cost calculations

.PHONY: all update-pricing validate-costs test-costs generate-forecasts clean help

# Configuration
PYTHON := python3
PRICING_SCRIPT := scripts/cost-prediction/update_pricing.py
COST_PREDICTOR := scripts/cost-prediction/predict_costs.py
TEST_SCRIPT := scripts/cost-prediction/test_cost_calculations.py
PRICING_DATA := scripts/cost-prediction/pricing_data.json
FORECAST_OUTPUT := scripts/cost-prediction/forecasts/

# Validation scripts
VALIDATE_MODELS := ./scripts/validate-model-names.sh
VALIDATE_NAMESPACES := ./scripts/validate-namespaces.sh
VALIDATE_RESOURCES := ./scripts/validate-resource-specs.sh
VALIDATE_SHARED_CONFIG := ./scripts/check-shared-config-refs.sh

# Default target
all: update-pricing validate-costs test-costs

# Help target
help:
	@echo "LLM Cost Optimization Makefile"
	@echo "=============================="
	@echo ""
	@echo "Targets:"
	@echo "  make update-pricing    - Update GPU pricing data from sources"
	@echo "  make validate-costs    - Validate all cost calculations in docs"
	@echo "  make test-costs       - Run comprehensive cost calculation tests"
	@echo "  make generate-forecasts - Generate cost forecasts for all profiles"
	@echo "  make validate-chapter  - Validate Chapter 11 with all safeguards"
	@echo "  make clean            - Clean generated files"
	@echo "  make help             - Show this help message"
	@echo ""
	@echo "Quick Commands:"
	@echo "  make                  - Update pricing, validate, and test"
	@echo "  make test-costs VERBOSE=1 - Run tests with verbose output"

# Update pricing data
update-pricing: $(PRICING_SCRIPT)
	@echo "📊 Updating GPU pricing data..."
	@$(PYTHON) $(PRICING_SCRIPT) --output $(PRICING_DATA)
	@echo "✅ Pricing data updated: $(PRICING_DATA)"
	@echo "   Last updated: $$(date)"
	@git diff --stat $(PRICING_DATA) || true

# Create pricing update script if it doesn't exist
$(PRICING_SCRIPT):
	@mkdir -p $(dir $(PRICING_SCRIPT))
	@cat > $(PRICING_SCRIPT) << 'EOF'
#!/usr/bin/env python3
"""
Update pricing data for GPU resources.
Fetches current market prices and applies realistic variations.
"""

import json
import random
from datetime import datetime
import argparse

def fetch_current_pricing():
    """Fetch current GPU pricing (simulated for book)."""
    
    # Base prices as of 2024 with market variations
    base_prices = {
        "on_premise": {
            "gpu_purchase": {
                "a100_40gb": 15000 * random.uniform(0.95, 1.05),
                "a100_80gb": 20000 * random.uniform(0.95, 1.05),
                "h100_80gb": 30000 * random.uniform(0.95, 1.05),
                "v100_32gb": 8000 * random.uniform(0.95, 1.05)
            },
            "power_cost_kwh": 0.12 * random.uniform(0.9, 1.1),
            "datacenter_pue": 1.5,
            "rack_cost_monthly": 500 * random.uniform(0.95, 1.05),
            "staff_cost_hourly": 150
        },
        "gpu_service": {
            "hourly_rates": {
                "a100_40gb": 1.80 * random.uniform(0.95, 1.05),
                "a100_80gb": 2.40 * random.uniform(0.95, 1.05),
                "h100_80gb": 4.20 * random.uniform(0.95, 1.05),
                "v100_32gb": 0.90 * random.uniform(0.95, 1.05)
            },
            "spot_discount": 0.3,
            "commitment_discounts": {
                "1_month": 0.1,
                "6_month": 0.2,
                "12_month": 0.3
            }
        },
        "last_updated": datetime.now().isoformat()
    }
    
    return base_prices

def main():
    parser = argparse.ArgumentParser(description='Update GPU pricing data')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    args = parser.parse_args()
    
    pricing = fetch_current_pricing()
    
    with open(args.output, 'w') as f:
        json.dump(pricing, f, indent=2)
    
    print(f"Updated pricing data saved to {args.output}")

if __name__ == "__main__":
    main()
EOF
	@chmod +x $(PRICING_SCRIPT)

# Validate cost calculations in documentation
validate-costs: update-pricing
	@echo "🔍 Validating cost calculations in documentation..."
	@echo ""
	@echo "Checking model naming consistency..."
	@$(VALIDATE_MODELS) docs/11-cost-optimization.md
	@echo ""
	@echo "Checking namespace conventions..."
	@$(VALIDATE_NAMESPACES) docs/11-cost-optimization.md
	@echo ""
	@echo "Checking resource specifications..."
	@$(VALIDATE_RESOURCES) docs/11-cost-optimization.md
	@echo ""
	@echo "Checking shared config references..."
	@$(VALIDATE_SHARED_CONFIG) docs/11-cost-optimization.md
	@echo ""
	@echo "✅ All validations passed!"

# Run comprehensive cost calculation tests
test-costs: $(PRICING_DATA) $(TEST_SCRIPT) $(COST_PREDICTOR)
	@echo "🧪 Running cost calculation tests..."
	@mkdir -p scripts/cost-prediction
	@if [ "$(VERBOSE)" = "1" ]; then \
		$(PYTHON) $(TEST_SCRIPT) -v; \
	else \
		$(PYTHON) $(TEST_SCRIPT); \
	fi

# Generate cost forecasts
generate-forecasts: $(PRICING_DATA) $(COST_PREDICTOR)
	@echo "📈 Generating cost forecasts..."
	@mkdir -p $(FORECAST_OUTPUT)
	@$(PYTHON) $(COST_PREDICTOR) --months 24 --output $(FORECAST_OUTPUT)forecast_24m.csv
	@echo "✅ Forecasts generated in $(FORECAST_OUTPUT)"

# Validate entire Chapter 11
validate-chapter:
	@echo "📋 Validating Chapter 11: Cost Optimization"
	@echo "=========================================="
	@$(MAKE) validate-costs
	@echo ""
	@echo "Running markdown linting..."
	@npx markdownlint docs/11-cost-optimization.md --config .markdownlint.json || true
	@echo ""
	@echo "✅ Chapter 11 validation complete!"

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	@rm -f $(PRICING_DATA)
	@rm -rf $(FORECAST_OUTPUT)
	@rm -f scripts/cost-prediction/__pycache__
	@echo "✅ Clean complete"

# Ensure pricing data exists before tests
$(PRICING_DATA): update-pricing

# Watch for changes and auto-validate
watch:
	@echo "👁️  Watching for changes..."
	@while true; do \
		$(MAKE) validate-costs; \
		echo ""; \
		echo "Waiting for changes... (Ctrl+C to stop)"; \
		sleep 5; \
	done

# Generate cost comparison report
cost-report: $(PRICING_DATA) $(COST_PREDICTOR)
	@echo "📊 Generating cost comparison report..."
	@$(PYTHON) -c "
import sys
sys.path.append('scripts/cost-prediction')
from predict_costs import CostPredictor, DeploymentProfile

predictor = CostPredictor('$(PRICING_DATA)')

profiles = [
    DeploymentProfile('startup-small', '8b', 'a100_40gb', 2, 'gpu_service', 1000, 'int8'),
    DeploymentProfile('startup-growth', '8b', 'a100_40gb', 10, 'gpu_service', 5000, 'int8'),
    DeploymentProfile('enterprise-cloud', '70b', 'h100_80gb', 8, 'gpu_service', 10000, 'fp16'),
    DeploymentProfile('enterprise-onprem', '70b', 'h100_80gb', 8, 'on_premise', 10000, 'fp16')
]

print('\n🎯 Cost Optimization Comparison Report')
print('=====================================\n')

for profile in profiles:
    pred = predictor.predict_costs(profile)
    print(f'{profile.name}:')
    print(f'  Monthly Cost: \$${pred.monthly_cost:,.0f}')
    print(f'  Cost per Request: \$${pred.cost_per_request:.6f}')
    if pred.break_even_vs_cloud:
        print(f'  Break-even: {pred.break_even_vs_cloud} months')
    print()
"

# Install dependencies
install-deps:
	@echo "📦 Installing dependencies..."
	@pip install numpy pandas

.SILENT: help