#!/bin/bash
# Simplified validation for monthly updates

set -e

echo "🔍 Running essential validation checks..."

# Basic validation - just check for major issues
echo "📋 Validating YAML syntax..."
if command -v yamllint >/dev/null 2>&1; then
    yamllint examples/ -q || echo "  ⚠️ YAML warnings found (non-blocking)"
else
    echo "  ℹ️ yamllint not available, skipping"
fi

echo "📋 Checking markdown files..."
if npm run lint >/dev/null 2>&1; then
    echo "  ✅ Markdown linting passed"
else
    echo "  ⚠️ Markdown linting issues found (non-blocking)"
fi

echo "📋 Basic file structure check..."
required_files=("docs/00-whats-new.md" "docs/00-whats-next.md" "package.json")
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file exists"
    else
        echo "  ❌ Missing required file: $file"
        exit 1
    fi
done

echo "✅ Essential validation complete"