#!/bin/bash
# Pre-push linting workflow for Life with llm-d Book
# Ensures code quality standards before pushing

set -e

echo "🔍 Running pre-push linting workflow..."
echo "======================================"

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Not in a git repository"
    exit 1
fi

# Check if we have changes to lint
CHANGED_PYTHON=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$' || true)
CHANGED_MARKDOWN=$(git diff --cached --name-only --diff-filter=ACM | grep '\.md$' || true)

if [[ -z "$CHANGED_PYTHON" && -z "$CHANGED_MARKDOWN" ]]; then
    echo "ℹ️  No Python or Markdown files to lint"
    exit 0
fi

# Python linting if we have Python files
if [[ -n "$CHANGED_PYTHON" ]]; then
    echo ""
    echo "🐍 Linting Python files..."
    echo "Files to lint: $CHANGED_PYTHON"
    
    # Check if Python tools are available
    if ! command -v black &> /dev/null; then
        echo "⚠️  black not found. Install with: pip install black"
        echo "   Or run: uv pip install -r requirements-test.txt"
        exit 1
    fi
    
    if ! command -v isort &> /dev/null; then
        echo "⚠️  isort not found. Install with: pip install isort"
        echo "   Or run: uv pip install -r requirements-test.txt"
        exit 1
    fi
    
    if ! command -v flake8 &> /dev/null; then
        echo "⚠️  flake8 not found. Install with: pip install flake8"
        echo "   Or run: uv pip install -r requirements-test.txt"
        exit 1
    fi
    
    # Format Python code
    echo "  Running black..."
    black examples/ tests/ docs/cost-optimization/ docs/security-configs/ || {
        echo "❌ black formatting failed"
        exit 1
    }
    
    echo "  Running isort..."
    isort examples/ tests/ docs/cost-optimization/ docs/security-configs/ || {
        echo "❌ isort import sorting failed"
        exit 1
    }
    
    echo "  Running flake8..."
    flake8 examples/ tests/ docs/cost-optimization/ docs/security-configs/ --max-line-length=88 || {
        echo "❌ flake8 linting failed"
        exit 1
    }
    
    echo "✅ Python linting complete"
fi

# Markdown linting if we have Markdown files
if [[ -n "$CHANGED_MARKDOWN" ]]; then
    echo ""
    echo "📝 Linting Markdown files..."
    echo "Files to lint: $CHANGED_MARKDOWN"
    
    # Check if markdownlint is available
    if ! command -v npx &> /dev/null || ! npx markdownlint --version &> /dev/null; then
        echo "⚠️  markdownlint not found. Install with: npm install"
        exit 1
    fi
    
    # Run markdownlint
    echo "  Running markdownlint..."
    npx markdownlint 'docs/**/*.md' --config .markdownlint.json || {
        echo "❌ Markdown linting failed"
        echo "   Fix with: npm run lint:fix"
        exit 1
    }
    
    echo "✅ Markdown linting complete"
fi

# Run tests if Python files changed
if [[ -n "$CHANGED_PYTHON" ]]; then
    echo ""
    echo "🧪 Running tests for changed Python files..."
    
    if command -v pytest &> /dev/null; then
        pytest tests/ || {
            echo "❌ Tests failed"
            exit 1
        }
        echo "✅ Tests passed"
    else
        echo "⚠️  pytest not found. Install with: uv pip install -r requirements-test.txt"
        echo "   Skipping tests..."
    fi
fi

echo ""
echo "✅ All linting checks passed!"
echo "🚀 Ready to push"