#!/bin/bash
# Initialize the claude-checkpoint-system for this repository

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="$SCRIPT_DIR/CLAUDE_STATE.md"

echo "🚀 Claude Checkpoint System Setup"
echo ""

# Create CLAUDE_STATE.md if it doesn't exist
if [[ ! -f "$STATE_FILE" ]]; then
    echo "📝 Creating CLAUDE_STATE.md..."
    
    cat > "$STATE_FILE" << 'EOF'
# Claude Session State

## Current Session Context

**Repository**: life-with-llm-d-book  
**Branch**: option2-condensed-book  
**Last Updated**: $(date '+%Y-%m-%d %H:%M:%S')  
**Session Focus**: Repository checkpoint system setup

## Project Overview

This is a comprehensive book repository for "Life with llm-d" - a guide to deploying and operating Large Language Model workloads using llm-d on Kubernetes and OpenShift.

### Repository Structure
- **docs/**: 12 chapters + 3 appendices covering complete llm-d lifecycle
- **examples/**: Code samples and configurations  
- **scripts/**: Validation and quality assurance tools
- **static/**: Images and assets for Docusaurus site

## Key Files and Locations

### Documentation
- `docs/`: Main book content
- `README.md`: Project overview and structure
- `appendices/`: Reference materials

### Configuration
- `package.json`: Node.js dependencies for Docusaurus
- `docusaurus.config.js`: Site configuration
- `scripts/`: Validation and utility scripts

### Development
- Local development: `npm start`
- Build: `npm run build`
- Validation: `make validate-chapter`

## Instructions for Claude

When resuming a session, read this file first to understand:
1. Current project context and structure
2. Recent changes and git status  
3. Available tools and commands
4. Previous session focus and progress

---

*This file maintains session state for token-efficient context management across Claude Code sessions.*
EOF
fi

echo "✅ Checkpoint system initialized!"
echo ""
echo "📋 How to use:"
echo "  - CLAUDE_STATE.md contains current session context"
echo "  - Tell Claude to read CLAUDE_STATE.md when resuming sessions"
echo "  - Update the file manually or ask Claude to update it as needed"
echo ""
echo "💡 Benefits:"
echo "- Saves token costs by managing context externally"
echo "- Preserves complete session history across Claude restarts"
echo "- Provides version control friendly state tracking"
echo ""
echo "🔗 For advanced features, install the full claude-checkpoint-system:"
echo "  git clone https://github.com/jeremyeder/claude-checkpoint-system.git"
echo ""
echo "🎉 Setup complete! The claude-checkpoint-system is now active in this repository."