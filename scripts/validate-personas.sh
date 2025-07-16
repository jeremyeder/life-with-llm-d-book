#!/bin/bash

# Persona Validation Framework
# Validates book content against target persona needs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Default values
CHAPTER_FILE=""
PERSONA=""
OUTPUT_FORMAT="text"
VERBOSE=false

# Available personas
AVAILABLE_PERSONAS=("alex-platform-engineer" "morgan-ml-engineer")

usage() {
    cat << EOF
Usage: $0 [OPTIONS] <chapter-file> [persona]

Validate book content against persona needs and expectations.

ARGUMENTS:
    chapter-file    Path to chapter markdown file (required)
    persona        Persona to validate against (optional, validates all if not specified)

OPTIONS:
    -f, --format    Output format (text, json, yaml) [default: text]
    -v, --verbose   Enable verbose output
    -h, --help      Show this help message

AVAILABLE PERSONAS:
    alex-platform-engineer    Early career platform engineer
    morgan-ml-engineer        Mid-career ML engineer

EXAMPLES:
    $0 docs/02-installation-setup.md alex-platform-engineer
    $0 docs/05-sre-operations.md --format json
    $0 docs/06-performance-optimization.md --verbose

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--format)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
        *)
            if [[ -z "$CHAPTER_FILE" ]]; then
                CHAPTER_FILE="$1"
            elif [[ -z "$PERSONA" ]]; then
                PERSONA="$1"
            else
                echo "Too many arguments" >&2
                usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate arguments
if [[ -z "$CHAPTER_FILE" ]]; then
    echo "Error: Chapter file is required" >&2
    usage
    exit 1
fi

if [[ ! -f "$CHAPTER_FILE" ]]; then
    echo "Error: Chapter file not found: $CHAPTER_FILE" >&2
    exit 1
fi

# Convert to absolute path
CHAPTER_FILE="$(cd "$(dirname "$CHAPTER_FILE")" && pwd)/$(basename "$CHAPTER_FILE")"

# Validate persona if specified
if [[ -n "$PERSONA" ]]; then
    if [[ ! " ${AVAILABLE_PERSONAS[@]} " =~ " ${PERSONA} " ]]; then
        echo "Error: Invalid persona: $PERSONA" >&2
        echo "Available personas: ${AVAILABLE_PERSONAS[*]}" >&2
        exit 1
    fi
fi

# Function to validate against a single persona
validate_persona() {
    local persona="$1"
    local chapter_file="$2"
    
    if [[ "$VERBOSE" == "true" ]]; then
        echo "Validating $chapter_file against persona: $persona" >&2
    fi
    
    # Run the Python validation script
    python3 "${SCRIPT_DIR}/persona-relevance-check.py" \
        --chapter "$chapter_file" \
        --persona "$persona" \
        --format "$OUTPUT_FORMAT" \
        ${VERBOSE:+--verbose}
}

# Main validation logic
if [[ -n "$PERSONA" ]]; then
    # Validate against single persona
    validate_persona "$PERSONA" "$CHAPTER_FILE"
else
    # Validate against all personas
    for persona in "${AVAILABLE_PERSONAS[@]}"; do
        if [[ "$OUTPUT_FORMAT" == "text" ]]; then
            echo "=== Validating against $persona ==="
            echo
        fi
        validate_persona "$persona" "$CHAPTER_FILE"
        if [[ "$OUTPUT_FORMAT" == "text" ]]; then
            echo
        fi
    done
fi