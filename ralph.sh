#!/bin/bash
# FerroML Ralph Loop
# Usage: ./ralph.sh [plan|build]

set -e

MODE="${1:-build}"
PROMPT_FILE="PROMPT_${MODE}.md"

if [ ! -f "$PROMPT_FILE" ]; then
    echo "Error: $PROMPT_FILE not found"
    echo "Usage: ./ralph.sh [plan|build]"
    exit 1
fi

echo "=== FerroML Ralph Loop ==="
echo "Mode: $MODE"
echo "Press Ctrl+C to stop"
echo ""

iteration=0
while true; do
    iteration=$((iteration + 1))
    echo "--- Iteration $iteration ---"

    # Run Claude with the prompt
    claude --dangerously-skip-permissions "$(cat "$PROMPT_FILE")"

    # Run validation after each iteration
    echo ""
    echo "--- Validation ---"
    cargo check 2>&1 || echo "cargo check failed"
    cargo test 2>&1 || echo "cargo test failed"

    echo ""
    echo "--- Iteration $iteration complete ---"
    echo ""

    # Small delay to allow Ctrl+C
    sleep 2
done
