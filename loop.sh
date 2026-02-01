#!/bin/bash
# FerroML Ralph Wiggum Loop
# Usage:
#   ./loop.sh           - Build mode, unlimited iterations
#   ./loop.sh plan      - Plan mode, unlimited iterations
#   ./loop.sh 20        - Build mode, max 20 iterations
#   ./loop.sh plan 5    - Plan mode, max 5 iterations

set -e

MODE="build"
MAX_ITERATIONS=0
ITERATION=0

# Parse arguments
for arg in "$@"; do
    if [[ "$arg" == "plan" ]]; then
        MODE="plan"
    elif [[ "$arg" =~ ^[0-9]+$ ]]; then
        MAX_ITERATIONS=$arg
    fi
done

PROMPT_FILE="PROMPT_${MODE}.md"

if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "Error: $PROMPT_FILE not found"
    exit 1
fi

echo "=============================================="
echo "FerroML Ralph Loop"
echo "Mode: $MODE"
echo "Max iterations: $([ $MAX_ITERATIONS -eq 0 ] && echo 'unlimited' || echo $MAX_ITERATIONS)"
echo "Prompt file: $PROMPT_FILE"
echo "=============================================="
echo ""

while :; do
    ITERATION=$((ITERATION + 1))

    echo ""
    echo "=============================================="
    echo "ITERATION $ITERATION - $(date)"
    echo "=============================================="
    echo ""

    # Run Claude with the prompt
    cat "$PROMPT_FILE" | claude -p \
        --dangerously-skip-permissions \
        --model claude-sonnet-4-20250514 \
        --verbose

    EXIT_CODE=$?

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo ""
        echo "Claude exited with code $EXIT_CODE"
        echo "Waiting 30 seconds before retry..."
        sleep 30
    fi

    # Check iteration limit
    if [[ $MAX_ITERATIONS -gt 0 && $ITERATION -ge $MAX_ITERATIONS ]]; then
        echo ""
        echo "=============================================="
        echo "Reached max iterations ($MAX_ITERATIONS)"
        echo "=============================================="
        break
    fi

    # Brief pause between iterations
    echo ""
    echo "Iteration $ITERATION complete. Starting next iteration in 5 seconds..."
    sleep 5
done

echo ""
echo "Ralph loop finished after $ITERATION iterations"
