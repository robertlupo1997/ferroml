#!/bin/bash
# FerroML Ralph Wiggum Loop
# Usage:
#   ./loop.sh           - Build mode, unlimited iterations
#   ./loop.sh plan      - Plan mode, unlimited iterations
#   ./loop.sh 20        - Build mode, max 20 iterations
#   ./loop.sh plan 5    - Plan mode, max 5 iterations

# Note: Do NOT use 'set -e' here - we handle errors manually
# and want the loop to continue even if individual iterations fail

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

# Find Claude CLI (works in both WSL and Docker)
if [[ -x "$HOME/.local/bin/claude" ]]; then
    CLAUDE_CMD="$HOME/.local/bin/claude"
elif command -v claude &> /dev/null; then
    CLAUDE_CMD="claude"
elif [[ -x "/usr/local/bin/claude" ]]; then
    CLAUDE_CMD="/usr/local/bin/claude"
else
    echo "Error: Claude CLI not found"
    exit 1
fi

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

    # Heartbeat logging
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting iteration $ITERATION" >> ralph-heartbeat.log

    # Run Claude with the prompt (using recommended Ralph Wiggum settings)
    # - timeout 1800s (30 min) for Opus which is slower
    # - stream-json enables real-time monitoring
    # - verbose provides additional debugging info
    timeout 1800 bash -c "cat '$PROMPT_FILE' | $CLAUDE_CMD -p \
        --dangerously-skip-permissions \
        --model opus \
        --output-format stream-json \
        --verbose"

    EXIT_CODE=$?

    # Log completion
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Iteration $ITERATION finished with exit code $EXIT_CODE" >> ralph-heartbeat.log

    if [[ $EXIT_CODE -eq 124 ]]; then
        echo ""
        echo "⚠️  Iteration $ITERATION TIMED OUT after 1800 seconds"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] TIMEOUT: Iteration $ITERATION exceeded 1800s limit" >> ralph-heartbeat.log
        echo "Waiting 30 seconds before next iteration..."
        sleep 30
    elif [[ $EXIT_CODE -ne 0 ]]; then
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
