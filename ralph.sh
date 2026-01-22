#!/bin/bash
# FerroML Ralph Loop (with hang detection + retry logic)
# Usage: ./ralph.sh [plan|build]
# See: https://github.com/anthropics/claude-code/issues/19060

set -e

MODE="${1:-build}"
PROMPT_FILE="PROMPT_${MODE}.md"
TIMEOUT_SECONDS=600    # 10 minute timeout per iteration
MAX_RETRIES=3          # Retry failed iterations
RETRY_DELAY=30         # Seconds to wait between retries

if [ ! -f "$PROMPT_FILE" ]; then
    echo "Error: $PROMPT_FILE not found"
    echo "Usage: ./ralph.sh [plan|build]"
    exit 1
fi

echo "=== FerroML Ralph Loop ==="
echo "Mode: $MODE"
echo "Press Ctrl+C to stop"
echo ""

run_claude() {
    local prompt="$1"
    local temp_output="$2"

    if command -v timeout &> /dev/null; then
        timeout --signal=KILL $TIMEOUT_SECONDS claude --dangerously-skip-permissions -p "$prompt" 2>&1 | tee "$temp_output"
        return $?
    elif command -v gtimeout &> /dev/null; then
        gtimeout --signal=KILL $TIMEOUT_SECONDS claude --dangerously-skip-permissions -p "$prompt" 2>&1 | tee "$temp_output"
        return $?
    else
        claude --dangerously-skip-permissions -p "$prompt" 2>&1 | tee "$temp_output" &
        local pid=$!
        local seconds=0
        while kill -0 $pid 2>/dev/null; do
            if [ $seconds -ge $TIMEOUT_SECONDS ]; then
                echo "[Timeout reached, killing process]"
                kill -9 $pid 2>/dev/null || true
                return 124
            fi
            sleep 1
            ((seconds++))
        done
        wait $pid 2>/dev/null
        return $?
    fi
}

iteration=0
consecutive_failures=0

while true; do
    iteration=$((iteration + 1))
    echo "--- Iteration $iteration ---"

    TEMP_OUTPUT=$(mktemp)
    PROMPT="$(<"$PROMPT_FILE")"

    # Try running Claude with retries
    success=false
    for retry in $(seq 1 $MAX_RETRIES); do
        set +e
        run_claude "$PROMPT" "$TEMP_OUTPUT"
        EXIT_CODE=$?
        set -e

        # Check for known error patterns in output
        if grep -q "No messages returned" "$TEMP_OUTPUT" 2>/dev/null; then
            echo "[Detected 'No messages returned' error - known CLI bug]"
            EXIT_CODE=1
        fi

        if [ $EXIT_CODE -eq 0 ]; then
            success=true
            consecutive_failures=0
            break
        elif [ $EXIT_CODE -eq 137 ] || [ $EXIT_CODE -eq 124 ]; then
            echo "[Process killed/timed out - attempt $retry/$MAX_RETRIES]"
        else
            echo "[Claude failed with exit code $EXIT_CODE - attempt $retry/$MAX_RETRIES]"
        fi

        if [ $retry -lt $MAX_RETRIES ]; then
            echo "[Waiting ${RETRY_DELAY}s before retry...]"
            sleep $RETRY_DELAY
            # Increase delay for subsequent retries (exponential backoff)
            RETRY_DELAY=$((RETRY_DELAY * 2))
        fi
    done

    # Reset retry delay for next iteration
    RETRY_DELAY=30

    if [ "$success" = false ]; then
        ((consecutive_failures++))
        echo "[WARNING: Iteration $iteration failed after $MAX_RETRIES attempts]"

        if [ $consecutive_failures -ge 3 ]; then
            echo "[ERROR: 3 consecutive iteration failures - stopping loop]"
            echo "[Check API status, rate limits, or network connectivity]"
            rm -f "$TEMP_OUTPUT"
            exit 1
        fi

        echo "[Waiting 60s before next iteration due to failures...]"
        sleep 60
    fi

    rm -f "$TEMP_OUTPUT"

    # Run validation after each iteration
    echo ""
    echo "--- Validation ---"
    cargo check 2>&1 || echo "cargo check failed"
    cargo test 2>&1 || echo "cargo test failed"

    echo ""
    echo "--- Iteration $iteration complete ---"
    echo ""

    # Small delay to allow Ctrl+C and prevent rate limiting
    sleep 2
done
