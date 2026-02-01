#!/bin/bash
# Setup script for Ralph loop in WSL

echo "=== Setting up Ralph Loop for WSL ==="

# Make loop executable
chmod +x loop.sh

# Check if claude is available
if ! command -v claude &> /dev/null; then
    echo "ERROR: claude CLI not found in WSL"
    echo "Install it with: npm install -g @anthropic-ai/claude-code"
    exit 1
fi

echo "Claude CLI found: $(which claude)"
echo "Version: $(claude --version 2>/dev/null || echo 'unknown')"

# Set up cron job for 3am
CRON_CMD="cd /mnt/c/Users/Trey/Downloads/ferroml && ./loop.sh >> /tmp/ralph_loop.log 2>&1"
CRON_LINE="0 3 * * * $CRON_CMD"

echo ""
echo "=== Cron Setup ==="
echo "To schedule for 3am daily, run:"
echo "  crontab -e"
echo "Then add this line:"
echo "  $CRON_LINE"
echo ""
echo "Or run this one-liner:"
echo "  (crontab -l 2>/dev/null; echo '$CRON_LINE') | crontab -"
echo ""
echo "=== Manual Test ==="
echo "Run now with: ./loop.sh plan 1"
echo ""
