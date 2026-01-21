#!/bin/bash
# FerroML WSL Setup Script
# Run this inside WSL: bash setup-wsl.sh

set -e

echo "=== FerroML WSL Setup ==="

# 1. Install Rust (if not installed)
if ! command -v rustc &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "Rust already installed: $(rustc --version)"
fi

# 2. Install Claude CLI (if not installed)
if ! command -v claude &> /dev/null; then
    echo "Installing Claude CLI..."
    # Check if npm is available
    if ! command -v npm &> /dev/null; then
        echo "Installing Node.js..."
        curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
        sudo apt-get install -y nodejs
    fi
    npm install -g @anthropic-ai/claude-code
else
    echo "Claude CLI already installed: $(claude --version)"
fi

# 3. Verify installations
echo ""
echo "=== Verification ==="
echo "Rust: $(rustc --version)"
echo "Cargo: $(cargo --version)"
echo "Claude: $(claude --version 2>/dev/null || echo 'not found')"

# 4. Test build
echo ""
echo "=== Testing FerroML Build ==="
cargo check

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run the Ralph loop:"
echo "  ./ralph.sh build"
echo ""
echo "Or manually:"
echo "  while true; do cat PROMPT_build.md | claude --dangerously-skip-permissions; cargo check && cargo test; sleep 2; done"
