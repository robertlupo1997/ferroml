#!/bin/bash
# Run Ralph Wiggum loop in Docker sandbox
# Usage:
#   ./ralph-docker.sh              # Plan mode, 1 iteration (test)
#   ./ralph-docker.sh build 10     # Build mode, 10 iterations
#   ./ralph-docker.sh plan         # Plan mode, unlimited

set -e

MODE="${1:-plan}"
ITERATIONS="${2:-1}"

echo "=============================================="
echo "FerroML Ralph Loop (Docker Sandbox)"
echo "Mode: $MODE"
echo "Iterations: $ITERATIONS"
echo "=============================================="

# Build and run
docker-compose -f docker-compose.ralph.yml build
docker-compose -f docker-compose.ralph.yml run --rm ralph ./loop.sh "$MODE" "$ITERATIONS"
