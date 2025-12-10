#!/bin/bash
# Run nano-vllm tests inside Gaudi container
#
# Usage:
#   ./scripts/run_tests.sh              # Run all tests
#   ./scripts/run_tests.sh --no-multi   # Skip multi-HPU tests
#   ./scripts/run_tests.sh --cov        # With coverage report

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Parse arguments
PYTEST_ARGS="-v"
COV_ARGS=""

for arg in "$@"; do
    case $arg in
        --no-multi)
            PYTEST_ARGS="$PYTEST_ARGS -m 'not multi_hpu'"
            ;;
        --cov)
            COV_ARGS="--cov=nanovllm --cov-report=html --cov-report=term"
            ;;
        *)
            PYTEST_ARGS="$PYTEST_ARGS $arg"
            ;;
    esac
done

echo "Running tests..."
echo "pytest tests/ $PYTEST_ARGS $COV_ARGS"
pytest tests/ $PYTEST_ARGS $COV_ARGS
