#!/usr/bin/env bash
# tools/test.sh — Run the test suite
#
# Usage:
#   ./tools/test.sh               # Run all tests
#   ./tools/test.sh <name>        # Run tests matching <name>
#   ./tools/test.sh --build       # Build then run all tests

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"

bold()  { printf "\033[1m%s\033[0m\n" "$*"; }
green() { printf "\033[1;32m%s\033[0m\n" "$*"; }
red()   { printf "\033[1;31m%s\033[0m\n" "$*"; }

DO_BUILD=false
TEST_FILTER=""

for arg in "$@"; do
    case "$arg" in
        --build|-b)
            DO_BUILD=true
            ;;
        *)
            TEST_FILTER="$arg"
            ;;
    esac
done

if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    bold "No build found, building first..."
    DO_BUILD=true
fi

if $DO_BUILD; then
    "$PROJECT_DIR/tools/build.sh"
    echo ""
fi

bold "=== Running Tests ==="

CTEST_ARGS=(--test-dir "$BUILD_DIR" --output-on-failure)

if [ -n "$TEST_FILTER" ]; then
    CTEST_ARGS+=(-R "$TEST_FILTER")
    bold "Filter: $TEST_FILTER"
fi

if ctest "${CTEST_ARGS[@]}"; then
    echo ""
    green "All matched tests passed."
else
    echo ""
    red "Some tests failed."
    exit 1
fi
