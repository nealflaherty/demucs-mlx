#!/usr/bin/env bash
# tools/build.sh — Configure and build the project
#
# Usage:
#   ./tools/build.sh              # Release build
#   ./tools/build.sh debug        # Debug build
#   ./tools/build.sh clean        # Clean and rebuild

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
BUILD_TYPE="${1:-Release}"

bold()  { printf "\033[1m%s\033[0m\n" "$*"; }
green() { printf "\033[1;32m%s\033[0m\n" "$*"; }

case "${1:-}" in
    clean)
        bold "Cleaning build directory..."
        rm -rf "$BUILD_DIR"
        BUILD_TYPE="Release"
        ;;
    debug|Debug)
        BUILD_TYPE="Debug"
        ;;
    release|Release|"")
        BUILD_TYPE="Release"
        ;;
    *)
        echo "Usage: $0 [release|debug|clean]"
        exit 1
        ;;
esac

bold "=== Build ($BUILD_TYPE) ==="

# Initialize MLX submodule if needed
MLX_DIR="$PROJECT_DIR/cpp/third_party/mlx"
if [ ! -f "$MLX_DIR/CMakeLists.txt" ]; then
    bold "Initializing MLX submodule..."
    git -C "$PROJECT_DIR" submodule update --init --recursive cpp/third_party/mlx
fi

# Download model if missing
MODEL_PATH="$PROJECT_DIR/models/htdemucs.safetensors"
if [ ! -f "$MODEL_PATH" ]; then
    bold "Model not found, downloading..."
    "$PROJECT_DIR/tools/download_model.sh"
fi

if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    bold "Configuring..."
    cmake -S "$PROJECT_DIR/cpp" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DBUILD_TESTS=ON
fi

bold "Building..."
cmake --build "$BUILD_DIR" -j

green "Build complete."
echo "Binary: $BUILD_DIR/demucs_separate"
