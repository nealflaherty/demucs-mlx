#!/usr/bin/env bash
# Download and convert the HTDemucs model to SafeTensors format.
#
# Usage:
#   ./tools/download_model.sh              # download default (htdemucs)
#   ./tools/download_model.sh htdemucs_6s  # download 6-source model
#
# Requires Python 3. Will create a venv at tools/.venv if needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_DIR/tools/.venv"
MODEL_NAME="${1:-htdemucs}"

echo "=== Download Model ($MODEL_NAME) ==="

# Create and setup venv if needed
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python venv..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip > /dev/null
    echo "Installing dependencies..."
    pip install -r "$SCRIPT_DIR/requirements.txt"
else
    source "$VENV_DIR/bin/activate"
fi

# Run converter
python3 "$SCRIPT_DIR/convert_model.py" -n "$MODEL_NAME" -o "$PROJECT_DIR/models"

deactivate 2>/dev/null || true
echo "Done."
