#!/usr/bin/env bash
# tools/run_benchmark.sh — Benchmark C++/Swift MLX vs Python Demucs
#
# Usage:
#   ./tools/run_benchmark.sh <audio_file>
#   ./tools/run_benchmark.sh  # uses default test file
#
# This script:
#   1. Creates a Python venv and installs demucs if needed
#   2. Builds the C++ binary if needed
#   3. Runs Python, C++, and Swift on the same audio file
#   4. Reports wall-clock time and audio quality comparison
#      (SDR, correlation, max absolute error vs Python reference)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_VENV="$PROJECT_DIR/tools/.venv"
CPP_BINARY="$PROJECT_DIR/cpp/build/demucs_separate"
SWIFT_BINARY="$PROJECT_DIR/swift/.build/release/demucs-separate"
MODEL_PATH="$PROJECT_DIR/models/htdemucs.safetensors"

# Output directories (cleaned between runs)
PYTHON_OUT="$PROJECT_DIR/tools/.bench_output/python"
CPP_OUT="$PROJECT_DIR/tools/.bench_output/cpp"
SWIFT_OUT="$PROJECT_DIR/tools/.bench_output/swift"

# Default test file
DEFAULT_AUDIO="$PROJECT_DIR/audio/test.mp3"

AUDIO_FILE="${1:-$DEFAULT_AUDIO}"
STEMS=("drums" "bass" "other" "vocals")

# ---------- helpers ----------

bold()  { printf "\033[1m%s\033[0m\n" "$*"; }
green() { printf "\033[1;32m%s\033[0m\n" "$*"; }
red()   { printf "\033[1;31m%s\033[0m\n" "$*"; }
dim()   { printf "\033[2m%s\033[0m\n" "$*"; }

# ---------- validate input ----------

if [ ! -f "$AUDIO_FILE" ]; then
    red "Error: audio file not found: $AUDIO_FILE"
    echo "Usage: $0 <audio_file>"
    exit 1
fi

AUDIO_NAME=$(basename "$AUDIO_FILE")
TRACK_NAME="${AUDIO_NAME%.*}"
AUDIO_DURATION=$(afinfo "$AUDIO_FILE" 2>/dev/null | grep "estimated duration" | awk '{print $3}' || echo "unknown")

bold "=== Demucs Benchmark ==="
echo "Audio:    $AUDIO_NAME"
echo "Duration: ${AUDIO_DURATION}s"
echo ""

# ---------- step 1: set up Python venv ----------

if [ ! -d "$BENCH_VENV" ]; then
    bold "Setting up Python venv..."
    python3 -m venv "$BENCH_VENV"
    "$BENCH_VENV/bin/pip" install --upgrade pip -q
    "$BENCH_VENV/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"
    green "Python venv ready."
else
    dim "Python venv already exists at tools/.venv"
fi

if ! "$BENCH_VENV/bin/python" -c "import demucs" 2>/dev/null; then
    bold "Installing dependencies into venv..."
    "$BENCH_VENV/bin/pip" install -r "$SCRIPT_DIR/requirements.txt"
fi

# Ensure numpy/scipy available for comparison
"$BENCH_VENV/bin/pip" install numpy scipy -q 2>/dev/null

# ---------- step 2: build C++ if needed ----------

if [ ! -f "$CPP_BINARY" ]; then
    bold "Building C++ binary..."
    mkdir -p "$PROJECT_DIR/cpp/build"
    cmake -S "$PROJECT_DIR/cpp" -B "$PROJECT_DIR/cpp/build" -DCMAKE_BUILD_TYPE=Release
    cmake --build "$PROJECT_DIR/cpp/build" --target demucs_separate -j
    green "C++ build complete."
else
    dim "C++ binary already built at cpp/build/demucs_separate"
fi

# ---------- step 3: build Swift if needed ----------

if [ ! -f "$SWIFT_BINARY" ]; then
    bold "Building Swift binary (release)..."
    (cd "$PROJECT_DIR/swift" && swift build -c release)
    green "Swift build complete."
else
    dim "Swift binary already built at swift/.build/release/demucs-separate"
fi

if [ ! -f "$MODEL_PATH" ]; then
    red "Error: model weights not found at $MODEL_PATH"
    echo "Run: python tools/convert_model.py"
    exit 1
fi

# ---------- step 4: clean output dirs ----------

rm -rf "$PYTHON_OUT" "$CPP_OUT" "$SWIFT_OUT"
mkdir -p "$PYTHON_OUT" "$CPP_OUT" "$SWIFT_OUT"

# ---------- step 5: run Python demucs (gold standard) ----------

bold "Running Python demucs (htdemucs, --shifts 1)..."
echo ""

PYTHON_START=$(python3 -c "import time; print(time.time())")
"$BENCH_VENV/bin/python" -m demucs.separate \
    -n htdemucs \
    --shifts 1 \
    --overlap 0.25 \
    -o "$PYTHON_OUT" \
    "$AUDIO_FILE"
PYTHON_END=$(python3 -c "import time; print(time.time())")
PYTHON_TIME=$(python3 -c "print(f'{$PYTHON_END - $PYTHON_START:.1f}')")

echo ""
green "Python finished in ${PYTHON_TIME}s"
echo ""

# ---------- step 6: run C++ MLX ----------

bold "Running C++ MLX demucs (--shifts 1)..."
echo ""

CPP_START=$(python3 -c "import time; print(time.time())")
"$CPP_BINARY" \
    --model "$MODEL_PATH" \
    --shifts 1 \
    --overlap 0.25 \
    -o "$CPP_OUT" \
    "$AUDIO_FILE"
CPP_END=$(python3 -c "import time; print(time.time())")
CPP_TIME=$(python3 -c "print(f'{$CPP_END - $CPP_START:.1f}')")

echo ""
green "C++ MLX finished in ${CPP_TIME}s"
echo ""

# ---------- step 7: run Swift MLX ----------

bold "Running Swift MLX demucs (--shifts 1)..."
echo ""

SWIFT_START=$(python3 -c "import time; print(time.time())")
"$SWIFT_BINARY" \
    --model "$MODEL_PATH" \
    --shifts 1 \
    --overlap 0.25 \
    --out "$SWIFT_OUT" \
    "$AUDIO_FILE" 2>/dev/null
SWIFT_END=$(python3 -c "import time; print(time.time())")
SWIFT_TIME=$(python3 -c "print(f'{$SWIFT_END - $SWIFT_START:.1f}')")

echo ""
green "Swift MLX finished in ${SWIFT_TIME}s"
echo ""

# ---------- step 8: quality comparison ----------

bold "=== Quality Comparison (vs Python reference) ==="
echo ""

# Python output lands in: <out>/htdemucs/<trackname>/<stem>.wav
# C++/Swift output lands in: <out>/<trackname>/<stem>.wav
PYTHON_STEM_DIR="$PYTHON_OUT/htdemucs/$TRACK_NAME"
CPP_STEM_DIR="$CPP_OUT/$TRACK_NAME"
SWIFT_STEM_DIR="$SWIFT_OUT/$TRACK_NAME"

"$BENCH_VENV/bin/python" - "$PYTHON_STEM_DIR" "$CPP_STEM_DIR" "$SWIFT_STEM_DIR" <<'PYEOF'
import sys, os
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate

ref_dir, cpp_dir, swift_dir = sys.argv[1], sys.argv[2], sys.argv[3]
stems = ["drums", "bass", "other", "vocals"]

def load_wav_float(path):
    """Load WAV and return float64 samples normalized to [-1, 1]."""
    sr, data = wavfile.read(path)
    if data.dtype == np.int16:
        return sr, data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        return sr, data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.float32 or data.dtype == np.float64:
        return sr, data.astype(np.float64)
    return sr, data.astype(np.float64)

def compute_metrics(ref, test):
    """Compute quality metrics between reference and test signals."""
    # Truncate to same length
    n = min(len(ref), len(test))
    ref = ref[:n]
    test = test[:n]

    # Signal-to-Distortion Ratio (SDR) in dB
    noise = ref - test
    ref_power = np.sum(ref ** 2)
    noise_power = np.sum(noise ** 2)
    if noise_power < 1e-30:
        sdr = float('inf')
    elif ref_power < 1e-30:
        sdr = float('-inf')
    else:
        sdr = 10.0 * np.log10(ref_power / noise_power)

    # Pearson correlation (per-channel, then average)
    corrs = []
    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)
        test = test.reshape(-1, 1)
    for ch in range(ref.shape[1]):
        r = ref[:, ch]
        t = test[:, ch]
        if np.std(r) < 1e-12 or np.std(t) < 1e-12:
            corrs.append(0.0)
        else:
            corrs.append(np.corrcoef(r, t)[0, 1])
    corr = np.mean(corrs)

    # Max absolute error
    max_err = np.max(np.abs(ref - test))

    # RMS error
    rms = np.sqrt(np.mean(noise ** 2))

    return sdr, corr, max_err, rms

def print_table(label, metrics_by_stem):
    print(f"  {label}")
    print(f"  {'Stem':<10} {'SDR (dB)':>10} {'Corr':>10} {'MaxErr':>10} {'RMS':>12}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")
    for stem, (sdr, corr, maxerr, rms) in metrics_by_stem.items():
        sdr_s = f"{sdr:.1f}" if sdr != float('inf') else "inf"
        print(f"  {stem:<10} {sdr_s:>10} {corr:>10.6f} {maxerr:>10.6f} {rms:>12.8f}")
    print()

for impl_label, impl_dir in [("C++ MLX vs Python", cpp_dir),
                               ("Swift MLX vs Python", swift_dir)]:
    if not os.path.isdir(impl_dir):
        print(f"  {impl_label}: output directory not found, skipping")
        continue

    metrics = {}
    for stem in stems:
        ref_path = os.path.join(ref_dir, f"{stem}.wav")
        test_path = os.path.join(impl_dir, f"{stem}.wav")
        if not os.path.exists(ref_path):
            print(f"  Warning: {ref_path} not found")
            continue
        if not os.path.exists(test_path):
            print(f"  Warning: {test_path} not found")
            continue

        _, ref_data = load_wav_float(ref_path)
        _, test_data = load_wav_float(test_path)
        metrics[stem] = compute_metrics(ref_data, test_data)

    if metrics:
        print_table(impl_label, metrics)

PYEOF

# ---------- step 9: timing summary ----------

bold "=== Timing Summary ==="
echo "Audio:       $AUDIO_NAME (${AUDIO_DURATION}s)"
echo ""
printf "  %-12s %8s\n" "Engine" "Time"
printf "  %-12s %8s\n" "------" "----"
printf "  %-12s %7ss\n" "Python" "$PYTHON_TIME"
printf "  %-12s %7ss\n" "C++ MLX" "$CPP_TIME"
printf "  %-12s %7ss\n" "Swift MLX" "$SWIFT_TIME"
echo ""

CPP_SPEEDUP=$(python3 -c "p=$PYTHON_TIME; c=$CPP_TIME; print(f'{p/c:.1f}x' if c > 0 else 'n/a')")
SWIFT_SPEEDUP=$(python3 -c "p=$PYTHON_TIME; s=$SWIFT_TIME; print(f'{p/s:.1f}x' if s > 0 else 'n/a')")
echo "  C++ speedup:   $CPP_SPEEDUP"
echo "  Swift speedup: $SWIFT_SPEEDUP"
echo ""
echo "Output dirs:"
echo "  Python: $PYTHON_OUT"
echo "  C++:    $CPP_OUT"
echo "  Swift:  $SWIFT_OUT"
