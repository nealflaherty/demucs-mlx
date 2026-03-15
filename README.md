# Demucs MLX

A C++ port of [Demucs](https://github.com/facebookresearch/demucs) (HTDemucs) using Apple's [MLX](https://github.com/ml-explore/mlx) framework for native Apple Silicon inference.

Separates audio into four stems: drums, bass, other, and vocals.

## Features

- Native C++ with Metal acceleration via MLX
- No Python runtime required
- HTDemucs model with full transformer cross-attention
- Output formats: WAV (16/24-bit, float32), M4A (AAC), FLAC, ALAC
- Two-stems mode (`--two-stems vocals` for vocal/instrumental split)
- Overlap-add chunked inference for arbitrary-length audio
- Random shift averaging for improved quality

## Requirements

- macOS (Apple Silicon recommended)
- CMake 3.24+
- C++17 compiler

## Build

```bash
git submodule update --init --recursive
mkdir -p build && cd build
cmake ..
make -j
```

## Usage

```bash
# Basic separation (outputs WAV stems)
./build/demucs_separate audio.mp3

# Extract just vocals + instrumental
./build/demucs_separate --two-stems vocals audio.mp3

# Output as M4A (AAC) with custom bitrate
./build/demucs_separate --m4a --bitrate 320 audio.mp3

# Output as FLAC lossless
./build/demucs_separate --flac audio.mp3

# 24-bit WAV output
./build/demucs_separate --int24 audio.mp3

# Custom output directory and filename pattern
./build/demucs_separate -o my_output --filename "{track}/{stem}.{ext}" audio.mp3
```

### Options

```
-o, --out DIR          Output directory (default: separated/htdemucs)
--model PATH           Model weights path
--filename PATTERN     Output filename pattern ({track}, {trackext}, {stem}, {ext})
--shifts N             Random shifts for quality averaging (default: 1)
--overlap F            Overlap between chunks (default: 0.25)
--no-split             Process entire track at once (more memory)
--segment N            Segment length in seconds
--clip-mode MODE       Clipping: rescale, clamp, none (default: rescale)
--two-stems STEM       Separate into STEM and no_STEM only
--other-method METHOD  Complement method: add, minus, none (default: add)
--wav / --m4a / --flac / --alac   Output format (default: wav)
--int24                24-bit integer WAV
--float32              32-bit float WAV
--bitrate N            AAC bitrate in kbps (default: 256)
```

## Model Weights

Download and convert the pretrained HTDemucs weights:

```bash
./tools/download_model.sh
```

This downloads the model from Meta's CDN and converts it to SafeTensors
format at `models/htdemucs.safetensors`. Requires Python 3 with `torch`
and `safetensors` (the script creates a temporary venv automatically).

The model weights are from the original [Demucs](https://github.com/facebookresearch/demucs) project by Meta Research.

## Project Structure

```
demucs-mlx/
├── demucs/           # C++ source (mirrors Python module structure)
├── tests/            # Verification tests
├── scripts/          # Python reference data generators
├── third_party/      # MLX submodule
└── CMakeLists.txt
```

## Acknowledgments

This project is a C++ port of [Demucs](https://github.com/facebookresearch/demucs) by Meta Research, originally created by Alexandre Défossez. The model architecture and pretrained weights are from the original project.

## License

MIT — see [LICENSE](LICENSE) for details.
