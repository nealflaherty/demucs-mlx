#!/usr/bin/env python3
"""Download and convert Demucs HTDemucs model to SafeTensors format.

Downloads the pretrained HTDemucs model from Meta's CDN and converts
the PyTorch state dict to SafeTensors format for use by the C++ code.
"""

import argparse
import sys
from pathlib import Path

MODELS = {
    "htdemucs": {
        "url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th",
        "output": "htdemucs.safetensors",
    },
    "htdemucs_6s": {
        "url": "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th",
        "output": "htdemucs_6s.safetensors",
    },
    "htdemucs_ft": {
        "url": None,  # bag of models, not a single file
        "output": None,
    },
}

DEFAULT_MODEL = "htdemucs"


def download_and_convert(model_name: str, dest_dir: Path) -> Path:
    """Download .th from Meta CDN and convert to .safetensors."""
    try:
        import torch
    except ImportError:
        print("Error: torch not found. Install with: pip install torch")
        sys.exit(1)

    try:
        from safetensors.torch import save_file
    except ImportError:
        print("Error: safetensors not found. Install with: pip install safetensors")
        sys.exit(1)

    info = MODELS.get(model_name)
    if not info or not info["url"]:
        print(f"Error: unknown or unsupported model '{model_name}'")
        print(f"Available models: {', '.join(k for k, v in MODELS.items() if v['url'])}")
        sys.exit(1)

    url = info["url"]
    output_name = info["output"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / output_name

    if dest_path.exists():
        print(f"Model already exists: {dest_path}")
        return dest_path

    print(f"Downloading {model_name} from {url}...")
    pkg = torch.hub.load_state_dict_from_url(
        url, map_location="cpu", check_hash=True, weights_only=False
    )

    # The .th file contains a full model checkpoint with 'state' key
    if isinstance(pkg, dict) and "state" in pkg:
        state_dict = pkg["state"]
    elif hasattr(pkg, "state_dict"):
        state_dict = pkg.state_dict()
    else:
        state_dict = pkg

    print(f"Converting {len(state_dict)} tensors to SafeTensors...")
    # Cast all tensors to float32 — the original checkpoint is float16
    # but MLX (unlike PyTorch) doesn't auto-upcast during computation,
    # causing overflow/NaN in the encoder layers.
    state_dict = {k: v.float() for k, v in state_dict.items()}
    save_file(state_dict, str(dest_path))

    print(f"Saved: {dest_path} ({dest_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return dest_path


def main():
    parser = argparse.ArgumentParser(description="Download and convert Demucs models")
    parser.add_argument(
        "-n", "--name",
        default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "models",
        help="Output directory (default: models/)",
    )
    args = parser.parse_args()

    download_and_convert(args.name, args.output_dir)


if __name__ == "__main__":
    main()
