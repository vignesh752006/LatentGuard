#!/usr/bin/env python3
"""
download-siglip-onnx.py — Download and convert SigLIP-Base-Patch16-224 to FP16 ONNX.

Requirements:
    pip install optimum[exporters] transformers torch onnx onnxruntime

Usage:
    python scripts/download-siglip-onnx.py [--out-dir client/public/models]

After running, place the generated .onnx file in `client/public/models/` and
update the extension's onnxModelUrl setting to point to it.

The script downloads the vision encoder only (no text encoder needed for VDLM).
Output file: siglip-base-patch16-224-vision.onnx (~180 MB fp32, ~90 MB fp16)

NOTE: This downloads from HuggingFace Hub. Set HF_TOKEN environment variable
      if you need authenticated access.
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download SigLIP ONNX model")
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).parent.parent / "client" / "public" / "models"),
        help="Output directory for the ONNX model file",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Convert to FP16 (default: True)",
    )
    parser.add_argument(
        "--model-id",
        default="google/siglip-base-patch16-224",
        help="HuggingFace model ID",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "siglip-base-patch16-224-vision.onnx"

    if out_file.exists():
        print(f"[INFO] Model already exists at {out_file}")
        print(f"[INFO] Set onnxModelUrl to: http://localhost:<port>/models/{out_file.name}")
        return 0

    try:
        from optimum.exporters.onnx import main_export
        from optimum.exporters.onnx.model_configs import SiglipVisionModelOnnxConfig
    except ImportError:
        print("[ERROR] optimum not installed. Run:")
        print("    pip install 'optimum[exporters]' transformers torch")
        return 1

    print(f"[INFO] Downloading and exporting {args.model_id} to ONNX...")
    print(f"[INFO] Output: {out_dir}")

    try:
        main_export(
            model_name_or_path=args.model_id,
            output=out_dir,
            task="feature-extraction",
            opset=17,
            framework="pt",
            fp16=args.fp16,
            device="cpu",
            no_post_process=False,
        )
    except Exception as exc:
        print(f"[ERROR] Export failed: {exc}")
        print()
        print("Manual fallback steps:")
        print("  1. pip install transformers optimum[exporters] onnxruntime")
        print("  2. python -c \"")
        print("     from optimum.exporters.onnx import main_export")
        print(f"     main_export('{args.model_id}', output='{out_dir}',")
        print("         task='feature-extraction', opset=17, fp16=True)")
        print("     \"")
        return 1

    # Rename the vision model output if needed
    vision_model = out_dir / "model.onnx"
    if vision_model.exists() and not out_file.exists():
        vision_model.rename(out_file)
        print(f"[INFO] Renamed model.onnx → {out_file.name}")

    if out_file.exists():
        size_mb = out_file.stat().st_size / 1024 / 1024
        print(f"\n✓ Success! Model: {out_file} ({size_mb:.1f} MB)")
        print(f"\nNext steps:")
        print(f"  1. Serve the model (or copy to extension's web-accessible resources):")
        print(f"     python -m http.server 9876 --directory {out_dir}")
        print(f"  2. In extension settings, set onnxModelUrl:")
        print(f"     http://localhost:9876/{out_file.name}")
        print(f"  3. Set visionMode to 'onnx' in extension settings.")
        return 0
    else:
        print("[ERROR] Export succeeded but output file not found.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

