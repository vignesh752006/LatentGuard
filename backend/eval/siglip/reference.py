#!/usr/bin/env python
"""Generate fixed SigLIP vision-tower reference arrays for the five fixtures."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import onnx
import onnxconverter_common
import onnxruntime
import torch
import transformers
from PIL import Image
from transformers import SiglipVisionModel

MODEL_ID = "google/siglip-base-patch16-224"
SEED = 0xDEADBEEF
HERE = Path(__file__).parent
IMAGE_DIR = HERE / "fixtures" / "images"
OUTPUT_DIR = HERE / "fixtures" / "reference"
IMAGE_FILES = (
    "login.png",
    "checkout.png",
    "natural-iss073e0763824.jpg",
    "natural-iss009e05944.jpg",
    "natural-iss009e05953.jpg",
)


def load_pixels(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - 0.5) / 0.5
    return torch.from_numpy(pixels.transpose(2, 0, 1).copy()).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-fixtures", action="store_true")
    args = parser.parse_args()
    if not args.write_fixtures:
        parser.error("Pass --write-fixtures to create the reference .npy arrays")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    model = SiglipVisionModel.from_pretrained(MODEL_ID).eval().cpu().float()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for filename in IMAGE_FILES:
            path = IMAGE_DIR / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing test image: {path}")
            output = model(pixel_values=load_pixels(path)).last_hidden_state
            if tuple(output.shape) != (1, 196, 768):
                raise RuntimeError(f"Unexpected SigLIP output shape: {tuple(output.shape)}")
            np.save(OUTPUT_DIR / f"{Path(filename).stem}.npy", output[0].numpy().astype(np.float32))
            print(f"{filename}: {tuple(output.shape[1:])}")

    metadata = {
        "seed": SEED,
        "model_id": MODEL_ID,
        "input": "RGB, PIL bilinear resize 224x224, float32 /255, normalize (x-0.5)/0.5, CHW",
        "reference_output": "SiglipVisionModel.last_hidden_state, [196,768], float32, CPU",
        "versions": {
            "python": __import__("platform").python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "pillow": Image.__version__ if hasattr(Image, "__version__") else __import__("PIL").__version__,
            "onnx": onnx.__version__,
            "onnxconverter_common": onnxconverter_common.__version__,
            "onnxruntime": onnxruntime.__version__,
        },
        "images": [
            {"file": name, "source": "Task 2 demo screenshot" if name.endswith(".png") else "NASA image API asset"}
            for name in IMAGE_FILES
        ],
    }
    (HERE / "fixtures" / "fixture-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
