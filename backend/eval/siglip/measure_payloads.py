"""Measure compact wire sizes and random-projection fidelity on real SigLIP fixtures."""

import json
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[2]))

from backend.eval.siglip.adapter import adapt_siglip_embeddings

REFERENCE_DIR = HERE / "fixtures" / "reference"
IMAGE_NAMES = (
    "login",
    "checkout",
    "natural-iss073e0763824",
    "natural-iss009e05944",
    "natural-iss009e05953",
)
TARGET_DIMS = (256, 128, 64)
SEED = 0xDEADBEEF


def quantize_rows(
    values: np.ndarray,
    keep_fraction: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = values.astype(np.float32, copy=True)
    if keep_fraction < 1:
        keep_count = max(1, round(matrix.shape[1] * keep_fraction))
        keep_indices = np.argpartition(
            np.abs(matrix),
            matrix.shape[1] - keep_count,
            axis=1,
        )[:, -keep_count:]
        keep_mask = np.zeros(matrix.shape, dtype=bool)
        np.put_along_axis(keep_mask, keep_indices, True, axis=1)
        matrix[~keep_mask] = 0

    scales = np.max(np.abs(matrix), axis=1).astype(np.float32)
    quantized = np.zeros(matrix.shape, dtype=np.int8)
    nonzero = scales > 0
    quantized[nonzero] = np.clip(
        np.rint(matrix[nonzero] / scales[nonzero, None] * 127),
        -127,
        127,
    ).astype(np.int8)
    return scales, quantized


def compact_size(values: np.ndarray, keep_fraction: float = 1.0) -> int:
    scales, quantized = quantize_rows(values, keep_fraction)
    packed = scales.tobytes() + quantized.tobytes()
    return len(zlib.compress(packed))


def quantization_metrics(values: np.ndarray, keep_fraction: float) -> dict[str, float]:
    scales, quantized = quantize_rows(values, keep_fraction)
    reconstructed = scales[:, None] * (quantized.astype(np.float32) / 127)
    cosine = np.sum(values * reconstructed, axis=1) / (
        np.linalg.norm(values, axis=1) * np.linalg.norm(reconstructed, axis=1)
    )
    return {
        "mean_patch_cosine": float(cosine.mean()),
        "minimum_patch_cosine": float(cosine.min()),
    }


def projection_metrics(
    values: np.ndarray,
    basis: np.ndarray,
    dimensions: int,
) -> dict[str, float]:
    reduced = values @ basis[:, :dimensions]
    reconstructed = reduced @ basis[:, :dimensions].T
    cosine = np.sum(values * reconstructed, axis=1) / (
        np.linalg.norm(values, axis=1) * np.linalg.norm(reconstructed, axis=1)
    )
    return {
        "mean_patch_cosine": float(cosine.mean()),
        "minimum_patch_cosine": float(cosine.min()),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    random_matrix = rng.standard_normal((1152, max(TARGET_DIMS)), dtype=np.float32)
    random_basis, _ = np.linalg.qr(random_matrix, mode="reduced")
    results: dict[str, dict[str, object]] = {}

    for name in IMAGE_NAMES:
        reference = np.load(REFERENCE_DIR / f"{name}.npy").astype(np.float32)
        embeddings = adapt_siglip_embeddings(reference)
        sizes = {
            "fp16_zlib": len(zlib.compress(embeddings.astype(np.float16).tobytes())),
            "int8_zlib": compact_size(embeddings),
            "int8_top10_zlib": compact_size(embeddings, keep_fraction=0.1),
        }
        projection_scores = {}
        for dimensions in TARGET_DIMS:
            reduced = embeddings @ random_basis[:, :dimensions]
            sizes[f"int8_random_{dimensions}_zlib"] = compact_size(reduced)
            projection_scores[str(dimensions)] = projection_metrics(
                embeddings,
                random_basis,
                dimensions,
            )
        results[name] = {
            "compressed_binary_bytes": sizes,
            "base64_data_bytes": {
                name: 4 * ((size + 2) // 3) for name, size in sizes.items()
            },
            "quantization_cosine": {
                "int8": quantization_metrics(embeddings, 1),
                "int8_top10": quantization_metrics(embeddings, 0.1),
            },
            "projection_cosine": projection_scores,
        }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
