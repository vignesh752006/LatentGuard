"""Fit the source-grid PCA basis and measure the complete payload path."""

import argparse
import base64
import hashlib
import json
import random
import shutil
import sys
import zlib
import platform
from pathlib import Path

import numpy as np
import onnxruntime as ort
import PIL
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))

from app.config.settings import Settings  # noqa: E402
from app.encoding.source_pca import (  # noqa: E402
    BASIS_PATH,
    MAX_COMPONENTS,
    SOURCE_DIM,
    SOURCE_PATCHES,
)
from app.schemas.messages import TensorPayload  # noqa: E402
from app.tensor.serialization import contract_from_settings, decode_tensor_payload  # noqa: E402
from eval.siglip.adapter import adapt_siglip_embeddings  # noqa: E402

TRAINING_SEED = 0x26171
DEFAULT_SYNTHETIC_PAGES = ROOT / "backend" / "eval" / "siglip" / "fixtures" / "pca-pages"
DEFAULT_PUBLIC_PAGES = ROOT / "backend" / "eval" / "siglip" / "fixtures" / "public-pages"
MODEL_PATH = ROOT / "client" / "public" / "models" / "siglip-base-patch16-224-vision-fp32.onnx"
SUMMARY_PATH = Path(__file__).resolve().parent / "fixtures" / "source-pca-metadata.json"
TARGET_BASIS_PATH = ROOT / "client" / "public" / "pca" / "source-pca-v1.bin"


def load_records(synthetic_dir: Path, public_dir: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for image_path in sorted(synthetic_dir.glob("sample_*.png")):
        metadata_path = image_path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        records.append({
            "image_path": image_path,
            "page_id": image_path.stem,
            "source": "synthetic",
            "sensitive_source_patch_indices": metadata["sensitive_source_patch_indices"],
        })

    manifest_path = public_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Public-page capture manifest is missing: {manifest_path}; "
            "run client/scripts/capture-public-pca-pages.mjs first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for page in manifest["pages"]:
        records.append({
            "image_path": public_dir / page["filename"],
            "page_id": page["filename"],
            "source": "public",
            "url": page["url"],
            "capture_timestamp": manifest.get("captured_at"),
            "sensitive_source_patch_indices": [],
        })
    return records


def image_tensor(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        pixels = np.asarray(
            image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
    return ((pixels - 0.5) / 0.5).transpose(2, 0, 1)[None].astype(np.float32)


def embed_records(records: list[dict[str, object]]) -> list[np.ndarray]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing FP32 SigLIP ONNX model: {MODEL_PATH}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    embeddings = []
    for index, record in enumerate(records):
        output = session.run(
            [output_name], {input_name: image_tensor(record["image_path"])}
        )[0][0]
        if output.shape != (SOURCE_PATCHES, SOURCE_DIM):
            raise ValueError(f"Expected [196, 768] source tokens, got {output.shape}")
        embeddings.append(np.asarray(output, dtype=np.float32))
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            print(f"[pca] embedded {index + 1}/{len(records)} pages")
    return embeddings


def sanitize_source(
    source: np.ndarray,
    sensitive_indices: list[int],
    alpha: float,
) -> np.ndarray:
    if not sensitive_indices:
        return source.copy()
    selected = source[np.asarray(sensitive_indices, dtype=np.int64)]
    _, _, right_vectors = np.linalg.svd(selected, full_matrices=False)
    private_basis = right_vectors[: min(4, len(sensitive_indices))].astype(np.float32)
    sanitized = source - (source @ private_basis.T) @ private_basis
    if alpha:
        patch = np.arange(SOURCE_PATCHES, dtype=np.float32)[:, None]
        dim = np.arange(SOURCE_DIM, dtype=np.float32)[None, :]
        noise = (np.sin(patch * 97 + dim * 13) * np.cos(dim * 7) * alpha).astype(np.float32)
        noise -= (noise @ private_basis.T) @ private_basis
        sanitized += noise
    return sanitized.astype(np.float32)


def make_wire_payload(
    coefficients: np.ndarray,
    component_count: int,
) -> tuple[TensorPayload, int]:
    values = np.asarray(coefficients[:, :component_count], dtype=np.float32)
    scales = np.max(np.abs(values), axis=1).astype("<f4")
    multipliers = np.divide(
        127.0,
        scales.astype(np.float64),
        out=np.zeros(scales.shape, dtype=np.float64),
        where=scales > 0,
    )
    quantized = np.clip(
        np.floor(values.astype(np.float64) * multipliers[:, None] + 0.5),
        -127,
        127,
    ).astype(np.int8)
    packed = scales.tobytes() + quantized.tobytes()
    compressed = zlib.compress(packed)
    payload = TensorPayload(
        dtype="int8_quantized",
        shape=[SOURCE_PATCHES, component_count],
        data=base64.b64encode(compressed).decode("ascii"),
        compression="zlib",
        transform="source_pca_v1",
    )
    return payload, len(payload.model_dump_json(exclude_none=True).encode("utf-8"))


def mean_patch_cosine(left: np.ndarray, right: np.ndarray) -> float:
    dots = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1) + 1e-12
    return float(np.mean(dots / denominator))


def fit_and_measure(records: list[dict[str, object]], embeddings: list[np.ndarray]) -> dict[str, object]:
    order = list(range(len(records)))
    random.Random(TRAINING_SEED).shuffle(order)
    heldout_count = max(1, int(np.ceil(len(order) * 0.2)))
    heldout_indices = order[:heldout_count]
    training_indices = order[heldout_count:]
    if len(training_indices) < MAX_COMPONENTS:
        raise ValueError("At least 129 training pages are required to fit rank-128 PCA")

    train_matrix = np.concatenate([embeddings[i] for i in training_indices]).astype(np.float64)
    mean = train_matrix.mean(axis=0)
    centered = train_matrix - mean
    covariance = centered.T @ centered / max(1, len(train_matrix) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order_by_variance = np.argsort(eigenvalues)[::-1][:MAX_COMPONENTS]
    components = eigenvectors[:, order_by_variance].T
    for component in components:
        pivot = int(np.argmax(np.abs(component)))
        if component[pivot] < 0:
            component *= -1

    mean32 = mean.astype("<f4")
    components32 = components.astype("<f4")
    packed_basis = np.concatenate((mean32, components32.reshape(-1))).astype("<f4")
    BASIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    packed_basis.tofile(BASIS_PATH)
    TARGET_BASIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(BASIS_PATH, TARGET_BASIS_PATH)

    settings = Settings()
    contract = contract_from_settings(settings)
    measurements: dict[str, dict[str, float]] = {}
    per_page: dict[str, dict[str, float]] = {}
    for rank in (64, 128):
        for alpha in (0.0, 0.01):
            wire_sizes = []
            cosines = []
            key = f"rank_{rank}_alpha_{alpha:g}"
            per_page[key] = {}
            for index in heldout_indices:
                record = records[index]
                source = embeddings[index]
                sanitized = sanitize_source(
                    source,
                    record["sensitive_source_patch_indices"],
                    alpha,
                )
                coefficients = (sanitized - mean32) @ components32[:rank].T
                payload, wire_size = make_wire_payload(coefficients, rank)
                decoded = decode_tensor_payload(
                    payload,
                    contract,
                    settings.vdlm_max_tensor_elements,
                ).numpy()
                expected = adapt_siglip_embeddings(sanitized)
                cosine = mean_patch_cosine(expected, decoded)
                wire_sizes.append(wire_size)
                cosines.append(cosine)
                per_page[key][record["page_id"]] = cosine
            measurements[key] = {
                "held_out_tensor_json_bytes_mean": float(np.mean(wire_sizes)),
                "held_out_tensor_json_bytes_min": float(np.min(wire_sizes)),
                "held_out_tensor_json_bytes_max": float(np.max(wire_sizes)),
                "held_out_mean_patch_cosine": float(np.mean(cosines)),
                "held_out_page_cosine_min": float(np.min(cosines)),
                "held_out_page_cosine_max": float(np.max(cosines)),
            }

    return {
        "version": "source_pca_v1",
        "training_seed": TRAINING_SEED,
        "total_pages": len(records),
        "training_pages": len(training_indices),
        "held_out_pages": len(heldout_indices),
        "training_split_by_page": True,
        "synthetic_pages": sum(record["source"] == "synthetic" for record in records),
        "public_pages": sum(record["source"] == "public" for record in records),
        "public_page_urls": [record["url"] for record in records if "url" in record],
        "public_capture_timestamp": next(
            (record["capture_timestamp"] for record in records if "capture_timestamp" in record),
            None,
        ),
        "source_tokens_used_for_fit": int(train_matrix.shape[0]),
        "source_dimension": SOURCE_DIM,
        "basis_rank": MAX_COMPONENTS,
        "basis_file_bytes": BASIS_PATH.stat().st_size,
        "basis_sha256": hashlib.sha256(BASIS_PATH.read_bytes()).hexdigest(),
        "input_model": "google/siglip-base-patch16-224 ONNX FP32",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "onnxruntime_version": ort.__version__,
        "pillow_version": PIL.__version__,
        "held_out_measurements": measurements,
        "held_out_page_cosines": per_page,
        "source_pca_wire_bytes_exclude_websocket_envelope": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-pages-dir", type=Path, default=DEFAULT_SYNTHETIC_PAGES)
    parser.add_argument("--public-pages-dir", type=Path, default=DEFAULT_PUBLIC_PAGES)
    parser.add_argument("--minimum-pages", type=int, default=300)
    args = parser.parse_args()

    records = load_records(args.synthetic_pages_dir, args.public_pages_dir)
    if len(records) < args.minimum_pages:
        raise ValueError(
            f"Need at least {args.minimum_pages} pages including public captures; "
            f"found {len(records)} ({sum(record['source'] == 'synthetic' for record in records)} synthetic, "
            f"{sum(record['source'] == 'public' for record in records)} public)"
        )
    if sum(record["source"] == "public" for record in records) < 20:
        raise ValueError("At least 20 Playwright-captured public pages are required")

    embeddings = embed_records(records)
    metadata = fit_and_measure(records, embeddings)
    SUMMARY_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata["held_out_measurements"], indent=2))
    print(
        f"[pca] basis saved to {BASIS_PATH}; "
        f"training={metadata['training_pages']} held-out={metadata['held_out_pages']}"
    )


if __name__ == "__main__":
    main()
