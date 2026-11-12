"""Render pages, run real SigLIP, and save exact source-PCA wire payloads."""

import base64
import argparse
import json
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))

from app.config.settings import Settings  # noqa: E402
from app.encoding.source_pca import load_basis  # noqa: E402
from app.schemas.messages import TensorPayload  # noqa: E402
from app.tensor.serialization import contract_from_settings, decode_tensor_payload  # noqa: E402
from eval.siglip.adapter import adapt_siglip_embeddings  # noqa: E402

MODEL_PATH = ROOT / "client" / "public" / "models" / "siglip-base-patch16-224-vision-fp32.onnx"
RENDER_SCRIPT = ROOT / "client" / "scripts" / "render-inversion-pages.mjs"
MODE_CONFIGS = {
    "none": (0, 0.0, False),
    "rank1_alpha0": (1, 0.0, True),
    "rank1_alpha001": (1, 0.01, True),
    "svd_k1_alpha0": (1, 0.0, False),
    "svd_k1_alpha001": (1, 0.01, False),
    "svd_k4_alpha0": (4, 0.0, False),
    "svd_k4_alpha001": (4, 0.01, False),
    "svd_k8_alpha0": (8, 0.0, False),
    "svd_k8_alpha001": (8, 0.01, False),
}
VALIDATION_TEMPLATES = {"wallet", "billing"}


def render_pages(out_dir: Path, count: int, seed: int) -> None:
    existing = list(out_dir.glob("sample_*.png"))
    if len(existing) == count:
        return
    subprocess.run(
        [
            "node",
            str(RENDER_SCRIPT),
            "--out-dir",
            str(out_dir),
            "--count",
            str(count),
            "--seed",
            str(seed),
        ],
        cwd=ROOT / "client",
        check=True,
    )


def image_tensor(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        pixels = np.asarray(
            image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
    return ((pixels - 0.5) / 0.5).transpose(2, 0, 1)[None].astype(np.float32)


def sanitize_source(
    source: np.ndarray,
    sensitive_indices: list[int],
    rank: int,
    alpha: float,
    rank1_baseline: bool = False,
    private_basis: np.ndarray | None = None,
) -> np.ndarray:
    if not sensitive_indices:
        return source.copy()
    if rank == 0:
        return source.copy()
    selected = source[np.asarray(sensitive_indices, dtype=np.int64)]
    if private_basis is None:
        if rank1_baseline:
            mean_vector = selected.mean(axis=0)
            vector_norm = float(np.linalg.norm(mean_vector))
            if vector_norm <= 1e-10:
                return source.copy()
            private_basis = (mean_vector / vector_norm).astype(np.float32)[None, :]
        else:
            _, _, right_vectors = np.linalg.svd(selected, full_matrices=False)
            private_basis = right_vectors[: min(rank, len(selected))].astype(np.float32)
    sanitized = source - (source @ private_basis.T) @ private_basis
    if alpha:
        patch = np.arange(source.shape[0], dtype=np.float32)[:, None]
        dim = np.arange(source.shape[1], dtype=np.float32)[None, :]
        noise = (np.sin(patch * 97 + dim * 13) * np.cos(dim * 7) * alpha).astype(np.float32)
        noise -= (noise @ private_basis.T) @ private_basis
        sanitized += noise
    return sanitized.astype(np.float32)


def right_singular_vectors(rows: np.ndarray) -> np.ndarray:
    rows64 = rows.astype(np.float64, copy=False)
    eigenvalues, eigenvectors = np.linalg.eigh(rows64 @ rows64.T)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    threshold = max(1e-12, float(eigenvalues[0]) * 1e-12)
    keep = eigenvalues > threshold
    if not np.any(keep):
        return np.empty((0, rows.shape[1]), dtype=np.float32)
    vectors = (eigenvectors[:, keep].T @ rows64) / np.sqrt(eigenvalues[keep])[:, None]
    return vectors.astype(np.float32)


def encode_source_pca_payload(
    source: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    rank: int,
) -> bytes:
    coefficients = (source - mean) @ components[:rank].T
    scales = np.max(np.abs(coefficients), axis=1).astype("<f4")
    multipliers = np.divide(
        127.0,
        scales.astype(np.float64),
        out=np.zeros(scales.shape, dtype=np.float64),
        where=scales > 0,
    )
    quantized = np.clip(
        np.floor(coefficients.astype(np.float64) * multipliers[:, None] + 0.5),
        -127,
        127,
    ).astype(np.int8)
    compressed = zlib.compress(scales.tobytes() + quantized.tobytes())
    payload = TensorPayload(
        dtype="int8_quantized",
        shape=[196, rank],
        data=base64.b64encode(compressed).decode("ascii"),
        compression="zlib",
        transform="source_pca_v1",
    )
    return payload.model_dump_json(exclude_none=True).encode("utf-8")


def generate_dataset(
    out_dir: Path,
    pages_dir: Path,
    count: int,
    seed: int,
    rank: int,
) -> dict[str, object]:
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if not pages_dir.is_absolute():
        pages_dir = ROOT / pages_dir
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing FP32 SigLIP ONNX model: {MODEL_PATH}")
    if not 1 <= rank <= 128:
        raise ValueError("PCA rank must be in [1, 128]")
    render_pages(pages_dir, count, seed)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    mean, components = load_basis()
    template_splits: dict[str, str] = {}
    for index in range(count):
        metadata_path = pages_dir / f"sample_{index:04d}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        template_id = str(metadata.get("template_id", ""))
        if not template_id:
            raise ValueError(f"Page {index} is missing template_id")
        template_splits[template_id] = (
            "validation" if template_id in VALIDATION_TEMPLATES else "train"
        )
    train_templates = sorted(
        template for template, split in template_splits.items() if split == "train"
    )
    validation_templates = sorted(
        template for template, split in template_splits.items() if split == "validation"
    )
    if not train_templates or not validation_templates:
        raise ValueError(
            "Dataset must include at least one training and one validation template"
        )
    settings = Settings()
    contract = contract_from_settings(settings)
    written = []

    for index in range(count):
        basename = f"sample_{index:04d}"
        image_path = pages_dir / f"{basename}.png"
        metadata_path = pages_dir / f"{basename}.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with Image.open(image_path) as image:
            screenshot = np.asarray(
                image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
        source = session.run(
            [output_name], {input_name: image_tensor(image_path)}
        )[0][0].astype(np.float32)
        if source.shape != (196, 768):
            raise ValueError(f"Expected SigLIP source tokens [196, 768], got {source.shape}")
        V = adapt_siglip_embeddings(source)
        payloads: dict[str, np.ndarray] = {}
        rebuilt_tensors: dict[str, np.ndarray] = {}
        sensitive_indices = metadata["sensitive_source_patch_indices"]
        private_bases: dict[tuple[int, bool], np.ndarray] = {}
        right_vectors = None
        if sensitive_indices:
            selected = source[np.asarray(sensitive_indices, dtype=np.int64)]
            right_vectors = right_singular_vectors(selected)
        for mode, (mode_rank, alpha, rank1_baseline) in MODE_CONFIGS.items():
            basis = None
            if mode_rank and sensitive_indices:
                basis_key = (mode_rank, rank1_baseline)
                if basis_key not in private_bases:
                    if rank1_baseline:
                        mean_vector = selected.mean(axis=0)
                        vector_norm = float(np.linalg.norm(mean_vector))
                        private_bases[basis_key] = (
                            (mean_vector / vector_norm).astype(np.float32)[None, :]
                            if vector_norm > 1e-10
                            else np.empty((0, source.shape[1]), dtype=np.float32)
                        )
                    else:
                        if right_vectors is None:
                            raise RuntimeError("SVD basis is unavailable for sensitive source rows")
                        private_bases[basis_key] = right_vectors[
                            : min(mode_rank, len(sensitive_indices))
                        ].astype(np.float32)
                basis = private_bases[basis_key]
                if basis.size == 0:
                    basis = None
            sanitized = sanitize_source(
                source,
                sensitive_indices,
                mode_rank,
                alpha,
                rank1_baseline,
                basis,
            )
            wire_json = encode_source_pca_payload(sanitized, mean, components, rank)
            payloads[f"wire_{mode}"] = np.frombuffer(wire_json, dtype=np.uint8)
            payload = TensorPayload.model_validate_json(wire_json)
            rebuilt_tensors[f"Vprime_{mode}"] = decode_tensor_payload(
                payload,
                contract,
                settings.vdlm_max_tensor_elements,
            ).numpy()

        record_meta = {
            "index": index,
            "seed": metadata["seed"],
            "template_id": metadata["template_id"],
            "page_type": metadata["template_id"],
            "card_number": metadata["card_number"],
            "password": metadata["password"],
            "cvv": metadata["cvv"],
            "sensitive_boxes": metadata["sensitive_boxes"],
            "train_val_split": template_splits[metadata["template_id"]],
        }
        np.savez(
            out_dir / f"{basename}.npz",
            screenshot=screenshot,
            I_sens=np.asarray(metadata["sensitive_patch_indices"], dtype=np.int32),
            V=V.astype(np.float32),
            metadata=np.asarray(json.dumps(record_meta)),
            **payloads,
            **rebuilt_tensors,
        )
        written.append({
            "sample": basename,
            "split": record_meta["train_val_split"],
            "template_id": record_meta["template_id"],
            "sensitive_target_patches": len(metadata["sensitive_patch_indices"]),
            "wire_bytes": {
                mode: int(payloads[f"wire_{mode}"].size) for mode in MODE_CONFIGS
            },
        })
        if (index + 1) % 25 == 0 or index + 1 == count:
            print(f"[dataset] {index + 1}/{count} real SigLIP pages")

    manifest = {
        "seed": seed,
        "page_count": count,
        "training_pages": sum(item["split"] == "train" for item in written),
        "validation_pages": sum(item["split"] == "validation" for item in written),
        "pages_dir": str(pages_dir),
        "split_strategy": "template-group holdout; pages never cross template partitions",
        "training_templates": train_templates,
        "validation_templates": validation_templates,
        "model": str(MODEL_PATH),
        "model_bytes": MODEL_PATH.stat().st_size,
        "onnxruntime": ort.__version__,
        "PCA_rank": rank,
        "wire_modes": list(MODE_CONFIGS),
        "samples": written,
    }
    (out_dir / "dataset-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "backend/eval/inversion/runs/data")
    parser.add_argument("--pages-dir", type=Path, default=ROOT / "backend/eval/inversion/runs/pages")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=26171)
    parser.add_argument("--pca-rank", type=int, default=64)
    args = parser.parse_args()
    if args.n < 2:
        parser.error("--n must be at least 2 for a page-level train/validation split")
    manifest = generate_dataset(args.out_dir, args.pages_dir, args.n, args.seed, args.pca_rank)
    print(
        f"[dataset] wrote {manifest['page_count']} samples "
        f"({manifest['training_pages']} train, {manifest['validation_pages']} validation)"
    )


if __name__ == "__main__":
    main()
