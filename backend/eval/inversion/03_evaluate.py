"""Evaluate convolutional inversion, EasyOCR recovery, and a digit-string probe."""

import argparse
import json
import math
import re
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import PIL
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from backend.eval.inversion.decoder import ConvDecoder

DEFAULT_KEYS = [
    "V",
    "Vprime_none",
    "Vprime_rank1_alpha0",
    "Vprime_rank1_alpha001",
    "Vprime_svd_k1_alpha0",
    "Vprime_svd_k1_alpha001",
    "Vprime_svd_k4_alpha0",
    "Vprime_svd_k4_alpha001",
    "Vprime_svd_k8_alpha0",
    "Vprime_svd_k8_alpha001",
]


def sensitive_mask(indices: np.ndarray) -> np.ndarray:
    mask = np.zeros((224, 224), dtype=bool)
    for index in indices:
        row, col = divmod(int(index), 24)
        y0, y1 = row * 224 // 24, (row + 1) * 224 // 24
        x0, x1 = col * 224 // 24, (col + 1) * 224 // 24
        mask[y0:y1, x0:x1] = True
    return mask


def psnr(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    diff = reference[mask] - prediction[mask]
    mse = float(np.mean(diff * diff))
    return 100.0 if mse == 0 else 20.0 * math.log10(1.0 / math.sqrt(mse))


def ssim(reference: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float:
    target = torch.from_numpy(reference.transpose(2, 0, 1)[None])
    output = torch.from_numpy(prediction.transpose(2, 0, 1)[None])
    mu_target = F.avg_pool2d(target, kernel_size=11, stride=1, padding=5)
    mu_output = F.avg_pool2d(output, kernel_size=11, stride=1, padding=5)
    target_var = F.avg_pool2d(target * target, 11, 1, 5) - mu_target * mu_target
    output_var = F.avg_pool2d(output * output, 11, 1, 5) - mu_output * mu_output
    covariance = F.avg_pool2d(target * output, 11, 1, 5) - mu_target * mu_output
    numerator = (2 * mu_target * mu_output + 0.01**2) * (2 * covariance + 0.03**2)
    denominator = (
        (mu_target * mu_target + mu_output * mu_output + 0.01**2)
        * (target_var + output_var + 0.03**2)
    )
    score = numerator / (denominator + 1e-12)
    selected = torch.from_numpy(mask)[None, None].expand_as(score)
    return float(score[selected].mean())


def char_accuracy(expected: str, actual: str) -> float:
    expected = re.sub(r"[^a-z0-9]", "", expected.lower())
    actual = re.sub(r"[^a-z0-9]", "", actual.lower())
    if not expected:
        return float(not actual)
    previous = list(range(len(actual) + 1))
    for row, left_char in enumerate(expected, start=1):
        current = [row]
        for col, right_char in enumerate(actual, start=1):
            current.append(min(
                current[-1] + 1,
                previous[col] + 1,
                previous[col - 1] + (left_char != right_char),
            ))
        previous = current
    return max(0.0, 1.0 - previous[-1] / len(expected))


def crop_for_box(image: np.ndarray, box: dict[str, object]) -> np.ndarray:
    x0 = max(0, int(float(box["x"]) * 224 / 1280))
    y0 = max(0, int(float(box["y"]) * 224 / 720))
    x1 = min(224, int(math.ceil((float(box["x"]) + float(box["width"])) * 224 / 1280)))
    y1 = min(224, int(math.ceil((float(box["y"]) + float(box["height"])) * 224 / 720)))
    crop = Image.fromarray(image[y0:y1, x0:x1]).resize((max(16, (x1 - x0) * 4), max(16, (y1 - y0) * 4)))
    return np.asarray(crop)


def run_ocr(image: np.ndarray, metadata: dict[str, object], reader) -> dict[str, float]:
    result = {}
    for key, selector in (("password", "#password"), ("card_number", "#card-number")):
        box = next(item for item in metadata["sensitive_boxes"] if item["selector"] == selector)
        recognized = " ".join(reader.readtext(crop_for_box(image, box), detail=0))
        result[key] = char_accuracy(str(metadata[key]), recognized)
    return result


def load_records(data_dir: Path, split: str) -> list[dict[str, object]]:
    records = []
    manifest = json.loads((data_dir / "dataset-manifest.json").read_text(encoding="utf-8"))
    for index in range(manifest["page_count"]):
        path = data_dir / f"sample_{index:04d}.npz"
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata["train_val_split"] != split:
                continue
            records.append({
                "path": path,
                "metadata": metadata,
                "screenshot": archive["screenshot"].astype(np.float32) / 255.0,
                "I_sens": archive["I_sens"].copy(),
            })
    if not records:
        raise ValueError(f"No {split} records in {data_dir}")
    return records


def load_latent(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        return archive[key].astype(np.float32, copy=False)


def digit_probe(records_train: list[dict[str, object]], records_test: list[dict[str, object]], key: str) -> dict[str, float]:
    x_train = np.stack([
        load_latent(record["path"], key).mean(axis=0) for record in records_train
    ]).astype(np.float64)
    x_test = np.stack([
        load_latent(record["path"], key).mean(axis=0) for record in records_test
    ]).astype(np.float64)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0) + 1e-6
    x_train = (x_train - mean) / scale
    x_test = (x_test - mean) / scale
    labels = np.asarray([
        list(str(record["metadata"]["card_number"])) for record in records_train
    ], dtype=np.int64)
    test_labels = np.asarray([
        list(str(record["metadata"]["card_number"])) for record in records_test
    ], dtype=np.int64)
    targets = np.eye(10, dtype=np.float64)[labels].reshape(len(labels), -1)
    gram = x_train @ x_train.T
    coefficients = np.linalg.solve(gram + np.eye(len(gram)) * 1.0, targets)
    predictions = (x_test @ x_train.T) @ coefficients
    predictions = predictions.reshape(len(x_test), 16, 10).argmax(axis=2)
    char_acc = float(np.mean(predictions == test_labels))
    train_majority = np.stack([
        np.bincount(labels[:, pos], minlength=10).argmax(axis=0)
        for pos in range(16)
    ])
    majority_acc = float(np.mean(test_labels == train_majority[None, :]))
    return {
        "character_accuracy": char_acc,
        "uniform_chance": 0.1,
        "training_majority_baseline": majority_acc,
        "test_pages": len(test_labels),
    }


def evaluate_key(
    key: str,
    records_train: list[dict[str, object]],
    records_test: list[dict[str, object]],
    model_dir: Path,
    reader,
    device: torch.device,
) -> dict[str, object]:
    model_path = model_dir / f"decoder_{key}.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained decoder is missing for {key}: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = ConvDecoder().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    psnr_scores = []
    ssim_scores = []
    ocr_password = []
    ocr_card = []
    with torch.no_grad():
        for record in records_test:
            latent = load_latent(record["path"], key)
            input_tensor = torch.from_numpy(
                latent.reshape(24, 24, 1152).transpose(2, 0, 1)[None].copy()
            ).to(device)
            prediction = model(input_tensor)[0].cpu().numpy().transpose(1, 2, 0)
            mask = sensitive_mask(record["I_sens"])
            psnr_scores.append(psnr(record["screenshot"], prediction, mask))
            ssim_scores.append(ssim(record["screenshot"], prediction, mask))
            if reader is not None:
                ocr = run_ocr(
                    (prediction * 255).clip(0, 255).astype(np.uint8),
                    record["metadata"],
                    reader,
                )
                ocr_password.append(ocr["password"])
                ocr_card.append(ocr["card_number"])
    digit_attack = digit_probe(records_train, records_test, key)
    return {
        "decoder_epochs": checkpoint["epochs"],
        "psnr_sensitive_mean_db": float(np.mean(psnr_scores)),
        "ssim_sensitive_mean": float(np.mean(ssim_scores)),
        "easyocr_password_character_accuracy": float(np.mean(ocr_password)) if reader else None,
        "easyocr_card_character_accuracy": float(np.mean(ocr_card)) if reader else None,
        "direct_card_digit_attack": digit_attack,
        "test_pages": len(records_test),
    }


def write_chart(path: Path, labels: list[str], values: list[float], title: str) -> None:
    width, height = 1200, 560
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), title, fill="#111827")
    plot_top, plot_bottom = 60, height - 90
    max_value = max(values, default=1.0) or 1.0
    bar_width = max(8, (width - 80) // max(1, len(values)) - 12)
    for index, (label, value) in enumerate(zip(labels, values)):
        x0 = 40 + index * (bar_width + 12)
        bar_height = int((plot_bottom - plot_top) * max(0.0, value) / max_value)
        draw.rectangle((x0, plot_bottom - bar_height, x0 + bar_width, plot_bottom), fill="#2563eb")
        draw.text((x0, plot_bottom + 8), label.replace("Vprime_", "").replace("_", "\n"), fill="#1f2937")
        draw.text((x0, plot_bottom - bar_height - 18), f"{value:.3f}", fill="#111827")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def evaluate(
    data_dir: Path,
    model_dir: Path,
    out_dir: Path,
    sanitizer_keys: list[str] | None = None,
    skip_ocr: bool = False,
) -> dict[str, object]:
    keys = sanitizer_keys or DEFAULT_KEYS
    train_records = load_records(data_dir, "train")
    test_records = load_records(data_dir, "validation")
    reader = None
    easyocr_version = None
    if not skip_ocr:
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError(
                "EasyOCR is required; install backend[evaluation] or pass --skip-ocr explicitly"
            ) from exc
        reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
        easyocr_version = version("easyocr")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {
        key: evaluate_key(key, train_records, test_records, model_dir, reader, device)
        for key in keys
    }
    manifest_path = data_dir / "dataset-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = {
        "setup": {
            "pages": manifest["page_count"],
            "train_pages": manifest["training_pages"],
            "held_out_pages": manifest["validation_pages"],
            "split_by_page": True,
            "pca_rank": manifest["PCA_rank"],
            "sanitizer_implementation": "Mean-direction rank-1 baseline; NumPy Gram-matrix eigendecomposition for k=1/4/8; client SVD uses power iteration with deflation",
            "wire_format": "source SVD/null-space -> centered PCA -> per-row int8 -> zlib/base64 JSON -> backend decode/rebuild",
            "decoder": "1x1 Conv2d 1152->64; transposed-convolution upsampling; no flattened MLP",
            "device": str(device),
            "easyocr": not skip_ocr,
            "easyocr_version": easyocr_version,
            "seed": manifest["seed"],
            "decoder_seed": 26171,
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "pytorch_version": str(torch.__version__),
            "pillow_version": PIL.__version__,
            "onnxruntime_version": manifest["onnxruntime"],
            "model_bytes": manifest["model_bytes"],
            "psnr": "Sensitive-pixel PSNR with RGB values in [0,1] and data range 1.0",
            "ssim": "11x11 average-pool SSIM, C1=0.01^2, C2=0.03^2, averaged over sensitive pixels",
            "digit_probe": "Ridge regression on global-mean latent features, lambda=1.0; 10% uniform character chance",
        },
        "results": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_inversion_results.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    labels = list(results)
    for metric, filename, title in (
        ("psnr_sensitive_mean_db", "inversion-psnr.png", "Sensitive-region PSNR (dB)"),
        ("ssim_sensitive_mean", "inversion-ssim.png", "Sensitive-region SSIM"),
    ):
        write_chart(
            out_dir / filename,
            labels,
            [float(results[key][metric]) for key in labels],
            title,
        )
    write_chart(
        out_dir / "inversion-digit-attack.png",
        labels,
        [float(results[key]["direct_card_digit_attack"]["character_accuracy"]) for key in labels],
        "Direct card-digit classifier (10% uniform chance)",
    )
    if not skip_ocr:
        write_chart(
            out_dir / "inversion-easyocr.png",
            labels,
            [float(results[key]["easyocr_card_character_accuracy"]) for key in labels],
            "EasyOCR card-number character accuracy",
        )
    markdown = [
        "# Inversion attack benchmark",
        "",
        f"Measured on {manifest['page_count']} Playwright-rendered pages "
        f"({manifest['training_pages']} train / {manifest['validation_pages']} held out), "
        f"with the PCA rank-{manifest['PCA_rank']} wire path and {device}.",
        "",
        "Every reported metric uses held-out pages only. V' is produced by source-grid sanitization, "
        "PCA projection, int8 quantization, zlib/base64 serialization, and the backend decoder's "
        "source-grid reconstruction plus 24x24 adapter.",
        "",
        f"Setup: seed {manifest['seed']}; {manifest['training_pages']} train / "
        f"{manifest['validation_pages']} held-out pages; Python {sys.version.split()[0]}, "
        f"NumPy {np.__version__}, PyTorch {torch.__version__}, ONNX Runtime "
        f"{manifest['onnxruntime']}; model size {manifest['model_bytes']:,} bytes. "
        f"Decoder seed is 26171; EasyOCR "
        f"{'not run' if easyocr_version is None else easyocr_version}.",
        "",
        "| Representation | PSNR (dB) | SSIM | EasyOCR password char acc. | EasyOCR card char acc. | Digit attack char acc. | Chance | Train-majority baseline |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, entry in results.items():
        ocr_password = entry["easyocr_password_character_accuracy"]
        ocr_card = entry["easyocr_card_character_accuracy"]
        digit = entry["direct_card_digit_attack"]
        markdown.append(
            f"| {key} | {entry['psnr_sensitive_mean_db']:.3f} | "
            f"{entry['ssim_sensitive_mean']:.4f} | "
            f"{'not run' if ocr_password is None else f'{ocr_password:.3f}'} | "
            f"{'not run' if ocr_card is None else f'{ocr_card:.3f}'} | "
            f"{digit['character_accuracy']:.3f} | 0.100 | "
            f"{digit['training_majority_baseline']:.3f} |"
        )
    markdown.extend([
        "",
        "## Charts",
        "",
        "![Sensitive-region PSNR](./inversion-psnr.png)",
        "",
        "![Sensitive-region SSIM](./inversion-ssim.png)",
        "",
        "![Direct card-digit attack](./inversion-digit-attack.png)",
        "",
        "## Interpretation",
        "",
        "A sanitization result is not a privacy guarantee. Compare the direct digit attack to its "
        "10% uniform character baseline and the EasyOCR accuracies to the unsanitized V baseline. "
        "This benchmark covers synthetic pages and must not be generalized to arbitrary websites.",
        "",
    ])
    (out_dir / "EVAL_INVERSION.md").write_text("\n".join(markdown), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("docs"))
    parser.add_argument("--sanitizers", nargs="+", default=None)
    parser.add_argument("--skip-ocr", action="store_true")
    args = parser.parse_args()
    report = evaluate(
        args.data_dir,
        args.model_dir,
        args.out_dir,
        args.sanitizers,
        args.skip_ocr,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
