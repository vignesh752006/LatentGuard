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
from PIL import Image, ImageDraw, ImageOps
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from backend.eval.inversion.decoder import ConvDecoder
from backend.eval.inversion.digit_probe import DigitGridProbe
from backend.eval.siglip.adapter import adapt_siglip_embeddings

VISION_MODEL_PATH = (
    ROOT / "client" / "public" / "models" / "siglip-base-patch16-224-vision-fp32.onnx"
)

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
    height, width = image.shape[:2]
    scale_x = width / 1280
    scale_y = height / 720
    x0 = max(0, int(float(box["x"]) * scale_x))
    y0 = max(0, int(float(box["y"]) * scale_y))
    x1 = min(width, int(math.ceil((float(box["x"]) + float(box["width"])) * scale_x)))
    y1 = min(height, int(math.ceil((float(box["y"]) + float(box["height"])) * scale_y)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Sensitive OCR crop is empty for box {box}")
    crop = Image.fromarray(image[y0:y1, x0:x1]).convert("L")
    crop = ImageOps.autocontrast(crop)
    if crop.height < 128:
        scale = 128 / crop.height
        crop = crop.resize(
            (max(16, round(crop.width * scale)), 128),
            Image.Resampling.LANCZOS,
        )
    return np.asarray(crop)


def run_ocr(image: np.ndarray, metadata: dict[str, object], reader) -> dict[str, float]:
    result = {}
    for key, selector in (("password", "#password"), ("card_number", "#card-number")):
        box = next(item for item in metadata["sensitive_boxes"] if item["selector"] == selector)
        allowlist = (
            "0123456789"
            if key == "card_number"
            else "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
        )
        recognized = " ".join(
            reader.readtext(crop_for_box(image, box), detail=0, allowlist=allowlist)
        )
        result[key] = char_accuracy(str(metadata[key]), recognized)
    return result


def load_records(data_dir: Path, split: str) -> list[dict[str, object]]:
    records = []
    manifest = json.loads((data_dir / "dataset-manifest.json").read_text(encoding="utf-8"))
    pages_dir = Path(manifest["pages_dir"])
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
                "original_path": pages_dir / f"sample_{index:04d}.png",
            })
    if not records:
        raise ValueError(f"No {split} records in {data_dir}")
    return records


def load_latent(path: Path, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        return archive[key].astype(np.float32, copy=False)


def _digit_labels(records: list[dict[str, object]]) -> np.ndarray:
    labels = np.asarray(
        [list(str(record["metadata"]["card_number"])) for record in records],
        dtype=np.int64,
    )
    if labels.ndim != 2 or labels.shape[1] != 16:
        raise ValueError("Expected 16-digit card labels for every benchmark page")
    return labels


def _feature_grid(records: list[dict[str, object]], key: str) -> np.ndarray:
    return np.stack([
        load_latent(record["path"], key)
        .reshape(24, 24, 1152)
        .transpose(2, 0, 1)
        for record in records
    ]).astype(np.float32, copy=False)


def _score_digit_probe(
    model: DigitGridProbe,
    features: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> float:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), 16):
            batch = torch.from_numpy(features[start:start + 16]).to(device)
            predictions.append(model(batch).argmax(dim=-1).cpu().numpy())
    return float(np.mean(np.concatenate(predictions) == labels))


def train_digit_probe(
    records_train: list[dict[str, object]],
    records_test: list[dict[str, object]],
    key: str,
    device: torch.device,
    epochs: int,
) -> tuple[dict[str, float], DigitGridProbe]:
    torch.manual_seed(26171)
    train_features = _feature_grid(records_train, key)
    test_features = _feature_grid(records_test, key)
    train_labels = _digit_labels(records_train)
    test_labels = _digit_labels(records_test)
    dataset = TensorDataset(
        torch.from_numpy(train_features),
        torch.from_numpy(train_labels),
    )
    generator = torch.Generator().manual_seed(26171)
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = DigitGridProbe().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = F.cross_entropy(logits.reshape(-1, 10), labels.reshape(-1))
            loss.backward()
            optimizer.step()

    accuracy = _score_digit_probe(model, test_features, test_labels, device)
    majority = np.asarray([
        np.bincount(train_labels[:, position], minlength=10).argmax()
        for position in range(16)
    ])
    majority_accuracy = float(np.mean(test_labels == majority[None, :]))
    return {
        "character_accuracy": accuracy,
        "uniform_chance": 0.1,
        "training_majority_baseline": majority_accuracy,
        "test_pages": len(test_labels),
        "probe_epochs": epochs,
        "probe_architecture": "Conv2d(1152->32,1x1)->Conv2d(32->32,3x3)->4x4 pool->MLP to 16x10",
    }, model


def _encode_image(session: ort.InferenceSession, image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    resized = np.asarray(
        Image.fromarray(image).convert("RGB").resize(
            (224, 224),
            Image.Resampling.BILINEAR,
        ),
        dtype=np.float32,
    )
    pixels = ((resized / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    tokens = session.run([output_name], {input_name: pixels.astype(np.float32)})[0][0]
    return adapt_siglip_embeddings(tokens)


def _score_grid_samples(
    model: DigitGridProbe,
    samples: list[np.ndarray],
    labels: np.ndarray,
    device: torch.device,
) -> float:
    features = np.stack([
        sample.reshape(24, 24, 1152).transpose(2, 0, 1)
        for sample in samples
    ]).astype(np.float32, copy=False)
    return _score_digit_probe(model, features, labels, device)


def positive_controls(
    records_train: list[dict[str, object]],
    records_test: list[dict[str, object]],
    model_dir: Path,
    reader,
    device: torch.device,
    probe_epochs: int,
) -> dict[str, object]:
    probe_result, probe_model = train_digit_probe(
        records_train,
        records_test,
        "V",
        device,
        probe_epochs,
    )
    digit_labels = _digit_labels(records_test)
    baseline = {
        key: float(value)
        for key, value in probe_result.items()
        if key in {"uniform_chance", "training_majority_baseline"}
    }

    original_cards: list[float] = []
    original_passwords: list[float] = []
    thumbnail_cards: list[float] = []
    thumbnail_passwords: list[float] = []
    decoder_cards: list[float] = []
    decoder_passwords: list[float] = []
    original_features: list[np.ndarray] = []
    thumbnail_features: list[np.ndarray] = []
    decoder_features: list[np.ndarray] = []
    model_path = model_dir / "decoder_V.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained unsanitized V decoder is missing: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    decoder = ConvDecoder().to(device)
    decoder.load_state_dict(checkpoint["state_dict"])
    decoder.eval()

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    if not VISION_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing SigLIP vision ONNX model: {VISION_MODEL_PATH}")
    vision = ort.InferenceSession(
        str(VISION_MODEL_PATH),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    with torch.no_grad():
        for record in records_test:
            original = np.asarray(
                Image.open(record["original_path"]).convert("RGB"),
                dtype=np.uint8,
            )
            thumbnail = (record["screenshot"] * 255).clip(0, 255).astype(np.uint8)
            latent = load_latent(record["path"], "V")
            input_tensor = torch.from_numpy(
                latent.reshape(24, 24, 1152).transpose(2, 0, 1)[None].copy()
            ).to(device)
            decoded = decoder(input_tensor)[0].cpu().numpy().transpose(1, 2, 0)
            decoded_image = (decoded * 255).clip(0, 255).astype(np.uint8)

            if reader is not None:
                original_ocr = run_ocr(original, record["metadata"], reader)
                thumbnail_ocr = run_ocr(thumbnail, record["metadata"], reader)
                decoder_ocr = run_ocr(decoded_image, record["metadata"], reader)
                original_cards.append(original_ocr["card_number"])
                original_passwords.append(original_ocr["password"])
                thumbnail_cards.append(thumbnail_ocr["card_number"])
                thumbnail_passwords.append(thumbnail_ocr["password"])
                decoder_cards.append(decoder_ocr["card_number"])
                decoder_passwords.append(decoder_ocr["password"])

            original_features.append(_encode_image(vision, original))
            thumbnail_features.append(_encode_image(vision, thumbnail))
            decoder_features.append(_encode_image(vision, decoded_image))

    controls = {
        "original_screenshot": {
            "easyocr_password_character_accuracy": (
                float(np.mean(original_passwords)) if reader is not None else None
            ),
            "easyocr_card_character_accuracy": (
                float(np.mean(original_cards)) if reader is not None else None
            ),
            "spatial_digit_probe_character_accuracy": _score_grid_samples(
                probe_model, original_features, digit_labels, device
            ),
        },
        "224_thumbnail": {
            "easyocr_password_character_accuracy": (
                float(np.mean(thumbnail_passwords)) if reader is not None else None
            ),
            "easyocr_card_character_accuracy": (
                float(np.mean(thumbnail_cards)) if reader is not None else None
            ),
            "spatial_digit_probe_character_accuracy": _score_grid_samples(
                probe_model, thumbnail_features, digit_labels, device
            ),
        },
        "unsanitized_V_decoder_output": {
            "decoder_epochs": checkpoint["epochs"],
            "easyocr_password_character_accuracy": (
                float(np.mean(decoder_passwords)) if reader is not None else None
            ),
            "easyocr_card_character_accuracy": (
                float(np.mean(decoder_cards)) if reader is not None else None
            ),
            "spatial_digit_probe_character_accuracy": _score_grid_samples(
                probe_model, decoder_features, digit_labels, device
            ),
        },
        **baseline,
        "validation_pages": len(records_test),
        "valid": False,
    }
    if reader is not None:
        card_scores = [
            float(controls[name]["easyocr_card_character_accuracy"])
            for name in (
                "original_screenshot",
                "224_thumbnail",
                "unsanitized_V_decoder_output",
            )
        ]
        password_scores = [
            float(controls[name]["easyocr_password_character_accuracy"])
            for name in (
                "original_screenshot",
                "224_thumbnail",
                "unsanitized_V_decoder_output",
            )
        ]
        probe_scores = [
            float(controls[name]["spatial_digit_probe_character_accuracy"])
            for name in (
                "original_screenshot",
                "224_thumbnail",
                "unsanitized_V_decoder_output",
            )
        ]
        controls["valid"] = (
            min(card_scores) >= 0.9
            and min(password_scores) >= 0.75
            and min(probe_scores) > float(controls["uniform_chance"])
        )
    return controls


def evaluate_key(
    key: str,
    records_train: list[dict[str, object]],
    records_test: list[dict[str, object]],
    model_dir: Path,
    reader,
    device: torch.device,
    digit_probe_epochs: int,
) -> tuple[dict[str, object], DigitGridProbe]:
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
    digit_attack, probe_model = train_digit_probe(
        records_train,
        records_test,
        key,
        device,
        digit_probe_epochs,
    )
    return {
        "decoder_epochs": checkpoint["epochs"],
        "psnr_sensitive_mean_db": float(np.mean(psnr_scores)),
        "ssim_sensitive_mean": float(np.mean(ssim_scores)),
        "easyocr_password_character_accuracy": float(np.mean(ocr_password)) if reader else None,
        "easyocr_card_character_accuracy": float(np.mean(ocr_card)) if reader else None,
        "direct_card_digit_attack": digit_attack,
        "test_pages": len(records_test),
    }, probe_model


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
    digit_probe_epochs: int = 30,
) -> dict[str, object]:
    keys = sanitizer_keys or DEFAULT_KEYS
    training_configs = {}
    for key in keys:
        training_path = model_dir / f"training_{key}.json"
        if not training_path.is_file():
            raise FileNotFoundError(f"Missing decoder training metadata: {training_path}")
        training = json.loads(training_path.read_text(encoding="utf-8"))
        if training.get("sanitizer") != key:
            raise ValueError(f"Training metadata sanitizer mismatch in {training_path}")
        config = {
            "epochs": training.get("epochs"),
            "batch_size": training.get("batch_size"),
            "learning_rate": training.get("learning_rate"),
        }
        if any(value is None for value in config.values()):
            raise ValueError(f"Incomplete decoder training metadata in {training_path}")
        training_configs[key] = config
    if len({tuple(config.values()) for config in training_configs.values()}) != 1:
        raise ValueError("Decoder training settings must match across sanitizer comparisons")

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
    controls = positive_controls(
        train_records,
        test_records,
        model_dir,
        reader,
        device,
        digit_probe_epochs,
    )
    print(json.dumps({"positive_controls": controls}, indent=2))
    manifest_path = data_dir / "dataset-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if reader is not None and not controls["valid"]:
        report = {
            "benchmark_valid": False,
            "positive_controls": controls,
            "conclusion": (
                "The positive-control gate failed; no sanitizer recovery or "
                "invertibility conclusion is valid."
            ),
            "results": {},
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "eval_inversion_results.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        (out_dir / "EVAL_INVERSION.md").write_text(
            "# Inversion attack benchmark\n\n"
            "## Positive controls\n\n"
            f"```json\n{json.dumps(controls, indent=2)}\n```\n\n"
            "The positive controls failed. No sanitizer results are reported; "
            "the benchmark is invalid until the visible-string controls succeed.\n",
            encoding="utf-8",
        )
        return report

    evaluated = {
        key: evaluate_key(
            key,
            train_records,
            test_records,
            model_dir,
            reader,
            device,
            digit_probe_epochs,
        )
        for key in keys
    }
    results = {key: value[0] for key, value in evaluated.items()}
    sanitized_keys = [
        key for key in results if key not in {"V", "Vprime_none"}
    ]
    if sanitized_keys:
        best_sanitized_key = max(
            sanitized_keys,
            key=lambda key: float(
                results[key]["direct_card_digit_attack"]["character_accuracy"]
            ),
        )
        best_digit_attack = results[best_sanitized_key]["direct_card_digit_attack"]
        best_accuracy = float(best_digit_attack["character_accuracy"])
        chance = float(best_digit_attack["uniform_chance"])
        majority = float(best_digit_attack["training_majority_baseline"])
        if best_accuracy > chance:
            baseline_comparison = (
                "It also exceeds" if best_accuracy > majority else "It does not exceed"
            )
            conclusion = (
                f"Sanitized V' remains partially recoverable by the tested spatial digit "
                f"probe: {best_sanitized_key} reached {best_accuracy:.4f} character "
                f"accuracy, above the {chance:.4f} uniform-chance baseline. "
                f"{baseline_comparison} the {majority:.4f} training-majority baseline. "
                "This is attack-level evidence on these synthetic held-out pages, not a "
                "claim that the representation is mathematically or generally invertible."
            )
        else:
            conclusion = (
                f"No evaluated sanitized V' exceeded the {chance:.4f} uniform-chance "
                f"digit baseline; the best was {best_sanitized_key} at "
                f"{best_accuracy:.4f} (training-majority baseline {majority:.4f}). "
                "This experiment did not demonstrate above-chance direct digit recovery, "
                "but it cannot establish that V' is non-invertible or private."
            )
    else:
        conclusion = "No sanitized representation was included, so V' recoverability was not evaluated."
    report = {
        "benchmark_valid": bool(controls["valid"]) if reader is not None else None,
        "positive_controls": controls,
        "conclusion": conclusion,
        "setup": {
            "pages": manifest["page_count"],
            "train_pages": manifest["training_pages"],
            "held_out_pages": manifest["validation_pages"],
            "split_strategy": manifest["split_strategy"],
            "training_templates": manifest["training_templates"],
            "validation_templates": manifest["validation_templates"],
            "pca_rank": manifest["PCA_rank"],
            "sanitizer_implementation": "Mean-direction rank-1 baseline; NumPy Gram-matrix eigendecomposition for k=1/4/8; client SVD uses power iteration with deflation",
            "wire_format": "source SVD/null-space -> centered PCA -> per-row int8 -> zlib/base64 JSON -> backend decode/rebuild",
            "decoder": "1x1 Conv2d 1152->64; transposed-convolution upsampling; no flattened MLP",
            "decoder_training": training_configs,
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
            "digit_probe": (
                "Per-representation spatial CNN on 24x24x1152 grid; "
                f"{digit_probe_epochs} epochs; 16-position digit classification"
            ),
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
        f"Benchmark valid: **{report['benchmark_valid']}**. Measured on "
        f"{manifest['page_count']} Playwright-rendered pages "
        f"({manifest['training_pages']} train / {manifest['validation_pages']} held out), "
        f"with the PCA rank-{manifest['PCA_rank']} wire path and {device}.",
        "",
        "## Positive controls",
        "",
        "These controls are reported before the sanitizer comparisons. The original "
        "and 224x224 inputs are independently run through the same SigLIP vision "
        "encoder; the decoder control is re-encoded after reconstruction. Digit "
        "scores use the spatial probe trained on unsanitized `V`.",
        "",
        "| Input | EasyOCR password char acc. | EasyOCR card char acc. | Spatial digit probe char acc. |",
        "|---|---:|---:|---:|",
    ]
    for label in (
        "original_screenshot",
        "224_thumbnail",
        "unsanitized_V_decoder_output",
    ):
        control = controls[label]
        markdown.append(
            f"| {label} | "
            f"{control['easyocr_password_character_accuracy']} | "
            f"{control['easyocr_card_character_accuracy']} | "
            f"{control['spatial_digit_probe_character_accuracy']:.4f} |"
        )
    markdown.extend([
        "",
        f"Uniform digit chance is {controls['uniform_chance']:.4f}; the "
        f"training-majority baseline is {controls['training_majority_baseline']:.4f}. "
        f"The control gate is `{controls['valid']}` (card OCR >=0.90, password "
        "OCR >=0.75, and digit probe above uniform chance for all three inputs).",
        "",
        "Every reported metric uses held-out pages only. V' is produced by source-grid sanitization, "
        "PCA projection, int8 quantization, zlib/base64 serialization, and the backend decoder's "
        "source-grid reconstruction plus 24x24 adapter.",
        "",
        f"Templates are held out as groups: train={manifest['training_templates']}; "
        f"validation={manifest['validation_templates']}.",
        "",
        f"Setup: seed {manifest['seed']}; {manifest['training_pages']} train / "
        f"{manifest['validation_pages']} held-out pages; Python {sys.version.split()[0]}, "
        f"NumPy {np.__version__}, PyTorch {torch.__version__}, ONNX Runtime "
        f"{manifest['onnxruntime']}; model size {manifest['model_bytes']:,} bytes. "
        f"Each decoder used {next(iter(training_configs.values()))['epochs']} epochs, "
        f"batch size {next(iter(training_configs.values()))['batch_size']}, Adam learning "
        f"rate {next(iter(training_configs.values()))['learning_rate']}, and seed 26171. "
        f"EasyOCR "
        f"{'not run' if easyocr_version is None else easyocr_version}.",
        "",
        "| Representation | PSNR (dB) | SSIM | EasyOCR password char acc. | EasyOCR card char acc. | Digit attack char acc. | Chance | Train-majority baseline |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
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
        "## Conclusion",
        "",
        conclusion,
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
    parser.add_argument("--digit-probe-epochs", type=int, default=30)
    args = parser.parse_args()
    report = evaluate(
        args.data_dir,
        args.model_dir,
        args.out_dir,
        args.sanitizers,
        args.skip_ocr,
        args.digit_probe_epochs,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
