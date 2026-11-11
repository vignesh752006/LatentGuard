"""Fit the shipped source-grid PCA basis from deterministic rendered pages."""

import json
import random
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import PIL
from PIL import Image, ImageDraw, ImageFont

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from eval.siglip.source_pca import BASIS_PATH, MAX_COMPONENTS, SOURCE_DIM, SOURCE_PATCHES
else:
    from .source_pca import BASIS_PATH, MAX_COMPONENTS, SOURCE_DIM, SOURCE_PATCHES

ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "client" / "public" / "models" / "siglip-base-patch16-224-vision-fp32.onnx"
SUMMARY_PATH = Path(__file__).resolve().parent / "fixtures" / "source-pca-metadata.json"
TRAINING_SEED = 0x26171
PAGE_COUNT = 128

PALETTES = [
    ("#f6f7fb", "#172554", "#2563eb"),
    ("#fffaf0", "#3f2d20", "#d97706"),
    ("#f0fdf4", "#14532d", "#16a34a"),
    ("#fdf2f8", "#831843", "#db2777"),
    ("#f8fafc", "#0f172a", "#475569"),
]
TITLES = [
    "Account overview", "Travel planner", "Project dashboard", "Member portal",
    "Order summary", "Research notes", "Workspace settings", "Community updates",
]
LABELS = [
    "Recent activity", "Account details", "Your saved items", "Payment options",
    "Notifications", "Quick search", "Upcoming events", "Help and support",
]
FONT_PATHS = [
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\calibri.ttf"),
    Path(r"C:\Windows\Fonts\times.ttf"),
    Path(r"C:\Windows\Fonts\verdana.ttf"),
]


def _font(rng: random.Random, size: int) -> ImageFont.ImageFont:
    available = [path for path in FONT_PATHS if path.exists()]
    if available:
        return ImageFont.truetype(str(rng.choice(available)), size)
    return ImageFont.load_default(size=size)


def render_page(index: int, rng: random.Random) -> Image.Image:
    background, foreground, accent = rng.choice(PALETTES)
    image = Image.new("RGB", (1280, 720), background)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1280, 76), fill="white")
    draw.text((42, 23), rng.choice(TITLES), fill=foreground, font=_font(rng, 28))
    draw.rounded_rectangle((900, 18, 1235, 58), radius=12, fill="#eef2ff")
    draw.text((920, 29), f"Search section {index}", fill="#475569", font=_font(rng, 16))

    left = rng.choice([36, 48, 64, 80])
    columns = rng.choice([2, 3, 4])
    gap = rng.choice([16, 20, 28])
    card_width = (1280 - left * 2 - gap * (columns - 1)) // columns
    top = rng.choice([110, 124, 142])
    for row in range(rng.choice([2, 3])):
        for col in range(columns):
            x = left + col * (card_width + gap)
            y = top + row * rng.choice([178, 190, 204])
            height = rng.choice([150, 164, 172])
            draw.rounded_rectangle((x, y, x + card_width, y + height), radius=14, fill="white")
            draw.rectangle((x + 18, y + 18, x + 62, y + 62), fill=accent)
            draw.text((x + 76, y + 23), rng.choice(LABELS), fill=foreground, font=_font(rng, 18))
            for line in range(rng.choice([2, 3, 4])):
                width = rng.randint(card_width // 3, max(card_width // 3, card_width - 44))
                line_y = y + 76 + line * 16
                draw.rounded_rectangle((x + 18, line_y, x + 18 + width, line_y + 5), radius=2, fill="#cbd5e1")
            if rng.random() < 0.6:
                button_y = y + height - 38
                draw.rounded_rectangle((x + 18, button_y, x + 118, button_y + 24), radius=6, fill=accent)
    return image


def image_tensor(image: Image.Image) -> np.ndarray:
    pixels = np.asarray(image.resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    return ((pixels - 0.5) / 0.5).transpose(2, 0, 1)[None].astype(np.float32)


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Missing FP32 ONNX model {MODEL_PATH}; export the SigLIP models first"
        )
    random.seed(TRAINING_SEED)
    rng = random.Random(TRAINING_SEED)
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    rows = []
    for page_index in range(PAGE_COUNT):
        image = render_page(page_index, rng)
        output = session.run(
            ["last_hidden_state"], {"pixel_values": image_tensor(image)}
        )[0][0]
        if output.shape != (SOURCE_PATCHES, SOURCE_DIM):
            raise ValueError(f"Expected [196, 768] SigLIP output, got {output.shape}")
        rows.append(output)
        if (page_index + 1) % 16 == 0:
            print(f"Embedded {page_index + 1}/{PAGE_COUNT} rendered pages")

    samples = np.concatenate(rows, axis=0).astype(np.float64)
    mean = samples.mean(axis=0)
    centered = samples - mean
    covariance = centered.T @ centered / max(1, len(samples) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:MAX_COMPONENTS]
    components = eigenvectors[:, order].T
    for row in components:
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            row *= -1
    mean32 = mean.astype("<f4")
    components32 = components.astype("<f4")
    BASIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.concatenate((mean32, components32.reshape(-1))).astype("<f4").tofile(BASIS_PATH)

    fixture_dir = Path(__file__).resolve().parent / "fixtures" / "reference"
    validation: dict[str, dict[str, float]] = {}
    for fixture_path in sorted(fixture_dir.glob("*.npy")):
        original = np.load(fixture_path).astype(np.float32)
        centered_fixture = original - mean32
        scores = centered_fixture @ components32.T
        for rank in (32, 64, 128):
            rebuilt = scores[:, :rank] @ components32[:rank] + mean32
            cosine = np.sum(original * rebuilt, axis=1) / (
                np.linalg.norm(original, axis=1) * np.linalg.norm(rebuilt, axis=1) + 1e-12
            )
            validation.setdefault(str(rank), {})[fixture_path.stem] = float(cosine.mean())

    metadata = {
        "version": "source_pca_v1",
        "training_seed": TRAINING_SEED,
        "rendered_pages": PAGE_COUNT,
        "source_tokens": int(samples.shape[0]),
        "source_dimension": SOURCE_DIM,
        "basis_rank": MAX_COMPONENTS,
        "input_model": "google/siglip-base-patch16-224 ONNX FP32",
        "numpy_version": np.__version__,
        "onnxruntime_version": ort.__version__,
        "pillow_version": PIL.__version__,
        "available_training_fonts": [str(path) for path in FONT_PATHS if path.exists()],
        "basis_file_bytes": BASIS_PATH.stat().st_size,
        "held_out_reference_mean_patch_cosine_by_rank": {
            rank: float(np.mean(list(per_image.values())))
            for rank, per_image in validation.items()
        },
        "held_out_reference_per_image_cosine_by_rank": validation,
        "training_pages_are_synthetic_rendered_ui_screenshots": True,
    }
    SUMMARY_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
