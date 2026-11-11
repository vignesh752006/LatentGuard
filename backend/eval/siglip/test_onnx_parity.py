"""Compare the exported FP16 ONNX vision graph with the HF reference fixtures."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from PIL import Image

from .adapter import adapt_siglip_embeddings

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[2]
FIXTURES = HERE / "fixtures"
MODEL = REPO_ROOT / "client" / "public" / "models" / "siglip-base-patch16-224-vision-fp16.onnx"
ADAPTER_MODEL = REPO_ROOT / "client" / "public" / "models" / "siglip-base-patch16-224-adapter.onnx"
IMAGE_FIXTURES = (
    ("login", ".png"),
    ("checkout", ".png"),
    ("natural-iss073e0763824", ".jpg"),
    ("natural-iss009e05944", ".jpg"),
    ("natural-iss009e05953", ".jpg"),
)


@pytest.fixture(scope="module")
def session() -> ort.InferenceSession:
    if not MODEL.is_file():
        pytest.skip("Export the local SigLIP ONNX models before running parity tests")
    return ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])


@pytest.fixture(scope="module")
def adapter_session() -> ort.InferenceSession:
    if not ADAPTER_MODEL.is_file():
        pytest.skip("Export the local SigLIP ONNX models before running adapter parity tests")
    return ort.InferenceSession(str(ADAPTER_MODEL), providers=["CPUExecutionProvider"])


def load_pixels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    return ((pixels - 0.5) / 0.5).transpose(2, 0, 1)[None].copy()


@pytest.mark.parametrize(("name", "extension"), IMAGE_FIXTURES)
def test_fp16_onnx_matches_hugging_face_reference(
    session: ort.InferenceSession,
    name: str,
    extension: str,
) -> None:
    image_path = FIXTURES / "images" / f"{name}{extension}"
    reference = np.load(FIXTURES / "reference" / f"{name}.npy")
    pixels = load_pixels(image_path)
    output = session.run(["last_hidden_state"], {"pixel_values": pixels})[0][0]

    assert output.shape == (196, 768)
    cosine = np.sum(output * reference, axis=1) / (
        np.linalg.norm(output, axis=1) * np.linalg.norm(reference, axis=1)
    )
    assert float(cosine.mean()) >= 0.99

    if name == "login":
        repeated = session.run(["last_hidden_state"], {"pixel_values": pixels})[0][0]
        np.testing.assert_array_equal(output, repeated)


@pytest.mark.parametrize(("name", "_extension"), IMAGE_FIXTURES)
def test_python_adapter_matches_exported_onnx_adapter(
    adapter_session: ort.InferenceSession,
    name: str,
    _extension: str,
) -> None:
    reference = np.load(FIXTURES / "reference" / f"{name}.npy")
    expected = adapter_session.run(
        ["adapted_embeddings"],
        {"patch_embeddings": reference[None].astype(np.float32)},
    )[0][0]
    actual = adapt_siglip_embeddings(reference)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-5)
