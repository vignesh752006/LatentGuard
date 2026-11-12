"""Verify the exported text tower and score adapter against Hugging Face SigLIP."""

from pathlib import Path
import os
import sys

import numpy as np
import onnxruntime as ort
import pytest
import torch
from tokenizers import Tokenizer
from transformers import SiglipModel, SiglipTokenizer

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT))
from backend.eval.siglip.projection import build_projection_matrix

MODEL_ID = "google/siglip-base-patch16-224"
MODEL_DIR = Path(
    os.environ.get(
        "VDLM_SIGLIP_TEXT_MODEL_DIR",
        str(REPO_ROOT / "client" / "public" / "models"),
    )
)
TEXT_MODEL = MODEL_DIR / "siglip-base-patch16-224-text-fp32.onnx"
SCORE_ADAPTER = MODEL_DIR / "siglip-base-patch16-224-text-image-score.onnx"
SOURCE_SCORE_ADAPTER = MODEL_DIR / "siglip-base-patch16-224-source-text-score.onnx"
TOKENIZER_DIR = MODEL_DIR / "siglip-base-patch16-224-text-tokenizer"
TASKS = (
    "click the sign in button",
    "enter the account password",
    "find the billing address field",
    "open the online banking dashboard",
    "cancel the checkout and return to the cart",
)


def _require_text_assets() -> None:
    if (
        TEXT_MODEL.is_file()
        and SCORE_ADAPTER.is_file()
        and SOURCE_SCORE_ADAPTER.is_file()
        and (TOKENIZER_DIR / "tokenizer.json").is_file()
    ):
        return
    message = (
        "Export the local SigLIP text ONNX, score adapter, and tokenizer with "
        "scripts/export-siglip-text-onnx.py"
    )
    if os.environ.get("CI", "").lower() == "true":
        pytest.fail(message)
    pytest.skip(message)


@pytest.fixture(scope="module")
def text_models():
    _require_text_assets()
    tokenizer = Tokenizer.from_file(str(TOKENIZER_DIR / "tokenizer.json"))
    reference_tokenizer = SiglipTokenizer.from_pretrained(MODEL_ID)
    reference = SiglipModel.from_pretrained(MODEL_ID).eval().cpu()
    text_session = ort.InferenceSession(
        str(TEXT_MODEL), providers=["CPUExecutionProvider"]
    )
    score_session = ort.InferenceSession(
        str(SCORE_ADAPTER), providers=["CPUExecutionProvider"]
    )
    source_score_session = ort.InferenceSession(
        str(SOURCE_SCORE_ADAPTER), providers=["CPUExecutionProvider"]
    )
    return (
        tokenizer,
        reference_tokenizer,
        reference,
        text_session,
        score_session,
        source_score_session,
    )


def _inputs(tokenizer, reference_tokenizer, task: str) -> tuple[np.ndarray, np.ndarray]:
    encoded = tokenizer.encode(task)
    expected = reference_tokenizer(
        task,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_attention_mask=True,
    )
    assert encoded.ids == expected["input_ids"]
    assert encoded.attention_mask == expected["attention_mask"]
    return (
        np.asarray([encoded.ids], dtype=np.int64),
        np.asarray([encoded.attention_mask], dtype=np.int64),
    )


@pytest.mark.parametrize("task", TASKS)
def test_text_onnx_embedding_matches_hugging_face(text_models, task: str) -> None:
    tokenizer, reference_tokenizer, reference, text_session, _, _ = text_models
    input_ids, attention_mask = _inputs(tokenizer, reference_tokenizer, task)
    with torch.inference_mode():
        pooled = reference.text_model(
            input_ids=torch.from_numpy(input_ids),
            attention_mask=torch.from_numpy(attention_mask),
        ).pooler_output
        expected = pooled.numpy()[0]
    actual = text_session.run(
        ["text_embedding"],
        {"input_ids": input_ids, "attention_mask": attention_mask},
    )[0][0]
    cosine = float(
        np.dot(actual, expected)
        / (np.linalg.norm(actual) * np.linalg.norm(expected))
    )
    assert actual.shape == (768,)
    assert cosine >= 0.99
    repeated = text_session.run(
        ["text_embedding"],
        {"input_ids": input_ids, "attention_mask": attention_mask},
    )[0][0]
    np.testing.assert_array_equal(actual, repeated)


@pytest.mark.parametrize("task", TASKS)
def test_score_adapter_matches_siglip_patch_text_cosine(text_models, task: str) -> None:
    tokenizer, reference_tokenizer, reference, _, score_session, _ = text_models
    input_ids, attention_mask = _inputs(tokenizer, reference_tokenizer, task)
    rng = np.random.default_rng(20261112)
    source_embeddings = rng.standard_normal((1, 576, 768), dtype=np.float32)
    projection = build_projection_matrix()
    adapted_embeddings = source_embeddings @ projection
    with torch.inference_mode():
        pooled = reference.text_model(
            input_ids=torch.from_numpy(input_ids),
            attention_mask=torch.from_numpy(attention_mask),
        ).pooler_output
        text_embedding = pooled.numpy()
    actual = score_session.run(
        ["score_grid"],
        {
            "adapted_embeddings": adapted_embeddings.astype(np.float32),
            "text_embedding": text_embedding.astype(np.float32),
        },
    )[0][0]
    normalized_image = source_embeddings[0] / np.linalg.norm(
        source_embeddings[0], axis=1, keepdims=True
    )
    normalized_text = text_embedding[0] / np.linalg.norm(text_embedding[0])
    expected = normalized_image @ normalized_text
    assert actual.shape == (576,)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("task", TASKS)
def test_source_score_adapter_matches_upsampled_patch_text_cosine(
    text_models, task: str
) -> None:
    tokenizer, reference_tokenizer, reference, _, _, source_score_session = text_models
    input_ids, attention_mask = _inputs(tokenizer, reference_tokenizer, task)
    rng = np.random.default_rng(20261112)
    source_embeddings = rng.standard_normal((1, 196, 768), dtype=np.float32)
    with torch.inference_mode():
        pooled = reference.text_model(
            input_ids=torch.from_numpy(input_ids),
            attention_mask=torch.from_numpy(attention_mask),
        ).pooler_output
        text_embedding = pooled.numpy()
        image_grid = torch.from_numpy(source_embeddings).reshape(1, 14, 14, 768)
        image_grid = image_grid.permute(0, 3, 1, 2)
        image_grid = torch.nn.functional.interpolate(
            image_grid,
            size=(24, 24),
            mode="bilinear",
            align_corners=False,
        )
        image_embedding = image_grid.permute(0, 2, 3, 1).reshape(1, 576, 768).numpy()
    actual = source_score_session.run(
        ["score_grid"],
        {
            "source_embeddings": source_embeddings,
            "text_embedding": text_embedding.astype(np.float32),
        },
    )[0][0]
    normalized_image = image_embedding[0] / np.linalg.norm(
        image_embedding[0], axis=1, keepdims=True
    )
    normalized_text = text_embedding[0] / np.linalg.norm(text_embedding[0])
    expected = normalized_image @ normalized_text
    assert actual.shape == (576,)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
