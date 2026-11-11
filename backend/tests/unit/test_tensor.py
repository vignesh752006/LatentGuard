"""Unit tests for tensor serialization / deserialization (Task 1)."""

import base64
import gzip
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch

from app.schemas.messages import TensorPayload
from app.tensor.serialization import contract_from_settings, decode_tensor_payload
from eval.siglip.adapter import adapt_siglip_embeddings
from eval.siglip.source_pca import load_basis, reconstruct_source_pca
from tests.conftest import make_sanitized_tensor, make_test_settings


def _make_int8_quantized_payload(tensor: torch.Tensor) -> TensorPayload:
    """Build a compact int8-quantized TensorPayload matching the client encoder."""
    arr = tensor.numpy().astype(np.float32)
    patch_count, embedding_dim = arr.shape
    scales = np.abs(arr).max(axis=1).astype(np.float32)  # (576,)
    inv_scales = np.where(scales > 0, 127.0 / scales, 0.0)
    int8_vals = np.clip(np.round(arr * inv_scales[:, None]), -127, 127).astype(np.int8)

    packed = np.concatenate([scales.view(np.uint8).ravel(), int8_vals.view(np.uint8).ravel()])
    compressed = zlib.compress(packed.tobytes())
    return TensorPayload(
        dtype="int8_quantized",
        shape=[patch_count, embedding_dim],
        data=base64.b64encode(compressed).decode("ascii"),
        compression="zlib",
    )


# ---- existing test (preserved) ------------------------------------------------

def test_zlib_compressed_tensor_roundtrip():
    settings = make_test_settings()
    contract = contract_from_settings(settings)
    tensor = make_sanitized_tensor()
    raw = tensor.numpy().astype(np.float16).tobytes()
    compressed = zlib.compress(raw)

    payload = TensorPayload(
        dtype="float16",
        shape=[576, 1152],
        data=base64.b64encode(compressed).decode("ascii"),
        compression="zlib",
    )
    decoded = decode_tensor_payload(payload, contract, settings.vdlm_max_tensor_elements)
    assert decoded.shape == (576, 1152)


# ---- Task 1 new tests ---------------------------------------------------------

def test_gzip_accepted_as_zlib():
    """Backend accepts gzip-encoded data labelled 'zlib' (old client bug)."""
    settings = make_test_settings()
    contract = contract_from_settings(settings)
    tensor = make_sanitized_tensor()
    raw = tensor.numpy().astype(np.float16).tobytes()
    # Encode with gzip but label it "zlib"
    gz_bytes = gzip.compress(raw)

    payload = TensorPayload(
        dtype="float16",
        shape=[576, 1152],
        data=base64.b64encode(gz_bytes).decode("ascii"),
        compression="zlib",
    )
    decoded = decode_tensor_payload(payload, contract, settings.vdlm_max_tensor_elements)
    assert decoded.shape == (576, 1152)
    # Values should round-trip through float16
    expected = tensor.to(torch.float16)
    assert torch.allclose(decoded.to(torch.float16), expected, atol=1e-3)


def test_int8_quantized_shape():
    """Decoder accepts int8_quantized payloads and returns correct shape."""
    settings = make_test_settings()
    contract = contract_from_settings(settings)
    tensor = make_sanitized_tensor()

    payload = _make_int8_quantized_payload(tensor)
    decoded = decode_tensor_payload(payload, contract, settings.vdlm_max_tensor_elements)
    assert decoded.shape == (576, 1152)
    assert decoded.dtype == torch.float32


def test_int8_quantized_fidelity():
    """Round-tripped int8 tensor has cosine similarity > 0.95 with the original."""
    settings = make_test_settings()
    contract = contract_from_settings(settings)
    tensor = make_sanitized_tensor()
    orig = tensor.to(torch.float32)

    payload = _make_int8_quantized_payload(tensor)
    decoded = decode_tensor_payload(payload, contract, settings.vdlm_max_tensor_elements)

    # Per-row cosine similarity, averaged
    dot = (orig * decoded).sum(dim=1)
    norm_a = orig.norm(dim=1)
    norm_b = decoded.norm(dim=1)
    cos_sim = (dot / (norm_a * norm_b + 1e-8)).mean().item()
    assert cos_sim > 0.95, f"int8 round-trip cosine similarity too low: {cos_sim:.4f}"


def test_int8_quantized_wire_size():
    """int8+zlib payload is smaller than raw fp16 (1,327,104 bytes)."""
    tensor = make_sanitized_tensor()
    payload = _make_int8_quantized_payload(tensor)
    wire_bytes = len(base64.b64decode(payload.data))  # compressed size
    assert wire_bytes < 1_327_104, f"int8+zlib ({wire_bytes} B) not smaller than fp16"
    # Document achieved size
    print(f"\n[TASK 1] int8+zlib wire bytes: {wire_bytes:,} "
          f"({wire_bytes / 1_327_104:.1%} of fp16 uncompressed)")


def test_source_pca_payload_rebuilds_full_contract_tensor():
    settings = make_test_settings()
    contract = contract_from_settings(settings)
    fixture = (
        Path(__file__).resolve().parents[2]
        / "eval"
        / "siglip"
        / "fixtures"
        / "reference"
        / "login.npy"
    )
    source = np.load(fixture).astype(np.float32)
    mean, components = load_basis()
    rank = 64
    coefficients = (source - mean) @ components[:rank].T
    scales = np.max(np.abs(coefficients), axis=1).astype(np.float32)
    multipliers = np.divide(127.0, scales, out=np.zeros_like(scales), where=scales > 0)
    quantized = np.clip(
        np.round(coefficients * multipliers[:, None]), -127, 127
    ).astype(np.int8)
    packed = scales.tobytes() + quantized.tobytes()
    payload = TensorPayload(
        dtype="int8_quantized",
        shape=[196, rank],
        data=base64.b64encode(zlib.compress(packed)).decode("ascii"),
        compression="zlib",
        transform="source_pca_v1",
    )

    decoded = decode_tensor_payload(payload, contract, settings.vdlm_max_tensor_elements)
    expected_source = reconstruct_source_pca(
        quantized.astype(np.float32) * scales[:, None] / 127.0,
        mean,
        components[:rank],
    )
    expected = adapt_siglip_embeddings(expected_source)
    assert decoded.shape == (576, 1152)
    np.testing.assert_allclose(decoded.numpy(), expected, rtol=0, atol=2e-6)


def test_source_pca_rejects_invalid_rank_before_decoding():
    settings = make_test_settings()
    contract = contract_from_settings(settings)
    payload = TensorPayload(
        dtype="int8_quantized",
        shape=[196, 129],
        data=base64.b64encode(b"").decode("ascii"),
        compression="zlib",
        transform="source_pca_v1",
    )
    from app.security.payload_guard import PayloadRejected

    with pytest.raises(PayloadRejected) as exc:
        decode_tensor_payload(payload, contract, settings.vdlm_max_tensor_elements)
    assert exc.value.code == "TENSOR_SHAPE_MISMATCH"


def test_delta_compression_rejected():
    """Backend rejects payloads with unsupported delta compression."""
    from app.security.payload_guard import PayloadRejected as PR
    settings = make_test_settings()
    contract = contract_from_settings(settings)
    payload = TensorPayload(
        dtype="float16",
        shape=[576, 1152],
        data=base64.b64encode(b"\x00" * 1327104).decode("ascii"),
        compression="delta",
    )
    with pytest.raises(PR):
        decode_tensor_payload(payload, contract, settings.vdlm_max_tensor_elements)
