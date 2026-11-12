"""Tensor serialization / deserialization for the VDLM wire protocol.

Supported compression labels (payload.compression):
  "none"  — raw bytes
  "zlib"  — zlib.compress / zlib.decompress (RFC 1950).
             For backward-compatibility the backend also accepts gzip-encoded
             data labelled "zlib" (detected by the 0x1f 0x8b magic bytes).
  "delta" — not yet decoded server-side; rejected with UNSUPPORTED_COMPRESSION.

Supported dtype values (payload.dtype):
  "float16"       — 2 bytes/element, reshape to payload.shape
  "float32"       — 4 bytes/element, reshape to payload.shape
  "int8_quantized"— compact encoding: float32 scales (patch_count values) +
                    int8 values (patch_count * embedding_dim), zlib-compressed.
                    Dequantized to float32 on decode.
"""

import base64
import gzip
import zlib
from dataclasses import dataclass

import numpy as np
import torch

from app.config.settings import Settings
from app.observability.logging import get_logger
from app.schemas.messages import TensorPayload
from app.security.payload_guard import PayloadRejected
from eval.siglip.adapter import adapt_siglip_embeddings
from app.encoding.source_pca import (
    MAX_COMPONENTS as SOURCE_PCA_MAX_COMPONENTS,
    SOURCE_PATCHES,
    load_basis as load_source_pca_basis,
    reconstruct_source_pca,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class TensorContract:
    patch_count: int
    embedding_dim: int
    dtype: str

    @property
    def shape(self) -> tuple[int, int]:
        return (self.patch_count, self.embedding_dim)

    @property
    def element_count(self) -> int:
        return self.patch_count * self.embedding_dim


def contract_from_settings(settings: Settings) -> TensorContract:
    return TensorContract(
        patch_count=settings.vdlm_patch_count,
        embedding_dim=settings.vdlm_embedding_dim,
        dtype="float16",
    )


def _decompress_zlib_or_gzip(raw: bytes) -> bytes:
    """Decompress data labelled 'zlib'.

    Accepts both RFC 1950 zlib streams and gzip streams (detected by magic
    bytes 0x1f 0x8b) for backward-compatibility with clients that mistakenly
    used gzip while labelling it 'zlib'.
    """
    if raw[:2] == b"\x1f\x8b":
        # gzip magic — client used gzipSync instead of zlibSync
        logger.warning(
            "tensor_compression_mismatch",
            detail="gzip data received with 'zlib' label; decompressing as gzip",
        )
        try:
            return gzip.decompress(raw)
        except Exception as exc:
            raise PayloadRejected(
                "TENSOR_DECODE_ERROR",
                f"gzip decompression failed: {exc}",
            ) from exc
    try:
        return zlib.decompress(raw)
    except zlib.error as exc:
        raise PayloadRejected(
            "TENSOR_DECODE_ERROR",
            f"zlib decompression failed: {exc}",
        ) from exc


def _decode_int8_quantized(
    raw: bytes,
    patch_count: int,
    embedding_dim: int,
) -> torch.Tensor:
    """Decode compact int8-quantized tensor.

    Wire format (before compression):
      [0 .. patch_count*4-1]              float32 scale per row
      [patch_count*4 .. end]              int8 values, row-major
    """
    scales_bytes = patch_count * 4  # float32 = 4 bytes each
    expected_total = scales_bytes + patch_count * embedding_dim

    if len(raw) != expected_total:
        raise PayloadRejected(
            "TENSOR_SIZE_MISMATCH",
            f"int8_quantized: expected {expected_total} bytes "
            f"({scales_bytes} scales + {patch_count * embedding_dim} int8), "
            f"got {len(raw)}",
        )

    scales = np.frombuffer(raw[:scales_bytes], dtype=np.float32).copy()  # (patch_count,)
    int8_vals = np.frombuffer(raw[scales_bytes:], dtype=np.int8).reshape(
        patch_count, embedding_dim
    ).copy()

    # Dequantize: scale * (int8 / 127)
    dequant = scales[:, None] * (int8_vals.astype(np.float32) / 127.0)

    if not np.isfinite(dequant).all():
        raise PayloadRejected(
            "TENSOR_INVALID_VALUES",
            "int8_quantized dequantized tensor contains non-finite values",
        )

    return torch.from_numpy(dequant)


def decode_tensor_payload(
    payload: TensorPayload,
    contract: TensorContract,
    max_elements: int,
) -> torch.Tensor:
    """Decode a TensorPayload from the wire into a torch.Tensor.

    Logs wire bytes received for every frame.
    """
    source_pca = payload.transform == "source_pca_v1"
    if source_pca:
        if (
            payload.dtype != "int8_quantized"
            or payload.shape[0] != SOURCE_PATCHES
            or not 1 <= payload.shape[1] <= SOURCE_PCA_MAX_COMPONENTS
        ):
            raise PayloadRejected(
                "TENSOR_SHAPE_MISMATCH",
                f"source_pca_v1 requires int8_quantized [196, 1..{SOURCE_PCA_MAX_COMPONENTS}], "
                f"got {payload.dtype} {payload.shape}",
            )
    elif payload.shape != list(contract.shape):
        raise PayloadRejected(
            "TENSOR_SHAPE_MISMATCH",
            f"Expected shape {list(contract.shape)}, got {payload.shape}",
        )

    element_count = payload.shape[0] * payload.shape[1]
    if element_count > max_elements:
        raise PayloadRejected(
            "TENSOR_TOO_LARGE",
            f"Tensor element count {element_count} exceeds limit {max_elements}",
        )

    try:
        raw = base64.b64decode(payload.data, validate=True)
    except Exception as exc:
        raise PayloadRejected(
            "TENSOR_DECODE_ERROR",
            f"Invalid base64 tensor data: {exc}",
        ) from exc

    wire_bytes = len(raw)

    # --- Decompress ---
    if payload.compression == "zlib":
        raw = _decompress_zlib_or_gzip(raw)
    elif payload.compression == "delta":
        raise PayloadRejected(
            "UNSUPPORTED_COMPRESSION",
            "delta compression is not yet supported server-side",
        )
    # "none" — no decompression needed

    logger.info(
        "tensor_decoded",
        wire_bytes=wire_bytes,
        decompressed_bytes=len(raw),
        dtype=payload.dtype,
        compression=payload.compression,
    )

    # --- int8_quantized path ---
    if payload.dtype == "int8_quantized":
        if source_pca:
            quantized = _decode_int8_quantized(raw, payload.shape[0], payload.shape[1])
            try:
                mean, components = load_source_pca_basis()
            except (FileNotFoundError, ValueError) as exc:
                raise PayloadRejected(
                    "SOURCE_PCA_BASIS_INVALID",
                    f"Unable to load source PCA basis: {exc}",
                ) from exc
            source = reconstruct_source_pca(
                quantized.numpy(), mean, components[: payload.shape[1]]
            )
            rebuilt = adapt_siglip_embeddings(source)
            return torch.from_numpy(rebuilt.copy())
        return _decode_int8_quantized(raw, contract.patch_count, contract.embedding_dim)

    # --- float16 / float32 path ---
    np_dtype = np.float16 if payload.dtype == "float16" else np.float32
    expected_bytes = element_count * np.dtype(np_dtype).itemsize
    if len(raw) != expected_bytes:
        raise PayloadRejected(
            "TENSOR_SIZE_MISMATCH",
            f"Expected {expected_bytes} bytes for {payload.dtype} tensor, got {len(raw)}",
        )

    array = np.frombuffer(raw, dtype=np_dtype).reshape(payload.shape)
    if not np.isfinite(array).all():
        raise PayloadRejected(
            "TENSOR_INVALID_VALUES",
            "Tensor contains non-finite values",
        )

    return torch.from_numpy(array.copy())


def encode_tensor_for_test(tensor: torch.Tensor) -> TensorPayload:
    array = tensor.detach().cpu().numpy().astype(np.float16)
    raw = array.tobytes()
    return TensorPayload(
        dtype="float16",
        shape=list(array.shape),
        data=base64.b64encode(raw).decode("ascii"),
        compression="none",
    )
