"""Shared source-grid PCA representation used by the compact wire decoder."""

from functools import lru_cache
from pathlib import Path

import numpy as np

SOURCE_DIM = 768
SOURCE_PATCHES = 196
MAX_COMPONENTS = 128
VERSION = "source_pca_v1"
BASIS_PATH = Path(__file__).resolve().parent / "data" / "source-pca-v1.bin"


@lru_cache(maxsize=1)
def load_basis() -> tuple[np.ndarray, np.ndarray]:
    if not BASIS_PATH.is_file():
        raise FileNotFoundError(f"Source PCA basis is missing: {BASIS_PATH}")
    values = np.fromfile(BASIS_PATH, dtype="<f4")
    expected = SOURCE_DIM * (MAX_COMPONENTS + 1)
    if values.size != expected:
        raise ValueError(f"Source PCA basis should contain {expected} float32 values; got {values.size}")
    if not np.isfinite(values).all():
        raise ValueError("Source PCA basis contains non-finite values")
    mean = values[:SOURCE_DIM].astype(np.float32, copy=True)
    components = values[SOURCE_DIM:].reshape(MAX_COMPONENTS, SOURCE_DIM).astype(
        np.float32, copy=True
    )
    return mean, components


def reconstruct_source_pca(
    coefficients: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    if coefficients.ndim != 2 or coefficients.shape[0] != SOURCE_PATCHES:
        raise ValueError(f"Expected PCA coefficients [196, k], got {coefficients.shape}")
    rank = coefficients.shape[1]
    if rank < 1 or rank > components.shape[0]:
        raise ValueError(f"Unsupported source PCA rank: {rank}")
    return (coefficients @ components[:rank] + mean).astype(np.float32)
