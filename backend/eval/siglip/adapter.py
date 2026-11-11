"""SigLIP patch-grid and fixed-projection adapter."""

import numpy as np

from .projection import build_projection_matrix

SOURCE_GRID = 14
SOURCE_DIM = 768
TARGET_GRID = 24
TARGET_DIM = 1152


def bilinear_upsample(
    embeddings: np.ndarray,
    source_grid: int = SOURCE_GRID,
    target_grid: int = TARGET_GRID,
) -> np.ndarray:
    """Upsample a square HWC patch grid with the TypeScript half-pixel formula."""
    source = np.asarray(embeddings, dtype=np.float32).reshape(
        source_grid, source_grid, SOURCE_DIM
    )
    target = np.empty((target_grid, target_grid, SOURCE_DIM), dtype=np.float32)
    scale = source_grid / target_grid
    for out_y in range(target_grid):
        y = max(0.0, min(source_grid - 1, (out_y + 0.5) * scale - 0.5))
        y0 = math_floor(y)
        y1 = min(source_grid - 1, y0 + 1)
        dy = y - y0
        for out_x in range(target_grid):
            x = max(0.0, min(source_grid - 1, (out_x + 0.5) * scale - 0.5))
            x0 = math_floor(x)
            x1 = min(source_grid - 1, x0 + 1)
            dx = x - x0
            top = source[y0, x0] * (1.0 - dx) + source[y0, x1] * dx
            bottom = source[y1, x0] * (1.0 - dx) + source[y1, x1] * dx
            target[out_y, out_x] = top * (1.0 - dy) + bottom * dy
    return target.reshape(target_grid * target_grid, SOURCE_DIM)


def adapt_siglip_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Convert [196, 768] patch tokens to the [576, 1152] client contract."""
    source = np.asarray(embeddings, dtype=np.float32)
    if source.shape != (SOURCE_GRID * SOURCE_GRID, SOURCE_DIM):
        raise ValueError(f"Expected (196, 768) SigLIP patches, got {source.shape}")
    upsampled = bilinear_upsample(source)
    return upsampled @ build_projection_matrix()


def math_floor(value: float) -> int:
    return int(np.floor(value))
