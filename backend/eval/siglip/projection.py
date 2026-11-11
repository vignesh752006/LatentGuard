"""Fixed 768-to-1152 projection shared with the TypeScript implementation."""

import math

import numpy as np

INPUT_DIM = 768
OUTPUT_DIM = 1152
PROJECTION_SEED = 0xDEADBEEF


def build_projection_matrix() -> np.ndarray:
    """Return row-major float32 weights using the shared uint32 LCG algorithm."""
    state = PROJECTION_SEED
    matrix = np.empty((INPUT_DIM, OUTPUT_DIM), dtype=np.float32)
    for index in range(matrix.size):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        matrix.flat[index] = (state / 4294967296.0) * 2.0 - 1.0

    for column in range(OUTPUT_DIM):
        squared_norm = 0.0
        for row in range(INPUT_DIM):
            value = float(matrix[row, column])
            squared_norm += value * value
        scale = math.sqrt(squared_norm) + 1e-8
        for row in range(INPUT_DIM):
            matrix[row, column] = float(matrix[row, column]) / scale

    return matrix
