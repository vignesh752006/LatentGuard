import hashlib
from pathlib import Path

import numpy as np

from eval.siglip.adapter import bilinear_upsample
from eval.siglip.projection import build_projection_matrix


def test_siglip_projection_matrix_matches_shared_fixture():
    fixture = Path(__file__).parents[1] / "eval" / "siglip" / "fixtures" / "projection-matrix.sha256"
    expected_hash = fixture.read_text(encoding="ascii").strip()
    matrix = build_projection_matrix()
    actual_hash = hashlib.sha256(matrix.tobytes(order="C")).hexdigest()
    assert actual_hash == expected_hash


def test_siglip_bilinear_adapter_clamps_edges():
    source = np.zeros((4, 768), dtype=np.float32)
    source[:, 0] = [1, 2, 3, 4]
    target = bilinear_upsample(source, source_grid=2, target_grid=4)
    assert target[[0, 3, 12, 15], 0].tolist() == [1, 2, 3, 4]
