import importlib.util
import sys
from pathlib import Path

import numpy as np

INVERSION_DIR = Path(__file__).parents[2] / "eval" / "inversion"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, INVERSION_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark module {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


dataset = _load("inversion_dataset", "01_generate_dataset.py")
evaluation = _load("inversion_evaluation", "03_evaluate.py")


def test_rank1_mean_baseline_projects_sensitive_mean_direction():
    source = np.random.default_rng(42).standard_normal((12, 8)).astype(np.float32)
    sensitive = [0, 1, 2, 3]
    direction = source[sensitive].mean(axis=0)
    direction /= np.linalg.norm(direction)

    sanitized = dataset.sanitize_source(
        source,
        sensitive,
        rank=1,
        alpha=0,
        rank1_baseline=True,
    )

    assert sanitized.shape == source.shape
    assert sanitized.dtype == np.float32
    np.testing.assert_allclose(sanitized @ direction, 0, atol=1e-5)


def test_svd_sanitizer_caps_rank_to_sensitive_rows_and_preserves_shape():
    source = np.random.default_rng(5).standard_normal((12, 8)).astype(np.float32)
    sanitized = dataset.sanitize_source(source, [0, 1], rank=8, alpha=0)

    assert sanitized.shape == source.shape
    assert sanitized.dtype == np.float32
    assert np.isfinite(sanitized).all()


def test_gram_eigendecomposition_matches_svd_right_subspace():
    rows = np.random.default_rng(16).standard_normal((6, 8)).astype(np.float32)
    expected = np.linalg.svd(rows, full_matrices=False)[2]
    actual = dataset.right_singular_vectors(rows)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual @ actual.T, np.eye(6), atol=1e-5)
    np.testing.assert_allclose(np.abs(actual @ expected.T), np.eye(6), atol=1e-4)


def test_empty_sensitive_set_is_identity():
    source = np.random.default_rng(7).standard_normal((12, 8)).astype(np.float32)
    sanitized = dataset.sanitize_source(source, [], rank=4, alpha=0.01)
    np.testing.assert_array_equal(sanitized, source)


def test_sensitive_mask_marks_requested_grid_cells():
    mask = evaluation.sensitive_mask(np.asarray([0, 24, 575], dtype=np.int32))

    assert mask.shape == (224, 224)
    assert mask.dtype == bool
    assert mask.any()
    assert mask[0, 0]
    assert mask[9, 0]
    assert mask[223, 223]


def test_psnr_and_ssim_are_perfect_for_identical_images():
    image = np.random.default_rng(12).random((224, 224, 3), dtype=np.float32)
    mask = np.ones((224, 224), dtype=bool)

    assert evaluation.psnr(image, image, mask) == 100
    assert evaluation.ssim(image, image, mask) > 0.999


def test_character_accuracy_uses_edit_distance():
    assert evaluation.char_accuracy("VDLM-1234", "vdlm1234") == 1
    assert evaluation.char_accuracy("123456", "1234") < 1


def test_digit_probe_reports_test_pages_and_chance_baseline(tmp_path: Path):
    train_records = []
    test_records = []
    rng = np.random.default_rng(19)
    for index in range(10):
        record_path = tmp_path / f"sample_{index:02}.npz"
        np.savez_compressed(record_path, Vprime=rng.standard_normal((576, 1152)).astype(np.float32))
        record = {
            "path": record_path,
            "metadata": {"card_number": f"{index:016d}"},
        }
        (train_records if index < 8 else test_records).append(record)

    result = evaluation.digit_probe(train_records, test_records, "Vprime")

    assert result["test_pages"] == 2
    assert result["uniform_chance"] == 0.1
    assert 0 <= result["character_accuracy"] <= 1
    assert 0 <= result["training_majority_baseline"] <= 1
