"""
Task 5 — Inversion benchmark unit tests (CI-safe, no GPU/Playwright)

Tests the synthetic data generation, sanitizers, and eval metrics.
"""

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Load the eval modules via importlib (they are standalone scripts, not packages)
import importlib.util

EVAL_DIR = Path(__file__).parent.parent.parent / "eval" / "inversion"


def _load(name, fname):
    path = EVAL_DIR / fname
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load("gen01", "01_generate_dataset.py")
ev  = _load("ev03",  "03_evaluate.py")


# ---- Dataset generation -------------------------------------------------------

def test_make_synthetic_embedding_shape():
    V = gen.make_synthetic_embedding(seed=42)
    assert V.shape == (576, 1152)
    assert V.dtype == np.float32


def test_make_synthetic_embedding_deterministic():
    V1 = gen.make_synthetic_embedding(seed=7)
    V2 = gen.make_synthetic_embedding(seed=7)
    np.testing.assert_array_equal(V1, V2)


def test_make_isens_login():
    isens = gen.make_isens("login", seed=0)
    assert len(isens) > 0
    assert int(isens.min()) >= 0
    assert int(isens.max()) < 576
    assert isens.dtype == np.int32


def test_make_isens_checkout():
    isens = gen.make_isens("checkout", seed=5)
    assert len(isens) > 0
    assert int(isens.max()) < 576


def test_generate_synthetic_dataset():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        out_dir = Path(tmpdir)
        gen._generate_synthetic(out_dir, n=5, pages=["login", "checkout"])

        files = sorted(out_dir.glob("sample_*.npz"))
        assert len(files) == 5

        # Validate first sample — use allow_pickle and close explicitly to avoid
        # Windows file-locking during TemporaryDirectory cleanup.
        d = np.load(files[0], allow_pickle=True)
        keys_present = list(d.keys())
        V_shape    = d["V"].shape
        shot_shape = d["screenshot"].shape
        isens_dtype = d["isens"].dtype
        meta_raw   = str(d["metadata"][0])
        d.close()  # release the file handle before the tempdir is deleted

        for key in ("V", "isens", "V_rank1", "V_svd_k1", "V_svd_k4", "V_svd_k8", "V_svd_k1_noise"):
            assert key in keys_present, f"Missing key: {key}"

        assert V_shape == (576, 1152)
        assert shot_shape == (224, 224, 3)
        assert isens_dtype == np.int32

        # Validate metadata
        meta = json.loads(meta_raw)
        assert "page_type" in meta
        assert meta["page_type"] in ("login", "checkout")


# ---- Sanitizer shapes and correctness ----------------------------------------

def test_rank1_sanitizer_shape_and_dtype():
    V = gen.make_synthetic_embedding(1)
    isens = gen.make_isens("login", 1)
    Vp = gen.apply_rank1_sanitizer(V, isens)
    assert Vp.shape == (576, 1152)
    assert Vp.dtype == np.float32


def test_rank1_sanitizer_modifies_embeddings():
    V = gen.make_synthetic_embedding(2)
    isens = gen.make_isens("login", 2)
    Vp = gen.apply_rank1_sanitizer(V, isens)
    assert not np.allclose(V, Vp)


def test_rank1_sanitizer_empty_isens_identity():
    V = gen.make_synthetic_embedding(3)
    Vp = gen.apply_rank1_sanitizer(V, np.array([], dtype=np.int32))
    np.testing.assert_array_equal(V, Vp)


@pytest.mark.parametrize("k", [1, 4, 8])
def test_svd_sanitizer_shape(k):
    V = gen.make_synthetic_embedding(10 + k)
    isens = gen.make_isens("checkout", 10 + k)
    Vp = gen.apply_svd_sanitizer(V, isens, k=k)
    assert Vp.shape == (576, 1152)
    assert Vp.dtype == np.float32


@pytest.mark.parametrize("k", [1, 4, 8])
def test_svd_sanitizer_modifies_embeddings(k):
    V = gen.make_synthetic_embedding(20 + k)
    isens = gen.make_isens("login", 20 + k)
    Vp = gen.apply_svd_sanitizer(V, isens, k=k)
    assert not np.allclose(V, Vp)


def test_svd_sanitizer_with_noise():
    V = gen.make_synthetic_embedding(50)
    isens = gen.make_isens("checkout", 50)
    Vp = gen.apply_svd_sanitizer(V, isens, k=1, alpha=0.1)
    assert Vp.shape == (576, 1152)


def test_svd_sanitizer_higher_k_more_mse():
    """Higher k removes more components → larger L2 diff from original."""
    V = gen.make_synthetic_embedding(99)
    isens = gen.make_isens("login", 99)[:20]
    mses = []
    for k in [1, 4, 8]:
        Vp = gen.apply_svd_sanitizer(V, isens, k=k)
        mses.append(float(np.mean((V - Vp)**2)))
    # Monotonically non-decreasing
    assert mses[0] <= mses[1] + 0.01
    assert mses[1] <= mses[2] + 0.01


# ---- Evaluation metrics -------------------------------------------------------

def test_psnr_identical():
    a = np.random.rand(100).astype(float)
    assert ev.psnr(a, a) == 100.0


def test_psnr_range():
    # psnr(zeros, ones): MSE=1.0, PSNR = 20*log10(max_val/sqrt(MSE)).
    # With max_val=1.0: PSNR = 20*log10(1/1) = 0.0 dB — this is correct by definition.
    # For a non-trivial case use a small perturbation so PSNR > 0.
    rng = np.random.default_rng(42)
    a = rng.random(100)
    b = a + rng.random(100) * 0.1   # small noise → positive PSNR
    p = ev.psnr(a, b)
    assert p >= 0.0        # always non-negative
    assert p < 100.0       # not identical (which would give 100.0 sentinel)


def test_ssim_identical():
    a = np.random.rand(1000).astype(float)
    s = ev.ssim(a, a)
    assert s > 0.9


def test_sensitive_region_mask_shape():
    isens = np.array([0, 24, 575], dtype=np.int32)
    mask = ev.sensitive_region_mask(isens, 224, 224)
    assert mask.shape == (224, 224)
    assert mask.dtype == bool
    assert mask.any()


def test_linear_probe_returns_valid_accuracy():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 576, 1152)).astype(np.float32)
    y = np.arange(30)
    acc = ev.linear_probe_accuracy(X, y)
    assert 0.0 <= acc <= 1.0
