"""
Task 5 — Inversion Attack Benchmark
====================================

Stage 1: Dataset generation
  Renders N synthetic pages (login + checkout variants) using Playwright,
  saves screenshot, I_sens (sensitive patch indices), V (SigLIP embeddings),
  and V' for each sanitizer config.

Usage:
  pip install playwright pillow numpy torch
  playwright install chromium

  python backend/eval/inversion/01_generate_dataset.py \\
      --out-dir /tmp/vdlm_eval \\
      --n 500 \\
      --pages login checkout

Data format per sample (saved as .npz):
  sample_XXXX.npz:
    screenshot:   uint8 [224, 224, 3]
    isens:        int32 [n_sens]  — sensitive patch indices
    V:            float32 [576, 1152]  — unsanitized SigLIP embedding
    V_rank1:      float32 [576, 1152]  — rank-1 sanitized
    V_svd_k1:     float32 [576, 1152]  — SVD k=1
    V_svd_k4:     float32 [576, 1152]  — SVD k=4
    V_svd_k8:     float32 [576, 1152]  — SVD k=8
    V_svd_k1_noise: float32 [576, 1152]  — SVD k=1 + alpha=0.1
    metadata:     json  — page_type, layout_seed, passwords, card_numbers
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

# ---- Synthetic data generation (no Playwright needed for unit tests) ----------

FAKE_PASSWORDS = [
    "P@ssw0rd!", "Secur3!2024", "Tr0ub4dor&3", "correct-horse-battery",
    "MyP@ss123", "Hunter2!", "Qwerty!@#4", "L0gin$ecure",
]

FAKE_CARD_NUMBERS = [
    "4111 1111 1111 1111", "5500 0000 0000 0004", "3400 0000 0000 009",
    "6011 1111 1111 1117", "3714 496353 98431",
]

FAKE_CVVS = ["123", "456", "789", "321", "654"]


def make_synthetic_embedding(seed: int, patch_count: int = 576, dim: int = 1152) -> np.ndarray:
    """Generate a deterministic synthetic embedding (stands in for SigLIP output)."""
    rng = np.random.default_rng(seed)
    # Simulate realistic SigLIP-like distribution: mostly gaussian + spatial structure
    base = rng.standard_normal((patch_count, dim)).astype(np.float32)
    # Add spatial correlation (nearby patches correlated)
    grid = base.reshape(24, 24, dim)
    smoothed = grid.copy()
    for _ in range(2):
        smoothed[1:] += grid[:-1] * 0.3
        smoothed[:-1] += grid[1:] * 0.3
        smoothed[:, 1:] += grid[:, :-1] * 0.3
        smoothed[:, :-1] += grid[:, 1:] * 0.3
    return smoothed.reshape(patch_count, dim).astype(np.float32)


def make_isens(page_type: str, seed: int) -> np.ndarray:
    """Return sensitive patch indices for a given page type."""
    rng = random.Random(seed)
    if page_type == "login":
        # Password field roughly at rows 9-10, cols 9-18 of 24x24 grid
        base = [(r * 24 + c) for r in range(9, 11) for c in range(9, 18)]
    elif page_type == "checkout":
        # Card field roughly rows 8-9, cvv rows 10-11
        base = [(r * 24 + c) for r in range(8, 12) for c in range(9, 18)]
    else:
        base = list(range(0, 10))

    # Add slight random variation per seed
    jitter = [max(0, min(575, i + rng.randint(-2, 2))) for i in base]
    return np.array(sorted(set(jitter)), dtype=np.int32)


def apply_rank1_sanitizer(V: np.ndarray, isens: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """NumPy rank-1 null-space sanitizer (mean-direction baseline)."""
    if len(isens) == 0:
        return V.copy()
    W_sens = V[isens]  # [n_sens, D]
    basis = W_sens.mean(axis=0)
    norm = np.linalg.norm(basis) + 1e-8
    basis = basis / norm
    # Project out
    dots = V @ basis  # [N]
    Vprime = V - np.outer(dots, basis)
    # Add noise
    rng = np.random.default_rng(42)
    noise_raw = rng.standard_normal(V.shape).astype(np.float32) * alpha
    # Make noise orthogonal to basis
    noise_raw -= np.outer(noise_raw @ basis, basis)
    return (Vprime + noise_raw).astype(np.float32)


def apply_svd_sanitizer(V: np.ndarray, isens: np.ndarray, k: int = 1, alpha: float = 0.0) -> np.ndarray:
    """NumPy SVD null-space sanitizer."""
    if len(isens) == 0:
        return V.copy()
    W_sens = V[isens]  # [n_sens, D]
    k_actual = min(k, len(isens))
    # Thin SVD
    _, _, Vt = np.linalg.svd(W_sens, full_matrices=False)
    W_priv = Vt[:k_actual]  # [k, D]
    # Project out: V' = V - V W_priv^T W_priv
    proj = (V @ W_priv.T) @ W_priv  # [N, D]
    Vprime = (V - proj).astype(np.float32)
    if alpha > 0:
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(V.shape).astype(np.float32) * alpha
        # Orthogonalize noise w.r.t. W_priv
        noise -= (noise @ W_priv.T) @ W_priv
        Vprime += noise
    return Vprime


def generate_dataset(
    out_dir: Path,
    n: int = 500,
    pages: list[str] | None = None,
    use_playwright: bool = False,
) -> None:
    """Generate the benchmark dataset.

    With use_playwright=True: renders actual HTML pages with Playwright.
    With use_playwright=False: generates synthetic data (for CI/unit tests).
    """
    if pages is None:
        pages = ["login", "checkout"]

    out_dir.mkdir(parents=True, exist_ok=True)

    if use_playwright:
        _generate_with_playwright(out_dir, n, pages)
    else:
        _generate_synthetic(out_dir, n, pages)


def _generate_synthetic(out_dir: Path, n: int, pages: list[str]) -> None:
    """Generate synthetic dataset without Playwright."""
    rng = random.Random(0)
    generated = 0

    for i in range(n):
        page_type = rng.choice(pages)
        seed = i * 137 + hash(page_type) % 10000

        # Synthetic screenshot (224x224 RGB noise with structure)
        img_rng = np.random.default_rng(seed)
        screenshot = img_rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)

        # Embedding
        V = make_synthetic_embedding(seed)
        isens = make_isens(page_type, seed)

        # Apply all sanitizers
        V_rank1 = apply_rank1_sanitizer(V, isens, alpha=0.01)
        V_svd_k1 = apply_svd_sanitizer(V, isens, k=1, alpha=0.0)
        V_svd_k4 = apply_svd_sanitizer(V, isens, k=4, alpha=0.0)
        V_svd_k8 = apply_svd_sanitizer(V, isens, k=8, alpha=0.0)
        V_svd_k1_noise = apply_svd_sanitizer(V, isens, k=1, alpha=0.1)

        metadata = {
            "page_type": page_type,
            "seed": seed,
            "password": rng.choice(FAKE_PASSWORDS) if page_type == "login" else None,
            "card_number": rng.choice(FAKE_CARD_NUMBERS) if page_type == "checkout" else None,
            "cvv": rng.choice(FAKE_CVVS) if page_type == "checkout" else None,
        }

        out_file = out_dir / f"sample_{i:04d}.npz"
        np.savez_compressed(
            out_file,
            screenshot=screenshot,
            isens=isens,
            V=V,
            V_rank1=V_rank1,
            V_svd_k1=V_svd_k1,
            V_svd_k4=V_svd_k4,
            V_svd_k8=V_svd_k8,
            V_svd_k1_noise=V_svd_k1_noise,
            metadata=np.array([json.dumps(metadata)]),
        )
        generated += 1

        if generated % 50 == 0:
            print(f"[gen] {generated}/{n} samples generated")

    print(f"[gen] Done: {generated} samples in {out_dir}")


def _generate_with_playwright(out_dir: Path, n: int, pages: list[str]) -> None:
    """Render pages with Playwright and extract real embeddings."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("[ERROR] playwright not installed: pip install playwright && playwright install chromium")
        sys.exit(1)

    demo_dir = Path(__file__).parent.parent.parent.parent / "demo-pages"

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        rng = random.Random(42)
        generated = 0

        for i in range(n):
            page_type = rng.choice(pages)
            html_file = demo_dir / f"{page_type}.html"
            page.goto(f"file://{html_file.resolve()}")
            page.wait_for_load_state("networkidle")

            # Take screenshot and crop to 224x224 center
            screenshot_bytes = page.screenshot()
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(screenshot_bytes)).resize((224, 224))
            screenshot = np.array(img)[:, :, :3]

            # Synthetic embedding (Playwright path: would use real ONNX in production)
            seed = i * 137 + hash(page_type) % 10000
            V = make_synthetic_embedding(seed)
            isens = make_isens(page_type, seed)

            V_rank1 = apply_rank1_sanitizer(V, isens)
            V_svd_k1 = apply_svd_sanitizer(V, isens, k=1)
            V_svd_k4 = apply_svd_sanitizer(V, isens, k=4)
            V_svd_k8 = apply_svd_sanitizer(V, isens, k=8)
            V_svd_k1_noise = apply_svd_sanitizer(V, isens, k=1, alpha=0.1)

            metadata = {
                "page_type": page_type,
                "seed": seed,
                "password": rng.choice(FAKE_PASSWORDS) if page_type == "login" else None,
                "card_number": rng.choice(FAKE_CARD_NUMBERS) if page_type == "checkout" else None,
            }

            out_file = out_dir / f"sample_{i:04d}.npz"
            np.savez_compressed(
                out_file,
                screenshot=screenshot, isens=isens, V=V,
                V_rank1=V_rank1, V_svd_k1=V_svd_k1,
                V_svd_k4=V_svd_k4, V_svd_k8=V_svd_k8,
                V_svd_k1_noise=V_svd_k1_noise,
                metadata=np.array([json.dumps(metadata)]),
            )
            generated += 1
            if generated % 50 == 0:
                print(f"[gen] {generated}/{n} Playwright samples")

        browser.close()
    print(f"[gen] Done: {generated} samples in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/tmp/vdlm_eval")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--pages", nargs="+", default=["login", "checkout"])
    parser.add_argument("--playwright", action="store_true")
    args = parser.parse_args()
    generate_dataset(Path(args.out_dir), args.n, args.pages, args.playwright)

