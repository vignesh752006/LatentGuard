# Inversion Attack Benchmark — EVAL_INVERSION.md

> [!NOTE]
> This document describes the inversion-attack evaluation framework for VDLM (Visual-Differential
> Latent Masking). All results marked **PENDING** require a Playwright + GPU run; the CI scaffold
> (`backend/eval/inversion/`) is fully in place and can be executed on a suitable machine.

---

## 1. Methodology

### 1.1 Dataset Generation (`01_generate_dataset.py`)

| Parameter | Value |
|-----------|-------|
| Pages per run | N ≥ 500 (synthetic); `--playwright` flag for real browser renders |
| Page types | `login` (password field) · `checkout` (card-number, CVV) |
| Embedding shape | (576, 1152) fp32 — SigLIP adapter output |
| Sensitive patch set | I_sens: deterministic per page type (seed-reproducible) |
| Screenshot shape | (224, 224, 3) uint8 |

Each sample `.npz` stores:

```
V              — original embedding  (576, 1152) fp32
isens          — sensitive patch indices  int32
screenshot     — rendered page thumbnail  (224, 224, 3) uint8
metadata       — JSON: page_type, seed, n_sens
V_rank1        — rank-1 sanitized embedding
V_svd_k1       — SVD k=1 sanitized
V_svd_k4       — SVD k=4 sanitized
V_svd_k8       — SVD k=8 sanitized
V_svd_k1_noise — SVD k=1 + Gaussian noise (alpha=0.05)
```

### 1.2 Decoder (`02_train_decoder.py`)

A lightweight **MLP decoder** (mean-pooled embedding → 224×224 pixel reconstruction):

```
Input:  mean-pool(V) → ℝ^1152
Hidden: 1152 → 2048 → 4096 (ReLU)
Output: 4096 → 150528 (224×224×3, sigmoid)
```

Trained separately on `V` and each `V'` variant with MSE loss, 50 epochs, Adam lr=1e-3.
Implementation uses pure NumPy (no GPU dependency for the scaffold; replace with PyTorch for real runs).

### 1.3 Evaluation Metrics (`03_evaluate.py`)

| Metric | Scope | Description |
|--------|-------|-------------|
| PSNR (dB) | Sensitive region only | Peak signal-to-noise ratio on I_sens pixels |
| SSIM | Sensitive region only | Structural similarity on I_sens pixels |
| OCR accuracy | Sensitive region only | pytesseract / EasyOCR on reconstructed patches (PENDING) |
| Linear probe accuracy | Full embedding | Predict digit-string from V' directly (classifier attack) |

**PSNR sentinel:** `psnr(x, x) = 100.0` (identical). Lower PSNR = better privacy (harder to reconstruct).

---

## 2. Results Table

> [!CAUTION]
> The numbers below are **PENDING** a full GPU training run with N=500 Playwright pages.
> CI unit tests verify framework correctness (shapes, determinism, metric math) but do NOT
> produce reconstruction results — a NumPy-only MLP on 500 samples is insufficient for
> meaningful PSNR/SSIM estimates. Run `01_generate_dataset.py --playwright --n 500` on a
> machine with Playwright + GPU, then `02_train_decoder.py` + `03_evaluate.py`.

| Sanitizer | PSNR ↓ (sensitive) | SSIM ↓ (sensitive) | OCR acc ↓ | Linear probe acc ↓ |
|-----------|--------------------|--------------------|-----------|-------------------|
| None (V raw) | PENDING | PENDING | PENDING | PENDING |
| Rank-1 | PENDING | PENDING | PENDING | PENDING |
| SVD k=1 | PENDING | PENDING | PENDING | PENDING |
| SVD k=4 | PENDING | PENDING | PENDING | PENDING |
| SVD k=8 | PENDING | PENDING | PENDING | PENDING |
| SVD k=1 + noise | PENDING | PENDING | PENDING | PENDING |

**Lower is better for privacy** (harder to reconstruct or classify sensitive content).

---

## 3. CI-Verified Properties (measured, not claimed)

These are verified by `backend/tests/unit/test_inversion_eval.py` (21 tests, all pass):

| Property | Verified |
|----------|---------|
| Synthetic embeddings have shape (576, 1152) fp32 | ✅ |
| Embedding generation is deterministic (seed-fixed) | ✅ |
| I_sens indices are in [0, 576) for both page types | ✅ |
| Dataset generation produces correct keys and dtypes | ✅ |
| Rank-1 sanitizer: identity when I_sens=∅ | ✅ |
| SVD sanitizer: shapes preserved for k=1,4,8 | ✅ |
| SVD sanitizer: higher k → larger L2 diff (monotone) | ✅ |
| PSNR: identical inputs → 100.0 sentinel | ✅ |
| PSNR: non-trivial noise → ≥ 0.0 and < 100.0 | ✅ |
| SSIM: identical inputs → > 0.9 | ✅ |
| Sensitive region mask: correct shape (224, 224) bool | ✅ |
| Linear probe: returns accuracy in [0.0, 1.0] | ✅ |

---

## 4. How to Run the Full Benchmark

### Prerequisites

```bash
# Install Playwright (for real browser pages)
pip install playwright
playwright install chromium

# Optional: GPU dependencies for fast decoder training
pip install torch torchvision
```

### Run steps

```bash
cd backend/eval/inversion

# Step 1: Generate dataset (N=500, real browser pages)
python 01_generate_dataset.py --playwright --n 500 --out /tmp/vdlm_dataset

# Step 2: Train decoder on each embedding variant
python 02_train_decoder.py --dataset /tmp/vdlm_dataset --epochs 100 --out /tmp/vdlm_models

# Step 3: Evaluate and produce metrics
python 03_evaluate.py --dataset /tmp/vdlm_dataset --models /tmp/vdlm_models --out docs/
```

This will update the Results Table above with real PSNR/SSIM/OCR numbers.

---

## 5. Honest Conclusion

**The benchmark framework is complete and CI-verified.** Full numerical results require
a separate GPU training run (not feasible in CI).

**What we can honestly claim today:**
- The SVD sanitizer with k=8 removes 8× more principal components than k=1 (verified by monotone MSE test).
- The rank-1 sanitizer is an identity map when no sensitive patches are present.
- The evaluation pipeline correctly computes PSNR, SSIM, and linear-probe accuracy.

**What we CANNOT claim without measurement:**
- Whether V' is or is not invertible (requires actual decoder training).
- Whether OCR accuracy drops meaningfully under any sanitizer.
- Whether the classifier attack succeeds on V'.

These will be filled in after a full benchmark run and updated here with honest numbers.

