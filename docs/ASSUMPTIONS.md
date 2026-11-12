# VDLM Engineering Assumptions vs. Verified Behavior

This document separates **research objectives** described in the master blueprint from **what this backend implementation actually guarantees today**.

## Aspirational (not verified by this codebase)

| Claim in blueprint | Status |
|--------------------|--------|
| Mathematically proven zero-knowledge privacy | **Not implemented.** No ZKP layer yet; see `provenance` extension fields. |
| 100% spatial layout preservation after null-space projection | **Not verified.** Requires client-side VDLM shader + benchmark suite. |
| Non-invertibility of `V'` against feature inversion decoders | **Not established.** See the measured but limited Task 5 benchmark below; no privacy guarantee follows from it. |
| `>92%` action accuracy parity vs. unredacted baseline | **Not measured.** Evaluation harness is future work. |

## Verified by this backend

| Behavior | Mechanism |
|----------|-----------|
| Raw screenshots rejected | `security/payload_guard.py` blocks forbidden fields and MIME types |
| DOM/credential fields rejected | Schema validation on `LatentFramePayload` |
| Tensor shape enforced | SigLIP-Base contract: 576×1152 float16 (configurable) |
| Stale plan detection | `state_version` comparison in session manager |
| Replaceable LLM provider | `models/provider.py` abstraction with mock default |
| Structured action output | Pydantic-validated `ActionDirective` schema |

## Client-side (extension)

| Behavior | Mechanism |
|----------|-----------|
| Raw screenshots not transmitted | Capture stays local; only sanitized `V'` tensors sent |
| Sensitivity geometry only to WASM | DOM scanner exports bounding boxes, not text |
| WebGPU vision + VDLM | WGSL shaders in offscreen document |
| Stale action rejection | Content script compares `state_version_required` |
| No arbitrary JS from model | Fixed action executor whitelist |

Client does **not** verify cryptographic provenance (`provenance.attestation` is always `null` today).

| Gap | Backend support |
|-----|-----------------|
| GAP 3: SPA state drift | `state_version`, `frame_id`, `plan.stale` responses |
| GAP 4: Bandwidth | Optional `compression`, delta frame types in protocol |
| GAP 5: Crypto provenance | Optional `SanitizationProvenance` fields (verification stub) |

---

## Task 3 — Real SigLIP Payload Sizes and Fidelity (2026-11-12)

Measurements use the five checked-in reference images (two 1280×720 demo
screenshots and three NASA images), with their real `[196,768]` SigLIP outputs
adapted to `[576,1152]` by bilinear 14×14→24×24 upsampling and the shared seeded
768→1152 projection. The Python adapted arrays matched the exported ONNX
adapter within max absolute error `1.96e-5`. Size results below are zlib
compressed binary bytes before Base64 and JSON framing; see the Base64 note for
the actual transport threshold.

| Encoding / transform | Mean bytes (min–max across 5 images) | Mean per-patch cosine vs. full adapted V | 45 KB result |
|---|---:|---:|---|
| fp16 + zlib | 1,225,069 (1,224,710–1,225,426) | Not measured | target NOT met |
| Per-row int8 + zlib | 611,777 (609,167–614,207) | 0.999970 (minimum patch 0.999940) | target NOT met |
| Per-row int8, top 10% + zlib | 139,009 (138,608–139,448) | 0.659037 (minimum patch 0.630010) | target NOT met |
| Orthogonal random projection to 256 dims, int8 + zlib | 140,178 (139,237–140,729) | 0.474077 (minimum patch 0.415922) | target NOT met |
| Orthogonal random projection to 128 dims, int8 + zlib | 71,750 (71,053–72,154) | 0.339937 (minimum patch 0.291390) | target NOT met |
| Orthogonal random projection to 64 dims, int8 + zlib | 37,460 (36,947–37,866) | 0.241495 (minimum patch 0.180520) | target NOT met on Base64 wire |

The 64-dimensional compressed binary body averages 37,460 bytes, but Base64
expands it to an average 49,947 bytes before JSON metadata. Thus the
adapter-space random projection does **not** meet the `<45 KB` target on the
current Base64 WebSocket wire. Even ignoring Base64, reaching the binary
threshold requires that projection, whose reconstructed per-patch cosine
averages only 0.241391. Top-10% sparsity is also lossy (mean cosine 0.659033).
The source-grid PCA representation measured below reaches the size target, with
substantial reconstruction-fidelity loss.

The exact client `fflate` serializer measurements on these five browser-adapted
embeddings were 1,227,736 compressed / 1,636,984 Base64 bytes for fp16,
612,314 / 816,420 bytes for per-row int8, and 143,211 / 190,950 bytes for
per-row int8 with top-10% sparsity. These measurements confirm that the current
client wire payload also misses the target for each enabled full-dimension
encoding.

The random projection uses seed `0xdeadbeef`, a NumPy standard-normal
`1152×256` matrix, reduced QR orthonormalization, and the first 64/128/256
columns. Fidelity is the mean cosine for each original adapted patch row
against its reconstruction through the transposed orthonormal basis. The
per-image measurements and exact procedure are reproducible with
`backend/eval/siglip/measure_payloads.py`.

### Source-PCA held-out payload and fidelity

The canonical rank-128 basis was refit using real SigLIP FP32-ONNX outputs from
303 Playwright screenshots: 280 controlled synthetic form pages and 23 public
website pages. Pages, not tokens, were split deterministically: 242 training
pages and 61 held-out pages. PCA was fit only to training-page vectors. The
held-out measurements serialize coefficients as the source-PCA tensor JSON,
then run the backend decoder to rebuild `[576,1152]`. Alpha is applied to the
source sanitizer before PCA (`k=4`); rank and alpha are varied as shown.

| PCA rank | Alpha | Mean tensor JSON bytes (min–max) | Held-out mean patch cosine (min–max per page) |
|---:|---:|---:|---:|
| 64 | 0 | 15,934 (15,710–16,414) | 0.759635 (0.620245–0.803485) |
| 64 | 0.01 | 15,934 (15,718–16,414) | 0.759635 (0.620245–0.803482) |
| 128 | 0 | 28,983 (28,419–30,703) | 0.842008 (0.708113–0.874339) |
| 128 | 0.01 | 28,984 (28,431–30,703) | 0.842008 (0.708113–0.874344) |

These are tensor-object JSON bytes (including Base64 data, excluding the
constant latent-frame envelope and WebSocket framing). Both measured ranks are
below the 45,056-byte target. Rank 128 costs about 13 KB more for higher
held-out cosine; alpha 0.01 had negligible impact on this reconstruction
metric. Cosine is not action accuracy or evidence that sensitive content
cannot be recovered.

The basis occupies 396,288 bytes. Fit seed, model/runtime versions, public URLs,
page split (52 synthetic / 9 public held out), SHA-256, and per-page held-out
results are recorded in
`backend/eval/siglip/fixtures/source-pca-metadata.json`. Public screenshots
were not committed; the URLs can be recaptured, but live page contents may
change after the recorded capture time.

The measured source-PCA tensor JSON sizes are below the 45,056-byte target at
both ranks, including the configured `alpha=0.01`. This size result has a
fidelity tradeoff: held-out mean patch cosine was `0.759635` at rank 64 and
`0.842008` at rank 128. It is not evidence of action-accuracy parity or
resistance to inversion.

### Task 5 — Inversion benchmark measurements

The 10-page CI smoke test passed with real SigLIP ONNX embeddings, an 8/2
page-level split, and two decoder epochs. It exercised the `V` and
`Vprime_svd_k4_alpha001` decoder paths; EasyOCR was skipped in that smoke test.

The full run used 500 synthetic Playwright pages (400 training / 100 held out),
the 343,344,204-byte FP32 SigLIP model, and the exact rank-64 source-PCA
wire path: source sanitization, PCA, per-row int8, zlib/Base64 JSON, and
backend reconstruction to `[576,1152]`. Ten convolutional decoder variants
were trained for two epochs each. Held-out sensitive-region PSNR ranged from
25.183 to 25.442 dB and SSIM from 0.7123 to 0.7218.

The direct ridge digit probe used global-mean latent features and predicted
1,600 held-out card-number characters. Uniform-character chance was 0.100;
the training-majority baseline was 0.09875:

| Representation | Held-out digit character accuracy |
|---|---:|
| V (unsanitized) | 0.12500 |
| None sanitizer / PCA+wire only | 0.11063 |
| Rank-1 baseline, alpha 0 | 0.12000 |
| Rank-1 baseline, alpha 0.01 | 0.11813 |
| SVD k=1, alpha 0 | 0.11313 |
| SVD k=1, alpha 0.01 | 0.11063 |
| SVD k=4, alpha 0 | 0.10750 |
| SVD k=4, alpha 0.01 | 0.10938 |
| SVD k=8, alpha 0 | 0.11000 |
| SVD k=8, alpha 0.01 | 0.11250 |

EasyOCR 1.7.2 reported zero password and card character accuracy on all ten
reconstructed representations. As a control, on the original 100 held-out
screenshots it scored 0.000 password and 0.0025 card character accuracy. The
OCR result therefore does not measure sanitizer benefit. The digit probe
results are modestly above chance for some settings and below the unsanitized
`V` result, but are only one simple attack.

The benchmark does not establish that `V'` is non-invertible: the decoders
produced sensitive-region image reconstructions with the PSNR/SSIM values
above, and no stronger decoder or attack was evaluated. Conversely, these
metrics do not prove that the original credentials were recovered. This
synthetic-only, two-epoch evaluation supports no privacy guarantee. Full
per-variant metrics, charts, setup, and limitations are in
[`EVAL_INVERSION.md`](./EVAL_INVERSION.md); machine-readable results are in
[`eval_inversion_results.json`](./eval_inversion_results.json).

The SVD integration test ran on a real login screenshot through the browser
SigLIP encoder and adapted `[576,1152]` embeddings with `k=4`, `alpha=0`, and
eight selected sensitive patches. The maximum absolute `V'·W_priv` dot over
all output patches and four basis vectors was `1.48e-6` (threshold `1e-4`).
Normal client defaults are `k=4`, configurable alpha `0.01`; noise is disabled
in this orthogonality measurement. The source-PCA reconstruction fidelity
measurement also used `alpha=0` to isolate the PCA quantization/reconstruction
loss; the configured normal default is `alpha=0.01`.

The oversized NASA fixture `natural-iss073e0763824.jpg` was downscaled from
13,934,907 bytes at 8256×5504 to 757,040 bytes at 2048×1365, and its reference
embedding was regenerated. The resized image's browser WASM parity score was
0.999975.

### SigLIP ONNX parity and runtime

The reference uses `google/siglip-base-patch16-224`, RGB PIL bilinear resize to
224×224, `/255`, `(x-0.5)/0.5`, and CHW. FP16 ONNX mean patch cosine on the five
fixtures using ONNX Runtime 1.31.1 CPU is:

| Image | Mean patch cosine |
|---|---:|
| Login demo | 0.999987 |
| Checkout demo | 0.999988 |
| NASA `iss073e0763824` | 0.999964 |
| NASA `iss009e05944` | 0.999992 |
| NASA `iss009e05953` | 0.999985 |

The browser WASM FP32 fallback measured cosines `0.9999999997`,
`0.9999989099`, `0.9999750477`, `0.9999990943`, and `0.9999974977` respectively, with output
shape `[576,1152]`. The browser test also passed deterministic-output and
missing-model error checks.

| Model artifact | Measured size |
|---|---:|
| FP16 vision graph (WebGPU path) | 171,804,953 bytes |
| FP32 vision graph (WASM fallback) | 343,344,204 bytes |
| Fixed projection adapter | 3,539,707 bytes |

On Windows, Playwright Chromium `158.0.8127.18`, 8 reported logical cores, and
cross-origin isolation enabled:

| Provider | Samples | p50 full encode | p95 full encode | Result |
|---|---:|---:|---:|---|
| WASM with FP32 model | 20 repeated login frames after warm-up | 979.930 ms | 1,109.120 ms | Measured |
| WebGPU with FP16 model | 0 | — | — | No adapter available |

Full encode includes preprocessing, vision inference, and the adapter. A
separate diagnostic FP16-on-WASM vision inference took `236,686 ms` in a
non-isolated browser run; this is not the 20-frame WASM benchmark. The extension
manifest and the Vite test server enable cross-origin isolation for WASM
threads. Model files are ignored by git; Vite copies locally exported models
into the ignored `client/dist` directory when building the extension.

Reference fixture and library versions are in
`backend/eval/siglip/fixtures/fixture-metadata.json`. Run the five-image FP16
ONNX check with `python -m pytest backend/eval/siglip/test_onnx_parity.py`, and
the browser parity/latency check with `cd client; npm run test:siglip`.

## Task 2 — Visible demo-page scan (2026-11-11)

Measured by the Playwright test with a 1280×720 viewport and the production
`scanSensitiveRegions()` and `mapRegionsToPatchesTs()` implementations:

| Page | Scanner regions | Target `I_sens` (24×24) | Target grid fraction | Source indices (14×14) | Source grid fraction |
|---|---:|---:|---:|---:|---:|
| Login | 3 | 40/576 | 6.94% | 18/196 | 9.18% |
| Checkout | 6 | 64/576 | 11.11% | 30/196 | 15.31% |
| No-sensitive-fields page | 0 | 0/576 | 0% | 0/196 | 0% |

Fractions are the percentage of patch-grid cells flagged, not exact pixel-area
coverage. Source indices use the production target-to-source center mapping.
Exact target patch-set assertions passed at 1280×720.

The scanner reports the marked canvas under both its `data-vdlm-sensitive`
custom selector and its accessible sensitive-role detection; those regions map
to the same patches. These counts describe only the deterministic demo pages
and tested viewport, not general-world detection accuracy or privacy.
