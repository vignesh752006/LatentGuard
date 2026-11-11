# VDLM Engineering Assumptions vs. Verified Behavior

This document separates **research objectives** described in the master blueprint from **what this backend implementation actually guarantees today**.

## Aspirational (not verified by this codebase)

| Claim in blueprint | Status |
|--------------------|--------|
| Mathematically proven zero-knowledge privacy | **Not implemented.** No ZKP layer yet; see `provenance` extension fields. |
| 100% spatial layout preservation after null-space projection | **Not verified.** Requires client-side VDLM shader + benchmark suite. |
| Non-invertibility of `V'` against feature inversion decoders | **Not verified.** Phase 3 penetration testing is future work. |
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
| fp16 + zlib | 1,225,180 (1,224,710–1,225,554) | Not measured | target NOT met |
| Per-row int8 + zlib | 611,811 (609,167–614,207) | 0.999970 (minimum patch 0.999940) | target NOT met |
| Per-row int8, top 10% + zlib | 139,001 (138,608–139,448) | 0.659033 (minimum patch 0.630010) | target NOT met |
| Orthogonal random projection to 256 dims, int8 + zlib | 140,193 (139,237–140,729) | 0.474137 (minimum patch 0.415922) | target NOT met |
| Orthogonal random projection to 128 dims, int8 + zlib | 71,765 (71,053–72,154) | 0.339956 (minimum patch 0.291390) | target NOT met |
| Orthogonal random projection to 64 dims, int8 + zlib | 37,469 (36,947–37,866) | 0.241391 (minimum patch 0.180520) | target NOT met on Base64 wire |

The 64-dimensional compressed binary body averages 37,469 bytes, but Base64
expands it to an average 49,959 bytes before JSON metadata. Thus the
blueprint's `<45 KB` target is **not met on the current Base64 WebSocket wire**.
Even ignoring Base64, reaching the binary threshold requires the 64-dimensional
projection, whose reconstructed per-patch cosine averages only 0.241391. The
lower-dimensional transforms are exploratory and are not enabled in the
576×1152 protocol. Top-10% sparsity is also lossy (mean cosine 0.659033).

The exact client `fflate` serializer measurements on these five browser-adapted
embeddings were 1,227,820 compressed / 1,637,095 Base64 bytes for fp16,
612,345 / 816,462 bytes for per-row int8, and 143,237 / 190,983 bytes for
per-row int8 with top-10% sparsity. These measurements confirm that the current
client wire payload also misses the target for each enabled full-dimension
encoding.

The random projection uses seed `0xdeadbeef`, a NumPy standard-normal
`1152×256` matrix, reduced QR orthonormalization, and the first 64/128/256
columns. Fidelity is the mean cosine for each original adapted patch row
against its reconstruction through the transposed orthonormal basis. The
per-image measurements and exact procedure are reproducible with
`backend/eval/siglip/measure_payloads.py`.

### SigLIP ONNX parity and runtime

The reference uses `google/siglip-base-patch16-224`, RGB PIL bilinear resize to
224×224, `/255`, `(x-0.5)/0.5`, and CHW. FP16 ONNX mean patch cosine on the five
fixtures using ONNX Runtime 1.31.1 CPU is:

| Image | Mean patch cosine |
|---|---:|
| Login demo | 0.999987 |
| Checkout demo | 0.999988 |
| NASA `iss073e0763824` | 0.999951 |
| NASA `iss009e05944` | 0.999992 |
| NASA `iss009e05953` | 0.999985 |

The browser WASM FP32 fallback measured cosines `0.9999999997`, `0.9999989099`,
`0.9999996683`, `0.9999990943`, and `0.9999974977` respectively, with output
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
| WASM with FP32 model | 20 repeated login frames after warm-up | 981.550 ms | 1,054.125 ms | Measured |
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

| Page | Scanner regions | `I_sens` patch count | Result |
|---|---:|---:|---|
| Login | 3 | 40 | Exact patch-set assertion passed for the visible password and marked canvas |
| Checkout | 6 | 64 | Exact patch-set assertion passed for card number, CVV, name, and marked canvas |
| No-sensitive-fields page | 0 | 0 | Empty `I_sens` assertion passed |

The scanner reports the marked canvas under both its `data-vdlm-sensitive`
custom selector and its accessible sensitive-role detection; those regions map
to the same patches. These counts describe only the deterministic demo pages
and tested viewport, not general-world detection accuracy or privacy.
