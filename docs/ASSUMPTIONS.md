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

## Task 1 — Measured Wire Sizes (2026-11-11)

Measurements are from the client Vitest tests with 576×1152 tensors. The seeded
size tests use an LCG seeded with `0xdeadbeef`, producing values in [-1, 1].
These are synthetic inputs, not captured SigLIP embeddings.

| Encoding / input | Measured bytes | Notes |
|---|---:|---|
| fp16, uncompressed | 1,327,104 | Raw tensor size |
| fp16 + zlib, repeating test pattern | 3,259 | Highly compressible synthetic input |
| fp16 + zlib, cross-language test pattern | 2,613 | Backend decoded it with max absolute error 0.00039065 |
| int8 per-row + zlib, seeded random input | 665,610 | 50.2% of uncompressed fp16; target NOT met |
| int8 per-row + zlib, seeded random input retaining top 10% per row | 123,425 | 9.3% of uncompressed fp16; target NOT met |

### Lower-dimensional projection experiment

The seeded random 576×1152 test matrix was projected with a fixed Gaussian matrix
generated from NumPy seed `0xdeadbeef`, normalized by `sqrt(d)`, then quantized
per row to int8 and zlib-compressed. Pairwise cosine fidelity compares all
165,600 off-diagonal patch pairs after quantization.

| Projected dimension | Wire bytes | Mean absolute pairwise-cosine error | Pairwise-cosine correlation | 45 KB target |
|---:|---:|---:|---:|---|
| 256 | 140,786 | 0.049684 | 0.427566 | NOT met |
| 128 | 72,312 | 0.070870 | 0.316395 | NOT met |
| 64 | 37,637 | 0.100167 | 0.223048 | Met on this seeded input |

The 64-dimensional seeded-random experiment reaches <45 KB, but its
measured pairwise-cosine correlation is only 0.223048. It is not enabled in the
wire protocol. All size and projection results above were measured on synthetic
seeded random data, not real embeddings. Re-measure after Task 3 with actual
SigLIP output before making a claim about payload size or utility for real frames.

The client logs serialized wire bytes per frame, and the backend logs received
and decompressed byte counts. The protocol currently requires 576×1152 tensors.

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
