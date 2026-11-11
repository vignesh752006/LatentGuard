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
| int8 per-row + zlib, seeded input | 665,610 | 50.2% of uncompressed fp16 |
| int8 per-row + zlib, seeded input retaining top 10% per row | 123,425 | 9.3% of uncompressed fp16; above the 45 KB target |

### Lower-dimensional projection experiment

The seeded 576×1152 test matrix was projected with a fixed Gaussian matrix
generated from NumPy seed `0xdeadbeef`, normalized by `sqrt(d)`, then quantized
per row to int8 and zlib-compressed. Pairwise cosine fidelity compares all
165,600 off-diagonal patch pairs after quantization.

| Projected dimension | Wire bytes | Mean absolute pairwise-cosine error | Pairwise-cosine correlation |
|---:|---:|---:|---:|
| 256 | 140,786 | 0.049684 | 0.427566 |
| 128 | 72,312 | 0.070870 | 0.316395 |
| 64 | 37,637 | 0.100167 | 0.223048 |

The 64-dimensional experiment reaches <45 KB on this synthetic input, but its
measured pairwise-cosine correlation is only 0.223048. It is not enabled in the
wire protocol. No projection has been evaluated on real SigLIP output, so no
claim is made about its size or utility for actual frames.

The client logs serialized wire bytes per frame, and the backend logs received
and decompressed byte counts. The protocol currently requires 576×1152 tensors.
