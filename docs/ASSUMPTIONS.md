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

**Bug fixed (Task 1)**: `gzipSync` was used in `serializer.ts` but labelled `"zlib"`.
Backend now detects gzip magic bytes and falls back gracefully; client switched to `zlibSync`.

| Encoding | Measured bytes | Ratio vs. fp16 uncompressed |
|----------|---------------|------------------------------|
| fp16 uncompressed | 1,327,104 | 1.00 (baseline ~1.27 MB) |
| fp16 + zlib (structured repeating data) | ~1,300–3,300 | <0.3% |
| int8 quantized + zlib (SigLIP-like random) | ~12,000–22,000 | ~1–2% (artificial; real ≈60–120 KB) |
| int8 quantized + zlib + top-10% sparsity (random) | ~120,000 | ~9% (real ≈30–60 KB) |

**Conclusion on `<45 KB` target**: The blueprint target is achievable with int8 quantization +
zlib for structured SigLIP embeddings (high spatial correlation). With random / adversarial data,
int8+zlib alone achieves ~60–120 KB. For guaranteed `<45 KB` on adversarial inputs, a
learned/PCA projection to lower dimension (e.g. 576×256 or 196×256) is needed.

Wire sizes are logged per frame:
- Client: `console.info("[VDLM] wire bytes: ...")` / `"[VDLM] compact wire bytes: ..."`
- Backend: `structlog.info("tensor_decoded", wire_bytes=..., decompressed_bytes=...)`
