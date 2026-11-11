# Engineering Notes

## Blueprint inconsistencies acknowledged

1. **Payload size vs. tensor size**: Real adapted SigLIP FP16 `576×1152` frames average 1,225,180 bytes with zlib; per-row int8 averages 611,811 bytes and top-10% sparsity averages 139,001 bytes. A 64-dimensional random projection compresses to a 37,469-byte mean binary body, but averages only 0.241391 reconstructed per-patch cosine and expands to 49,959 Base64 bytes. The `<45 KB` target is not met on the current wire format; measurements and setup are in `docs/ASSUMPTIONS.md`.

2. **"Text-only LLM" on spatial latents**: LLaMA-3-70B-Instruct is text-only; interpreting raw 576×1152 vision embeddings requires either (a) a learned projection head trained on sanitized latents, or (b) a VLM decoder. This implementation passes a **derived spatial summary** (patch statistics + grid features) to the text planner as an interim bridge, with a clear interface for a future VLM decoder module.

3. **SVD in WGSL**: Client-side null-space projection is specified for WebGPU; server never performs VDLM sanitization.

## Module boundaries

```
app/
  config/          Settings from environment
  schemas/         Pydantic wire + domain models
  validation/      Latent + action schema validation
  tensor/          Serialize/deserialize, shape contracts
  security/        Forbidden payload detection
  session/         Browser state, version sync
  planning/        Reasoning pipeline orchestration
  models/          Replaceable planner providers
  transport/       WebSocket handler + protocol
  observability/   Structured logging
  api/             HTTP routes (health, readiness)
```

## Client integration status

The extension now:

1. Capture tab frame locally (never sent to server)
2. Run SigLIP-Base via ONNX Runtime Web (FP16 WebGPU graph, FP32 WASM fallback; mock remains default)
3. Apply WGSL null-space shader → `V'`
4. Stream `latent.frame` over WebSocket
5. Receive `action.directive` and dispatch DOM events

The WebGPU provider was unavailable in the measured browser, and the complete
ONNX model files are large. In the measured Windows Playwright Chromium runtime,
the FP32 WASM path completed 20 full encodes at p50 981.550 ms and p95
1,054.125 ms. No backend redesign is required if protocol v1 is followed.
