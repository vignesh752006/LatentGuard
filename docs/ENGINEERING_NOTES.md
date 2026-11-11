# Engineering Notes

## Blueprint inconsistencies acknowledged

1. **Payload size vs. tensor size**: Real adapted SigLIP FP16 `576×1152` frames average 1,225,069 compressed bytes; per-row int8 averages 611,777 bytes. A 64-dimensional adapter-space random projection misses the 45 KB Base64 target. Source-grid PCA reaches a 15,666-byte mean serialized JSON frame but averages only 0.497006 per-patch cosine against the full adapted sanitized tensor (minimum observed patch cosine -0.176877). The target is reached at a significant, image-dependent fidelity cost; see `docs/ASSUMPTIONS.md`.

2. **"Text-only LLM" on spatial latents**: LLaMA-3-70B-Instruct is text-only; interpreting raw 576×1152 vision embeddings requires either (a) a learned projection head trained on sanitized latents, or (b) a VLM decoder. This implementation passes a **derived spatial summary** (patch statistics + grid features) to the text planner as an interim bridge, with a clear interface for a future VLM decoder module.

3. **SVD execution**: The default multi-rank sanitizer is a CPU implementation in the offscreen extension; the backend receives sanitized tensors and does not perform privacy sanitization.

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
