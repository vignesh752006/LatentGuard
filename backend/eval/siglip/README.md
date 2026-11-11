# SigLIP vision reference, parity, and payload measurements

This directory contains the Python reference path, the fixed-projection
implementation, and five image/output fixture pairs. The two demo screenshots
use a 1280×720 viewport. The remaining three source images are NASA public
images `iss073e0763824`, `iss009e05944`, and `iss009e05953`, downloaded from
`https://images-assets.nasa.gov/image/{asset-id}/{asset-id}~orig.jpg`; cite
those NASA IDs if reusing the images.

All five `.npy` arrays contain the Hugging Face SigLIP vision tower
`last_hidden_state` with shape `[196, 768]` and dtype `float32`, before the
VDLM 24×24 adapter. `fixture-metadata.json` records the fixed seed, model ID, image sources, and
the exact installed reference/export/runtime library versions.

## Reproduce the fixtures and export

From the repository root, install the optional dependencies:

```powershell
python -m pip install -e "backend[siglip-eval]"
```

Generate the reference outputs:

```powershell
python backend/eval/siglip/reference.py --write-fixtures
```

Export the FP16 vision graph for WebGPU, a numerically matching FP32 graph for
WASM, and the separate fixed-projection adapter (all three are required):

```powershell
python scripts/download-siglip-onnx.py --out-dir client/public/models
```

The vision exports use `SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")`,
`torch.onnx.export(..., input_names=["pixel_values"],
output_names=["last_hidden_state"], opset_version=17, dynamo=False)`, followed by
`onnxconverter_common.float16.convert_float_to_float16(..., keep_io_types=True)`
with explicit Cast nodes at both boundaries for the WebGPU FP16 graph. The WASM
fallback retains the exported FP32 graph. A separate diagnostic FP16-on-WASM
vision inference took 236,686 ms in a non-isolated browser run; in the
cross-origin-isolated 20-frame benchmark, FP32 WASM measured p50 981.550 ms and
p95 1,054.125 ms for full encode. The corresponding model files are large
(171,804,953-byte FP16; 343,344,204-byte FP32; 3,539,707-byte adapter). No
model file belongs in source control.

The browser uses separable bilinear resize with the same half-pixel sampling
and 8-bit intermediate rounding as PIL, then RGB `/255`, normalization
`(x-0.5)/0.5`, and CHW layout. The adapter graph performs 14×14→24×24
bilinear upsampling followed by the shared 768→1152 projection. Keeping the
adapter in ONNX lets ONNX Runtime use optimized matrix kernels rather than a
JavaScript triple loop.

Install the browser and run the actual ONNX parity check:

```powershell
cd client
npx playwright install chromium
npm run test:siglip
```

`python -m pytest backend/eval/siglip/test_onnx_parity.py` compares the FP16
ONNX vision graph with all five PyTorch references and requires mean per-patch
cosine ≥0.99. `cd client; npm run test:siglip` verifies the 576×1152 browser
adapter output, determinism, missing-model errors, and browser WASM parity on
the same fixtures; it measures 20 full encode calls after warm-up. It probes
WebGPU separately and reports it unavailable if Chromium has no adapter rather
than substituting another provider.

Reproduce the real-embedding compact payload and projection measurements with:

```powershell
python backend/eval/siglip/measure_payloads.py
```

The script applies the shared 768→1152 adapter to the five reference arrays,
then measures fp16/zlib and per-row int8/zlib sizes, optional top-10% sparsity,
and seeded orthogonal random projections to 256, 128, and 64 dimensions. On
these five images, mean compressed binary sizes were 1,225,180 bytes (fp16),
611,811 bytes (int8), and 139,001 bytes (int8 with top-10% sparsity). The
projection sizes and reconstructed per-patch cosine were 140,193 bytes / 0.474137
(256D), 71,765 bytes / 0.339956 (128D), and 37,469 bytes / 0.241391 (64D).
The 64D Base64 body averages 49,959 bytes before JSON metadata, so the 45 KB
wire target is not met. For a projected tensor, cosine fidelity is the
per-patch cosine between the original adapted row and its reconstruction
through the transpose of the orthonormal projection basis. Compressed sizes
are binary bytes before Base64 and JSON framing.

## Projection matrix

The TypeScript implementation is in `client/src/vision/projection.ts`; the
Python implementation is in `projection.py`. Both use the same uint32
linear-congruential generator (`state = (1664525 * state + 1013904223) mod
2^32`, seed `0xdeadbeef`), fill row-major float32 values in [-1, 1], and
normalize each output column to unit L2 norm. `projection-matrix.sha256` is
the shared byte-for-byte fixture checked by both test suites.
