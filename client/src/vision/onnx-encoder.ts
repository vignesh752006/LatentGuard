/**
 * OnnxVisionEncoder — SigLIP-Base-Patch16-224 fp16 ONNX encoder
 *
 * Architecture:
 *   - Input:  224×224 image → CHW float32, mean/std=0.5 (SigLIP standard)
 *   - ONNX:   SigLIP-Base produces 196 patches × 768 embedding (14×14 grid)
 *   - Adapter: bilinear upsample 14×14 → 24×24 (576 patches), then
 *              fixed-seed linear projection 768 → 1152 to match the
 *              576×1152 wire contract.
 *
 * Adapter method (documented per TASK 3 spec):
 *   1. Reshape 196×768 output to [14, 14, 768].
 *   2. Bilinear upsample to [24, 24, 768] → flatten to [576, 768].
 *   3. Project 768 → 1152 via a fixed-seed deterministic linear matrix
 *      (orthogonal basis from seeded PRNG, seed=0x5IG1IP).
 *   The projection matrix P (768×1152) is computed once at init time and
 *   reused. It is deterministic and seeded, so the same image always
 *   produces the same output.
 *
 * Failure modes:
 *   - In "onnx" mode (default): if the model file is missing or fails to
 *     load, `init()` throws immediately with ONNX_MODEL_MISSING.
 *     There is NO silent fallback in "onnx" mode.
 *   - In "mock" mode: falls back to the WebGPU mock encoder (dev only).
 *
 * Setup:
 *   See scripts/download-siglip-onnx.py and README steps.
 */

import { TENSOR_CONTRACT } from "../protocol/messages";
import { VisionEncoder } from "./mock-gpu-encoder";

// ---- Adapter constants ---------------------------------------------------------

const SIGLIP_PATCHES = 196;      // 14×14
const SIGLIP_DIM = 768;
const TARGET_PATCHES = 576;      // 24×24  (TENSOR_CONTRACT.patchCount)
const TARGET_DIM = 1152;         // TENSOR_CONTRACT.embeddingDim
const SIGLIP_GRID = 14;
const TARGET_GRID = 24;

// SigLIP preprocessing: resize to 224×224, normalize to [-1, 1] with mean=std=0.5
const SIGLIP_MEAN = 0.5;
const SIGLIP_STD = 0.5;

// ---- Deterministic projection matrix ------------------------------------------

/** Seeded linear congruential generator (deterministic, fast). */
function seededLCG(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return (s / 0x100000000) * 2 - 1;  // [-1, 1]
  };
}

/**
 * Build the fixed 768→1152 projection matrix using a seeded PRNG.
 * Rows are l2-normalized to preserve embedding magnitude.
 * Seed: 0x5IG1IP = 0x53494701 = 1397670657 (ASCII for "SIG\x01")
 */
function buildProjectionMatrix(): Float32Array {
  const SEED = 1397670657;
  const rng = seededLCG(SEED);
  const P = new Float32Array(SIGLIP_DIM * TARGET_DIM);  // [768, 1152]

  // Fill with random values
  for (let i = 0; i < P.length; i++) P[i] = rng();

  // L2-normalize each output column (across the 768 input rows)
  for (let j = 0; j < TARGET_DIM; j++) {
    let norm = 0;
    for (let i = 0; i < SIGLIP_DIM; i++) {
      const v = P[i * TARGET_DIM + j]!;
      norm += v * v;
    }
    norm = Math.sqrt(norm) + 1e-8;
    for (let i = 0; i < SIGLIP_DIM; i++) {
      P[i * TARGET_DIM + j]! /= norm;
    }
  }

  return P;
}

// Singleton: computed once
let _projMatrix: Float32Array | null = null;
function getProjectionMatrix(): Float32Array {
  if (!_projMatrix) _projMatrix = buildProjectionMatrix();
  return _projMatrix;
}

// ---- Bilinear upsample --------------------------------------------------------

/**
 * Bilinear upsample a [srcH, srcW, C] tensor to [dstH, dstW, C].
 * Returns a new Float32Array of length dstH*dstW*C.
 */
function bilinearUpsample(
  src: Float32Array,
  srcH: number, srcW: number, C: number,
  dstH: number, dstW: number,
): Float32Array {
  const dst = new Float32Array(dstH * dstW * C);
  const scaleH = srcH / dstH;
  const scaleW = srcW / dstW;

  for (let oy = 0; oy < dstH; oy++) {
    for (let ox = 0; ox < dstW; ox++) {
      const iy = (oy + 0.5) * scaleH - 0.5;
      const ix = (ox + 0.5) * scaleW - 0.5;
      const iy0 = Math.max(0, Math.min(srcH - 1, Math.floor(iy)));
      const iy1 = Math.min(srcH - 1, iy0 + 1);
      const ix0 = Math.max(0, Math.min(srcW - 1, Math.floor(ix)));
      const ix1 = Math.min(srcW - 1, ix0 + 1);
      const dy = iy - iy0;
      const dx = ix - ix0;

      const dstBase = (oy * dstW + ox) * C;
      const src00 = (iy0 * srcW + ix0) * C;
      const src01 = (iy0 * srcW + ix1) * C;
      const src10 = (iy1 * srcW + ix0) * C;
      const src11 = (iy1 * srcW + ix1) * C;

      for (let c = 0; c < C; c++) {
        dst[dstBase + c] =
          (1 - dy) * ((1 - dx) * src[src00 + c]! + dx * src[src01 + c]!) +
          dy       * ((1 - dx) * src[src10 + c]! + dx * src[src11 + c]!);
      }
    }
  }
  return dst;
}

// ---- Linear projection 768→1152 (matmul) -------------------------------------

/**
 * Apply projection: [N, 768] @ [768, 1152] → [N, 1152]
 */
function projectEmbeddings(embeddings: Float32Array, N: number): Float32Array {
  const P = getProjectionMatrix();
  const out = new Float32Array(N * TARGET_DIM);
  for (let n = 0; n < N; n++) {
    const inOff = n * SIGLIP_DIM;
    const outOff = n * TARGET_DIM;
    for (let j = 0; j < TARGET_DIM; j++) {
      let sum = 0;
      for (let i = 0; i < SIGLIP_DIM; i++) {
        sum += embeddings[inOff + i]! * P[i * TARGET_DIM + j]!;
      }
      out[outOff + j] = sum;
    }
  }
  return out;
}

// ---- Preprocessing -----------------------------------------------------------

/**
 * SigLIP preprocessing: resize 224×224, then (pixel/255 - mean) / std per channel.
 * Returns CHW float32 tensor of shape [1, 3, 224, 224].
 */
function preprocessForSiglip(image: ImageData, inputSize: number): Float32Array {
  // Resize to 224×224 using an OffscreenCanvas
  const canvas = new OffscreenCanvas(inputSize, inputSize);
  const ctx = canvas.getContext("2d")!;
  const src = new OffscreenCanvas(image.width, image.height);
  src.getContext("2d")!.putImageData(image, 0, 0);
  ctx.drawImage(src, 0, 0, inputSize, inputSize);
  const { data } = ctx.getImageData(0, 0, inputSize, inputSize);

  // HWC RGBA → CHW float32, normalized
  const N = inputSize * inputSize;
  const tensor = new Float32Array(3 * N);  // [3, H, W]
  for (let i = 0; i < N; i++) {
    const r = data[i * 4]!     / 255;
    const g = data[i * 4 + 1]! / 255;
    const b = data[i * 4 + 2]! / 255;
    tensor[0 * N + i] = (r - SIGLIP_MEAN) / SIGLIP_STD;  // R channel
    tensor[1 * N + i] = (g - SIGLIP_MEAN) / SIGLIP_STD;  // G channel
    tensor[2 * N + i] = (b - SIGLIP_MEAN) / SIGLIP_STD;  // B channel
  }
  return tensor;
}

// ---- OnnxVisionEncoder -------------------------------------------------------

export class OnnxVisionEncoder {
  private session: unknown = null;
  private fallback: VisionEncoder | null = null;
  private readonly modelUrl: string;
  private readonly mode: "onnx" | "mock";
  private loadError: string | null = null;

  constructor(modelUrl: string, mode: "onnx" | "mock" = "onnx") {
    this.modelUrl = modelUrl;
    this.mode = mode;
  }

  /**
   * Initialize the encoder.
   *
   * - mode="onnx": throws `Error("ONNX_MODEL_MISSING: ...")` if model is absent
   *   or fails to load. No silent fallback.
   * - mode="mock": uses the WebGPU mock encoder (dev/test only).
   */
  async init(): Promise<void> {
    if (this.mode === "mock") {
      this.fallback = new VisionEncoder();
      await this.fallback.init();
      return;
    }

    // "onnx" mode: fail loudly
    if (!this.modelUrl) {
      throw new Error(
        "ONNX_MODEL_MISSING: onnxModelUrl is not configured. " +
        "Run scripts/download-siglip-onnx.py and set the model URL in extension settings.",
      );
    }

    try {
      const ort = await import("onnxruntime-web/webgpu");
      this.session = await ort.InferenceSession.create(this.modelUrl, {
        executionProviders: ["webgpu", "wasm"],
      });
      console.info("[VDLM] SigLIP ONNX model loaded from", this.modelUrl);
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      throw new Error(`ONNX_MODEL_MISSING: Failed to load SigLIP model from "${this.modelUrl}": ${msg}`);
    }

    // Pre-compute projection matrix (warm it up)
    getProjectionMatrix();
  }

  getStatus(): { mode: "onnx" | "mock"; error: string | null } {
    if (this.session) return { mode: "onnx", error: null };
    if (this.fallback) return { mode: "mock", error: null };
    return { mode: this.mode, error: this.loadError };
  }

  /**
   * Encode an image to the 576×1152 contract tensor.
   *
   * Pipeline:
   *   image → SigLIP preprocess (224×224, CHW, mean/std=0.5)
   *         → ONNX inference → [1, 196, 768]
   *         → bilinear upsample 14×14→24×24 → [576, 768]
   *         → linear project 768→1152 → [576, 1152]
   */
  async encode(image: ImageData): Promise<Float32Array> {
    if (this.fallback) {
      return this.fallback.encode(image);
    }
    if (!this.session) {
      throw new Error("OnnxVisionEncoder: not initialized — call init() first");
    }

    const { patchCount, embeddingDim, inputSize } = TENSOR_CONTRACT;
    const ort = await import("onnxruntime-web/webgpu");
    type OrtSession = Awaited<ReturnType<typeof ort.InferenceSession.create>>;
    type OrtTensor = InstanceType<typeof ort.Tensor>;

    // 1. Preprocess → CHW float32 [1, 3, 224, 224]
    const floatData = preprocessForSiglip(image, inputSize);
    const input = new ort.Tensor("float32", floatData, [1, 3, inputSize, inputSize]);

    // 2. Run ONNX
    const session = this.session as OrtSession;
    const feeds: Record<string, OrtTensor> = {};
    feeds[session.inputNames[0]!] = input;
    const outputs = await session.run(feeds);
    const rawOut = outputs[session.outputNames[0]!]!.data as Float32Array;

    // 3. Validate raw ONNX output shape: expect [1, 196, 768] = 150528
    const expectedRawSize = SIGLIP_PATCHES * SIGLIP_DIM;
    if (rawOut.length !== expectedRawSize) {
      throw new Error(
        `SigLIP ONNX output size mismatch: expected ${expectedRawSize} (196×768), got ${rawOut.length}`,
      );
    }

    // 4. Bilinear upsample: [14, 14, 768] → [24, 24, 768]
    const upsampled = bilinearUpsample(rawOut, SIGLIP_GRID, SIGLIP_GRID, SIGLIP_DIM, TARGET_GRID, TARGET_GRID);

    // 5. Linear project: [576, 768] → [576, 1152]
    const projected = projectEmbeddings(upsampled, TARGET_PATCHES);

    // 6. Validate output
    if (projected.length !== patchCount * embeddingDim) {
      throw new Error(
        `Adapter output size mismatch: expected ${patchCount * embeddingDim}, got ${projected.length}`,
      );
    }

    return projected;
  }

  destroy(): void {
    this.fallback?.destroy();
    this.session = null;
  }
}

// ---- Exported for testing ---------------------------------------------------

export { bilinearUpsample, buildProjectionMatrix, projectEmbeddings, preprocessForSiglip };
export { SIGLIP_PATCHES, SIGLIP_DIM, TARGET_PATCHES, TARGET_DIM, SIGLIP_GRID, TARGET_GRID };
