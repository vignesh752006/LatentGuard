import type { InferenceSession, Tensor as OrtTensor } from "onnxruntime-web";
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url";
import { TENSOR_CONTRACT } from "../protocol/messages";
import {
  buildProjectionMatrix,
  projectEmbeddings,
  SIGLIP_DIM,
  TARGET_DIM,
} from "./projection";

export const SIGLIP_PATCHES = 196;
export const TARGET_PATCHES = TENSOR_CONTRACT.patchCount;
export const SIGLIP_GRID = 14;
export const TARGET_GRID = 24;
const SIGLIP_INPUT_SIZE = 224;
const MODEL_OUTPUT_NAME = "last_hidden_state";
const MODEL_INPUT_NAME = "pixel_values";

type ExecutionProvider = "webgpu" | "wasm";

export function bilinearUpsample(
  src: Float32Array,
  srcH: number,
  srcW: number,
  channels: number,
  dstH: number,
  dstW: number,
): Float32Array {
  if (src.length !== srcH * srcW * channels) {
    throw new Error(`Expected ${srcH * srcW * channels} source values, got ${src.length}`);
  }
  const dst = new Float32Array(dstH * dstW * channels);
  const scaleH = srcH / dstH;
  const scaleW = srcW / dstW;

  for (let oy = 0; oy < dstH; oy++) {
    for (let ox = 0; ox < dstW; ox++) {
      const iy = Math.max(0, Math.min(srcH - 1, (oy + 0.5) * scaleH - 0.5));
      const ix = Math.max(0, Math.min(srcW - 1, (ox + 0.5) * scaleW - 0.5));
      const iy0 = Math.floor(iy);
      const iy1 = Math.min(srcH - 1, iy0 + 1);
      const ix0 = Math.floor(ix);
      const ix1 = Math.min(srcW - 1, ix0 + 1);
      const dy = iy - iy0;
      const dx = ix - ix0;
      const dstBase = (oy * dstW + ox) * channels;
      const src00 = (iy0 * srcW + ix0) * channels;
      const src01 = (iy0 * srcW + ix1) * channels;
      const src10 = (iy1 * srcW + ix0) * channels;
      const src11 = (iy1 * srcW + ix1) * channels;

      for (let channel = 0; channel < channels; channel++) {
        dst[dstBase + channel] =
          (1 - dy) * ((1 - dx) * src[src00 + channel]! + dx * src[src01 + channel]!) +
          dy * ((1 - dx) * src[src10 + channel]! + dx * src[src11 + channel]!);
      }
    }
  }
  return dst;
}

export function preprocessForSiglip(image: ImageData): Float32Array {
  if (image.width <= 0 || image.height <= 0) {
    throw new Error(`Invalid image dimensions: ${image.width}x${image.height}`);
  }
  const horizontal = resizeBilinear(image.data, image.width, image.height, SIGLIP_INPUT_SIZE, true);
  const resized = resizeBilinear(horizontal, SIGLIP_INPUT_SIZE, image.height, SIGLIP_INPUT_SIZE, false);

  const planeSize = SIGLIP_INPUT_SIZE * SIGLIP_INPUT_SIZE;
  const tensor = new Float32Array(3 * planeSize);
  for (let pixel = 0; pixel < planeSize; pixel++) {
    for (let channel = 0; channel < 3; channel++) {
      tensor[channel * planeSize + pixel] = resized[pixel * 4 + channel]! / 127.5 - 1;
    }
  }
  return tensor;
}

function resizeBilinear(
  source: Uint8ClampedArray,
  sourceWidth: number,
  sourceHeight: number,
  outputSize: number,
  horizontal: boolean,
): Uint8ClampedArray {
  const inputSize = horizontal ? sourceWidth : sourceHeight;
  const scale = inputSize / outputSize;
  const filterScale = Math.max(1, scale);
  const support = filterScale;
  const weights = Array.from({ length: outputSize }, (_, output) => {
    const center = (output + 0.5) * scale;
    const start = Math.max(0, Math.floor(center - support + 0.5));
    const end = Math.min(inputSize, Math.floor(center + support + 0.5));
    const values: number[] = [];
    let total = 0;
    for (let input = start; input < end; input++) {
      const weight = Math.max(0, 1 - Math.abs((input + 0.5 - center) / filterScale));
      values.push(weight);
      total += weight;
    }
    return { start, values: values.map((value) => value / total) };
  });

  const outputWidth = horizontal ? outputSize : sourceWidth;
  const outputHeight = horizontal ? sourceHeight : outputSize;
  const output = new Uint8ClampedArray(outputWidth * outputHeight * 4);
  for (let y = 0; y < outputHeight; y++) {
    for (let x = 0; x < outputWidth; x++) {
      const outputCoordinate = horizontal ? x : y;
      const { start, values } = weights[outputCoordinate]!;
      for (let channel = 0; channel < 4; channel++) {
        let value = 0;
        for (let offset = 0; offset < values.length; offset++) {
          const inputCoordinate = start + offset;
          const sourcePixel = horizontal
            ? (y * sourceWidth + inputCoordinate) * 4 + channel
            : (inputCoordinate * sourceWidth + x) * 4 + channel;
          value += source[sourcePixel]! * values[offset]!;
        }
        const outputPixel = (y * outputWidth + x) * 4 + channel;
        output[outputPixel] = Math.floor(value + 0.5);
      }
    }
  }
  return output;
}

export class OnnxVisionEncoder {
  private session: InferenceSession | null = null;
  private adapterSession: InferenceSession | null = null;
  private readonly adapterModelUrl: string;
  private readonly wasmModelUrl: string;

  constructor(
    private readonly modelUrl: string,
    private readonly executionProviders: ExecutionProvider[] = ["webgpu", "wasm"],
  ) {
    this.adapterModelUrl = modelUrl.replace(
      /siglip-base-patch16-224-vision-(?:fp16|fp32)\.onnx(?=[?#]|$)/,
      "siglip-base-patch16-224-adapter.onnx",
    );
    this.wasmModelUrl = modelUrl.replace(
      /siglip-base-patch16-224-vision-fp16\.onnx(?=[?#]|$)/,
      "siglip-base-patch16-224-vision-fp32.onnx",
    );
  }

  async init(): Promise<void> {
    if (!this.modelUrl) {
      throw new Error(
        "ONNX_MODEL_MISSING: onnxModelUrl is not configured. " +
        "Run scripts/download-siglip-onnx.py and configure the model URL.",
      );
    }
    if (this.session) return;

    const ort = await import("onnxruntime-web/webgpu");
    ort.env.wasm.wasmPaths = { wasm: wasmUrl };
    const failures: string[] = [];

    for (const provider of this.executionProviders) {
      const modelUrl = provider === "wasm" ? this.wasmModelUrl : this.modelUrl;
      let candidateSession: InferenceSession | null = null;
      let candidateAdapter: InferenceSession | null = null;
      try {
        candidateSession = await ort.InferenceSession.create(modelUrl, {
          executionProviders: [provider],
        });
        candidateAdapter = await ort.InferenceSession.create(this.adapterModelUrl, {
          executionProviders: [provider],
        });
        if (!candidateSession.inputNames.includes(MODEL_INPUT_NAME) ||
            !candidateSession.outputNames.includes(MODEL_OUTPUT_NAME) ||
            !candidateAdapter.inputNames.includes("patch_embeddings") ||
            !candidateAdapter.outputNames.includes("adapted_embeddings")) {
          throw new Error("Model must expose the SigLIP vision and fixed patch-adapter inputs and outputs");
        }
        this.session = candidateSession;
        this.adapterSession = candidateAdapter;
        buildProjectionMatrix();
        return;
      } catch (error) {
        if (candidateAdapter) await candidateAdapter.release();
        if (candidateSession) await candidateSession.release();
        const detail = error instanceof Error ? error.message : String(error);
        failures.push(`${provider} (${modelUrl}): ${detail}`);
      }
    }

    throw new Error(
      `ONNX_MODEL_LOAD_FAILED: unable to load a supported model for "${this.modelUrl}": ${failures.join("; ")}`,
    );
  }

  async encodePatches(image: ImageData): Promise<Float32Array> {
    if (!this.session) throw new Error("OnnxVisionEncoder is not initialized; call init() first");
    const ort = await import("onnxruntime-web/webgpu");
    const input = new ort.Tensor(
      "float32",
      preprocessForSiglip(image),
      [1, 3, SIGLIP_INPUT_SIZE, SIGLIP_INPUT_SIZE],
    );
    const feeds: Record<string, OrtTensor> = { [MODEL_INPUT_NAME]: input };
    const outputs = await this.session.run(feeds);
    const output = outputs[MODEL_OUTPUT_NAME];
    if (!output || output.dims.join(",") !== "1,196,768" || !(output.data instanceof Float32Array)) {
      throw new Error(
        `Invalid SigLIP output; expected float32 [1,196,768], got ${
          output ? `${output.data.constructor.name} [${output.dims.join(",")}]` : "no output"
        }`,
      );
    }
    return output.data;
  }

  async adaptPatches(patchEmbeddings: Float32Array): Promise<Float32Array> {
    if (!this.adapterSession) throw new Error("SigLIP adapter model is not initialized");
    const ort = await import("onnxruntime-web/webgpu");
    const adapterInput = new ort.Tensor("float32", patchEmbeddings, [1, SIGLIP_PATCHES, SIGLIP_DIM]);
    const outputs = await this.adapterSession.run({ patch_embeddings: adapterInput });
    const output = outputs.adapted_embeddings;
    if (!output || output.dims.join(",") !== "1,576,1152" || !(output.data instanceof Float32Array)) {
      throw new Error(
        `Invalid adapter output; expected float32 [1,576,1152], got ${
          output ? `${output.data.constructor.name} [${output.dims.join(",")}]` : "no output"
        }`,
      );
    }
    return output.data;
  }

  async encode(image: ImageData): Promise<Float32Array> {
    return this.adaptPatches(await this.encodePatches(image));
  }

  getStatus(): { mode: "onnx"; error: string | null } {
    return { mode: "onnx", error: null };
  }

  destroy(): void {
    if (this.session) void this.session.release();
    if (this.adapterSession) void this.adapterSession.release();
    this.session = null;
    this.adapterSession = null;
  }
}

export { buildProjectionMatrix, projectEmbeddings };
export { SIGLIP_DIM, TARGET_DIM };
