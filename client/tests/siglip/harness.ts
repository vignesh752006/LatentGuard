import { OnnxVisionEncoder, SIGLIP_DIM } from "../../src/vision/onnx-encoder";
import {
  buildLatentFramePayload,
  encodeCompactTensorPayload,
  encodeTensorPayload,
  decodeBase64ToUint8,
} from "../../src/protocol/serializer";
import type { TensorPayload } from "../../src/protocol/messages";
import { serializeLatentFrame } from "../../src/protocol/websocket-client";
import { sanitizeSVD } from "../../src/vdlm/svd-sanitizer";
import { SiglipTextEncoder } from "../../src/vision/text-similarity";
import {
  loadSourcePcaBasis,
  projectSourcePca,
  reconstructSourcePca,
  SOURCE_PCA_DEFAULT_COMPONENTS,
} from "../../src/vision/source-pca";
import { unzlibSync } from "fflate";

type Provider = "webgpu" | "wasm";
type EncoderResult = {
  cosine: number;
  outputShape: number[];
  latencyMs: { p50: number; p95: number; count: number } | null;
  wireBytes: {
    fp16Zlib: { compressedBytes: number; base64Bytes: number };
    int8Zlib: { compressedBytes: number; base64Bytes: number };
    int8Top10Zlib: { compressedBytes: number; base64Bytes: number };
  };
};
const encoders = new Map<Provider, OnnxVisionEncoder>();
let textEncoder: SiglipTextEncoder | null = null;

function encodedSize(payload: TensorPayload): { compressedBytes: number; base64Bytes: number } {
  return { compressedBytes: atob(payload.data).length, base64Bytes: payload.data.length };
}

declare global {
  interface Window {
    encodeSiglip: (
      provider: Provider,
      imageUrl: string,
      reference: number[],
      measureLatency: boolean,
    ) => Promise<EncoderResult>;
    verifySyntheticDeterminism: (modelUrl: string) => Promise<boolean>;
    closeSiglipEncoders: () => void;
    sanitizeSiglipEmbedding: (imageUrl: string, sensitiveIndices: number[]) => Promise<{
      outputShape: number[];
      privateRank: number;
      maxAbsDot: number;
    }>;
    measureSourcePca: (imageUrl: string) => Promise<{
      compressedBytes: number;
      base64Bytes: number;
      frameJsonBytes: number;
      meanPatchCosineVsSanitized: number;
      minimumPatchCosineVsSanitized: number;
      meanPatchCosineVsFull: number;
    }>;
    scoreSiglipSourceIntent: (sourceEmbeddings: number[], taskIntent: string) => Promise<{
      shape: number[];
      minimum: number;
      maximum: number;
      finite: boolean;
    }>;
  }
}

window.encodeSiglip = async (provider, imageUrl, reference, measureLatency) => {
  const image = new Image();
  image.src = imageUrl;
  await image.decode();
  const canvas = new OffscreenCanvas(image.width, image.height);
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Could not create fixture image canvas");
  context.drawImage(image, 0, 0);
  const imageData = context.getImageData(0, 0, image.width, image.height);
  let encoder = encoders.get(provider);
  if (!encoder) {
    encoder = new OnnxVisionEncoder("/models/siglip-base-patch16-224-vision-fp16.onnx", [provider]);
    await encoder.init();
    encoders.set(provider, encoder);
  }
  const patches = await encoder.encodePatches(imageData);
  const adapted = await encoder.adaptPatches(patches);
  let patchCosines = 0;
  for (let patch = 0; patch < 196; patch++) {
    let patchDot = 0;
    let leftNorm = 0;
    let rightNorm = 0;
    for (let channel = 0; channel < 768; channel++) {
      const left = patches[patch * SIGLIP_DIM + channel]!;
      const right = reference[patch * 768 + channel]!;
      patchDot += left * right;
      leftNorm += left * left;
      rightNorm += right * right;
    }
    patchCosines += patchDot / Math.sqrt(leftNorm * rightNorm);
  }
  const cosine = patchCosines / 196;
  const wireBytes = {
    fp16Zlib: encodedSize(encodeTensorPayload(adapted, { compress: true })),
    int8Zlib: encodedSize(encodeCompactTensorPayload(adapted)),
    int8Top10Zlib: encodedSize(encodeCompactTensorPayload(adapted, { topKPercent: 0.1 })),
  };
  if (!measureLatency) {
    return { cosine, outputShape: [adapted.length / 1152, 1152], latencyMs: null, wireBytes };
  }
  const durations: number[] = [];
  for (let iteration = 0; iteration < 20; iteration++) {
    const start = performance.now();
    await encoder.encode(imageData);
    durations.push(performance.now() - start);
  }
  durations.sort((a, b) => a - b);
  return {
    cosine,
    outputShape: [adapted.length / 1152, 1152],
    latencyMs: { p50: durations[9]!, p95: durations[18]!, count: durations.length },
    wireBytes,
  };
};

window.verifySyntheticDeterminism = async (modelUrl) => {
  const provider: Provider = "wasm";
  let encoder = encoders.get(provider);
  if (!encoder) {
    encoder = new OnnxVisionEncoder(modelUrl, [provider]);
    await encoder.init();
    encoders.set(provider, encoder);
  }
  const pixels = new Uint8ClampedArray(224 * 224 * 4);
  for (let index = 0; index < 224 * 224; index++) {
    pixels[index * 4] = index % 251;
    pixels[index * 4 + 1] = (index * 7) % 253;
    pixels[index * 4 + 2] = (index * 13) % 255;
    pixels[index * 4 + 3] = 255;
  }
  const image = new ImageData(pixels, 224, 224);
  const first = await encoder.encode(image);
  const second = await encoder.encode(image);
  return first.length === 576 * 1152 && first.every((value, index) => value === second[index]);
};

window.closeSiglipEncoders = () => {
  for (const encoder of encoders.values()) encoder.destroy();
  encoders.clear();
  textEncoder?.destroy();
  textEncoder = null;
};

window.sanitizeSiglipEmbedding = async (imageUrl, sensitiveIndices) => {
  const encoder = await getWasmEncoder();
  const embedding = await encoder.encode(await imageDataFromUrl(imageUrl));
  const { sanitized, privateBasis } = sanitizeSVD(embedding, sensitiveIndices, {
    k: 4,
    alpha: 0,
  });
  let maxAbsDot = 0;
  for (let patch = 0; patch < 576; patch++) {
    for (let rank = 0; rank < privateBasis.length / 1152; rank++) {
      let dot = 0;
      for (let dim = 0; dim < 1152; dim++) {
        dot += sanitized[patch * 1152 + dim]! * privateBasis[rank * 1152 + dim]!;
      }
      maxAbsDot = Math.max(maxAbsDot, Math.abs(dot));
    }
  }
  return { outputShape: [576, 1152], privateRank: privateBasis.length / 1152, maxAbsDot };
};

window.measureSourcePca = async (imageUrl) => {
  const encoder = await getWasmEncoder();
  const source = await encoder.encodePatches(await imageDataFromUrl(imageUrl));
  const original = await encoder.adaptPatches(source);
  const sensitiveSourceIndices = [1, 2, 24, 25, 26, 48, 49, 50].map((index) => {
    const row = Math.floor(index / 24);
    const col = index % 24;
    return Math.min(13, Math.floor((row + 0.5) * 14 / 24)) * 14 +
      Math.min(13, Math.floor((col + 0.5) * 14 / 24));
  });
  const sanitizedSource = sanitizeSVD(source, [...new Set(sensitiveSourceIndices)], {
    k: 4,
    alpha: 0,
    patchCount: 196,
    embeddingDim: 768,
  }).sanitized;
  const fullSanitized = await encoder.adaptPatches(sanitizedSource);
  const basis = await loadSourcePcaBasis();
  const coefficients = projectSourcePca(
    sanitizedSource,
    basis,
    SOURCE_PCA_DEFAULT_COMPONENTS,
  );
  const frame = buildLatentFramePayload({
    frameId: "pca-measurement",
    stateVersion: 1,
    sourcePcaCoefficients: coefficients,
    pcaComponentCount: SOURCE_PCA_DEFAULT_COMPONENTS,
    sensitivePatchIndices: [],
    viewportHash: "measurement",
    lastActionId: null,
    sanitizationMethod: "svd_nullspace_k4",
  });
  const payload = frame.tensor;
  const raw = unzlibSync(decodeBase64ToUint8(payload.data));
  const scales = new Float32Array(raw.buffer, raw.byteOffset, 196);
  const quantized = new Int8Array(raw.buffer, raw.byteOffset + 196 * 4);
  const dequantized = new Float32Array(coefficients.length);
  for (let patch = 0; patch < 196; patch++) {
    for (let component = 0; component < SOURCE_PCA_DEFAULT_COMPONENTS; component++) {
      const offset = patch * SOURCE_PCA_DEFAULT_COMPONENTS + component;
      dequantized[offset] = quantized[offset]! * scales[patch]! / 127;
    }
  }
  const reconstructedSource = reconstructSourcePca(
    dequantized,
    basis,
    SOURCE_PCA_DEFAULT_COMPONENTS,
  );
  const reconstructed = await encoder.adaptPatches(reconstructedSource);
  let cosineSum = 0;
  let cosineMin = 1;
  let fullCosineSum = 0;
  for (let patch = 0; patch < 576; patch++) {
    let dot = 0;
    let normA = 0;
    let normB = 0;
    let fullDot = 0;
    let fullNormA = 0;
    let fullNormB = 0;
    for (let dim = 0; dim < 1152; dim++) {
      const a = fullSanitized[patch * 1152 + dim]!;
      const full = original[patch * 1152 + dim]!;
      const b = reconstructed[patch * 1152 + dim]!;
      dot += a * b;
      normA += a * a;
      normB += b * b;
      fullDot += full * b;
      fullNormA += full * full;
      fullNormB += b * b;
    }
    const cosine = dot / (Math.sqrt(normA * normB) + 1e-12);
    cosineSum += cosine;
    cosineMin = Math.min(cosineMin, cosine);
    fullCosineSum += fullDot / (Math.sqrt(fullNormA * fullNormB) + 1e-12);
  }
  return {
    compressedBytes: decodeBase64ToUint8(payload.data).byteLength,
    base64Bytes: payload.data.length,
    frameJsonBytes: new TextEncoder().encode(
      serializeLatentFrame(frame, "pca-measurement-session"),
    ).byteLength,
    meanPatchCosineVsSanitized: cosineSum / 576,
    minimumPatchCosineVsSanitized: cosineMin,
    meanPatchCosineVsFull: fullCosineSum / 576,
  };
};

window.scoreSiglipSourceIntent = async (sourceValues, taskIntent) => {
  const source = Float32Array.from(sourceValues);
  if (source.length !== 196 * 768) {
    throw new Error(`Expected 196x768 SigLIP source tokens, got ${source.length}`);
  }
  const sanitized = sanitizeSVD(
    source,
    [1, 2, 24, 25, 26, 48, 49, 50],
    { k: 4, alpha: 0, patchCount: 196, embeddingDim: 768 },
  ).sanitized;
  source.fill(0);
  textEncoder ??= new SiglipTextEncoder(
    "/models/siglip-base-patch16-224-text-fp32.onnx",
    ["wasm"],
  );
  await textEncoder.init();
  try {
    const scores = await textEncoder.scoreGrid(sanitized, taskIntent);
    return {
      shape: [24, 24],
      minimum: Math.min(...scores),
      maximum: Math.max(...scores),
      finite: scores.every(Number.isFinite),
    };
  } finally {
    sanitized.fill(0);
  }
};

async function getWasmEncoder(): Promise<OnnxVisionEncoder> {
  let encoder = encoders.get("wasm");
  if (!encoder) {
    encoder = new OnnxVisionEncoder(
      "/models/siglip-base-patch16-224-vision-fp16.onnx",
      ["wasm"],
    );
    await encoder.init();
    encoders.set("wasm", encoder);
  }
  return encoder;
}

async function imageDataFromUrl(imageUrl: string): Promise<ImageData> {
  const image = new Image();
  image.src = imageUrl;
  await image.decode();
  const canvas = new OffscreenCanvas(image.width, image.height);
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Could not create fixture image canvas");
  context.drawImage(image, 0, 0);
  return context.getImageData(0, 0, image.width, image.height);
}
