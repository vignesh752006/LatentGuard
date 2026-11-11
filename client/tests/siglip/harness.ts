import { OnnxVisionEncoder, SIGLIP_DIM } from "../../src/vision/onnx-encoder";
import {
  encodeCompactTensorPayload,
  encodeTensorPayload,
} from "../../src/protocol/serializer";
import type { TensorPayload } from "../../src/protocol/messages";

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
};
