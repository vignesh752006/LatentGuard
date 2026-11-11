import { describe, expect, it } from "vitest";
import { unzlibSync } from "fflate";

import { TENSOR_CONTRACT } from "../src/protocol/messages";
import {
  assertOutboundPayloadSafe,
  buildLatentFramePayload,
  decodeBase64ToUint8,
  encodeCompactTensorPayload,
  encodeSourcePcaTensorPayload,
  encodeTensorPayload,
} from "../src/protocol/serializer";

function makeSeededMatrix(size: number): Float32Array {
  const matrix = new Float32Array(size);
  let state = 0xdeadbeef;
  for (let i = 0; i < size; i++) {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    matrix[i] = (state / 0x100000000) * 2 - 1;
  }
  return matrix;
}

describe("tensor serialization", () => {
  it("encodes float32 matrix to fp16 base64", () => {
    const size = TENSOR_CONTRACT.patchCount * TENSOR_CONTRACT.embeddingDim;
    const matrix = new Float32Array(size);
    matrix[0] = 1.0;
    matrix[100] = 3.0;
    const payload = encodeTensorPayload(matrix);
    expect(payload.shape).toEqual([576, 1152]);
    expect(payload.dtype).toBe("float16");
    expect(payload.data.length).toBeGreaterThan(0);
  });

  it("builds latent frame without forbidden fields", () => {
    const size = TENSOR_CONTRACT.patchCount * TENSOR_CONTRACT.embeddingDim;
    const matrix = new Float32Array(size);
    const frame = buildLatentFramePayload({
      frameId: "f-00001",
      stateVersion: 2,
      sanitizedMatrix: matrix,
      sensitivePatchIndices: [10, 20],
      viewportHash: "sha256:abc",
      lastActionId: null,
    });
    expect(() => assertOutboundPayloadSafe(frame)).not.toThrow();
    expect(frame.spatial_metadata.sensitive_patch_indices).toEqual([10, 20]);
  });

  it("rejects forbidden outbound fields", () => {
    expect(() => assertOutboundPayloadSafe({ screenshot: "x" })).toThrow();
  });

  // ---- Task 1: zlib compression fix ----

  it("compresses with zlib (not gzip) when compress=true", () => {
    const size = TENSOR_CONTRACT.patchCount * TENSOR_CONTRACT.embeddingDim;
    const matrix = new Float32Array(size);
    const payload = encodeTensorPayload(matrix, { compress: true });
    expect(payload.compression).toBe("zlib");

    // zlib stream starts with 0x78 (CMF byte); gzip starts with 0x1f 0x8b
    const bytes = decodeBase64ToUint8(payload.data);
    expect(bytes[0]).toBe(0x78);  // zlib CMF
    expect(bytes[0]).not.toBe(0x1f); // NOT gzip magic
  });

  it("compressed payload is smaller than uncompressed fp16", () => {
    const size = TENSOR_CONTRACT.patchCount * TENSOR_CONTRACT.embeddingDim;
    const matrix = new Float32Array(size);
    // Fill with some pattern to aid compression
    for (let i = 0; i < size; i++) matrix[i] = (i % 10) * 0.1;
    const uncompressed = encodeTensorPayload(matrix);
    const compressed = encodeTensorPayload(matrix, { compress: true });
    const rawUncompressed = decodeBase64ToUint8(uncompressed.data).byteLength;
    const rawCompressed = decodeBase64ToUint8(compressed.data).byteLength;
    expect(rawCompressed).toBeLessThan(rawUncompressed);
  });

  // ---- Task 1: compact int8_quantized encoder ----

  it("encodeCompactTensorPayload returns dtype int8_quantized + zlib", () => {
    const size = TENSOR_CONTRACT.patchCount * TENSOR_CONTRACT.embeddingDim;
    const matrix = new Float32Array(size);
    for (let i = 0; i < size; i++) matrix[i] = Math.sin(i * 0.01);

    const payload = encodeCompactTensorPayload(matrix);
    expect(payload.dtype).toBe("int8_quantized");
    expect(payload.compression).toBe("zlib");
    expect(payload.shape).toEqual([576, 1152]);
    expect(payload.encoding).toBe("base64");
    expect(payload.data.length).toBeGreaterThan(0);
  });

  it("compact payload wire size is smaller than fp16 uncompressed (1,327,104 bytes)", () => {
    const size = TENSOR_CONTRACT.patchCount * TENSOR_CONTRACT.embeddingDim;
    const matrix = makeSeededMatrix(size);

    const payload = encodeCompactTensorPayload(matrix);
    const wireBytes = decodeBase64ToUint8(payload.data).byteLength;

    // Must be smaller than uncompressed fp16
    expect(wireBytes).toBeLessThan(1_327_104);
    console.info(`[Task 1] seeded int8+zlib wire bytes: ${wireBytes.toLocaleString()} ` +
      `(${(wireBytes / 1_327_104 * 100).toFixed(1)}% of fp16)`);
  });

  it("compact payload with topKPercent sparsity is smaller than without", () => {
    const size = TENSOR_CONTRACT.patchCount * TENSOR_CONTRACT.embeddingDim;
    const matrix = makeSeededMatrix(size);

    const densePayload = encodeCompactTensorPayload(matrix, { topKPercent: 0 });
    const sparsePayload = encodeCompactTensorPayload(matrix, { topKPercent: 0.1 });

    const denseBytes = decodeBase64ToUint8(densePayload.data).byteLength;
    const sparseBytes = decodeBase64ToUint8(sparsePayload.data).byteLength;
    // With 90% zeros, compression should do much better
    expect(sparseBytes).toBeLessThan(denseBytes);
    console.info(`[Task 1] seeded top-10% int8+zlib wire bytes: ${sparseBytes.toLocaleString()}`);
  });

  it("compact payload rejects wrong tensor size", () => {
    const wrongMatrix = new Float32Array(100);
    expect(() => encodeCompactTensorPayload(wrongMatrix)).toThrow(/Tensor size mismatch/);
  });

  it("encodes source PCA coefficients as a compact 196xN zlib payload", () => {
    const components = 64;
    const coefficients = new Float32Array(196 * components);
    for (let i = 0; i < coefficients.length; i++) coefficients[i] = Math.sin(i * 0.03);
    const payload = encodeSourcePcaTensorPayload(coefficients, components);
    const raw = unzlibSync(decodeBase64ToUint8(payload.data));

    expect(payload.transform).toBe("source_pca_v1");
    expect(payload.shape).toEqual([196, components]);
    expect(payload.dtype).toBe("int8_quantized");
    expect(raw.byteLength).toBe(196 * 4 + 196 * components);
  });

  it("rejects non-finite source PCA coefficients", () => {
    const coefficients = new Float32Array(196 * 64);
    coefficients[123] = Number.NaN;
    expect(() => encodeSourcePcaTensorPayload(coefficients, 64)).toThrow(
      /non-finite value at index 123/,
    );
  });

  // Cross-language fp16 round-trip is tested in serializer-backend-roundtrip.test.ts.
});
