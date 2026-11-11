import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import {
  bilinearUpsample,
  buildProjectionMatrix,
  projectEmbeddings,
  SIGLIP_DIM,
  SIGLIP_GRID,
  SIGLIP_PATCHES,
  TARGET_DIM,
  TARGET_GRID,
  TARGET_PATCHES,
} from "../src/vision/onnx-encoder";
import { TENSOR_CONTRACT } from "../src/protocol/messages";

describe("SigLIP projection adapter", () => {
  it("builds the 768x1152 projection with unit-norm columns", () => {
    const matrix = buildProjectionMatrix();
    expect(matrix).toBeInstanceOf(Float32Array);
    expect(matrix.length).toBe(SIGLIP_DIM * TARGET_DIM);
    for (let column = 0; column < 5; column++) {
      let squaredNorm = 0;
      for (let row = 0; row < SIGLIP_DIM; row++) {
        const value = matrix[row * TARGET_DIM + column]!;
        squaredNorm += value * value;
      }
      expect(Math.sqrt(squaredNorm)).toBeCloseTo(1, 5);
    }
  });

  it("matches the Python projection-matrix fixture byte for byte", () => {
    const matrix = buildProjectionMatrix();
    const hash = createHash("sha256")
      .update(Buffer.from(matrix.buffer, matrix.byteOffset, matrix.byteLength))
      .digest("hex");
    const expected = readFileSync(
      path.resolve(process.cwd(), "..", "backend", "eval", "siglip", "fixtures", "projection-matrix.sha256"),
      "ascii",
    ).trim();
    expect(hash).toBe(expected);
  });

  it("upsamples 14x14x768 to 24x24x768 and preserves a constant", () => {
    const source = new Float32Array(SIGLIP_GRID * SIGLIP_GRID * SIGLIP_DIM).fill(42);
    const target = bilinearUpsample(source, SIGLIP_GRID, SIGLIP_GRID, SIGLIP_DIM, TARGET_GRID, TARGET_GRID);
    expect(target.length).toBe(TARGET_PATCHES * SIGLIP_DIM);
    for (const value of target) expect(value).toBeCloseTo(42, 5);
  });

  it("clamps bilinear edge coordinates instead of extrapolating", () => {
    const source = Float32Array.of(1, 2, 3, 4);
    const target = bilinearUpsample(source, 2, 2, 1, 4, 4);
    expect(target[0]).toBe(1);
    expect(target[3]).toBe(2);
    expect(target[12]).toBe(3);
    expect(target[15]).toBe(4);
  });

  it("adapts a fixed [196,768] tensor to [576,1152] deterministically", () => {
    const source = Float32Array.from(
      { length: SIGLIP_PATCHES * SIGLIP_DIM },
      (_, index) => Math.sin(index * 0.01),
    );
    const upsampled = bilinearUpsample(source, SIGLIP_GRID, SIGLIP_GRID, SIGLIP_DIM, TARGET_GRID, TARGET_GRID);
    const first = projectEmbeddings(upsampled, TARGET_PATCHES);
    const second = projectEmbeddings(upsampled, TARGET_PATCHES);
    expect(first.length).toBe(TARGET_PATCHES * TARGET_DIM);
    expect(first.length).toBe(TENSOR_CONTRACT.patchCount * TENSOR_CONTRACT.embeddingDim);
    expect(first).toEqual(second);
  });
});

describe("OnnxVisionEncoder model loading", () => {
  it("throws a clear error when no model URL is configured", async () => {
    const { OnnxVisionEncoder } = await import("../src/vision/onnx-encoder");
    await expect(new OnnxVisionEncoder("").init()).rejects.toThrow("ONNX_MODEL_MISSING");
  });

  it("throws a clear load error when the model file is missing", async () => {
    const { OnnxVisionEncoder } = await import("../src/vision/onnx-encoder");
    await expect(
      new OnnxVisionEncoder("http://127.0.0.1:9/nonexistent.onnx", ["wasm"]).init(),
    ).rejects.toThrow("ONNX_MODEL_LOAD_FAILED");
  });
});
