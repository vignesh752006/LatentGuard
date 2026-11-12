import { describe, expect, it } from "vitest";
import {
  computeTextImageScores,
  SiglipTextEncoder,
} from "../src/vision/text-similarity";
import { TENSOR_CONTRACT } from "../src/protocol/messages";
import { SIGLIP_DIM } from "../src/vision/projection";

const { patchCount } = TENSOR_CONTRACT;
const embeddingDim = SIGLIP_DIM;

describe("computeTextImageScores", () => {
  it("returns one normalized cosine score per sanitized patch", () => {
    const embeddings = new Float32Array(patchCount * embeddingDim);
    const text = new Float32Array(embeddingDim);
    embeddings[0] = 4;
    text[0] = 2;

    const scores = computeTextImageScores(embeddings, text);

    expect(scores).toHaveLength(patchCount);
    expect(scores[0]).toBeCloseTo(1);
    expect(scores[1]).toBe(0);
    expect(scores.every((score) => score >= -1 && score <= 1)).toBe(true);
  });

  it("rejects embeddings with incorrect shapes", () => {
    expect(() =>
      computeTextImageScores(new Float32Array(10), new Float32Array(embeddingDim)),
    ).toThrow(/SigLIP image values/);
    expect(() =>
      computeTextImageScores(
        new Float32Array(patchCount * embeddingDim),
        new Float32Array(10),
      ),
    ).toThrow(/text embedding length/);
  });

  it("rejects non-finite sanitized or text values", () => {
    const embeddings = new Float32Array(patchCount * embeddingDim);
    const text = new Float32Array(embeddingDim);
    embeddings[3] = Number.NaN;
    expect(() => computeTextImageScores(embeddings, text)).toThrow(/non-finite/);
    embeddings[3] = 0;
    text[7] = Number.POSITIVE_INFINITY;
    expect(() => computeTextImageScores(embeddings, text)).toThrow(/non-finite/);
  });
});

describe("SiglipTextEncoder", () => {
  it("fails clearly when no model URL is configured", async () => {
    await expect(new SiglipTextEncoder("").init()).rejects.toThrow(
      /ONNX_TEXT_MODEL_MISSING/,
    );
  });
});
