/**
 * Task 6 — text-similarity score grid unit tests.
 */

import { describe, expect, it } from "vitest";
import {
  buildScoreGrid,
  computeTextImageScores,
  mockTextEmbedding,
} from "../src/vision/text-similarity";
import { TENSOR_CONTRACT } from "../src/protocol/messages";

const { patchCount, embeddingDim } = TENSOR_CONTRACT;

function makeEmbeddings(seed = 0): Float32Array {
  const arr = new Float32Array(patchCount * embeddingDim);
  let x = seed + 1;
  for (let i = 0; i < arr.length; i++) {
    x = (x * 1664525 + 1013904223) >>> 0;
    arr[i] = (x / 0x100000000) * 2 - 1;
  }
  return arr;
}

describe("mockTextEmbedding", () => {
  it("returns vector of correct length", () => {
    const v = mockTextEmbedding("click login button");
    expect(v.length).toBe(embeddingDim);
  });

  it("is deterministic for same input", () => {
    const v1 = mockTextEmbedding("fill password field");
    const v2 = mockTextEmbedding("fill password field");
    expect(v1).toEqual(v2);
  });

  it("differs for different inputs", () => {
    const v1 = mockTextEmbedding("click login");
    const v2 = mockTextEmbedding("scroll down");
    expect(v1).not.toEqual(v2);
  });

  it("is approximately unit-norm", () => {
    const v = mockTextEmbedding("test intent");
    let norm = 0;
    for (const x of v) norm += x * x;
    expect(Math.sqrt(norm)).toBeCloseTo(1.0, 4);
  });
});

describe("computeTextImageScores", () => {
  it("returns Float32Array of length patchCount", () => {
    const emb = makeEmbeddings(1);
    const scores = computeTextImageScores(emb);
    expect(scores).toBeInstanceOf(Float32Array);
    expect(scores.length).toBe(patchCount);
  });

  it("all scores are in [-1, 1]", () => {
    const emb = makeEmbeddings(2);
    const scores = computeTextImageScores(emb);
    for (const s of scores) {
      expect(s).toBeGreaterThanOrEqual(-1.0 - 1e-5);
      expect(s).toBeLessThanOrEqual(1.0 + 1e-5);
    }
  });

  it("identical text and patch embedding → score = 1", () => {
    const textVec = new Float32Array(embeddingDim).fill(0);
    textVec[0] = 1; // unit vector
    // Build embedding where patch 0 matches the text vector exactly
    const emb = new Float32Array(patchCount * embeddingDim);
    emb[0] = 1; // patch 0 col 0
    const scores = computeTextImageScores(emb, { textEmbedding: textVec });
    expect(scores[0]).toBeCloseTo(1.0, 4);
  });

  it("throws on wrong textEmbedding length", () => {
    const emb = makeEmbeddings(3);
    expect(() =>
      computeTextImageScores(emb, { textEmbedding: new Float32Array(100) }),
    ).toThrow(/mismatch/);
  });
});

describe("buildScoreGrid", () => {
  it("returns Float32Array of length patchCount", () => {
    const emb = makeEmbeddings(5);
    const grid = buildScoreGrid(emb, "click the submit button");
    expect(grid).toBeInstanceOf(Float32Array);
    expect(grid.length).toBe(patchCount);
  });

  it("is deterministic for same intent", () => {
    const emb = makeEmbeddings(7);
    const g1 = buildScoreGrid(emb, "fill card number");
    const g2 = buildScoreGrid(emb, "fill card number");
    expect(g1).toEqual(g2);
  });

  it("differs for different intents", () => {
    const emb = makeEmbeddings(8);
    const g1 = buildScoreGrid(emb, "click login");
    const g2 = buildScoreGrid(emb, "scroll down page");
    // Not all values should be identical
    let diffCount = 0;
    for (let i = 0; i < patchCount; i++) {
      if (Math.abs(g1[i]! - g2[i]!) > 1e-6) diffCount++;
    }
    expect(diffCount).toBeGreaterThan(0);
  });
});

