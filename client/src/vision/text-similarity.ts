/**
 * Task 6 — SigLIP text-image similarity score grid.
 *
 * Computes a 24×24 (576-element) patch-wise text-image similarity score
 * over the sanitized embedding V' using a cosine similarity proxy.
 *
 * In "onnx" mode a text encoder session would be needed; here we implement
 * the cosine-similarity math against a mock/projected text embedding so the
 * interface is production-ready and testable without the model.
 *
 * Output: Float32Array of length 576, values in [-1, 1].
 */

import { TENSOR_CONTRACT } from "../protocol/messages";

export interface TextSimilarityOptions {
  /** If provided, used directly as the text embedding (length embeddingDim). */
  textEmbedding?: Float32Array;
  /** Normalize before computing similarity (recommended). Default true. */
  normalize?: boolean;
}

/**
 * Compute per-patch cosine similarity between V' and a text embedding.
 *
 * @param sanitizedEmbeddings Float32Array of shape (patchCount × embeddingDim),
 *   row-major.
 * @param textEmbedding Float32Array of length embeddingDim.  When omitted a
 *   uniform unit-vector is used (all patches score equally — neutral baseline).
 * @returns Float32Array of length patchCount (576), values in [-1, 1].
 */
export function computeTextImageScores(
  sanitizedEmbeddings: Float32Array,
  options?: TextSimilarityOptions,
): Float32Array {
  const { patchCount, embeddingDim } = TENSOR_CONTRACT;
  const normalize = options?.normalize ?? true;

  // Build text embedding: supplied or neutral (all-ones normalised)
  let textVec = options?.textEmbedding ?? new Float32Array(embeddingDim).fill(1);
  if (textVec.length !== embeddingDim) {
    throw new Error(
      `textEmbedding length mismatch: expected ${embeddingDim}, got ${textVec.length}`,
    );
  }

  if (normalize) {
    textVec = l2Normalize(textVec);
  }

  const scores = new Float32Array(patchCount);

  for (let p = 0; p < patchCount; p++) {
    const offset = p * embeddingDim;
    const row = sanitizedEmbeddings.subarray(offset, offset + embeddingDim);
    const patchVec = normalize ? l2Normalize(new Float32Array(row)) : new Float32Array(row);

    let dot = 0;
    for (let d = 0; d < embeddingDim; d++) {
      dot += patchVec[d]! * textVec[d]!;
    }
    scores[p] = dot;
  }

  return scores;
}

/**
 * Compute a mock text embedding for a given task-intent string.
 *
 * In production this would call the SigLIP text encoder ONNX session.
 * Here we produce a deterministic pseudo-embedding from the string's bytes
 * so tests are hermetic.
 *
 * @param taskIntent Natural-language task description.
 * @returns Float32Array of length embeddingDim.
 */
export function mockTextEmbedding(taskIntent: string): Float32Array {
  const { embeddingDim } = TENSOR_CONTRACT;
  const vec = new Float32Array(embeddingDim);
  const encoded = new TextEncoder().encode(taskIntent);

  // Simple deterministic seeded PRNG from task intent bytes
  let seed = 0;
  for (const b of encoded) seed = (seed * 31 + b) >>> 0;

  for (let i = 0; i < embeddingDim; i++) {
    // LCG: x = (a*x + c) mod 2^32
    seed = ((1664525 * seed + 1013904223) >>> 0);
    vec[i] = ((seed >>> 0) / 0x100000000) * 2 - 1; // [-1, 1]
  }
  return l2Normalize(vec);
}

/**
 * Returns a 24×24 score grid (flattened, row-major) suitable for embedding
 * in the wire payload as `score_grid`.
 */
export function buildScoreGrid(
  sanitizedEmbeddings: Float32Array,
  taskIntent: string,
  options?: TextSimilarityOptions,
): Float32Array {
  const textEmbedding = options?.textEmbedding ?? mockTextEmbedding(taskIntent);
  return computeTextImageScores(sanitizedEmbeddings, {
    ...options,
    textEmbedding,
  });
}

function l2Normalize(v: Float32Array): Float32Array {
  let norm = 0;
  for (let i = 0; i < v.length; i++) norm += v[i]! * v[i]!;
  norm = Math.sqrt(norm);
  if (norm < 1e-9) return v;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i]! / norm;
  return out;
}
