/**
 * VDLM Multi-rank SVD null-space sanitizer (Task 4)
 *
 * Implements the blueprint math:
 *   W_sens  = rows of V at I_sens indices           [|I_sens|, D]
 *   W_priv  = top-k singular vectors from SVD of W_sens  [k, D]
 *   P_null  = I_D - W_priv^T W_priv               [D, D]  (implicit)
 *   V'[i,*] = V[i,*] @ P_null + alpha * N_orth[i,*]
 *
 * The SVD is computed on CPU (JS); the projection is applied row-by-row in JS
 * (GPU path would need a compute shader — kept as a future optimization).
 *
 * Edge cases:
 *   - |I_sens| = 0: return V unchanged (identity)
 *   - |I_sens| < k: use k' = |I_sens| singular vectors
 *   - NaN in V: replace with 0 before projection
 *
 * See docs/ASSUMPTIONS.md for measured MSE / cosine-similarity results.
 */

import { TENSOR_CONTRACT } from "../protocol/messages";

// ---- Thin SVD (power iteration, CPU) ----------------------------------------

/**
 * Compute the top-k left singular vectors of matrix A ([m, n]) via power iteration.
 * Returns U: [k, n] (each row is a right singular vector of A).
 * This is fast for small m (|I_sens| ≤ 576) and moderate n (1152).
 */
function topKRightSingularVectors(A: Float32Array, m: number, n: number, k: number): Float32Array {
  // We work with A^T A (n×n) to find right singular vectors via power iteration.
  // For stability with deflation, iteratively extract top singular vectors.
  const result = new Float32Array(k * n);  // [k, n]

  // Work on a copy (deflation modifies it)
  const Acopy = A.slice();

  for (let ki = 0; ki < k; ki++) {
    // Power iteration: v <- A^T A v / ||A^T A v||
    let v = new Float32Array(n);
    // Initialize with a deterministic non-zero vector
    for (let j = 0; j < n; j++) v[j] = Math.sin((j + ki * 37) * 0.1);
    normalizeInPlace(v, n);

    for (let iter = 0; iter < 50; iter++) {
      // u = A v  [m]
      const u = matvecMul(Acopy, m, n, v);
      normalizeInPlace(u, m);
      // v_new = A^T u  [n]
      const vNew = matvecMulT(Acopy, m, n, u);
      normalizeInPlace(vNew, n);
      v = vNew;
    }

    // Store this singular vector
    result.set(v, ki * n);

    // Deflation: A <- A - sigma * u * v^T  (removes this component from Acopy)
    // u = A v
    const u = matvecMul(Acopy, m, n, v);
    const sigma = norm(u, m);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        Acopy[i * n + j] -= (u[i]! / (sigma + 1e-10)) * v[j]! * sigma;
      }
    }
  }

  return result;
}

function matvecMul(A: Float32Array, m: number, n: number, v: Float32Array): Float32Array {
  const u = new Float32Array(m);
  for (let i = 0; i < m; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += A[i * n + j]! * v[j]!;
    u[i] = s;
  }
  return u;
}

function matvecMulT(A: Float32Array, m: number, n: number, u: Float32Array): Float32Array {
  const v = new Float32Array(n);
  for (let j = 0; j < n; j++) {
    let s = 0;
    for (let i = 0; i < m; i++) s += A[i * n + j]! * u[i]!;
    v[j] = s;
  }
  return v;
}

function norm(v: Float32Array, len: number): number {
  let s = 0;
  for (let i = 0; i < len; i++) s += v[i]! * v[i]!;
  return Math.sqrt(s);
}

function normalizeInPlace(v: Float32Array, len: number): void {
  const n = norm(v, len) + 1e-10;
  for (let i = 0; i < len; i++) v[i]! /= n;
}

// ---- Orthogonal noise (deterministic, independent of private basis) ----------

/**
 * Deterministic pseudo-random noise orthogonal to the private basis W_priv.
 * Uses a fixed hash based on patch index and basis vectors for reproducibility.
 * The noise is projected out of the span of W_priv so it doesn't re-introduce
 * the private direction.
 */
function orthoNoise(
  patch: number,
  dim: number,
  W_priv: Float32Array,
  k: number,
  alpha: number,
): Float32Array {
  if (alpha === 0) return new Float32Array(dim);

  // Generate pseudo-random noise vector from patch index
  const noise = new Float32Array(dim);
  for (let d = 0; d < dim; d++) {
    // Simple hash: sin/cos combination deterministic per (patch, d)
    noise[d] = Math.sin(patch * 97 + d * 13) * Math.cos(d * 7) * alpha;
  }

  // Project out the W_priv directions so noise is truly orthogonal
  for (let ki = 0; ki < k; ki++) {
    let dot = 0;
    for (let d = 0; d < dim; d++) dot += noise[d]! * W_priv[ki * dim + d]!;
    for (let d = 0; d < dim; d++) noise[d]! -= dot * W_priv[ki * dim + d]!;
  }

  return noise;
}

// ---- Diagnostic metrics -----------------------------------------------------

export interface SanitizationMetrics {
  /** MSE between V and V' on sensitive rows */
  mse_sens: number;
  /** MSE between V and V' on non-sensitive rows */
  mse_non_sens: number;
  /** Mean cosine similarity V·V' on sensitive rows */
  cos_sim_sens: number;
  /** Mean cosine similarity V·V' on non-sensitive rows */
  cos_sim_non_sens: number;
  /** k used (may differ from requested if |I_sens| < k) */
  k_used: number;
  /** Number of sensitive patches */
  n_sens: number;
}

function cosineSim(a: Float32Array, b: Float32Array, offset: number, dim: number): number {
  let dot = 0, na = 0, nb = 0;
  for (let d = 0; d < dim; d++) {
    dot += a[offset + d]! * b[offset + d]!;
    na  += a[offset + d]! * a[offset + d]!;
    nb  += b[offset + d]! * b[offset + d]!;
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}

function rowMse(a: Float32Array, b: Float32Array, offset: number, dim: number): number {
  let s = 0;
  for (let d = 0; d < dim; d++) {
    const diff = a[offset + d]! - b[offset + d]!;
    s += diff * diff;
  }
  return s / dim;
}

// ---- Main sanitizer ----------------------------------------------------------

export interface SanitizerResult {
  sanitized: Float32Array;
  metrics: SanitizationMetrics;
}

export interface MultiRankSanitizerOptions {
  /** Number of singular vectors to remove (k=1..8). Default: 1. */
  k?: number;
  /** Orthogonal noise magnitude. Default: 0.01. */
  alpha?: number;
  /** If true, compute and return diagnostic metrics. Default: true. */
  computeMetrics?: boolean;
}

/**
 * Apply the SVD multi-rank null-space projection to sanitize embeddings.
 *
 * @param embeddings  Float32Array of shape [patchCount, embeddingDim]
 * @param sensitivePatchIndices  Indices of sensitive patches (I_sens)
 * @param opts  Options
 * @returns  { sanitized, metrics }
 */
export function sanitizeSVD(
  embeddings: Float32Array,
  sensitivePatchIndices: number[],
  opts: MultiRankSanitizerOptions = {},
): SanitizerResult {
  const { patchCount, embeddingDim } = TENSOR_CONTRACT;
  const k = Math.max(1, Math.min(8, opts.k ?? 1));
  const alpha = opts.alpha ?? 0.01;
  const computeMetrics = opts.computeMetrics ?? true;

  // Validate / sanitize inputs
  const nSens = sensitivePatchIndices.length;
  const validIndices = sensitivePatchIndices.filter(i => i >= 0 && i < patchCount);

  // Fix NaNs in input
  const V = new Float32Array(embeddings.length);
  for (let i = 0; i < embeddings.length; i++) {
    V[i] = Number.isFinite(embeddings[i]!) ? embeddings[i]! : 0;
  }

  // Edge case: no sensitive patches → identity transform
  if (validIndices.length === 0) {
    const metrics: SanitizationMetrics = {
      mse_sens: 0, mse_non_sens: 0,
      cos_sim_sens: 1, cos_sim_non_sens: 1,
      k_used: 0, n_sens: 0,
    };
    return { sanitized: V.slice(), metrics };
  }

  // Build W_sens: rows of V at sensitive indices [|I_sens|, D]
  const kActual = Math.min(k, validIndices.length);
  const W_sens = new Float32Array(validIndices.length * embeddingDim);
  for (let i = 0; i < validIndices.length; i++) {
    const patchIdx = validIndices[i]!;
    W_sens.set(V.subarray(patchIdx * embeddingDim, (patchIdx + 1) * embeddingDim), i * embeddingDim);
  }

  // Compute top-k right singular vectors of W_sens → W_priv: [k, D]
  const W_priv = topKRightSingularVectors(W_sens, validIndices.length, embeddingDim, kActual);

  // Apply projection to all patches: V'[i,*] = V[i,*] - Σ_ki (V[i,*]·w_ki) w_ki + noise
  const Vprime = new Float32Array(patchCount * embeddingDim);

  for (let patch = 0; patch < patchCount; patch++) {
    const base = patch * embeddingDim;

    // Project out all k private directions
    const proj = new Float32Array(embeddingDim);  // sum of projections
    for (let ki = 0; ki < kActual; ki++) {
      let dot = 0;
      for (let d = 0; d < embeddingDim; d++) {
        dot += V[base + d]! * W_priv[ki * embeddingDim + d]!;
      }
      for (let d = 0; d < embeddingDim; d++) {
        proj[d]! += dot * W_priv[ki * embeddingDim + d]!;
      }
    }

    // Add orthogonal noise (only to sensitive patches if needed, here all)
    const noise = orthoNoise(patch, embeddingDim, W_priv, kActual, alpha);

    for (let d = 0; d < embeddingDim; d++) {
      Vprime[base + d] = V[base + d]! - proj[d]! + noise[d]!;
    }
  }

  // Compute metrics
  let metrics: SanitizationMetrics;
  if (computeMetrics) {
    const sensSet = new Set(validIndices);
    let mse_s = 0, mse_ns = 0, cos_s = 0, cos_ns = 0;
    let ns_count = 0, nns_count = 0;

    for (let patch = 0; patch < patchCount; patch++) {
      const off = patch * embeddingDim;
      if (sensSet.has(patch)) {
        mse_s += rowMse(V, Vprime, off, embeddingDim);
        cos_s += cosineSim(V, Vprime, off, embeddingDim);
        ns_count++;
      } else {
        mse_ns += rowMse(V, Vprime, off, embeddingDim);
        cos_ns += cosineSim(V, Vprime, off, embeddingDim);
        nns_count++;
      }
    }

    metrics = {
      mse_sens: ns_count > 0 ? mse_s / ns_count : 0,
      mse_non_sens: nns_count > 0 ? mse_ns / nns_count : 0,
      cos_sim_sens: ns_count > 0 ? cos_s / ns_count : 1,
      cos_sim_non_sens: nns_count > 0 ? cos_ns / nns_count : 1,
      k_used: kActual,
      n_sens: validIndices.length,
    };

    console.info(
      "[VDLM SVD] sanitization metrics",
      `k=${kActual}`, `n_sens=${validIndices.length}`,
      `mse_sens=${metrics.mse_sens.toFixed(6)}`,
      `cos_sens=${metrics.cos_sim_sens.toFixed(4)}`,
      `mse_non_sens=${metrics.mse_non_sens.toFixed(6)}`,
      `cos_non_sens=${metrics.cos_sim_non_sens.toFixed(4)}`,
    );
  } else {
    metrics = {
      mse_sens: 0, mse_non_sens: 0,
      cos_sim_sens: 0, cos_sim_non_sens: 0,
      k_used: kActual, n_sens: validIndices.length,
    };
  }

  return { sanitized: Vprime, metrics };
}

/**
 * Rank-1 baseline sanitizer (original mean-direction approach).
 * Kept for comparison / A-B benchmarking.
 */
export function sanitizeRankOne(
  embeddings: Float32Array,
  sensitivePatchIndices: number[],
  alpha = 0.01,
): Float32Array {
  return sanitizeSVD(embeddings, sensitivePatchIndices, { k: 1, alpha, computeMetrics: false }).sanitized;
}

