/**
 * Task 3 — SigLIP adapter unit tests
 *
 * Tests the shape/determinism of:
 *   - buildProjectionMatrix: 768×1152, columns normalized
 *   - bilinearUpsample: 14×14×768 → 24×24×768
 *   - projectEmbeddings: [N, 768] → [N, 1152]
 *   - Full adapter pipeline shape (without actual ONNX inference)
 *
 * Note: OnnxVisionEncoder.init() in "onnx" mode requires the actual ONNX model
 * file and will throw ONNX_MODEL_MISSING in a test environment (no model present).
 * This is the correct behavior per TASK 3 spec ("fail loudly").
 */

import { describe, it, expect } from 'vitest';
import {
  buildProjectionMatrix,
  bilinearUpsample,
  projectEmbeddings,
  SIGLIP_PATCHES,
  SIGLIP_DIM,
  TARGET_PATCHES,
  TARGET_DIM,
  SIGLIP_GRID,
  TARGET_GRID,
} from '../src/vision/onnx-encoder';
import { TENSOR_CONTRACT } from '../src/protocol/messages';

describe('Task 3: SigLIP adapter — projection matrix', () => {
  it('buildProjectionMatrix returns Float32Array of shape [768, 1152]', () => {
    const P = buildProjectionMatrix();
    expect(P).toBeInstanceOf(Float32Array);
    expect(P.length).toBe(SIGLIP_DIM * TARGET_DIM);  // 768 * 1152 = 884736
  });

  it('projection matrix columns are l2-normalized (norm ≈ 1.0)', () => {
    const P = buildProjectionMatrix();
    // Check first 5 columns
    for (let j = 0; j < 5; j++) {
      let norm = 0;
      for (let i = 0; i < SIGLIP_DIM; i++) {
        const v = P[i * TARGET_DIM + j]!;
        norm += v * v;
      }
      norm = Math.sqrt(norm);
      expect(norm).toBeCloseTo(1.0, 2);  // ±0.005
    }
  });

  it('buildProjectionMatrix is deterministic (same output every call)', () => {
    const P1 = buildProjectionMatrix();
    const P2 = buildProjectionMatrix();
    // Both should be identical (singleton or same seed)
    expect(P1[0]).toBe(P2[0]);
    expect(P1[100]).toBe(P2[100]);
    expect(P1[P1.length - 1]).toBe(P2[P2.length - 1]);
  });
});

describe('Task 3: SigLIP adapter — bilinear upsample', () => {
  it('upsample 14×14×768 → 24×24×768 produces correct size', () => {
    const src = new Float32Array(SIGLIP_GRID * SIGLIP_GRID * SIGLIP_DIM);
    // Fill with index values for verification
    for (let i = 0; i < src.length; i++) src[i] = i * 0.001;

    const dst = bilinearUpsample(src, SIGLIP_GRID, SIGLIP_GRID, SIGLIP_DIM, TARGET_GRID, TARGET_GRID);
    expect(dst).toBeInstanceOf(Float32Array);
    expect(dst.length).toBe(TARGET_GRID * TARGET_GRID * SIGLIP_DIM);  // 24*24*768 = 442368
  });

  it('upsample of a constant array returns the same constant', () => {
    const C = 42;
    const src = new Float32Array(SIGLIP_GRID * SIGLIP_GRID * 3).fill(C);
    const dst = bilinearUpsample(src, SIGLIP_GRID, SIGLIP_GRID, 3, TARGET_GRID, TARGET_GRID);
    for (let i = 0; i < dst.length; i++) {
      expect(dst[i]).toBeCloseTo(C, 4);
    }
  });

  it('upsample 1×1→2×2 center pixel (identity check)', () => {
    const src = new Float32Array([3.14]);
    const dst = bilinearUpsample(src, 1, 1, 1, 2, 2);
    expect(dst.length).toBe(4);
    for (const v of dst) expect(v).toBeCloseTo(3.14, 4);
  });

  it('upsample is deterministic (same input → same output)', () => {
    const src = new Float32Array(SIGLIP_GRID * SIGLIP_GRID * SIGLIP_DIM);
    for (let i = 0; i < src.length; i++) src[i] = Math.sin(i * 0.1);
    const dst1 = bilinearUpsample(src, SIGLIP_GRID, SIGLIP_GRID, SIGLIP_DIM, TARGET_GRID, TARGET_GRID);
    const dst2 = bilinearUpsample(src, SIGLIP_GRID, SIGLIP_GRID, SIGLIP_DIM, TARGET_GRID, TARGET_GRID);
    expect(dst1[0]).toBe(dst2[0]);
    expect(dst1[1000]).toBe(dst2[1000]);
  });
});

describe('Task 3: SigLIP adapter — projection', () => {
  it('projectEmbeddings [196, 768] → [196, 1152] correct shape', () => {
    const embeddings = new Float32Array(SIGLIP_PATCHES * SIGLIP_DIM);
    for (let i = 0; i < embeddings.length; i++) embeddings[i] = Math.random();
    const out = projectEmbeddings(embeddings, SIGLIP_PATCHES);
    expect(out.length).toBe(SIGLIP_PATCHES * TARGET_DIM);  // 196 * 1152
  });

  it('projectEmbeddings [576, 768] → [576, 1152] correct shape', () => {
    const embeddings = new Float32Array(TARGET_PATCHES * SIGLIP_DIM);
    for (let i = 0; i < embeddings.length; i++) embeddings[i] = Math.random();
    const out = projectEmbeddings(embeddings, TARGET_PATCHES);
    expect(out.length).toBe(TARGET_PATCHES * TARGET_DIM);  // 576 * 1152 = 663552
  });

  it('projectEmbeddings is deterministic (same seed → same projection)', () => {
    const embeddings = new Float32Array(5 * SIGLIP_DIM);
    for (let i = 0; i < embeddings.length; i++) embeddings[i] = i * 0.01;
    const out1 = projectEmbeddings(embeddings, 5);
    const out2 = projectEmbeddings(embeddings, 5);
    expect(out1[0]).toBe(out2[0]);
    expect(out1[out1.length - 1]).toBe(out2[out2.length - 1]);
  });

  it('zero input → zero output', () => {
    const embeddings = new Float32Array(2 * SIGLIP_DIM);  // all zeros
    const out = projectEmbeddings(embeddings, 2);
    for (const v of out) expect(Math.abs(v)).toBeLessThan(1e-10);
  });
});

describe('Task 3: SigLIP adapter — full pipeline shape (no ONNX)', () => {
  it('upsample+project pipeline: 196×768 → 576×1152', () => {
    // Simulate SigLIP output [196, 768]
    const siglipOut = new Float32Array(SIGLIP_PATCHES * SIGLIP_DIM);
    for (let i = 0; i < siglipOut.length; i++) siglipOut[i] = Math.sin(i * 0.1);

    // 1. Bilinear upsample 14×14×768 → 24×24×768
    const upsampled = bilinearUpsample(
      siglipOut, SIGLIP_GRID, SIGLIP_GRID, SIGLIP_DIM, TARGET_GRID, TARGET_GRID,
    );
    expect(upsampled.length).toBe(TARGET_PATCHES * SIGLIP_DIM);

    // 2. Project 576×768 → 576×1152
    const projected = projectEmbeddings(upsampled, TARGET_PATCHES);
    expect(projected.length).toBe(TARGET_PATCHES * TARGET_DIM);
    expect(projected.length).toBe(TENSOR_CONTRACT.patchCount * TENSOR_CONTRACT.embeddingDim);
  });

  it('pipeline is deterministic for fixed SigLIP output', () => {
    const siglipOut = new Float32Array(SIGLIP_PATCHES * SIGLIP_DIM);
    for (let i = 0; i < siglipOut.length; i++) siglipOut[i] = Math.cos(i * 0.05);

    const run = () => {
      const up = bilinearUpsample(siglipOut, SIGLIP_GRID, SIGLIP_GRID, SIGLIP_DIM, TARGET_GRID, TARGET_GRID);
      return projectEmbeddings(up, TARGET_PATCHES);
    };

    const r1 = run();
    const r2 = run();
    for (let i = 0; i < r1.length; i += 1000) {
      expect(r1[i]).toBe(r2[i]);
    }
  });
});

describe('Task 3: OnnxVisionEncoder — fail loudly in onnx mode', () => {
  it('init() throws ONNX_MODEL_MISSING when no modelUrl given', async () => {
    // Dynamically import to avoid top-level module issues in test environment
    const { OnnxVisionEncoder } = await import('../src/vision/onnx-encoder');
    const encoder = new OnnxVisionEncoder('', 'onnx');
    await expect(encoder.init()).rejects.toThrow('ONNX_MODEL_MISSING');
  });

  it('init() throws ONNX_MODEL_MISSING for a bad model URL in onnx mode', async () => {
    const { OnnxVisionEncoder } = await import('../src/vision/onnx-encoder');
    const encoder = new OnnxVisionEncoder('http://localhost:9/nonexistent.onnx', 'onnx');
    await expect(encoder.init()).rejects.toThrow('ONNX_MODEL_MISSING');
  });
});

