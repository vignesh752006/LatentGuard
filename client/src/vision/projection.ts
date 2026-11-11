export const SIGLIP_DIM = 768;
export const TARGET_DIM = 1152;
export const PROJECTION_SEED = 0xdeadbeef;

/** Build the row-major projection shared with backend/eval/siglip/projection.py. */
export function buildProjectionMatrix(): Float32Array {
  const matrix = new Float32Array(SIGLIP_DIM * TARGET_DIM);
  let state = PROJECTION_SEED >>> 0;

  for (let index = 0; index < matrix.length; index++) {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    matrix[index] = (state / 0x100000000) * 2 - 1;
  }

  for (let column = 0; column < TARGET_DIM; column++) {
    let squaredNorm = 0;
    for (let row = 0; row < SIGLIP_DIM; row++) {
      const value = matrix[row * TARGET_DIM + column]!;
      squaredNorm += value * value;
    }
    const scale = Math.sqrt(squaredNorm) + 1e-8;
    for (let row = 0; row < SIGLIP_DIM; row++) {
      matrix[row * TARGET_DIM + column]! /= scale;
    }
  }

  return matrix;
}

let projectionMatrix: Float32Array | null = null;

export function projectEmbeddings(embeddings: Float32Array, rows: number): Float32Array {
  if (embeddings.length !== rows * SIGLIP_DIM) {
    throw new Error(`Expected ${rows * SIGLIP_DIM} embedding values, got ${embeddings.length}`);
  }
  const matrix = projectionMatrix ??= buildProjectionMatrix();
  const output = new Float32Array(rows * TARGET_DIM);
  for (let row = 0; row < rows; row++) {
    const inputOffset = row * SIGLIP_DIM;
    const outputOffset = row * TARGET_DIM;
    for (let inputColumn = 0; inputColumn < SIGLIP_DIM; inputColumn++) {
      const value = embeddings[inputOffset + inputColumn]!;
      const projectionOffset = inputColumn * TARGET_DIM;
      for (let outputColumn = 0; outputColumn < TARGET_DIM; outputColumn++) {
        output[outputOffset + outputColumn] += value * matrix[projectionOffset + outputColumn]!;
      }
    }
  }
  return output;
}
