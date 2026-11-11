export const SOURCE_PCA_VERSION = "source_pca_v1" as const;
export const SOURCE_PCA_DIM = 768;
export const SOURCE_PCA_MAX_COMPONENTS = 128;
export const SOURCE_PCA_DEFAULT_COMPONENTS = 64;

export interface SourcePcaBasis {
  mean: Float32Array;
  components: Float32Array;
  rank: number;
}

let cachedBasis: Promise<SourcePcaBasis> | null = null;

export function loadSourcePcaBasis(): Promise<SourcePcaBasis> {
  cachedBasis ??= fetch("/pca/source-pca-v1.bin")
    .then(async (response) => {
      if (!response.ok) {
        throw new Error(`SOURCE_PCA_BASIS_MISSING: HTTP ${response.status} loading /pca/source-pca-v1.bin`);
      }
      const buffer = await response.arrayBuffer();
      if (buffer.byteLength % 4 !== 0) {
        throw new Error(`SOURCE_PCA_BASIS_INVALID: byte length ${buffer.byteLength} is not float32-aligned`);
      }
      const values = new Float32Array(buffer.byteLength / 4);
      const view = new DataView(buffer);
      for (let i = 0; i < values.length; i++) values[i] = view.getFloat32(i * 4, true);
      const expected = SOURCE_PCA_DIM * (SOURCE_PCA_MAX_COMPONENTS + 1);
      if (values.length !== expected) {
        throw new Error(`SOURCE_PCA_BASIS_INVALID: expected ${expected} float32 values, got ${values.length}`);
      }
      return {
        mean: values.slice(0, SOURCE_PCA_DIM),
        components: values.slice(SOURCE_PCA_DIM),
        rank: SOURCE_PCA_MAX_COMPONENTS,
      };
    })
    .catch((error: unknown) => {
      cachedBasis = null;
      throw error;
    });
  return cachedBasis;
}

export function projectSourcePca(
  source: Float32Array,
  basis: SourcePcaBasis,
  componentCount = SOURCE_PCA_DEFAULT_COMPONENTS,
): Float32Array {
  if (source.length !== 196 * SOURCE_PCA_DIM) {
    throw new Error(`Expected 196x768 source SigLIP tokens, got ${source.length} values`);
  }
  if (!Number.isInteger(componentCount) || componentCount < 1 || componentCount > basis.rank) {
    throw new Error(`PCA component count must be between 1 and ${basis.rank}`);
  }
  const result = new Float32Array(196 * componentCount);
  for (let patch = 0; patch < 196; patch++) {
    const sourceOffset = patch * SOURCE_PCA_DIM;
    for (let component = 0; component < componentCount; component++) {
      let sum = 0;
      const basisOffset = component * SOURCE_PCA_DIM;
      for (let dim = 0; dim < SOURCE_PCA_DIM; dim++) {
        sum += (source[sourceOffset + dim]! - basis.mean[dim]!) *
          basis.components[basisOffset + dim]!;
      }
      result[patch * componentCount + component] = sum;
    }
  }
  return result;
}

export function reconstructSourcePca(
  coefficients: Float32Array,
  basis: SourcePcaBasis,
  componentCount: number,
): Float32Array {
  if (coefficients.length !== 196 * componentCount ||
      !Number.isInteger(componentCount) || componentCount < 1 || componentCount > basis.rank) {
    throw new Error(`Expected 196xN PCA coefficients with 1 <= N <= ${basis.rank}`);
  }
  const result = new Float32Array(196 * SOURCE_PCA_DIM);
  for (let patch = 0; patch < 196; patch++) {
    const targetOffset = patch * SOURCE_PCA_DIM;
    for (let dim = 0; dim < SOURCE_PCA_DIM; dim++) {
      let value = basis.mean[dim]!;
      for (let component = 0; component < componentCount; component++) {
        value += coefficients[patch * componentCount + component]! *
          basis.components[component * SOURCE_PCA_DIM + dim]!;
      }
      result[targetOffset + dim] = value;
    }
  }
  return result;
}
