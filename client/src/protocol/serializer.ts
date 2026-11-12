import { zlibSync } from "fflate";

import {
  TENSOR_CONTRACT,
  type LatentFramePayload,
  type ScoreGrid,
  type TensorPayload,
} from "./messages";
import { ATTESTATION_SCHEME, buildAttestation } from "./attestation";
import {
  SOURCE_PCA_DEFAULT_COMPONENTS,
  SOURCE_PCA_MAX_COMPONENTS,
  SOURCE_PCA_VERSION,
} from "../vision/source-pca";

const FORBIDDEN_KEYS =
  /screenshot|raw_pixels?|image_data|dom_html|password|credential|payment/i;

export function assertOutboundPayloadSafe(obj: unknown, path = ""): void {
  if (obj === null || obj === undefined) return;
  if (typeof obj === "object") {
    if (Array.isArray(obj)) {
      obj.forEach((item, i) => assertOutboundPayloadSafe(item, `${path}[${i}]`));
      return;
    }
    for (const [key, value] of Object.entries(obj as Record<string, unknown>)) {
      const keyPath = path ? `${path}.${key}` : key;
      if (FORBIDDEN_KEYS.test(key)) {
        throw new Error(`Forbidden outbound field: ${key}`);
      }
      assertOutboundPayloadSafe(value, keyPath);
    }
  }
}

export function float32ToFloat16Buffer(matrix: Float32Array): ArrayBuffer {
  const out = new Uint16Array(matrix.length);
  for (let i = 0; i < matrix.length; i++) {
    out[i] = float32ToFloat16(matrix[i]!);
  }
  return out.buffer;
}

function float32ToFloat16(value: number): number {
  const floatView = new Float32Array(1);
  const intView = new Int32Array(floatView.buffer);
  floatView[0] = value;
  const bits = intView[0]!;
  const sign = (bits >> 16) & 0x8000;
  let exponent = ((bits >> 23) & 0xff) - 127 + 15;
  let mantissa = bits & 0x7fffff;

  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa = (mantissa | 0x800000) >> (1 - exponent);
    return sign | (mantissa >> 13);
  }
  if (exponent >= 31) return sign | 0x7c00;
  return sign | (exponent << 10) | (mantissa >> 13);
}

export function encodeTensorPayload(
  matrix: Float32Array,
  options?: { compress?: boolean },
): TensorPayload {
  const { patchCount, embeddingDim } = TENSOR_CONTRACT;
  if (matrix.length !== patchCount * embeddingDim) {
    throw new Error(
      `Tensor size mismatch: expected ${patchCount * embeddingDim}, got ${matrix.length}`,
    );
  }

  const fp16 = new Uint8Array(float32ToFloat16Buffer(matrix));
  let bytes: Uint8Array = fp16;
  let compression: TensorPayload["compression"] = "none";

  if (options?.compress) {
    bytes = zlibSync(fp16);
    compression = "zlib";
  }

  const base64 = uint8ToBase64(bytes);
  console.info(
    "[VDLM] wire bytes:",
    base64.length,
    "base64 bytes; compressed bytes:",
    bytes.byteLength,
    "compression:",
    compression,
  );
  return {
    encoding: "base64",
    dtype: "float16",
    shape: [patchCount, embeddingDim],
    data: base64,
    compression,
  };
}

export function buildLatentFramePayload(params: {
  frameId: string;
  stateVersion: number;
  sanitizedMatrix?: Float32Array;
  sourcePcaCoefficients?: Float32Array;
  pcaComponentCount?: number;
  sensitivePatchIndices: number[];
  viewportHash: string;
  lastActionId: string | null;
  compress?: boolean;
  /** Optional Task 6: per-patch text-image similarity scores (Float32Array of length 576). */
  scoreGrid?: Float32Array | null;
  /** Task 10: shader/model version string for attestation. Defaults to "0.1.0". */
  shaderVersion?: string;
  sanitizationMethod?: string;
}): LatentFramePayload {
  let scoreGridField: ScoreGrid | undefined;
  if (params.scoreGrid) {
    if (params.scoreGrid.length !== TENSOR_CONTRACT.patchCount) {
      throw new Error(
        `Expected ${TENSOR_CONTRACT.patchCount} task-similarity scores, got ${params.scoreGrid.length}`,
      );
    }
    scoreGridField = {
      data: uint8ToBase64(
        new Uint8Array(
          params.scoreGrid.buffer,
          params.scoreGrid.byteOffset,
          params.scoreGrid.byteLength,
        ),
      ),
      shape: [24, 24],
    };
  }

  const payload: LatentFramePayload = {
    frame_id: params.frameId,
    state_version: params.stateVersion,
    tensor: params.sourcePcaCoefficients
      ? encodeSourcePcaTensorPayload(
        params.sourcePcaCoefficients,
        params.pcaComponentCount ?? SOURCE_PCA_DEFAULT_COMPONENTS,
      )
      : encodeTensorPayload(params.sanitizedMatrix ?? new Float32Array(0), {
        compress: params.compress ?? false,
      }),
    spatial_metadata: {
      patch_grid: {
        rows: TENSOR_CONTRACT.gridRows,
        cols: TENSOR_CONTRACT.gridCols,
      },
      sensitive_patch_indices: [...params.sensitivePatchIndices].sort((a, b) => a - b),
      viewport_hash: params.viewportHash,
    },
    action_context: {
      last_action_id: params.lastActionId,
      pending_user_input: false,
    },
    provenance: {
      vdlm_shader_version: "0.1.0",
      sanitization_method: params.sanitizationMethod ?? "null_space_projection",
      attestation: null,
      attestation_scheme: ATTESTATION_SCHEME,
    },
    ...(scoreGridField ? { score_grid: scoreGridField } : {}),
  };
  assertOutboundPayloadSafe(payload);
  return payload;
}

/**
 * Compact tensor encoding: int8 per-row quantization + optional top-k sparsity + zlib.
 *
 * Wire format (before compression):
 *   [0..2303]       Float32Array(576) — per-row scale factors (max(abs(row)))
 *   [2304..665855]  Int8Array(576*1152) — quantized row values, row-major
 *
 * This encodes the full 576×1152 adapted tensor and does not meet the 45 KB
 * target on real SigLIP inputs. The source-grid PCA representation is encoded
 * separately by encodeSourcePcaTensorPayload.
 *
 * @param topKPercent  fraction [0,1] of values to retain by absolute magnitude
 *                     per row; 0 = keep all; 0.1 = keep top 10%.
 */
export function encodeCompactTensorPayload(
  matrix: Float32Array,
  options?: { topKPercent?: number },
): TensorPayload {
  const { patchCount, embeddingDim } = TENSOR_CONTRACT;
  if (matrix.length !== patchCount * embeddingDim) {
    throw new Error(
      `Tensor size mismatch: expected ${patchCount * embeddingDim}, got ${matrix.length}`,
    );
  }

  const topK = options?.topKPercent ?? 0;

  // Build scale array and int8 buffer
  const scales = new Float32Array(patchCount);
  const int8Data = new Int8Array(patchCount * embeddingDim);

  for (let row = 0; row < patchCount; row++) {
    const offset = row * embeddingDim;
    const rowSlice = matrix.subarray(offset, offset + embeddingDim);

    // Optional top-k sparsity: zero out all but top-k% by absolute value
    let rowValues = new Float32Array(rowSlice);
    if (topK > 0 && topK < 1) {
      const keepCount = Math.max(1, Math.round(embeddingDim * topK));
      const absVals = Array.from(rowValues).map((v, i) => ({ v: Math.abs(v), i }));
      absVals.sort((a, b) => b.v - a.v);
      const keepSet = new Set(absVals.slice(0, keepCount).map((x) => x.i));
      for (let j = 0; j < embeddingDim; j++) {
        if (!keepSet.has(j)) rowValues[j] = 0;
      }
    }

    // Per-row scale: max absolute value
    let maxAbs = 0;
    for (let j = 0; j < embeddingDim; j++) {
      const a = Math.abs(rowValues[j]!);
      if (a > maxAbs) maxAbs = a;
    }
    scales[row] = maxAbs;

    // Quantize to int8
    const invScale = maxAbs > 0 ? 127 / maxAbs : 0;
    for (let j = 0; j < embeddingDim; j++) {
      const q = Math.round(rowValues[j]! * invScale);
      int8Data[offset + j] = Math.max(-127, Math.min(127, q)) as unknown as number;
    }
  }

  // Pack: scales (Float32) + int8 values
  const packedSize = scales.byteLength + int8Data.byteLength;
  const packed = new Uint8Array(packedSize);
  packed.set(new Uint8Array(scales.buffer), 0);
  packed.set(new Uint8Array(int8Data.buffer), scales.byteLength);

  // Compress with zlib
  const compressed = zlibSync(packed);
  const base64 = uint8ToBase64(compressed);
  console.info(
    "[VDLM] compact wire bytes:",
    base64.length,
    "base64 bytes; compressed bytes:",
    compressed.byteLength,
    "ratio:",
    (compressed.byteLength / 1327104).toFixed(3),
    "topKPercent:",
    topK,
  );
  return {
    encoding: "base64",
    dtype: "int8_quantized",
    shape: [patchCount, embeddingDim],
    data: base64,
    compression: "zlib",
  };
}

export function encodeSourcePcaTensorPayload(
  coefficients: Float32Array,
  componentCount: number,
): TensorPayload {
  const patchCount = 196;
  if (!Number.isInteger(componentCount) || componentCount < 1 ||
      componentCount > SOURCE_PCA_MAX_COMPONENTS ||
      coefficients.length !== patchCount * componentCount) {
    throw new Error(`Expected 196xN source-PCA coefficients, got ${coefficients.length}`);
  }
  const scales = new Float32Array(patchCount);
  const values = new Int8Array(coefficients.length);
  for (let index = 0; index < coefficients.length; index++) {
    if (!Number.isFinite(coefficients[index]!)) {
      throw new Error(`Source PCA coefficients contain a non-finite value at index ${index}`);
    }
  }
  for (let row = 0; row < patchCount; row++) {
    const offset = row * componentCount;
    let maxAbs = 0;
    for (let dim = 0; dim < componentCount; dim++) {
      maxAbs = Math.max(maxAbs, Math.abs(coefficients[offset + dim]!));
    }
    scales[row] = maxAbs;
    const multiplier = maxAbs > 0 ? 127 / maxAbs : 0;
    for (let dim = 0; dim < componentCount; dim++) {
      values[offset + dim] = Math.max(
        -127,
        Math.min(127, Math.round(coefficients[offset + dim]! * multiplier)),
      );
    }
  }
  const packed = new Uint8Array(scales.byteLength + values.byteLength);
  packed.set(new Uint8Array(scales.buffer), 0);
  packed.set(new Uint8Array(values.buffer), scales.byteLength);
  const compressed = zlibSync(packed);
  const data = uint8ToBase64(compressed);
  console.info(
    "[VDLM] source PCA wire bytes:",
    data.length,
    "base64 bytes; compressed bytes:",
    compressed.byteLength,
    "shape:",
    `${patchCount}x${componentCount}`,
  );
  return {
    encoding: "base64",
    dtype: "int8_quantized",
    shape: [patchCount, componentCount],
    data,
    compression: "zlib",
    transform: SOURCE_PCA_VERSION,
  };
}

export async function hashViewport(width: number, height: number, stateVersion: number): Promise<string> {
  const data = new TextEncoder().encode(`${width}x${height}@v${stateVersion}`);
  const digest = await crypto.subtle.digest("SHA-256", data);
  const hex = [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
  return `sha256:${hex.slice(0, 16)}`;
}

/**
 * Task 10: Async variant of buildLatentFramePayload that generates and embeds
 * a signed ECDSA attestation in the provenance field.
 *
 * NOTE: This is an INTEGRITY CHECK, not a ZKP. Labelled clearly in the payload.
 */
export async function buildLatentFramePayloadWithAttestation(
  params: Parameters<typeof buildLatentFramePayload>[0] & { shaderVersion?: string },
): Promise<LatentFramePayload> {
  if (!params.sanitizedMatrix) {
    throw new Error("Attestation requires a full adapted sanitized matrix");
  }
  const payload = buildLatentFramePayload(params);
  try {
    const attestation = await buildAttestation(
      params.shaderVersion ?? payload.provenance.vdlm_shader_version ?? "0.1.0",
      params.sanitizedMatrix,
    );
    payload.provenance.attestation = attestation;
  } catch {
    // Non-fatal: attestation failure should not block the frame send.
    payload.provenance.attestation = null;
  }
  return payload;
}

function uint8ToBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

export function decodeBase64ToUint8(base64: string): Uint8Array {
  const binary = atob(base64);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}
