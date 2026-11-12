import type { InferenceSession } from "onnxruntime-web";
import type { PreTrainedTokenizer } from "@huggingface/transformers";
import { TENSOR_CONTRACT } from "../protocol/messages";
import { SIGLIP_DIM } from "./projection";

const TEXT_LENGTH = 64;
const TOKENIZER_ID = "siglip-base-patch16-224-text-tokenizer";
const FP16_TEXT_MODEL = "siglip-base-patch16-224-text-fp16.onnx";
const FP32_TEXT_MODEL = "siglip-base-patch16-224-text-fp32.onnx";
const SCORE_ADAPTER_MODEL = "siglip-base-patch16-224-text-image-score.onnx";
const SOURCE_SCORE_ADAPTER_MODEL = "siglip-base-patch16-224-source-text-score.onnx";

type ExecutionProvider = "webgpu" | "wasm";

export function computeTextImageScores(
  siglipImageEmbeddings: Float32Array,
  textEmbedding: Float32Array,
): Float32Array {
  const { patchCount } = TENSOR_CONTRACT;
  if (siglipImageEmbeddings.length !== patchCount * SIGLIP_DIM) {
    throw new Error(
      `Expected ${patchCount * SIGLIP_DIM} SigLIP image values, got ${siglipImageEmbeddings.length}`,
    );
  }
  if (textEmbedding.length !== SIGLIP_DIM) {
    throw new Error(
      `Expected text embedding length ${SIGLIP_DIM}, got ${textEmbedding.length}`,
    );
  }
  assertFinite(siglipImageEmbeddings, "siglipImageEmbeddings");
  assertFinite(textEmbedding, "textEmbedding");
  const normalizedText = l2Normalize(textEmbedding);
  const scores = new Float32Array(patchCount);
  for (let patch = 0; patch < patchCount; patch++) {
    const offset = patch * SIGLIP_DIM;
    let squaredNorm = 0;
    for (let dimension = 0; dimension < SIGLIP_DIM; dimension++) {
      const value = siglipImageEmbeddings[offset + dimension]!;
      squaredNorm += value * value;
    }
    const norm = Math.sqrt(squaredNorm);
    if (norm === 0) continue;
    let dot = 0;
    for (let dimension = 0; dimension < SIGLIP_DIM; dimension++) {
      dot +=
        (siglipImageEmbeddings[offset + dimension]! / norm) *
        normalizedText[dimension]!;
    }
    scores[patch] = Math.max(-1, Math.min(1, dot));
  }
  return scores;
}

export class SiglipTextEncoder {
  private textSession: InferenceSession | null = null;
  private scoreSession: InferenceSession | null = null;
  private sourceScoreSession: InferenceSession | null = null;
  private tokenizer: PreTrainedTokenizer | null = null;

  constructor(
    private readonly modelUrl = `/models/${FP16_TEXT_MODEL}`,
    private readonly executionProviders: ExecutionProvider[] = ["webgpu", "wasm"],
    private readonly scoreAdapterUrl = `/models/${SCORE_ADAPTER_MODEL}`,
    private readonly sourceScoreAdapterUrl = `/models/${SOURCE_SCORE_ADAPTER_MODEL}`,
  ) {}

  async init(): Promise<void> {
    if (!this.modelUrl || !this.scoreAdapterUrl || !this.sourceScoreAdapterUrl) {
      throw new Error(
        "ONNX_TEXT_MODEL_MISSING: configure the SigLIP text and score-adapter model URLs.",
      );
    }
    if (
      this.textSession &&
      this.scoreSession &&
      this.sourceScoreSession &&
      this.tokenizer
    ) return;

    const ort = await import("onnxruntime-web/webgpu");
    const modelUrls = this.modelUrls();
    const failures: string[] = [];
    for (const provider of this.executionProviders) {
      const textModelUrl = provider === "wasm" ? modelUrls.fp32 : modelUrls.fp16;
      let textSession: InferenceSession | null = null;
      let scoreSession: InferenceSession | null = null;
      let sourceScoreSession: InferenceSession | null = null;
      try {
        textSession = await ort.InferenceSession.create(textModelUrl, {
          executionProviders: [provider],
        });
        scoreSession = await ort.InferenceSession.create(this.scoreAdapterUrl, {
          executionProviders: [provider],
        });
        sourceScoreSession = await ort.InferenceSession.create(
          this.sourceScoreAdapterUrl,
          { executionProviders: [provider] },
        );
        if (
          !textSession.inputNames.includes("input_ids") ||
          !textSession.inputNames.includes("attention_mask") ||
          !textSession.outputNames.includes("text_embedding")
        ) {
          throw new Error(
            "Text graph must expose input_ids, attention_mask and text_embedding",
          );
        }
        if (
          !scoreSession.inputNames.includes("adapted_embeddings") ||
          !scoreSession.inputNames.includes("text_embedding") ||
          !scoreSession.outputNames.includes("score_grid")
        ) {
          throw new Error(
            "Score graph must expose adapted_embeddings, text_embedding and score_grid",
          );
        }
        if (
          !sourceScoreSession.inputNames.includes("source_embeddings") ||
          !sourceScoreSession.inputNames.includes("text_embedding") ||
          !sourceScoreSession.outputNames.includes("score_grid")
        ) {
          throw new Error(
            "Source score graph must expose source_embeddings, text_embedding and score_grid",
          );
        }
        this.textSession = textSession;
        this.scoreSession = scoreSession;
        this.sourceScoreSession = sourceScoreSession;
        break;
      } catch (error) {
        if (textSession) await textSession.release();
        if (scoreSession) await scoreSession.release();
        if (sourceScoreSession) await sourceScoreSession.release();
        failures.push(
          `${provider} (${textModelUrl}, ${this.scoreAdapterUrl}, ${this.sourceScoreAdapterUrl}): ${
            error instanceof Error ? error.message : String(error)
          }`,
        );
      }
    }
    if (!this.textSession || !this.scoreSession || !this.sourceScoreSession) {
      throw new Error(
        `ONNX_TEXT_MODEL_LOAD_FAILED: unable to load the SigLIP text tower and score adapter: ${failures.join("; ")}`,
      );
    }

    try {
      const { AutoTokenizer, env } = await import("@huggingface/transformers");
      env.allowRemoteModels = false;
      env.allowLocalModels = true;
      env.localModelPath = this.localModelsPath();
      this.tokenizer = await AutoTokenizer.from_pretrained(TOKENIZER_ID);
    } catch (error) {
      await this.textSession.release();
      await this.scoreSession.release();
      await this.sourceScoreSession.release();
      this.textSession = null;
      this.scoreSession = null;
      this.sourceScoreSession = null;
      throw new Error(
        "ONNX_TEXT_TOKENIZER_LOAD_FAILED: the local SigLIP tokenizer assets are missing or invalid. " +
          `Export them with scripts/export-siglip-text-onnx.py. ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  async encodeTask(taskIntent: string): Promise<Float32Array> {
    if (!this.textSession || !this.tokenizer) {
      throw new Error("SiglipTextEncoder is not initialized; call init() first");
    }
    if (!taskIntent.trim()) throw new Error("taskIntent must not be empty");
    const tokenizerOptions = {
      padding: "max_length",
      truncation: true,
      max_length: TEXT_LENGTH,
      return_attention_mask: true,
    } as const;
    const encoded = await this.tokenizer(taskIntent, tokenizerOptions);
    if (
      encoded.input_ids.dims.join(",") !== `1,${TEXT_LENGTH}` ||
      encoded.attention_mask.dims.join(",") !== `1,${TEXT_LENGTH}`
    ) {
      throw new Error(
        `SigLIP tokenizer must return [1,${TEXT_LENGTH}] input_ids and attention_mask`,
      );
    }
    const inputIds = toInt64TensorData(encoded.input_ids.data, "input_ids");
    const attentionMask = toInt64TensorData(
      encoded.attention_mask.data,
      "attention_mask",
    );
    const ort = await import("onnxruntime-web/webgpu");
    const outputs = await this.textSession.run({
      input_ids: new ort.Tensor("int64", inputIds, [1, TEXT_LENGTH]),
      attention_mask: new ort.Tensor("int64", attentionMask, [1, TEXT_LENGTH]),
    });
    const output = outputs.text_embedding;
    if (
      !output ||
      output.dims.join(",") !== `1,${SIGLIP_DIM}` ||
      !(output.data instanceof Float32Array)
    ) {
      throw new Error(
        `Invalid SigLIP text output; expected float32 [1,${SIGLIP_DIM}]`,
      );
    }
    assertFinite(output.data, "SigLIP text output");
    return output.data;
  }

  async scoreGrid(
    sanitizedEmbeddings: Float32Array,
    taskIntent: string,
  ): Promise<Float32Array> {
    if (!this.scoreSession || !this.sourceScoreSession) {
      throw new Error("SiglipTextEncoder is not initialized; call init() first");
    }
    const { patchCount, embeddingDim } = TENSOR_CONTRACT;
    const isSourceGrid = sanitizedEmbeddings.length === 196 * SIGLIP_DIM;
    const isAdaptedGrid = sanitizedEmbeddings.length === patchCount * embeddingDim;
    if (!isSourceGrid && !isAdaptedGrid) {
      throw new Error(
        `Expected ${196 * SIGLIP_DIM} sanitized source values or ${patchCount * embeddingDim} adapted values, got ${sanitizedEmbeddings.length}`,
      );
    }
    assertFinite(sanitizedEmbeddings, "sanitizedEmbeddings");
    const textEmbedding = await this.encodeTask(taskIntent);
    const ort = await import("onnxruntime-web/webgpu");
    const outputs = isSourceGrid
      ? await this.sourceScoreSession.run({
          source_embeddings: new ort.Tensor(
            "float32",
            sanitizedEmbeddings,
            [1, 196, SIGLIP_DIM],
          ),
          text_embedding: new ort.Tensor("float32", textEmbedding, [1, SIGLIP_DIM]),
        })
      : await this.scoreSession.run({
          adapted_embeddings: new ort.Tensor(
            "float32",
            sanitizedEmbeddings,
            [1, patchCount, embeddingDim],
          ),
          text_embedding: new ort.Tensor("float32", textEmbedding, [1, SIGLIP_DIM]),
        });
    const output = outputs.score_grid;
    if (
      !output ||
      output.dims.join(",") !== `1,${patchCount}` ||
      !(output.data instanceof Float32Array)
    ) {
      throw new Error(
        `Invalid SigLIP score output; expected float32 [1,${patchCount}]`,
      );
    }
    assertFinite(output.data, "SigLIP score grid");
    return output.data;
  }

  destroy(): void {
    if (this.textSession) void this.textSession.release();
    if (this.scoreSession) void this.scoreSession.release();
    if (this.sourceScoreSession) void this.sourceScoreSession.release();
    this.textSession = null;
    this.scoreSession = null;
    this.sourceScoreSession = null;
    this.tokenizer = null;
  }

  private modelUrls(): { fp16: string; fp32: string } {
    const fp16 = this.modelUrl.replace(FP32_TEXT_MODEL, FP16_TEXT_MODEL);
    const fp32 = this.modelUrl.replace(FP16_TEXT_MODEL, FP32_TEXT_MODEL);
    return { fp16, fp32 };
  }

  private localModelsPath(): string {
    if (typeof chrome !== "undefined" && chrome.runtime?.getURL) {
      return chrome.runtime.getURL("models/");
    }
    return "/models/";
  }
}

function toInt64TensorData(
  values: ArrayLike<number | bigint>,
  name: string,
): BigInt64Array {
  if (values.length !== TEXT_LENGTH) {
    throw new Error(`Expected ${TEXT_LENGTH} token values for ${name}`);
  }
  return BigInt64Array.from(Array.from(values, (value) => BigInt(value)));
}

function l2Normalize(values: Float32Array): Float32Array {
  let squaredNorm = 0;
  for (const value of values) squaredNorm += value * value;
  const norm = Math.sqrt(squaredNorm);
  if (norm === 0) throw new Error("Cannot normalize a zero text embedding");
  const output = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) output[i] = values[i]! / norm;
  return output;
}

function assertFinite(values: Float32Array, name: string): void {
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) {
      throw new Error(`${name} contains a non-finite value at index ${i}`);
    }
  }
}
