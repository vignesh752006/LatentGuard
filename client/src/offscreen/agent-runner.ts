import { loadConfig, type ClientConfig } from "../config/settings";
import { log } from "../logging/logger";
import type {
  ActionDirectivePayload,
  LatentFramePayload,
  SessionInitPayload,
} from "../protocol/messages";
import { defaultEncoderSpec } from "../protocol/messages";
import {
  parseServerMessage,
  serializeActionResult,
  serializeLatentFrame,
  serializeSessionInit,
} from "../protocol/websocket-client";
import type { AgentStatus, SensitivityScanResult } from "../messaging/types";
import { persistStatusFromOffscreen } from "../storage/agent-status";
import { decodeCaptureToImageData } from "../capture/decode";
import { buildLatentFramePayload, hashViewport } from "../protocol/serializer";
import { VisionEncoder } from "../vision/mock-gpu-encoder";
import { sanitizeRankOne, sanitizeSVD } from "../vdlm/svd-sanitizer";
import {
  loadSourcePcaBasis,
  projectSourcePca,
  SOURCE_PCA_DEFAULT_COMPONENTS,
} from "../vision/source-pca";

type VisionEncoderLike = {
  init(): Promise<void>;
  encode(image: ImageData): Promise<Float32Array>;
  encodePatches?(image: ImageData): Promise<Float32Array>;
};

const persistStatus = persistStatusFromOffscreen;

export interface StageTiming {
  stage: string;
  durationMs: number;
}

export interface FrameTiming {
  frameId: string;
  totalMs: number;
  stages: StageTiming[];
}

/** Rolling buffer of the last 100 frame timings (for p50/p95 reporting). */
const _framingHistory: FrameTiming[] = [];

export function getFrameTimings(): readonly FrameTiming[] {
  return _framingHistory;
}

export function computeLatencyPercentiles(key: "totalMs" | string): {
  p50: number;
  p95: number;
  count: number;
} {
  const values = _framingHistory
    .map((f) => {
      if (key === "totalMs") return f.totalMs;
      return f.stages.find((s) => s.stage === key)?.durationMs ?? NaN;
    })
    .filter((v) => !isNaN(v))
    .sort((a, b) => a - b);

  if (values.length === 0) return { p50: 0, p95: 0, count: 0 };
  const p50 = values[Math.floor(values.length * 0.5)]!;
  const p95 = values[Math.floor(values.length * 0.95)]!;
  return { p50, p95, count: values.length };
}

export class OffscreenAgentRunner {
  private ws: WebSocket | null = null;
  private sessionId: string | null = null;
  private running = false;
  private tabId: number | null = null;
  private frameCount = 0;
  private config: ClientConfig | null = null;
  private loopTimer: ReturnType<typeof setTimeout> | null = null;
  private sessionReadyWaiters: Array<(id: string) => void> = [];
  private visionEncoder: VisionEncoderLike | null = null;
  private pipelineReady = false;

  async start(tabId: number, config?: ClientConfig): Promise<void> {
    this.config = config ?? (await loadConfig());
    this.tabId = tabId;
    this.running = true;
    this.frameCount = 0;
    this.sessionId = null;

    try {
      await persistStatus({
        running: true,
        tabId,
        lastError: null,
        frameCount: 0,
        sessionId: null,
        pipelineStage: "starting",
        visionMode: this.config.visionMode,
      });

      await this.ensurePipeline();
      await this.connectWebSocket();

      const viewport = await this.getViewport();
      await this.initSession(viewport);

      void this.runCycle();
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      this.running = false;
      this.ws?.close();
      this.ws = null;
      await persistStatus({
        running: false,
        connected: false,
        pipelineStage: "error",
        lastError: msg,
        tabId: null,
      });
      throw error;
    }
  }

  async stop(): Promise<void> {
    this.running = false;
    if (this.loopTimer) clearTimeout(this.loopTimer);
    this.loopTimer = null;
    this.ws?.close();
    this.ws = null;
    this.sessionId = null;
    await persistStatus({
      running: false,
      connected: false,
      sessionId: null,
      pipelineStage: "idle",
      tabId: null,
    });
  }

  private async ensurePipeline(): Promise<void> {
    if (this.pipelineReady) return;
    if (this.config?.visionMode === "onnx") {
      const { OnnxVisionEncoder } = await import("../vision/onnx-encoder");
      this.visionEncoder = new OnnxVisionEncoder(this.config.onnxModelUrl);
    } else {
      this.visionEncoder = new VisionEncoder();
    }
    await this.visionEncoder.init();
    this.pipelineReady = true;
  }

  private async connectWebSocket(): Promise<void> {
    if (!this.config) throw new Error("Config missing");
    await this.setStage("connecting");

    await new Promise<void>((resolve, reject) => {
      const ws = new WebSocket(this.config!.backendWsUrl);
      this.ws = ws;
      ws.onopen = () => resolve();
      ws.onerror = () => reject(new Error("WebSocket failed — is backend running on port 8080?"));
      ws.onclose = () => {
        if (this.running) void this.setStage("error", "WebSocket closed");
      };
      ws.onmessage = (e) => void this.onServerMessage(String(e.data));
    });

    await persistStatus({ connected: true });
  }

  private waitForSessionReady(): Promise<string> {
    if (this.sessionId) return Promise.resolve(this.sessionId);
    return new Promise((resolve, reject) => {
      const t = setTimeout(() => reject(new Error("session.ready timeout")), 10000);
      this.sessionReadyWaiters.push((id) => {
        clearTimeout(t);
        resolve(id);
      });
    });
  }

  private async initSession(viewport: { width: number; height: number }): Promise<void> {
    if (!this.ws || !this.config) throw new Error("Not connected");
    await this.setStage("session_init");

    const ready = this.waitForSessionReady();
    const payload: SessionInitPayload = {
      task_intent: this.config.taskIntent,
      encoder_spec: defaultEncoderSpec(),
      viewport,
      client_capabilities: ["webgpu", "onnx-runtime-web"],
    };
    this.ws.send(serializeSessionInit(payload));

    this.sessionId = await ready;
    await persistStatus({ sessionId: this.sessionId, pipelineStage: "ready" });
    log("info", "Session ready", { sessionId: this.sessionId });
  }

  private async runCycle(): Promise<void> {
    if (!this.running || this.tabId === null || !this.config || !this.sessionId) return;

    const cycleStart = performance.now();
    const stages: StageTiming[] = [];
    const mark = (stage: string, start: number) => {
      stages.push({ stage, durationMs: performance.now() - start });
    };

    try {
      let t = performance.now();
      await this.setStage("scanning");
      const sensitivity = await this.relayTab<SensitivityScanResult>({ type: "SCAN_SENSITIVITY" });
      const stateVersion = await this.relayTab<{ stateVersion: number }>({ type: "GET_STATE_VERSION" });
      mark("scan", t);

      t = performance.now();
      await this.setStage("capturing");
      const captureJpeg = await this.requestCapture();
      mark("capture", t);

      this.frameCount += 1;
      const frameId = `f-${String(this.frameCount).padStart(5, "0")}`;

      t = performance.now();
      await this.setStage("processing");
      const imageData = await decodeCaptureToImageData(captureJpeg);
      mark("decode", t);

      const sanitizerMethod = this.config.sanitizerMode === "rank1" ? "rank1" : "svd";
      let sanitized: Float32Array | undefined;
      let sourcePcaCoefficients: Float32Array | undefined;
      let pcaComponentCount: number | undefined;
      let encodeStart = performance.now();
      if (this.config.payloadMode === "source-pca") {
        if (!this.visionEncoder?.encodePatches) {
          throw new Error("Source PCA payloads require ONNX SigLIP mode; Mock test mode cannot produce source tokens.");
        }
        const source = await this.visionEncoder.encodePatches(imageData);
        mark("encode", encodeStart);
        encodeStart = performance.now();
        const sourceIndices = mapTargetIndicesToSourceGrid(sensitivity.sensitivePatchIndices);
        const sanitizedSource = sanitizerMethod === "rank1"
          ? sanitizeRankOne(
            source,
            sourceIndices,
            this.config.sanitizerAlpha,
            { patchCount: 196, embeddingDim: 768 },
          )
          : sanitizeSVD(source, sourceIndices, {
            k: this.config.sanitizerK,
            alpha: this.config.sanitizerAlpha,
            patchCount: 196,
            embeddingDim: 768,
          }).sanitized;
        const basis = await loadSourcePcaBasis();
        pcaComponentCount = SOURCE_PCA_DEFAULT_COMPONENTS;
        sourcePcaCoefficients = projectSourcePca(
          sanitizedSource,
          basis,
          pcaComponentCount,
        );
        source.fill(0);
        sanitizedSource.fill(0);
      } else {
        const embeddings = await this.visionEncoder!.encode(imageData);
        mark("encode", encodeStart);
        encodeStart = performance.now();
        sanitized = sanitizerMethod === "rank1"
          ? sanitizeRankOne(embeddings, sensitivity.sensitivePatchIndices, this.config.sanitizerAlpha)
          : sanitizeSVD(embeddings, sensitivity.sensitivePatchIndices, {
            k: this.config.sanitizerK,
            alpha: this.config.sanitizerAlpha,
          }).sanitized;
        embeddings.fill(0);
      }
      mark("sanitize", encodeStart);

      t = performance.now();
      const viewportHash = await hashViewport(
        sensitivity.viewport.width,
        sensitivity.viewport.height,
        stateVersion.stateVersion,
      );

      const latentFrame = buildLatentFramePayload({
        frameId,
        stateVersion: stateVersion.stateVersion,
        ...(sanitized ? { sanitizedMatrix: sanitized } : {}),
        ...(sourcePcaCoefficients ? { sourcePcaCoefficients, pcaComponentCount } : {}),
        sensitivePatchIndices: sensitivity.sensitivePatchIndices,
        viewportHash,
        lastActionId: null,
        compress: this.config.compressPayload,
        sanitizationMethod: sanitizerMethod === "svd"
          ? `svd_nullspace_k${this.config.sanitizerK}`
          : "rank1_nullspace_baseline",
      });
      sanitized?.fill(0);
      sourcePcaCoefficients?.fill(0);
      mark("serialize", t);

      t = performance.now();
      await this.setStage("transmitting");
      this.ws!.send(serializeLatentFrame(latentFrame, this.sessionId));
      mark("network_send", t);

      await this.setStage("awaiting_action");
      await persistStatus({ frameCount: this.frameCount, lastError: null });

      const totalMs = performance.now() - cycleStart;
      const timing: FrameTiming = { frameId, totalMs, stages };
      _framingHistory.push(timing);
      if (_framingHistory.length > 100) _framingHistory.shift();
      log("debug", "[VDLM] frame timing", { frameId, totalMs: totalMs.toFixed(1), stages });
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      log("error", "Cycle failed", { error: msg });
      await this.setStage("error", msg);
    } finally {
      if (this.running) {
        this.loopTimer = setTimeout(() => void this.runCycle(), this.config!.frameIntervalMs);
      }
    }
  }

  private async onServerMessage(raw: string): Promise<void> {
    const parsed = parseServerMessage(raw);

    if (parsed.kind === "session.ready") {
      const id = parsed.message.payload.session_id;
      this.sessionId = id;
      this.sessionReadyWaiters.splice(0).forEach((fn) => fn(id));
      return;
    }

    if (parsed.kind === "action.directive" && this.tabId !== null) {
      await this.setStage("executing");
      const directive = parsed.message.payload;
      const result = await this.relayTab<{ stateVersion: number; status: string; errorCode: string | null }>({
        type: "EXECUTE_ACTION",
        payload: directive,
      });

      if (this.ws && this.sessionId) {
        this.ws.send(
          serializeActionResult(
            {
              action_id: directive.action_id,
              state_version: result.stateVersion,
              status: result.status as "success" | "failure" | "cancelled",
              error_code: result.errorCode,
            },
            this.sessionId,
          ),
        );
      }

      if (directive.action === "DONE") await this.stop();
      else await this.setStage("awaiting_action");
    } else if (parsed.kind === "error") {
      await this.setStage("error", parsed.message.payload.message);
    }
  }

  private async setStage(stage: string, error: string | null = null): Promise<void> {
    await persistStatus({
      pipelineStage: stage,
      lastError: error,
      connected: this.ws?.readyState === WebSocket.OPEN,
      running: this.running,
      sessionId: this.sessionId,
      frameCount: this.frameCount,
      tabId: this.tabId,
    });
  }

  private async getViewport(): Promise<{ width: number; height: number }> {
    try {
      const scan = await this.relayTab<SensitivityScanResult>({ type: "SCAN_SENSITIVITY" });
      return scan.viewport;
    } catch {
      return { width: 1280, height: 720 };
    }
  }

  private relayTab<T>(message: Record<string, unknown>): Promise<T> {
    if (this.tabId === null) {
      return Promise.reject(new Error("No target tab"));
    }
    const tabId = this.tabId;
    return new Promise((resolve, reject) => {
      chrome.runtime.sendMessage(
        { type: "RELAY_TO_TAB", payload: { tabId, message } },
        (response) => {
          const err = chrome.runtime.lastError;
          if (err) reject(new Error(err.message));
          else if (response && typeof response === "object" && "error" in response) {
            reject(new Error(String((response as { error: string }).error)));
          } else resolve(response as T);
        },
      );
    });
  }

  private requestCapture(): Promise<string> {
    return new Promise((resolve, reject) => {
      chrome.runtime.sendMessage({ type: "CAPTURE_TAB" }, (response) => {
        const err = chrome.runtime.lastError;
        if (err) reject(new Error(err.message));
        else if (response?.error) reject(new Error(response.error));
        else resolve(response.captureJpeg as string);
      });
    });
  }
}

export function mapTargetIndicesToSourceGrid(targetIndices: number[]): number[] {
  const mapped = new Set<number>();
  for (const index of targetIndices) {
    if (!Number.isInteger(index) || index < 0 || index >= 24 * 24) continue;
    const row = Math.floor(index / 24);
    const col = index % 24;
    const sourceRow = Math.min(13, Math.floor((row + 0.5) * 14 / 24));
    const sourceCol = Math.min(13, Math.floor((col + 0.5) * 14 / 24));
    mapped.add(sourceRow * 14 + sourceCol);
  }
  return [...mapped];
}

let runner: OffscreenAgentRunner | null = null;

export function getRunner(): OffscreenAgentRunner {
  if (!runner) runner = new OffscreenAgentRunner();
  return runner;
}
