import { StrictMode, useCallback, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { loadConfig, saveConfig } from "../config/settings";
import type { AgentStatus } from "../messaging/types";

const defaultStatus: AgentStatus = {
  running: false,
  connected: false,
  sessionId: null,
  stateVersion: 0,
  frameCount: 0,
  lastError: null,
  visionMode: "onnx",
  processingMode: "auto",
  pipelineStage: "idle",
  tabId: null,
};

function Popup() {
  const [status, setStatus] = useState<AgentStatus>(defaultStatus);
  const [backendUrl, setBackendUrl] = useState("ws://127.0.0.1:8080/ws/v1/session");
  const [taskIntent, setTaskIntent] = useState("Click the Learn more link");
  const [visionMode, setVisionMode] = useState<"mock" | "onnx">("onnx");
  const [onnxModelUrl, setOnnxModelUrl] = useState("/models/siglip-base-patch16-224-vision-fp16.onnx");
  const [sanitizerMode, setSanitizerMode] = useState<"svd" | "rank1">("svd");
  const [sanitizerK, setSanitizerK] = useState(4);
  const [sanitizerAlpha, setSanitizerAlpha] = useState(0.01);
  const [payloadMode, setPayloadMode] = useState<"standard" | "source-pca">("standard");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refreshStatus = useCallback(async () => {
    const response = await chrome.runtime.sendMessage({ type: "GET_STATUS" });
    if (response) setStatus({ ...defaultStatus, ...response });
  }, []);

  useEffect(() => {
    void loadConfig().then((cfg) => {
      setBackendUrl(cfg.backendWsUrl);
      setTaskIntent(cfg.taskIntent);
      setVisionMode(cfg.visionMode);
      setOnnxModelUrl(cfg.onnxModelUrl);
      setSanitizerMode(cfg.sanitizerMode);
      setSanitizerK(cfg.sanitizerK);
      setSanitizerAlpha(cfg.sanitizerAlpha);
      setPayloadMode(cfg.payloadMode);
    });
    void refreshStatus();
    const id = setInterval(() => void refreshStatus(), 800);
    return () => clearInterval(id);
  }, [refreshStatus]);

  useEffect(() => {
    if (visionMode !== "onnx") {
      setError(null);
      return;
    }
    let active = true;
    const modelUrls = [
      onnxModelUrl,
      onnxModelUrl.replace("vision-fp16.onnx", "vision-fp32.onnx"),
      onnxModelUrl.replace(/vision-(fp16|fp32)\.onnx(?=[?#]|$)/, "adapter.onnx"),
    ];
    const requiredUrls = payloadMode === "source-pca"
      ? [...modelUrls, "/pca/source-pca-v1.bin"]
      : modelUrls;
    void Promise.all(requiredUrls.map(async (url) => {
      try {
        return (await fetch(url, { method: "HEAD" })).ok;
      } catch {
        return false;
      }
    })).then((available) => {
      if (!active) return;
      if (available.slice(0, 3).some((exists) => !exists)) {
        setError(
          "SigLIP ONNX model files are missing. Export them with " +
          "`python scripts/download-siglip-onnx.py --out-dir client/public/models`, then rebuild the extension.",
        );
      } else if (payloadMode === "source-pca" && !available[3]) {
        setError("Source PCA basis is missing. Run backend/eval/siglip/fit_source_pca.py, then rebuild the extension.");
      } else {
        setError(null);
      }
    });
    return () => {
      active = false;
    };
  }, [visionMode, onnxModelUrl, payloadMode]);

  const startAgent = async () => {
    setError(null);
    setBusy(true);
    try {
      if (visionMode === "onnx" && !onnxModelUrl.trim()) {
        throw new Error("SigLIP ONNX model URL is missing. Build the local models or select Mock test mode.");
      }
      if (!Number.isInteger(sanitizerK) || sanitizerK < 1 || sanitizerK > 8) {
        throw new Error("SVD rank must be an integer from 1 to 8.");
      }
      if (!Number.isFinite(sanitizerAlpha) || sanitizerAlpha < 0) {
        throw new Error("Sanitizer alpha must be a non-negative number.");
      }
      await saveConfig({
        backendWsUrl: backendUrl,
        taskIntent,
        visionMode,
        onnxModelUrl,
        sanitizerMode,
        sanitizerK,
        sanitizerAlpha,
        payloadMode,
      });
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab?.id) throw new Error("No active tab — click the webpage first");
      if (!tab.url?.startsWith("http")) {
        throw new Error("Open a normal website (e.g. https://example.com) first");
      }

      const result = await chrome.runtime.sendMessage({
        type: "AGENT_START",
        payload: { tabId: tab.id },
      });

      if (!result?.ok) throw new Error(result?.error ?? "Start failed");
      await refreshStatus();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const stopAgent = async () => {
    setBusy(true);
    await chrome.runtime.sendMessage({ type: "AGENT_STOP" });
    await refreshStatus();
    setBusy(false);
  };

  return (
    <main style={styles.main}>
      <h1 style={styles.title}>VDLM Agent</h1>
      <p style={styles.sub}>Privacy-preserving visual browser agent</p>

      <label style={styles.label}>
        Backend WebSocket
        <input style={styles.input} value={backendUrl} onChange={(e) => setBackendUrl(e.target.value)} />
      </label>

      <label style={styles.label}>
        Task intent
        <input style={styles.input} value={taskIntent} onChange={(e) => setTaskIntent(e.target.value)} />
      </label>

      <label style={styles.label}>
        Vision mode
        <select
          style={styles.input}
          value={visionMode}
          onChange={(e) => setVisionMode(e.target.value as "mock" | "onnx")}
        >
          <option value="mock">Mock (test mode)</option>
          <option value="onnx">ONNX SigLIP</option>
        </select>
      </label>

      {visionMode === "onnx" && (
        <label style={styles.label}>
          SigLIP ONNX model URL
          <input style={styles.input} value={onnxModelUrl} onChange={(e) => setOnnxModelUrl(e.target.value)} />
          <span style={styles.hint}>Missing local models will be reported when starting.</span>
        </label>
      )}

      <label style={styles.label}>
        Sanitizer
        <select
          style={styles.input}
          value={sanitizerMode}
          onChange={(e) => setSanitizerMode(e.target.value as "svd" | "rank1")}
        >
          <option value="svd">SVD null-space (default)</option>
          <option value="rank1">Rank-1 baseline</option>
        </select>
      </label>

      {sanitizerMode === "svd" && (
        <label style={styles.label}>
          SVD rank (1–8)
          <input style={styles.input} type="number" min={1} max={8} step={1} value={sanitizerK} onChange={(e) => setSanitizerK(Number(e.target.value))} />
        </label>
      )}

      <label style={styles.label}>
        Sanitizer alpha
        <input style={styles.input} type="number" min={0} step={0.001} value={sanitizerAlpha} onChange={(e) => setSanitizerAlpha(Number(e.target.value))} />
      </label>

      <label style={styles.label}>
        Payload representation
        <select
          style={styles.input}
          value={payloadMode}
          onChange={(e) => setPayloadMode(e.target.value as "standard" | "source-pca")}
        >
          <option value="standard">Full adapted tensor</option>
          <option value="source-pca">Source SigLIP with PCA (experimental)</option>
        </select>
      </label>

      <div style={styles.row}>
        <button
          style={styles.btnPrimary}
          onClick={() => void startAgent()}
          disabled={status.running || busy}
        >
          Start
        </button>
        <button style={styles.btn} onClick={() => void stopAgent()} disabled={!status.running || busy}>
          Stop
        </button>
      </div>

      {(error || status.lastError) && (
        <p role="alert" style={styles.error}>Error: {error ?? status.lastError}</p>
      )}

      <section style={styles.status}>
        <div>Running: {status.running ? "yes" : "no"}</div>
        <div>Connected: {status.connected ? "yes" : "no"}</div>
        <div>Stage: {status.pipelineStage}</div>
        <div>Session: {status.sessionId ?? "—"}</div>
        <div>Frames: {status.frameCount}</div>
        <div>Vision: {status.visionMode}</div>
      </section>

      <p style={styles.hint}>
        1. Start backend on :8080 · 2. Reload extension · 3. Refresh example.com · 4. Click Start
      </p>
    </main>
  );
}

const styles: Record<string, React.CSSProperties> = {
  main: { width: 320, padding: 12, fontFamily: "system-ui, sans-serif", fontSize: 13 },
  title: { margin: "0 0 4px", fontSize: 16 },
  sub: { margin: "0 0 12px", color: "#555" },
  label: { display: "block", marginBottom: 8 },
  input: { width: "100%", marginTop: 4, boxSizing: "border-box" },
  row: { display: "flex", gap: 8, marginTop: 8 },
  btnPrimary: { flex: 1, padding: "6px 8px" },
  btn: { flex: 1, padding: "6px 8px" },
  status: { marginTop: 12, padding: 8, background: "#f5f5f5", borderRadius: 6, lineHeight: 1.5 },
  error: { color: "#b00020", marginTop: 8, fontSize: 12 },
  hint: { marginTop: 10, fontSize: 11, color: "#666" },
};

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Popup />
  </StrictMode>,
);
