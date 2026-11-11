# Latency & Utility Benchmarks — BENCHMARKS.md

> [!NOTE]
> This document tracks per-stage latency and task utility metrics for VDLM.
> Numbers marked **PENDING** require a live browser session with the full pipeline running.
> The timing instrumentation is wired into `OffscreenAgentRunner.runCycle()` and reports
> `p50`/`p95` via `computeLatencyPercentiles()` exported from `agent-runner.ts`.

---

## 1. Latency Benchmark

### 1.1 Target

| Metric | Target |
|--------|--------|
| End-to-end (capture → execute) | < 350 ms |

### 1.2 Per-Stage Timings (p50 / p95)

> [!CAUTION]
> Numbers below are **PENDING** real measurements. The instrumentation is live —
> run the extension on a local page and read `computeLatencyPercentiles()` to fill in.

| Stage | Description | p50 (ms) | p95 (ms) |
|-------|-------------|----------|----------|
| `scan` | DOM sensitivity scan + state version relay | PENDING | PENDING |
| `capture` | JPEG tab capture via chrome.tabCapture | PENDING | PENDING |
| `decode` | JPEG → ImageData decode | PENDING | PENDING |
| `encode` | SigLIP mock/ONNX encode (224×224 → 576×1152) | PENDING | PENDING |
| `sanitize` | SVD null-space sanitization (k=1 default) | PENDING | PENDING |
| `serialize` | fp16/int8 quantize + zlib compress + base64 | PENDING | PENDING |
| `network_send` | WebSocket send + backend plan + action.directive received | PENDING | PENDING |
| `execute` | Action execution (CLICK/TYPE/SCROLL) | PENDING | PENDING |
| **Total** | **End-to-end** | **PENDING** | **PENDING** |

### 1.3 How to Measure

```typescript
// In the browser console (offscreen context), after several frames:
import { computeLatencyPercentiles, getFrameTimings } from "./offscreen/agent-runner";

const total = computeLatencyPercentiles("totalMs");
console.table(total);
// { p50: ..., p95: ..., count: N }

const encodeP = computeLatencyPercentiles("encode");
console.table(encodeP);
```

---

## 2. Utility Benchmark

### 2.1 Methodology

Measure click accuracy on the demo pages (login.html, checkout.html) with:
- **Unsanitized V**: raw SigLIP embeddings sent to planner.
- **Sanitized V'**: SVD null-space sanitized embeddings (k=1, k=4, k=8).

**Task set (manual / Mind2Web-style):**

| Task ID | Page | Task Description |
|---------|------|------------------|
| U1 | login.html | "Click the Login button" |
| U2 | login.html | "Fill the username field" |
| U3 | login.html | "Click the Cancel button" |
| U4 | checkout.html | "Click the Pay button" |
| U5 | checkout.html | "Click the Cancel button" |
| U6 | checkout.html | "Fill the name on card field" |

### 2.2 Accuracy Results

> [!CAUTION]
> Numbers below are **PENDING** real benchmark runs. The mock planner always returns
> CLICK at the highest-score patch, so accuracy depends on the score_grid correctly
> identifying the target element.

| Sanitizer | Correct Clicks / Total | Accuracy |
|-----------|------------------------|----------|
| None (V raw) | PENDING | PENDING |
| SVD k=1 (V') | PENDING | PENDING |
| SVD k=4 (V') | PENDING | PENDING |
| SVD k=8 (V') | PENDING | PENDING |

**Target:** Utility parity ≥ 85% (sanitized vs. unsanitized) for non-sensitive tasks.

---

## 3. Measured Payload Sizes

These are verified by CI tests:

| Mode | Wire bytes | Compression ratio |
|------|-----------|------------------|
| fp16 uncompressed | 1,327,104 | 1.00× |
| fp16 + zlib (zeros) | ~1,308 | ~1015× |
| fp16 + zlib (random) | ~3,259 | ~407× |
| int8 + zlib (structured) | ~12,238 | ~108× |
| int8 + zlib (top-10% sparse, random) | ~123,375 | ~11× |

---

## 4. Honest Conclusions

**What is measured today:**
- Payload sizes under different compression modes (CI-verified, see Task 1 test output).
- SVD sanitizer MSE/cosine-similarity on test embeddings (CI-verified, see Task 4 test output).
- Stage-timing instrumentation is wired (Task 8) but requires a live run to produce numbers.

**What is NOT yet measured:**
- Real end-to-end latency (requires a Chrome extension session with backend).
- Utility accuracy (requires running tasks on demo pages with the full pipeline).
- ONNX encode latency (no real SigLIP ONNX model in CI — download script provided).

These will be filled in when a full benchmark run is performed and results verified.

