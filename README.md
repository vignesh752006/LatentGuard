# vdlm — visual-differential latent masking (hackathon prototype)

privacy-preserving browser agent for isro problem statement 26171.

**github:** https://github.com/prempalhade3-code/smart-privacy-agent

this repo is a **working demo prototype**. the full pipeline runs end-to-end, but several parts are **mocked or simplified** on purpose so you can demo without downloading huge ai models. read the "what is mocked" section before presenting to judges.

---

## what this demo actually does

```
your browser (chrome extension)              your laptop (python backend)
─────────────────────────────────            ─────────────────────────────
1. captures the visible tab locally          4. receives sanitized tensor V' only
2. scans page for sensitive field boxes      5. never gets screenshot / password text
3. ONNX SigLIP + local SVD sanitization ──►  6. mock planner picks an action
   sends V' over websocket                       (click / scroll / type)
7. executes click/scroll in the page      ◄──  8. sends action.directive back
```

**real today:** chrome extension architecture, tab capture, dom sensitivity scan, websocket protocol, tensor shape validation, stale-state checks, action execution in the browser.

**implemented today:** ONNX SigLIP is the default vision mode (FP16 WebGPU with FP32 WASM fallback); the procedural mock is available only as an explicit test mode. SVD null-space sanitization is the default (k=4), with a rank-1 baseline. The planner still uses keyword rules, and there is no cryptographic privacy proof.

---

## what you need on your laptop

| tool | version | why |
|------|---------|-----|
| git | any recent | clone the repo |
| python | 3.10 or 3.11 recommended | backend server |
| node.js | 18+ | build the chrome extension |
| google chrome | 121+ | extension + webgpu |
| rust + wasm-pack | optional | only if you want rust wasm patch mapper; **not required for demo** |

you do **not** need gpu drivers beyond what chrome already uses for webgpu.

---

## step 1 — clone the repo

```bash
git clone https://github.com/prempalhade3-code/smart-privacy-agent.git
cd smart-privacy-agent
```

if your friend already has the folder, just:

```bash
cd smart-privacy-agent
git pull
```

---

## step 2 — start the backend (terminal 1)

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate          # windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # default settings are fine for demo
PYTHONPATH=. python run.py
```

you should see uvicorn running on **port 8080**.

**check it works:**

```bash
curl http://127.0.0.1:8080/health
```

expected:

```json
{"service":"vdlm-reasoning-server","status":"ok", ...}
```

```bash
curl http://127.0.0.1:8080/ready
```

expected: `"model_provider": "mock"` and `"status": "ready"`.

leave this terminal open the whole demo.

---

## step 3 — build the chrome extension (terminal 2)

```bash
cd client
npm install
npm run build
```

notes:
- `npm run build:wasm` runs automatically. if rust/wasm-pack is **not** installed, the build still succeeds — it uses a typescript patch mapper instead. that's fine for the demo.
- after build, the loadable extension folder is **`client/dist/`**

**optional — run client tests:**

```bash
npm test
```

---

## step 4 — load the extension in chrome

1. open chrome and go to `chrome://extensions`
2. turn on **developer mode** (top right toggle)
3. click **load unpacked**
4. select the folder: `smart-privacy-agent/client/dist`
5. pin **VDLM Browser Agent** to the toolbar if you want

if you rebuild later (`npm run build`), come back here and click **reload** on the extension card.

---

## step 5 — run the live demo

### 5a. open a test page

open any normal website in a tab, for example:

```
https://example.com
```

click inside that tab so it is the **active tab**.

### 5b. open the extension popup

click the vdlm icon. you will see:

- backend websocket url (leave as `ws://127.0.0.1:8080/ws/v1/session`)
- task intent (text box)
- vision mode dropdown — **ONNX SigLIP** by default; mock is test-only
- start / stop buttons
- status section at the bottom

### 5c. set a task and start

try these task intents:

| task intent | expected backend action |
|-------------|-------------------------|
| `Click the Learn more link` | click at dominant patch coords |
| `Scroll down the page` | scroll down ~300px |
| `Type in the search box` | type `example_input` at a patch |
| `Navigate to https://example.com` | navigate action |

click **start**.

### 5d. what you should see (expected output)

**in the popup (within a few seconds):**

```
running: yes
connected: yes
stage: processing   (cycles through capturing / processing / transmitting)
session: <some uuid>
frames: 1, 2, 3 ... (number goes up every ~1.5s)
vision: onnx
```

**in the backend terminal:**

```
session_initialized ...
latent_frame_received ...
action_planned action=CLICK ...
```

(or scroll / type depending on your task intent)

**in the browser:**

- the page may scroll, or a click may fire at computed coordinates
- on example.com this is subtle — you mainly prove the **loop is alive**, not perfect ai targeting

### 5e. stop

click **stop** in the popup. `running` should go back to `no`.

---

## how to know everything is working

use this checklist:

| check | pass? |
|-------|-------|
| `curl http://127.0.0.1:8080/health` returns `"status":"ok"` | |
| extension loaded without errors in `chrome://extensions` | |
| popup shows `connected: yes` after start | |
| `frames` counter increases | |
| backend logs show `latent_frame_received` | |
| backend logs show `action_planned` | |
| no error line in popup | |

if `connected: no`:
- backend not running, or wrong port
- firewall blocking localhost

if `frames` stays at 0:
- webgpu not available — try updating chrome
- check extension errors: `chrome://extensions` → details → service worker → inspect

if backend shows `FORBIDDEN_FIELD`:
- good — security guard works; client tried to send something it shouldn't

---

## repo folder map

```
smart-privacy-agent/
├── README.md                 ← you are here
├── backend/                  ← python fastapi websocket server
│   ├── run.py                ← start with: PYTHONPATH=. python run.py
│   ├── app/models/mock_planner.py   ← keyword mock ai (default)
│   └── tests/                ← pytest suite
├── client/                   ← chrome mv3 extension
│   ├── dist/                 ← load THIS folder in chrome (after npm run build)
│   ├── src/
│   │   ├── background/       ← orchestration + websocket
│   │   ├── content/          ← dom scan + action execution
│   │   ├── offscreen/        ← webgpu vision + vdlm shaders
│   │   └── popup/            ← ui
│   └── public/shaders/       ← wgsl compute shaders
└── docs/
    ├── PROTOCOL.md           ← websocket message format
    └── ASSUMPTIONS.md        ← research claims vs what code actually proves
```

---

## what is real vs mocked (be honest in the demo)

### real (actually implemented)

- chrome manifest v3 extension with offscreen document
- visible tab capture stays on device — **raw screenshot is not sent to backend**
- dom scanner finds password/input field **bounding boxes** (geometry only, not text values)
- ONNX SigLIP runs locally (WebGPU when available, WASM otherwise); SVD sanitization runs locally on CPU
- source-grid PCA is the default payload representation (`196×64` int8 coefficients); standard full-tensor fp16 remains selectable
- backend rejects forbidden fields (screenshot, dom html, credentials)
- session uuid, frame ids, state_version sync
- stale plan detection when page state drifts
- structured actions: click, scroll, type, navigate, done
- content script executes actions in the real page

### mocked or simplified (do not claim these are production-ready)

| part | what the demo actually does |
|------|----------------------------|
| **vision encoder** | ONNX SigLIP runs by default; the procedural encoder is an explicit test mode |
| **onnx / siglip mode** | Requires local FP16, FP32 fallback, and adapter models; missing or invalid models raise an error and never silently use the mock |
| **VDLM sanitization** | CPU SVD null-space projection is the default (k=4); rank-1 is selectable as a baseline. This is not a formal privacy guarantee |
| **source PCA payload** | Default mode projects sanitized 196×768 SigLIP tokens; backend rebuilds the 576×1152 tensor. On 61 held-out pages, rank 64 averaged 15,934 B and 0.7596 patch cosine; rank 128 averaged 28,983 B and 0.8420. See `docs/ASSUMPTIONS.md` for limits and methodology |
| **backend planner** | `VDLM_MODEL_PROVIDER=mock` — keyword rules (`scroll` → scroll, `click` → click at dominant patch). **not** llama-3-70b reasoning |
| **llama provider** | code stub exists (`llama_planner.py`) but needs a separate openai-compatible api server + api key |
| **payload size** | Source-PCA is the client default; it reaches the size target only with measured reconstruction-fidelity loss recorded in `docs/ASSUMPTIONS.md` |
| **privacy guarantees** | no zero-knowledge proof, no cryptographic attestation; inversion results are benchmark-specific and are not a privacy guarantee |
| **wasm patch mapper** | rust source is in repo but wasm binary is **not** prebuilt — demo uses typescript fallback unless you run `npm run build:wasm` with rust installed |

---

## what is left to do (future work)

these are the main gaps between this prototype and the full research blueprint:

1. replace keyword mock planner with **visual grounding** or llama/vlm that uses page information
2. validate sanitizer utility against broader real-world tasks beyond the synthetic inversion benchmark
3. ship prebuilt **rust wasm** patch mapper in the default path
4. reduce payloads toward 45 KB without the measured fidelity loss
5. validate reconstruction utility/action accuracy after the inversion benchmark
6. improve latency toward the 350 ms target
7. cryptographic provenance / attestation (research hardening)

---

## configuration (usually you can ignore this)

### SigLIP ONNX encoder (default)

Install the export dependencies and create the local ONNX models:

```powershell
python -m pip install -e "backend[siglip-eval]"
python scripts/download-siglip-onnx.py --out-dir client/public/models
```

For the measured source-PCA basis, capture 20+ public pages and render the
controlled training pages before fitting:

```powershell
cd client
node scripts/capture-public-pca-pages.mjs
node scripts/render-inversion-pages.mjs --out-dir ..\backend\eval\siglip\fixtures\pca-pages --count 280 --seed 156017
cd ..
python backend/eval/siglip/fit_source_pca.py
```

The backend canonical basis lives at
`backend/app/encoding/data/source-pca-v1.bin`; the identical file is copied to
`client/public/pca/source-pca-v1.bin` for browser inference.

For the PowerShell wrapper use `scripts/download-siglip-onnx.ps1`; on
POSIX-compatible systems use `scripts/download-siglip-onnx.sh`. The wrappers
export the public `google/siglip-base-patch16-224` checkpoint. The popup
defaults to ONNX mode and the packaged FP16 model URL. ONNX mode tries the FP16 graph with WebGPU, then uses the
FP32 graph on WASM; either path reports a clear initialization error if its
required model or the adapter is missing or invalid. It never substitutes mock
embeddings. Local model files are ignored by git and are copied into `client/dist`
by Vite when present. The popup checks for missing model files and displays a
setup error; select **Mock (test mode)** only when intentionally testing the
procedural encoder. Source PCA payload mode requires the fitted basis. See
[backend/eval/siglip/README.md](backend/eval/siglip/README.md) for reference
fixtures, exact preprocessing, PCA fit, measured payload sizes, export details,
and parity/latency tests.

### Inversion benchmark (Task 5)

The full evaluation uses 500 Playwright-rendered pages, actual local SigLIP
ONNX embeddings, the exact source-PCA int8/zlib wire format and backend
reconstruction, and convolutional decoders:

```powershell
python -m pip install -e "backend[inversion-eval]"
python backend/eval/inversion/run_benchmark.py --n 500 --epochs 30
```

For a quick two-epoch smoke run, use `--n 10 --epochs 2 --skip-ocr`. Generated
pages and model files remain in ignored local run directories. The benchmark
writes `docs/EVAL_INVERSION.md`, machine-readable metrics, and PNG charts.

backend `.env`:

| variable | default | meaning |
|----------|---------|---------|
| `VDLM_MODEL_PROVIDER` | `mock` | keep as mock for hackathon demo |
| `VDLM_PORT` | `8080` | websocket + http port |

client popup settings are saved in chrome storage automatically when you click start.

---

## running tests (optional)

**backend:**

```bash
cd backend
source .venv/bin/activate
PYTHONPATH=. pytest tests/ -v
```

expect all tests to pass.

**client:**

```bash
cd client
npm test
```

---

## troubleshooting

**`pip install` fails on torch (too big / slow)**  
use python 3.10 or 3.11. on slow networks torch download takes time — wait or use a machine with better internet.

**`npm run build` fails on typescript errors**  
make sure you pulled latest main and ran `npm install` inside `client/`.

**popup says "no active tab"**  
click on the website tab first, then open the popup and hit start.

**popup says "open a normal website"**  
chrome blocks capture on `chrome://` pages. use `https://example.com` or any https site.

**webgpu errors in console**  
update chrome. try `chrome://gpu` and check webgpu status.

**backend connection refused**  
terminal 1 must be running. check port 8080 is not used by another app.

---

## quick demo script for judges (2 minutes)

1. show backend terminal already running on :8080
2. show extension loaded in chrome
3. open https://example.com
4. open popup → task: `Scroll down the page` → vision: ONNX (or choose Mock test mode if models are not installed) → **start**
5. point at popup: `connected: yes`, frames counting up
6. point at backend logs: `latent_frame_received`, `action_planned`
7. page scrolls (or click task if scroll is hard to see)
8. say clearly: *"ONNX mode runs SigLIP locally by default; the procedural mock is for explicit testing. SVD sanitization and PCA compression have measured behavior but no formal privacy guarantee, and planning remains a keyword-based prototype."*
9. click **stop**

---

## more docs

- websocket message format: [docs/PROTOCOL.md](docs/PROTOCOL.md)
- research claims vs verified behavior: [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md)
- client-only details: [client/README.md](client/README.md)
- backend-only details: [backend/README.md](backend/README.md)
