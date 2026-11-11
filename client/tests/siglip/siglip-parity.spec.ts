import { expect, test, type Page } from "@playwright/test";
import { existsSync, readFileSync, statSync } from "node:fs";
import path from "node:path";

const fixtureRoot = path.resolve(process.cwd(), "..", "backend", "eval", "siglip", "fixtures");
const imageNames = [
  "login",
  "checkout",
  "natural-iss073e0763824",
  "natural-iss009e05944",
  "natural-iss009e05953",
];
const modelDir = path.resolve(process.cwd(), "public", "models");
const modelPath = path.join(modelDir, "siglip-base-patch16-224-vision-fp16.onnx");
const wasmModelPath = path.join(modelDir, "siglip-base-patch16-224-vision-fp32.onnx");
const adapterPath = path.join(modelDir, "siglip-base-patch16-224-adapter.onnx");

function readNpyFloat32(filePath: string): Float32Array {
  const file = readFileSync(filePath);
  if (file[0] !== 0x93 || file.toString("ascii", 1, 6) !== "NUMPY") {
    throw new Error(`Not an NPY file: ${filePath}`);
  }
  const major = file[6]!;
  const headerLength = major === 1 ? file.readUInt16LE(8) : file.readUInt32LE(8);
  const headerStart = major === 1 ? 10 : 12;
  const header = file.toString("ascii", headerStart, headerStart + headerLength);
  if (!header.includes("'descr': '<f4'") || !header.includes("(196, 768)")) {
    throw new Error(`Unexpected reference fixture header: ${header}`);
  }
  const dataStart = headerStart + headerLength;
  const values = new Float32Array(196 * 768);
  for (let index = 0; index < values.length; index++) {
    values[index] = file.readFloatLE(dataStart + index * 4);
  }
  return values;
}

async function encodeFixture(
  page: Page,
  name: string,
  provider: "webgpu" | "wasm",
  measureLatency: boolean,
) {
  const extension = name.startsWith("natural-") ? ".jpg" : ".png";
  const source = readFileSync(path.join(fixtureRoot, "images", `${name}${extension}`));
  const mimeType = extension === ".jpg" ? "image/jpeg" : "image/png";
  const imageUrl = `data:${mimeType};base64,${source.toString("base64")}`;
  const reference = readNpyFloat32(path.join(fixtureRoot, "reference", `${name}.npy`));
  return page.evaluate(
    async ({ providerName, dataUrl, referenceValues, shouldMeasure }) =>
      window.encodeSiglip(providerName, dataUrl, referenceValues, shouldMeasure),
    {
      providerName: provider,
      dataUrl: imageUrl,
      referenceValues: Array.from(reference),
      shouldMeasure: measureLatency,
    },
  );
}

test("SigLIP ONNX matches five PyTorch reference fixtures and reports provider timings", async ({ page }) => {
  if (!existsSync(path.join(fixtureRoot, "fixture-metadata.json"))) {
    throw new Error("Missing SigLIP references; run backend/eval/siglip/reference.py --write-fixtures");
  }
  if (!existsSync(modelPath) || !existsSync(wasmModelPath) || !existsSync(adapterPath)) {
    throw new Error("Missing ONNX model files; run scripts/download-siglip-onnx.py");
  }
  if (!existsSync(path.resolve(process.cwd(), "public", "pca", "source-pca-v1.bin"))) {
    throw new Error("Missing source PCA basis; run backend/eval/siglip/fit_source_pca.py");
  }

  page.on("console", (message) => console.log(`BROWSER_${message.type()}: ${message.text()}`));
  page.on("pageerror", (error) => console.error(`BROWSER_PAGE_ERROR: ${error.message}`));
  page.on("requestfailed", (request) => console.error(`BROWSER_REQUEST_FAILED: ${request.url()} ${request.failure()?.errorText}`));
  page.on("response", (response) => {
    if (response.status() >= 400) console.error(`BROWSER_HTTP_${response.status()}: ${response.url()}`);
  });
  await page.goto("/tests/siglip/harness.html");
  await page.waitForFunction(() => typeof window.verifySyntheticDeterminism === "function", null, { timeout: 10_000 });
  expect(await page.evaluate(() => window.verifySyntheticDeterminism(
    "/models/siglip-base-patch16-224-vision-fp16.onnx",
  ))).toBe(true);

  const scores: Record<string, number> = {};
  const latencies: Record<string, { p50: number; p95: number }> = {};
  const wireBytes: Record<string, Awaited<ReturnType<typeof encodeFixture>>["wireBytes"]> = {};

  for (const provider of ["wasm", "webgpu"] as const) {
    if (provider === "webgpu") {
      const hasAdapter = await page.evaluate(async () => {
        return Boolean(navigator.gpu && (await navigator.gpu.requestAdapter()));
      });
      if (!hasAdapter) {
        console.warn("WEBGPU_UNAVAILABLE: browser exposes no WebGPU adapter");
        continue;
      }
    }

    for (const name of imageNames) {
      const result = await encodeFixture(page, name, provider, name === "login");
      expect(result.outputShape).toEqual([576, 1152]);
      scores[`${provider}:${name}`] = result.cosine;
      expect(result.cosine, `${provider} cosine for ${name}`).toBeGreaterThanOrEqual(0.99);
      wireBytes[`${provider}:${name}`] = result.wireBytes;
      if (result.latencyMs) latencies[provider] = result.latencyMs;
    }
  }

  const loginImage = readFileSync(path.join(fixtureRoot, "images", "login.png"));
  const loginUrl = `data:image/png;base64,${loginImage.toString("base64")}`;
  const svdResult = await page.evaluate(
    async ({ imageUrl, sensitiveIndices }) =>
      window.sanitizeSiglipEmbedding(imageUrl, sensitiveIndices),
    { imageUrl: loginUrl, sensitiveIndices: [1, 2, 24, 25, 26, 48, 49, 50] },
  );
  expect(svdResult.outputShape).toEqual([576, 1152]);
  expect(svdResult.privateRank).toBeGreaterThan(0);
  expect(svdResult.maxAbsDot).toBeLessThan(1e-4);

  const pcaSummaries: Record<string, {
    compressedBytes: number;
    base64Bytes: number;
    frameJsonBytes: number;
    meanPatchCosineVsSanitized: number;
    minimumPatchCosineVsSanitized: number;
    meanPatchCosineVsFull: number;
  }> = {};
  for (const name of imageNames) {
    const extension = name.startsWith("natural-") ? ".jpg" : ".png";
    const image = readFileSync(path.join(fixtureRoot, "images", `${name}${extension}`));
    const mimeType = extension === ".jpg" ? "image/jpeg" : "image/png";
    const result = await page.evaluate(
      (imageUrl) => window.measureSourcePca(imageUrl),
      `data:${mimeType};base64,${image.toString("base64")}`,
    );
    expect(result.frameJsonBytes).toBeLessThan(45_056);
    pcaSummaries[name] = result;
  }

  await page.evaluate(() => window.closeSiglipEncoders());
  console.info("SIGLIP_PARITY_AND_LATENCY", JSON.stringify({
    scores,
    latencies,
    wireBytes,
    realSiglipSvd: svdResult,
    sourcePca: pcaSummaries,
    modelBytes: {
      visionFp16: statSync(modelPath).size,
      visionWasmFp32: statSync(wasmModelPath).size,
      adapter: statSync(adapterPath).size,
    },
  }));
});
