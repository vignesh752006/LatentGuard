import { chromium } from "@playwright/test";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const defaultOutput = path.resolve(
  scriptDir,
  "..",
  "..",
  "backend",
  "eval",
  "siglip",
  "fixtures",
  "public-pages",
);
const outputDir = path.resolve(process.argv[2] ?? defaultOutput);
const urls = [
  "https://example.com/",
  "https://www.wikipedia.org/",
  "https://en.wikipedia.org/wiki/Main_Page",
  "https://www.python.org/",
  "https://docs.python.org/3/",
  "https://developer.mozilla.org/en-US/",
  "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
  "https://www.w3.org/",
  "https://www.gnu.org/",
  "https://www.rust-lang.org/",
  "https://nodejs.org/en",
  "https://www.typescriptlang.org/",
  "https://react.dev/",
  "https://vite.dev/",
  "https://www.nasa.gov/",
  "https://science.nasa.gov/",
  "https://www.noaa.gov/",
  "https://www.nps.gov/",
  "https://www.loc.gov/",
  "https://www.si.edu/",
  "https://www.metmuseum.org/",
  "https://docs.github.com/",
  "https://support.google.com/",
  "https://www.openstreetmap.org/",
  "https://archive.org/",
];

await mkdir(outputDir, { recursive: true });
const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({
  viewport: { width: 1280, height: 720 },
  deviceScaleFactor: 1,
  ignoreHTTPSErrors: true,
});
const results = [];

try {
  for (let index = 0; index < urls.length; index++) {
    const url = urls[index];
    const page = await context.newPage();
    try {
      const response = await page.goto(url, {
        waitUntil: "domcontentloaded",
        timeout: 12000,
      });
      await page.waitForTimeout(1200);
      if (!response?.ok()) {
        throw new Error(`HTTP ${response?.status() ?? "no response"}`);
      }
      const filename = `public-${String(index + 1).padStart(2, "0")}.png`;
      await page.screenshot({ path: path.join(outputDir, filename) });
      results.push({ url, title: await page.title(), filename });
      console.log(`[capture] ${results.length}/${urls.length} ${url}`);
    } catch (error) {
      console.warn(`[capture] skipped ${url}: ${error.message}`);
    } finally {
      await page.close();
    }
  }
} finally {
  await context.close();
  await browser.close();
}

if (results.length < 20) {
  throw new Error(`Captured ${results.length} pages; at least 20 are required`);
}

await writeFile(
  path.join(outputDir, "manifest.json"),
  `${JSON.stringify({ captured_at: new Date().toISOString(), viewport: [1280, 720], pages: results }, null, 2)}\n`,
);
console.log(`[capture] saved ${results.length} public page screenshots to ${outputDir}`);
