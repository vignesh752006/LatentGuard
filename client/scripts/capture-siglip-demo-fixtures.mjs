import { chromium } from "@playwright/test";
import { mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const clientRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const fixtureDir = path.resolve(clientRoot, "..", "backend", "eval", "siglip", "fixtures", "images");
await mkdir(fixtureDir, { recursive: true });

const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 }, deviceScaleFactor: 1 });
  for (const pageName of ["login", "checkout"]) {
    const source = path.join(clientRoot, "demo-pages", `${pageName}.html`);
    await page.goto(pathToFileURL(source).href, { waitUntil: "load" });
    await page.screenshot({
      path: path.join(fixtureDir, `${pageName}.png`),
      animations: "disabled",
      fullPage: false,
    });
  }
} finally {
  await browser.close();
}
