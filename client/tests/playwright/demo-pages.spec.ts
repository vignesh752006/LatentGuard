import { expect, test, type Locator } from "@playwright/test";

import type { SensitiveRegion } from "../../src/messaging/types";
import { mapRegionsToPatchesTs } from "../../src/sensitivity/patch-mapper";

const VIEWPORT = { width: 1280, height: 720 };

async function injectScanner(page: import("@playwright/test").Page): Promise<void> {
  await page.addScriptTag({
    type: "module",
    url: "/tests/playwright/inject-scanner.ts",
  });
}

async function expectedRegion(
  locator: Locator,
  category: SensitiveRegion["category"],
): Promise<SensitiveRegion> {
  const box = await locator.boundingBox();
  if (!box) throw new Error(`Expected visible element for category ${category}`);
  return {
    x: box.x / VIEWPORT.width,
    y: box.y / VIEWPORT.height,
    width: box.width / VIEWPORT.width,
    height: box.height / VIEWPORT.height,
    category,
  };
}

function sortedRegions(regions: SensitiveRegion[]): SensitiveRegion[] {
  return [...regions].sort(
    (a, b) =>
      a.category.localeCompare(b.category) ||
      a.y - b.y ||
      a.x - b.x ||
      a.width - b.width ||
      a.height - b.height,
  );
}

async function assertExactSensitiveRegions(
  page: import("@playwright/test").Page,
  expected: SensitiveRegion[],
  fields: SensitiveRegion[],
  label: string,
): Promise<void> {
  const scan = await page.evaluate(() => {
    if (!window.__vdlmRunSensitiveScan) throw new Error("Sensitive scanner was not injected");
    return window.__vdlmRunSensitiveScan();
  });

  expect(scan.viewport).toEqual(VIEWPORT);
  expect(sortedRegions(scan.regions)).toEqual(sortedRegions(expected));

  const expectedIndices = mapRegionsToPatchesTs(VIEWPORT.width, VIEWPORT.height, expected);
  const fieldIndices = mapRegionsToPatchesTs(VIEWPORT.width, VIEWPORT.height, fields);
  expect(scan.indices).toEqual(expectedIndices);
  expect(scan.indices).toEqual(expect.arrayContaining(fieldIndices));

  console.info(
    `[Task 2] ${label} viewport=${scan.viewport.width}x${scan.viewport.height}; ` +
      `sensitive regions=${scan.regions.length}; I_sens patches=${scan.indices.length}; ` +
      `indices=${scan.indices.join(",")}`,
  );
}

test.use({ viewport: VIEWPORT });

test("login page maps exactly the visible password and canvas regions", async ({ page }) => {
  await page.goto("/demo-pages/login.html");
  await injectScanner(page);

  await expect(page.locator("#password")).toBeVisible();
  await expect(page.locator("#search")).toBeVisible();
  await expect(page.locator("#btn-login")).toBeVisible();
  await expect(page.locator("#btn-cancel")).toBeVisible();
  await expect(page.locator("#sensitive-box")).toBeVisible();

  const canvasHasVisibleDrawing = await page.locator("#sensitive-box").evaluate((element) => {
    const canvas = element as HTMLCanvasElement;
    const pixel = canvas.getContext("2d")?.getImageData(4, 4, 1, 1).data;
    return Boolean(pixel && (pixel[0] !== 255 || pixel[1] !== 255 || pixel[2] !== 255));
  });
  expect(canvasHasVisibleDrawing).toBe(true);

  const password = await expectedRegion(page.locator("#password"), "password");
  const canvas = await expectedRegion(page.locator("[data-vdlm-sensitive]"), "custom");
  const canvasRole = await expectedRegion(page.locator("#sensitive-box"), "payment");
  await assertExactSensitiveRegions(page, [password, canvas, canvasRole], [password], "login");
});

test("checkout page maps exactly the visible payment fields and canvas region", async ({ page }) => {
  await page.goto("/demo-pages/checkout.html");
  await injectScanner(page);

  for (const selector of ["#card-number", "#cvv", "#name-on-card", "#search", "#btn-pay", "#btn-cancel", "#sensitive-box"]) {
    await expect(page.locator(selector)).toBeVisible();
  }

  const card = await expectedRegion(page.locator("#card-number"), "payment");
  const cvvPayment = await expectedRegion(page.locator("#cvv"), "payment");
  const cvvPassword = await expectedRegion(page.locator("#cvv"), "password");
  const name = await expectedRegion(page.locator("#name-on-card"), "payment");
  const canvas = await expectedRegion(page.locator("[data-vdlm-sensitive]"), "custom");
  const canvasRole = await expectedRegion(page.locator("#sensitive-box"), "payment");
  await assertExactSensitiveRegions(
    page,
    [card, cvvPayment, cvvPassword, name, canvas, canvasRole],
    [card, cvvPayment, cvvPassword, name],
    "checkout",
  );
});

test("page without sensitive fields produces an empty I_sens", async ({ page }) => {
  await page.goto("/demo-pages/login.html");
  await page.setContent(
    '<main><label>Search <input type="search" placeholder="Query"></label><p>Ordinary public content</p></main>',
  );
  await injectScanner(page);

  const scan = await page.evaluate(() => {
    if (!window.__vdlmRunSensitiveScan) throw new Error("Sensitive scanner was not injected");
    return window.__vdlmRunSensitiveScan();
  });
  expect(scan.viewport).toEqual(VIEWPORT);
  expect(scan.regions).toEqual([]);
  expect(scan.indices).toEqual([]);
});
