import { chromium } from "@playwright/test";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const args = new Map(
  process.argv.slice(2).reduce((pairs, value, index, values) => {
    if (value.startsWith("--")) pairs.push([value.slice(2), values[index + 1]]);
    return pairs;
  }, []),
);
const outputDir = path.resolve(args.get("out-dir") ?? "backend/eval/inversion/runs/pages");
const count = Number(args.get("count") ?? 500);
const seed = Number(args.get("seed") ?? 26171);
if (!Number.isInteger(count) || count < 1) throw new Error("--count must be a positive integer");

function randomFactory(start) {
  let state = start >>> 0;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 0x100000000;
  };
}

function digits(rng, length) {
  return Array.from({ length }, () => Math.floor(rng() * 10)).join("");
}

function mapBoxesToPatches(boxes, rows, columns, width, height) {
  const patchW = Math.max(1, Math.ceil(width / columns));
  const patchH = Math.max(1, Math.ceil(height / rows));
  const result = new Set();
  for (const { x, y, width: boxWidth, height: boxHeight } of boxes) {
    const colStart = Math.min(columns - 1, Math.floor(x / patchW));
    const colEnd = Math.min(columns - 1, Math.floor((x + boxWidth) / patchW));
    const rowStart = Math.min(rows - 1, Math.floor(y / patchH));
    const rowEnd = Math.min(rows - 1, Math.floor((y + boxHeight) / patchH));
    for (let row = rowStart; row <= rowEnd; row++) {
      for (let col = colStart; col <= colEnd; col++) result.add(row * columns + col);
    }
  }
  return [...result].sort((a, b) => a - b);
}

function syntheticHtml(index, rng, password, cardNumber, cvv) {
  const hue = Math.floor(rng() * 360);
  const font = ["Arial", "Georgia", "Verdana", "Tahoma"][Math.floor(rng() * 4)];
  const layout = Math.floor(rng() * 3);
  const width = [560, 640, 720][layout];
  const left = [24, 48, 96][Math.floor(rng() * 3)];
  return `<!doctype html>
  <html><head><meta charset="utf-8"><style>
    *{box-sizing:border-box} body{margin:0;width:1280px;height:720px;font-family:${font},sans-serif;
      color:#182337;background:hsl(${hue} 24% 95%)}
    header{height:76px;background:white;border-bottom:1px solid #d6deea;padding:24px 40px;font-size:23px}
    main{display:grid;grid-template-columns:1fr 250px;gap:24px;margin:32px ${left}px}
    section,.aside{background:white;border:1px solid #d8deea;border-radius:12px;padding:24px}
    h1{font-size:23px;margin:0 0 8px} p{margin:0 0 20px;color:#64748b;font-size:14px}
    label{display:block;font-size:13px;font-weight:600;margin:12px 0 5px}
    input{display:block;width:100%;height:40px;border:1px solid #bac5d4;border-radius:6px;
      padding:0 11px;font-size:16px;background:#fff;color:#111827}
    .row{display:grid;grid-template-columns:1fr 120px;gap:12px}
    button{margin-top:18px;border:0;border-radius:6px;background:hsl(${hue} 65% 42%);
      color:white;padding:11px 22px;font-size:14px}
    canvas{display:block;width:100%;height:76px;margin-top:18px;background:#f8fafc;border:1px solid #d1d9e5}
    .aside{height:230px}.line{height:8px;background:#e4e9f0;border-radius:5px;margin:14px 0}
    .search{float:right;width:200px;height:34px}
  </style></head><body>
    <header>Northstar ${["Account", "Membership", "Orders", "Workspace"][index % 4]}
      <input class="search" id="search" type="search" placeholder="Search">
    </header><main><section><h1>${["Sign in", "Review your order", "Confirm account", "Secure checkout"][index % 4]}</h1>
    <p>Page ${String(index + 1).padStart(4, "0")} · verify your information to continue</p>
    <label for="name-on-card">Name</label><input id="name-on-card" name="name-on-card" value="Taylor Sample ${index}">
    <label for="password">Password</label><input id="password" name="password" type="password" value="${password}">
    <label for="card-number">Card number</label><input id="card-number" name="card-number" autocomplete="cc-number" value="${cardNumber}">
    <div class="row"><div><label for="cvv">CVV</label><input id="cvv" name="cvv" autocomplete="cc-csc" value="${cvv}"></div>
    <div><label for="expiry">Expiry</label><input id="expiry" value="12/29"></div></div>
    <canvas id="sensitive-box" data-vdlm-sensitive width="570" height="76" aria-label="Sensitive box"></canvas>
    <button id="btn-pay">Continue</button><button id="btn-cancel" style="background:#64748b">Cancel</button>
    </section><div class="aside"><div class="line" style="width:60%"></div><div class="line"></div>
    <div class="line" style="width:75%"></div></div></main><script>
      const c=document.querySelector("canvas"),x=c.getContext("2d");x.fillStyle="#fecaca";x.fillRect(8,8,90,60);
      x.fillStyle="#7f1d1d";x.font="bold 14px sans-serif";x.fillText("Sensitive",18,42);
    </script></body></html>`;
}

await mkdir(outputDir, { recursive: true });
const browser = await chromium.launch({ headless: true });
try {
  const context = await browser.newContext({
    viewport: { width: 1280, height: 720 },
    deviceScaleFactor: 1,
  });
  try {
    for (let index = 0; index < count; index++) {
      const rng = randomFactory(seed + index * 0x9e3779b9);
      const password = `Vdlm-${digits(rng, 4)}-${String.fromCharCode(65 + Math.floor(rng() * 26))}${digits(rng, 3)}`;
      const cardNumber = digits(rng, 16);
      const cvv = digits(rng, 3);
      const page = await context.newPage();
      await page.setContent(syntheticHtml(index, rng, password, cardNumber, cvv), {
        waitUntil: "load",
      });
      await page.screenshot({
        path: path.join(outputDir, `sample_${String(index).padStart(4, "0")}.png`),
        animations: "disabled",
      });
      const sensitiveBoxes = await page.evaluate(() =>
        ["#password", "#card-number", "#cvv", "#name-on-card", "#sensitive-box"].map((selector) => {
          const rect = document.querySelector(selector).getBoundingClientRect();
          return { x: rect.x, y: rect.y, width: rect.width, height: rect.height, selector };
        }),
      );
      const sensitivePatchIndices = mapBoxesToPatches(sensitiveBoxes, 24, 24, 1280, 720);
      const sensitiveSourcePatchIndices = [...new Set(sensitivePatchIndices.map((index) => {
        const row = Math.floor(index / 24);
        const col = index % 24;
        const sourceRow = Math.min(13, Math.floor((row + 0.5) * 14 / 24));
        const sourceCol = Math.min(13, Math.floor((col + 0.5) * 14 / 24));
        return sourceRow * 14 + sourceCol;
      }))].sort((a, b) => a - b);
      await writeFile(
        path.join(outputDir, `sample_${String(index).padStart(4, "0")}.json`),
        `${JSON.stringify({
          index,
          seed: seed + index * 0x9e3779b9,
          password,
          card_number: cardNumber,
          cvv,
          viewport: [1280, 720],
          sensitive_boxes: sensitiveBoxes,
          sensitive_patch_indices: sensitivePatchIndices,
          sensitive_source_patch_indices: sensitiveSourcePatchIndices,
        })}\n`,
      );
      await page.close();
      if ((index + 1) % 50 === 0 || index + 1 === count) {
        console.log(`[render] ${index + 1}/${count}`);
      }
    }
  } finally {
    await context.close();
  }
} finally {
  await browser.close();
}
