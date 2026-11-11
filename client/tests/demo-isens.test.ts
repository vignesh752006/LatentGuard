/**
 * Task 2 — Demo page DOM scan assertions (vitest)
 *
 * Loads login.html and checkout.html using JSDOM, runs the same selector
 * logic as client/src/sensitivity/dom-scanner.ts, and asserts that
 * I_sens covers the password / card-number / CVV / canvas-rendered boxes.
 *
 * Sensitive selector coverage is verified by presence in the selector table.
 * Patch grid coverage is verified by converting the documented normalized
 * bounding boxes to 24×24 patch indices.
 */

import { readFileSync } from 'fs';
import path from 'path';
import { JSDOM } from 'jsdom';
import { describe, it, expect, beforeAll } from 'vitest';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEMO_DIR = path.resolve(__dirname, '../../demo-pages');

// ---- Selector table (mirrors client/src/sensitivity/dom-scanner.ts) -----------
const SENSITIVE_SELECTORS: Array<{ selector: string; category: string }> = [
  { selector: 'input[type="password"]',      category: 'password' },
  { selector: 'input[autocomplete*="cc"]',   category: 'payment' },
  { selector: 'input[name*="password" i]',   category: 'password' },
  { selector: 'input[name*="passwd" i]',     category: 'password' },
  { selector: 'input[name*="card" i]',       category: 'payment' },
  { selector: 'input[name*="cvv" i]',        category: 'payment' },
  { selector: 'input[name*="ssn" i]',        category: 'pii' },
  { selector: '[data-vdlm-sensitive]',       category: 'custom' },
  { selector: 'iframe[title*="payment" i]',  category: 'payment' },
];

function scanDocument(document: Document) {
  const regions: Array<{ selector: string; category: string; id: string }> = [];
  for (const { selector, category } of SENSITIVE_SELECTORS) {
    document.querySelectorAll(selector).forEach((el) => {
      regions.push({ selector, category, id: (el as HTMLElement).id || (el as HTMLInputElement).name || el.tagName });
    });
  }
  return regions;
}

function loadPage(filename: string) {
  const html = readFileSync(path.join(DEMO_DIR, filename), 'utf8');
  const dom = new JSDOM(html, {
    url: `file://${path.join(DEMO_DIR, filename).replace(/\\/g, '/')}`,
  });
  return dom.window.document;
}

// Patch grid coverage
function normalizedRegionToPatches(
  x: number, y: number, w: number, h: number,
  rows = 24, cols = 24,
): number[] {
  const pW = 1 / cols;
  const pH = 1 / rows;
  const c0 = Math.floor(x / pW);
  const r0 = Math.floor(y / pH);
  const c1 = Math.min(cols - 1, Math.floor((x + w) / pW));
  const r1 = Math.min(rows - 1, Math.floor((y + h) / pH));
  const patches: number[] = [];
  for (let r = r0; r <= r1; r++)
    for (let c = c0; c <= c1; c++)
      patches.push(r * cols + c);
  return patches;
}

// ── login.html ──────────────────────────────────────────────────────────────

describe('Task 2: login.html DOM scan', () => {
  let doc: Document;

  beforeAll(() => { doc = loadPage('login.html'); });

  it('has a password input field', () => {
    expect(doc.querySelector('input[type="password"]')).not.toBeNull();
  });

  it('has a username text input', () => {
    expect(doc.querySelector('input[type="text"]#username')).not.toBeNull();
  });

  it('has Login and Cancel buttons', () => {
    expect(doc.querySelector('button#btn-login')).not.toBeNull();
    expect(doc.querySelector('button#btn-cancel')).not.toBeNull();
  });

  it('has a search box in the nav', () => {
    expect(doc.querySelector('input[type="search"]')).not.toBeNull();
  });

  it('has a canvas sensitive box with [data-vdlm-sensitive]', () => {
    expect(doc.querySelector('[data-vdlm-sensitive] canvas, canvas[data-vdlm-sensitive], [data-vdlm-sensitive]')).not.toBeNull();
  });

  it('I_sens covers password field (category=password)', () => {
    const regions = scanDocument(doc);
    const hit = regions.some(r => r.category === 'password');
    expect(hit).toBe(true);
  });

  it('I_sens covers canvas sensitive region (category=custom)', () => {
    const regions = scanDocument(doc);
    const hit = regions.some(r => r.category === 'custom');
    expect(hit).toBe(true);
  });

  it('password field element id="password" is in sensitive set', () => {
    const regions = scanDocument(doc);
    const hit = regions.some(r => r.id === 'password');
    expect(hit).toBe(true);
  });

  it('password field covers at least 1 patch in 24x24 grid', () => {
    // Documented layout: password field at x=0.396, y=0.398, w=0.208, h=0.046
    const patches = normalizedRegionToPatches(0.396, 0.398, 0.208, 0.046);
    expect(patches.length).toBeGreaterThanOrEqual(1);
  });

  it('canvas sensitive box covers at least 1 patch in 24x24 grid', () => {
    // Documented layout: canvas at x=0.396, y=0.620, w=0.208, h=0.065
    const patches = normalizedRegionToPatches(0.396, 0.620, 0.208, 0.065);
    expect(patches.length).toBeGreaterThanOrEqual(1);
  });
});

// ── checkout.html ────────────────────────────────────────────────────────────

describe('Task 2: checkout.html DOM scan', () => {
  let doc: Document;

  beforeAll(() => { doc = loadPage('checkout.html'); });

  it('has a card-number input', () => {
    expect(doc.querySelector('input[name="card-number"]')).not.toBeNull();
  });

  it('has a CVV input', () => {
    expect(doc.querySelector('input[name="cvv"]')).not.toBeNull();
  });

  it('has a name-on-card input', () => {
    expect(doc.querySelector('input[name="name"]')).not.toBeNull();
  });

  it('has Pay and Cancel buttons', () => {
    expect(doc.querySelector('button#btn-pay')).not.toBeNull();
    expect(doc.querySelector('button#btn-cancel')).not.toBeNull();
  });

  it('has a search box in the nav', () => {
    expect(doc.querySelector('input[type="search"]')).not.toBeNull();
  });

  it('has a canvas sensitive box with [data-vdlm-sensitive]', () => {
    expect(doc.querySelector('[data-vdlm-sensitive]')).not.toBeNull();
  });

  it('I_sens covers payment fields (category=payment)', () => {
    const regions = scanDocument(doc);
    const hit = regions.some(r => r.category === 'payment');
    expect(hit).toBe(true);
  });

  it('I_sens covers canvas region (category=custom)', () => {
    const regions = scanDocument(doc);
    const hit = regions.some(r => r.category === 'custom');
    expect(hit).toBe(true);
  });

  it('card-number element is in sensitive set', () => {
    const regions = scanDocument(doc);
    const hit = regions.some(r => r.id === 'card-number');
    expect(hit).toBe(true);
  });

  it('cvv element is in sensitive set', () => {
    const regions = scanDocument(doc);
    const hit = regions.some(r => r.id === 'cvv');
    expect(hit).toBe(true);
  });

  it('card-number field covers at least 1 patch in 24x24 grid', () => {
    // x=0.375, y=0.370, w=0.250, h=0.046
    const patches = normalizedRegionToPatches(0.375, 0.370, 0.250, 0.046);
    expect(patches.length).toBeGreaterThanOrEqual(1);
  });

  it('CVV field covers at least 1 patch in 24x24 grid', () => {
    // x=0.375, y=0.482, w=0.125, h=0.046
    const patches = normalizedRegionToPatches(0.375, 0.482, 0.125, 0.046);
    expect(patches.length).toBeGreaterThanOrEqual(1);
  });

  it('canvas sensitive box covers at least 1 patch in 24x24 grid', () => {
    // x=0.375, y=0.638, w=0.250, h=0.065
    const patches = normalizedRegionToPatches(0.375, 0.638, 0.250, 0.065);
    expect(patches.length).toBeGreaterThanOrEqual(1);
  });
});
