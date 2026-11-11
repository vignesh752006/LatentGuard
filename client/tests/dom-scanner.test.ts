/**
 * Task 9 — Extended DOM scanner unit tests.
 *
 * jsdom does not compute CSS layout, so getBoundingClientRect() always returns
 * zero for all elements. We mock it at the prototype level to return a non-zero
 * rect for any element that has a data-testid or is of a detectable kind,
 * then verify the scanner classifies them correctly.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { scanSensitiveRegions } from "../src/sensitivity/dom-scanner";

/** Mock all elements to report a 100×40 rect at (10, 10). */
function mockBoundingRects(): void {
  vi.spyOn(Element.prototype, "getBoundingClientRect").mockReturnValue({
    left: 10,
    top: 10,
    right: 110,
    bottom: 50,
    width: 100,
    height: 40,
    x: 10,
    y: 10,
    toJSON() {
      return this;
    },
  } as DOMRect);
  // Also mock window.getComputedStyle so elements are not filtered as hidden.
  vi.spyOn(window, "getComputedStyle").mockReturnValue({
    display: "block",
    visibility: "visible",
  } as unknown as CSSStyleDeclaration);
}

function setBody(html: string): void {
  document.body.innerHTML = html;
}

// Ensure window dimensions are set for normalisation math.
Object.defineProperty(window, "innerWidth", { value: 1024, configurable: true });
Object.defineProperty(window, "innerHeight", { value: 768, configurable: true });

describe("Task 9: Extended DOM scanner", () => {
  beforeEach(() => {
    document.body.innerHTML = "";
    mockBoundingRects();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  // ---- Original selector-based detection -----------------------------------

  it("detects password input by type", () => {
    setBody('<input type="password" />');
    const regions = scanSensitiveRegions();
    expect(regions.some((r) => r.category === "password")).toBe(true);
  });

  it("detects input[name*=card]", () => {
    setBody('<input name="card-number" />');
    const regions = scanSensitiveRegions();
    expect(regions.some((r) => r.category === "payment")).toBe(true);
  });

  it("detects [data-vdlm-sensitive]", () => {
    setBody('<div data-vdlm-sensitive></div>');
    const regions = scanSensitiveRegions();
    expect(regions.some((r) => r.category === "custom")).toBe(true);
  });

  it("detects input[name*=cvv]", () => {
    setBody('<input name="cvv" />');
    const regions = scanSensitiveRegions();
    expect(regions.some((r) => r.category === "payment")).toBe(true);
  });

  // ---- Task 9: contenteditable detection ----------------------------------

  it("detects contenteditable with sensitive aria-label (card)", () => {
    setBody('<div contenteditable="true" aria-label="Enter card number"></div>');
    const regions = scanSensitiveRegions();
    expect(regions.some((r) => r.category === "pii" || r.category === "payment")).toBe(true);
  });

  it("detects contenteditable with sensitive aria-label (password)", () => {
    setBody('<div contenteditable="true" aria-label="Enter password"></div>');
    const regions = scanSensitiveRegions();
    expect(regions.some((r) => r.category === "pii" || r.category === "password")).toBe(true);
  });

  it("does NOT flag generic contenteditable without sensitive label", () => {
    setBody('<div contenteditable="true" aria-label="Your comment"></div>');
    const regions = scanSensitiveRegions();
    // Only contenteditable is on the page — none should match (non-sensitive label)
    const fromContenteditable = regions.filter(
      (r) => r.category === "pii" || r.category === "password",
    );
    expect(fromContenteditable.length).toBe(0);
  });

  // ---- Task 9: canvas detection -------------------------------------------

  it("detects canvas with payment aria-label", () => {
    setBody('<canvas aria-label="payment canvas" width="200" height="100"></canvas>');
    const regions = scanSensitiveRegions();
    expect(regions.some((r) => r.category === "payment")).toBe(true);
  });

  it("detects canvas with role=payment", () => {
    setBody('<canvas role="payment"></canvas>');
    const regions = scanSensitiveRegions();
    expect(regions.some((r) => r.category === "payment")).toBe(true);
  });

  it("does NOT flag generic canvas", () => {
    setBody('<canvas aria-label="chart"></canvas>');
    const regions = scanSensitiveRegions();
    expect(regions.length).toBe(0);
  });

  // ---- Task 9: local PII text scan ----------------------------------------

  it("flags visible credit card number in text (geometry only, text not stored)", () => {
    setBody("<p>Your saved card: 4111 1111 1111 1111</p>");
    const regions = scanSensitiveRegions();
    expect(regions.some((r) => r.category === "payment")).toBe(true);
  });

  it("flags SSN pattern in visible text", () => {
    setBody("<p>SSN: 123-45-6789</p>");
    const regions = scanSensitiveRegions();
    expect(regions.some((r) => r.category === "pii")).toBe(true);
  });

  it("flags email address in visible text", () => {
    setBody("<p>Email: user@example.com</p>");
    const regions = scanSensitiveRegions();
    expect(regions.some((r) => r.category === "pii")).toBe(true);
  });

  it("does NOT flag generic text", () => {
    setBody("<p>Hello World, the price is 99 dollars</p>");
    const regions = scanSensitiveRegions();
    expect(regions.length).toBe(0);
  });

  // ---- Geometry invariants -------------------------------------------------

  it("all region coordinates are normalised to [0, 1]", () => {
    setBody(`
      <input type="password" />
      <input name="card" />
    `);
    const regions = scanSensitiveRegions();
    expect(regions.length).toBeGreaterThan(0);
    for (const r of regions) {
      expect(r.x).toBeGreaterThanOrEqual(0);
      expect(r.y).toBeGreaterThanOrEqual(0);
      expect(r.x + r.width).toBeLessThanOrEqual(1 + 1e-6);
      expect(r.y + r.height).toBeLessThanOrEqual(1 + 1e-6);
    }
  });

  it("scanSensitiveRegions returns empty array when page has no sensitive elements", () => {
    setBody("<p>Hello world</p>");
    const regions = scanSensitiveRegions();
    expect(regions).toEqual([]);
  });

  // ---- Deduplication -------------------------------------------------------

  it("deduplicates identical overlapping regions", () => {
    // Two inputs at exactly the same position → getBoundingClientRect returns
    // the same value for both → deduplicated.
    setBody(`
      <input type="password" />
      <input name="passwd" />
    `);
    const regions = scanSensitiveRegions();
    const passwordRegions = regions.filter((r) => r.category === "password");
    // Both have exact same rect (mocked to same value) → should be deduped to 1.
    expect(passwordRegions.length).toBe(1);
  });
});
