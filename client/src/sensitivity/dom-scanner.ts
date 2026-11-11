import type { SensitiveRegion } from "../messaging/types";

const SENSITIVE_SELECTORS: Array<{ selector: string; category: string }> = [
  { selector: 'input[type="password"]', category: "password" },
  { selector: 'input[autocomplete*="cc"]', category: "payment" },
  { selector: 'input[name*="password" i]', category: "password" },
  { selector: 'input[name*="passwd" i]', category: "password" },
  { selector: 'input[name*="card" i]', category: "payment" },
  { selector: 'input[name*="cvv" i]', category: "payment" },
  { selector: 'input[name*="ssn" i]', category: "pii" },
  { selector: "[data-vdlm-sensitive]", category: "custom" },
  { selector: 'iframe[title*="payment" i]', category: "payment" },
];

// ---- Task 9: additional detection patterns ---------------------------------

/** Task 9: contenteditable elements with sensitive roles / labels. */
const CONTENTEDITABLE_SENSITIVE_RE =
  /password|card|credit|debit|ssn|social.?security|cvv|cvc|pin\b/i;

/**
 * Task 9: local OCR/regex patterns that flag PII in VISIBLE TEXT.
 * Text is NEVER sent off-device — only the bounding-box geometry leaves the browser.
 *
 * Patterns (all applied to the textContent of rendered elements):
 *   - Credit card: 4 groups of 4 digits (various separators)
 *   - SSN: XXX-XX-XXXX or XXXXXXXXX
 *   - Email: basic RFC5322 subset
 */
const LOCAL_PII_PATTERNS: Array<{ name: string; re: RegExp; category: string }> = [
  {
    name: "credit_card",
    re: /\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b/,
    category: "payment",
  },
  {
    name: "ssn",
    re: /\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b/,
    category: "pii",
  },
  {
    name: "email",
    re: /\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b/,
    category: "pii",
  },
];

export function scanSensitiveRegions(): SensitiveRegion[] {
  const regions: SensitiveRegion[] = [];
  const viewportW = window.innerWidth;
  const viewportH = window.innerHeight;

  // --- 1. Existing selector-based detection ---
  for (const { selector, category } of SENSITIVE_SELECTORS) {
    document.querySelectorAll(selector).forEach((el) => {
      const r = _visibleRect(el, viewportW, viewportH);
      if (r) regions.push({ ...r, category });
    });
  }

  // --- 2. Task 9: contenteditable detection ---
  document.querySelectorAll("[contenteditable]").forEach((el) => {
    const label =
      el.getAttribute("aria-label") ||
      el.getAttribute("placeholder") ||
      el.getAttribute("data-vdlm-label") ||
      "";
    if (CONTENTEDITABLE_SENSITIVE_RE.test(label)) {
      const r = _visibleRect(el, viewportW, viewportH);
      if (r) regions.push({ ...r, category: "pii" });
    }
  });

  // --- 3. Task 9: canvas bounding-box detection ---
  // Canvases tagged with [data-vdlm-sensitive] are detected by selector above.
  // Additionally, any canvas with a recognisable payment/sensitive role attribute:
  document.querySelectorAll("canvas").forEach((el) => {
    const role = el.getAttribute("role") || el.getAttribute("aria-label") || "";
    if (/card|payment|sensitive/i.test(role)) {
      const r = _visibleRect(el, viewportW, viewportH);
      if (r) regions.push({ ...r, category: "payment" });
    }
  });

  // --- 4. Task 9: same-origin iframe injection ---
  // We can only access iframes on the same origin.  For cross-origin iframes,
  // geometry is detected by selector (title*="payment") already above.
  document.querySelectorAll("iframe").forEach((iframe) => {
    try {
      const doc = (iframe as HTMLIFrameElement).contentDocument;
      if (!doc) return; // cross-origin: blocked by browser security
      const iframeRegions = _scanDoc(doc, viewportW, viewportH, iframe);
      regions.push(...iframeRegions);
    } catch {
      // Cross-origin SecurityError — expected, ignore silently.
    }
  });

  // --- 5. Task 9: local OCR/regex pass over visible text ---
  // Text never leaves the device — only bounding-box coordinates are recorded.
  _scanVisibleTextForPii(regions, viewportW, viewportH);

  return _deduplicateRegions(regions);
}

/** Never extracts text content — geometry only. */
export function getViewportSize(): { width: number; height: number } {
  return { width: window.innerWidth, height: window.innerHeight };
}

// ---- Private helpers -------------------------------------------------------

function _visibleRect(
  el: Element,
  viewportW: number,
  viewportH: number,
): Omit<SensitiveRegion, "category"> | null {
  const rect = el.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return null;
  if (rect.bottom < 0 || rect.right < 0) return null;
  if (rect.top > viewportH || rect.left > viewportW) return null;
  return {
    x: rect.left / viewportW,
    y: rect.top / viewportH,
    width: rect.width / viewportW,
    height: rect.height / viewportH,
  };
}

/** Scan a same-origin iframe document for sensitive selectors. */
function _scanDoc(
  doc: Document,
  viewportW: number,
  viewportH: number,
  iframe: Element,
): SensitiveRegion[] {
  const iframeRect = iframe.getBoundingClientRect();
  const regions: SensitiveRegion[] = [];

  for (const { selector, category } of SENSITIVE_SELECTORS) {
    doc.querySelectorAll(selector).forEach((el) => {
      const rect = el.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) return;
      // Map to top-level viewport coordinates via iframe offset.
      const absLeft = iframeRect.left + rect.left;
      const absTop = iframeRect.top + rect.top;
      regions.push({
        x: absLeft / viewportW,
        y: absTop / viewportH,
        width: rect.width / viewportW,
        height: rect.height / viewportH,
        category,
      });
    });
  }
  return regions;
}

/**
 * Task 9: local PII text scan.
 *
 * Walks visible leaf text nodes; if a node's textContent matches a PII pattern,
 * records the bounding rectangle of the PARENT element as a sensitive region.
 * TEXT CONTENT IS NEVER STORED OR TRANSMITTED — only the geometry.
 */
function _scanVisibleTextForPii(
  regions: SensitiveRegion[],
  viewportW: number,
  viewportH: number,
): void {
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const parent = node.parentElement;
      if (!parent) return NodeFilter.FILTER_REJECT;
      // Skip script/style/hidden elements.
      const tag = parent.tagName.toLowerCase();
      if (["script", "style", "noscript"].includes(tag)) return NodeFilter.FILTER_REJECT;
      const style = window.getComputedStyle(parent);
      if (style.display === "none" || style.visibility === "hidden") return NodeFilter.FILTER_REJECT;
      return NodeFilter.FILTER_ACCEPT;
    },
  });

  let node: Text | null;
  while ((node = walker.nextNode() as Text | null)) {
    const text = node.textContent ?? "";
    for (const { re, category } of LOCAL_PII_PATTERNS) {
      if (re.test(text)) {
        const parent = node.parentElement!;
        const r = _visibleRect(parent, viewportW, viewportH);
        if (r) {
          regions.push({ ...r, category });
        }
        break; // one match per node is enough
      }
    }
    // TEXT IS DISCARDED HERE — only the bounding box was recorded above.
  }
}

/** Remove duplicate/overlapping regions (same category and very close geometry). */
function _deduplicateRegions(regions: SensitiveRegion[]): SensitiveRegion[] {
  const out: SensitiveRegion[] = [];
  for (const r of regions) {
    const dup = out.some(
      (o) =>
        o.category === r.category &&
        Math.abs(o.x - r.x) < 0.001 &&
        Math.abs(o.y - r.y) < 0.001 &&
        Math.abs(o.width - r.width) < 0.001 &&
        Math.abs(o.height - r.height) < 0.001,
    );
    if (!dup) out.push(r);
  }
  return out;
}
