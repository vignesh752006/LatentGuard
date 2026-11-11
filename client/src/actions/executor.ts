/**
 * Task 7 — Safety executor.
 *
 * Extends the original executor with:
 *   1. CLICK blocked on elements inside I_sens regions or matching dangerous labels.
 *      Dangerous labels: pay, purchase, delete, submit (and variants), unless user confirms.
 *   2. Rate limiting: max 10 actions per 10-second rolling window.
 *   3. Structured action log (append-only in-memory, exported for inspection/tests).
 *
 * "User confirm" is simulated via window.confirm() in the browser extension context.
 * In tests (jsdom) window.confirm always returns false by default — override via
 * `vi.spyOn(window, 'confirm').mockReturnValue(true)` to simulate user approval.
 */

import type { ActionDirectivePayload } from "../protocol/messages";
import {
  isDirectiveStale,
  normalizedToClientCoords,
  validateDirectiveSchema,
} from "../state/state-tracker";

export interface ExecutionResult {
  status: "success" | "failure" | "cancelled";
  errorCode: string | null;
}

export interface ActionLogEntry {
  timestamp: number;
  actionId: string;
  action: string;
  coords: [number, number] | null;
  status: "success" | "failure" | "cancelled";
  errorCode: string | null;
  blockedReason?: string;
}

// In-memory append-only action log (cleared per session reset).
const _actionLog: ActionLogEntry[] = [];
export function getActionLog(): readonly ActionLogEntry[] {
  return _actionLog;
}
export function clearActionLog(): void {
  _actionLog.length = 0;
}

// Rate-limiter: max 10 actions per 10 seconds.
const RATE_WINDOW_MS = 10_000;
const RATE_MAX = 10;
const _actionTimestamps: number[] = [];

function checkRateLimit(): boolean {
  const now = Date.now();
  // Evict entries older than the window.
  while (_actionTimestamps.length > 0 && now - _actionTimestamps[0]! > RATE_WINDOW_MS) {
    _actionTimestamps.shift();
  }
  if (_actionTimestamps.length >= RATE_MAX) return false;
  _actionTimestamps.push(now);
  return true;
}

/** Labels that require user confirmation before CLICK is dispatched. */
const DANGEROUS_LABEL_RE =
  /\b(pay|purchase|buy|delete|remove|submit|confirm|checkout|place.?order|send.?payment)\b/i;

const ALLOWED_ACTIONS = new Set(["CLICK", "SCROLL", "TYPE", "WAIT", "NAVIGATE", "DONE"]);

export async function executeDirective(
  directive: ActionDirectivePayload,
  currentStateVersion: number,
  /** Sensitive patch indices from the current frame (for CLICK blocking). */
  sensitivePatchIndices: number[] = [],
): Promise<ExecutionResult> {
  if (!validateDirectiveSchema(directive)) {
    return _log(directive, { status: "failure", errorCode: "INVALID_DIRECTIVE_SCHEMA" });
  }

  if (!ALLOWED_ACTIONS.has(directive.action)) {
    return _log(directive, { status: "failure", errorCode: "DISALLOWED_ACTION" });
  }

  if (isDirectiveStale(directive, currentStateVersion)) {
    return _log(directive, { status: "failure", errorCode: "STALE_STATE_VERSION" });
  }

  if (!checkRateLimit()) {
    return _log(directive, { status: "failure", errorCode: "RATE_LIMITED" });
  }

  try {
    let result: ExecutionResult;
    switch (directive.action) {
      case "CLICK":
        result = await executeClick(directive, sensitivePatchIndices);
        break;
      case "SCROLL":
        result = executeScroll(directive);
        break;
      case "TYPE":
        result = executeType(directive);
        break;
      case "WAIT":
        await sleep(500);
        result = { status: "success", errorCode: null };
        break;
      case "NAVIGATE":
        result = executeNavigate(directive);
        break;
      case "DONE":
        result = { status: "success", errorCode: null };
        break;
      default:
        result = { status: "failure", errorCode: "UNSUPPORTED_ACTION" };
    }
    return _log(directive, result);
  } catch (error) {
    return _log(directive, {
      status: "failure",
      errorCode: error instanceof Error ? error.message : "EXECUTION_ERROR",
    });
  }
}

async function executeClick(
  directive: ActionDirectivePayload,
  sensitivePatchIndices: number[],
): Promise<ExecutionResult> {
  if (!directive.coords) {
    return { status: "failure", errorCode: "MISSING_COORDS" };
  }
  const { x, y } = normalizedToClientCoords(directive.coords);

  // Block click if the target patch is sensitive.
  if (directive.target_id !== null && directive.target_id !== undefined) {
    if (sensitivePatchIndices.includes(directive.target_id)) {
      const confirmed = window.confirm(
        `[VDLM Safety] The agent wants to click a sensitive region (patch ${directive.target_id}). Allow?`,
      );
      if (!confirmed) {
        return { status: "cancelled", errorCode: "BLOCKED_SENSITIVE_REGION" };
      }
    }
  }

  const target = document.elementFromPoint(x, y);
  if (!target || !isSafeElement(target)) {
    return { status: "failure", errorCode: "UNSAFE_OR_MISSING_TARGET" };
  }

  // Block click on elements with dangerous labels unless user confirms.
  if (isDangerousTarget(target)) {
    const label = getElementLabel(target);
    const confirmed = window.confirm(
      `[VDLM Safety] The agent wants to click "${label}". This may trigger a purchase/delete/submit. Allow?`,
    );
    if (!confirmed) {
      return { status: "cancelled", errorCode: "BLOCKED_DANGEROUS_LABEL" };
    }
  }

  if (target instanceof HTMLElement) {
    target.focus({ preventScroll: true });
    target.click();
  }
  return { status: "success", errorCode: null };
}

function executeScroll(directive: ActionDirectivePayload): ExecutionResult {
  const delta = directive.scroll_delta ?? [0, 300];
  window.scrollBy({ left: delta[0], top: delta[1], behavior: "smooth" });
  return { status: "success", errorCode: null };
}

function executeType(directive: ActionDirectivePayload): ExecutionResult {
  if (!directive.text || !directive.coords) {
    return { status: "failure", errorCode: "MISSING_TYPE_PARAMS" };
  }
  const { x, y } = normalizedToClientCoords(directive.coords);
  const target = document.elementFromPoint(x, y);
  if (!(target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement)) {
    return { status: "failure", errorCode: "TARGET_NOT_TYPABLE" };
  }
  if (target.type === "password") {
    return { status: "failure", errorCode: "REFUSE_PASSWORD_FIELD" };
  }
  if (!isSafeElement(target)) {
    return { status: "failure", errorCode: "UNSAFE_TARGET" };
  }
  target.focus();
  target.value = directive.text;
  target.dispatchEvent(new Event("input", { bubbles: true }));
  target.dispatchEvent(new Event("change", { bubbles: true }));
  return { status: "success", errorCode: null };
}

function executeNavigate(directive: ActionDirectivePayload): ExecutionResult {
  if (!directive.text) {
    return { status: "failure", errorCode: "MISSING_URL" };
  }
  let url: URL;
  try {
    url = new URL(directive.text, window.location.href);
  } catch {
    return { status: "failure", errorCode: "INVALID_URL" };
  }
  if (!["http:", "https:"].includes(url.protocol)) {
    return { status: "failure", errorCode: "UNSAFE_URL_SCHEME" };
  }
  window.location.assign(url.toString());
  return { status: "success", errorCode: null };
}

function isSafeElement(el: Element): boolean {
  if (el.closest("script, iframe[src^='javascript:']")) return false;
  return true;
}

function isDangerousTarget(el: Element): boolean {
  const label = getElementLabel(el).toLowerCase();
  return DANGEROUS_LABEL_RE.test(label);
}

function getElementLabel(el: Element): string {
  if (el instanceof HTMLInputElement || el instanceof HTMLButtonElement) {
    return el.value || el.textContent || el.getAttribute("aria-label") || "";
  }
  return el.textContent?.trim() || el.getAttribute("aria-label") || "";
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function _log(
  directive: ActionDirectivePayload,
  result: ExecutionResult,
  blockedReason?: string,
): ExecutionResult {
  _actionLog.push({
    timestamp: Date.now(),
    actionId: directive.action_id,
    action: directive.action,
    coords: directive.coords as [number, number] | null,
    status: result.status,
    errorCode: result.errorCode,
    ...(blockedReason ? { blockedReason } : {}),
  });
  return result;
}
