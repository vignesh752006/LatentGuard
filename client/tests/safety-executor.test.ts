/**
 * Task 7 — Executor safety + patch mapper parity tests.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { clearActionLog, executeDirective, getActionLog } from "../src/actions/executor";
import { mapRegionsToPatchesTs, patchToNormalizedCoordsTs } from "../src/sensitivity/patch-mapper";
import type { ActionDirectivePayload } from "../src/protocol/messages";

// ---- Helpers ------------------------------------------------------------------

let _stateVersion = 1;

function makeDirective(overrides: Partial<ActionDirectivePayload> = {}): ActionDirectivePayload {
  return {
    action_id: "test-action-id",
    action: "WAIT",
    target_id: null,
    coords: null,
    text: null,
    scroll_delta: null,
    state_version_required: _stateVersion,
    confidence: 0.9,
    reasoning_summary: "test",
    ...overrides,
  };
}

// ---- Executor tests -----------------------------------------------------------

describe("Task 7: executor safety", () => {
  beforeEach(() => {
    clearActionLog();
    // Reset rate-limit timestamps (access module internals via re-import isn't easy
    // in vitest without resetting the module; instead we just run tests that don't
    // rely on a fresh state for rate-limiting in a separate describe block).
  });

  it("logs every executed action", async () => {
    await executeDirective(makeDirective({ action: "WAIT" }), _stateVersion);
    expect(getActionLog().length).toBeGreaterThanOrEqual(1);
    const entry = getActionLog().at(-1)!;
    expect(entry.action).toBe("WAIT");
    expect(entry.status).toBe("success");
  });

  it("rejects disallowed action (schema check catches unknown actions first)", async () => {
    // validateDirectiveSchema rejects the unknown action "HACK" before ALLOWED_ACTIONS check.
    const result = await executeDirective(
      makeDirective({ action: "HACK" as "CLICK" }),
      _stateVersion,
    );
    expect(result.status).toBe("failure");
    // Either INVALID_DIRECTIVE_SCHEMA (schema check first) or DISALLOWED_ACTION is acceptable.
    expect(["INVALID_DIRECTIVE_SCHEMA", "DISALLOWED_ACTION"]).toContain(result.errorCode);
  });

  it("rejects STALE_STATE_VERSION", async () => {
    const result = await executeDirective(
      makeDirective({ action: "WAIT", state_version_required: 99 }),
      1, // current version = 1, required = 99 → stale
    );
    // isDirectiveStale: required > current → stale
    expect(result.status).toBe("failure");
    expect(result.errorCode).toBe("STALE_STATE_VERSION");
  });

  it("blocks CLICK on sensitive patch and user denies", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(false);
    const result = await executeDirective(
      makeDirective({
        action: "CLICK",
        coords: [0.5, 0.5],
        target_id: 42,
        state_version_required: _stateVersion,
      }),
      _stateVersion,
      [42, 100, 200], // sensitive patch indices include 42
    );
    expect(result.status).toBe("cancelled");
    expect(result.errorCode).toBe("BLOCKED_SENSITIVE_REGION");
    vi.restoreAllMocks();
  });

  it("allows CLICK on non-sensitive patch", async () => {
    // In jsdom there's no real DOM element at 0.5,0.5 → returns failure, NOT cancel.
    const result = await executeDirective(
      makeDirective({
        action: "CLICK",
        coords: [0.5, 0.5],
        target_id: 300,
        state_version_required: _stateVersion,
      }),
      _stateVersion,
      [42, 100], // 300 is NOT in sensitive set
    );
    // Will fail because jsdom has no element there, but NOT blocked for safety reasons.
    expect(result.errorCode).not.toBe("BLOCKED_SENSITIVE_REGION");
    expect(result.errorCode).not.toBe("BLOCKED_DANGEROUS_LABEL");
  });

  it("logs cancelled actions", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(false);
    clearActionLog();
    await executeDirective(
      makeDirective({ action: "CLICK", coords: [0.5, 0.5], target_id: 1 }),
      _stateVersion,
      [1],
    );
    const log = getActionLog();
    expect(log.some((e) => e.status === "cancelled")).toBe(true);
    vi.restoreAllMocks();
  });
});

describe("Task 7: rate limiting", () => {
  beforeEach(() => clearActionLog());

  it("rejects after exceeding rate limit", async () => {
    // We'll fire WAIT actions rapidly; after 10 the 11th should be rate-limited.
    // Note: the rate-limiter state is module-level, so we can only verify the
    // behaviour structurally (not reset it between tests in this setup).
    // This test verifies the rate-limit logic path exists and returns the right code.
    // A full integration test would mock Date.now().
    const results: string[] = [];
    for (let i = 0; i < 15; i++) {
      const r = await executeDirective(makeDirective({ action: "WAIT" }), _stateVersion);
      results.push(r.errorCode ?? "ok");
    }
    // Some should be rate-limited (RATE_LIMITED error code).
    expect(results.some((r) => r === "RATE_LIMITED")).toBe(true);
  });
});

// ---- Patch mapper parity tests -----------------------------------------------

describe("Task 7: TS patch mapper parity (WASM not available in test env)", () => {
  it("mapRegionsToPatchesTs: empty regions → empty", () => {
    expect(mapRegionsToPatchesTs(1920, 1080, [])).toEqual([]);
  });

  it("mapRegionsToPatchesTs: full-screen region → all 576 patches", () => {
    const all = mapRegionsToPatchesTs(1920, 1080, [
      { x: 0, y: 0, width: 1, height: 1, category: "test" },
    ]);
    expect(all.length).toBe(576);
    expect(all[0]).toBe(0);
    expect(all[575]).toBe(575);
  });

  it("mapRegionsToPatchesTs: top-left corner → patch 0 included", () => {
    const patches = mapRegionsToPatchesTs(1920, 1080, [
      { x: 0, y: 0, width: 0.01, height: 0.01, category: "test" },
    ]);
    expect(patches).toContain(0);
  });

  it("mapRegionsToPatchesTs: bottom-right corner → patch 575 included", () => {
    const patches = mapRegionsToPatchesTs(1920, 1080, [
      { x: 0.99, y: 0.99, width: 0.01, height: 0.01, category: "test" },
    ]);
    expect(patches).toContain(575);
  });

  it("patchToNormalizedCoordsTs: patch 0 → near (0,0)", () => {
    const [x, y] = patchToNormalizedCoordsTs(0);
    expect(x).toBeCloseTo(1 / 48, 3); // (0+0.5)/24
    expect(y).toBeCloseTo(1 / 48, 3);
  });

  it("patchToNormalizedCoordsTs: patch 575 → near (1,1)", () => {
    const [x, y] = patchToNormalizedCoordsTs(575);
    expect(x).toBeCloseTo(23.5 / 24, 3);
    expect(y).toBeCloseTo(23.5 / 24, 3);
  });

  it("mapRegionsToPatchesTs result is sorted and unique", () => {
    const patches = mapRegionsToPatchesTs(1920, 1080, [
      { x: 0.0, y: 0.0, width: 0.5, height: 0.5, category: "a" },
      { x: 0.0, y: 0.0, width: 0.3, height: 0.3, category: "b" }, // overlapping
    ]);
    for (let i = 1; i < patches.length; i++) {
      expect(patches[i]!).toBeGreaterThan(patches[i - 1]!);
    }
  });

  it("parity: TS mapRegions matches expected patch indices for a known region", () => {
    // Region x=0,y=0,w=0.5,h=0.5 on 1920×1080:
    // patchW = ceil(1920/24)=80, patchH = ceil(1080/24)=45
    // px=0,py=0,pw=960,ph=540
    // colEnd = floor((0+960)/80) = 12 (clamped to 23)  → cols 0..12 = 13 cols
    // rowEnd = floor((0+540)/45) = 12 → rows 0..12 = 13 rows
    // → 13×13 = 169 patches
    const patches = mapRegionsToPatchesTs(1920, 1080, [
      { x: 0, y: 0, width: 0.5, height: 0.5, category: "test" },
    ]);
    expect(patches.length).toBe(169); // 13 rows × 13 cols (ceiling-division boundary)
    expect(patches[0]).toBe(0);
  });
});
