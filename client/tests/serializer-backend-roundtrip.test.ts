import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { encodeTensorPayload } from "../src/protocol/serializer";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const backendDir = path.join(repoRoot, "backend");
const decoderScript = path.join(backendDir, "tests", "helpers", "decode_client_payload.py");

function findPython(): string {
  const localVenv =
    process.platform === "win32"
      ? path.join(backendDir, ".venv", "Scripts", "python.exe")
      : path.join(backendDir, ".venv", "bin", "python");
  if (existsSync(localVenv)) return localVenv;
  return process.env.VDLM_PYTHON ?? (process.platform === "win32" ? "python" : "python3");
}

describe("client-to-backend tensor round-trip", () => {
  it("decodes the zlib-compressed fp16 serializer output within fp16 tolerance", () => {
    const matrix = new Float32Array(576 * 1152);
    const pattern = [0.1, -0.2, 0.3, -1.1, 0, 2.1, 0.07];
    for (let i = 0; i < matrix.length; i++) {
      matrix[i] = pattern[i % pattern.length]!;
    }

    const payload = encodeTensorPayload(matrix, { compress: true });
    const result = spawnSync(findPython(), [decoderScript], {
      cwd: backendDir,
      encoding: "utf8",
      env: {
        ...process.env,
        PYTHONPATH: backendDir,
      },
      input: JSON.stringify({ payload }),
    });

    if (result.error || result.status !== 0) {
      throw new Error(
        `Backend tensor decoder failed (${result.error?.message ?? result.status}): ${result.stderr}`,
      );
    }

    const decoded = JSON.parse(result.stdout.trim().split(/\r?\n/).at(-1) ?? "{}") as {
      shape: number[];
      max_abs_error: number;
    };
    expect(decoded.shape).toEqual([576, 1152]);
    expect(decoded.max_abs_error).toBeLessThanOrEqual(1e-3);
    console.info(`[Task 1] client/backend fp16 max absolute error: ${decoded.max_abs_error}`);
  });
});
