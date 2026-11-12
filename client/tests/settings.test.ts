import { describe, expect, it } from "vitest";

import { DEFAULT_CONFIG } from "../src/config/settings";

describe("default client configuration", () => {
  it("selects ONNX SigLIP and SVD while keeping mock as an explicit mode", () => {
    expect(DEFAULT_CONFIG.visionMode).toBe("onnx");
    expect(DEFAULT_CONFIG.sanitizerMode).toBe("svd");
    expect(DEFAULT_CONFIG.sanitizerK).toBe(4);
    expect(DEFAULT_CONFIG.sanitizerAlpha).toBe(0.01);
    expect(DEFAULT_CONFIG.payloadMode).toBe("source-pca");
    expect(DEFAULT_CONFIG.onnxModelUrl).toContain("vision-fp16.onnx");
  });
});
