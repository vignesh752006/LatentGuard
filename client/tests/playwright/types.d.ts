import type { SensitiveRegion } from "../../src/messaging/types";

declare global {
  interface Window {
    __vdlmRunSensitiveScan?: () => {
      viewport: { width: number; height: number };
      regions: SensitiveRegion[];
      indices: number[];
    };
  }
}

export {};
