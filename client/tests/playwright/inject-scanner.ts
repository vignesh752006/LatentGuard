import { scanSensitiveRegions } from "../../src/sensitivity/dom-scanner";
import { mapRegionsToPatchesTs } from "../../src/sensitivity/patch-mapper";

window.__vdlmRunSensitiveScan = () => {
  const regions = scanSensitiveRegions();
  return {
    viewport: { width: window.innerWidth, height: window.innerHeight },
    regions,
    indices: mapRegionsToPatchesTs(window.innerWidth, window.innerHeight, regions),
  };
};
