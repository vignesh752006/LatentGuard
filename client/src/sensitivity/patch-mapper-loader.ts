/**
 * Task 7 — Unified patch mapper.
 *
 * Tries to load the Rust/WASM patch mapper (layout-parser.wasm) first.
 * Falls back transparently to the pure-TS implementation if WASM loading fails.
 *
 * Usage:
 *   const mapper = await createPatchMapper();
 *   const indices = mapper.mapRegionsToPatches(vw, vh, regions);
 */

import type { SensitiveRegion } from "../messaging/types";
import { mapRegionsToPatchesTs, patchToNormalizedCoordsTs } from "./patch-mapper";
import { TENSOR_CONTRACT } from "../protocol/messages";

export interface PatchMapper {
  mapRegionsToPatches(
    viewportWidth: number,
    viewportHeight: number,
    regions: SensitiveRegion[],
  ): number[];
  patchToNormalizedCoords(patchIndex: number): [number, number];
  /** Whether the WASM implementation is active. */
  isWasm: boolean;
}

interface WasmPatchMapperModule {
  default?: () => Promise<unknown>;
  map_regions_to_patch_indices: (
    viewportWidth: number,
    viewportHeight: number,
    gridRows: number,
    gridCols: number,
    regions: Float32Array,
  ) => Uint32Array;
  patch_index_to_normalized_coords: (
    patchIndex: number,
    gridRows: number,
    gridCols: number,
  ) => Float32Array;
}

/** Lazy singleton. */
let _mapper: PatchMapper | null = null;

/**
 * Returns the shared PatchMapper instance, loading WASM on first call.
 * Subsequent calls return the cached instance.
 */
export async function createPatchMapper(): Promise<PatchMapper> {
  if (_mapper) return _mapper;
  _mapper = await _tryLoadWasm();
  return _mapper;
}

/** For tests: reset cached instance. */
export function resetPatchMapper(): void {
  _mapper = null;
}

async function _tryLoadWasm(): Promise<PatchMapper> {
  try {
    // Dynamic import — bundler resolves this at build time.
    // The wasm-bindgen glue exports `map_regions_to_patch_indices` and
    // `patch_index_to_normalized_coords`.
    const wasmUrl: string = "/wasm/layout_parser.js";
    const wasmModule = (await import(/* @vite-ignore */ wasmUrl)) as WasmPatchMapperModule;
    // Trigger WASM init if the module exposes a default init function.
    if (typeof wasmModule.default === "function") {
      await wasmModule.default();
    }
    if (typeof wasmModule.map_regions_to_patch_indices !== "function") {
      throw new Error("WASM module missing map_regions_to_patch_indices");
    }
    console.info("[VDLM] WASM patch mapper loaded");
    return _makeWasmMapper(wasmModule);
  } catch (err) {
    console.warn("[VDLM] WASM patch mapper unavailable, using TS fallback:", err);
    return _makeTsMapper();
  }
}

function _makeWasmMapper(wasmModule: WasmPatchMapperModule): PatchMapper {
  const { gridRows, gridCols } = TENSOR_CONTRACT;
  return {
    isWasm: true,
    mapRegionsToPatches(vw, vh, regions) {
      const flat = new Float32Array(regions.length * 4);
      regions.forEach((r, i) => {
        flat[i * 4] = r.x;
        flat[i * 4 + 1] = r.y;
        flat[i * 4 + 2] = r.width;
        flat[i * 4 + 3] = r.height;
      });
      const result = wasmModule.map_regions_to_patch_indices(
        vw,
        vh,
        gridRows,
        gridCols,
        flat,
      );
      return Array.from(result);
    },
    patchToNormalizedCoords(idx) {
      const coords = wasmModule.patch_index_to_normalized_coords(
        idx,
        gridRows,
        gridCols,
      );
      return [coords[0]!, coords[1]!];
    },
  };
}

function _makeTsMapper(): PatchMapper {
  return {
    isWasm: false,
    mapRegionsToPatches: mapRegionsToPatchesTs,
    patchToNormalizedCoords: patchToNormalizedCoordsTs,
  };
}
