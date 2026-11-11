/**
 * Task 10 — Attestation unit tests.
 *
 * Tests:
 * - hashEmbedding: deterministic, correct length
 * - buildAttestation: produces valid base64url JSON record
 * - verifyAttestation: round-trips correctly, rejects tampered records
 * - buildLatentFramePayloadWithAttestation: populates provenance.attestation
 */

import { describe, expect, it, beforeEach } from "vitest";
import {
  ATTESTATION_SCHEME,
  buildAttestation,
  hashEmbedding,
  resetSessionKey,
  verifyAttestation,
  type AttestationRecord,
} from "../src/protocol/attestation";
import { buildLatentFramePayloadWithAttestation } from "../src/protocol/serializer";
import { TENSOR_CONTRACT } from "../src/protocol/messages";

const { patchCount, embeddingDim } = TENSOR_CONTRACT;

function makeEmbedding(fill = 0.5): Float32Array {
  return new Float32Array(patchCount * embeddingDim).fill(fill);
}

describe("Task 10: attestation", () => {
  beforeEach(() => {
    resetSessionKey();
  });
  // ---- hashEmbedding -------------------------------------------------------

  it("hashEmbedding returns 64-char hex string", async () => {
    const emb = makeEmbedding(0.5);
    const h = await hashEmbedding(emb);
    expect(h).toMatch(/^[0-9a-f]{64}$/);
  });

  it("hashEmbedding is deterministic for same input", async () => {
    const emb = makeEmbedding(0.123);
    const h1 = await hashEmbedding(emb);
    const h2 = await hashEmbedding(emb);
    expect(h1).toBe(h2);
  });

  it("hashEmbedding differs for different inputs", async () => {
    const h1 = await hashEmbedding(makeEmbedding(0.1));
    const h2 = await hashEmbedding(makeEmbedding(0.9));
    expect(h1).not.toBe(h2);
  });

  // ---- buildAttestation ----------------------------------------------------

  it("buildAttestation returns non-empty string", async () => {
    const emb = makeEmbedding();
    const att = await buildAttestation("0.1.0", emb);
    expect(typeof att).toBe("string");
    expect(att.length).toBeGreaterThan(100);
  });

  it("buildAttestation decodes to a valid JSON record", async () => {
    const emb = makeEmbedding(0.42);
    const att = await buildAttestation("0.1.0", emb);

    // Decode base64url
    const padded = att.replace(/-/g, "+").replace(/_/g, "/");
    const json = atob(padded + "==".slice(0, (4 - padded.length % 4) % 4));
    const record: AttestationRecord = JSON.parse(json);

    expect(record.v).toBe(1);
    expect(record.alg).toBe("ECDSA-P256-SHA256");
    expect(record.kid).toMatch(/^[0-9a-f]{8}$/);
    expect(record.shader_version).toBe("0.1.0");
    expect(record.embedding_hash).toMatch(/^[0-9a-f]{64}$/);
    expect(typeof record.sig).toBe("string");
    expect(record.sig.length).toBeGreaterThan(10);
  });

  // ---- verifyAttestation ---------------------------------------------------

  it("verifyAttestation round-trips: valid → {valid: true}", async () => {
    const emb = makeEmbedding(0.7);
    const att = await buildAttestation("0.2.0", emb);
    const result = await verifyAttestation(att);
    expect(result.valid).toBe(true);
    expect(result.record?.shader_version).toBe("0.2.0");
  });

  it("verifyAttestation rejects null attestation", async () => {
    const result = await verifyAttestation(null as unknown as string);
    expect(result.valid).toBe(false);
  });

  it("verifyAttestation rejects empty string", async () => {
    const result = await verifyAttestation("");
    expect(result.valid).toBe(false);
  });

  it("verifyAttestation rejects garbled base64", async () => {
    const result = await verifyAttestation("not-valid-base64!!!");
    expect(result.valid).toBe(false);
  });

  it("verifyAttestation rejects wrong shader_version when strict", async () => {
    const emb = makeEmbedding();
    const att = await buildAttestation("1.0.0", emb);
    const result = await verifyAttestation(att, "0.1.0");
    expect(result.valid).toBe(false);
    expect(result.reason).toMatch(/mismatch/);
  });

  it("verifyAttestation detects tampered embedding_hash", async () => {
    const emb = makeEmbedding(0.5);
    const att = await buildAttestation("0.1.0", emb);

    // Decode, tamper, re-encode
    const padded = att.replace(/-/g, "+").replace(/_/g, "/");
    const json = atob(padded + "==".slice(0, (4 - padded.length % 4) % 4));
    const record = JSON.parse(json);
    record.embedding_hash = "a".repeat(64); // fake hash
    const tampered = btoa(JSON.stringify(record))
      .replace(/\+/g, "-")
      .replace(/\//g, "_")
      .replace(/=+$/, "");

    // Structural check passes (hash format still valid) but sig will fail
    const result = await verifyAttestation(tampered);
    // Signature check should fail because canonical payload changed
    expect(result.valid).toBe(false);
  });

  // ---- ATTESTATION_SCHEME constant -----------------------------------------

  it("ATTESTATION_SCHEME is a non-empty string", () => {
    expect(typeof ATTESTATION_SCHEME).toBe("string");
    expect(ATTESTATION_SCHEME.length).toBeGreaterThan(0);
  });

  // ---- buildLatentFramePayloadWithAttestation ------------------------------

  it("buildLatentFramePayloadWithAttestation populates provenance.attestation", async () => {
    const matrix = makeEmbedding(0.3);
    const payload = await buildLatentFramePayloadWithAttestation({
      frameId: "f-00001",
      stateVersion: 1,
      sanitizedMatrix: matrix,
      sensitivePatchIndices: [0, 1],
      viewportHash: "sha256:abc123",
      lastActionId: null,
      compress: false,
      shaderVersion: "0.1.0",
    });

    expect(payload.provenance.attestation).not.toBeNull();
    expect(typeof payload.provenance.attestation).toBe("string");
    expect(payload.provenance.attestation_scheme).toBe(ATTESTATION_SCHEME);
  });

  it("buildLatentFramePayloadWithAttestation attestation verifies correctly", async () => {
    const matrix = makeEmbedding(0.6);
    const payload = await buildLatentFramePayloadWithAttestation({
      frameId: "f-00002",
      stateVersion: 2,
      sanitizedMatrix: matrix,
      sensitivePatchIndices: [],
      viewportHash: "sha256:xyz",
      lastActionId: null,
    });

    const result = await verifyAttestation(payload.provenance.attestation!);
    expect(result.valid).toBe(true);
  });
});
