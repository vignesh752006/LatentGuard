/**
 * Task 10 — Signed provenance attestation.
 *
 * Produces a signed integrity record that covers:
 *   - shader/model version string
 *   - SHA-256 hash of the sanitized embedding V'
 *   - per-session ECDSA P-256 key (generated once per extension session)
 *
 * This is an INTEGRITY CHECK, NOT a zero-knowledge proof.
 * The backend verifies the structure and signature; it cannot verify the
 * actual sanitization math without the raw embedding (which is withheld).
 *
 * Wire format for `attestation` field (base64url-encoded JSON):
 * {
 *   "v": 1,
 *   "alg": "ECDSA-P256-SHA256",
 *   "kid": "<session key id, 8 hex chars>",
 *   "shader_version": "<string>",
 *   "embedding_hash": "<hex SHA-256 of V' bytes>",
 *   "sig": "<base64url ECDSA signature over canonical payload>"
 * }
 */

export const ATTESTATION_SCHEME = "ECDSA-P256-SHA256-v1";

export interface AttestationRecord {
  v: number;
  alg: string;
  kid: string;
  shader_version: string;
  embedding_hash: string;
  sig: string;
}

/** Per-session ECDSA key pair (lazy-initialised). */
let _sessionKey: CryptoKeyPair | null = null;
let _sessionKid: string = "";

async function getSessionKey(): Promise<{ keyPair: CryptoKeyPair; kid: string }> {
  if (_sessionKey) return { keyPair: _sessionKey, kid: _sessionKid };

  _sessionKey = await crypto.subtle.generateKey(
    { name: "ECDSA", namedCurve: "P-256" },
    false, // non-exportable private key
    ["sign", "verify"],
  );

  // Derive a short key-id from the public key bytes.
  const pubRaw = await crypto.subtle.exportKey("raw", _sessionKey.publicKey);
  const hash = await crypto.subtle.digest("SHA-256", pubRaw);
  _sessionKid = Array.from(new Uint8Array(hash))
    .slice(0, 4)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");

  return { keyPair: _sessionKey, kid: _sessionKid };
}

/** For tests: reset the session key so a fresh one is generated. */
export function resetSessionKey(): void {
  _sessionKey = null;
  _sessionKid = "";
}

/**
 * Compute SHA-256 of the sanitized embedding bytes and return as hex string.
 * Input is a Float32Array (V'); we hash its raw bytes.
 */
export async function hashEmbedding(embedding: Float32Array): Promise<string> {
  const bytes = new Uint8Array(embedding.byteLength);
  bytes.set(new Uint8Array(embedding.buffer, embedding.byteOffset, embedding.byteLength));
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

/**
 * Build and sign a provenance attestation record.
 *
 * @param shaderVersion   Version string of the WGSL shader / sanitizer.
 * @param sanitizedEmb    The sanitized embedding V' (Float32Array).
 * @returns base64url-encoded JSON attestation string.
 */
export async function buildAttestation(
  shaderVersion: string,
  sanitizedEmb: Float32Array,
): Promise<string> {
  const { keyPair, kid } = await getSessionKey();

  const embeddingHash = await hashEmbedding(sanitizedEmb);

  // Canonical payload: deterministic JSON with sorted keys.
  const canonicalPayload = JSON.stringify({
    alg: "ECDSA-P256-SHA256",
    embedding_hash: embeddingHash,
    kid,
    shader_version: shaderVersion,
    v: 1,
  });

  const payloadBytes = new TextEncoder().encode(canonicalPayload);
  const sigBytes = await crypto.subtle.sign(
    { name: "ECDSA", hash: "SHA-256" },
    keyPair.privateKey,
    payloadBytes,
  );

  const record: AttestationRecord = {
    v: 1,
    alg: "ECDSA-P256-SHA256",
    kid,
    shader_version: shaderVersion,
    embedding_hash: embeddingHash,
    sig: _base64url(new Uint8Array(sigBytes)),
  };

  return _base64url(new TextEncoder().encode(JSON.stringify(record)));
}

/**
 * Verify an attestation produced by this session's key.
 *
 * Returns `{ valid: true }` if the signature checks out, or
 * `{ valid: false, reason: string }` otherwise.
 *
 * NOTE: This is an INTEGRITY CHECK — it confirms the attestation was signed
 * by this session's key and that the payload fields are intact.
 * It is NOT a zero-knowledge proof of correct sanitization.
 */
export async function verifyAttestation(
  attestationB64: string,
  expectedShaderVersion?: string,
): Promise<{ valid: boolean; reason?: string; record?: AttestationRecord }> {
  try {
    const json = new TextDecoder().decode(_fromBase64url(attestationB64));
    const record: AttestationRecord = JSON.parse(json);

    if (record.v !== 1) return { valid: false, reason: "Unknown attestation version" };
    if (record.alg !== "ECDSA-P256-SHA256") return { valid: false, reason: "Unknown algorithm" };
    if (expectedShaderVersion && record.shader_version !== expectedShaderVersion) {
      return { valid: false, reason: "Shader version mismatch" };
    }

    // Verify signature against canonical payload.
    const { keyPair } = await getSessionKey();
    const canonical = JSON.stringify({
      alg: record.alg,
      embedding_hash: record.embedding_hash,
      kid: record.kid,
      shader_version: record.shader_version,
      v: record.v,
    });
    const payloadBytes = new TextEncoder().encode(canonical);
    const sigBytes = _fromBase64url(record.sig);

    const ok = await crypto.subtle.verify(
      { name: "ECDSA", hash: "SHA-256" },
      keyPair.publicKey,
      sigBytes,
      payloadBytes,
    );
    if (!ok) return { valid: false, reason: "Signature verification failed" };
    return { valid: true, record };
  } catch (err) {
    return { valid: false, reason: `Parse error: ${err}` };
  }
}

function _base64url(bytes: Uint8Array): string {
  let binary = "";
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function _fromBase64url(s: string): Uint8Array<ArrayBuffer> {
  const padded = s.replace(/-/g, "+").replace(/_/g, "/");
  const binary = atob(padded);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}
