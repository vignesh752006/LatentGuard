import re
from typing import Any
import base64
import json
import logging

logger = logging.getLogger(__name__)

FORBIDDEN_FIELD_PATTERN = re.compile(
    r"(screenshot|raw_pixels?|image_data|dom_html|password|credential|"
    r"payment_info|accessibility_tree|png|jpeg|jpg|webp|canvas_blob)",
    re.IGNORECASE,
)

FORBIDDEN_VALUE_PATTERNS = (
    re.compile(r"^data:image/", re.IGNORECASE),
    re.compile(r"iVBORw0KGgo", re.IGNORECASE),  # PNG header in base64
)


class PayloadRejected(Exception):
    def __init__(self, code: str, message: str, recoverable: bool = False):
        self.code = code
        self.message = message
        self.recoverable = recoverable
        super().__init__(message)


def scan_for_forbidden_content(obj: Any, path: str = "") -> None:
    """Recursively reject payloads that may carry raw private visual content."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_path = f"{path}.{key}" if path else key
            if FORBIDDEN_FIELD_PATTERN.search(str(key)):
                raise PayloadRejected(
                    "FORBIDDEN_FIELD",
                    f"Forbidden field: {key}",
                    recoverable=False,
                )
            scan_for_forbidden_content(value, key_path)
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            scan_for_forbidden_content(item, f"{path}[{index}]")
    elif isinstance(obj, str):
        if path.endswith(".data") and "tensor" in path:
            return
        if len(obj) > 500_000:
            raise PayloadRejected(
                "PAYLOAD_TOO_LARGE",
                f"String field too large at {path or 'root'}",
                recoverable=False,
            )
        for pattern in FORBIDDEN_VALUE_PATTERNS:
            if pattern.search(obj):
                raise PayloadRejected(
                    "FORBIDDEN_CONTENT",
                    f"Forbidden content pattern at {path or 'root'}",
                    recoverable=False,
                )


def enforce_payload_size(raw_bytes: int, max_bytes: int) -> None:
    if raw_bytes > max_bytes:
        raise PayloadRejected(
            "PAYLOAD_TOO_LARGE",
            f"Payload size {raw_bytes} exceeds limit {max_bytes}",
            recoverable=False,
        )


# ---------------------------------------------------------------------------
# Task 10 — Attestation structural verification
# ---------------------------------------------------------------------------

_EXPECTED_ALG = "ECDSA-P256-SHA256"
_EXPECTED_VERSION = 1
_REQUIRED_FIELDS = {"v", "alg", "kid", "shader_version", "embedding_hash", "sig"}
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX8_RE  = re.compile(r"^[0-9a-f]{8}$")


def _b64url_decode(s: str) -> bytes:
    padded = s.replace("-", "+").replace("_", "/")
    missing = (4 - len(padded) % 4) % 4
    return base64.b64decode(padded + "=" * missing)


def verify_attestation(
    attestation_b64: str | None,
    *,
    expected_shader_version: str | None = None,
) -> dict:
    """
    Structural integrity check for a VDLM attestation string.

    Returns: { "valid": bool, "reason": str | None, "record": dict | None }

    NOTE: This is an INTEGRITY CHECK — it confirms the record is well-formed
    and uses the expected algorithm. It is NOT a zero-knowledge proof and does
    NOT cryptographically verify the ECDSA signature (that would require the
    client to share its session public key — a future extension).
    """
    if not attestation_b64:
        return {"valid": False, "reason": "No attestation present", "record": None}

    try:
        raw_json = _b64url_decode(attestation_b64).decode("utf-8")
        record = json.loads(raw_json)
    except Exception as exc:
        return {"valid": False, "reason": f"Cannot decode attestation: {exc}", "record": None}

    missing = _REQUIRED_FIELDS - set(record.keys())
    if missing:
        return {"valid": False, "reason": f"Missing fields: {missing}", "record": record}

    if record.get("v") != _EXPECTED_VERSION:
        return {"valid": False, "reason": f"Unsupported version: {record.get('v')}", "record": record}

    if record.get("alg") != _EXPECTED_ALG:
        return {"valid": False, "reason": f"Unsupported algorithm: {record.get('alg')}", "record": record}

    if not _HEX64_RE.match(str(record.get("embedding_hash", ""))):
        return {"valid": False, "reason": "embedding_hash must be 64 hex chars", "record": record}

    if not _HEX8_RE.match(str(record.get("kid", ""))):
        return {"valid": False, "reason": "kid must be 8 hex chars", "record": record}

    if expected_shader_version and record.get("shader_version") != expected_shader_version:
        return {
            "valid": False,
            "reason": (
                f"shader_version mismatch: expected {expected_shader_version!r}, "
                f"got {record.get('shader_version')!r}"
            ),
            "record": record,
        }

    logger.info(
        "[VDLM] Attestation structural check PASSED "
        "(integrity only — NOT a ZKP, crypto sig not verified without client pubkey). "
        "kid=%s shader=%s hash=%.8s...",
        record["kid"],
        record["shader_version"],
        record["embedding_hash"],
    )
    return {"valid": True, "reason": None, "record": record}
