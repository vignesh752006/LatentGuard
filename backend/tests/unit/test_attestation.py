"""Task 10 — Backend attestation verification unit tests."""

import base64
import json
import pytest

from app.security.payload_guard import verify_attestation


def _make_attestation(overrides: dict | None = None) -> str:
    """Build a syntactically valid attestation base64url string."""
    record = {
        "v": 1,
        "alg": "ECDSA-P256-SHA256",
        "kid": "deadbeef",
        "shader_version": "0.1.0",
        "embedding_hash": "a" * 64,
        "sig": base64.urlsafe_b64encode(b"fakesig" * 10).decode().rstrip("="),
    }
    if overrides:
        record.update(overrides)
    raw = json.dumps(record).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def test_verify_valid_attestation():
    att = _make_attestation()
    result = verify_attestation(att)
    assert result["valid"] is True
    assert result["reason"] is None
    assert result["record"]["kid"] == "deadbeef"


def test_verify_none_attestation():
    result = verify_attestation(None)
    assert result["valid"] is False
    assert "No attestation" in result["reason"]


def test_verify_empty_attestation():
    result = verify_attestation("")
    assert result["valid"] is False


def test_verify_garbled_attestation():
    result = verify_attestation("not-valid!!!")
    assert result["valid"] is False


def test_verify_missing_field():
    # Remove 'sig' field
    att = _make_attestation({"sig": None})
    # Remove sig entirely by building manually
    record = {
        "v": 1,
        "alg": "ECDSA-P256-SHA256",
        "kid": "deadbeef",
        "shader_version": "0.1.0",
        "embedding_hash": "a" * 64,
        # sig intentionally omitted
    }
    att = base64.urlsafe_b64encode(json.dumps(record).encode()).decode().rstrip("=")
    result = verify_attestation(att)
    assert result["valid"] is False
    assert "Missing fields" in result["reason"]


def test_verify_wrong_version():
    att = _make_attestation({"v": 99})
    result = verify_attestation(att)
    assert result["valid"] is False
    assert "version" in result["reason"].lower()


def test_verify_wrong_algorithm():
    att = _make_attestation({"alg": "RSA-SHA256"})
    result = verify_attestation(att)
    assert result["valid"] is False
    assert "algorithm" in result["reason"].lower()


def test_verify_bad_embedding_hash_length():
    att = _make_attestation({"embedding_hash": "abc123"})  # too short
    result = verify_attestation(att)
    assert result["valid"] is False
    assert "embedding_hash" in result["reason"]


def test_verify_bad_kid_format():
    att = _make_attestation({"kid": "ZZZZZZZZ"})  # not hex
    result = verify_attestation(att)
    assert result["valid"] is False
    assert "kid" in result["reason"]


def test_verify_shader_version_mismatch():
    att = _make_attestation({"shader_version": "9.9.9"})
    result = verify_attestation(att, expected_shader_version="0.1.0")
    assert result["valid"] is False
    assert "mismatch" in result["reason"]


def test_verify_shader_version_matches():
    att = _make_attestation({"shader_version": "0.1.0"})
    result = verify_attestation(att, expected_shader_version="0.1.0")
    assert result["valid"] is True


def test_verify_returns_record_on_success():
    att = _make_attestation()
    result = verify_attestation(att)
    assert isinstance(result["record"], dict)
    assert result["record"]["alg"] == "ECDSA-P256-SHA256"
