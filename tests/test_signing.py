"""Tests for Ed25519 passport signing."""

import base64
import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

from rankigi._signing import sign_payload, _canonical_json


@pytest.fixture
def ed25519_keypair():
    private_key = Ed25519PrivateKey.generate()
    private_der = private_key.private_bytes(Encoding.DER, PrivateFormat.PKCS8, NoEncryption())
    public_der = private_key.public_key().public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)
    return {
        "private_b64": base64.b64encode(private_der).decode(),
        "public_b64": base64.b64encode(public_der).decode(),
        "private_key": private_key,
    }


SAMPLE_PAYLOAD = {
    "agent_id": "test-uuid",
    "action": "tool_call",
    "tool": "web_search",
}


def test_sign_payload_returns_base64(ed25519_keypair):
    sig = sign_payload(ed25519_keypair["private_b64"], SAMPLE_PAYLOAD)
    # Should be valid base64
    decoded = base64.b64decode(sig)
    assert len(decoded) > 0


def test_signature_is_64_bytes(ed25519_keypair):
    sig = sign_payload(ed25519_keypair["private_b64"], SAMPLE_PAYLOAD)
    decoded = base64.b64decode(sig)
    assert len(decoded) == 64


def test_signature_verifies_with_matching_key(ed25519_keypair):
    sig = sign_payload(ed25519_keypair["private_b64"], SAMPLE_PAYLOAD)
    sig_bytes = base64.b64decode(sig)
    canonical = _canonical_json(SAMPLE_PAYLOAD)
    # Should not raise
    ed25519_keypair["private_key"].public_key().verify(sig_bytes, canonical.encode("utf-8"))


def test_signature_fails_with_different_key(ed25519_keypair):
    sig = sign_payload(ed25519_keypair["private_b64"], SAMPLE_PAYLOAD)
    sig_bytes = base64.b64decode(sig)
    canonical = _canonical_json(SAMPLE_PAYLOAD)

    other_key = Ed25519PrivateKey.generate()
    with pytest.raises(Exception):
        other_key.public_key().verify(sig_bytes, canonical.encode("utf-8"))


def test_canonical_json_determinism():
    payload_a = {"z": 1, "a": 2, "m": {"b": 3, "a": 4}}
    payload_b = {"a": 2, "m": {"a": 4, "b": 3}, "z": 1}
    assert _canonical_json(payload_a) == _canonical_json(payload_b)
    # Keys should be sorted
    result = _canonical_json(payload_a)
    parsed_keys = list(json.loads(result).keys())
    assert parsed_keys == sorted(parsed_keys)
