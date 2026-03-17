"""Ed25519 event signing for RANKIGI passports.

Requires the ``cryptography`` library: ``pip install rankigi[signing]``
"""

from __future__ import annotations

import base64
import json
from typing import Any, Dict


def _canonical_json(value: Any) -> str:
    """Deterministic JSON with sorted keys and compact separators."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sign_payload(private_key_b64: str, payload_dict: Dict[str, Any]) -> str:
    """Sign a canonical JSON payload with an Ed25519 private key.

    Parameters
    ----------
    private_key_b64 : str
        Base64-encoded DER pkcs8 Ed25519 private key.
    payload_dict : dict
        The payload to sign (will be canonicalized).

    Returns
    -------
    str
        Base64-encoded Ed25519 signature.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import load_der_private_key

    der_bytes = base64.b64decode(private_key_b64)
    private_key = load_der_private_key(der_bytes, password=None)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError("Key is not an Ed25519 private key")

    canonical = _canonical_json(payload_dict)
    signature = private_key.sign(canonical.encode("utf-8"))
    return base64.b64encode(signature).decode("ascii")
