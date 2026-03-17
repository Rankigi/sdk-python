"""Intent Chain AES-256-GCM encryption for RANKIGI Python SDK.

Requires the ``cryptography`` package: pip install rankigi[intent]
"""

from __future__ import annotations

import hashlib
import os
from base64 import b64encode
from typing import Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def encrypt_intent(plaintext: str, key_hex: str) -> Tuple[str, str]:
    """Encrypt agent reasoning with AES-256-GCM and return (packed, intent_hash).

    Parameters
    ----------
    plaintext : str
        Agent reasoning text.
    key_hex : str
        AES-256 key as 64-char hex string.

    Returns
    -------
    tuple of (packed, intent_hash)
        packed: ``iv_hex:tag_hex:ciphertext_b64``
        intent_hash: SHA-256 hex digest of packed string
    """
    key = bytes.fromhex(key_hex)
    if len(key) != 32:
        raise ValueError("AES-256 key must be 32 bytes (64 hex chars)")

    nonce = os.urandom(12)
    aesgcm = AESGCM(key)

    # AESGCM.encrypt returns ciphertext + 16-byte tag appended
    ct_with_tag = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
    ciphertext = ct_with_tag[:-16]
    tag = ct_with_tag[-16:]

    packed = f"{nonce.hex()}:{tag.hex()}:{b64encode(ciphertext).decode('ascii')}"
    intent_hash = hashlib.sha256(packed.encode("utf-8")).hexdigest()

    return packed, intent_hash
