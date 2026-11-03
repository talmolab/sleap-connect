"""Ed25519 keypair utilities for P2P authentication.

This module provides key generation, signing, and verification for the
Ed25519-based P2P challenge-response protocol.

Protocol:
1. Worker sends AUTH_CHALLENGE::<nonce>
2. Client signs nonce with private key, sends AUTH_RESPONSE::<signature_b64>
3. Worker fetches authorized public keys from server, verifies signature
"""

import base64

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)


def generate_keypair() -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    """Generate a new Ed25519 keypair.

    Returns:
        Tuple of (private_key, public_key).
    """
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key


def sign_nonce(private_key: Ed25519PrivateKey, nonce: str) -> str:
    """Sign a nonce with the private key.

    Args:
        private_key: Ed25519 private key.
        nonce: Challenge nonce string from the worker.

    Returns:
        URL-safe base64-encoded signature (no padding).
    """
    signature = private_key.sign(nonce.encode())
    return base64.urlsafe_b64encode(signature).rstrip(b"=").decode()


def verify_signature(
    public_key: Ed25519PublicKey, nonce: str, signature_b64: str
) -> bool:
    """Verify a nonce signature against a public key.

    Args:
        public_key: Ed25519 public key.
        nonce: The original challenge nonce.
        signature_b64: URL-safe base64-encoded signature to verify.

    Returns:
        True if signature is valid, False otherwise.
    """
    try:
        padding = 4 - (len(signature_b64) % 4)
        if padding != 4:
            signature_b64 += "=" * padding
        sig_bytes = base64.urlsafe_b64decode(signature_b64)
        public_key.verify(sig_bytes, nonce.encode())
        return True
    except InvalidSignature:
        return False


def private_key_to_b64(private_key: Ed25519PrivateKey) -> str:
    """Serialize a private key to URL-safe base64 (raw bytes, no padding).

    Args:
        private_key: Ed25519 private key.

    Returns:
        URL-safe base64-encoded private key bytes.
    """
    raw = private_key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def public_key_to_b64(public_key: Ed25519PublicKey) -> str:
    """Serialize a public key to URL-safe base64 (raw bytes, no padding).

    Args:
        public_key: Ed25519 public key.

    Returns:
        URL-safe base64-encoded public key bytes.
    """
    raw = public_key.public_bytes(Encoding.Raw, PublicFormat.Raw)
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def private_key_from_b64(b64: str) -> Ed25519PrivateKey:
    """Restore a private key from URL-safe base64.

    Args:
        b64: URL-safe base64-encoded private key bytes.

    Returns:
        Ed25519PrivateKey.
    """
    padding = 4 - (len(b64) % 4)
    if padding != 4:
        b64 += "=" * padding
    raw = base64.urlsafe_b64decode(b64)
    return Ed25519PrivateKey.from_private_bytes(raw)


def public_key_from_b64(b64: str) -> Ed25519PublicKey:
    """Restore a public key from URL-safe base64.

    Args:
        b64: URL-safe base64-encoded public key bytes.

    Returns:
        Ed25519PublicKey.
    """
    padding = 4 - (len(b64) % 4)
    if padding != 4:
        b64 += "=" * padding
    raw = base64.urlsafe_b64decode(b64)
    return Ed25519PublicKey.from_public_bytes(raw)
