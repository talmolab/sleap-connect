"""ECDH P-256 key exchange and AES-256-GCM encryption for relay E2E encryption.

This module provides key generation, Diffie-Hellman key derivation, and
authenticated encryption for the relay transport path. The P2P data channel
path uses DTLS (built into WebRTC) instead.

Protocol:
1. Client sends key_exchange with ephemeral P-256 public key
2. Worker generates ephemeral P-256 keypair, derives shared AES key via ECDH+HKDF
3. Worker sends key_exchange_response with its public key
4. Client derives same AES key
5. All subsequent relay messages encrypted with AES-256-GCM

Parameters (must match JavaScript Web Crypto API side):
- Curve: P-256 (secp256r1)
- KDF: HKDF-SHA256, no salt, info=b"sleap-rtc-relay-e2e-v1"
- Cipher: AES-256-GCM, 12-byte nonce
"""

import base64
import json
import os

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes, serialization

# HKDF info string — must match the JavaScript side exactly.
HKDF_INFO = b"sleap-rtc-relay-e2e-v1"

# AES-GCM nonce size in bytes (96-bit).
NONCE_SIZE = 12


def generate_keypair() -> tuple[ec.EllipticCurvePrivateKey, bytes]:
    """Generate an ephemeral P-256 keypair for ECDH key exchange.

    Returns:
        Tuple of (private_key, public_key_bytes) where public_key_bytes is the
        uncompressed point encoding (65 bytes: 0x04 || x || y).
    """
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key_bytes = private_key.public_key().public_bytes(
        serialization.Encoding.X962,
        serialization.PublicFormat.UncompressedPoint,
    )
    return private_key, public_key_bytes


def public_key_to_b64(public_key_bytes: bytes) -> str:
    """Encode a public key as URL-safe base64 (no padding).

    Args:
        public_key_bytes: Uncompressed P-256 point (65 bytes).

    Returns:
        URL-safe base64 string without padding.
    """
    return base64.urlsafe_b64encode(public_key_bytes).rstrip(b"=").decode()


def public_key_from_b64(b64: str) -> bytes:
    """Decode a public key from URL-safe base64 (no padding).

    Args:
        b64: URL-safe base64-encoded public key.

    Returns:
        Raw public key bytes (65 bytes uncompressed point).
    """
    padding = 4 - (len(b64) % 4)
    if padding != 4:
        b64 += "=" * padding
    return base64.urlsafe_b64decode(b64)


def derive_shared_key(
    private_key: ec.EllipticCurvePrivateKey,
    peer_public_key_bytes: bytes,
) -> bytes:
    """Derive a 256-bit AES key from ECDH shared secret + HKDF.

    Both sides must call this with the other's public key to arrive at the
    same derived key.

    Args:
        private_key: Our P-256 private key.
        peer_public_key_bytes: Peer's public key (uncompressed point, 65 bytes).

    Returns:
        32-byte AES-256 key.
    """
    peer_public_key = ec.EllipticCurvePublicKey.from_encoded_point(
        ec.SECP256R1(), peer_public_key_bytes
    )
    shared_secret = private_key.exchange(ec.ECDH(), peer_public_key)

    derived_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=HKDF_INFO,
    ).derive(shared_secret)

    return derived_key


def encrypt(key: bytes, plaintext: dict) -> tuple[bytes, bytes]:
    """Encrypt a JSON-serializable dict with AES-256-GCM.

    Args:
        key: 32-byte AES key from derive_shared_key().
        plaintext: Dict to encrypt (will be JSON-serialized).

    Returns:
        Tuple of (nonce, ciphertext) as raw bytes. The ciphertext includes
        the 16-byte GCM authentication tag appended.
    """
    aesgcm = AESGCM(key)
    nonce = os.urandom(NONCE_SIZE)
    plaintext_bytes = json.dumps(plaintext, separators=(",", ":")).encode("utf-8")
    ciphertext = aesgcm.encrypt(nonce, plaintext_bytes, None)
    return nonce, ciphertext


def decrypt(key: bytes, nonce: bytes, ciphertext: bytes) -> dict:
    """Decrypt an AES-256-GCM ciphertext back to a dict.

    Args:
        key: 32-byte AES key from derive_shared_key().
        nonce: 12-byte nonce used during encryption.
        ciphertext: Ciphertext with appended GCM auth tag.

    Returns:
        Decrypted JSON dict.

    Raises:
        cryptography.exceptions.InvalidTag: If decryption fails (wrong key,
            tampered ciphertext, or wrong nonce).
    """
    aesgcm = AESGCM(key)
    plaintext_bytes = aesgcm.decrypt(nonce, ciphertext, None)
    return json.loads(plaintext_bytes.decode("utf-8"))
