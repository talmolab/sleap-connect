"""Pre-Shared Key (PSK) authentication utilities for P2P verification.

This module provides cryptographic primitives for the PSK challenge-response
protocol used to verify peers over WebRTC data channels.

The protocol:
1. Worker generates a random nonce and sends AUTH_CHALLENGE
2. Client computes HMAC-SHA256(secret, nonce) and sends AUTH_RESPONSE
3. Worker verifies the HMAC matches

The secret is distributed out-of-band (shared filesystem, env var, config file)
and never touches the signaling server.
"""

import base64
import hashlib
import hmac
import secrets


def generate_secret() -> str:
    """Generate a cryptographically secure room secret.

    Returns:
        A 256-bit (32-byte) random secret, URL-safe base64-encoded.
        The result is 43 characters (no padding).

    Example:
        >>> secret = generate_secret()
        >>> len(secret)
        43
    """
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()


def generate_nonce() -> str:
    """Generate a random nonce for the challenge.

    Returns:
        A 256-bit (32-byte) random nonce, URL-safe base64-encoded.
        The result is 43 characters (no padding).

    Example:
        >>> nonce = generate_nonce()
        >>> len(nonce)
        43
    """
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()


def _decode_base64(data: str) -> bytes:
    """Decode URL-safe base64 string, handling missing padding.

    Args:
        data: URL-safe base64-encoded string (with or without padding).

    Returns:
        Decoded bytes.
    """
    # Add padding if needed (base64 requires length to be multiple of 4)
    padding = 4 - (len(data) % 4)
    if padding != 4:
        data += "=" * padding
    return base64.urlsafe_b64decode(data)


def compute_hmac(secret: str, nonce: str) -> str:
    """Compute HMAC-SHA256 of the nonce using the secret.

    Args:
        secret: The room secret (base64-encoded).
        nonce: The challenge nonce (base64-encoded).

    Returns:
        HMAC-SHA256 digest, URL-safe base64-encoded (43 characters, no padding).

    Example:
        >>> secret = generate_secret()
        >>> nonce = generate_nonce()
        >>> mac = compute_hmac(secret, nonce)
        >>> len(mac)
        43
    """
    secret_bytes = _decode_base64(secret)
    nonce_bytes = _decode_base64(nonce)

    digest = hmac.new(secret_bytes, nonce_bytes, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def verify_hmac(secret: str, nonce: str, received_hmac: str) -> bool:
    """Verify that the received HMAC matches the expected value.

    Uses constant-time comparison to prevent timing attacks.

    Args:
        secret: The room secret (base64-encoded).
        nonce: The challenge nonce (base64-encoded).
        received_hmac: The HMAC received from the peer (base64-encoded).

    Returns:
        True if the HMAC is valid, False otherwise.

    Example:
        >>> secret = generate_secret()
        >>> nonce = generate_nonce()
        >>> mac = compute_hmac(secret, nonce)
        >>> verify_hmac(secret, nonce, mac)
        True
        >>> verify_hmac(secret, nonce, "wrong_hmac")
        False
    """
    expected_hmac = compute_hmac(secret, nonce)
    return hmac.compare_digest(expected_hmac, received_hmac)
