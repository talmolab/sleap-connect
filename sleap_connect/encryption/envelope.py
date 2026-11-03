"""Encrypted relay message envelope for E2E encryption.

Wraps and unwraps relay messages into encrypted envelopes that preserve
plaintext routing fields (type, session_id, job_id, req_id) while encrypting
the actual message payload.

Envelope format (what the signaling server sees):
{
    "type": "encrypted_relay",
    "session_id": "<uuid>",
    "job_id": "<optional, for relay channel routing>",
    "req_id": "<optional, for request matching>",
    "nonce": "<base64, 12 bytes>",
    "ciphertext": "<base64, AES-GCM encrypted payload>"
}

The signaling server can route based on the plaintext fields but cannot
read the encrypted payload.
"""

import base64
import logging

from cryptography.exceptions import InvalidTag

from sleap_connect.encryption.ecdh import decrypt, encrypt

logger = logging.getLogger(__name__)

# Message type used for encrypted relay envelopes.
ENCRYPTED_RELAY_TYPE = "encrypted_relay"


def wrap(
    key: bytes,
    session_id: str,
    message: dict,
    job_id: str | None = None,
    req_id: str | None = None,
) -> dict:
    """Encrypt a message and wrap it in a relay-compatible envelope.

    Args:
        key: 32-byte AES key from ECDH key exchange.
        session_id: Session identifier for key lookup on the receiving side.
        message: The message dict to encrypt.
        job_id: Optional job ID for relay channel routing (stays plaintext).
        req_id: Optional request ID for request/response matching (stays plaintext).

    Returns:
        Encrypted envelope dict with plaintext routing fields.
    """
    nonce, ciphertext = encrypt(key, message)

    envelope = {
        "type": ENCRYPTED_RELAY_TYPE,
        "session_id": session_id,
        "nonce": base64.b64encode(nonce).decode(),
        "ciphertext": base64.b64encode(ciphertext).decode(),
    }
    if job_id is not None:
        envelope["job_id"] = job_id
    if req_id is not None:
        envelope["req_id"] = req_id
    return envelope


def unwrap(
    envelope: dict,
    key_lookup: dict[str, bytes],
) -> dict | None:
    """Decrypt an encrypted relay envelope back to the original message.

    Args:
        envelope: Encrypted envelope dict (must have type="encrypted_relay").
        key_lookup: Dict mapping session_id → AES key.

    Returns:
        Decrypted message dict, or None if decryption failed (unknown session,
        wrong key, or tampered message). Failures are logged but not raised.
    """
    session_id = envelope.get("session_id")
    if not session_id:
        logger.warning("Encrypted relay message missing session_id, discarding")
        return None

    key = key_lookup.get(session_id)
    if key is None:
        logger.debug("Unknown session_id %s, discarding encrypted message", session_id)
        return None

    try:
        nonce = base64.b64decode(envelope["nonce"])
        ciphertext = base64.b64decode(envelope["ciphertext"])
    except (KeyError, Exception) as e:
        logger.warning("Malformed encrypted envelope: %s", e)
        return None

    try:
        return decrypt(key, nonce, ciphertext)
    except InvalidTag:
        logger.warning(
            "Decryption failed for session %s (invalid tag), discarding", session_id
        )
        return None
