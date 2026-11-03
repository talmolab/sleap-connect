"""E2E encryption for the relay transport path.

Provides ECDH P-256 key exchange and AES-256-GCM message encryption so that
relay messages between dashboard/sleap-app clients and workers are encrypted
end-to-end. The signaling server can route messages but cannot read payloads.
"""

from sleap_connect.encryption.ecdh import (
    decrypt,
    derive_shared_key,
    encrypt,
    generate_keypair,
    public_key_from_b64,
    public_key_to_b64,
)
from sleap_connect.encryption.envelope import ENCRYPTED_RELAY_TYPE, unwrap, wrap

__all__ = [
    "ENCRYPTED_RELAY_TYPE",
    "decrypt",
    "derive_shared_key",
    "encrypt",
    "generate_keypair",
    "public_key_from_b64",
    "public_key_to_b64",
    "unwrap",
    "wrap",
]
