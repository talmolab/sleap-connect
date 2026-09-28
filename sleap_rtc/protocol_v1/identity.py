"""Persistent worker identity (Ed25519 keypair) for protocol v1 pairing/auth.

Reuses the existing `sleap_rtc.auth.keypair` primitives (already used by the
legacy P2P challenge-response protocol) rather than reimplementing Ed25519
handling. What's new here is just persistence of a single long-lived
identity keypair, separate from the legacy `credentials.py` schema — that
file is account/room-shaped (JWT, account_key, room_secrets) for the
signaling-server model being retired for the 1:1 case; this identity is
pairing-shaped (just a keypair, no account).
"""

import json
import os
import stat
from pathlib import Path
from typing import Union

from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)

from sleap_rtc.auth.keypair import (
    Ed25519PrivateKey,
    generate_keypair,
    private_key_from_b64,
    private_key_to_b64,
    public_key_to_b64,
    sign_nonce,
)

DEFAULT_IDENTITY_PATH = Path.home() / ".sleap-rtc" / "protocol_v1_identity.json"


class WorkerIdentity:
    """A worker's persistent Ed25519 identity.

    Loads an existing keypair from `path` if present, otherwise generates
    one and persists it. `node_id` (the base64-encoded public key) is this
    worker's identity as used in `hello.node_id` and the trust store.
    """

    def __init__(self, path: Union[str, Path] = DEFAULT_IDENTITY_PATH):
        """Load or generate this worker's identity.

        Args:
            path: Where to persist the keypair. Defaults to a per-user path
                under the home directory; tests should pass a `tmp_path`.
        """
        self._path = Path(path)
        self._private_key = self._load_or_generate()

    @property
    def node_id(self) -> str:
        """This worker's public identity — base64-encoded Ed25519 public key."""
        return public_key_to_b64(self._private_key.public_key())

    def sign(self, nonce: str) -> str:
        """Sign a nonce (e.g. the other side's `hello.nonce`) with this identity.

        Args:
            nonce: The nonce string to sign.

        Returns:
            URL-safe base64-encoded signature.
        """
        return sign_nonce(self._private_key, nonce)

    @property
    def iroh_secret_key_bytes(self) -> bytes:
        """This identity's raw 32-byte Ed25519 seed, for `iroh.SecretKey.from_bytes`
        / `EndpointOptions(secret_key=...)` (item 2.2).

        Feeding the *same* keypair to iroh's endpoint makes its `EndpointId`
        numerically identical to `node_id` — one identity for the worker
        across both transports, instead of the client needing to track a
        separate "how do I dial you" id alongside "who are you" for trust.
        """
        return self._private_key.private_bytes(
            Encoding.Raw, PrivateFormat.Raw, NoEncryption()
        )

    def _load_or_generate(self) -> Ed25519PrivateKey:
        if self._path.exists():
            data = json.loads(self._path.read_text())
            return private_key_from_b64(data["private_key"])

        private_key, _ = generate_keypair()
        self._path.parent.mkdir(parents=True, exist_ok=True)

        temp_path = self._path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps({"private_key": private_key_to_b64(private_key)})
        )
        os.chmod(temp_path, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(temp_path, self._path)

        return private_key
