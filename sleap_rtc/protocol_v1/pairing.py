"""Pairing-ticket generation and validation.

A pairing ticket is a short-lived, one-time secret that lets a new client
establish trust with this worker on first contact (protocol spec §3.2).

By default, pending tickets live in memory only — fine when the same
process both mints and validates them (e.g. in tests, or a single-process
embedding). When a `path` is given, pending secrets are additionally
persisted to a small JSON file: this is what lets a *separate* CLI
invocation (``sleap-rtc pair``) mint a ticket that an already-running
``sleap-rtc serve`` process — a different OS process entirely — will
recognize, by having both point at the same file. Secrets are still
short-lived (5 min default) and single-use either way; persisting them
doesn't change that, it just makes them visible across processes.
"""

import json
import os
import secrets
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

# 5 minutes, per protocol spec §3.2.
DEFAULT_TICKET_TTL_SECS = 300


@dataclass
class PairingTicket:
    """A one-time pairing ticket, as printed/QR-coded for a new client.

    Attributes:
        node_id: This worker's public identity.
        addrs: Direct address(es) the client can dial (e.g. ``ws://host:port``).
        secret: One-time secret proving possession of this ticket.
        expires_at: Unix timestamp after which `secret` is no longer valid.
        iroh: Optional iroh dial info (``node_id``, ``relay_url``,
            ``direct_addrs`` — see `iroh_live.build_iroh_section`) for a
            client that can't reach `addrs` directly. Omitted from the
            serialized ticket entirely when ``None``, so tickets from a
            worker not serving iroh are unchanged.
    """

    node_id: str
    addrs: List[str]
    secret: str
    expires_at: float
    iroh: Optional[dict] = None

    def to_dict(self) -> dict:
        """Serialize to the ticket's wire/QR-code shape."""
        data = {
            "node_id": self.node_id,
            "addrs": self.addrs,
            "secret": self.secret,
            "expires_at": self.expires_at,
        }
        if self.iroh is not None:
            data["iroh"] = self.iroh
        return data


class PendingPairings:
    """Tracks outstanding, not-yet-claimed pairing secrets.

    In-memory only unless `path` is given (see module docstring).
    """

    def __init__(
        self,
        ttl_secs: float = DEFAULT_TICKET_TTL_SECS,
        path: Optional[Union[str, Path]] = None,
    ):
        """Initialize with no pending tickets (or load them from `path`).

        Args:
            ttl_secs: How long a generated ticket's secret remains valid.
            path: If given, persist pending secrets here so a separate
                process can mint/claim tickets against the same set.
        """
        self._ttl_secs = ttl_secs
        self._path = Path(path) if path is not None else None
        self._secrets: Dict[str, float] = self._load() if self._path else {}

    def create(
        self, node_id: str, addrs: List[str], iroh: Optional[dict] = None
    ) -> PairingTicket:
        """Generate a new one-time pairing ticket.

        Args:
            node_id: This worker's public identity, to embed in the ticket.
            addrs: Direct address(es) the client should dial.
            iroh: Optional iroh dial info to embed in the ticket.

        Returns:
            The new `PairingTicket`.
        """
        if self._path is not None:
            self._secrets = self._load()  # pick up secrets from other processes

        # 16 random bytes (128-bit), stored as the URL-safe base64 string of
        # those exact bytes — `token_urlsafe(16)` does precisely that (it's
        # `base64.urlsafe_b64encode(token_bytes(16))` with padding
        # stripped), which matters because the one-line pairing code format
        # (protocol_v1.pair_code) embeds this secret as 16 raw bytes and
        # must reconstruct the identical string on decode for `claim` to
        # recognize it.
        secret = secrets.token_urlsafe(16)
        expires_at = time.time() + self._ttl_secs
        self._secrets[secret] = expires_at
        self._prune_expired()

        if self._path is not None:
            self._save()

        return PairingTicket(
            node_id=node_id,
            addrs=addrs,
            secret=secret,
            expires_at=expires_at,
            iroh=iroh,
        )

    def claim(self, secret: str) -> bool:
        """Consume a pairing secret if it's valid and unexpired.

        Single-use: whether it succeeds or fails on expiry, the secret is
        removed and can't be claimed again.

        Args:
            secret: The secret from a `pair.claim` request.

        Returns:
            True if the secret was valid and is now consumed; False if it
            was unknown or had already expired.
        """
        if self._path is not None:
            self._secrets = self._load()  # pick up secrets from other processes

        expires_at = self._secrets.pop(secret, None)

        if self._path is not None:
            self._save()

        if expires_at is None:
            return False
        return time.time() <= expires_at

    def _prune_expired(self) -> None:
        now = time.time()
        self._secrets = {s: e for s, e in self._secrets.items() if e >= now}

    def _load(self) -> Dict[str, float]:
        if not self._path.exists():
            return {}
        data = json.loads(self._path.read_text())
        return {s: e for s, e in data.get("pending", {}).items() if e >= time.time()}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._path.with_suffix(".tmp")
        temp_path.write_text(json.dumps({"pending": self._secrets}))
        os.chmod(temp_path, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(temp_path, self._path)
