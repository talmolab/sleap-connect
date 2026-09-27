"""In-memory pairing-ticket generation and validation.

A pairing ticket is a short-lived, one-time secret that lets a new client
establish trust with this worker on first contact (protocol spec §3.2).
Tickets are intentionally NOT persisted — they're meant to be used within
minutes of being generated (e.g. scanned from a QR code); if the worker
restarts mid-pairing-window, the operator just generates a new one.
"""

import secrets
import time
from dataclasses import dataclass
from typing import Dict, List

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
    """

    node_id: str
    addrs: List[str]
    secret: str
    expires_at: float

    def to_dict(self) -> dict:
        """Serialize to the ticket's wire/QR-code shape."""
        return {
            "node_id": self.node_id,
            "addrs": self.addrs,
            "secret": self.secret,
            "expires_at": self.expires_at,
        }


class PendingPairings:
    """Tracks outstanding, not-yet-claimed pairing secrets, in memory only."""

    def __init__(self, ttl_secs: float = DEFAULT_TICKET_TTL_SECS):
        """Initialize with no pending tickets.

        Args:
            ttl_secs: How long a generated ticket's secret remains valid.
        """
        self._ttl_secs = ttl_secs
        self._secrets: Dict[str, float] = {}  # secret -> expires_at

    def create(self, node_id: str, addrs: List[str]) -> PairingTicket:
        """Generate a new one-time pairing ticket.

        Args:
            node_id: This worker's public identity, to embed in the ticket.
            addrs: Direct address(es) the client should dial.

        Returns:
            The new `PairingTicket`.
        """
        secret = secrets.token_urlsafe(24)
        expires_at = time.time() + self._ttl_secs
        self._secrets[secret] = expires_at
        return PairingTicket(
            node_id=node_id, addrs=addrs, secret=secret, expires_at=expires_at
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
        expires_at = self._secrets.pop(secret, None)
        if expires_at is None:
            return False
        return time.time() <= expires_at
