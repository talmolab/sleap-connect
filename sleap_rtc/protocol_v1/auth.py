"""pair.claim / auth.prove method handlers for the protocol v1 server.

See protocol spec §3.2-3.3. **Implements the client-proves-to-worker
direction only.** The spec's symmetric option (the worker also proving
itself to the client) is deferred for two reasons: (1) it would need
outgoing-request infrastructure this server doesn't have yet — today it
only dispatches incoming client requests, it never issues its own `req` and
correlates the reply; (2) it protects against a *spoofed worker*, not
against an *untrusted client* — a real gap, but a lower-priority one than
what this closes, since the worker is what executes jobs and exposes
mounted filesystem paths. Revisit once the server needs outgoing requests
for another reason anyway.

Trust model: `pair.claim` is how a brand-new client establishes trust on
first contact (via a one-time secret from a pairing ticket — see
`pairing.py`). Once trusted, `auth.prove` is how that same client
re-authenticates on every later connection, by signing this connection's
`hello` nonce with the private key matching the `node_id` it already
claimed.
"""

from sleap_rtc.auth.keypair import public_key_from_b64, verify_signature
from sleap_rtc.protocol_v1.errors import (
    AUTH_BAD_SIGNATURE,
    AUTH_PAIRING_EXPIRED,
    AUTH_REQUIRED,
    AUTH_UNTRUSTED,
    ProtocolError,
)
from sleap_rtc.protocol_v1.identity import WorkerIdentity
from sleap_rtc.protocol_v1.pairing import PendingPairings
from sleap_rtc.protocol_v1.server import Connection, ProtocolV1Server
from sleap_rtc.protocol_v1.trust_store import TrustStore


class AuthMethods:
    """Registers `pair.claim` and `auth.prove` on a server."""

    def __init__(
        self,
        server: ProtocolV1Server,
        identity: WorkerIdentity,
        trust_store: TrustStore,
        pending_pairings: PendingPairings,
    ):
        """Wire up and register the auth method handlers.

        Args:
            server: The `ProtocolV1Server` to register methods on.
            identity: This worker's persistent Ed25519 identity.
            trust_store: The durable set of paired client node_ids.
            pending_pairings: Outstanding, not-yet-claimed pairing secrets.
        """
        self.server = server
        self.identity = identity
        self.trust_store = trust_store
        self.pending_pairings = pending_pairings

        server.register("pair.claim", self.pair_claim)
        server.register("auth.prove", self.auth_prove)

    async def pair_claim(self, params: dict, conn: Connection) -> dict:
        """Handle `pair.claim` — first-contact trust via a one-time secret.

        On success, this connection is *also* immediately authenticated
        (per spec §3.2: pairing establishes trust, it doesn't require a
        separate `auth.prove` round-trip on the very same connection) —
        subsequent reconnects use `auth.prove` instead.
        """
        secret = params["secret"]
        node_id = params["node_id"]

        if not self.pending_pairings.claim(secret):
            raise ProtocolError(
                AUTH_PAIRING_EXPIRED, "Pairing secret is invalid or already used"
            )

        self.trust_store.add_trusted(node_id)
        conn.peer_node_id = node_id
        conn.authenticated = True
        return {}

    async def auth_prove(self, params: dict, conn: Connection) -> dict:
        """Handle `auth.prove` — an already-paired client re-proving its identity.

        Verifies `params["sig"]` is this connection's claimed node_id
        signing *our* nonce from this connection's `hello` (see
        `server.Connection.own_nonce`), then checks that node_id is
        actually in the trust store.
        """
        if conn.peer_node_id is None or conn.own_nonce is None:
            raise ProtocolError(
                AUTH_REQUIRED, "auth.prove requires a completed hello handshake first"
            )

        public_key = public_key_from_b64(conn.peer_node_id)
        if not verify_signature(public_key, conn.own_nonce, params["sig"]):
            raise ProtocolError(
                AUTH_BAD_SIGNATURE, "Signature does not match this connection's nonce"
            )

        if not self.trust_store.is_trusted(conn.peer_node_id):
            raise ProtocolError(
                AUTH_UNTRUSTED,
                f"Node {conn.peer_node_id!r} is not paired with this worker",
            )

        conn.authenticated = True
        return {}
