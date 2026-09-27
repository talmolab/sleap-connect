"""Integration tests for pair.claim / auth.prove against a real WS server.

Uses real Ed25519 keypairs (via sleap_rtc.auth.keypair) for the test
"client" side, and a real WorkerIdentity/TrustStore/PendingPairings backing
a real ProtocolV1Server — the whole point is verifying real signatures
against a real nonce exchanged over a real connection.
"""

import pytest
import websockets

from sleap_rtc.auth.keypair import generate_keypair, public_key_to_b64, sign_nonce
from sleap_rtc.protocol_v1.auth import AuthMethods
from sleap_rtc.protocol_v1.envelope import Hello, Req, parse_envelope
from sleap_rtc.protocol_v1.identity import WorkerIdentity
from sleap_rtc.protocol_v1.pairing import PendingPairings
from sleap_rtc.protocol_v1.server import ProtocolV1Server
from sleap_rtc.protocol_v1.trust_store import TrustStore


@pytest.fixture
def identity(tmp_path):
    return WorkerIdentity(tmp_path / "identity.json")


@pytest.fixture
def trust_store(tmp_path):
    return TrustStore(tmp_path / "trusted.json")


@pytest.fixture
def pending_pairings():
    return PendingPairings()


@pytest.fixture
async def running_server(identity, trust_store, pending_pairings):
    """A ProtocolV1Server with AuthMethods wired in, bound to a free port."""
    server_obj = ProtocolV1Server(node_id=identity.node_id)
    AuthMethods(server_obj, identity, trust_store, pending_pairings)

    # A protected method to prove the auth gate actually blocks/unblocks it.
    async def echo(params, conn):
        return {"echoed": params}

    server_obj.register("test.echo", echo)

    ws_server = await server_obj.serve("127.0.0.1", 0)
    port = ws_server.sockets[0].getsockname()[1]
    try:
        yield server_obj, port
    finally:
        ws_server.close()
        await ws_server.wait_closed()


def _client_identity():
    """Generate a fresh Ed25519 keypair for a test 'client'."""
    private_key, public_key = generate_keypair()
    node_id = public_key_to_b64(public_key)
    return private_key, node_id


async def _hello(port, node_id, nonce="client-nonce"):
    """Connect and exchange hello; return (ws, server's hello frame)."""
    ws = await websockets.connect(f"ws://127.0.0.1:{port}")
    await ws.send(
        Hello(
            proto={"min": 1, "max": 1},
            agent={"name": "test-client", "version": "0.0.0", "platform": "test"},
            node_id=node_id,
            nonce=nonce,
        ).to_json()
    )
    server_hello = parse_envelope(await ws.recv())
    assert isinstance(server_hello, Hello)
    return ws, server_hello


async def _call(ws, req_id, method, params=None):
    await ws.send(Req(id=req_id, method=method, params=params or {}).to_json())
    return parse_envelope(await ws.recv())


class TestUnauthenticatedGate:
    """Tests that non-auth methods are blocked before authentication."""

    async def test_protected_method_is_blocked_before_auth(self, running_server):
        _server, port = running_server
        _priv, node_id = _client_identity()
        ws, _server_hello = await _hello(port, node_id)

        reply = await _call(ws, 1, "test.echo", {"x": 1})

        assert reply.error["code"] == "auth.required"
        await ws.close()


class TestPairClaim:
    """Tests for the pair.claim flow."""

    async def test_valid_secret_authenticates_and_trusts_the_client(
        self, running_server, identity, trust_store, pending_pairings
    ):
        server_obj, port = running_server
        _priv, node_id = _client_identity()
        ticket = pending_pairings.create(identity.node_id, [f"ws://127.0.0.1:{port}"])
        ws, _server_hello = await _hello(port, node_id)

        reply = await _call(
            ws, 1, "pair.claim", {"secret": ticket.secret, "node_id": node_id}
        )

        assert reply.result == {}
        assert trust_store.is_trusted(node_id) is True

        # Now authenticated on this same connection — protected methods work.
        echo_reply = await _call(ws, 2, "test.echo", {"x": 1})
        assert echo_reply.result == {"echoed": {"x": 1}}
        await ws.close()

    async def test_secret_is_single_use(
        self, running_server, identity, pending_pairings
    ):
        _server, port = running_server
        _priv1, node_id1 = _client_identity()
        _priv2, node_id2 = _client_identity()
        ticket = pending_pairings.create(identity.node_id, [])

        ws1, _ = await _hello(port, node_id1)
        first = await _call(
            ws1, 1, "pair.claim", {"secret": ticket.secret, "node_id": node_id1}
        )
        assert first.result == {}

        ws2, _ = await _hello(port, node_id2)
        second = await _call(
            ws2, 1, "pair.claim", {"secret": ticket.secret, "node_id": node_id2}
        )
        assert second.error["code"] == "auth.pairing_expired"

        await ws1.close()
        await ws2.close()

    async def test_unknown_secret_is_rejected(self, running_server):
        _server, port = running_server
        _priv, node_id = _client_identity()
        ws, _server_hello = await _hello(port, node_id)

        reply = await _call(
            ws, 1, "pair.claim", {"secret": "not-a-real-secret", "node_id": node_id}
        )

        assert reply.error["code"] == "auth.pairing_expired"
        await ws.close()


class TestAuthProve:
    """Tests for the auth.prove flow (already-paired client reconnecting)."""

    async def test_correct_signature_from_a_trusted_node_authenticates(
        self, running_server, trust_store
    ):
        _server, port = running_server
        priv, node_id = _client_identity()
        trust_store.add_trusted(node_id)
        ws, server_hello = await _hello(port, node_id)

        sig = sign_nonce(priv, server_hello.nonce)
        reply = await _call(ws, 1, "auth.prove", {"sig": sig})

        assert reply.result == {}

        echo_reply = await _call(ws, 2, "test.echo", {"x": 1})
        assert echo_reply.result == {"echoed": {"x": 1}}
        await ws.close()

    async def test_wrong_signature_is_rejected(self, running_server, trust_store):
        _server, port = running_server
        _priv, node_id = _client_identity()
        trust_store.add_trusted(node_id)
        ws, _server_hello = await _hello(port, node_id)

        # Sign the wrong thing entirely — not this connection's nonce.
        wrong_priv, _ = _client_identity()
        bad_sig = sign_nonce(wrong_priv, "some-other-value")
        reply = await _call(ws, 1, "auth.prove", {"sig": bad_sig})

        assert reply.error["code"] == "auth.bad_signature"
        await ws.close()

    async def test_correct_signature_from_an_untrusted_node_is_rejected(
        self, running_server
    ):
        _server, port = running_server
        priv, node_id = _client_identity()  # never added to trust_store
        ws, server_hello = await _hello(port, node_id)

        sig = sign_nonce(priv, server_hello.nonce)
        reply = await _call(ws, 1, "auth.prove", {"sig": sig})

        assert reply.error["code"] == "auth.untrusted"
        await ws.close()
