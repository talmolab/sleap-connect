"""Integration tests for the iroh transport binding — a real iroh
Endpoint, a real client Endpoint, real QUIC streams.

Uses `iroh.preset_minimal()` (relay disabled, direct addresses only) for
both endpoints so these tests have no external network dependency and no
real relay/n0 involvement — the default preset's `online()` reaches out to
a real internet relay to determine reachability, which is not something a
test suite should depend on. `preset_minimal`'s own docstring calls it out
as "no external dependencies; good for tests / offline".

`ProtocolV1Server.serve_iroh` drives every connection through the exact
same `_do_hello`/`_dispatch` logic the WS transport uses (see
`test_server.py`) — these tests exist to prove the iroh-specific plumbing
(the length-prefixed framing, `IrohStreamTransport`, the accept loop)
actually carries that logic correctly over a real QUIC connection, not to
re-test the dispatch logic itself.
"""

import asyncio
import struct

import iroh
import pytest

from sleap_rtc.auth.keypair import generate_keypair, public_key_to_b64
from sleap_rtc.protocol_v1.auth import AuthMethods
from sleap_rtc.protocol_v1.envelope import Hello, Req, parse_envelope
from sleap_rtc.protocol_v1.errors import ProtocolError
from sleap_rtc.protocol_v1.identity import WorkerIdentity
from sleap_rtc.protocol_v1.iroh_transport import (
    ALPN,
    FrameTooLarge,
    IrohStreamTransport,
    read_frame,
)
from sleap_rtc.protocol_v1.pairing import PendingPairings
from sleap_rtc.protocol_v1.server import (
    DEFAULT_WS_MAX_SIZE,
    Connection,
    ProtocolV1Server,
    TransportClosed,
)
from sleap_rtc.protocol_v1.trust_store import TrustStore

_MINIMAL = iroh.preset_minimal


async def _wait_for_direct_addresses(endpoint, timeout=5.0):
    """Poll until `endpoint.addr()` reports at least one direct address.

    Stands in for `online()`, which hangs forever with relay disabled
    (it specifically waits for a usable *relay*, which `preset_minimal`
    has none of) — direct addresses show up almost immediately without it.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if endpoint.addr().direct_addresses():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("endpoint never reported a direct address")


@pytest.fixture
async def running_iroh_server(tmp_path):
    """A ProtocolV1Server (with pairing/auth wired in) on a real local iroh endpoint."""
    identity = WorkerIdentity(tmp_path / "identity.json")
    trust_store = TrustStore(tmp_path / "trusted.json")
    pending_pairings = PendingPairings()
    server_obj = ProtocolV1Server(node_id=identity.node_id)
    AuthMethods(server_obj, identity, trust_store, pending_pairings)

    endpoint = await iroh.Endpoint.bind(
        iroh.EndpointOptions(preset=_MINIMAL(), alpns=[ALPN])
    )
    await _wait_for_direct_addresses(endpoint)
    serve_task = asyncio.create_task(server_obj.serve_iroh(endpoint))
    try:
        yield server_obj, endpoint, pending_pairings, identity
    finally:
        await endpoint.close()
        serve_task.cancel()


async def _dial_and_hello(server_addr) -> tuple:
    """Dial the server, complete the iroh handshake and hello exchange.

    Returns `(transport, client_ep, client_node_id)` — *not* paired/
    authenticated yet, for tests of the pre-auth surface specifically.
    """
    client_ep = await iroh.Endpoint.bind(iroh.EndpointOptions(preset=_MINIMAL()))
    _priv, public_key = generate_keypair()
    client_node_id = public_key_to_b64(public_key)
    conn = await client_ep.connect(server_addr, ALPN)
    bi = await conn.open_bi()
    transport = IrohStreamTransport(conn, bi)

    await transport.send(
        Hello(
            proto={"min": 1, "max": 1},
            agent={"name": "test-client", "version": "0.0.0", "platform": "test"},
            node_id=client_node_id,
            nonce="client-nonce",
        ).to_json()
    )
    reply = parse_envelope(await transport.recv())
    assert isinstance(reply, Hello)
    return transport, client_ep, client_node_id


async def _connect_and_pair(server_addr, pending_pairings, identity) -> tuple:
    """Dial, hello, and complete a real pair.claim. Returns `(transport, client_ep)`."""
    transport, client_ep, client_node_id = await _dial_and_hello(server_addr)

    ticket = pending_pairings.create(identity.node_id, [])
    await transport.send(
        Req(
            id=0,
            method="pair.claim",
            params={"secret": ticket.secret, "node_id": client_node_id},
        ).to_json()
    )
    pair_reply = parse_envelope(await transport.recv())
    assert pair_reply.result == {}, f"pairing failed in test setup: {pair_reply}"
    return transport, client_ep


class TestIrohHelloHandshake:
    """The iroh transport must complete the same hello exchange as WS."""

    async def test_completes_with_matching_protocol_versions(self, running_iroh_server):
        _server_obj, endpoint, _pending_pairings, _identity = running_iroh_server
        transport, client_ep, _client_node_id = await _dial_and_hello(endpoint.addr())
        try:
            pass  # _dial_and_hello already asserts a valid Hello came back
        finally:
            await client_ep.close()

    async def test_announces_the_workers_node_id(self, running_iroh_server):
        _server_obj, endpoint, _pending_pairings, identity = running_iroh_server
        client_ep = await iroh.Endpoint.bind(iroh.EndpointOptions(preset=_MINIMAL()))
        try:
            conn = await client_ep.connect(endpoint.addr(), ALPN)
            bi = await conn.open_bi()
            transport = IrohStreamTransport(conn, bi)
            await transport.send(
                Hello(
                    proto={"min": 1, "max": 1},
                    agent={},
                    node_id="test-client-node",
                    nonce="n",
                ).to_json()
            )
            reply = parse_envelope(await transport.recv())
            assert reply.node_id == identity.node_id
        finally:
            await client_ep.close()


class TestIrohMethodDispatch:
    """Registered methods must be reachable the same way as over WS."""

    async def test_calls_a_registered_method_and_returns_its_result(
        self, running_iroh_server
    ):
        server_obj, endpoint, pending_pairings, identity = running_iroh_server

        async def echo(params, conn):
            assert isinstance(conn, Connection)
            return {"echo": params["value"]}

        server_obj.register("test.echo", echo)
        transport, client_ep = await _connect_and_pair(
            endpoint.addr(), pending_pairings, identity
        )
        try:
            await transport.send(
                Req(id=1, method="test.echo", params={"value": 42}).to_json()
            )
            res = parse_envelope(await transport.recv())
            assert res.result == {"echo": 42}
        finally:
            await client_ep.close()

    async def test_unauthenticated_connection_is_rejected(self, running_iroh_server):
        # No pair.claim/auth.prove was done -- everything except those two
        # methods must be refused, exactly as the WS transport enforces.
        server_obj, endpoint, _pending_pairings, _identity = running_iroh_server
        server_obj.register("test.echo", lambda params, conn: {"echo": True})
        transport, client_ep, _client_node_id = await _dial_and_hello(endpoint.addr())
        try:
            await transport.send(Req(id=0, method="test.echo", params={}).to_json())
            res = parse_envelope(await transport.recv())
            assert res.error is not None
            assert res.error["code"] == "auth.required"
        finally:
            await client_ep.close()

    async def test_protocol_error_is_reported_with_its_code(self, running_iroh_server):
        server_obj, endpoint, pending_pairings, identity = running_iroh_server

        async def echo_then_boom(params, conn):
            raise ProtocolError("job.not_found", "no such job")

        server_obj.register("test.boom", echo_then_boom)
        transport, client_ep = await _connect_and_pair(
            endpoint.addr(), pending_pairings, identity
        )
        try:
            await transport.send(Req(id=1, method="test.boom", params={}).to_json())
            res = parse_envelope(await transport.recv())
            assert res.error["code"] == "job.not_found"
            assert res.error["msg"] == "no such job"
        finally:
            await client_ep.close()


class TestIrohDisconnect:
    """Closing the client's stream must surface as a clean end, not a crash."""

    async def test_worker_handles_a_client_disconnecting_mid_session(
        self, running_iroh_server, caplog
    ):
        _server_obj, endpoint, _pending_pairings, _identity = running_iroh_server
        transport, client_ep, _client_node_id = await _dial_and_hello(endpoint.addr())
        await client_ep.close()
        # The accept-loop task handling this connection must exit cleanly
        # (via TransportClosed) rather than propagate an unhandled error --
        # an unhandled exception in that background task would surface as
        # an ERROR-level log from asyncio's default exception handler, not
        # as a raised exception here, so that's what's checked for.
        await asyncio.sleep(0.2)
        errors = [r for r in caplog.records if r.levelname == "ERROR"]
        assert errors == [], f"unhandled error in accept-loop task: {errors}"


class _FakeRecvStream:
    """Just enough of `iroh.RecvStream` for `read_frame`'s size check: a
    length-prefix header, then (if the check passes and a second
    `read_exact` is actually attempted) a fixed payload.
    """

    def __init__(self, header: bytes, payload: bytes = b""):
        self._reads = [header, payload]

    async def read_exact(self, n: int) -> bytes:
        return self._reads.pop(0)


class _FakeBiStream:
    """Just enough of an iroh `BiStream` for `IrohStreamTransport.__init__`."""

    def __init__(self, recv: _FakeRecvStream):
        self._recv = recv

    def send(self):
        return None

    def recv(self):
        return self._recv


class TestReadFrameSizeLimit:
    """A raw QUIC stream has no built-in size limit the way a WS connection
    does — these are the iroh-side counterpart of the WS `max_size` tests
    above, proving the new defensive check in `read_frame` actually works.
    Uses a minimal fake stream rather than a real iroh connection: the
    check fires off the 4-byte length prefix alone, before any real bytes
    would need to flow, so there's nothing a real connection would add.
    """

    async def test_read_frame_rejects_a_length_prefix_over_max_size(self):
        header = struct.pack(">I", 2000)
        with pytest.raises(FrameTooLarge):
            await read_frame(_FakeRecvStream(header), max_size=1000)

    async def test_read_frame_accepts_a_length_prefix_at_or_under_max_size(self):
        # Sanity check the boundary isn't off-by-one in the wrong direction.
        header = struct.pack(">I", 4)
        fake = _FakeRecvStream(header, payload=b"test")
        assert await read_frame(fake, max_size=4) == "test"

    async def test_transport_recv_converts_an_oversized_frame_to_transport_closed(self):
        # IrohStreamTransport.recv() doesn't expose a max_size override —
        # it always uses read_frame's default (DEFAULT_WS_MAX_SIZE) — so
        # the claimed length here has to exceed THAT, not an arbitrary
        # small number.
        header = struct.pack(">I", DEFAULT_WS_MAX_SIZE + 1)
        transport = IrohStreamTransport(
            connection=None, bi=_FakeBiStream(_FakeRecvStream(header))
        )
        with pytest.raises(TransportClosed):
            await transport.recv()
