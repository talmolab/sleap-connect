"""Integration tests for ProtocolV1Server — a real WS server, a real client.

Uses the `websockets` client against a real, locally-bound server (port 0 —
let the OS assign a free port) rather than mocking the transport, since the
whole point of this module is the actual hello handshake / req-res /
event-push wire behavior.

Every connection must pair before calling any protected method (item 1.5's
auth gate applies unconditionally) — `_connect_and_hello` does a real
pair.claim as part of connecting, so these tests can focus on generic
dispatch mechanics rather than re-deriving the pairing dance each time.
Auth-specific behavior (pairing edge cases, auth.prove, rejection paths) is
covered in test_auth.py, not here.
"""

import pytest
import websockets

from sleap_rtc.auth.keypair import generate_keypair, public_key_to_b64
from sleap_rtc.protocol_v1.auth import AuthMethods
from sleap_rtc.protocol_v1.envelope import Event, Hello, Req, parse_envelope
from sleap_rtc.protocol_v1.errors import ProtocolError
from sleap_rtc.protocol_v1.identity import WorkerIdentity
from sleap_rtc.protocol_v1.pairing import PendingPairings
from sleap_rtc.protocol_v1.server import ProtocolV1Server
from sleap_rtc.protocol_v1.trust_store import TrustStore


@pytest.fixture
async def running_server(tmp_path):
    """A ProtocolV1Server (with pairing/auth wired in) bound to a free port."""
    identity = WorkerIdentity(tmp_path / "identity.json")
    trust_store = TrustStore(tmp_path / "trusted.json")
    pending_pairings = PendingPairings()
    server_obj = ProtocolV1Server(node_id=identity.node_id)
    AuthMethods(server_obj, identity, trust_store, pending_pairings)

    ws_server = await server_obj.serve("127.0.0.1", 0)
    port = ws_server.sockets[0].getsockname()[1]
    try:
        yield server_obj, port, pending_pairings, identity
    finally:
        ws_server.close()
        await ws_server.wait_closed()


async def _connect_and_hello(port: int, pending_pairings: PendingPairings, identity):
    """Connect, complete hello, and pair; return the open, authenticated client ws."""
    ws = await websockets.connect(f"ws://127.0.0.1:{port}")
    _priv, public_key = generate_keypair()
    client_node_id = public_key_to_b64(public_key)

    await ws.send(
        Hello(
            proto={"min": 1, "max": 1},
            agent={"name": "test-client", "version": "0.0.0", "platform": "test"},
            node_id=client_node_id,
            nonce="client-nonce",
        ).to_json()
    )
    reply = parse_envelope(await ws.recv())
    assert isinstance(reply, Hello)

    ticket = pending_pairings.create(identity.node_id, [])
    await ws.send(
        Req(
            id=0,
            method="pair.claim",
            params={"secret": ticket.secret, "node_id": client_node_id},
        ).to_json()
    )
    pair_reply = parse_envelope(await ws.recv())
    assert pair_reply.result == {}, f"pairing failed in test setup: {pair_reply}"

    return ws


class TestHelloHandshake:
    """Tests for the hello handshake."""

    async def test_completes_with_matching_protocol_versions(self, running_server):
        _server, port, pending_pairings, identity = running_server

        ws = await _connect_and_hello(port, pending_pairings, identity)
        await ws.close()

    async def test_announces_blob_port_in_its_own_hello_when_configured(self, tmp_path):
        identity = WorkerIdentity(tmp_path / "identity.json")
        server_obj = ProtocolV1Server(node_id=identity.node_id, blob_port=9632)
        ws_server = await server_obj.serve("127.0.0.1", 0)
        port = ws_server.sockets[0].getsockname()[1]
        try:
            ws = await websockets.connect(f"ws://127.0.0.1:{port}")
            await ws.send(
                Hello(
                    proto={"min": 1, "max": 1},
                    agent={},
                    node_id="client-node",
                    nonce="client-nonce",
                ).to_json()
            )
            reply = parse_envelope(await ws.recv())
            assert reply.blob_port == 9632
            await ws.close()
        finally:
            ws_server.close()
            await ws_server.wait_closed()

    async def test_omits_blob_port_when_not_configured(self, running_server):
        _server, port, _pending_pairings, _identity = running_server

        ws = await websockets.connect(f"ws://127.0.0.1:{port}")
        await ws.send(
            Hello(
                proto={"min": 1, "max": 1},
                agent={},
                node_id="client-node",
                nonce="client-nonce",
            ).to_json()
        )
        reply = parse_envelope(await ws.recv())
        assert reply.blob_port is None
        await ws.close()

    async def test_closes_connection_on_protocol_mismatch(self, running_server):
        _server, port, _pending_pairings, _identity = running_server

        ws = await websockets.connect(f"ws://127.0.0.1:{port}")
        await ws.send(
            Hello(
                proto={"min": 99, "max": 100},  # doesn't overlap with server's [1, 1]
                agent={},
                node_id="client-node",
                nonce="x",
            ).to_json()
        )
        with pytest.raises(websockets.exceptions.ConnectionClosed):
            await ws.recv()

    async def test_closes_connection_if_first_frame_is_not_hello(self, running_server):
        _server, port, _pending_pairings, _identity = running_server

        ws = await websockets.connect(f"ws://127.0.0.1:{port}")
        await ws.send(Req(id=1, method="jobs.list").to_json())
        with pytest.raises(websockets.exceptions.ConnectionClosed):
            await ws.recv()


class TestMethodDispatch:
    """Tests for req/res method dispatch."""

    async def test_calls_a_registered_method_and_returns_its_result(
        self, running_server
    ):
        server_obj, port, pending_pairings, identity = running_server

        async def echo(params, conn):
            return {"echoed": params}

        server_obj.register("test.echo", echo)
        ws = await _connect_and_hello(port, pending_pairings, identity)

        await ws.send(Req(id=1, method="test.echo", params={"x": 1}).to_json())
        reply = parse_envelope(await ws.recv())

        assert reply.id == 1
        assert reply.result == {"echoed": {"x": 1}}
        await ws.close()

    async def test_accepts_a_request_well_over_the_old_1_mib_default(
        self, running_server
    ):
        """Regression test for item 3.3: `websockets`' own default `max_size`
        (1 MiB) would silently reject a real `labels_content` embed before
        `ProtocolV1Server.serve` started passing `DEFAULT_WS_MAX_SIZE` (256
        MiB) — confirm a multi-MB request actually round-trips rather than
        closing the connection.
        """
        server_obj, port, pending_pairings, identity = running_server

        received_size = {}

        async def echo_size(params, conn):
            received_size["n"] = len(params["payload"])
            return {"ok": True}

        server_obj.register("test.echo", echo_size)
        ws = await _connect_and_hello(port, pending_pairings, identity)

        big_payload = "x" * (2 * 1024 * 1024)  # 2 MiB — over the old 1 MiB default
        await ws.send(
            Req(id=1, method="test.echo", params={"payload": big_payload}).to_json()
        )
        reply = parse_envelope(await ws.recv())

        assert reply.result == {"ok": True}
        assert received_size["n"] == 2 * 1024 * 1024
        await ws.close()

    async def test_max_size_still_rejects_a_request_over_the_configured_cap(
        self, tmp_path
    ):
        """The previous test only proves a payload BELOW the new 256 MiB
        ceiling now round-trips — it doesn't prove the ceiling still exists
        at all (e.g. a future refactor that drops `max_size` entirely, or
        passes `None`, would leave this file fully green). Uses its own
        tiny `max_size` rather than the real 256 MiB one so this stays a
        fast unit test, not a 256 MiB transfer.
        """
        identity = WorkerIdentity(tmp_path / "identity.json")
        server_obj = ProtocolV1Server(node_id=identity.node_id)
        ws_server = await server_obj.serve("127.0.0.1", 0, max_size=1024)
        port = ws_server.sockets[0].getsockname()[1]
        try:
            ws = await websockets.connect(f"ws://127.0.0.1:{port}")
            with pytest.raises(websockets.exceptions.ConnectionClosedError):
                await ws.send("x" * 2048)  # over the 1024-byte cap just configured
                await ws.recv()
        finally:
            ws_server.close()
            await ws_server.wait_closed()

    async def test_unknown_method_returns_proto_unknown_method_error(
        self, running_server
    ):
        _server, port, pending_pairings, identity = running_server
        ws = await _connect_and_hello(port, pending_pairings, identity)

        await ws.send(Req(id=1, method="does.not.exist").to_json())
        reply = parse_envelope(await ws.recv())

        assert reply.error["code"] == "proto.unknown_method"
        await ws.close()

    async def test_protocol_error_is_reported_with_its_code(self, running_server):
        server_obj, port, pending_pairings, identity = running_server

        async def raises_not_found(params, conn):
            raise ProtocolError("job.not_found", "No such job", data={"job_id": "x"})

        server_obj.register("test.fail", raises_not_found)
        ws = await _connect_and_hello(port, pending_pairings, identity)

        await ws.send(Req(id=1, method="test.fail").to_json())
        reply = parse_envelope(await ws.recv())

        assert reply.error == {
            "code": "job.not_found",
            "msg": "No such job",
            "data": {"job_id": "x"},
        }
        await ws.close()

    async def test_unexpected_exception_is_reported_as_internal(self, running_server):
        server_obj, port, pending_pairings, identity = running_server

        async def raises_boom(params, conn):
            raise RuntimeError("boom")

        server_obj.register("test.boom", raises_boom)
        ws = await _connect_and_hello(port, pending_pairings, identity)

        await ws.send(Req(id=1, method="test.boom").to_json())
        reply = parse_envelope(await ws.recv())

        assert reply.error["code"] == "internal"
        assert "boom" in reply.error["msg"]
        await ws.close()

    async def test_multiple_requests_are_correlated_by_id(self, running_server):
        server_obj, port, pending_pairings, identity = running_server

        async def echo(params, conn):
            return params

        server_obj.register("test.echo", echo)
        ws = await _connect_and_hello(port, pending_pairings, identity)

        await ws.send(Req(id=1, method="test.echo", params={"n": 1}).to_json())
        await ws.send(Req(id=2, method="test.echo", params={"n": 2}).to_json())

        reply1 = parse_envelope(await ws.recv())
        reply2 = parse_envelope(await ws.recv())

        assert (reply1.id, reply1.result) == (1, {"n": 1})
        assert (reply2.id, reply2.result) == (2, {"n": 2})
        await ws.close()


class TestEventBus:
    """Tests for event push via a registered subscription."""

    async def test_published_events_reach_a_subscribed_connection(self, running_server):
        server_obj, port, pending_pairings, identity = running_server

        async def subscribe(params, conn):
            server_obj.events.subscribe("job-1", conn)
            return {}

        server_obj.register("test.subscribe", subscribe)
        ws = await _connect_and_hello(port, pending_pairings, identity)
        await ws.send(Req(id=1, method="test.subscribe").to_json())
        await ws.recv()  # the res for test.subscribe

        await server_obj.events.publish(
            "job-1", Event(topic="job.log", seq=1, data={"line": "hi"}, job_id="job-1")
        )
        frame = parse_envelope(await ws.recv())

        assert isinstance(frame, Event)
        assert frame.data == {"line": "hi"}
        await ws.close()

    async def test_events_for_a_different_job_are_not_delivered(self, running_server):
        server_obj, port, pending_pairings, identity = running_server

        async def subscribe(params, conn):
            server_obj.events.subscribe("job-1", conn)
            return {}

        server_obj.register("test.subscribe", subscribe)
        ws = await _connect_and_hello(port, pending_pairings, identity)
        await ws.send(Req(id=1, method="test.subscribe").to_json())
        await ws.recv()

        await server_obj.events.publish(
            "job-2",
            Event(topic="job.log", seq=1, data={"line": "other job"}, job_id="job-2"),
        )
        # Nothing to receive for job-1's subscriber; sending our own req and
        # getting its res back (rather than the job-2 event) proves it.
        await ws.send(Req(id=2, method="test.subscribe").to_json())
        reply = parse_envelope(await ws.recv())
        assert reply.id == 2
        await ws.close()
