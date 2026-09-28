"""WebSocket transport binding for the sleap-connect protocol v1.

This is the Stage-1 "worker IS the server" transport: the worker runs this
directly (no signaling server, no rooms) and a client dials it over
localhost/LAN/Tailscale, per the architecture conclusion in the sleap-app
planning docs.

**Auth gating lives here; the actual pair.claim/auth.prove logic lives in
`sleap_rtc.protocol_v1.auth`** (item 1.5). This module only enforces that no
method other than `pair.claim`/`auth.prove` runs on an unauthenticated
connection, and records what a `hello` handshake needs for `auth.prove` to
later verify a signature (the nonce *we* sent, and the node_id *the client*
claimed). Only the client-proves-to-worker direction is implemented — see
`auth.py`'s module docstring for why the reverse direction is deferred. Do
not point this at an untrusted network without pairing configured.
"""

import asyncio
import logging
import secrets
from typing import Any, Awaitable, Callable, Dict, Optional, Protocol, Set

import websockets.exceptions
from websockets.asyncio.server import Server, ServerConnection, serve

from sleap_rtc.protocol_v1.envelope import (
    Event,
    EnvelopeError,
    Hello,
    PROTOCOL_VERSION,
    Req,
    Res,
    parse_envelope,
)
from sleap_rtc.protocol_v1.errors import (
    AUTH_REQUIRED,
    INTERNAL,
    PROTO_UNKNOWN_METHOD,
    ProtocolError,
)

AGENT_NAME = "sleap-connect-worker"

# The only methods an unauthenticated connection may call — everything else
# gets AUTH_REQUIRED. See protocol spec §3.3.
_UNAUTHENTICATED_METHODS = frozenset({"pair.claim", "auth.prove"})


class TransportClosed(Exception):
    """Raised by a `Transport.recv()` once the peer is gone.

    The one signal every transport normalizes to, whether the underlying
    close was clean or abrupt — the dispatch logic in this module (used by
    every transport binding) only ever needs to know "this connection is
    over now", never the transport-specific reason why.
    """


class Transport(Protocol):
    """What `Connection`/`ProtocolV1Server` need from any transport binding.

    `WsStreamTransport` (below) and `iroh_transport.IrohStreamTransport`
    (item 2.2) are the two concrete implementations — a WebSocket already
    frames each `.send()`/`.recv()` call as one discrete message, while
    iroh's raw QUIC streams don't, so that framing is each adapter's own
    problem to solve; this module's handshake/dispatch logic is written
    once against this shape and never touches either transport's native
    API directly.
    """

    async def send(self, data: str) -> None: ...

    async def recv(self) -> str:
        """Return the next frame, or raise `TransportClosed` once there
        won't be one.
        """
        ...

    async def close(self, code: int = 1000, reason: str = "") -> None: ...


class WsStreamTransport:
    """Adapts a `websockets` connection to the `Transport` shape."""

    def __init__(self, ws: ServerConnection):
        self._ws = ws

    async def send(self, data: str) -> None:
        await self._ws.send(data)

    async def recv(self) -> str:
        try:
            return await self._ws.recv()
        except websockets.exceptions.ConnectionClosed as e:
            raise TransportClosed() from e

    async def close(self, code: int = 1000, reason: str = "") -> None:
        await self._ws.close(code=code, reason=reason)


class Connection:
    """Per-connection state: the transport, auth status, and event subscriptions."""

    def __init__(self, transport: Transport):
        """Wrap a transport (a WebSocket or an iroh stream).

        Args:
            transport: Anything satisfying the `Transport` shape.
        """
        self.transport = transport
        self.subscribed_job_ids: Set[str] = set()
        # Populated by _do_hello; used by auth.py's auth_prove to verify a
        # signature against the nonce *we* sent and the node_id *the peer*
        # claimed in *its* hello.
        self.own_nonce: Optional[str] = None
        self.peer_node_id: Optional[str] = None
        self.peer_agent: Optional[dict] = None
        # Set True by pair.claim or auth.prove (sleap_rtc.protocol_v1.auth).
        # Until then, only those two methods may be dispatched — see
        # ProtocolV1Server._dispatch.
        self.authenticated: bool = False

    async def send_event(self, event: Event) -> None:
        """Push an `event` frame to this connection."""
        await self.transport.send(event.to_json())


class EventBus:
    """Fan-out of per-job `event` frames to whichever connections subscribed."""

    def __init__(self):
        """Initialize with no subscribers."""
        self._subscribers: Dict[str, Set[Connection]] = {}

    def subscribe(self, job_id: str, conn: Connection) -> None:
        """Subscribe a connection to a job's events."""
        self._subscribers.setdefault(job_id, set()).add(conn)
        conn.subscribed_job_ids.add(job_id)

    def unsubscribe_all(self, conn: Connection) -> None:
        """Remove a connection from every job it was subscribed to (on disconnect)."""
        for job_id in conn.subscribed_job_ids:
            self._subscribers.get(job_id, set()).discard(conn)
        conn.subscribed_job_ids.clear()

    async def publish(self, job_id: str, event: Event) -> None:
        """Push an event to every connection currently subscribed to `job_id`."""
        for conn in list(self._subscribers.get(job_id, ())):
            try:
                await conn.send_event(event)
            except Exception:
                logging.exception(
                    f"[protocol_v1] Failed to push event to a subscriber of {job_id}"
                )


MethodHandler = Callable[[Dict[str, Any], Connection], Awaitable[dict]]


class ProtocolV1Server:
    """A WebSocket server implementing the sleap-connect protocol v1 envelope.

    Method handlers are registered by dotted name (e.g. ``"jobs.submit"``)
    and dispatched from incoming `req` frames; replies go back as `res`.
    `events` is the `EventBus` job-execution code should push through.
    """

    def __init__(
        self,
        node_id: str,
        proto_min: int = PROTOCOL_VERSION,
        proto_max: int = PROTOCOL_VERSION,
        agent_version: str = "0.0.0",
        agent_platform: str = "unknown",
        blob_port: Optional[int] = None,
    ):
        """Initialize the server (does not start listening — see `serve`).

        Args:
            node_id: This worker's persistent identity (base64 public key —
                or, until pairing/keys land in 1.5, any stable placeholder
                string uniquely identifying this worker).
            proto_min: Lowest protocol version this server accepts.
            proto_max: Highest protocol version this server accepts.
            agent_version: Reported in `hello.agent.version`.
            agent_platform: Reported in `hello.agent.platform`.
            blob_port: Reported in `hello.blob_port` (spec §6.3) if this
                worker is also running the blob HTTP server. `None` if not
                (`job.result` blobs won't be fetchable either way).
        """
        self.node_id = node_id
        self.proto_min = proto_min
        self.proto_max = proto_max
        self.agent_version = agent_version
        self.agent_platform = agent_platform
        self.blob_port = blob_port
        self.events = EventBus()
        self._methods: Dict[str, MethodHandler] = {}

    def register(self, name: str, handler: MethodHandler) -> None:
        """Register a method handler.

        Args:
            name: Dotted method name, e.g. ``"jobs.submit"``.
            handler: ``async def handler(params: dict, conn: Connection) -> dict``.
        """
        self._methods[name] = handler

    async def serve(self, host: str, port: int) -> Server:
        """Start listening for WebSocket connections.

        Args:
            host: Bind address.
            port: Bind port.

        Returns:
            The running `websockets` `Server` (use as an async context
            manager, or call `.close()` / `.wait_closed()` on it).
        """
        return await serve(self._handle_ws_connection, host, port)

    async def serve_iroh(self, endpoint) -> None:
        """Accept iroh connections and run each through the same handshake
        and method dispatch as the WebSocket transport (item 2.2).

        Runs until `endpoint.accept_next()` reports the endpoint has been
        closed (returns `None`) — call this from a background task, the
        same way `serve`'s returned WS `Server` runs in the background.
        Each connection is handled in its own task so one slow or
        misbehaving client can't stop new connections from being accepted.

        Args:
            endpoint: A bound, online `iroh.Endpoint`.
        """
        while True:
            incoming = await endpoint.accept_next()
            if incoming is None:
                return
            asyncio.create_task(self._handle_iroh_incoming(incoming))

    async def _handle_iroh_incoming(self, incoming) -> None:
        # Deferred import: only sleap-rtc installs that actually use iroh
        # need the dependency at all (see iroh_transport.py's own docstring).
        from sleap_rtc.protocol_v1.iroh_transport import IrohStreamTransport

        try:
            accepting = await incoming.accept()
            iroh_conn = await accepting.connect()
        except Exception:
            logging.exception("[protocol_v1] iroh handshake failed")
            return

        try:
            # One control stream per connection — protocol v1's envelope
            # frames all multiplex over this single stream, the same as one
            # WebSocket connection carries every frame for that connection.
            bi = await iroh_conn.accept_bi()
        except Exception:
            logging.exception("[protocol_v1] iroh failed to accept its control stream")
            return

        await self._run_connection(IrohStreamTransport(iroh_conn, bi))

    async def _handle_ws_connection(self, ws: ServerConnection) -> None:
        await self._run_connection(WsStreamTransport(ws))

    async def _run_connection(self, transport: Transport) -> None:
        """Drive one connection through hello + frame dispatch until it closes.

        Transport-agnostic: works identically whether `transport` wraps a
        WebSocket or an iroh stream — every transport-specific detail (wire
        framing, how a close is signaled) is that transport's own adapter's
        job, not this method's.
        """
        conn = Connection(transport)
        try:
            if not await self._do_hello(conn):
                return
            while True:
                try:
                    raw = await transport.recv()
                except TransportClosed:
                    return
                await self._handle_frame(conn, raw)
        finally:
            self.events.unsubscribe_all(conn)

    async def _do_hello(self, conn: Connection) -> bool:
        """Exchange `hello` frames and check protocol-version compatibility.

        Also records, on `conn`, what `auth.prove` will later need: the
        node_id the peer claimed (`conn.peer_node_id`) and the nonce *we*
        sent (`conn.own_nonce`) — `auth.prove` verifies the peer's signature
        of that nonce against that node_id.

        Args:
            conn: The connection being established.

        Returns:
            True if the handshake succeeded and the connection should
            proceed to normal message handling; False if it was closed.
        """
        transport = conn.transport
        try:
            raw = await transport.recv()
        except TransportClosed:
            return False
        try:
            frame = parse_envelope(raw)
        except EnvelopeError as e:
            await transport.close(code=1002, reason=str(e))
            return False

        if not isinstance(frame, Hello):
            await transport.close(code=1002, reason="expected hello as the first frame")
            return False

        client_min = frame.proto.get("min")
        client_max = frame.proto.get("max")
        if (
            client_min is None
            or client_max is None
            or client_max < self.proto_min
            or client_min > self.proto_max
        ):
            await transport.close(code=1002, reason="proto.mismatch")
            return False

        conn.peer_node_id = frame.node_id
        conn.peer_agent = frame.agent

        conn.own_nonce = secrets.token_urlsafe(16)
        our_hello = Hello(
            proto={"min": self.proto_min, "max": self.proto_max},
            agent={
                "name": AGENT_NAME,
                "version": self.agent_version,
                "platform": self.agent_platform,
            },
            node_id=self.node_id,
            nonce=conn.own_nonce,
            blob_port=self.blob_port,
        )
        await transport.send(our_hello.to_json())
        return True

    async def _handle_frame(self, conn: Connection, raw: str) -> None:
        try:
            frame = parse_envelope(raw)
        except EnvelopeError as e:
            logging.warning(f"[protocol_v1] Dropping unparseable frame: {e}")
            return

        if not isinstance(frame, Req):
            # `hello` is only valid as the very first frame (handled in
            # _do_hello); this server doesn't issue its own outgoing `req`s
            # to the client yet, so an incoming `res`/`event` is unexpected.
            logging.warning(
                f"[protocol_v1] Unexpected frame type from client: "
                f"{type(frame).__name__}"
            )
            return

        res = await self._dispatch(frame, conn)
        await conn.transport.send(res.to_json())

    async def _dispatch(self, req: Req, conn: Connection) -> Res:
        if req.method not in _UNAUTHENTICATED_METHODS and not conn.authenticated:
            return Res.err(req.id, AUTH_REQUIRED, "Connection is not authenticated")

        handler = self._methods.get(req.method)
        if handler is None:
            return Res.err(
                req.id, PROTO_UNKNOWN_METHOD, f"Unknown method: {req.method}"
            )

        try:
            result = await handler(req.params, conn)
            return Res.ok(req.id, result)
        except ProtocolError as e:
            return Res.err(req.id, e.code, str(e), e.data)
        except Exception as e:
            logging.exception(f"[protocol_v1] Unhandled error in method {req.method}")
            return Res.err(req.id, INTERNAL, str(e))
