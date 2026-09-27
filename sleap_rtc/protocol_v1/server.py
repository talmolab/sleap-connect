"""WebSocket transport binding for the sleap-connect protocol v1.

This is the Stage-1 "worker IS the server" transport: the worker runs this
directly (no signaling server, no rooms) and a client dials it over
localhost/LAN/Tailscale, per the architecture conclusion in the sleap-app
planning docs.

**Auth is intentionally NOT implemented here.** Every `hello` currently
succeeds as long as protocol versions overlap, and every connection is
implicitly trusted — there is no `auth.prove` challenge and no trusted-
client allowlist yet. That's item 1.5 (pairing/trust) per the Stage 1 plan;
this PR wires the envelope and method-dispatch mechanics only, so 1.5 can
bolt the security layer onto the same connection lifecycle without
reworking it. Do not point this at an untrusted network yet.
"""

import logging
import secrets
from typing import Any, Awaitable, Callable, Dict, Optional, Set

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
from sleap_rtc.protocol_v1.errors import INTERNAL, PROTO_UNKNOWN_METHOD, ProtocolError

AGENT_NAME = "sleap-connect-worker"


class Connection:
    """Per-connection state: the websocket, and its live event subscriptions."""

    def __init__(self, ws: ServerConnection):
        """Wrap a websocket connection.

        Args:
            ws: The underlying `websockets` server connection.
        """
        self.ws = ws
        self.subscribed_job_ids: Set[str] = set()

    async def send_event(self, event: Event) -> None:
        """Push an `event` frame to this connection."""
        await self.ws.send(event.to_json())


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
        """
        self.node_id = node_id
        self.proto_min = proto_min
        self.proto_max = proto_max
        self.agent_version = agent_version
        self.agent_platform = agent_platform
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
        """Start listening for connections.

        Args:
            host: Bind address.
            port: Bind port.

        Returns:
            The running `websockets` `Server` (use as an async context
            manager, or call `.close()` / `.wait_closed()` on it).
        """
        return await serve(self._handle_connection, host, port)

    async def _handle_connection(self, ws: ServerConnection) -> None:
        conn = Connection(ws)
        try:
            if not await self._do_hello(ws):
                return
            async for raw in ws:
                await self._handle_frame(conn, raw)
        finally:
            self.events.unsubscribe_all(conn)

    async def _do_hello(self, ws: ServerConnection) -> bool:
        """Exchange `hello` frames and check protocol-version compatibility.

        Returns:
            True if the handshake succeeded and the connection should
            proceed to normal message handling; False if it was closed.
        """
        raw = await ws.recv()
        try:
            frame = parse_envelope(raw)
        except EnvelopeError as e:
            await ws.close(code=1002, reason=str(e))
            return False

        if not isinstance(frame, Hello):
            await ws.close(code=1002, reason="expected hello as the first frame")
            return False

        client_min = frame.proto.get("min")
        client_max = frame.proto.get("max")
        if (
            client_min is None
            or client_max is None
            or client_max < self.proto_min
            or client_min > self.proto_max
        ):
            await ws.close(code=1002, reason="proto.mismatch")
            return False

        our_hello = Hello(
            proto={"min": self.proto_min, "max": self.proto_max},
            agent={
                "name": AGENT_NAME,
                "version": self.agent_version,
                "platform": self.agent_platform,
            },
            node_id=self.node_id,
            nonce=secrets.token_urlsafe(16),
        )
        await ws.send(our_hello.to_json())
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
        await conn.ws.send(res.to_json())

    async def _dispatch(self, req: Req, conn: Connection) -> Res:
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
