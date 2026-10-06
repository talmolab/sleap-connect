"""iroh QUIC transport binding for the sleap-connect protocol v1 (item 2.2).

Runs alongside the existing WebSocket transport (`server.py`), not instead
of it: a worker accepts both, so a plain WS-only client keeps working
unchanged, and a client that can't reach the worker directly (no shared
network/VPN/Tailscale) falls back to iroh's own direct-then-relay dialing
instead. See `ProtocolV1Server.serve_iroh`, which drives connections
accepted here through the exact same handshake/dispatch logic as the WS
path — nothing above the transport layer needed to change to support this.

Deferred import throughout sleap_rtc (see `server.py`'s `_handle_iroh_incoming`):
`iroh` is only imported by code paths that actually use it, so a `sleap-rtc`
install that never touches iroh doesn't need the dependency installed.

Wire framing: an iroh `BiStream` is just two raw byte pipes with no
built-in message boundaries — unlike a WebSocket connection, where each
`.send()`/`.recv()` call already carries one discrete message. Each
envelope JSON frame here is therefore length-prefixed: a 4-byte big-endian
length, then that many bytes of UTF-8-encoded JSON.

End-of-stream detection (verified by hand, not just from the docs): a
clean `SendStream.finish()` with nothing left to read shows up on the
receiving side as an `iroh.IrohError` from `RecvStream.read_exact` —
`kind() == IrohErrorKind.STREAM`, message `"FinishedEarly(0)"` — the same
generic error type and kind an abrupt disconnect mid-read also raises.
There's no reliable way (and, for this protocol, no need) to tell those
apart: either way the connection is over, exactly like a WebSocket's
`ConnectionClosed` already covers both a clean and an abrupt close with
one exception type.
"""

import struct
from typing import TYPE_CHECKING, Optional

from sleap_rtc.protocol_v1.server import DEFAULT_WS_MAX_SIZE, TransportClosed

if TYPE_CHECKING:
    import iroh

_LENGTH_PREFIX = struct.Struct(">I")  # 4-byte big-endian length prefix


class FrameTooLarge(Exception):
    """A peer's length prefix claimed more than `max_size` bytes.

    Raised before the (potentially huge) second `read_exact` is ever
    attempted — a raw QUIC stream has no built-in size limit the way
    `websockets` enforces one on a WS connection (`server.py`'s
    `DEFAULT_WS_MAX_SIZE`), so this is iroh's counterpart: without it, a
    malformed or hostile 4-byte length prefix (up to ~4 GiB) would make
    `read_exact` try to buffer that much.
    """


# The ALPN identifying this protocol on the wire — an iroh connection with
# any other ALPN is a different application entirely, not a malformed
# request, so the worker's Endpoint is configured to only ever accept this
# one.
ALPN = b"sleap-connect/protocol-v1"


async def write_frame(send: "iroh.SendStream", data: str) -> None:
    """Write one length-prefixed frame: a 4-byte big-endian length, then
    that many UTF-8 bytes. Shared by the control-stream transport below and
    item 2.4's blob-range-read stream (`server.py`'s `_serve_iroh_blob_stream`)
    — the same framing convention, reused rather than duplicated.
    """
    payload = data.encode("utf-8")
    await send.write_all(_LENGTH_PREFIX.pack(len(payload)) + payload)


async def read_frame(
    recv: "iroh.RecvStream", max_size: int = DEFAULT_WS_MAX_SIZE
) -> Optional[str]:
    """Read one length-prefixed frame (the read-side counterpart of
    `write_frame`). Returns `None` once the stream ends, clean or abrupt —
    per the module-level docstring above, iroh doesn't reliably distinguish
    those and this protocol doesn't need to.

    Args:
        recv: The stream to read from.
        max_size: Largest frame accepted, in bytes — see `FrameTooLarge`.
            Defaults to the same ceiling the WS transport enforces
            (`DEFAULT_WS_MAX_SIZE`), so a `labels_content` embed that fits
            over one transport fits over the other too.

    Raises:
        FrameTooLarge: the claimed length exceeds `max_size`.
    """
    import iroh

    try:
        header = await recv.read_exact(_LENGTH_PREFIX.size)
        (length,) = _LENGTH_PREFIX.unpack(header)
        if length > max_size:
            raise FrameTooLarge(f"frame length {length} exceeds max_size {max_size}")
        payload = await recv.read_exact(length)
    except iroh.IrohError as e:
        if e.kind() in (iroh.IrohErrorKind.STREAM, iroh.IrohErrorKind.CONNECTION):
            return None
        raise
    return payload.decode("utf-8")


class IrohStreamTransport:
    """Adapts one iroh `BiStream` to the `server.Transport` shape.

    One `BiStream` carries every envelope frame for the connection's
    lifetime — protocol v1's hello/req/res/event frames all multiplex over
    this single stream, the same as one WebSocket connection carries every
    frame for that connection.
    """

    def __init__(self, connection: "iroh.Connection", bi: "iroh.BiStream"):
        """Wrap an already-open bidirectional stream.

        Args:
            connection: The iroh `Connection` this stream belongs to —
                kept only so `close()` can end the whole connection, not
                just this one stream.
            bi: The `BiStream` to frame envelope traffic over.
        """
        self._connection = connection
        self._send = bi.send()
        self._recv = bi.recv()

    async def send(self, data: str) -> None:
        await write_frame(self._send, data)

    async def recv(self) -> str:
        # The `Transport` protocol requires raising TransportClosed once
        # there won't be another frame — read_frame's `None` sentinel (a
        # simpler primitive, reused as-is by the blob stream's read loop,
        # which prefers a plain `if x is None: return` over an exception)
        # gets translated to that exception right here, at the boundary.
        # An oversized frame is treated the same way, for the same reason
        # the WS transport's own "message too big" close already looks like
        # a plain disconnect to `_run_connection` (ConnectionClosed ->
        # TransportClosed) rather than an unhandled exception.
        try:
            text = await read_frame(self._recv)
        except FrameTooLarge:
            raise TransportClosed()
        if text is None:
            raise TransportClosed()
        return text

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self._connection.close(code, reason.encode("utf-8"))
