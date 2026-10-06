"""One-line pairing codes (protocol spec §3.2, decided 2026-10-05).

`sleap-rtc pair` used to print a multi-line, indented JSON ticket that a
user had to carefully select (awkward across a wrapped terminal line) and
paste into the app. This module packs the same information into one short,
double-click-selectable line instead: ``sleap1`` + lowercase, unpadded
RFC 4648 base32 of a small binary layout —

| bytes | field |
|---|---|
| 1 | format version = 1 |
| 32 | worker node id (raw Ed25519 public key) |
| 16 | one-time secret (raw bytes) |
| 4 | expiry, uint32 big-endian Unix seconds |
| 1 + … | LAN addresses: count, then each: tag ``4`` + IPv4(4) + port(2) / tag ``6`` + IPv6(16) + port(2) / tag ``0`` + len(1) + UTF-8 URL |
| 1 + … | iroh relay: ``0`` none / ``1..n`` index into `RELAY_TABLE` / ``255`` + len(1) + UTF-8 URL |
| 2 | checksum: first 2 bytes of SHA-256 over everything before it |

iroh *direct* addresses are deliberately left out — a client dials the
worker's relay and iroh upgrades to a direct path itself once connected, so
carrying them in the code would only add bytes for no benefit. ``--json``
output (``sleap-rtc pair --json``) is unaffected and still includes them,
for scripts/back-compat and for a client that wants to skip the relay hop.

This module is deliberately standalone (no dependency on `cli.py` or
`server.py`), so it has a small, easy-to-test, pure encode/decode surface;
`cli.py` just calls `encode_pair_code`. The TypeScript app keeps an
identical mirror of this exact format (`src/lib/protocolV1/pairCode.ts`) —
the two must be kept in lock-step. `tests/protocol_v1/pair_code_vectors.json`
holds fixed, deterministic encode/decode vectors both test suites assert
against, so a format change in one language that isn't mirrored in the
other gets caught immediately.
"""

import base64
import binascii
import hashlib
import struct
from ipaddress import IPv4Address, IPv6Address, ip_address
from typing import List, Optional, Tuple
from urllib.parse import urlsplit

from sleap_rtc.protocol_v1.iroh_live import build_iroh_section
from sleap_rtc.protocol_v1.pairing import PairingTicket

CODE_PREFIX = "sleap1"
FORMAT_VERSION = 1

# iroh's current default (n0) relay URLs, verified against the installed
# `iroh` (1.1.0) Python package:
#
#   >>> import iroh
#   >>> iroh.RelayMode.default_mode().relay_map().urls()
#   ['https://aps1-1.relay.n0.iroh.link./', 'https://euc1-1.relay.n0.iroh.link./',
#    'https://use1-1.relay.n0.iroh.link./', 'https://usw1-1.relay.n0.iroh.link./']
#
# — n0's production relay fleet (one region each for Asia-Pacific,
# EU-Central, US-East, US-West; see https://iroh.computer and the
# `iroh-relay` crate's bundled default `RelayMap`). A worker's live
# `endpoint.addr().relay_url()` matches one of these strings exactly
# (confirmed by binding a real endpoint and watching it pick a home relay),
# so a pairing code's relay byte can be a bare index into this table
# instead of spelling the URL out.
#
# APPEND-ONLY. A pairing code's relay byte for a known region is this
# tuple's position + 1 (index 0 is reserved for "no relay" — see
# `_encode_relay`); reordering or removing an entry would silently
# reinterpret already-minted, still-unexpired codes as the wrong relay.
# If iroh ever adds a region, only append it — and update the identical
# copy in `src/lib/protocolV1/pairCode.ts` at the same time.
RELAY_TABLE: Tuple[str, ...] = (
    "https://aps1-1.relay.n0.iroh.link./",
    "https://euc1-1.relay.n0.iroh.link./",
    "https://use1-1.relay.n0.iroh.link./",
    "https://usw1-1.relay.n0.iroh.link./",
)

_ADDR_TAG_IPV4 = 4
_ADDR_TAG_IPV6 = 6
_ADDR_TAG_URL = 0

_RELAY_TAG_NONE = 0
_RELAY_TAG_URL = 255

_CHECKSUM_LEN = 2
# version(1) + node_id(32) + secret(16) + expires_at(4) + addr count(1) +
# relay tag(1) + checksum(2) — the smallest a well-formed code's decoded
# body can be (no addrs, no relay).
_MIN_BODY_LEN = 1 + 32 + 16 + 4 + 1 + 1 + _CHECKSUM_LEN


class PairCodeError(ValueError):
    """A pairing code is malformed: bad prefix/version, truncated, or a bad checksum."""


def encode_pair_code(ticket: PairingTicket) -> str:
    """Pack `ticket` into a one-line, double-click-selectable pairing code.

    See the module docstring for the exact byte layout. Only
    ``ticket.iroh["relay_url"]`` is encoded — direct addresses are left out
    (the `--json` ticket still has them).

    Args:
        ticket: The ticket to encode. `ticket.node_id` must decode (as
            URL-safe base64) to exactly 32 bytes and `ticket.secret` to
            exactly 16 bytes — true of every ticket `PendingPairings.create`
            mints.

    Returns:
        The pairing code, e.g. ``"sleap1abc...xyz"``.

    Raises:
        ValueError: If `ticket.node_id`/`ticket.secret` aren't 32/16 raw
            bytes, there are more than 255 addrs, or an addr or custom
            relay URL is too long to fit the format's 1-byte length prefix
            (255 UTF-8 bytes).
    """
    node_id = _b64url_decode(ticket.node_id)
    if len(node_id) != 32:
        raise ValueError(f"node_id must decode to 32 bytes, got {len(node_id)}")

    secret = _b64url_decode(ticket.secret)
    if len(secret) != 16:
        raise ValueError(f"secret must decode to 16 bytes, got {len(secret)}")

    if len(ticket.addrs) > 255:
        raise ValueError(f"too many addrs ({len(ticket.addrs)}); max 255")

    body = bytearray()
    body.append(FORMAT_VERSION)
    body += node_id
    body += secret
    body += struct.pack(">I", int(ticket.expires_at))
    body.append(len(ticket.addrs))
    for addr in ticket.addrs:
        body += _encode_addr(addr)
    body += _encode_relay(ticket.iroh)

    checksum = hashlib.sha256(bytes(body)).digest()[:_CHECKSUM_LEN]
    body += checksum

    return CODE_PREFIX + _b32_encode(bytes(body))


def decode_pair_code(code: str) -> PairingTicket:
    """Reverse `encode_pair_code`.

    Whitespace around `code` is stripped and the whole thing is matched
    case-insensitively (the base32 alphabet this format emits is lowercase
    a-z/2-7, but a terminal, editor, or autocapitalizing phone keyboard
    might upper-case a pasted code).

    Args:
        code: A pairing code, as printed by `sleap-rtc pair`.

    Returns:
        The decoded `PairingTicket`. Its `iroh` field is `None` if the code
        carries no relay, else a `build_iroh_section`-shaped dict with an
        empty `direct_addrs` (the code never carries any — see the module
        docstring) and `node_id` equal to the ticket's own `node_id` (true
        of every ticket this worker has ever minted).

    Raises:
        PairCodeError: `code` doesn't start with the expected prefix, has
            an unsupported format version, fails its checksum (a strong
            signal of a truncated or mistyped paste), or is otherwise too
            short/malformed to parse.
    """
    lowered = code.strip().lower()
    if not lowered.startswith(CODE_PREFIX):
        raise PairCodeError(
            f"Not a sleap-connect pairing code (expected it to start with "
            f"{CODE_PREFIX!r})"
        )

    payload = _b32_decode(lowered[len(CODE_PREFIX) :])
    if len(payload) < _MIN_BODY_LEN:
        raise PairCodeError(
            "This pairing code is too short — it's probably truncated or mistyped"
        )

    body, checksum = payload[:-_CHECKSUM_LEN], payload[-_CHECKSUM_LEN:]
    if hashlib.sha256(body).digest()[:_CHECKSUM_LEN] != checksum:
        raise PairCodeError(
            "This pairing code is incomplete or mistyped (checksum mismatch)"
        )

    pos = 0
    version = body[pos]
    pos += 1
    if version != FORMAT_VERSION:
        raise PairCodeError(f"Unsupported pairing code version: {version}")

    try:
        node_id = _b64url_encode(body[pos : pos + 32])
        pos += 32
        secret = _b64url_encode(body[pos : pos + 16])
        pos += 16
        (expires_at,) = struct.unpack(">I", body[pos : pos + 4])
        pos += 4

        addr_count = body[pos]
        pos += 1
        addrs: List[str] = []
        for _ in range(addr_count):
            addr, pos = _decode_addr(body, pos)
            addrs.append(addr)

        relay_tag = body[pos]
        pos += 1
        if relay_tag == _RELAY_TAG_NONE:
            relay_url: Optional[str] = None
        elif relay_tag == _RELAY_TAG_URL:
            url_len = body[pos]
            pos += 1
            relay_url = body[pos : pos + url_len].decode("utf-8")
            pos += url_len
        elif 1 <= relay_tag <= len(RELAY_TABLE):
            relay_url = RELAY_TABLE[relay_tag - 1]
        else:
            raise PairCodeError(f"Unknown relay table index: {relay_tag}")
    except (IndexError, struct.error, UnicodeDecodeError) as e:
        raise PairCodeError(
            "This pairing code is too short — it's probably truncated or mistyped"
        ) from e

    if pos != len(body):
        raise PairCodeError(
            "This pairing code has unexpected trailing data — it's probably mistyped"
        )

    iroh = build_iroh_section(node_id, relay_url, []) if relay_url else None
    return PairingTicket(
        node_id=node_id,
        addrs=addrs,
        secret=secret,
        expires_at=float(expires_at),
        iroh=iroh,
    )


def _encode_addr(addr: str) -> bytes:
    """Encode one ``ws://...`` dial address as a tagged addr entry."""
    parsed = _parse_ws_host_port(addr)
    if parsed is not None:
        ip, port = parsed
        tag = _ADDR_TAG_IPV4 if isinstance(ip, IPv4Address) else _ADDR_TAG_IPV6
        return bytes([tag]) + ip.packed + struct.pack(">H", port)

    raw = addr.encode("utf-8")
    if len(raw) > 255:
        raise ValueError(f"addr is too long to encode ({len(raw)} bytes): {addr!r}")
    return bytes([_ADDR_TAG_URL, len(raw)]) + raw


def _decode_addr(body: bytes, pos: int) -> Tuple[str, int]:
    """Decode one tagged addr entry starting at `pos`.

    Returns:
        `(addr, next_pos)`.
    """
    tag = body[pos]
    pos += 1
    if tag == _ADDR_TAG_IPV4:
        ip = IPv4Address(bytes(body[pos : pos + 4]))
        pos += 4
        (port,) = struct.unpack(">H", body[pos : pos + 2])
        pos += 2
        return f"ws://{ip}:{port}", pos
    if tag == _ADDR_TAG_IPV6:
        ip = IPv6Address(bytes(body[pos : pos + 16]))
        pos += 16
        (port,) = struct.unpack(">H", body[pos : pos + 2])
        pos += 2
        return f"ws://[{ip}]:{port}", pos
    if tag == _ADDR_TAG_URL:
        length = body[pos]
        pos += 1
        url = body[pos : pos + length].decode("utf-8")
        pos += length
        return url, pos
    raise PairCodeError(f"Unknown address tag: {tag}")


def _parse_ws_host_port(addr: str):
    """`(ip, port)` if `addr` is exactly ``ws://<ipv4>:<port>`` or
    ``ws://[<ipv6>]:<port>`` — the only shapes the addr format's IPv4/IPv6
    tags cover. Anything else (a hostname, a path, a different scheme, ...)
    returns `None` so the caller falls back to the generic URL tag.
    """
    parsed = urlsplit(addr)
    if (
        parsed.scheme != "ws"
        or parsed.hostname is None
        or parsed.port is None
        or parsed.path
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        return None
    try:
        return ip_address(parsed.hostname), parsed.port
    except ValueError:
        return None


def _encode_relay(iroh: Optional[dict]) -> bytes:
    """Encode the relay tag (+ optional URL) from a ticket's `iroh` section.

    Only `relay_url` is ever encoded — direct addresses are left out (see
    the module docstring).
    """
    relay_url = (iroh or {}).get("relay_url")
    if not relay_url:
        return bytes([_RELAY_TAG_NONE])
    if relay_url in RELAY_TABLE:
        return bytes([RELAY_TABLE.index(relay_url) + 1])
    raw = relay_url.encode("utf-8")
    if len(raw) > 255:
        raise ValueError(f"relay_url is too long to encode ({len(raw)} bytes)")
    return bytes([_RELAY_TAG_URL, len(raw)]) + raw


def _b32_encode(data: bytes) -> str:
    return base64.b32encode(data).decode("ascii").rstrip("=").lower()


def _b32_decode(payload: str) -> bytes:
    upper = payload.upper()
    padded = upper + "=" * (-len(upper) % 8)
    try:
        return base64.b32decode(padded)
    except binascii.Error as e:
        raise PairCodeError(f"Malformed pairing code: {e}") from e


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))
