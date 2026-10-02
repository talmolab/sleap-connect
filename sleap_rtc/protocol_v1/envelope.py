"""Wire envelope for the sleap-connect protocol v1.

Implements the four frame shapes from the protocol spec
(``docs/plans/2026-09-26-sleap-connect-protocol-v1-spec.md`` in sleap-app):
every message on the wire is one JSON object of the form
``{"v": 1, "type": "hello" | "req" | "res" | "event", ...}``.

This module only handles frame (de)serialization and identification;
dispatch and transport live in `server.py`.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

PROTOCOL_VERSION = 1


class EnvelopeError(Exception):
    """Raised when a raw wire message doesn't parse as a valid envelope frame."""


@dataclass
class Hello:
    """The `hello` frame — sent by both sides immediately on connect.

    Attributes:
        proto: This side's supported protocol version range, e.g.
            ``{"min": 1, "max": 1}``.
        agent: Identifies the sender — ``{"name", "version", "platform"}``.
        node_id: This side's persistent public-key identity (base64).
        nonce: Random per-connection nonce for the other side to sign back
            in an `auth.prove` request (see protocol spec §3.3).
        blob_port: The worker's blob-serving HTTP port (spec §6.3), on the
            same host the client dialed for this WS connection — e.g.
            `GET http://<that host>:<blob_port>/blobs/<sha256>`. Additive
            per §2.4 (new optional fields don't bump `v`); `None` on a
            worker not running the blob HTTP server, or when sent by a
            client (only workers serve blobs today). Only meaningful in the
            worker's own `hello`, never the client's.
        proof: Symmetric auth: the WORKER's signature over the CLIENT's
            `nonce` (the one in the client's own hello, previously unused —
            see `nonce` above), proving the worker genuinely holds the
            private key for the `node_id` it just claimed in this same
            frame. The client verifies it immediately, before ever
            attempting `pair.claim`/`auth.prove` — closing the gap where an
            impostor could otherwise just lie about `node_id` in its hello
            with no cryptographic proof at all. Only meaningful in the
            worker's own hello; `None` when sent by a client (a client has
            no long-lived key the OTHER side verifies this way — the
            reverse direction, client-proves-to-worker, is what
            `pair.claim`/`auth.prove` already do).
        v: Envelope version.
    """

    proto: Dict[str, int]
    agent: Dict[str, str]
    node_id: str
    nonce: str
    blob_port: Optional[int] = None
    proof: Optional[str] = None
    v: int = PROTOCOL_VERSION

    def to_dict(self) -> dict:
        """Serialize to the wire dict shape."""
        d = {
            "v": self.v,
            "type": "hello",
            "proto": self.proto,
            "agent": self.agent,
            "node_id": self.node_id,
            "nonce": self.nonce,
        }
        if self.blob_port is not None:
            d["blob_port"] = self.blob_port
        if self.proof is not None:
            d["proof"] = self.proof
        return d

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "Hello":
        """Deserialize from a parsed wire dict."""
        return cls(
            v=d.get("v", PROTOCOL_VERSION),
            proto=d["proto"],
            agent=d["agent"],
            node_id=d["node_id"],
            nonce=d["nonce"],
            blob_port=d.get("blob_port"),
            proof=d.get("proof"),
        )


@dataclass
class Req:
    """The `req` frame — a request, correlated to its `res` by `id`.

    Attributes:
        id: Per-connection monotonic integer, owned by the requester (each
            side counts its own outgoing requests independently).
        method: Dotted method name, e.g. ``"jobs.submit"``.
        params: Method-specific parameters.
        v: Envelope version.
    """

    id: int
    method: str
    params: Dict[str, Any] = field(default_factory=dict)
    v: int = PROTOCOL_VERSION

    def to_dict(self) -> dict:
        """Serialize to the wire dict shape."""
        return {
            "v": self.v,
            "type": "req",
            "id": self.id,
            "method": self.method,
            "params": self.params,
        }

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "Req":
        """Deserialize from a parsed wire dict."""
        return cls(
            v=d.get("v", PROTOCOL_VERSION),
            id=d["id"],
            method=d["method"],
            params=d.get("params", {}),
        )


@dataclass
class Res:
    """The `res` frame — a reply to a `req`, carrying `result` xor `error`.

    Attributes:
        id: Echoes the `req.id` this replies to.
        result: The method's return value. Mutually exclusive with `error`.
        error: ``{"code": str, "msg": str, "data": Any}`` — see
            `sleap_rtc.protocol_v1.errors` for the code taxonomy. Mutually
            exclusive with `result`.
        v: Envelope version.
    """

    id: int
    result: Optional[dict] = None
    error: Optional[dict] = None
    v: int = PROTOCOL_VERSION

    def to_dict(self) -> dict:
        """Serialize to the wire dict shape."""
        d = {"v": self.v, "type": "res", "id": self.id}
        if self.error is not None:
            d["error"] = self.error
        else:
            d["result"] = self.result if self.result is not None else {}
        return d

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "Res":
        """Deserialize from a parsed wire dict."""
        return cls(
            v=d.get("v", PROTOCOL_VERSION),
            id=d["id"],
            result=d.get("result"),
            error=d.get("error"),
        )

    @classmethod
    def ok(cls, id: int, result: Optional[dict] = None) -> "Res":
        """Build a success reply."""
        return cls(id=id, result=result if result is not None else {})

    @classmethod
    def err(cls, id: int, code: str, msg: str, data: Any = None) -> "Res":
        """Build an error reply."""
        error: Dict[str, Any] = {"code": code, "msg": msg}
        if data is not None:
            error["data"] = data
        return cls(id=id, error=error)


@dataclass
class Event:
    """The `event` frame — unsolicited, server→client, per-job sequenced.

    Attributes:
        topic: Event topic, e.g. ``"job.status"``, ``"job.log"``,
            ``"job.metric"``, ``"job.curve"``, ``"job.result"``.
        seq: Per-job monotonic sequence number (starts at 1) — what makes
            events-since-N replay possible.
        data: Topic-specific payload.
        job_id: The job this event belongs to. Omitted for connection-level
            events (none defined yet in this PR).
        v: Envelope version.
    """

    topic: str
    seq: int
    data: Dict[str, Any]
    job_id: Optional[str] = None
    v: int = PROTOCOL_VERSION

    def to_dict(self) -> dict:
        """Serialize to the wire dict shape."""
        d = {
            "v": self.v,
            "type": "event",
            "topic": self.topic,
            "seq": self.seq,
            "data": self.data,
        }
        if self.job_id is not None:
            d["job_id"] = self.job_id
        return d

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        """Deserialize from a parsed wire dict."""
        return cls(
            v=d.get("v", PROTOCOL_VERSION),
            topic=d["topic"],
            seq=d["seq"],
            data=d.get("data", {}),
            job_id=d.get("job_id"),
        )


Envelope = Union[Hello, Req, Res, Event]

_FRAME_TYPES = {
    "hello": Hello,
    "req": Req,
    "res": Res,
    "event": Event,
}


def parse_envelope(raw: str) -> Envelope:
    """Parse a raw wire message into the appropriate frame type.

    Args:
        raw: The raw JSON text received from the transport.

    Returns:
        A `Hello`, `Req`, `Res`, or `Event` instance.

    Raises:
        EnvelopeError: If `raw` isn't valid JSON, has an unrecognized or
            missing `type`, or is missing a field required for its type.
    """
    try:
        d = json.loads(raw)
    except json.JSONDecodeError as e:
        raise EnvelopeError(f"Invalid JSON: {e}") from e

    if not isinstance(d, dict):
        raise EnvelopeError("Envelope frame must be a JSON object")

    frame_type = d.get("type")
    cls = _FRAME_TYPES.get(frame_type)
    if cls is None:
        raise EnvelopeError(f"Unknown or missing envelope type: {frame_type!r}")

    try:
        return cls.from_dict(d)
    except KeyError as e:
        raise EnvelopeError(
            f"Missing required field {e} for type {frame_type!r}"
        ) from e
