"""Assembles and starts the protocol v1 worker server.

This is what the ``sleap-rtc serve`` CLI command (item 1.6) actually runs:
wires together identity, trust store, pairing tickets, job store, job
queue, and the protocol v1 server + its method handlers (jobs.*, fs.*,
pair.claim, auth.prove), reconciles the job store against reality
(`reattach_all`) before accepting new connections, then starts listening.
"""

import asyncio
import logging
import threading
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

from websockets.asyncio.server import Server

from sleap_rtc.jobs.builder import DEFAULT_ZMQ_PORTS
from sleap_rtc.jobs.process import reattach_all
from sleap_rtc.jobs.queue import JobQueue
from sleap_rtc.jobs.store import JobStore
from sleap_rtc.protocol_v1.auth import AuthMethods
from sleap_rtc.protocol_v1.blob_http import (
    make_blob_http_server,
    run_blob_http_server_in_thread,
)
from sleap_rtc.protocol_v1.blobs import BlobIndex
from sleap_rtc.protocol_v1.identity import WorkerIdentity
from sleap_rtc.protocol_v1.iroh_live import (
    iroh_live_path,
    keep_iroh_live_updated,
    remove_iroh_live,
    snapshot_iroh_section,
    write_iroh_live,
)
from sleap_rtc.protocol_v1.job_methods import JobMethods
from sleap_rtc.protocol_v1.pairing import PendingPairings
from sleap_rtc.protocol_v1.server import ProtocolV1Server
from sleap_rtc.protocol_v1.trust_store import TrustStore

DEFAULT_DATA_DIR = Path.home() / ".sleap-rtc" / "protocol_v1"
DEFAULT_PORT = 9631
# The blob HTTP server (spec §6.3) defaults to the WS port + 1 — one port
# to remember (or forward through a firewall/NAT), not a second one to
# separately configure.
DEFAULT_BLOB_PORT_OFFSET = 1


def identity_path(data_dir: Path) -> Path:
    """Where a worker's identity keypair lives under `data_dir`."""
    return data_dir / "identity.json"


def trust_store_path(data_dir: Path) -> Path:
    """Where the trusted-client allowlist lives under `data_dir`."""
    return data_dir / "trusted_clients.json"


def pairing_path(data_dir: Path) -> Path:
    """Where pending pairing tickets live under `data_dir`."""
    return data_dir / "pairing_tickets.json"


def job_store_path(data_dir: Path) -> Path:
    """Where the SQLite job store lives under `data_dir`."""
    return data_dir / "jobs.sqlite"


def job_log_dir(data_dir: Path) -> Path:
    """Where per-job subprocess logs live under `data_dir`."""
    return data_dir / "job-logs"


def blob_index_path(data_dir: Path) -> Path:
    """Where the result-blob index lives under `data_dir`."""
    return data_dir / "blobs.sqlite"


@dataclass
class WorkerServer:
    """A fully-assembled, running protocol v1 worker."""

    server: ProtocolV1Server
    identity: WorkerIdentity
    trust_store: TrustStore
    pending_pairings: PendingPairings
    job_methods: JobMethods
    store: JobStore
    ws_server: Server
    data_dir: Path
    reattach_outcomes: Dict[str, str]
    blob_index: BlobIndex
    blob_port: int
    _blob_http_server: ThreadingHTTPServer
    _blob_http_thread: threading.Thread
    # `None` unless `enable_iroh=True` (item 2.2) — a plain-WS-only worker
    # (still the default for `start_worker_server` itself; `sleap-rtc serve`
    # opts in) has no iroh endpoint to close.
    iroh_endpoint: Optional[Any] = None
    _iroh_serve_task: Optional[asyncio.Task] = None
    _iroh_live_task: Optional[asyncio.Task] = None

    async def close(self) -> None:
        """Stop listening and close the job store."""
        self.ws_server.close()
        await self.ws_server.wait_closed()
        await self.store.close()
        self._blob_http_server.shutdown()
        self._blob_http_server.server_close()
        self._blob_http_thread.join(timeout=5)
        self.blob_index.close()
        if self.iroh_endpoint is not None:
            self._iroh_live_task.cancel()
            remove_iroh_live(iroh_live_path(self.data_dir))
            await self.iroh_endpoint.close()
            self._iroh_serve_task.cancel()


async def start_worker_server(
    host: str = "0.0.0.0",
    port: int = DEFAULT_PORT,
    data_dir: Path = DEFAULT_DATA_DIR,
    file_manager: Optional[object] = None,
    blob_port: Optional[int] = None,
    enable_iroh: bool = False,
    iroh_preset: Optional[Any] = None,
    enable_metrics: bool = False,
) -> WorkerServer:
    """Assemble every protocol v1 piece and start listening.

    Args:
        host: Bind address.
        port: Bind port.
        data_dir: Where to persist identity/trust/pairing/job-store state
            and per-job logs.
        file_manager: An existing `FileManager`, if `fs.mounts`/`fs.list`
            should be exposed.
        blob_port: Bind port for the blob-serving HTTP endpoint (spec
            §6.3), which is what makes a completed track job's `job.result`
            actually carry a fetchable blob ref instead of `{}` (item
            1.10). Defaults to `port + 1`.
        enable_iroh: Also accept connections over iroh (item 2.2), so a
            client that can't reach this worker directly (no shared
            network/VPN/Tailscale) can still pair and connect via iroh's
            own direct-then-relay dialing, instead of only the plain-WS
            binding above. Defaults `False` here so every existing caller
            of this function (tests especially) is unaffected; `sleap-rtc
            serve` (the real CLI) opts in.
        enable_metrics: Forward each training job's local ZMQ epoch/loss
            stream as `job.metric`/`job.curve` events (item 3.1). Defaults
            `False` so every existing caller of this function (tests
            especially) is unaffected by a real ZMQ socket bind; `sleap-rtc
            serve` (the real CLI) opts in. Uses
            `sleap_rtc.jobs.builder.DEFAULT_ZMQ_PORTS` — the same ports
            `CommandBuilder` wires into the `sleap-nn` invocation itself.
        iroh_preset: Overrides iroh's own default `Preset` (which includes
            a real relay and reaches out to the real internet) — tests
            pass `iroh.preset_minimal()` so they have no external network
            dependency. Ignored unless `enable_iroh=True`.

    Returns:
        The running `WorkerServer` — `reattach_all` has already run by the
        time this returns, so any jobs left "running" from a previous
        instance have already been reconciled against reality.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if blob_port is None:
        # port=0 means "OS, pick any free port" for the WS server (tests
        # rely on this) — port + 1 would then usually land on a real,
        # privileged, unrelated port instead of mirroring that intent, so
        # mirror it exactly: 0 in, let the OS pick this one independently.
        blob_port = 0 if port == 0 else port + DEFAULT_BLOB_PORT_OFFSET

    identity = WorkerIdentity(identity_path(data_dir))
    trust_store = TrustStore(trust_store_path(data_dir))
    pending_pairings = PendingPairings(path=pairing_path(data_dir))

    store = JobStore(job_store_path(data_dir))
    await store.connect()

    reattach_outcomes = await reattach_all(store)
    for job_id, outcome in reattach_outcomes.items():
        logging.info(f"[serve] Startup reattach: job {job_id} -> {outcome}")

    blob_index = BlobIndex(blob_index_path(data_dir))
    blob_http_server = make_blob_http_server(blob_index, host, blob_port)
    blob_port = blob_http_server.server_address[1]  # the real bound port
    blob_http_thread = run_blob_http_server_in_thread(blob_http_server)

    server = ProtocolV1Server(
        node_id=identity.node_id,
        blob_port=blob_port,
        blob_index=blob_index,
        sign_nonce=identity.sign,
    )
    AuthMethods(server, identity, trust_store, pending_pairings)
    job_methods = JobMethods(
        server,
        store,
        JobQueue(max_concurrent=1),
        job_log_dir(data_dir),
        file_manager=file_manager,
        blob_index=blob_index,
        metrics_ports=DEFAULT_ZMQ_PORTS if enable_metrics else None,
    )

    ws_server = await server.serve(host, port)

    iroh_endpoint = None
    iroh_serve_task = None
    iroh_live_task = None
    if enable_iroh:
        # Deferred import: only a worker that actually enables iroh needs
        # the dependency importable at all.
        import iroh

        from sleap_rtc.protocol_v1.iroh_transport import ALPN

        endpoint_kwargs: Dict[str, Any] = {
            "secret_key": identity.iroh_secret_key_bytes,
            "alpns": [ALPN],
        }
        if iroh_preset is not None:
            endpoint_kwargs["preset"] = iroh_preset
        iroh_endpoint = await iroh.Endpoint.bind(
            iroh.EndpointOptions(**endpoint_kwargs)
        )
        iroh_serve_task = asyncio.create_task(server.serve_iroh(iroh_endpoint))
        # Publish this endpoint's reachability for `sleap-rtc pair` (a
        # separate process) to embed in tickets (item 2.1).
        write_iroh_live(
            iroh_live_path(data_dir),
            snapshot_iroh_section(iroh_endpoint, identity.node_id),
        )
        iroh_live_task = asyncio.create_task(
            keep_iroh_live_updated(
                iroh_endpoint, iroh_live_path(data_dir), identity.node_id
            )
        )

    return WorkerServer(
        server=server,
        identity=identity,
        trust_store=trust_store,
        pending_pairings=pending_pairings,
        job_methods=job_methods,
        store=store,
        blob_index=blob_index,
        blob_port=blob_port,
        _blob_http_server=blob_http_server,
        _blob_http_thread=blob_http_thread,
        ws_server=ws_server,
        data_dir=data_dir,
        reattach_outcomes=reattach_outcomes,
        iroh_endpoint=iroh_endpoint,
        _iroh_serve_task=iroh_serve_task,
        _iroh_live_task=iroh_live_task,
    )
