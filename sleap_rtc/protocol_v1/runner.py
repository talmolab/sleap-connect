"""Assembles and starts the protocol v1 worker server.

This is what the ``sleap-rtc serve`` CLI command (item 1.6) actually runs:
wires together identity, trust store, pairing tickets, job store, job
queue, and the protocol v1 server + its method handlers (jobs.*, fs.*,
pair.claim, auth.prove), reconciles the job store against reality
(`reattach_all`) before accepting new connections, then starts listening.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from websockets.asyncio.server import Server

from sleap_rtc.jobs.process import reattach_all
from sleap_rtc.jobs.queue import JobQueue
from sleap_rtc.jobs.store import JobStore
from sleap_rtc.protocol_v1.auth import AuthMethods
from sleap_rtc.protocol_v1.identity import WorkerIdentity
from sleap_rtc.protocol_v1.job_methods import JobMethods
from sleap_rtc.protocol_v1.pairing import PendingPairings
from sleap_rtc.protocol_v1.server import ProtocolV1Server
from sleap_rtc.protocol_v1.trust_store import TrustStore

DEFAULT_DATA_DIR = Path.home() / ".sleap-rtc" / "protocol_v1"
DEFAULT_PORT = 9631


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

    async def close(self) -> None:
        """Stop listening and close the job store."""
        self.ws_server.close()
        await self.ws_server.wait_closed()
        await self.store.close()


async def start_worker_server(
    host: str = "0.0.0.0",
    port: int = DEFAULT_PORT,
    data_dir: Path = DEFAULT_DATA_DIR,
    file_manager: Optional[object] = None,
) -> WorkerServer:
    """Assemble every protocol v1 piece and start listening.

    Args:
        host: Bind address.
        port: Bind port.
        data_dir: Where to persist identity/trust/pairing/job-store state
            and per-job logs.
        file_manager: An existing `FileManager`, if `fs.mounts`/`fs.list`
            should be exposed.

    Returns:
        The running `WorkerServer` — `reattach_all` has already run by the
        time this returns, so any jobs left "running" from a previous
        instance have already been reconciled against reality.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    identity = WorkerIdentity(identity_path(data_dir))
    trust_store = TrustStore(trust_store_path(data_dir))
    pending_pairings = PendingPairings(path=pairing_path(data_dir))

    store = JobStore(job_store_path(data_dir))
    await store.connect()

    reattach_outcomes = await reattach_all(store)
    for job_id, outcome in reattach_outcomes.items():
        logging.info(f"[serve] Startup reattach: job {job_id} -> {outcome}")

    server = ProtocolV1Server(node_id=identity.node_id)
    AuthMethods(server, identity, trust_store, pending_pairings)
    job_methods = JobMethods(
        server,
        store,
        JobQueue(max_concurrent=1),
        job_log_dir(data_dir),
        file_manager=file_manager,
    )

    ws_server = await server.serve(host, port)

    return WorkerServer(
        server=server,
        identity=identity,
        trust_store=trust_store,
        pending_pairings=pending_pairings,
        job_methods=job_methods,
        store=store,
        ws_server=ws_server,
        data_dir=data_dir,
        reattach_outcomes=reattach_outcomes,
    )
