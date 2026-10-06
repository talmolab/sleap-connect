"""CLI commands for the protocol v1 worker: `serve`, `pair`, and extensions
to the existing `doctor`/`status` commands.

Kept in its own module (rather than added directly to the ~4000-line
`cli.py`) so the new, still-evolving protocol v1 surface stays easy to
find; `cli.py` just imports and registers `serve`/`pair`, and calls the
`print_*_section` functions from within the existing `doctor`/`status`
commands.

**Not included in this PR:** OS service installation (`service install` —
systemd --user+linger / LaunchAgent / Windows Task Scheduler). `serve` runs
in the foreground; wrapping it as a persistent background service is a
separate, substantial, OS-specific piece of work and a natural follow-up,
not bundled in here.
"""

import asyncio
import json
import textwrap
from pathlib import Path
from typing import Optional

import click

from sleap_rtc.config import MountConfig
from sleap_rtc.protocol_v1.runner import (
    DEFAULT_DATA_DIR,
    DEFAULT_PORT,
    identity_path,
    pairing_path,
    start_worker_server,
    trust_store_path,
)
from sleap_rtc.worker.file_manager import FileManager


def _parse_mount(raw: str) -> MountConfig:
    """Parse a `--mount` value: "PATH" or "PATH:LABEL"."""
    path, sep, label = raw.partition(":")
    return MountConfig(path=path, label=label if sep else Path(path).name or path)


@click.command(name="serve")
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind address.")
@click.option("--port", default=DEFAULT_PORT, show_default=True, help="Bind port.")
@click.option(
    "--blob-port",
    default=None,
    type=int,
    help="Bind port for the blob-serving HTTP endpoint (spec §6.3) — a "
    "completed track job's predictions are fetched from here. Defaults "
    "to --port + 1.",
)
@click.option(
    "--data-dir",
    default=str(DEFAULT_DATA_DIR),
    show_default=True,
    help="Where to persist identity, trust store, pairing tickets, and jobs.",
)
@click.option(
    "--mount",
    "mounts",
    multiple=True,
    help="A directory clients may browse (fs.mounts/fs.list), as PATH or "
    "PATH:LABEL. Repeatable. fs.mounts/fs.list are always available even "
    "with none given — they just report an empty mount list.",
)
@click.option(
    "--iroh/--no-iroh",
    default=True,
    show_default=True,
    help="Also accept connections over iroh (item 2.2) — works even "
    "without a shared network/VPN/Tailscale, via iroh's own "
    "direct-then-relay dialing. Additive: the plain WS binding above is "
    "always available regardless of this flag.",
)
def serve(
    host: str,
    port: int,
    blob_port: Optional[int],
    data_dir: str,
    mounts: tuple,
    iroh: bool,
):
    """Run this machine as a sleap-connect worker (protocol v1).

    Starts the worker's own server directly — no signaling server, no
    rooms. A client pairs with this worker (see `sleap-rtc pair`) and
    connects straight to it over localhost/LAN/Tailscale, or via iroh if
    it can't reach this machine directly (see --iroh).

    Runs in the foreground; press Ctrl-C to stop. To keep it running
    persistently, use your OS's own service manager for now (systemd
    --user, a LaunchAgent, or Task Scheduler) — first-class `service
    install` support is a planned follow-up.

    Example:
        sleap-rtc serve --port 9631 --mount /data/videos:lab-data
    """
    asyncio.run(_serve_async(host, port, blob_port, Path(data_dir), mounts, iroh))


async def _serve_async(
    host: str,
    port: int,
    blob_port: Optional[int],
    data_dir: Path,
    mounts: tuple = (),
    enable_iroh: bool = True,
) -> None:
    file_manager = FileManager(mounts=[_parse_mount(m) for m in mounts])
    worker = await start_worker_server(
        host=host,
        port=port,
        data_dir=data_dir,
        blob_port=blob_port,
        file_manager=file_manager,
        enable_iroh=enable_iroh,
    )
    try:
        click.echo(click.style("sleap-connect worker", bold=True))
        click.echo(f"  node_id:   {worker.identity.node_id}")
        click.echo(f"  address:   ws://{host}:{port}")
        click.echo(f"  blob port: {worker.blob_port}")
        click.echo(f"  data dir:  {worker.data_dir}")
        if worker.iroh_endpoint is not None:
            click.echo(
                "  iroh:      enabled (same node_id as above; a client "
                "reachable only via relay/hole-punch can still connect)"
            )

        if worker.reattach_outcomes:
            click.echo("")
            click.echo(click.style("Reattached from a previous run:", bold=True))
            for job_id, outcome in worker.reattach_outcomes.items():
                click.echo(f"  {job_id}: {outcome}")

        if not worker.trust_store.list_trusted():
            click.echo("")
            click.echo(
                click.style(
                    "No paired clients yet — run 'sleap-rtc pair' to generate a "
                    "pairing ticket.",
                    fg="yellow",
                )
            )

        click.echo("")
        click.echo("Listening. Press Ctrl-C to stop.")
        await worker.ws_server.wait_closed()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await worker.close()


@click.command(name="pair")
@click.option(
    "--addr",
    "addrs",
    multiple=True,
    help="A direct address a client can dial (e.g. ws://192.168.1.42:9631). "
    "Repeatable. If omitted, the ticket carries no address hints — the "
    "client must already know how to reach this worker.",
)
@click.option(
    "--ttl",
    default=300,
    show_default=True,
    help="How many seconds the pairing secret stays valid.",
)
@click.option(
    "--data-dir",
    default=str(DEFAULT_DATA_DIR),
    show_default=True,
    help="Must match the --data-dir a running 'sleap-rtc serve' is using.",
)
def pair(addrs: tuple, ttl: int, data_dir: str):
    """Generate a pairing ticket for a new client.

    Prints a ticket (as JSON) that a new client uses to establish trust
    with this worker on first contact — via 'pair.claim' in the protocol.
    The ticket's secret is single-use and expires after --ttl seconds.

    If a 'sleap-rtc serve --iroh' worker is running against the same
    --data-dir, the ticket also carries an 'iroh' section (node_id,
    relay_url, direct_addrs) so a client with no direct route can dial it.

    Works whether or not 'sleap-rtc serve' is currently running: pending
    tickets are shared via a small file under --data-dir, so a running
    serve process picks up tickets minted here without needing to be
    restarted (see PendingPairings).

    Example:
        sleap-rtc pair --addr ws://192.168.1.42:9631
    """
    from sleap_rtc.protocol_v1.identity import WorkerIdentity
    from sleap_rtc.protocol_v1.pairing import PendingPairings

    data_dir_path = Path(data_dir)
    identity = WorkerIdentity(identity_path(data_dir_path))
    pending_pairings = PendingPairings(ttl_secs=ttl, path=pairing_path(data_dir_path))

    from sleap_rtc.protocol_v1.iroh_live import iroh_live_path, read_iroh_live

    iroh_section = read_iroh_live(iroh_live_path(data_dir_path), identity.node_id)
    ticket = pending_pairings.create(identity.node_id, list(addrs), iroh=iroh_section)

    click.echo(click.style("Pairing ticket", bold=True))
    click.echo(json.dumps(ticket.to_dict(), indent=2))
    click.echo("")
    if iroh_section is not None:
        click.echo(
            "This ticket includes iroh dial info from the running worker, so "
            "the client can connect even without a shared network."
        )
        click.echo("")
    click.echo(textwrap.dedent(f"""\
            Give this to the new client — it's single-use and expires in
            {ttl} seconds. The client sends a 'pair.claim' request with this
            ticket's secret and its own node_id to establish trust.
            """))


def print_doctor_section(data_dir: Path = DEFAULT_DATA_DIR) -> bool:
    """Print a "Protocol v1 Worker" section for the `doctor` command.

    Args:
        data_dir: Same default `serve`/`pair` use.

    Returns:
        True if everything checked out; False if something looks wrong
        (mirrors the existing `doctor` command's `all_ok` convention).
    """
    all_ok = True

    click.echo("")
    click.echo(click.style("Protocol v1 Worker:", bold=True))

    for dep in ("psutil", "aiosqlite", "websockets", "cryptography"):
        try:
            __import__(dep)
            click.echo(f"  {dep}: {click.style('✓', fg='green')}")
        except ImportError:
            click.echo(f"  {dep}: {click.style('✗ missing', fg='red')}")
            all_ok = False

    id_path = identity_path(data_dir)
    if id_path.exists():
        from sleap_rtc.protocol_v1.identity import WorkerIdentity

        identity = WorkerIdentity(id_path)
        click.echo(f"  Identity: {click.style('✓', fg='green')} ({identity.node_id})")
    else:
        click.echo(
            f"  Identity: {click.style('not yet generated', fg='yellow')} "
            f"(created on first 'sleap-rtc serve' or 'sleap-rtc pair')"
        )

    trust_path = trust_store_path(data_dir)
    if trust_path.exists():
        from sleap_rtc.protocol_v1.trust_store import TrustStore

        n = len(TrustStore(trust_path).list_trusted())
        click.echo(f"  Paired clients: {n}")
    else:
        click.echo("  Paired clients: 0 (no trust store yet)")

    return all_ok


def print_status_section(data_dir: Path = DEFAULT_DATA_DIR) -> None:
    """Print a "Protocol v1 Worker" section for the `status` command.

    Args:
        data_dir: Same default `serve`/`pair` use.
    """
    click.echo("")
    click.echo(click.style("Protocol v1 Worker:", bold=True))

    id_path = identity_path(data_dir)
    if not id_path.exists():
        click.echo("  Not yet initialized (no 'sleap-rtc serve' or 'pair' run yet)")
        return

    from sleap_rtc.protocol_v1.identity import WorkerIdentity
    from sleap_rtc.protocol_v1.trust_store import TrustStore

    identity = WorkerIdentity(id_path)
    click.echo(f"  node_id: {identity.node_id}")

    trust_path = trust_store_path(data_dir)
    n_trusted = len(TrustStore(trust_path).list_trusted()) if trust_path.exists() else 0
    click.echo(f"  Paired clients: {n_trusted}")

    store_path = data_dir / "jobs.sqlite"
    if store_path.exists():
        asyncio.run(_print_recent_jobs(store_path))
    else:
        click.echo("  Jobs: none yet")


async def _print_recent_jobs(store_path: Path, limit: int = 5) -> None:
    from sleap_rtc.jobs.store import JobStore

    async with JobStore(store_path) as store:
        jobs = await store.list_jobs()

    if not jobs:
        click.echo("  Jobs: none yet")
        return

    click.echo(f"  Recent jobs (showing up to {limit}):")
    for record in jobs[:limit]:
        click.echo(f"    {record.job_id}: {record.state}")
