"""CLI commands for the protocol v1 worker: `serve`, `pair`, and extensions
to the existing `doctor`/`status` commands.

Kept in its own module (rather than added directly to the ~4000-line
`cli.py`) so the new, still-evolving protocol v1 surface stays easy to
find; `cli.py` just imports and registers `serve`/`pair`, and calls the
`print_*_section` functions from within the existing `doctor`/`status`
commands.

`serve --daemonize` / `serve --stop` run and stop the worker in the
background (see `daemon.py`). OS service installation (`service install` —
systemd --user / LaunchAgent / Windows Task Scheduler), for a worker that
also comes back after a reboot, is a separate follow-up.
"""

import asyncio
import json
import os
import signal
import sys
import textwrap
from pathlib import Path
from typing import Optional

import click

from sleap_rtc.config import MountConfig
from sleap_rtc.protocol_v1 import daemon
from sleap_rtc.protocol_v1.runner import (
    DEFAULT_DATA_DIR,
    DEFAULT_PORT,
    identity_path,
    pairing_path,
    start_worker_server,
    trust_store_path,
)
from sleap_rtc.worker.file_manager import FileManager

# How long `serve` waits at startup for its iroh endpoint to reach a home
# relay, so the first `pair` ticket is dialable from outside the LAN.
IROH_ONLINE_TIMEOUT_SECS = 10.0


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
@click.option(
    "--metrics/--no-metrics",
    default=True,
    show_default=True,
    help="Forward each training job's local ZMQ epoch/loss stream as "
    "job.metric/job.curve events (item 3.1), so a remote client sees live "
    "training progress instead of only raw log lines.",
)
@click.option(
    "--daemonize",
    is_flag=True,
    help="Run in the background: detach from this terminal (so it keeps "
    "running after the terminal or SSH session closes) and return once the "
    "worker is listening. Output goes to serve.log under --data-dir.",
)
@click.option(
    "--stop",
    is_flag=True,
    help="Stop the worker running against --data-dir (e.g. one started with "
    "--daemonize) and exit. Running jobs keep running and are reattached "
    "by the next 'serve'.",
)
@click.option(
    "--ready-file",
    default=None,
    hidden=True,
    help="Internal: written once listening, so 'serve --daemonize' knows "
    "its child is up.",
)
def serve(
    host: str,
    port: int,
    blob_port: Optional[int],
    data_dir: str,
    mounts: tuple,
    iroh: bool,
    metrics: bool,
    daemonize: bool,
    stop: bool,
    ready_file: Optional[str],
):
    """Run this machine as a sleap-connect worker (protocol v1).

    Starts the worker's own server directly — no signaling server, no
    rooms. A client pairs with this worker (see `sleap-rtc pair`) and
    connects straight to it over localhost/LAN/Tailscale, or via iroh if
    it can't reach this machine directly (see --iroh).

    Runs in the foreground by default; press Ctrl-C to stop. With
    --daemonize it runs in the background instead, surviving the terminal
    or SSH session that started it, until 'serve --stop'. (It does not come
    back after a reboot — that needs an OS service, a planned follow-up.)

    Only one worker can run per --data-dir at a time.

    Examples:
        sleap-rtc serve --port 9631 --mount /data/videos:lab-data
        sleap-rtc serve --daemonize --mount /data/videos:lab-data
        sleap-rtc serve --stop
    """
    data_dir_path = Path(data_dir)

    if stop:
        _stop(data_dir_path)
        return

    if daemonize:
        _daemonize(
            daemon.serve_argv(
                host=host,
                port=port,
                blob_port=blob_port,
                data_dir=data_dir_path,
                mounts=mounts,
                iroh=iroh,
                metrics=metrics,
            ),
            data_dir_path,
        )
        return

    try:
        daemon.acquire_pidfile(data_dir_path)
    except daemon.AlreadyRunningError as e:
        raise click.ClickException(_already_running_message(e, data_dir_path))
    try:
        asyncio.run(
            _serve_async(
                host,
                port,
                blob_port,
                data_dir_path,
                mounts,
                iroh,
                metrics,
                ready_file=Path(ready_file) if ready_file else None,
            )
        )
    finally:
        daemon.release_pidfile(data_dir_path)


def _already_running_message(e: daemon.AlreadyRunningError, data_dir: Path) -> str:
    return (
        f"A worker is already running against {data_dir} (pid "
        f"{e.running.pid}). Stop it first with 'sleap-rtc serve --stop', or "
        f"use a different --data-dir."
    )


def _daemonize(argv: list, data_dir: Path) -> None:
    running = daemon.read_running(data_dir)
    if running is not None:
        raise click.ClickException(
            _already_running_message(daemon.AlreadyRunningError(running), data_dir)
        )

    click.echo("Starting worker in the background...")
    try:
        info = daemon.spawn_daemon(argv, data_dir)
    except daemon.DaemonStartError as e:
        if e.log_tail:
            click.echo("")
            click.echo(e.log_tail, err=True)
            click.echo("")
        raise click.ClickException(f"{e} (full log: {daemon.log_path(data_dir)})")

    click.echo(click.style("sleap-connect worker running in the background", bold=True))
    click.echo(f"  pid:       {info['pid']}")
    click.echo(f"  node_id:   {info['node_id']}")
    click.echo(f"  address:   {info['address']}")
    click.echo(f"  log:       {daemon.log_path(data_dir)}")
    click.echo("")
    click.echo(
        "Pair a client with 'sleap-rtc pair'; stop with 'sleap-rtc serve --stop'."
    )


def _stop(data_dir: Path) -> None:
    try:
        stopped = daemon.stop_running(data_dir)
    except TimeoutError as e:
        raise click.ClickException(str(e))
    if stopped is None:
        click.echo(f"No worker is running against {data_dir}.")
    else:
        click.echo(f"Stopped worker (pid {stopped.pid}).")


async def _serve_async(
    host: str,
    port: int,
    blob_port: Optional[int],
    data_dir: Path,
    mounts: tuple = (),
    enable_iroh: bool = True,
    enable_metrics: bool = True,
    ready_file: Optional[Path] = None,
) -> None:
    # SIGTERM (`serve --stop`, `kill`, a service manager) gets the same clean
    # shutdown as Ctrl-C instead of dying mid-write. Windows has no SIGTERM
    # delivery to hook (see `daemon.stop_running`).
    if sys.platform != "win32":
        asyncio.get_running_loop().add_signal_handler(
            signal.SIGTERM, asyncio.current_task().cancel
        )

    file_manager = FileManager(mounts=[_parse_mount(m) for m in mounts])
    worker = await start_worker_server(
        host=host,
        port=port,
        data_dir=data_dir,
        blob_port=blob_port,
        file_manager=file_manager,
        enable_iroh=enable_iroh,
        enable_metrics=enable_metrics,
        iroh_online_timeout=IROH_ONLINE_TIMEOUT_SECS,
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
        if ready_file is not None:
            daemon.write_ready(
                ready_file,
                {
                    "pid": os.getpid(),
                    "node_id": worker.identity.node_id,
                    "address": f"ws://{host}:{port}",
                    "blob_port": worker.blob_port,
                },
            )
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
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Print the full ticket as JSON (today's format: node_id, addrs, "
    "secret, expires_at, and — if iroh is live — an iroh section including "
    "direct_addrs) instead of the default one-line pairing code. For "
    "scripts, or a client that wants to dial iroh direct addresses "
    "without going through a relay first.",
)
def pair(addrs: tuple, ttl: int, data_dir: str, as_json: bool):
    """Generate a pairing ticket for a new client.

    By default, prints a single-line pairing code (`sleap1...`) — select
    and copy that one line, paste it into the app's Connect tab. Pass
    --json to print the full ticket as JSON instead (today's format;
    scripts/back-compat).

    The ticket's secret is single-use and expires after --ttl seconds. If
    a 'sleap-rtc serve --iroh' worker is running against the same
    --data-dir, a client with no direct route can still dial it: the
    pairing code carries the worker's current relay, and --json's ticket
    additionally carries iroh's node_id/relay_url/direct_addrs.

    Works whether or not 'sleap-rtc serve' is currently running: pending
    tickets are shared via a small file under --data-dir, so a running
    serve process picks up tickets minted here without needing to be
    restarted (see PendingPairings).

    Example:
        sleap-rtc pair --addr ws://192.168.1.42:9631
    """
    from sleap_rtc.protocol_v1.identity import WorkerIdentity
    from sleap_rtc.protocol_v1.pair_code import encode_pair_code
    from sleap_rtc.protocol_v1.pairing import PendingPairings

    data_dir_path = Path(data_dir)
    identity = WorkerIdentity(identity_path(data_dir_path))
    pending_pairings = PendingPairings(ttl_secs=ttl, path=pairing_path(data_dir_path))

    from sleap_rtc.protocol_v1.iroh_live import iroh_live_path, read_iroh_live

    iroh_section = read_iroh_live(iroh_live_path(data_dir_path), identity.node_id)
    ticket = pending_pairings.create(identity.node_id, list(addrs), iroh=iroh_section)

    if as_json:
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
        return

    click.echo(click.style("Pairing code", bold=True))
    click.echo(encode_pair_code(ticket))
    click.echo("")
    click.echo("In SLEAP: Connect → Pair a worker → paste the code")
    click.echo(f"Expires in {ttl} seconds.")


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

    running = daemon.read_running(data_dir)
    if running is None:
        click.echo("  Worker: not running")
    else:
        click.echo(f"  Worker: running (pid {running.pid})")

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
