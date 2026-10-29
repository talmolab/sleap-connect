"""CLI commands for the protocol v1 worker: `serve`, `pair`, `status`,
`shell-hook`, and an extension to the existing `doctor` command.

Kept in its own module (rather than added directly to the ~4000-line
`cli.py`) so the new, still-evolving protocol v1 surface stays easy to
find; `cli.py` just imports and registers these commands, and calls
`print_doctor_section` from within the existing `doctor` command.

`serve` runs the worker in the background by default (see `daemon.py`;
`--foreground` keeps it attached), and `serve --stop` stops it. The first
background start also adds a new-terminal reminder to the user's shell
startup file (see `shell_hook.py`). OS service installation (`service
install` — systemd --user / LaunchAgent / Windows Task Scheduler), for a
worker that also comes back after a reboot, is a separate follow-up.
"""

import asyncio
import json
import os
import shlex
import signal
import sys
import textwrap
from pathlib import Path
import time
from typing import Optional

import click

from sleap_rtc.protocol_v1 import daemon, shell_hook
from sleap_rtc.protocol_v1.runner import (
    DEFAULT_DATA_DIR,
    DEFAULT_PORT,
    devices_path,
    identity_path,
    open_shared_folders,
    open_trust_store,
    pairing_path,
    start_worker_server,
    trust_store_path,
)
from sleap_rtc.protocol_v1.shared_folders import (
    SharedFolderFileManager,
    SharedFolders,
    is_broad,
    parse_mount,
    resolve_folder,
)

# How long `serve` waits at startup for its iroh endpoint to reach a home
# relay, so the first `pair` ticket is dialable from outside the LAN.
IROH_ONLINE_TIMEOUT_SECS = 10.0


def _can_ask() -> bool:
    """Whether someone is at a terminal to answer a y/N prompt.

    Both ends: a script or test that captures output may still leave stdin
    attached to a console (e.g. on Windows).
    """
    return sys.stdin.isatty() and sys.stdout.isatty()


def _glyph(symbol: str, fallback: str) -> str:
    """`symbol`, or `fallback` where stdout can't encode it (e.g. a
    Windows pipe in a legacy code page)."""
    try:
        symbol.encode(sys.stdout.encoding or "ascii")
    except (UnicodeEncodeError, LookupError):
        return fallback
    return symbol


def _share_folder(
    shared: SharedFolders,
    path: Path,
    label: Optional[str] = None,
    *,
    yes: bool,
    interactive: bool,
) -> bool:
    """Share `path` after confirming (y/N). True if it's shared now.

    A folder already shared (itself or inside a shared folder) needs no
    confirmation. Without a terminal to ask on, it's left unshared unless
    `yes`.
    """
    if shared.find(path) is not None:
        return True
    if not path.is_dir():
        raise click.ClickException(f"Not a folder: {path}")
    if not yes:
        if not interactive:
            return False
        click.echo(
            f"Share {click.style(str(path), bold=True)} with devices paired to this worker?"
        )
        click.echo(
            "  They can browse it, and their training/inference jobs can read and write inside it."
        )
        if is_broad(path):
            click.echo(
                click.style(
                    "  This is your whole home folder (or more). Consider sharing just "
                    "your data folder instead.",
                    fg="yellow",
                )
            )
        if not click.confirm("Share this folder?", default=True):
            return False
    shared.add(path, label)
    return True


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
    help="Also share this folder (PATH or PATH:LABEL; relative paths are "
    "fine), like 'sleap-rtc add-folder'. Repeatable. Asked once, then "
    "remembered.",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Share the folder 'serve' starts in (and any --mount) without asking.",
)
@click.option(
    "--serve-folder",
    default=None,
    hidden=True,
    help="Internal: the folder the user ran 'serve --daemonize' in.",
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
    "--foreground",
    is_flag=True,
    help="Stay attached to this terminal (press Ctrl-C to stop) instead of "
    "running in the background.",
)
@click.option(
    "--daemonize",
    is_flag=True,
    hidden=True,
    help="Run in the background. This is now the default; kept so existing "
    "commands keep working.",
)
@click.option(
    "--stop",
    is_flag=True,
    help="Stop the worker running against --data-dir and exit. Running jobs "
    "keep running and are reattached by the next 'serve'.",
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
    yes: bool,
    serve_folder: Optional[str],
    iroh: bool,
    metrics: bool,
    foreground: bool,
    daemonize: bool,
    stop: bool,
    ready_file: Optional[str],
):
    """Run this machine as a sleap-connect worker (protocol v1).

    Starts the worker's own server directly — no signaling server, no
    rooms. A client pairs with this worker (see `sleap-rtc pair`) and
    connects straight to it over localhost/LAN/Tailscale, or via iroh if
    it can't reach this machine directly (see --iroh).

    Runs in the background: it returns once the worker is listening, and
    keeps running after you close the terminal or log out, until 'serve
    --stop'. (It does not come back after a reboot — that needs an OS
    service, a planned follow-up.) Check on it with 'sleap-rtc status'. The
    first time, it also adds a one-line reminder to your shell's startup
    file, shown in new terminals while the worker runs (remove it with
    'sleap-rtc shell-hook uninstall'). With --foreground it stays attached
    to this terminal instead; press Ctrl-C to stop.

    Only one worker can run per --data-dir at a time.

    The folder you start it in is shared with paired devices — asked the
    first time (y/N), then remembered. Run it in your data folder. Share
    more with --mount or 'sleap-rtc add-folder'.

    Examples:
        cd /data/project && sleap-rtc serve
        sleap-rtc serve --mount /data/videos:lab-data
        sleap-rtc status
        sleap-rtc serve --stop
    """
    data_dir_path = Path(data_dir).expanduser()

    if stop:
        _stop(data_dir_path)
        return

    serve_folder_path = Path(serve_folder) if serve_folder else Path.cwd().resolve()
    if ready_file is None:
        # The user's own invocation (not a background child): refresh the
        # reminder scripts so they match this version.
        try:
            shell_hook.write_notice_scripts()
        except OSError:
            pass
    # Ask here, in the user's terminal — before the background child (which
    # has none) is spawned; the child then finds the answers already saved.
    shared = open_shared_folders(data_dir_path)
    interactive = _can_ask() and ready_file is None
    _share_folder(shared, serve_folder_path, yes=yes, interactive=interactive)
    for raw in mounts:
        path, label = parse_mount(raw)
        _share_folder(
            shared, resolve_folder(path), label, yes=yes, interactive=interactive
        )

    if not foreground and ready_file is None:
        _daemonize(
            daemon.serve_argv(
                host=host,
                port=port,
                blob_port=blob_port,
                data_dir=data_dir_path,
                mounts=(),
                serve_folder=serve_folder_path,
                iroh=iroh,
                metrics=metrics,
            ),
            data_dir_path,
        )
        message = shell_hook.auto_install(interactive=_can_ask())
        if message:
            click.echo("")
            click.echo(message)
        return

    try:
        daemon.acquire_pidfile(
            data_dir_path,
            {
                "host": host,
                "port": port,
                "blob_port": blob_port,
                "iroh": iroh,
                "background": ready_file is not None,
            },
        )
    except daemon.AlreadyRunningError as e:
        raise click.ClickException(_already_running_message(e, data_dir_path))
    try:
        asyncio.run(
            _serve_async(
                host,
                port,
                blob_port,
                data_dir_path,
                serve_folder_path,
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

    click.echo("")
    click.echo(
        click.style(_glyph("✓", "*"), fg="green")
        + click.style(
            f" sleap-connect worker running in the background (pid {info['pid']}).",
            bold=True,
        )
    )
    click.echo("  It keeps running after you close this terminal or log out,")
    click.echo("  until you stop it.")
    click.echo("")
    click.echo(f"  Check status:  sleap-rtc status{_data_dir_arg(data_dir)}")
    click.echo(f"  Stop it:       sleap-rtc serve --stop{_data_dir_arg(data_dir)}")
    click.echo(f"  Logs:          {shell_hook.display(daemon.log_path(data_dir))}")
    click.echo("")
    click.echo(f"  Pair a device: sleap-rtc pair{_data_dir_arg(data_dir)}")


def _data_dir_arg(data_dir: Path) -> str:
    """` --data-dir <dir>` for a non-default data dir, so a shown command
    targets the same worker; empty for the default."""
    if data_dir.expanduser().resolve() == DEFAULT_DATA_DIR.expanduser().resolve():
        return ""
    return f" --data-dir {shlex.quote(str(data_dir))}"


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
    serve_folder: Optional[Path] = None,
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

    file_manager = SharedFolderFileManager(
        open_shared_folders(data_dir), data_dir, serve_folder
    )
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

        if worker.requeued_job_ids:
            n = len(worker.requeued_job_ids)
            click.echo("")
            click.echo(
                f"Re-queued {n} job{'s' if n != 1 else ''} left waiting by the previous run."
            )

        _print_shared_folders(file_manager)

        n_devices = len(worker.trust_store.list_trusted())
        click.echo("")
        if n_devices:
            click.echo(f"Paired devices: {n_devices} (see 'sleap-rtc clients')")
        else:
            click.echo(
                click.style(
                    "No paired devices yet — run 'sleap-rtc pair' to get a pairing code.",
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


def _print_shared_folders(file_manager: SharedFolderFileManager) -> None:
    """The startup "what can paired devices see" section."""
    mounts = file_manager.get_mounts()
    click.echo("")
    if not mounts:
        click.echo(
            click.style(
                "No shared folders — paired devices can't browse files on this "
                "worker. Share one with 'sleap-rtc add-folder <folder>', or run "
                "'serve' in your data folder.",
                fg="yellow",
            )
        )
        return
    click.echo(
        click.style("Shared folders", bold=True)
        + " — paired devices can browse these, and their jobs can read and write inside them:"
    )
    width = max(len(m["label"]) for m in mounts)
    for m in mounts:
        click.echo(f"  {m['label']:<{width}}  {m['path']}")


def _ago(ts: Optional[float]) -> str:
    """How long ago Unix time `ts` was ("5 min ago"), or "never" if None."""
    if ts is None:
        return "never"
    secs = max(0, int(time.time() - ts))
    for unit, size in (("day", 86400), ("hr", 3600), ("min", 60)):
        if secs >= size:
            n = secs // size
            return f"{n} {unit}{'s' if n != 1 and unit == 'day' else ''} ago"
    return "just now"


_DATA_DIR_OPTION = click.option(
    "--data-dir",
    default=None,
    help=f"Worker data directory (same as 'serve'). [default: {DEFAULT_DATA_DIR}]",
)


def _platform_label(d) -> str:
    label = {"tauri": "desktop", "web": "browser"}.get(
        d.platform or "", d.platform or "—"
    )
    return f"{label} {d.app_version}" if d.app_version else label


def _describe(d) -> str:
    """One line for a device in prompts: name, app, last seen, short ID."""
    return (
        f'"{d.name or "(unnamed)"}" ({_platform_label(d)}, '
        f"last seen {_ago(d.last_seen_at)}, ID {d.node_id[:12]})"
    )


@click.group(name="clients", invoke_without_command=True)
@_DATA_DIR_OPTION
@click.pass_context
def clients(ctx: click.Context, data_dir: Optional[str]):
    """List the devices paired with this worker: who can use it.

    Each device's name (as its app reports it), platform and app version,
    when it paired and when it was last seen (connected or disconnected).
    Works whether or not 'sleap-rtc serve' is running. To unpair devices,
    see 'sleap-rtc clients remove --help'.
    """
    ctx.obj = {"data_dir": data_dir}
    if ctx.invoked_subcommand is not None:
        return
    data_dir = Path(data_dir or DEFAULT_DATA_DIR).expanduser()
    data_dir = Path(data_dir).expanduser()
    if not (devices_path(data_dir).exists() or trust_store_path(data_dir).exists()):
        click.echo("No paired devices yet. Pair one with 'sleap-rtc pair'.")
        return
    devices = open_trust_store(data_dir).list_devices()
    if not devices:
        click.echo("No paired devices yet. Pair one with 'sleap-rtc pair'.")
        return

    rows = [
        (
            d.name or "(unnamed)",
            _platform_label(d),
            _ago(d.paired_at) if d.paired_at else "—",
            _ago(d.last_seen_at),
            d.node_id[:12],
        )
        for d in devices
    ]
    headers = ("Device", "App", "Paired", "Last seen", "ID")
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = "  ".join(f"{{:<{w}}}" for w in widths)
    click.echo(click.style(line.format(*headers), bold=True))
    for r in rows:
        click.echo(line.format(*r))
    click.echo(f"\n{len(devices)} paired device{'s' if len(devices) != 1 else ''}.")


@clients.command(name="remove")
@click.argument("targets", nargs=-1)
@click.option(
    "--all",
    "remove_all",
    is_flag=True,
    help="Unpair every device and invalidate every unused pairing code.",
)
@click.option("--yes", "-y", is_flag=True, help="Don't ask for confirmation.")
@_DATA_DIR_OPTION
@click.pass_context
def clients_remove(
    ctx: click.Context,
    targets: tuple,
    remove_all: bool,
    yes: bool,
    data_dir: Optional[str],
):
    """Unpair devices, by name or ID prefix (as 'sleap-rtc clients' shows).

    A removed device can't use this worker any more: if it's connected
    right now, its next request or job update is refused and it's
    disconnected. Jobs it submitted keep running. It can pair again with a
    new code from 'sleap-rtc pair'. Works whether or not 'sleap-rtc serve'
    is running.

    \b
    Examples:
      sleap-rtc clients remove "Rig 4"
      sleap-rtc clients remove 7HV7KAOR
      sleap-rtc clients remove --all
    """
    data_dir_path = Path(
        data_dir or (ctx.obj or {}).get("data_dir") or DEFAULT_DATA_DIR
    ).expanduser()
    if remove_all == bool(targets):
        raise click.UsageError("Name the devices to remove, or pass --all.")
    store = open_trust_store(data_dir_path)
    devices = store.list_devices()

    if remove_all:
        from sleap_rtc.protocol_v1.pairing import PendingPairings

        pairings = PendingPairings(path=pairing_path(data_dir_path))
        n_codes = pairings.pending_count()
        if not devices and not n_codes:
            click.echo("No paired devices and no unused pairing codes.")
            return
        prompt = (
            f"Remove ALL {len(devices)} paired device{'s' if len(devices) != 1 else ''}"
            f" and invalidate {n_codes} unused pairing code{'s' if n_codes != 1 else ''}?"
        )
        if not yes and not click.confirm(prompt, default=False):
            click.echo("Nothing removed.")
            return
        n_devices = store.remove_all()
        n_codes = pairings.clear()
        click.echo(
            f"Removed {n_devices} device{'s' if n_devices != 1 else ''}; invalidated "
            f"{n_codes} pairing code{'s' if n_codes != 1 else ''}. "
            "Pair again with 'sleap-rtc pair'."
        )
        return

    chosen = []
    for target in targets:
        matches = _match_devices(devices, target)
        if not matches:
            raise click.ClickException(
                f"No paired device matches \"{target}\". See 'sleap-rtc clients'."
            )
        if len(matches) > 1:
            listing = "\n".join(f"  {_describe(d)}" for d in matches)
            raise click.ClickException(
                f'"{target}" matches {len(matches)} devices:\n{listing}\n'
                "Use the full name or a longer ID prefix."
            )
        if matches[0] not in chosen:
            chosen.append(matches[0])

    if not yes:
        listing = "\n".join(f"  {_describe(d)}" for d in chosen)
        if not click.confirm(
            f"Remove {len(chosen)} device(s)?\n{listing}\n", default=False
        ):
            click.echo("Nothing removed.")
            return
    store.remove([d.node_id for d in chosen])
    for d in chosen:
        click.echo(f'Removed "{d.name or "(unnamed)"}" ({d.node_id[:12]}).')


def _match_devices(devices, target: str) -> list:
    """Devices whose name is `target` (any case), else whose ID starts with it."""
    by_name = [d for d in devices if d.name and d.name.casefold() == target.casefold()]
    if by_name:
        return by_name
    return [d for d in devices if d.node_id.startswith(target)]


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

    if devices_path(data_dir).exists() or trust_store_path(data_dir).exists():
        n = len(open_trust_store(data_dir).list_trusted())
        click.echo(f"  Paired clients: {n} (see 'sleap-rtc clients')")
    else:
        click.echo("  Paired clients: 0 (no trust store yet)")

    return all_ok


def _duration(secs: float) -> str:
    """A short duration: "3h 12m", "2d 4h", "45s"."""
    secs = int(max(0, secs))
    days, rem = divmod(secs, 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {mins}m"
    if mins:
        return f"{mins}m"
    return f"{secs}s"


async def _job_counts(store_path: Path) -> dict:
    from sleap_rtc.jobs.store import JobStore

    async with JobStore(store_path) as store:
        jobs = await store.list_jobs()
    counts: dict = {}
    for record in jobs:
        counts[record.state] = counts.get(record.state, 0) + 1
    return counts


def _print_worker_status(data_dir: Path, running: daemon.ServeProcess) -> None:
    details = daemon.read_details(data_dir)
    mode = "in the background" if details.get("background", True) else "in a terminal"
    click.echo(
        click.style(_glyph("●", "*"), fg="green")
        + click.style(" sleap-connect worker: running ", bold=True)
        + f"{mode} (pid {running.pid}, up {_duration(time.time() - running.started_at)})"
    )

    rows = []
    if "port" in details:
        address = f"ws://{details.get('host', '0.0.0.0')}:{details['port']}"
        if details.get("iroh"):
            address += " · iroh"
        rows.append(("Address", address))
    rows.append(("Data dir", shell_hook.display(data_dir)))

    shared = open_shared_folders(data_dir).list()
    rows.append(
        ("Shared folders", ", ".join(f.path for f in shared) if shared else "none")
    )

    has_store = devices_path(data_dir).exists() or trust_store_path(data_dir).exists()
    devices = open_trust_store(data_dir).list_devices() if has_store else []
    names = ", ".join(d.name or "(unnamed)" for d in devices)
    rows.append(("Paired devices", f"{len(devices)} ({names})" if devices else "0"))

    store_path = data_dir / "jobs.sqlite"
    counts = asyncio.run(_job_counts(store_path)) if store_path.exists() else {}
    rows.append(
        (
            "Jobs",
            f"{counts.get('running', 0)} running, {counts.get('queued', 0)} queued",
        )
    )
    if details.get("background", True):
        rows.append(("Log", shell_hook.display(daemon.log_path(data_dir))))

    width = max(len(label) for label, _ in rows) + 1
    for label, value in rows:
        click.echo(f"  {label + ':':<{width}} {value}")
    click.echo("")
    click.echo(
        f"Stop: sleap-rtc serve --stop{_data_dir_arg(data_dir)} · "
        f"pair a device: sleap-rtc pair{_data_dir_arg(data_dir)}"
    )


@click.command(name="status")
@_DATA_DIR_OPTION
def status(data_dir: Optional[str]):
    """Show whether a sleap-connect worker is running, and its details.

    Lists every running worker (normally just one). Exits with status 1
    when none is running.
    """
    if data_dir is not None:
        path = _data_dir(data_dir)
        found = [(path, r)] if (r := daemon.read_running(path)) else []
    else:
        found = daemon.list_running()
        # A worker started by an older version isn't in the registry.
        default = DEFAULT_DATA_DIR.expanduser()
        if not any(d.resolve() == default.resolve() for d, _ in found):
            if (r := daemon.read_running(default)) is not None:
                found.append((default, r))

    if not found:
        click.echo(
            click.style(_glyph("○", "-"), fg="yellow")
            + click.style(" sleap-connect worker: not running", bold=True)
        )
        click.echo(
            "  Start one in your data folder:  cd /path/to/data && sleap-rtc serve"
        )
        raise SystemExit(1)

    for i, (path, running) in enumerate(found):
        if i:
            click.echo("")
        _print_worker_status(path, running)


@click.group(name="shell-hook", invoke_without_command=True)
@click.pass_context
def shell_hook_group(ctx: click.Context):
    """Manage the new-terminal reminder shown while a worker is running.

    The first 'sleap-rtc serve' adds a small block to your shell's startup
    file (~/.zshrc, ~/.bashrc, fish config.fish, or the PowerShell profile)
    so new terminals show a one-line notice while a worker runs. Set
    SLEAP_RTC_NO_SHELL_HOOK=1 to stop 'serve' from adding it.
    """
    if ctx.invoked_subcommand is None:
        ctx.invoke(shell_hook_status)


@shell_hook_group.command(name="install")
@click.option(
    "--shell",
    type=click.Choice(shell_hook.SHELLS),
    default=None,
    help="Shell to add the reminder for. [default: your current shell]",
)
def shell_hook_install(shell: Optional[str]):
    """Add the new-terminal reminder to your shell's startup file."""
    shell = shell or shell_hook.detect_shell()
    if shell is None:
        raise click.ClickException(
            "Couldn't tell which shell you use. Pass --shell "
            f"({', '.join(shell_hook.SHELLS)})."
        )
    result = shell_hook.install(shell)
    for path in result.added:
        click.echo(f"Added the reminder to {shell_hook.display(path)}.")
    for path in result.already:
        click.echo(f"{shell_hook.display(path)} already has the reminder.")
    if result.added:
        click.echo("It shows up in terminals you open from now on.")


@shell_hook_group.command(name="uninstall")
def shell_hook_uninstall():
    """Remove the new-terminal reminder (and stop 'serve' re-adding it)."""
    changed = shell_hook.uninstall()
    for path in changed:
        click.echo(f"Removed the reminder from {shell_hook.display(path)}.")
    if not changed:
        click.echo("No startup file had the reminder.")
    click.echo(
        "'sleap-rtc serve' won't add it again ('sleap-rtc shell-hook install' to undo)."
    )


@shell_hook_group.command(name="status")
def shell_hook_status():
    """Show which startup files have the new-terminal reminder."""
    files = shell_hook.installed_files()
    if files:
        for path in files:
            click.echo(f"Reminder installed in {shell_hook.display(path)}.")
    else:
        click.echo(
            "The new-terminal reminder isn't installed ('sleap-rtc shell-hook install')."
        )


def _data_dir(data_dir: Optional[str]) -> Path:
    return Path(data_dir or DEFAULT_DATA_DIR).expanduser()


@click.command(name="add-folder")
@click.argument("folder", default=".", required=False)
@click.option(
    "--label", default=None, help="Display name (default: the folder's name)."
)
@click.option("--yes", "-y", is_flag=True, help="Don't ask for confirmation.")
@_DATA_DIR_OPTION
def add_folder(folder: str, label: Optional[str], yes: bool, data_dir: Optional[str]):
    """Share a folder with the devices paired to this worker.

    They can browse it, and their training/inference jobs can read and write
    inside it. FOLDER may be relative (default: the current folder). A
    running worker picks it up right away.
    """
    path = resolve_folder(folder)
    shared = open_shared_folders(_data_dir(data_dir))
    already = shared.find(path)
    if already is not None:
        click.echo(f'Already shared (as part of "{already.label}": {already.path}).')
        return
    if not _share_folder(shared, path, label, yes=yes, interactive=_can_ask()):
        if _can_ask():
            click.echo("Not shared.")
        else:
            raise click.ClickException(
                "Not shared: pass --yes to share without a prompt."
            )
        return
    added = shared.find(path)
    click.echo(f'Shared {added.path} as "{added.label}".')


@click.command(name="remove-folder")
@click.argument("folder")
@_DATA_DIR_OPTION
def remove_folder(folder: str, data_dir: Optional[str]):
    """Stop sharing a folder (by path or label, as 'sleap-rtc folders' shows).

    Paired devices can no longer browse it; a running worker applies this
    right away.
    """
    removed = open_shared_folders(_data_dir(data_dir)).remove(folder)
    if removed is None:
        raise click.ClickException(
            f"No shared folder matches \"{folder}\". See 'sleap-rtc folders'."
        )
    click.echo(f'Stopped sharing "{removed.label}" ({removed.path}).')


@click.command(name="folders")
@_DATA_DIR_OPTION
def folders(data_dir: Optional[str]):
    """List the folders shared with the devices paired to this worker."""
    shared = open_shared_folders(_data_dir(data_dir)).list()
    if not shared:
        click.echo(
            "No shared folders. Run 'sleap-rtc serve' in your data folder, or "
            "'sleap-rtc add-folder <folder>'."
        )
        return
    rows = [
        (
            f.label,
            f.path + ("" if Path(f.path).is_dir() else "  (missing)"),
            _ago(f.added_at),
        )
        for f in shared
    ]
    headers = ("Label", "Path", "Shared")
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = "  ".join(f"{{:<{w}}}" for w in widths)
    click.echo(click.style(line.format(*headers), bold=True))
    for r in rows:
        click.echo(line.format(*r))
