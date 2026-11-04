"""`sleap-connect jobs` — the worker's jobs, from the worker machine itself.

Reads (`jobs`, `jobs queue`, `jobs show`, `jobs logs`) go straight to the
job store (`jobs.sqlite`) and the per-job log files under --data-dir, so
they work whether or not the worker is running. `jobs stop` and `jobs
cancel` need the running worker (it owns the job processes and the queue):
they connect to it as a local client, authenticating with the worker's
own identity key, which the worker trusts without pairing (see
`AuthMethods._is_self`).
"""

import asyncio
import json
import secrets
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import rich_click as click

from sleap_connect.jobs.spec import TrackJobSpec, TrainJobSpec
from sleap_connect.jobs.store import TERMINAL_STATES, JobRecord, JobStore
from sleap_connect.protocol_v1 import daemon
from sleap_connect.protocol_v1.cli import (
    _DATA_DIR_OPTION,
    _ago,
    _data_dir,
    _data_dir_arg,
    _duration,
)
from sleap_connect.protocol_v1.runner import (
    DEFAULT_PORT,
    devices_path,
    identity_path,
    job_log_dir,
    job_store_path,
    open_trust_store,
    trust_store_path,
)

# How many lines `jobs logs` shows by default, and how many jobs `jobs` lists.
DEFAULT_LOG_LINES = 50
DEFAULT_LIST_LIMIT = 20

# How often `jobs logs -f` checks the log file and the job's state.
FOLLOW_POLL_SECS = 0.5

# How long `jobs cancel` waits for the worker to record the job as ended
# before reporting "requested" instead of "canceled".
CANCEL_SETTLE_SECS = 3.0

# How long to wait for the worker to answer before giving up.
WORKER_TIMEOUT_SECS = 10.0


# --- reading the job store ---------------------------------------------------


async def _read_jobs(data_dir: Path) -> List[JobRecord]:
    async with JobStore(job_store_path(data_dir)) as store:
        return await store.list_jobs()


def _load_jobs(data_dir: Path) -> List[JobRecord]:
    """Every job the worker has recorded, newest first ([] if none yet)."""
    if not job_store_path(data_dir).exists():
        return []
    jobs = asyncio.run(_read_jobs(data_dir))
    return sorted(jobs, key=lambda r: r.created_at, reverse=True)


def _load_job(data_dir: Path, job_id: str) -> Optional[JobRecord]:
    for record in _load_jobs(data_dir):
        if record.job_id == job_id:
            return record
    return None


def _find_job(jobs: List[JobRecord], target: str) -> JobRecord:
    """The job whose ID is `target`, or starts with it (unique prefix)."""
    exact = [r for r in jobs if r.job_id == target]
    if exact:
        return exact[0]
    matches = [r for r in jobs if r.job_id.startswith(target)]
    if not matches:
        raise click.ClickException(
            f"No job matches \"{target}\". See 'sleap-connect jobs'."
        )
    if len(matches) > 1:
        ids = ", ".join(r.job_id for r in matches[:5])
        raise click.ClickException(
            f'"{target}" matches {len(matches)} jobs ({ids}'
            f"{', ...' if len(matches) > 5 else ''}). Use a longer ID prefix."
        )
    return matches[0]


def _device_names(data_dir: Path) -> Dict[str, str]:
    """Paired devices' names by node_id."""
    if not (devices_path(data_dir).exists() or trust_store_path(data_dir).exists()):
        return {}
    return {
        d.node_id: d.name or "(unnamed)"
        for d in open_trust_store(data_dir).list_devices()
    }


def _device(record: JobRecord, names: Dict[str, str]) -> str:
    if record.submitted_by is None:
        return "—"
    return names.get(record.submitted_by, f"unpaired ({record.submitted_by[:8]})")


def _kind(record: JobRecord) -> str:
    return "train" if isinstance(record.spec, TrainJobSpec) else "track"


def _labels_path(record: JobRecord) -> Optional[str]:
    if isinstance(record.spec, TrainJobSpec):
        return record.spec.labels_path
    return record.spec.data_path


def _model(record: JobRecord) -> str:
    """What the job trains or runs: model types, or model folder names."""
    spec = record.spec
    if isinstance(spec, TrainJobSpec):
        return ", ".join(spec.model_types) or "—"
    return ", ".join(Path(p).name for p in spec.model_paths) or "—"


def _model_dir(record: JobRecord) -> Optional[str]:
    return (record.result or {}).get("model_dir")


def _started(record: JobRecord) -> str:
    if record.process_started_at is None:
        return "—"
    return _ago(record.process_started_at)


def _log_path(data_dir: Path, record: JobRecord) -> Path:
    if record.log_path:
        return Path(record.log_path)
    return job_log_dir(data_dir) / f"{record.job_id}.log"


def _predictions_path(data_dir: Path, record: JobRecord) -> Optional[str]:
    """Where a track job's predictions went (or go), as best we can tell."""
    from sleap_connect.protocol_v1.job_methods import JobMethods

    spec = record.spec
    if not isinstance(spec, TrackJobSpec):
        return None
    captured = JobMethods._captured_output_path_from_log(_log_path(data_dir, record))
    return captured or spec.output_path


def _job_summary(record: JobRecord, names: Dict[str, str]) -> dict:
    """The --json shape of one job in `jobs` / `jobs queue`."""
    return {
        "job_id": record.job_id,
        "kind": _kind(record),
        "state": record.state,
        "model_types": list(getattr(record.spec, "model_types", None) or []),
        "model_paths": list(getattr(record.spec, "model_paths", None) or []),
        "model_dir": _model_dir(record),
        "labels_path": _labels_path(record),
        "submitted_by": record.submitted_by,
        "device": names.get(record.submitted_by) if record.submitted_by else None,
        "created_at": record.created_at,
        "started_at": record.process_started_at,
        "updated_at": record.updated_at,
        "error": record.error,
        "run": getattr(record.spec, "run", None),
    }


def _echo_table(headers: Tuple[str, ...], rows: List[Tuple[str, ...]]) -> None:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = "  ".join(f"{{:<{w}}}" for w in widths)
    click.echo(click.style(line.format(*headers).rstrip(), bold=True))
    for r in rows:
        click.echo(line.format(*r).rstrip())


_STATE_COLORS = {
    "running": "cyan",
    "queued": "yellow",
    "completed": "green",
    "failed": "red",
    "canceled": "bright_black",
}


def _state(state: str, width: int = 0) -> str:
    """`state`, colored, padded to `width` (before styling, so columns align)."""
    return click.style(f"{state:<{width}}", fg=_STATE_COLORS.get(state))


def _not_running_note(data_dir: Path) -> Optional[str]:
    if daemon.read_running(data_dir) is not None:
        return None
    return (
        "The worker isn't running; queued jobs start when it does "
        f"('sleap-connect serve{_data_dir_arg(data_dir)}')."
    )


# --- `jobs` ------------------------------------------------------------------


@click.group(name="jobs", invoke_without_command=True)
@click.option(
    "-n",
    "--limit",
    default=DEFAULT_LIST_LIMIT,
    show_default=True,
    help="Show at most this many jobs (0 for all).",
)
@click.option("--json", "as_json", is_flag=True, help="Print the jobs as JSON.")
@_DATA_DIR_OPTION
@click.pass_context
def jobs(ctx: click.Context, limit: int, as_json: bool, data_dir: Optional[str]):
    """List this worker's recent jobs, newest first.

    One row per model trained or run: a multi-model training run shows one job
    per model. Works whether or not the worker is running. See one job with
    'sleap-connect jobs show ID', its log with 'sleap-connect jobs logs ID'.

    Examples:
        sleap-connect jobs
        sleap-connect jobs queue
        sleap-connect jobs show job_Xy12
        sleap-connect jobs logs job_Xy12 -f
        sleap-connect jobs stop job_Xy12
    """
    ctx.obj = {"data_dir": data_dir}
    if ctx.invoked_subcommand is not None:
        return
    data_dir_path = _worker_data_dir(_data_dir(data_dir))
    records = _load_jobs(data_dir_path)
    shown = records[:limit] if limit > 0 else records
    names = _device_names(data_dir_path)

    if as_json:
        click.echo(json.dumps([_job_summary(r, names) for r in shown], indent=2))
        return
    if not records:
        click.echo("No jobs yet. Jobs submitted from SLEAP's Connect window show here.")
        return

    _print_job_table(shown, names)
    if len(shown) < len(records):
        click.echo(
            f"\nShowing the {len(shown)} newest of {len(records)} jobs "
            "(-n 0 to show all)."
        )


def _print_job_table(records: List[JobRecord], names: Dict[str, str]) -> None:
    width = max(len(r.state) for r in records)
    rows = [
        (
            r.job_id,
            _kind(r),
            _model(r),
            _state(r.state, width),
            _device(r, names),
            _started(r),
            Path(_labels_path(r) or "—").name,
        )
        for r in records
    ]
    # Size the state column by its plain text, not the styled one.
    headers = ("ID", "Kind", "Model", "State", "Device", "Started", "Labels")
    widths = [
        max(len(h), *((width if i == 3 else len(row[i])) for row in rows))
        for i, h in enumerate(headers)
    ]
    line = "  ".join(f"{{:<{w}}}" for w in widths)
    click.echo(click.style(line.format(*headers).rstrip(), bold=True))
    for row in rows:
        click.echo(line.format(*row).rstrip())


def _parent_data_dir(ctx: click.Context, data_dir: Optional[str]) -> Path:
    """--data-dir given to the subcommand, else to `jobs` itself."""
    return _worker_data_dir(_data_dir(data_dir or (ctx.obj or {}).get("data_dir")))


def _worker_data_dir(data_dir: Path) -> Path:
    """`data_dir`, if a worker has been set up there.

    The jobs commands read the worker's own files, so they only work on the
    worker machine. Run anywhere else, say so rather than "No jobs yet".
    """
    if not identity_path(data_dir).exists():
        raise click.ClickException(
            f"No sleap-connect worker is set up on this machine (nothing in "
            f"{data_dir}). Run 'sleap-connect jobs' on the worker machine (machine "
            "where 'sleap-connect serve' was run) or manage jobs from SLEAP's "
            "Connect window."
        )
    return data_dir


# --- `jobs queue` ------------------------------------------------------------


def _queue(
    records: List[JobRecord], positions: Optional[Dict[str, int]] = None
) -> List[JobRecord]:
    """Running jobs, then waiting ones in the order they'll run.

    `positions` is the running worker's own line (job_id -> place); without
    it, waiting jobs run in submission order, as the worker queues them.
    """
    positions = positions or {}
    running = [r for r in records if r.state == "running"]
    waiting = [r for r in records if r.state == "queued"]
    return sorted(running, key=lambda r: r.created_at) + sorted(
        waiting, key=lambda r: (positions.get(r.job_id, float("inf")), r.created_at)
    )


def _worker_queue_positions(
    data_dir: Path,
) -> Tuple[Optional[Dict[str, int]], Optional[str]]:
    """The running worker's queue positions right now, and why not if not.

    (None, None) when no worker is running: the job store is all there is.
    """
    if daemon.read_running(data_dir) is None:
        return None, None
    try:
        (reply,) = asyncio.run(_call_worker(data_dir, [("jobs.list", {})]))
    except click.ClickException as e:
        return None, e.message
    if reply.error:
        return None, reply.error.get("msg")
    return {
        j["job_id"]: j["queue_position"]
        for j in reply.result.get("jobs", [])
        if j.get("queue_position") is not None
    }, None


@jobs.command(name="queue")
@click.option("--json", "as_json", is_flag=True, help="Print the queue as JSON.")
@_DATA_DIR_OPTION
@click.pass_context
def jobs_queue(ctx: click.Context, as_json: bool, data_dir: Optional[str]):
    """Show the running job and the jobs waiting behind it, in run order.

    The worker runs one job at a time; the rest wait their turn. When the worker
    is running, the order is its queue at the moment you ask. Cancel waiting jobs
    with 'sleap-connect jobs cancel ID' or 'sleap-connect jobs cancel --queued'.
    """
    data_dir_path = _parent_data_dir(ctx, data_dir)
    positions, problem = _worker_queue_positions(data_dir_path)
    line_up = _queue(_load_jobs(data_dir_path), positions)
    names = _device_names(data_dir_path)

    if as_json:
        out = []
        position = 0
        for r in line_up:
            if r.state == "queued":
                position += 1
            out.append(
                {
                    **_job_summary(r, names),
                    "queue_position": position if r.state == "queued" else None,
                }
            )
        click.echo(json.dumps(out, indent=2))
        return
    if not line_up:
        click.echo("Nothing running or waiting.")
        return

    rows = []
    position = 0
    now = time.time()
    for r in line_up:
        if r.state == "running":
            place, since = (
                "▶",
                f"running {_duration(now - (r.process_started_at or r.created_at))}",
            )
        else:
            position += 1
            place, since = str(position), f"waiting {_duration(now - r.created_at)}"
        rows.append(
            (
                place,
                r.job_id,
                _kind(r),
                _model(r),
                _device(r, names),
                since,
                Path(_labels_path(r) or "—").name,
            )
        )
    _echo_table(("#", "ID", "Kind", "Model", "Device", "Status", "Labels"), rows)
    note = _not_running_note(data_dir_path)
    if problem and position:
        note = (
            f"Couldn't ask the worker for its queue ({problem}); waiting jobs are "
            "shown in the order they were submitted."
        )
    if note and position:
        click.echo("")
        click.echo(click.style(note, fg="yellow"))


# --- `jobs show` -------------------------------------------------------------


@jobs.command(name="show")
@click.argument("job_id")
@click.option("--json", "as_json", is_flag=True, help="Print the job as JSON.")
@_DATA_DIR_OPTION
@click.pass_context
def jobs_show(ctx: click.Context, job_id: str, as_json: bool, data_dir: Optional[str]):
    """Show one job: what it ran, where its output is, timing, and any error.

    JOB_ID may be a unique prefix of the ID 'sleap-connect jobs' shows.
    """
    data_dir_path = _parent_data_dir(ctx, data_dir)
    record = _find_job(_load_jobs(data_dir_path), job_id)
    names = _device_names(data_dir_path)
    log_path = _log_path(data_dir_path, record)
    predictions = _predictions_path(data_dir_path, record)

    if as_json:
        spec = record.spec.to_dict()
        spec.pop("labels_content", None)
        click.echo(
            json.dumps(
                {
                    **_job_summary(record, names),
                    "predictions_path": predictions,
                    "log_path": str(log_path),
                    "result": record.result,
                    "spec": spec,
                },
                indent=2,
            )
        )
        return

    rows: List[Tuple[str, str]] = [
        ("Kind", _kind(record)),
        ("Model", _model(record)),
        ("State", _state(record.state)),
    ]
    if record.error:
        rows.append(("Error", click.style(record.error, fg="red")))
    rows.append(("Device", _device(record, names)))
    rows.append(("Labels", _labels_path(record) or "—"))
    run = getattr(record.spec, "run", None) or {}
    if run.get("name") or run.get("id"):
        rows.append(("Run", str(run.get("name") or run.get("id"))))

    rows.append(
        ("Submitted", f"{_ago(record.created_at)} ({_when(record.created_at)})")
    )
    if record.process_started_at is not None:
        rows.append(
            (
                "Started",
                f"{_ago(record.process_started_at)} ({_when(record.process_started_at)})",
            )
        )
        end = record.updated_at if record.state in TERMINAL_STATES else time.time()
        label = "Took" if record.state in TERMINAL_STATES else "Running for"
        rows.append((label, _duration(end - record.process_started_at)))
    if record.state in TERMINAL_STATES:
        rows.append(
            ("Ended", f"{_ago(record.updated_at)} ({_when(record.updated_at)})")
        )

    if isinstance(record.spec, TrainJobSpec):
        model_dir = _model_dir(record)
        if model_dir:
            rows.append(("Model folder", model_dir))
        elif record.spec.ckpt_dir:
            rows.append(("Models in", record.spec.ckpt_dir))
    elif predictions:
        rows.append(("Predictions", predictions))
    for chained in (record.result or {}).get("chained_job_ids") or []:
        rows.append(("Inference job", chained))
    if (record.result or {}).get("chain_error"):
        rows.append(("Inference", click.style(record.result["chain_error"], fg="red")))
    rows.append(("Log", str(log_path) + ("" if log_path.exists() else "  (missing)")))

    click.echo(click.style(record.job_id, bold=True))
    width = max(len(label) for label, _ in rows) + 1
    for label, value in rows:
        click.echo(f"  {label + ':':<{width}} {value}")


def _when(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


# --- `jobs logs` -------------------------------------------------------------


def _visible(raw: bytes) -> str:
    """One log line as a terminal shows it (a progress bar's last state)."""
    from sleap_connect.protocol_v1.job_methods import _visible_text

    return _visible_text(raw.decode("utf-8", errors="replace"))


def _tail(path: Path, n: int) -> Tuple[List[str], int]:
    """The last `n` lines of `path` (all if n <= 0), and its size."""
    data = path.read_bytes()
    lines = [_visible(raw) for raw in data.splitlines()]
    return (lines[-n:] if n > 0 else lines), len(data)


@jobs.command(name="logs")
@click.argument("job_id")
@click.option(
    "-n",
    "--lines",
    default=DEFAULT_LOG_LINES,
    show_default=True,
    help="How many lines from the end to show (0 for the whole log).",
)
@click.option(
    "-f",
    "--follow",
    is_flag=True,
    help="Keep printing new lines until the job ends (Ctrl-C to stop watching; "
    "the job keeps running).",
)
@_DATA_DIR_OPTION
@click.pass_context
def jobs_logs(
    ctx: click.Context, job_id: str, lines: int, follow: bool, data_dir: Optional[str]
):
    """Show a job's log (sleap-nn's output).

    JOB_ID may be a unique prefix of the ID 'sleap-connect jobs' shows.

    Examples:
        sleap-connect jobs logs job_Xy12
        sleap-connect jobs logs job_Xy12 -n 0
        sleap-connect jobs logs job_Xy12 -f
    """
    data_dir_path = _parent_data_dir(ctx, data_dir)
    record = _find_job(_load_jobs(data_dir_path), job_id)
    path = _log_path(data_dir_path, record)

    offset = 0
    if path.exists():
        shown, offset = _tail(path, lines)
        for line in shown:
            click.echo(line)
    elif record.state == "queued":
        click.echo(f"Job {record.job_id} hasn't started yet; no log so far.")
    elif not follow:
        raise click.ClickException(f"No log file for job {record.job_id} ({path}).")

    if not follow:
        return
    try:
        _follow(data_dir_path, record.job_id, path, offset)
    except KeyboardInterrupt:
        pass


def _follow(data_dir: Path, job_id: str, path: Path, offset: int) -> None:
    """Print what's appended to `path` until the job is terminal."""
    pending = b""
    last_state_check = 0.0
    state = None
    while True:
        if path.exists():
            with open(path, "rb") as f:
                f.seek(offset)
                chunk = f.read()
            offset += len(chunk)
            pending += chunk
            *complete, pending = pending.split(b"\n")
            for raw in complete:
                click.echo(_visible(raw))
        if state in TERMINAL_STATES:
            if pending:
                click.echo(_visible(pending))
            record = _load_job(data_dir, job_id)
            click.echo(click.style(f"\nJob {job_id} {record.state}.", bold=True))
            return
        now = time.monotonic()
        if now - last_state_check >= 2 * FOLLOW_POLL_SECS:
            last_state_check = now
            record = _load_job(data_dir, job_id)
            # Read once more after seeing it end: its last lines may have
            # landed since this pass's read.
            state = record.state if record is not None else "canceled"
            if state in TERMINAL_STATES:
                continue
        time.sleep(FOLLOW_POLL_SECS)


# --- talking to the running worker (stop / cancel) ---------------------------


class WorkerNotRunning(click.ClickException):
    pass


def _worker_address(data_dir: Path) -> str:
    """The running worker's WS address, from its pidfile."""
    if daemon.read_running(data_dir) is None:
        raise WorkerNotRunning(
            f"No worker is running against {data_dir}, so there's nothing to stop "
            "or cancel: jobs only run while it does. Start it with "
            f"'sleap-connect serve{_data_dir_arg(data_dir)}'."
        )
    details = daemon.read_details(data_dir)
    host = details.get("host") or "127.0.0.1"
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    if ":" in host:
        host = f"[{host}]"
    return f"ws://{host}:{details.get('port') or DEFAULT_PORT}"


async def _call_worker(data_dir: Path, calls: List[Tuple[str, dict]]) -> list:
    """Make each (method, params) call on the running worker, in order, over
    one connection. Returns each call's `Res`.
    """
    import websockets

    from sleap_connect.protocol_v1.envelope import Hello, Req, Res, parse_envelope
    from sleap_connect.protocol_v1.identity import WorkerIdentity
    from sleap_connect.protocol_v1.runner import worker_version

    identity = WorkerIdentity(identity_path(data_dir))
    address = _worker_address(data_dir)
    next_id = 0

    async def call(ws, method: str, params: dict) -> Res:
        nonlocal next_id
        next_id += 1
        await ws.send(Req(id=next_id, method=method, params=params).to_json())
        while True:
            reply = parse_envelope(await ws.recv())
            if isinstance(reply, Res) and reply.id == next_id:
                return reply

    async def run() -> list:
        try:
            ws = await websockets.connect(address, max_size=None)
        except OSError as e:
            raise click.ClickException(
                f"Couldn't reach the worker at {address}: {e}"
            ) from None
        try:
            await ws.send(
                Hello(
                    proto={"min": 1, "max": 1},
                    agent={
                        "name": "sleap-connect-cli",
                        "version": worker_version(),
                        "platform": sys.platform,
                    },
                    node_id=identity.node_id,
                    nonce=secrets.token_urlsafe(16),
                ).to_json()
            )
            server_hello = parse_envelope(await ws.recv())
            if not isinstance(server_hello, Hello):
                raise click.ClickException(f"Unexpected reply from {address}.")
            if server_hello.node_id != identity.node_id:
                raise click.ClickException(
                    f"{address} is a different worker than the one in {data_dir}."
                )
            proved = await call(
                ws, "auth.prove", {"sig": identity.sign(server_hello.nonce)}
            )
            if proved.error:
                raise click.ClickException(
                    "The worker refused this command "
                    f"({proved.error.get('msg')}). If it's an older version, "
                    "restart it: 'sleap-connect serve --stop', then "
                    "'sleap-connect serve'."
                )
            return [await call(ws, method, params) for method, params in calls]
        finally:
            await ws.close()

    try:
        return await asyncio.wait_for(run(), WORKER_TIMEOUT_SECS)
    except asyncio.TimeoutError:
        raise click.ClickException(
            f"The worker at {address} didn't answer within {WORKER_TIMEOUT_SECS:.0f}s."
        ) from None


def _settle(data_dir: Path, job_ids: List[str]) -> Dict[str, str]:
    """Each job's state once it ends, waiting up to CANCEL_SETTLE_SECS."""
    deadline = time.monotonic() + CANCEL_SETTLE_SECS
    while True:
        states = {
            r.job_id: r.state for r in _load_jobs(data_dir) if r.job_id in job_ids
        }
        if all(s in TERMINAL_STATES for s in states.values()):
            return states
        if time.monotonic() >= deadline:
            return states
        time.sleep(0.2)


def _end_jobs(data_dir: Path, records: List[JobRecord], mode: str) -> None:
    """Stop/cancel `records` through the worker and report each outcome."""
    active = []
    for r in records:
        if r.state in TERMINAL_STATES:
            click.echo(f"Job {r.job_id} already {r.state}.")
        else:
            active.append(r)
    if not active:
        return
    ids = [r.job_id for r in active]
    replies = asyncio.run(
        _call_worker(
            data_dir, [("jobs.cancel", {"job_id": i, "mode": mode}) for i in ids]
        )
    )
    errors = {
        i: reply.error.get("msg") if reply.error else None
        for i, reply in zip(ids, replies)
    }
    states = _settle(data_dir, [i for i in ids if errors.get(i) is None])
    failed = False
    for r in active:
        if errors.get(r.job_id):
            failed = True
            click.echo(
                click.style(f"Job {r.job_id}: {errors[r.job_id]}", fg="red"), err=True
            )
            continue
        state = states.get(r.job_id)
        if state in TERMINAL_STATES:
            click.echo(f"Job {r.job_id} {state}.")
        elif mode == "stop":
            click.echo(
                f"Stopping job {r.job_id}: training ends after the current step and "
                "keeps the best model so far. Watch it with "
                f"'sleap-connect jobs logs {r.job_id} -f{_data_dir_arg(data_dir)}'."
            )
        else:
            click.echo(f"Canceling job {r.job_id}; it ends in a moment.")
    if failed:
        raise SystemExit(1)


# --- `jobs stop` / `jobs cancel` ---------------------------------------------


@jobs.command(name="stop")
@click.argument("job_id")
@_DATA_DIR_OPTION
@click.pass_context
def jobs_stop(ctx: click.Context, job_id: str, data_dir: Optional[str]):
    """Stop a training job early, keeping the best model trained so far.

    Like SLEAP's "Stop early" button: training finishes cleanly and the job
    completes, so inference after training still runs. A job that
    hasn't started is just canceled. To end a job and discard it, use 'sleap-connect
    jobs cancel'. Needs the worker running. JOB_ID may be a unique prefix.

    Example:
        sleap-connect jobs stop job_Xy12
    """
    data_dir_path = _parent_data_dir(ctx, data_dir)
    record = _find_job(_load_jobs(data_dir_path), job_id)
    _end_jobs(data_dir_path, [record], "stop")


@jobs.command(name="cancel")
@click.argument("job_ids", nargs=-1)
@click.option(
    "--queued",
    is_flag=True,
    help="Cancel every job waiting in the queue (not the running one).",
)
@click.option("--yes", "-y", is_flag=True, help="Don't ask for confirmation.")
@_DATA_DIR_OPTION
@click.pass_context
def jobs_cancel(
    ctx: click.Context,
    job_ids: tuple,
    queued: bool,
    yes: bool,
    data_dir: Optional[str],
):
    """Cancel jobs: end a running one now, or take waiting ones off the queue.

    A canceled training job keeps nothing; to keep the best model so far, use
    'sleap-connect jobs stop'. Needs the worker running. JOB_IDS may be unique
    prefixes.

    Examples:
        sleap-connect jobs cancel job_Xy12
        sleap-connect jobs cancel --queued
    """
    data_dir_path = _parent_data_dir(ctx, data_dir)
    if queued == bool(job_ids):
        raise click.UsageError("Name the jobs to cancel, or pass --queued.")
    records = _load_jobs(data_dir_path)

    if queued:
        chosen = [r for r in _queue(records) if r.state == "queued"]
        if not chosen:
            click.echo("No jobs are waiting in the queue.")
            return
        _worker_address(data_dir_path)  # fail before asking if it's not running
        if not yes:
            listing = "\n".join(
                f"  {r.job_id}  {_kind(r)}  {_model(r)}  "
                f"{Path(_labels_path(r) or '—').name}"
                for r in chosen
            )
            n = len(chosen)
            if not click.confirm(
                f"Cancel {n} waiting job{'s' if n != 1 else ''}?\n{listing}\n",
                default=False,
            ):
                click.echo("Nothing canceled.")
                return
    else:
        chosen = []
        for target in job_ids:
            record = _find_job(records, target)
            if record not in chosen:
                chosen.append(record)
    _end_jobs(data_dir_path, chosen, "cancel")
