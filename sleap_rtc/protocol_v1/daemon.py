"""Background-process support for `sleap-rtc serve` (`--daemonize` / `--stop`).

`serve --daemonize` re-executes `serve` as a fresh, fully detached child
process rather than `os.fork()`-ing the current one: forking a process that
may already hold threads (or, later, an event loop) is unsafe, and `fork`
doesn't exist on Windows at all. A clean re-exec works the same way on every
platform.

The parent then *blocks until the child is actually listening* — the child
writes a small "ready" file once `start_worker_server` has returned — so a
`serve --daemonize` that returns 0 means the worker really is up, and a
child that dies during startup (port already in use, bad --data-dir, ...)
surfaces as a non-zero exit plus the tail of its log, not as a silent
"started" followed by nothing listening.

Every `serve` (foreground or daemonized) also holds a pidfile under its
--data-dir, so two workers can never run against the same data dir at once —
they would share one SQLite job store and both try to reattach the same
running jobs on startup. The pidfile records the process's own start time
alongside its PID (the same PID-reuse guard as `jobs.process.is_alive`), so
a stale pidfile left by a crash or reboot is detected and cleared rather
than blocking the next start forever.
"""

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import psutil

from sleap_rtc.jobs.process import is_alive

# How long `serve --daemonize` waits for the child to report ready. Generous:
# with --iroh, startup includes binding an iroh endpoint and reaching its
# relay, which can take several seconds on a slow or captive network.
DEFAULT_READY_TIMEOUT_SECS = 60.0

# How long `serve --stop` waits for the worker to exit after asking it to.
DEFAULT_STOP_TIMEOUT_SECS = 15.0

_POLL_INTERVAL_SECS = 0.1

# How many trailing lines of the child's log to show when startup fails.
_LOG_TAIL_LINES = 20


def pid_path(data_dir: Path) -> Path:
    """Where a running `serve` records its PID and start time."""
    return data_dir / "serve.pid"


def log_path(data_dir: Path) -> Path:
    """Where a daemonized `serve` writes its stdout/stderr."""
    return data_dir / "serve.log"


def ready_path(data_dir: Path) -> Path:
    """Where a daemonized `serve` child reports it is listening."""
    return data_dir / "serve.ready"


@dataclass
class ServeProcess:
    """A `serve` process recorded in a --data-dir's pidfile."""

    pid: int
    started_at: float


class AlreadyRunningError(RuntimeError):
    """Another `serve` is already running against this --data-dir."""

    def __init__(self, running: ServeProcess):
        self.running = running
        super().__init__(f"a worker is already running (pid {running.pid})")


def read_running(data_dir: Path) -> Optional[ServeProcess]:
    """Return the live `serve` process for `data_dir`, if there is one.

    A pidfile whose process is gone (or whose PID now belongs to a different
    process, per the recorded start time) is stale: it is removed, and this
    returns None.
    """
    path = pid_path(data_dir)
    try:
        data = json.loads(path.read_text())
        recorded = ServeProcess(
            pid=int(data["pid"]), started_at=float(data["started_at"])
        )
    except FileNotFoundError:
        return None
    except (ValueError, KeyError, TypeError):
        # Truncated or hand-edited — no way to tell what it referred to.
        path.unlink(missing_ok=True)
        return None

    if is_alive(recorded.pid, recorded.started_at):
        return recorded
    path.unlink(missing_ok=True)
    return None


def acquire_pidfile(data_dir: Path) -> ServeProcess:
    """Record the current process as `data_dir`'s running `serve`.

    Raises:
        AlreadyRunningError: if another live `serve` already holds it.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    me = ServeProcess(pid=os.getpid(), started_at=psutil.Process().create_time())
    payload = json.dumps({"pid": me.pid, "started_at": me.started_at})
    path = pid_path(data_dir)

    # Two attempts: the first can only fail on an existing file, which is
    # either a live worker (raise) or stale (`read_running` removes it).
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            running = read_running(data_dir)
            if running is None:
                continue
            if running.pid == me.pid:
                return me
            raise AlreadyRunningError(running)
        with os.fdopen(fd, "w") as f:
            f.write(payload)
        return me

    # Lost a race with another starting worker between removal and re-create.
    running = read_running(data_dir)
    raise AlreadyRunningError(running or me)


def release_pidfile(data_dir: Path) -> None:
    """Remove `data_dir`'s pidfile, but only if it is this process's."""
    path = pid_path(data_dir)
    try:
        if int(json.loads(path.read_text())["pid"]) == os.getpid():
            path.unlink(missing_ok=True)
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        pass


def write_ready(path: Path, info: dict) -> None:
    """Report (atomically) that this `serve` child is listening."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(info))
    os.replace(tmp, path)


def serve_argv(
    *,
    host: str,
    port: int,
    blob_port: Optional[int],
    data_dir: Path,
    mounts: Sequence[str],
    iroh: bool,
    metrics: bool,
) -> List[str]:
    """Build the `serve` arguments that reproduce these options in a child."""
    argv = ["serve", "--host", host, "--port", str(port), "--data-dir", str(data_dir)]
    if blob_port is not None:
        argv += ["--blob-port", str(blob_port)]
    for mount in mounts:
        argv += ["--mount", mount]
    argv.append("--iroh" if iroh else "--no-iroh")
    argv.append("--metrics" if metrics else "--no-metrics")
    return argv


class DaemonStartError(RuntimeError):
    """A daemonized `serve` child failed to come up."""

    def __init__(self, message: str, log_tail: str):
        self.log_tail = log_tail
        super().__init__(message)


def spawn_daemon(
    argv: Sequence[str],
    data_dir: Path,
    ready_timeout: float = DEFAULT_READY_TIMEOUT_SECS,
) -> dict:
    """Start `sleap-rtc <argv>` detached and wait until it reports ready.

    Args:
        argv: `serve` arguments, as built by `serve_argv`.
        data_dir: The child's --data-dir (holds its log and ready file).
        ready_timeout: Seconds to wait for the child to report ready.

    Returns:
        The child's ready info (pid, node_id, address, blob_port, ...).

    Raises:
        DaemonStartError: if the child exits before reporting ready, or
            doesn't report ready within `ready_timeout` (in which case it is
            left running — it may just be slow — and `serve --stop` stops it).
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    ready = ready_path(data_dir)
    ready.unlink(missing_ok=True)
    log = log_path(data_dir)

    # `-c` rather than `-m sleap_rtc.cli`: running cli.py as `__main__` would
    # import it a second time (as `sleap_rtc.cli`) the moment anything else
    # imports it.
    cmd = [
        sys.executable,
        "-c",
        "from sleap_rtc.cli import cli; cli(prog_name='sleap-rtc')",
        *argv,
        "--ready-file",
        str(ready),
    ]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    popen_kwargs: dict = {}
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        # Own session: no controlling terminal, so closing the terminal or
        # dropping the SSH session that launched it doesn't SIGHUP it.
        popen_kwargs["start_new_session"] = True

    log_start = log.stat().st_size if log.exists() else 0
    with open(log, "ab", buffering=0) as log_file:
        log_file.write(
            f"\n--- serve --daemonize starting at {time.ctime()} ---\n".encode()
        )
        child = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            close_fds=True,
            **popen_kwargs,
        )

    deadline = time.monotonic() + ready_timeout
    while time.monotonic() < deadline:
        if ready.exists():
            info = json.loads(ready.read_text())
            ready.unlink(missing_ok=True)
            return info
        if child.poll() is not None:
            raise DaemonStartError(
                f"the worker exited during startup (exit code {child.returncode})",
                _read_log_tail(log, log_start),
            )
        time.sleep(_POLL_INTERVAL_SECS)

    raise DaemonStartError(
        f"the worker (pid {child.pid}) didn't report ready within "
        f"{ready_timeout:.0f}s; it is still running and may just be slow",
        _read_log_tail(log, log_start),
    )


def _read_log_tail(log: Path, start: int) -> str:
    """The last few lines this run appended to `log`."""
    try:
        with open(log, "rb") as f:
            f.seek(start)
            text = f.read().decode(errors="replace")
    except FileNotFoundError:
        return ""
    return "\n".join(text.strip().splitlines()[-_LOG_TAIL_LINES:])


def stop_running(
    data_dir: Path, timeout: float = DEFAULT_STOP_TIMEOUT_SECS
) -> Optional[ServeProcess]:
    """Stop `data_dir`'s running `serve`, if any, and wait for it to exit.

    On POSIX this sends SIGTERM, which `serve` handles as a clean shutdown
    (closing the job store, removing its iroh live-info file and pidfile).
    Running training jobs are not affected: they run in their own process
    sessions and are reattached by the next `serve`. Windows has no
    deliverable equivalent for a process with no console, so there it is
    terminated outright and its leftover state files are removed here.

    Returns:
        The process that was stopped, or None if none was running.

    Raises:
        TimeoutError: if it is still running after `timeout` seconds.
    """
    running = read_running(data_dir)
    if running is None:
        return None

    proc = psutil.Process(running.pid)
    if sys.platform == "win32":
        proc.terminate()
    else:
        os.kill(running.pid, signal.SIGTERM)

    try:
        proc.wait(timeout=timeout)
    except psutil.TimeoutExpired:
        raise TimeoutError(
            f"the worker (pid {running.pid}) did not exit within {timeout:.0f}s"
        ) from None

    if sys.platform == "win32":
        from sleap_rtc.protocol_v1.iroh_live import iroh_live_path, remove_iroh_live

        remove_iroh_live(iroh_live_path(data_dir))
        pid_path(data_dir).unlink(missing_ok=True)
    return running
