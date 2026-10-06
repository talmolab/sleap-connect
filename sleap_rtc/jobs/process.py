"""Detached subprocess spawning, liveness checks, and reattach-on-restart.

The core idea: a job's ``sleap-nn`` subprocess must survive the worker
process itself dying or restarting (a training run can take hours; a worker
crash mid-run shouldn't kill the run too). This module provides:

- `spawn_detached`: starts a subprocess in its own session (immune to the
  worker's own signals/lifetime) with its stdout/stderr redirected to a log
  *file* rather than an in-process pipe — a pipe's read end only exists for
  the process that created it, but a file on disk can be tailed by anyone,
  including a brand new worker process after a restart. The command runs
  under `_exit_code_wrapper.py`, which records its exit code next to the
  log (`read_exit_code`) — the only way a *restarted* worker, no longer the
  job's parent, can tell whether it succeeded.
- `is_alive`: checks whether a recorded PID is still the *same* process we
  spawned (not a different process that happens to have recycled the PID),
  using the OS-reported process start time as a fingerprint.
- `reattach_all`: called once at worker startup, before accepting new jobs.
  Reconciles the job store's belief ("this job is running") against reality
  (is its subprocess actually still alive?).
- `send_stop_signal` / `send_cancel_signal`: graceful vs. hard job
  termination, matching the exact signal/process-group convention already
  established in `job_executor.py` (SIGINT for graceful stop, SIGTERM
  escalating to SIGKILL for hard cancel, `os.killpg` so distributed-worker
  child processes exit too, Windows fallback to `terminate()`/`kill()`) —
  reimplemented here with `asyncio` escalation instead of a thread, to match
  the rest of this module.

Does not touch `job_coordinator.py` / `job_executor.py` — wiring this into
actual job submission is a follow-up change.
"""

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import psutil
from asyncio import create_subprocess_exec
from asyncio.subprocess import STDOUT, Process

from sleap_rtc.jobs.store import JobStore

# Grace period after SIGTERM before escalating to SIGKILL — matches
# job_executor.py's `_CANCEL_GRACE_SECS`.
_CANCEL_GRACE_SECS = 5

# asyncio only holds a *weak* reference to a task once nothing else does —
# a fire-and-forget task like the SIGKILL escalation below can be garbage
# collected mid-`sleep()` with no warning. Keep a strong reference here for
# its lifetime; the done-callback discards it once it finishes.
_background_tasks: set = set()

# How much clock skew to tolerate when comparing a process's recorded start
# time against its live-queried one. `psutil.Process.create_time()` has
# sub-second precision that can vary slightly by measurement method; this is
# generous enough to absorb that without being so loose it'd accept a
# different process that happened to start within the same second.
_START_TIME_TOLERANCE_SECS = 2.0

_EXIT_CODE_WRAPPER = Path(__file__).with_name("_exit_code_wrapper.py")


def exit_code_path(log_path: Union[str, Path]) -> Path:
    """Where a job's exit code is recorded, next to its log file."""
    return Path(log_path).with_suffix(".exit")


def read_exit_code(log_path: Union[str, Path]) -> Optional[int]:
    """The recorded exit code of the job logging to `log_path`, if it has
    exited and recorded one (see `_exit_code_wrapper.py`).
    """
    try:
        return int(exit_code_path(log_path).read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


async def spawn_detached(
    cmd: Sequence[str],
    *,
    job_id: str,
    store: JobStore,
    log_path: Union[str, Path],
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
) -> Process:
    """Spawn a job's subprocess, detached, and record it in the job store.

    The subprocess is started in its own session (`start_new_session=True`
    — POSIX `setsid`-equivalent) so it isn't tied to the worker's own
    process group and survives the worker exiting. Its combined
    stdout/stderr are redirected to `log_path` rather than a pipe, so the
    job's output remains readable (e.g. by `reattach_all` or a `job.log`
    tailer) even after the spawning worker process is gone.

    Records `pid` / the subprocess's own start time / `log_path` on the job
    via `store.set_process_info` immediately after spawning, before the
    caller awaits it — if the *worker* dies between spawn and that call
    landing, `reattach_all` has nothing to find, and the job is correctly
    treated as lost rather than silently orphaned with no record at all.
    But if the *worker survives* and `set_process_info` itself fails (e.g. a
    transient SQLite error), the subprocess is already running and we have
    no other way to find it later — so that case kills the just-spawned
    process before re-raising, rather than leaving an orphan that
    `reattach_all` can never discover.

    Args:
        cmd: Full command to execute (e.g. ``["sleap-nn", "train", ...]``).
        job_id: The job this subprocess belongs to. Must already exist in
            `store`.
        store: The JobStore to record process info in.
        log_path: Where to redirect the subprocess's stdout/stderr.
        cwd: Working directory for the subprocess.
        env: Environment variables for the subprocess.

    Returns:
        The spawned `asyncio.subprocess.Process`.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    exit_path = exit_code_path(log_path)
    exit_path.unlink(missing_ok=True)

    with open(log_path, "ab", buffering=0) as log_file:
        process = await create_subprocess_exec(
            sys.executable,
            str(_EXIT_CODE_WRAPPER),
            str(exit_path),
            "--",
            *cmd,
            stdout=log_file,
            stderr=STDOUT,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
    # `log_file` is closed here in the parent, but the child received its own
    # dup'd file descriptor at spawn time (standard POSIX/Windows subprocess
    # semantics) and keeps writing to it independently.

    try:
        started_at = psutil.Process(process.pid).create_time()
        await store.set_process_info(job_id, process.pid, started_at, str(log_path))
    except Exception:
        logging.warning(
            f"[jobs] Failed to record process info for job {job_id} "
            f"(pid {process.pid}) — killing the just-spawned process to "
            f"avoid an untracked orphan"
        )
        try:
            process.kill()
        except ProcessLookupError:
            pass
        raise

    return process


def is_alive(pid: int, started_at: float) -> bool:
    """Check whether `pid` is still the same process recorded at `started_at`.

    A bare `pid` existence check isn't enough: PIDs get reused by the OS, so
    an old, long-dead job's PID could now belong to an unrelated process. We
    additionally compare the process's own reported start time (recorded at
    spawn time in `spawn_detached`) to rule that out.

    Args:
        pid: The process ID to check.
        started_at: The process's recorded start time (`psutil.Process.
            create_time()` at spawn time).

    Returns:
        True if a process with this PID exists, we can read its start time,
        and that start time matches (within `_START_TIME_TOLERANCE_SECS`);
        False otherwise — including when the PID exists but now belongs to
        a different user's process we don't have permission to inspect
        (`psutil.AccessDenied`), plausible on a shared multi-user machine
        after PID reuse. We can't confirm it's ours, so treat it the same as
        "not alive" rather than letting the exception propagate — this is
        called in a loop by `reattach_all` over every "running" job, and one
        permission error shouldn't abort reconciling the rest.
    """
    try:
        proc = psutil.Process(pid)
        actual_started_at = proc.create_time()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False

    return abs(actual_started_at - started_at) <= _START_TIME_TOLERANCE_SECS


async def reattach_all(store: JobStore) -> Dict[str, str]:
    """Reconcile the job store against reality at worker startup.

    For every job the store believes is still "running": if its recorded
    subprocess is genuinely still alive, leave it running (the caller is
    expected to resume tailing its `log_path` for progress — this function
    only determines the *true* state, it doesn't re-establish log tailing
    itself). If not — the worker crashed, or the machine rebooted, and the
    job's subprocess died with it despite being detached — mark the job
    "failed" so it doesn't sit "running" forever with nothing actually
    running behind it.

    Call this once, before the worker starts accepting new job submissions.

    Args:
        store: The JobStore to reconcile.

    Returns:
        A dict of ``{job_id: outcome}`` for every job that was "running",
        where outcome is one of ``"reattached"`` (still running),
        ``"exited"`` (finished while no worker was running, with a recorded
        exit code — still stored as "running", for the caller to finalize
        from `read_exit_code`), or ``"marked_failed"``.
    """
    outcomes: Dict[str, str] = {}

    for record in await store.list_jobs():
        if record.state != "running":
            continue

        if (
            record.pid is not None
            and record.process_started_at is not None
            and is_alive(record.pid, record.process_started_at)
        ):
            outcomes[record.job_id] = "reattached"
            continue

        if record.log_path is not None and read_exit_code(record.log_path) is not None:
            outcomes[record.job_id] = "exited"
            continue

        await store.update_state(
            record.job_id,
            "failed",
            error=(
                "worker restarted and this job's subprocess was no longer "
                "running (crash, reboot, or the process was otherwise lost)"
            ),
        )
        outcomes[record.job_id] = "marked_failed"

    return outcomes


def send_stop_signal(pid: int) -> None:
    """Ask a job's process group to stop gracefully (SIGINT; terminate() on Windows).

    Mirrors `job_executor.py`'s `stop_running_job` exactly — same signal
    choice, same `os.killpg` (so distributed-worker child processes exit
    alongside the main one, allowing sleap-nn to save a checkpoint), same
    Windows fallback.

    Args:
        pid: PID of the job's subprocess (the process group leader, since
            it was spawned with `start_new_session=True`).
    """
    try:
        if sys.platform == "win32":
            for proc in _windows_job_processes(pid):
                proc.terminate()
        else:
            pgid = os.getpgid(pid)
            logging.info(f"Sending SIGINT to process group {pgid} (graceful stop)")
            os.killpg(pgid, signal.SIGINT)
    except (ProcessLookupError, psutil.NoSuchProcess):
        pass


def send_cancel_signal(pid: int) -> None:
    """Hard-cancel a job's process group (SIGTERM, escalating to SIGKILL).

    Mirrors `job_executor.py`'s `cancel_running_job`, with the escalation
    delay implemented as an `asyncio` task instead of a `threading.Thread`,
    to match the rest of this module. Must be called from a running event
    loop.

    Args:
        pid: PID of the job's subprocess (the process group leader).
    """
    try:
        if sys.platform == "win32":
            for proc in _windows_job_processes(pid):
                proc.kill()
            return

        pgid = os.getpgid(pid)
        logging.info(f"Sending SIGTERM to process group {pgid} (hard cancel)")
        os.killpg(pgid, signal.SIGTERM)
        task = asyncio.create_task(_escalate_to_sigkill(pid))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
    except (ProcessLookupError, psutil.NoSuchProcess):
        pass


def _windows_job_processes(pid: int) -> list:
    """The job's own processes under its `_exit_code_wrapper` (Windows has
    no process groups to signal). Falls back to `pid` itself if it has no
    children, e.g. it already exited them.
    """
    wrapper = psutil.Process(pid)
    return wrapper.children(recursive=True) or [wrapper]


async def _escalate_to_sigkill(pid: int) -> None:
    await asyncio.sleep(_CANCEL_GRACE_SECS)
    try:
        if not psutil.pid_exists(pid):
            return
        pgid = os.getpgid(pid)
        logging.warning(
            f"Process group {pgid} did not exit after SIGTERM "
            f"({_CANCEL_GRACE_SECS}s), escalating to SIGKILL"
        )
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, psutil.NoSuchProcess):
        pass
