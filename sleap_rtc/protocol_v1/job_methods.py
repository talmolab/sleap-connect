"""Job-lifecycle method handlers for the protocol v1 server.

Wires the SQLite job store, detached execution + reattach, and the job
queue into the wire protocol: `jobs.submit`, `jobs.cancel`, `jobs.status`,
`jobs.list`, `jobs.subscribe`. Also registers `fs.mounts`/`fs.list`/`fs.stat`/
`fs.read` when a `FileManager` is supplied, since that underlying capability
already exists and wiring it up is cheap.

Forwards sleap-nn's ZMQ epoch/loss stream as rate-capped `job.metric`/
`job.curve` events for training jobs (opt-in via `metrics_ports`, since every
existing caller of this class — tests especially — shouldn't suddenly start
binding real ZMQ sockets); see `sleap_rtc.protocol_v1.metrics`.
"""

import asyncio
import base64
import binascii
import json
import logging
import os
import re
import secrets
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from sleap_rtc.jobs.builder import DEFAULT_ZMQ_PORTS, CommandBuilder
from sleap_rtc.jobs.process import (
    is_alive,
    read_exit_code,
    send_cancel_signal,
    send_stop_signal,
    spawn_detached,
)
from sleap_rtc.jobs.queue import JobQueue
from sleap_rtc.jobs.spec import TrackJobSpec, TrainJobSpec, parse_job_spec
from sleap_rtc.jobs.store import TERMINAL_STATES, JobRecord, JobStore
from sleap_rtc.protocol_v1.blobs import BlobIndex, compute_chunk_hashes, hash_file
from sleap_rtc.protocol_v1.envelope import Event
from sleap_rtc.protocol_v1.errors import JOB_NOT_FOUND, JOB_SPEC_INVALID, ProtocolError
from sleap_rtc.protocol_v1.metrics import JobMetricsConsumer, read_training_log_epochs
from sleap_rtc.protocol_v1.server import Connection, ProtocolV1Server

# The line sleap-nn's track CLI prints on completion, e.g.
# "Predictions output path: /data/video.predictions.slp" — mirrors the
# legacy worker's job_executor.py capture (grep for it there for the
# original, since-verified-in-production wording this must keep matching).
_OUTPUT_PATH_MARKER = "Predictions output path:"

# How often to poll a job's log file for new lines. Polling (rather than an
# OS-level file-watch) is deliberate: it's portable across platforms with no
# extra dependency, and job.log traffic is inherently low-frequency (line-
# buffered subprocess output, not the high-rate ZMQ batch_end stream).
_LOG_POLL_INTERVAL_SECS = 0.5

# How often an in-place progress bar's current state (a tqdm `\r` redraw) is
# sent as a `job.log` `progress` event. tqdm redraws many times a second;
# forwarding every redraw is what flooded clients' logs.
_PROGRESS_EMIT_INTERVAL_SECS = 1.0


# CSI (ESC [ ... final byte) and OSC (ESC ] ... BEL/ST) sequences, as emitted
# by rich/tqdm progress output. They render invisibly but end up in copied logs.
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


def _visible_text(raw: str) -> str:
    """What a terminal shows for `raw`: its last non-empty `\r` segment."""
    raw = _ANSI_RE.sub("", raw)
    for segment in reversed(raw.split("\r")):
        segment = segment.rstrip()
        if segment:
            return segment
    return ""


def generate_job_id() -> str:
    """Generate a short, worker-assigned, URL-safe job identifier."""
    return f"job_{secrets.token_urlsafe(9)}"


class JobMethods:
    """Registers jobs.* (and, optionally, fs.mounts/fs.list) on a server."""

    def __init__(
        self,
        server: ProtocolV1Server,
        store: JobStore,
        queue: JobQueue,
        log_dir: Path,
        file_manager: Optional[object] = None,
        command_builder: Optional[object] = None,
        blob_index: Optional[BlobIndex] = None,
        metrics_ports: Optional[Dict[str, int]] = None,
    ):
        """Wire up and register the method handlers.

        Args:
            server: The `ProtocolV1Server` to register methods on.
            store: The job store (item 1.2).
            queue: The concurrency-gating job queue (item 1.3).
            log_dir: Directory to write each job's subprocess log file into.
            file_manager: An existing `FileManager` instance, if `fs.mounts`
                /`fs.list`/`fs.stat`/`fs.read` should be exposed. Omit to
                leave those methods unregistered.
            command_builder: An object with `build_command(spec) -> list[str]`.
                Defaults to the real `CommandBuilder` (builds actual
                ``sleap-nn`` invocations); tests inject a fake one so they
                can exercise the full submit → spawn → tail → complete flow
                with a trivial command instead of requiring sleap-nn.
            blob_index: Where to register a completed track job's output
                file so it becomes fetchable as a `job.result` blob (item
                1.10). Omit to leave `job.result` reporting `{blobs: {}}`
                like before — e.g. a worker not running the blob HTTP
                server has nowhere for a client to fetch the bytes from.
            metrics_ports: ZMQ `{"controller": port, "publish": port}` to
                bind a `JobMetricsConsumer` on for each train job, forwarding
                sleap-nn's epoch/loss stream as `job.metric`/`job.curve`
                events. Omit (default) to leave metrics forwarding disabled
                — e.g. for every existing caller of this class (tests
                especially), which shouldn't suddenly start binding real
                ZMQ sockets just by constructing a `JobMethods`. Production
                code passes `sleap_rtc.jobs.builder.DEFAULT_ZMQ_PORTS` (the
                same ports `CommandBuilder` wires into the actual `sleap-nn`
                invocation when no override is given).
        """
        self.server = server
        self.store = store
        self.queue = queue
        self.log_dir = Path(log_dir)
        # Per-job working folders (sleap-nn's ckpt_dir, materialized labels),
        # alongside the logs, so a train job's model is always findable.
        self.runs_dir = self.log_dir.parent / "job-runs"
        self.file_manager = file_manager
        self.blob_index = blob_index
        self.metrics_ports = metrics_ports
        self._builder = (
            command_builder if command_builder is not None else CommandBuilder()
        )
        # Tracks each job's background execution task, so a graceful worker
        # shutdown (or a test) can wait for one to genuinely finish —
        # including its trailing event emits — rather than just polling the
        # store for a terminal state, which races with those emits still
        # landing.
        self._tasks: Dict[str, asyncio.Task] = {}
        # Jobs waiting for a queue slot, in arrival order (the queue is FIFO),
        # so clients can show "#2 in line".
        self._waiting: List[str] = []

        server.register("jobs.submit", self.submit)
        server.register("jobs.cancel", self.cancel)
        server.register("jobs.status", self.status)
        server.register("jobs.list", self.list_jobs)
        server.register("jobs.subscribe", self.subscribe)
        if file_manager is not None:
            server.register("fs.mounts", self.fs_mounts)
            server.register("fs.list", self.fs_list)
            server.register("fs.stat", self.fs_stat)
            server.register("fs.read", self.fs_read)

    async def wait_for_job(self, job_id: str, timeout: Optional[float] = None) -> None:
        """Wait for a job's background execution task to fully finish.

        Unlike polling the store for a terminal state, this waits for the
        task itself to complete — including its trailing event emits — so
        there's no window where the job "looks done" in the store but is
        still about to touch it again. Intended for graceful worker
        shutdown and for tests; returns immediately if the job isn't
        currently tracked (never submitted, or already finished).

        Args:
            job_id: The job to wait for.
            timeout: Max seconds to wait, or None to wait indefinitely.
        """
        task = self._tasks.get(job_id)
        if task is not None:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)

    async def submit(self, params: dict, conn: Connection) -> dict:
        """Handle `jobs.submit` — validate the spec, create the job, run it in the background."""
        try:
            spec = parse_job_spec(json.dumps(params["spec"]))
        except Exception as e:
            raise ProtocolError(JOB_SPEC_INVALID, f"Invalid job spec: {e}") from e

        job_id = generate_job_id()
        await self.store.create_job(job_id, spec)
        self._tasks[job_id] = asyncio.create_task(self._run_job(job_id, spec))
        return {"job_id": job_id}

    async def _run_job(self, job_id: str, spec) -> None:
        """Run a job to completion: queue, spawn, tail its log, record the result."""
        try:
            await self._run_job_body(job_id, spec)
        finally:
            self._tasks.pop(job_id, None)

    def queue_position(self, job_id: str) -> Optional[int]:
        """1-based place in line of a job waiting for a slot, else None."""
        try:
            return self._waiting.index(job_id) + 1
        except ValueError:
            return None

    async def _run_job_body(self, job_id: str, spec) -> None:
        self._waiting.append(job_id)
        try:
            await self.queue.acquire()
        finally:
            self._waiting.remove(job_id)
        try:
            metrics_consumer = self._make_metrics_consumer(job_id, spec)
            try:
                await self._run_job_inner(job_id, spec, metrics_consumer)
            finally:
                # Always torn down, including on an exception mid-run —
                # ports are fixed (one job at a time per worker), so a
                # leaked ZMQ bind here would break the *next* job too.
                if metrics_consumer is not None:
                    await metrics_consumer.stop()
        finally:
            self.queue.release()

    async def _run_job_inner(
        self, job_id: str, spec, metrics_consumer: Optional[JobMetricsConsumer]
    ) -> None:
        try:
            await self.store.update_state(job_id, "running")
            await self._emit(job_id, "job.status", {"state": "running"})

            self._materialize_config_contents(spec)
            if isinstance(spec, TrainJobSpec):
                job_dir = self._job_dir(job_id)
                job_dir.mkdir(parents=True, exist_ok=True)
                spec.ckpt_dir = str(job_dir / "models")
                self._materialize_labels_content(spec, job_dir / "labels.slp")
            missing = _missing_inputs(spec)
            if missing:
                detail = f"not found on worker: {missing[0]}"
                await self.store.update_state(job_id, "failed", error=detail)
                await self._emit(
                    job_id,
                    "job.status",
                    {"state": "failed", "detail": detail, "missing": missing},
                )
                return
            cmd = self._builder.build_command(spec)
            log_path = self.log_dir / f"{job_id}.log"
            if metrics_consumer is not None:
                self._start_metrics_consumer(job_id, metrics_consumer)
            process = await spawn_detached(
                cmd, job_id=job_id, store=self.store, log_path=log_path
            )

            tail_task = asyncio.create_task(
                self._tail_log(job_id, log_path, lambda: process.returncode is None)
            )
            try:
                returncode = await process.wait()
                await tail_task  # final read: flush lines written right before exit
            finally:
                # On cancellation (worker shutting down) the tail task would
                # otherwise outlive this one and keep emitting.
                tail_task.cancel()
            await self._finish(job_id, spec, log_path, returncode)
        except Exception as e:
            logging.exception(f"[jobs] Job {job_id} failed with an unexpected error")
            await self.store.update_state(job_id, "failed", error=str(e))
            await self._emit(
                job_id, "job.status", {"state": "failed", "detail": str(e)}
            )

    async def _finish(
        self, job_id: str, spec, log_path: Path, returncode: Optional[int]
    ) -> None:
        """Record a job's terminal state once its process has exited."""
        if returncode == 0:
            blobs = await self._register_result_blobs(job_id, spec, log_path)
            result = {"blobs": blobs}
            if isinstance(spec, TrainJobSpec):
                result.update(self._train_outputs(job_id, spec))
            await self.store.update_state(job_id, "completed", result=result)
            # job.result BEFORE job.status: completed — a client that
            # resolves its "wait for this job" promise as soon as it
            # sees the terminal status (the natural, simplest thing
            # for it to do) must already have the result in hand at
            # that point, or it has no further chance to see it: the
            # instant a client considers a job over, it typically
            # unsubscribes, so a job.result arriving a message later
            # would silently go nowhere.
            await self._emit(job_id, "job.result", result)
            await self._emit(job_id, "job.status", {"state": "completed"})
        else:
            if returncode is None:
                detail = (
                    "job's process exited without recording an exit code "
                    "(killed outright, or started by an older worker version)"
                )
            else:
                detail = f"exit code {returncode}"
            await self.store.update_state(job_id, "failed", error=detail)
            await self._emit(
                job_id, "job.status", {"state": "failed", "detail": detail}
            )

    def _job_dir(self, job_id: str) -> Path:
        return self.runs_dir / job_id

    def _train_outputs(self, job_id: str, spec) -> dict:
        """Where a finished train job's model and training labels are.

        `model_dir` is what `sleap-nn track --model_paths` takes, so a client
        can run inference with it as a follow-up track job.
        """
        models = self._job_dir(job_id) / "models"
        model_dirs = sorted(
            (d for d in models.glob("*") if (d / "training_config.yaml").exists()),
            key=lambda d: d.stat().st_mtime,
        )
        labels = self._job_dir(job_id) / "labels.slp"
        return {
            "model_dir": str(model_dirs[-1]) if model_dirs else None,
            "labels_path": str(labels) if labels.exists() else spec.labels_path,
        }

    def resume_reattached(self, reattach_outcomes: Dict[str, str]) -> None:
        """Take over jobs a previous worker left running (see `reattach_all`).

        A reattached job is no longer this process's child, so nothing would
        otherwise ever notice it finish: it would stay "running" forever. For
        each one still running (or that finished while no worker was up),
        resume tailing its log where the previous worker stopped and record
        its terminal state from the exit code `_exit_code_wrapper.py` wrote.

        For a training job, also resumes live metrics: sleap-nn connects out
        to the worker's ZMQ ports and reconnects on its own once they're
        bound again. Epochs that finished while no worker was listening are
        backfilled from sleap-nn's `training_log.csv`.
        """
        for job_id, outcome in reattach_outcomes.items():
            if outcome in ("reattached", "exited"):
                self._tasks[job_id] = asyncio.create_task(self._run_reattached(job_id))

    async def _run_reattached(self, job_id: str) -> None:
        try:
            # Holds the queue slot like any running job, so a new submission
            # waits instead of sharing the GPU with it.
            async with self.queue.slot():
                record = await self.store.get_job(job_id)
                log_path = Path(record.log_path)
                events = await self.store.get_events_since(job_id, 0)
                already_emitted = sum(
                    1
                    for ev in events
                    if ev.topic == "job.log" and not ev.data.get("progress")
                )
                # Epochs that finished while no worker was listening, before
                # the live stream takes over again.
                await self._backfill_epochs(job_id)
                metrics_consumer = self._make_metrics_consumer(job_id, record.spec)
                try:
                    if metrics_consumer is not None:
                        self._start_metrics_consumer(job_id, metrics_consumer)
                    await self._tail_log(
                        job_id,
                        log_path,
                        lambda: is_alive(record.pid, record.process_started_at),
                        skip_lines=already_emitted,
                    )
                finally:
                    if metrics_consumer is not None:
                        await metrics_consumer.stop()
                # Anything that slipped between the two (e.g. an epoch ending
                # right as the live stream was rebinding).
                await self._backfill_epochs(job_id)
                await self._finish(
                    job_id, record.spec, log_path, read_exit_code(log_path)
                )
        except Exception as e:
            logging.exception(f"[jobs] Reattached job {job_id} failed unexpectedly")
            await self.store.update_state(job_id, "failed", error=str(e))
            await self._emit(
                job_id, "job.status", {"state": "failed", "detail": str(e)}
            )
        finally:
            self._tasks.pop(job_id, None)

    async def _backfill_epochs(self, job_id: str) -> None:
        """Emit `job.epoch` for epochs in the run's `training_log.csv` that no
        `job.epoch` event reports yet (they finished while no worker was
        listening to the job's ZMQ stream).
        """
        reported = {
            ev.data.get("epoch")
            for ev in await self.store.get_events_since(job_id, 0)
            if ev.topic == "job.epoch"
        }
        models = self._job_dir(job_id) / "models"
        for log_csv in sorted(models.glob("*/training_log.csv")):
            for epoch in read_training_log_epochs(log_csv):
                if epoch["epoch"] not in reported:
                    reported.add(epoch["epoch"])
                    await self._emit(job_id, "job.epoch", epoch)

    @staticmethod
    def _materialize_config_contents(spec) -> None:
        """Write `spec.config_contents` to temp files and populate
        `spec.config_paths`, since `CommandBuilder` only ever reads
        `config_paths` (it has no notion of inline config text).

        The client sends training config as inline YAML strings
        (`config_contents`) specifically so it doesn't need a separate
        "upload the config file first" round trip — but nothing on this
        (protocol v1) worker path ever materialized those into real files
        for `CommandBuilder` to point sleap-nn at, an integration gap only
        surfaced once a real training job was actually run end-to-end.
        Mirrors the legacy `worker_class.py`'s already-proven behavior
        (same temp-file-per-config + `path_mappings` text substitution),
        which was never ported to this newer path.

        A no-op for `TrackJobSpec` (no `config_contents` concept) and for
        a `TrainJobSpec` that came with `config_paths` already set instead.
        """
        if not isinstance(spec, TrainJobSpec) or not spec.config_contents:
            return

        temp_paths: list[str] = []
        for idx, content in enumerate(spec.config_contents):
            for old_path, new_path in (spec.path_mappings or {}).items():
                content = content.replace(old_path, new_path)
            fd, temp_path = tempfile.mkstemp(
                suffix=".yaml", prefix=f"job_config_{idx}_"
            )
            with os.fdopen(fd, "w") as f:
                f.write(content)
            temp_paths.append(temp_path)
        spec.config_paths = temp_paths

    @staticmethod
    def _materialize_labels_content(spec, dest: Optional[Path] = None) -> None:
        """Write `spec.labels_content` (base64-encoded raw .slp bytes) to a
        temp file and populate `spec.labels_path`, mirroring
        `_materialize_config_contents`'s exact approach for the same reason:
        `CommandBuilder` only ever reads `labels_path`, so an inline-content
        field is useless to it until materialized onto disk.

        A no-op for `TrackJobSpec` (no `labels_content` concept) and for a
        `TrainJobSpec` that didn't set `labels_content` (client resolved the
        labels file to a worker-visible path instead of inlining it).
        """
        if not isinstance(spec, TrainJobSpec) or not spec.labels_content:
            return

        try:
            raw = base64.b64decode(spec.labels_content, validate=True)
        except (ValueError, binascii.Error) as e:
            raise ProtocolError(
                JOB_SPEC_INVALID, f"Invalid labels_content (not valid base64): {e}"
            ) from e

        if dest is None:
            fd, temp_path = tempfile.mkstemp(suffix=".slp", prefix="job_labels_")
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
            spec.labels_path = temp_path
        else:
            dest.write_bytes(raw)
            spec.labels_path = str(dest)

    @staticmethod
    def _start_metrics_consumer(
        job_id: str, metrics_consumer: JobMetricsConsumer
    ) -> None:
        """Start a `JobMetricsConsumer`, degrading to "no metrics" instead of
        failing the whole job if its ZMQ sockets can't bind.

        `job.metric`/`job.curve` forwarding is best-effort telemetry, not a
        training-correctness concern — a bind failure (e.g. a stale process
        still holding the configured ports) must not turn into the entire
        training job failing before sleap-nn even gets a chance to run.
        """
        try:
            metrics_consumer.start()
        except Exception:
            logging.exception(
                f"[jobs] Job {job_id} could not start metrics forwarding — "
                "continuing without job.metric/job.curve"
            )

    def _make_metrics_consumer(self, job_id: str, spec) -> Optional[JobMetricsConsumer]:
        """Build a `JobMetricsConsumer` for a training job, or None if metrics
        forwarding isn't configured (`metrics_ports` omitted) or `spec` isn't
        a `TrainJobSpec` (track/inference jobs have no epoch/loss stream).
        """
        if self.metrics_ports is None or not isinstance(spec, TrainJobSpec):
            return None
        return JobMetricsConsumer(
            control_port=self.metrics_ports.get(
                "controller", DEFAULT_ZMQ_PORTS["controller"]
            ),
            publish_port=self.metrics_ports.get(
                "publish", DEFAULT_ZMQ_PORTS["publish"]
            ),
            emit=lambda topic, data: self._emit(job_id, topic, data),
            total_epochs=spec.max_epochs,
        )

    async def _tail_log(
        self,
        job_id: str,
        log_path: Path,
        is_running: Callable[[], bool],
        skip_lines: int = 0,
    ) -> None:
        """Poll a job's log file and emit its output as `job.log` events.

        Each finished (newline-terminated) line is one `job.log` event with
        what a terminal would show for it: a line redrawn in place with
        `\r` (a tqdm progress bar) collapses to its final state. That keeps
        lines 1:1 with events, which is what lets a restarted worker resume
        a reattached job's log by skipping the `skip_lines` lines the
        previous worker already emitted.

        A progress bar that is still being redrawn (no newline yet) is sent
        as `{"line": <current state>, "progress": True}` at most every
        `_PROGRESS_EMIT_INTERVAL_SECS`; clients replace the previous progress
        line with it instead of appending.

        Stops once `is_running()` is false (i.e. the subprocess itself has
        exited) — not once the job's *stored* state is no longer "running",
        since that state is only set *after* awaiting this task, which would
        otherwise deadlock the two waiting on each other.
        """
        pos = 0
        partial = ""
        last_progress = ""
        last_progress_at = 0.0
        while True:
            await asyncio.sleep(_LOG_POLL_INTERVAL_SECS)
            done = not is_running()  # checked before reading: no lost tail
            if log_path.exists():
                with open(log_path, "r", errors="replace", newline="") as f:
                    f.seek(pos)
                    new_data = f.read()
                    pos = f.tell()
                lines = (partial + new_data).split("\n")
                partial = lines.pop()
                if done and partial:
                    lines.append(partial)
                    partial = ""
                for line in lines:
                    line = _visible_text(line)
                    if not line:
                        continue
                    last_progress = ""
                    if skip_lines > 0:
                        skip_lines -= 1
                        continue
                    await self._emit(job_id, "job.log", {"line": line})

                if "\r" in partial:
                    # Only the latest redraw matters; don't let a long epoch's
                    # thousands of redraws accumulate in memory.
                    cut = partial.rstrip("\r").rfind("\r")
                    partial = partial[cut:]
                    current = _visible_text(partial)
                    now = time.monotonic()
                    if (
                        current
                        and current != last_progress
                        and now - last_progress_at >= _PROGRESS_EMIT_INTERVAL_SECS
                    ):
                        last_progress, last_progress_at = current, now
                        await self._emit(
                            job_id, "job.log", {"line": current, "progress": True}
                        )

            if done:
                return

    async def _register_result_blobs(self, job_id: str, spec, log_path: Path) -> dict:
        """Register a completed track job's output file as a result blob.

        Only track (inference) jobs produce a result worth fetching back —
        a training job's "result" is a checkpoint left in its own run
        directory, which the client doesn't need streamed to it the same
        way (see the module's connectStore-side counterpart notes). Never
        raises: a hashing/registration failure only means the client can't
        fetch this job's blob, not that the job itself failed — it already
        exited 0 by the time this runs.

        Returns:
            `{"predictions": {"sha256": ..., "size": ...}}` if a result
            file was found and registered, else `{}` — the same shape
            `job.result`'s `blobs` field has always had (spec §6.4).
        """
        if not isinstance(spec, TrackJobSpec) or self.blob_index is None:
            return {}
        try:
            output_path = self._resolve_track_output_path(spec, log_path)
            if output_path is None or not output_path.is_file():
                return {}
            sha256, size = await hash_file(output_path)
            chunk_hashes = await compute_chunk_hashes(output_path)
            await self.blob_index.register(sha256, str(output_path), size, chunk_hashes)
            return {"predictions": {"sha256": sha256, "size": size}}
        except Exception:
            logging.exception(
                f"[jobs] Job {job_id} completed but its result blob could not "
                "be registered — job.result will report no blobs"
            )
            return {}

    @staticmethod
    def _resolve_track_output_path(
        spec: TrackJobSpec, log_path: Path
    ) -> Optional[Path]:
        """Find a completed track job's output file.

        Priority (mirrors the legacy `job_executor.py`'s proven behavior —
        see its own docstring for the original wording this must keep
        matching): (1) `spec.output_path`, if the caller set one explicitly;
        (2) the path sleap-nn itself printed on completion ("Predictions
        output path: ..."), scraped from the job's persisted log file rather
        than a live stdout hook, since protocol v1 already writes the whole
        log to disk; (3) the naming convention sleap-nn falls back to when
        neither of the above applies.
        """
        if spec.output_path is not None:
            return Path(spec.output_path)

        captured = JobMethods._captured_output_path_from_log(log_path)
        if captured is not None:
            return Path(captured)

        base = Path(spec.data_path)
        return base.with_suffix(".predictions" + base.suffix)

    @staticmethod
    def _captured_output_path_from_log(log_path: Path) -> Optional[str]:
        if not log_path.exists():
            return None
        with open(log_path, "r", errors="replace") as f:
            for line in f:
                if _OUTPUT_PATH_MARKER in line:
                    return line.split(_OUTPUT_PATH_MARKER, 1)[1].strip()
        return None

    async def _emit(self, job_id: str, topic: str, data: dict) -> None:
        seq = await self.store.append_event(job_id, topic, data)
        await self.server.events.publish(
            job_id, Event(topic=topic, seq=seq, data=data, job_id=job_id)
        )

    async def cancel(self, params: dict, conn: Connection) -> dict:
        """Handle `jobs.cancel` — graceful stop or hard cancel of a running job.

        If the job is already in a terminal state, this is a no-op. If the
        store still shows it as active but its recorded process isn't
        actually alive (or was never recorded) — the worker restarted
        without a `reattach_all` pass, or some other inconsistency — this
        corrects the store to "failed" instead of silently doing nothing
        and leaving the job stuck "running" forever with no way to cancel
        it (there's nothing left to send a signal to).
        """
        job_id = params["job_id"]
        mode = params.get("mode", "cancel")
        record = await self._get_job_or_raise(job_id)

        if record.state in TERMINAL_STATES:
            return {}

        if (
            record.pid is not None
            and record.process_started_at is not None
            and is_alive(record.pid, record.process_started_at)
        ):
            if mode == "stop":
                send_stop_signal(record.pid)
            else:
                send_cancel_signal(record.pid)
        else:
            detail = "job's subprocess was not running when cancel was requested"
            await self.store.update_state(job_id, "failed", error=detail)
            await self._emit(
                job_id, "job.status", {"state": "failed", "detail": detail}
            )

        return {}

    async def status(self, params: dict, conn: Connection) -> dict:
        """Handle `jobs.status` — a full snapshot of one job."""
        record = await self._get_job_or_raise(params["job_id"])
        summary = _spec_summary(record.spec)
        return {
            **_record_to_dict(record),
            **summary,
            "queue_position": self.queue_position(record.job_id),
        }

    async def list_jobs(self, params: dict, conn: Connection) -> dict:
        """Handle `jobs.list` — a summary of every job this worker knows about.

        Newest first. Each entry is small enough to list hundreds of jobs: no
        full spec (`jobs.status` has it) and never inline labels.
        """
        jobs = []
        for r in await self.store.list_jobs():
            summary = _spec_summary(r.spec)
            summary.pop("spec")
            jobs.append(
                {
                    "job_id": r.job_id,
                    "state": r.state,
                    "created_at": r.created_at,
                    "updated_at": r.updated_at,
                    "error": r.error,
                    "queue_position": self.queue_position(r.job_id),
                    **summary,
                }
            )
        return {"jobs": jobs}

    async def subscribe(self, params: dict, conn: Connection) -> dict:
        """Handle `jobs.subscribe` — replay events-since-N, then live-subscribe."""
        job_id = params["job_id"]
        since_seq = params.get("since_seq", 0)
        await self._get_job_or_raise(job_id)  # 404s before subscribing to a ghost job

        self.server.events.subscribe(job_id, conn)

        backlog = await self.store.get_events_since(job_id, since_seq)
        for ev in backlog:
            await conn.send_event(
                Event(topic=ev.topic, seq=ev.seq, data=ev.data, job_id=job_id)
            )
        return {}

    async def fs_mounts(self, params: dict, conn: Connection) -> dict:
        """Handle `fs.mounts` — the configured, browsable mount roots."""
        return {"mounts": self.file_manager.get_mounts()}

    async def fs_list(self, params: dict, conn: Connection) -> dict:
        """Handle `fs.list` — directory listing, scoped to configured mounts."""
        return self.file_manager.list_directory(params["path"], params.get("offset", 0))

    async def fs_stat(self, params: dict, conn: Connection) -> dict:
        """Handle `fs.stat` — metadata for one path, scoped to configured mounts."""
        return self.file_manager.stat_path(params["path"])

    async def fs_read(self, params: dict, conn: Connection) -> dict:
        """Handle `fs.read` — a small direct byte range read, scoped to
        configured mounts. NOT the bulk-transfer path (see the blob API).
        """
        return self.file_manager.read_file(
            params["path"], params.get("offset", 0), params.get("length")
        )

    async def _get_job_or_raise(self, job_id: str) -> JobRecord:
        record = await self.store.get_job(job_id)
        if record is None:
            raise ProtocolError(JOB_NOT_FOUND, f"No such job: {job_id}")
        return record


def _missing_inputs(spec) -> List[str]:
    """Input paths a job names that don't exist on this worker.

    Checked before spawning, so a moved or mistyped file fails the job in
    seconds instead of partway into training. Video files referenced *inside*
    an SLP are not checked here (that would mean parsing it); clients check
    those with `fs.stat` before submitting.
    """
    if isinstance(spec, TrainJobSpec):
        paths = [spec.labels_path, spec.val_labels_path, *spec.config_paths]
    else:
        paths = [spec.data_path, *spec.model_paths]
    return [p for p in paths if p and not Path(p).exists()]


def _spec_summary(spec) -> dict:
    """What a client needs to list or re-run a job, minus bulky inline labels.

    `labels_content` (a base64 SLP, possibly hundreds of MB) is dropped;
    `config_contents` is kept, since "run again" and "start from a past job"
    need it.
    """
    data = spec.to_dict()
    data.pop("labels_content", None)
    kind = data.get("type")
    return {
        "kind": kind,
        "model_types": data.get("model_types") or [],
        "labels_path": (
            data.get("labels_path") if kind == "train" else data.get("data_path")
        ),
        "project": data.get("project"),
        "spec": data,
    }


def _record_to_dict(record: JobRecord) -> dict:
    return {
        "job_id": record.job_id,
        "state": record.state,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "result": record.result,
        "error": record.error,
    }
