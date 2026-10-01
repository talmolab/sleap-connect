"""Forwards a training job's local ZMQ progress stream as job.metric/job.curve
protocol v1 events.

Reuses `ProgressReporter`'s already-proven ZMQ SUB-socket draining loop (the
same ports/messages `job_executor.py`'s legacy RTC path already forwards
verbatim as `PROGRESS_REPORT::<raw>`) by handing it a small `channel`-shaped
sink instead of a real `RTCDataChannel` — only the *consumption* is reused,
since `ProgressReporter.start_progress_listener`'s forwarding loop only ever
calls one method (`channel.send(str)`) on whatever object it's given.

The raw message shape (confirmed against `sleap-app`'s existing
`trainingStore.ts` parsing of these same `PROGRESS_REPORT::` messages, which
has been live since before protocol v1 existed) is a JSON object with an
`event` field (`"train_begin"`, `"epoch_begin"`, `"epoch_end"`, `"batch_end"`,
...), `epoch`, and a `logs` dict with `"loss"`/`"train/loss"`/`"val/loss"`
keys (occasionally nested under `py_dict` instead — jsonpickle encoding
quirk from older sleap-nn versions).
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from sleap_rtc.worker.progress_reporter import ProgressReporter

# Rate cap for job.metric/job.curve emission — negligible bandwidth even over
# a deliberately tiny relay throttle (see the roadmap's own math: ~1 KB/s).
EMIT_INTERVAL_SECS = 1.0

# M4 downsampling caps the WIRE payload at roughly this many points. Each
# bucket can contribute up to 4 points (first/min/max/last), so the bucket
# count is a quarter of the point budget.
MAX_CURVE_POINTS = 1000
MAX_CURVE_BUCKETS = MAX_CURVE_POINTS // 4

EmitFn = Callable[[str, dict], Awaitable[None]]


def downsample_m4(
    points: List[Tuple[float, float]], max_buckets: int = MAX_CURVE_BUCKETS
) -> List[Tuple[float, float]]:
    """M4 (min/max/first/last-per-bucket) downsampling of an (x, y) series.

    Visually lossless compared to naive decimation or reservoir sampling —
    every bucket keeps its own min and max y value, so a real loss spike
    always survives, which is exactly why the spec picked M4 over naive/
    reservoir sampling for a live loss curve.

    Args:
        points: (x, y) pairs, already sorted by non-decreasing x.
        max_buckets: Upper bound on the number of x-buckets.

    Returns:
        A subset of `points` (each returned point is one of the originals,
        never interpolated), no more than `max_buckets * 4` of them.
    """
    if max_buckets <= 0 or len(points) <= max_buckets * 4:
        return list(points)

    x0 = points[0][0]
    x1 = points[-1][0]
    span = x1 - x0
    if span <= 0:
        # All points share one x (shouldn't happen for a monotonic step
        # counter, but don't crash on it) — fall back to first/min/max/last.
        lo = min(points, key=lambda p: p[1])
        hi = max(points, key=lambda p: p[1])
        out = {points[0], points[-1], lo, hi}
        return sorted(out, key=lambda p: p[1])

    bucket_width = span / max_buckets
    out: List[Tuple[float, float]] = []
    current_bucket = 0
    bucket: List[Tuple[float, float]] = []

    def flush(pts: List[Tuple[float, float]]) -> None:
        if not pts:
            return
        first, last = pts[0], pts[-1]
        lo = min(pts, key=lambda p: p[1])
        hi = max(pts, key=lambda p: p[1])
        seen = set()
        for p in sorted((first, lo, hi, last), key=lambda p: p[0]):
            if p not in seen:
                seen.add(p)
                out.append(p)

    for p in points:
        idx = min(int((p[0] - x0) / bucket_width), max_buckets - 1)
        if idx != current_bucket and bucket:
            flush(bucket)
            bucket = []
            current_bucket = idx
        bucket.append(p)
    flush(bucket)
    return out


class _SinkChannel:
    """Adapts `ProgressReporter.start_progress_listener`'s one-method
    `channel.send(str)` forwarding call onto a plain callback, so that
    loop's real ZMQ-draining logic (NOBLOCK recv_all, polling interval,
    cleanup) can be reused verbatim with no RTCDataChannel in sight.
    """

    _PREFIX = "PROGRESS_REPORT::"

    def __init__(self, on_message: Callable[[str], None]):
        self._on_message = on_message

    def send(self, msg: str) -> None:
        if msg.startswith(self._PREFIX):
            msg = msg[len(self._PREFIX) :]
        self._on_message(msg)


@dataclass
class _MetricsState:
    epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    latest_train_loss: Optional[float] = None
    latest_val_loss: Optional[float] = None
    best_loss: Optional[float] = None
    wandb_url: Optional[str] = None
    eta_seconds: Optional[float] = None
    curve: List[Tuple[float, float]] = field(default_factory=list)
    dirty: bool = False
    _epoch_started_at: Optional[float] = None
    _epoch_durations: List[float] = field(default_factory=list)


class JobMetricsConsumer:
    """Subscribes to one training job's ZMQ progress stream and emits
    rate-capped `job.metric`/`job.curve` protocol v1 events for it.

    One instance per running train job (the worker only ever runs one job
    at a time — `JobQueue(max_concurrent=1)` — so there's no port-sharing
    concern reusing the same default ZMQ ports every time).
    """

    def __init__(
        self,
        control_port: int,
        publish_port: int,
        emit: EmitFn,
        total_epochs: Optional[int] = None,
        emit_interval: float = EMIT_INTERVAL_SECS,
    ):
        self._reporter = ProgressReporter(
            control_address=f"tcp://127.0.0.1:{control_port}",
            progress_address=f"tcp://127.0.0.1:{publish_port}",
        )
        self._emit = emit
        self._state = _MetricsState(total_epochs=total_epochs)
        self._step = 0.0
        self._emit_interval = emit_interval
        self._emit_task: Optional[asyncio.Task] = None

    def start(self) -> None:
        """Bind the ZMQ sockets and start draining + emitting in the background."""
        self._reporter.start_control_socket()
        self._reporter.start_progress_listener_task(_SinkChannel(self._on_raw_message))
        self._emit_task = asyncio.create_task(self._emit_loop())

    async def stop(self) -> None:
        """Stop the emit loop and tear down the ZMQ sockets.

        Safe to call even if `start()` was never called or already torn
        down — mirrors `ProgressReporter.async_cleanup`'s own idempotence.
        Uses the *async* teardown (not `ProgressReporter.cleanup`) since
        this always runs from an event loop — `async_cleanup` awaits the
        listener task's cancellation before closing the sockets it's still
        using, instead of racing it the way the sync version's own
        docstring warns against.
        """
        if self._emit_task is not None and not self._emit_task.done():
            self._emit_task.cancel()
            try:
                await self._emit_task
            except asyncio.CancelledError:
                pass
        await self._reporter.async_cleanup()

    def _on_raw_message(self, msg: str) -> None:
        try:
            data = json.loads(msg)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(data, dict):
            return

        event = data.get("event")
        payload = data
        if event is None and isinstance(data.get("py_dict"), dict):
            # Older sleap-nn jsonpickle encoding nests the real payload.
            payload = data["py_dict"]
            event = payload.get("event")

        if event == "train_begin":
            url = payload.get("wandb_url")
            if url:
                self._state.wandb_url = url
                self._state.dirty = True
            return

        if event == "epoch_end":
            self._record_epoch_end(payload)
            return

        if event == "batch_end":
            logs = payload.get("logs") or {}
            loss = logs.get("loss", logs.get("train/loss"))
            if isinstance(loss, (int, float)):
                self._step += 1.0
                self._state.curve.append((self._step, float(loss)))
                self._state.dirty = True
            return

    def _record_epoch_end(self, payload: dict) -> None:
        import time

        epoch = payload.get("epoch")
        logs = payload.get("logs") or {}
        train_loss = logs.get("train/loss", logs.get("loss"))
        val_loss = logs.get("val/loss")

        now = time.monotonic()
        if self._state._epoch_started_at is not None:
            self._state._epoch_durations.append(now - self._state._epoch_started_at)
            # Keep a short rolling window — early epochs (data loader warmup,
            # first-batch compilation) are not representative of steady state.
            self._state._epoch_durations = self._state._epoch_durations[-5:]
        self._state._epoch_started_at = now

        if isinstance(epoch, (int, float)):
            self._state.epoch = int(epoch)
        if isinstance(train_loss, (int, float)):
            self._state.latest_train_loss = float(train_loss)
            if self._state.best_loss is None or train_loss < self._state.best_loss:
                self._state.best_loss = float(train_loss)
            self._step += 1.0
            self._state.curve.append((self._step, float(train_loss)))
        if isinstance(val_loss, (int, float)):
            self._state.latest_val_loss = float(val_loss)

        self._state.eta_seconds = self._estimate_eta()
        self._state.dirty = True

    def _estimate_eta(self) -> Optional[float]:
        durations = self._state._epoch_durations
        total = self._state.total_epochs
        epoch = self._state.epoch
        if not durations or total is None or epoch is None:
            return None
        remaining = total - (epoch + 1)
        if remaining <= 0:
            return 0.0
        avg = sum(durations) / len(durations)
        return avg * remaining

    async def _emit_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._emit_interval)
                if not self._state.dirty:
                    continue
                self._state.dirty = False
                await self._emit_metric_and_curve()
        except asyncio.CancelledError:
            # Flush one last update so a job that completes between emit
            # ticks doesn't silently drop its final epoch's numbers.
            if self._state.dirty:
                self._state.dirty = False
                await self._emit_metric_and_curve()
            raise

    async def _emit_metric_and_curve(self) -> None:
        s = self._state
        try:
            await self._emit(
                "job.metric",
                {
                    "epoch": s.epoch,
                    "total_epochs": s.total_epochs,
                    "latest_train_loss": s.latest_train_loss,
                    "latest_val_loss": s.latest_val_loss,
                    "best_loss": s.best_loss,
                    "wandb_url": s.wandb_url,
                    "eta_seconds": s.eta_seconds,
                },
            )
            curve = downsample_m4(s.curve)
            await self._emit(
                "job.curve",
                {"points": [{"x": x, "y": y} for x, y in curve]},
            )
        except Exception:
            logging.exception("[metrics] Failed to emit job.metric/job.curve")
