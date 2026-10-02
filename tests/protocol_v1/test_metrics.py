"""Tests for job.metric/job.curve ZMQ forwarding (item 3.1).

`TestJobMetricsConsumer` uses real ZMQ sockets (a real PUB publisher
standing in for sleap-nn, a real `JobMetricsConsumer` SUB/PUB pair
underneath) — no mocking of zmq itself, matching this repo's established
"real infra over mocks" test convention (see `test_iroh_transport.py`,
`test_blob_http.py`, etc.).
"""

import asyncio
import json
import time

import pytest
import zmq

from sleap_rtc.protocol_v1.metrics import JobMetricsConsumer, downsample_m4

# Distinct from the real DEFAULT_ZMQ_PORTS (9000/9001) to avoid colliding
# with anything else that might be using those during a test run.
_TEST_CONTROL_PORT = 19100
_TEST_PUBLISH_PORT = 19101


class TestDownsampleM4:
    """Tests for the M4 (min/max/first/last-per-bucket) downsampler."""

    def test_returns_everything_under_the_bucket_budget(self):
        points = [(float(i), float(i)) for i in range(10)]
        assert downsample_m4(points, max_buckets=10) == points

    def test_caps_output_size_for_a_large_series(self):
        points = [(float(i), float(i % 7)) for i in range(10000)]
        out = downsample_m4(points, max_buckets=100)
        assert len(out) <= 100 * 4

    def test_preserves_a_spike(self):
        # A single huge spike buried in an otherwise-flat series must
        # survive downsampling — this is the whole point of M4 over naive
        # decimation or reservoir sampling.
        points = [(float(i), 1.0) for i in range(1000)]
        points[500] = (500.0, 999.0)
        out = downsample_m4(points, max_buckets=50)
        assert any(y == 999.0 for _, y in out)

    def test_empty_and_singleton_series(self):
        assert downsample_m4([], max_buckets=10) == []
        assert downsample_m4([(1.0, 2.0)], max_buckets=10) == [(1.0, 2.0)]

    def test_output_points_are_real_input_points_not_interpolated(self):
        points = [(float(i), float(i) * 1.37) for i in range(5000)]
        out = downsample_m4(points, max_buckets=50)
        point_set = set(points)
        assert all(p in point_set for p in out)


class _EmitRecorder:
    def __init__(self):
        self.calls = []

    async def __call__(self, topic, data):
        self.calls.append((topic, data))


async def _wait_for_emit(recorder, topic, timeout=5.0):
    """Return the first emitted payload for `topic`."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for t, d in recorder.calls:
            if t == topic:
                return d
        await asyncio.sleep(0.02)
    raise AssertionError(f"no {topic!r} event emitted within {timeout}s")


async def _wait_for_condition(recorder, topic, predicate, timeout=5.0):
    """Return the latest emitted payload for `topic` once it satisfies `predicate`.

    Unlike `_wait_for_emit`, this is for asserting on *final* state after
    several updates to the same job — metric state is cumulative, so the
    first emit is stale by the time later ones have landed.
    """
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        matches = [d for t, d in recorder.calls if t == topic]
        if matches:
            last = matches[-1]
            if predicate(last):
                return last
        await asyncio.sleep(0.02)
    raise AssertionError(
        f"condition on {topic!r} not met within {timeout}s (last seen: {last})"
    )


@pytest.fixture
async def fake_sleap_nn_publisher():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.connect(f"tcp://127.0.0.1:{_TEST_PUBLISH_PORT}")
    # PUB/SUB slow-joiner: a message sent immediately after connect can be
    # silently dropped before the subscription has propagated, even though
    # the SUB side is already bound. A real network property, not a bug —
    # give it a moment before the test sends anything for real.
    await asyncio.sleep(0.3)
    try:
        yield sock
    finally:
        sock.setsockopt(zmq.LINGER, 0)
        sock.close()
        ctx.term()


@pytest.fixture
async def consumer():
    recorder = _EmitRecorder()
    c = JobMetricsConsumer(
        control_port=_TEST_CONTROL_PORT,
        publish_port=_TEST_PUBLISH_PORT,
        emit=recorder,
        total_epochs=10,
        emit_interval=0.05,
    )
    c.start()
    # Slow-joiner: give the SUB socket's bind a moment to land before any
    # test publishes — otherwise early messages are silently dropped (a
    # real PUB/SUB property, not a bug in the consumer).
    for _ in range(50):
        if c._reporter.progress_socket is not None:
            break
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.1)
    c.recorder = recorder
    try:
        yield c
    finally:
        await c.stop()


class TestJobMetricsConsumer:
    """Tests for `JobMetricsConsumer` — real ZMQ messages in, job.metric/
    job.curve events out, rate-capped."""

    async def test_epoch_end_updates_metric_and_curve(
        self, consumer, fake_sleap_nn_publisher
    ):
        fake_sleap_nn_publisher.send_string(
            json.dumps(
                {
                    "event": "epoch_end",
                    "epoch": 0,
                    "logs": {"train/loss": 0.5, "val/loss": 0.6},
                }
            )
        )

        metric = await _wait_for_emit(consumer.recorder, "job.metric")
        assert metric["epoch"] == 0
        assert metric["latest_train_loss"] == 0.5
        assert metric["latest_val_loss"] == 0.6
        assert metric["best_loss"] == 0.5
        assert metric["total_epochs"] == 10

        curve = await _wait_for_emit(consumer.recorder, "job.curve")
        assert curve["points"][-1]["y"] == 0.5

    async def test_best_loss_tracks_the_minimum_across_epochs(
        self, consumer, fake_sleap_nn_publisher
    ):
        for epoch, loss in [(0, 0.9), (1, 0.3), (2, 0.7)]:
            fake_sleap_nn_publisher.send_string(
                json.dumps(
                    {"event": "epoch_end", "epoch": epoch, "logs": {"loss": loss}}
                )
            )
            await asyncio.sleep(0.15)

        metric = await _wait_for_condition(
            consumer.recorder, "job.metric", lambda m: m["epoch"] == 2
        )
        assert metric["best_loss"] == 0.3
        assert metric["latest_train_loss"] == 0.7

    async def test_train_begin_captures_wandb_url(
        self, consumer, fake_sleap_nn_publisher
    ):
        fake_sleap_nn_publisher.send_string(
            json.dumps({"event": "train_begin", "wandb_url": "https://wandb.ai/x/y"})
        )

        metric = await _wait_for_emit(consumer.recorder, "job.metric")
        assert metric["wandb_url"] == "https://wandb.ai/x/y"

    async def test_malformed_message_is_ignored_not_raised(
        self, consumer, fake_sleap_nn_publisher
    ):
        fake_sleap_nn_publisher.send_string("not json")
        fake_sleap_nn_publisher.send_string(
            json.dumps({"event": "epoch_end", "epoch": 0, "logs": {"loss": 0.1}})
        )

        metric = await _wait_for_emit(consumer.recorder, "job.metric")
        assert metric["epoch"] == 0  # the malformed message was silently skipped

    async def test_stop_is_idempotent_and_safe_without_start(self):
        c = JobMetricsConsumer(
            control_port=_TEST_CONTROL_PORT + 10,
            publish_port=_TEST_PUBLISH_PORT + 10,
            emit=_EmitRecorder(),
        )
        await c.stop()  # must not raise even though start() was never called
        await c.stop()  # and must be safe to call twice

    async def test_stop_uses_the_async_teardown_not_the_sync_one(
        self, consumer, monkeypatch
    ):
        # Regression guard: stop() must await ProgressReporter.async_cleanup()
        # — not call the sync cleanup() — since stop() always runs from an
        # event loop. The sync version only *requests* the listener task's
        # cancellation without awaiting it before closing the sockets it's
        # still using, which can race a still-running executor thread (see
        # ProgressReporter.cleanup's own docstring warning against exactly
        # this). The race is timing-dependent and doesn't reliably fail a
        # real-ZMQ test, so this pins down the *call*, not the race itself.
        calls = []
        monkeypatch.setattr(consumer._reporter, "cleanup", lambda: calls.append("sync"))
        orig_async_cleanup = consumer._reporter.async_cleanup

        async def spy_async_cleanup():
            calls.append("async")
            await orig_async_cleanup()

        monkeypatch.setattr(consumer._reporter, "async_cleanup", spy_async_cleanup)

        await consumer.stop()

        assert calls == ["async"]
