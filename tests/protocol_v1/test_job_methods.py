"""Integration tests for the jobs.* / fs.* protocol v1 method handlers.

Uses a real JobStore (tmp-file SQLite), a real JobQueue, and real
subprocesses — swapping in a fake `CommandBuilder` that returns a plain
Python invocation instead of a real ``sleap-nn`` command, so the full
submit → spawn → tail-log → complete flow is exercised for real without
requiring sleap-nn to be installed.
"""

import asyncio
import json
import hashlib
import sys
import time
from pathlib import Path

import pytest
import zmq

from sleap_rtc.jobs.queue import JobQueue
from sleap_rtc.jobs.spec import TrackJobSpec, TrainJobSpec
from sleap_rtc.jobs.store import JobStore
from sleap_rtc.protocol_v1.blobs import BlobIndex
from sleap_rtc.protocol_v1.errors import JOB_NOT_FOUND, ProtocolError
from sleap_rtc.protocol_v1.job_methods import JobMethods
from sleap_rtc.protocol_v1 import job_methods as job_methods_module
from sleap_rtc.protocol_v1.server import Connection, ProtocolV1Server

TERMINAL_STATES = ("completed", "failed", "canceled")


class _FakeCommandBuilder:
    """Returns a fixed command regardless of the spec, for testing."""

    def __init__(self, cmd):
        self._cmd = cmd

    def build_command(self, spec):
        return self._cmd


class _FakeWs:
    """A fake websocket connection that records what was sent to it."""

    def __init__(self):
        self.sent = []

    async def send(self, data: str) -> None:
        self.sent.append(data)


class _FakeFileManager:
    def __init__(self, mounts, listing, stat=None, read=None):
        self._mounts = mounts
        self._listing = listing
        self._stat = stat
        self._read = read

    def get_mounts(self):
        return self._mounts

    def list_directory(self, path, offset=0):
        return self._listing

    def stat_path(self, path):
        return self._stat

    def read_file(self, path, offset=0, length=None):
        return self._read


async def _wait_for_terminal(methods, store, job_id, timeout=10):
    # Waits for the job's background task to fully finish (including its
    # trailing event emits), not just for the store to show a terminal
    # state — polling the store alone races with those emits still landing.
    await methods.wait_for_job(job_id, timeout=timeout)
    record = await store.get_job(job_id)
    assert (
        record.state in TERMINAL_STATES
    ), f"job {job_id} ended in state {record.state!r}"
    return record


async def _wait_for_state(store, job_id, state, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = await store.get_job(job_id)
        if record.state == state:
            return record
        await asyncio.sleep(0.05)
    raise AssertionError(
        f"job {job_id} did not reach state {state!r} within {timeout}s"
    )


@pytest.fixture(autouse=True)
def _skip_input_preflight(request, monkeypatch):
    """Most tests here run a fake command against made-up input paths (they
    never read them); only `TestPreflight` exercises the missing-input check."""
    if request.cls is None or request.cls.__name__ != "TestPreflight":
        monkeypatch.setattr(job_methods_module, "_missing_inputs", lambda spec: [])


@pytest.fixture
def spec():
    return TrainJobSpec(config_path="/data/centroid.yaml")


@pytest.fixture
async def store(tmp_path):
    async with JobStore(tmp_path / "jobs.sqlite") as store:
        yield store


def _make_methods(
    store, tmp_path, cmd, file_manager=None, blob_index=None, metrics_ports=None
):
    server = ProtocolV1Server(node_id="test-node")
    queue = JobQueue(max_concurrent=1)
    return JobMethods(
        server,
        store,
        queue,
        log_dir=tmp_path / "logs",
        file_manager=file_manager,
        command_builder=_FakeCommandBuilder(cmd),
        blob_index=blob_index,
        metrics_ports=metrics_ports,
    )


class TestSubmitAndRunToCompletion:
    """Tests for jobs.submit driving a job through to completion."""

    async def test_successful_job_streams_log_and_completes(
        self, store, tmp_path, spec
    ):
        methods = _make_methods(
            store, tmp_path, [sys.executable, "-c", "print('line1'); print('line2')"]
        )

        result = await methods.submit({"spec": spec.to_dict()}, conn=None)
        job_id = result["job_id"]

        record = await _wait_for_terminal(methods, store, job_id)

        assert record.state == "completed"
        events = await store.get_events_since(job_id)
        log_lines = [e.data["line"] for e in events if e.topic == "job.log"]
        assert "line1" in log_lines
        assert "line2" in log_lines
        statuses = [e.data["state"] for e in events if e.topic == "job.status"]
        assert statuses == ["running", "completed"]
        assert any(e.topic == "job.result" for e in events)

    async def test_failing_command_marks_job_failed(self, store, tmp_path, spec):
        methods = _make_methods(
            store, tmp_path, [sys.executable, "-c", "import sys; sys.exit(1)"]
        )

        result = await methods.submit({"spec": spec.to_dict()}, conn=None)
        job_id = result["job_id"]

        record = await _wait_for_terminal(methods, store, job_id)

        assert record.state == "failed"
        assert "exit code 1" in record.error

    async def test_invalid_spec_raises_job_spec_invalid(self, store, tmp_path):
        methods = _make_methods(store, tmp_path, [sys.executable, "-c", "pass"])

        with pytest.raises(ProtocolError) as exc_info:
            await methods.submit({"spec": {"type": "not-a-real-type"}}, conn=None)

        assert exc_info.value.code == "job.spec_invalid"


class TestResultBlobs:
    """Tests for registering a completed track job's output as a blob."""

    def _track_spec(self, data_path):
        return TrackJobSpec(data_path=str(data_path), model_paths=["/models/centroid"])

    async def test_registers_the_blob_at_spec_output_path(self, store, tmp_path):
        content = b"predicted labels"
        output_path = tmp_path / "out.slp"
        index = BlobIndex(tmp_path / "blobs.sqlite")
        spec = TrackJobSpec(
            data_path=str(tmp_path / "video.mp4"),
            model_paths=["/models/centroid"],
            output_path=str(output_path),
        )
        # Fake command: write the output file sleap-nn would have produced.
        cmd = [
            sys.executable,
            "-c",
            f"open({str(output_path)!r}, 'wb').write({content!r})",
        ]
        methods = _make_methods(store, tmp_path, cmd, blob_index=index)

        result = await methods.submit({"spec": spec.to_dict()}, conn=None)
        record = await _wait_for_terminal(methods, store, result["job_id"])

        expected_sha256 = hashlib.sha256(content).hexdigest()
        assert record.result == {
            "blobs": {"predictions": {"sha256": expected_sha256, "size": len(content)}}
        }
        blob = await index.get(expected_sha256)
        assert blob is not None
        assert blob.path == str(output_path)
        assert blob.size == len(content)

        status = await methods.status({"job_id": result["job_id"]}, conn=None)
        assert status["result"]["blobs"]["predictions"]["sha256"] == expected_sha256

        events = await store.get_events_since(result["job_id"])
        result_events = [e for e in events if e.topic == "job.result"]
        assert len(result_events) == 1
        assert (
            result_events[0].data["blobs"]["predictions"]["sha256"] == expected_sha256
        )

    async def test_falls_back_to_the_captured_stdout_path(self, store, tmp_path):
        content = b"captured-path predictions"
        captured_path = tmp_path / "captured.slp"
        index = BlobIndex(tmp_path / "blobs.sqlite")
        spec = self._track_spec(tmp_path / "video.mp4")  # no output_path set
        cmd = [
            sys.executable,
            "-c",
            f"open({str(captured_path)!r}, 'wb').write({content!r}); "
            f"print('Predictions output path: {captured_path}')",
        ]
        methods = _make_methods(store, tmp_path, cmd, blob_index=index)

        result = await methods.submit({"spec": spec.to_dict()}, conn=None)
        await _wait_for_terminal(methods, store, result["job_id"])

        expected_sha256 = hashlib.sha256(content).hexdigest()
        blob = await index.get(expected_sha256)
        assert blob is not None
        assert blob.path == str(captured_path)

    async def test_falls_back_to_the_naming_convention(self, store, tmp_path):
        data_path = tmp_path / "video.slp"
        content = b"convention-fallback predictions"
        convention_path = tmp_path / "video.predictions.slp"
        index = BlobIndex(tmp_path / "blobs.sqlite")
        spec = self._track_spec(data_path)  # no output_path, nothing captured
        cmd = [
            sys.executable,
            "-c",
            f"open({str(convention_path)!r}, 'wb').write({content!r})",
        ]
        methods = _make_methods(store, tmp_path, cmd, blob_index=index)

        result = await methods.submit({"spec": spec.to_dict()}, conn=None)
        await _wait_for_terminal(methods, store, result["job_id"])

        expected_sha256 = hashlib.sha256(content).hexdigest()
        blob = await index.get(expected_sha256)
        assert blob is not None
        assert blob.path == str(convention_path)

    async def test_no_blobs_when_the_output_file_never_materializes(
        self, store, tmp_path
    ):
        index = BlobIndex(tmp_path / "blobs.sqlite")
        spec = self._track_spec(tmp_path / "video.mp4")
        methods = _make_methods(
            store, tmp_path, [sys.executable, "-c", "pass"], blob_index=index
        )

        result = await methods.submit({"spec": spec.to_dict()}, conn=None)
        record = await _wait_for_terminal(methods, store, result["job_id"])

        assert record.state == "completed"
        assert record.result == {"blobs": {}}

    async def test_no_blobs_without_a_blob_index_configured(self, store, tmp_path):
        output_path = tmp_path / "out.slp"
        spec = TrackJobSpec(
            data_path=str(tmp_path / "video.mp4"),
            model_paths=["/models/centroid"],
            output_path=str(output_path),
        )
        cmd = [sys.executable, "-c", f"open({str(output_path)!r}, 'wb').write(b'x')"]
        methods = _make_methods(store, tmp_path, cmd)  # no blob_index

        result = await methods.submit({"spec": spec.to_dict()}, conn=None)
        record = await _wait_for_terminal(methods, store, result["job_id"])

        assert record.result == {"blobs": {}}

    async def test_train_jobs_never_register_a_result_blob(self, store, tmp_path, spec):
        index = BlobIndex(tmp_path / "blobs.sqlite")
        methods = _make_methods(
            store, tmp_path, [sys.executable, "-c", "pass"], blob_index=index
        )

        result = await methods.submit({"spec": spec.to_dict()}, conn=None)
        record = await _wait_for_terminal(methods, store, result["job_id"])

        assert record.result["blobs"] == {}
        # A train job reports where its model went instead (see TestTrainOutputs).
        assert "model_dir" in record.result

    async def test_job_result_is_emitted_before_job_status_completed(
        self, store, tmp_path
    ):
        # A client that resolves its "wait for this job" promise as soon as
        # it sees the terminal job.status (the natural, simplest thing for
        # it to do — see sleap-app's connectStore) must already have seen
        # job.result by then, or it has no further chance to: the instant a
        # client considers a job over it typically unsubscribes, so a
        # job.result arriving a message later would silently go nowhere.
        output_path = tmp_path / "out.slp"
        index = BlobIndex(tmp_path / "blobs.sqlite")
        spec = TrackJobSpec(
            data_path=str(tmp_path / "video.mp4"),
            model_paths=["/models/centroid"],
            output_path=str(output_path),
        )
        cmd = [sys.executable, "-c", f"open({str(output_path)!r}, 'wb').write(b'x')"]
        methods = _make_methods(store, tmp_path, cmd, blob_index=index)

        result = await methods.submit({"spec": spec.to_dict()}, conn=None)
        await _wait_for_terminal(methods, store, result["job_id"])

        events = await store.get_events_since(result["job_id"])
        topics_in_order = [e.topic for e in events]
        result_idx = topics_in_order.index("job.result")
        completed_idx = next(
            i
            for i, e in enumerate(events)
            if e.topic == "job.status" and e.data.get("state") == "completed"
        )
        assert result_idx < completed_idx


class TestStatusAndList:
    """Tests for jobs.status / jobs.list."""

    async def test_status_returns_a_snapshot(self, store, tmp_path, spec):
        methods = _make_methods(store, tmp_path, [sys.executable, "-c", "pass"])
        result = await methods.submit({"spec": spec.to_dict()}, conn=None)
        await _wait_for_terminal(methods, store, result["job_id"])

        status = await methods.status({"job_id": result["job_id"]}, conn=None)

        assert status["job_id"] == result["job_id"]
        assert status["state"] == "completed"

    async def test_status_raises_not_found_for_unknown_job(self, store, tmp_path):
        methods = _make_methods(store, tmp_path, [sys.executable, "-c", "pass"])

        with pytest.raises(ProtocolError) as exc_info:
            await methods.status({"job_id": "does-not-exist"}, conn=None)

        assert exc_info.value.code == JOB_NOT_FOUND

    async def test_list_jobs_returns_every_known_job(self, store, tmp_path, spec):
        methods = _make_methods(store, tmp_path, [sys.executable, "-c", "pass"])
        r1 = await methods.submit({"spec": spec.to_dict()}, conn=None)
        r2 = await methods.submit({"spec": spec.to_dict()}, conn=None)
        await _wait_for_terminal(methods, store, r1["job_id"])
        await _wait_for_terminal(methods, store, r2["job_id"])

        listing = await methods.list_jobs({}, conn=None)

        job_ids = {j["job_id"] for j in listing["jobs"]}
        assert job_ids == {r1["job_id"], r2["job_id"]}


class TestCancel:
    """Tests for jobs.cancel against a genuinely long-running job."""

    async def test_stop_mode_terminates_a_running_job(self, store, tmp_path, spec):
        methods = _make_methods(
            store, tmp_path, [sys.executable, "-c", "import time; time.sleep(30)"]
        )
        result = await methods.submit({"spec": spec.to_dict()}, conn=None)
        job_id = result["job_id"]
        await _wait_for_state(store, job_id, "running")

        await methods.cancel({"job_id": job_id, "mode": "stop"}, conn=None)

        await _wait_for_terminal(methods, store, job_id, timeout=5)

    async def test_cancel_mode_terminates_a_running_job(self, store, tmp_path, spec):
        methods = _make_methods(
            store, tmp_path, [sys.executable, "-c", "import time; time.sleep(30)"]
        )
        result = await methods.submit({"spec": spec.to_dict()}, conn=None)
        job_id = result["job_id"]
        await _wait_for_state(store, job_id, "running")

        await methods.cancel({"job_id": job_id, "mode": "cancel"}, conn=None)

        # A plain script has no SIGTERM handler, so it should exit well
        # within the SIGKILL escalation grace period — this test doesn't
        # exercise the escalation path itself, only that cancel actually
        # terminates the job.
        await _wait_for_terminal(methods, store, job_id, timeout=5)

    async def test_cancel_corrects_a_stuck_running_job_with_no_live_process(
        self, store, tmp_path, spec
    ):
        """If the store shows "running" but no process is actually alive for
        it (e.g. the worker restarted without a reattach_all pass), cancel
        must correct the store to "failed" instead of silently doing
        nothing — there's no process to signal, but the job shouldn't be
        stuck "running" forever with no way to resolve it.
        """
        methods = _make_methods(store, tmp_path, [sys.executable, "-c", "pass"])
        job = await store.create_job("job-stuck", spec)
        await store.update_state(job.job_id, "running")  # no pid ever recorded

        await methods.cancel({"job_id": job.job_id, "mode": "cancel"}, conn=None)

        record = await store.get_job(job.job_id)
        assert record.state == "failed"
        assert "not running" in record.error

    async def test_cancel_handles_a_pid_recorded_without_a_process_started_at(
        self, store, tmp_path, spec
    ):
        """`set_process_info` always writes `pid` and `process_started_at`
        together, so this combination shouldn't occur via any real code
        path — but `cancel`'s liveness guard must not assume that. Without
        also checking `process_started_at is not None`, `is_alive` would be
        called with `None` and raise `TypeError` from
        `abs(actual_started_at - started_at)`, unlike `reattach_all`'s
        equivalent guard, which already checks both.
        """
        import os

        methods = _make_methods(store, tmp_path, [sys.executable, "-c", "pass"])
        job = await store.create_job("job-partial", spec)
        await store.update_state(job.job_id, "running")
        # Simulate the otherwise-unreachable inconsistency directly,
        # bypassing set_process_info's atomic pid+process_started_at write.
        await store._conn.execute(
            "UPDATE jobs SET pid = ? WHERE job_id = ?", (os.getpid(), job.job_id)
        )
        await store._conn.commit()

        await methods.cancel({"job_id": job.job_id, "mode": "cancel"}, conn=None)

        record = await store.get_job(job.job_id)
        assert record.state == "failed"


class TestMaterializeConfigContents:
    """Tests for `_materialize_config_contents`.

    The client only ever sends training config as inline YAML strings
    (`config_contents`), never as worker-local file paths — but
    `CommandBuilder` only ever reads `config_paths`, an integration gap
    that surfaced as `IndexError: list index out of range` (indexing an
    empty `config_paths`) the first time a real training job ran end to
    end. `_materialize_config_contents` writes each content string to a
    temp file and populates `config_paths` from that, mirroring the
    legacy `worker_class.py`'s already-proven behavior for this same
    problem.
    """

    def test_writes_each_content_to_its_own_temp_file(self):
        spec = TrainJobSpec(
            config_contents=["centroid: yaml", "centered_instance: yaml"],
        )

        JobMethods._materialize_config_contents(spec)

        assert len(spec.config_paths) == 2
        assert Path(spec.config_paths[0]).read_text() == "centroid: yaml"
        assert Path(spec.config_paths[1]).read_text() == "centered_instance: yaml"

    def test_applies_path_mappings_to_the_content(self):
        spec = TrainJobSpec(
            config_contents=["data_config.train_labels_path=/local/labels.slp"],
            path_mappings={"/local/labels.slp": "/worker/labels.slp"},
        )

        JobMethods._materialize_config_contents(spec)

        written = Path(spec.config_paths[0]).read_text()
        assert written == "data_config.train_labels_path=/worker/labels.slp"

    def test_is_a_noop_when_config_paths_already_given(self):
        spec = TrainJobSpec(config_paths=["/already/there.yaml"])

        JobMethods._materialize_config_contents(spec)

        assert spec.config_paths == ["/already/there.yaml"]

    def test_is_a_noop_for_a_track_spec(self):
        spec = TrackJobSpec(data_path="/x.slp", model_paths=["/m"])

        JobMethods._materialize_config_contents(spec)  # must not raise

        assert spec.data_path == "/x.slp"

    def test_materialized_spec_builds_a_real_command_without_crashing(self):
        # The actual bug this closes: build_train_command only ever read
        # config_paths[config_index], and the client only ever sends
        # config_contents — so this raised IndexError before the fix.
        from sleap_rtc.jobs.builder import CommandBuilder

        spec = TrainJobSpec(config_contents=["centroid: yaml"])

        JobMethods._materialize_config_contents(spec)
        cmd = CommandBuilder().build_command(spec)

        assert cmd[:2] == ["sleap-nn", "train"]


class TestMaterializeLabelsContent:
    """Tests for `_materialize_labels_content` (item 3.1).

    Mirrors `_materialize_config_contents`'s exact approach: the client
    sends raw .slp bytes inline (base64-encoded, since .slp is a binary
    HDF5 file, not text) when it has no worker-resolvable path for the
    labels file; the worker writes them to a temp file and points
    `labels_path` at it, since `CommandBuilder` only ever reads paths.
    """

    def test_writes_decoded_bytes_to_a_temp_file(self):
        import base64

        raw = b"\x89HDF\r\n\x1a\nnot a real slp file"
        spec = TrainJobSpec(
            config_contents=["x: y"], labels_content=base64.b64encode(raw).decode()
        )

        JobMethods._materialize_labels_content(spec)

        assert spec.labels_path is not None
        assert Path(spec.labels_path).read_bytes() == raw

    def test_is_a_noop_when_labels_content_absent(self):
        spec = TrainJobSpec(config_contents=["x: y"], labels_path="/already/there.slp")

        JobMethods._materialize_labels_content(spec)

        assert spec.labels_path == "/already/there.slp"

    def test_is_a_noop_for_a_track_spec(self):
        spec = TrackJobSpec(data_path="/x.slp", model_paths=["/m"])

        JobMethods._materialize_labels_content(spec)  # must not raise

        assert spec.data_path == "/x.slp"

    def test_invalid_base64_raises_job_spec_invalid(self):
        spec = TrainJobSpec(
            config_contents=["x: y"], labels_content="not-valid-base64!!!"
        )

        with pytest.raises(ProtocolError) as exc_info:
            JobMethods._materialize_labels_content(spec)

        assert exc_info.value.code == "job.spec_invalid"


class TestSubscribe:
    """Tests for jobs.subscribe — backlog replay plus live delivery."""

    async def test_replays_backlog_and_then_delivers_live_events(
        self, store, tmp_path, spec
    ):
        methods = _make_methods(store, tmp_path, [sys.executable, "-c", "pass"])
        result = await methods.submit({"spec": spec.to_dict()}, conn=None)
        job_id = result["job_id"]
        await _wait_for_terminal(methods, store, job_id)  # backlog already exists

        ws = _FakeWs()
        conn = Connection(ws)
        await methods.subscribe({"job_id": job_id, "since_seq": 0}, conn)

        assert len(ws.sent) > 0  # the backlog was replayed

        ws.sent.clear()
        await methods._emit(job_id, "job.log", {"line": "a live one"})

        assert len(ws.sent) == 1

    async def test_subscribe_raises_not_found_for_unknown_job(self, store, tmp_path):
        methods = _make_methods(store, tmp_path, [sys.executable, "-c", "pass"])
        conn = Connection(_FakeWs())

        with pytest.raises(ProtocolError) as exc_info:
            await methods.subscribe({"job_id": "does-not-exist"}, conn)

        assert exc_info.value.code == JOB_NOT_FOUND


class TestFsMethods:
    """Tests for fs.mounts / fs.list delegating to a FileManager."""

    async def test_fs_mounts_delegates_to_file_manager(self, store, tmp_path):
        fm = _FakeFileManager(mounts=[{"path": "/data", "label": "Data"}], listing=None)
        methods = _make_methods(
            store, tmp_path, [sys.executable, "-c", "pass"], file_manager=fm
        )

        result = await methods.fs_mounts({}, conn=None)

        assert result == {"mounts": [{"path": "/data", "label": "Data"}]}

    async def test_fs_list_delegates_to_file_manager(self, store, tmp_path):
        listing = {"entries": [{"name": "a.slp", "type": "file"}], "total_count": 1}
        fm = _FakeFileManager(mounts=[], listing=listing)
        methods = _make_methods(
            store, tmp_path, [sys.executable, "-c", "pass"], file_manager=fm
        )

        result = await methods.fs_list({"path": "/data"}, conn=None)

        assert result == listing

    async def test_fs_stat_delegates_to_file_manager(self, store, tmp_path):
        stat_result = {
            "path": "/data/a.slp",
            "type": "file",
            "size": 123,
            "modified": 0.0,
        }
        fm = _FakeFileManager(mounts=[], listing=None, stat=stat_result)
        methods = _make_methods(
            store, tmp_path, [sys.executable, "-c", "pass"], file_manager=fm
        )

        result = await methods.fs_stat({"path": "/data/a.slp"}, conn=None)

        assert result == stat_result

    async def test_fs_read_delegates_to_file_manager(self, store, tmp_path):
        read_result = {"path": "/data/a.slp", "content_base64": "aGVsbG8=", "size": 5}
        fm = _FakeFileManager(mounts=[], listing=None, read=read_result)
        methods = _make_methods(
            store, tmp_path, [sys.executable, "-c", "pass"], file_manager=fm
        )

        result = await methods.fs_read({"path": "/data/a.slp"}, conn=None)

        assert result == read_result

    def test_fs_methods_not_registered_without_a_file_manager(self, tmp_path):
        server = ProtocolV1Server(node_id="test-node")
        queue = JobQueue(max_concurrent=1)
        JobMethods(
            server,
            store=None,
            queue=queue,
            log_dir=tmp_path,
            command_builder=_FakeCommandBuilder([sys.executable]),
        )
        assert "fs.mounts" not in server._methods
        assert "fs.list" not in server._methods
        assert "fs.stat" not in server._methods
        assert "fs.read" not in server._methods


class TestMetricsWiring:
    """End-to-end: a real ZMQ-publishing fake train job, through
    jobs.submit, emits job.metric events alongside the usual job.log/
    job.status — exercising `_make_metrics_consumer`'s integration into
    `_run_job_body`, not just `JobMetricsConsumer` in isolation (see
    test_metrics.py for that).
    """

    _CONTROL_PORT = 19200
    _PUBLISH_PORT = 19201

    def _fake_train_cmd(self):
        # A fake "sleap-nn" that publishes one real epoch_end ZMQ message
        # on the same ports a real JobMetricsConsumer would bind to, then
        # exits — mirrors sleap-nn's actual wire format (see metrics.py's
        # module docstring).
        script = (
            "import json, time, zmq; "
            "ctx = zmq.Context(); "
            "sock = ctx.socket(zmq.PUB); "
            f"sock.connect('tcp://127.0.0.1:{self._PUBLISH_PORT}'); "
            "time.sleep(0.5); "  # slow-joiner
            "sock.send_string(json.dumps({"
            "'event': 'epoch_end', 'epoch': 0, "
            "'logs': {'train/loss': 0.42}"
            "})); "
            "time.sleep(0.3); "  # give the consumer's 0.05s poll a chance
            "sock.close(); ctx.term()"
        )
        return [sys.executable, "-c", script]

    async def test_training_job_emits_job_metric(self, store, tmp_path, spec):
        methods = _make_methods(
            store,
            tmp_path,
            self._fake_train_cmd(),
            metrics_ports={
                "controller": self._CONTROL_PORT,
                "publish": self._PUBLISH_PORT,
            },
        )

        result = await methods.submit({"spec": spec.to_dict()}, conn=None)
        job_id = result["job_id"]
        await _wait_for_terminal(methods, store, job_id, timeout=15)

        events = await store.get_events_since(job_id)
        metrics = [e.data for e in events if e.topic == "job.metric"]
        assert any(m["latest_train_loss"] == 0.42 for m in metrics)
        epochs = [e.data for e in events if e.topic == "job.epoch"]
        assert any(ep["train_loss"] == 0.42 for ep in epochs)

    async def test_track_job_never_starts_a_metrics_consumer(self, store, tmp_path):
        # TrackJobSpec has no epoch/loss stream — must not try to bind a
        # ZMQ socket for it even when metrics_ports is configured.
        spec = TrackJobSpec(data_path=str(tmp_path / "video.mp4"), model_paths=["/m"])
        methods = _make_methods(
            store,
            tmp_path,
            [sys.executable, "-c", "pass"],
            metrics_ports={
                "controller": self._CONTROL_PORT + 1,
                "publish": self._PUBLISH_PORT + 1,
            },
        )

        result = await methods.submit({"spec": spec.to_dict()}, conn=None)
        record = await _wait_for_terminal(methods, store, result["job_id"])

        assert record.state == "completed"
        events = await store.get_events_since(result["job_id"])
        assert not any(e.topic == "job.metric" for e in events)

    async def test_metrics_disabled_by_default(self, store, tmp_path, spec):
        # No metrics_ports passed — the default used by every other test in
        # this file. Must not raise or bind anything.
        methods = _make_methods(store, tmp_path, [sys.executable, "-c", "pass"])
        assert methods._make_metrics_consumer("job-1", spec) is None

    async def test_job_still_completes_when_metrics_port_is_already_bound(
        self, store, tmp_path, spec
    ):
        # A stale process (or just bad luck) holding one of the configured
        # ZMQ ports must not take the whole training job down — job.metric/
        # job.curve are best-effort telemetry, not a training-correctness
        # concern. Regression test for the gap where
        # `JobMetricsConsumer.start()`'s bind failure propagated out of
        # `_run_job_inner` and failed the job before sleap-nn ever ran.
        control_port = self._CONTROL_PORT + 10
        publish_port = self._PUBLISH_PORT + 10

        ctx = zmq.Context()
        blocker = ctx.socket(zmq.PUB)
        blocker.bind(f"tcp://127.0.0.1:{control_port}")
        try:
            methods = _make_methods(
                store,
                tmp_path,
                [sys.executable, "-c", "print('hello')"],
                metrics_ports={"controller": control_port, "publish": publish_port},
            )
            result = await methods.submit({"spec": spec.to_dict()}, conn=None)
            record = await _wait_for_terminal(methods, store, result["job_id"])

            assert record.state == "completed"
            events = await store.get_events_since(result["job_id"])
            assert not any(e.topic == "job.metric" for e in events)
        finally:
            blocker.setsockopt(zmq.LINGER, 0)
            blocker.close()
            ctx.term()


class TestResumeReattached:
    """A job left running by a previous worker is finished by the next one.

    Simulates a worker restart in-process: the first `JobMethods` spawns
    the job, then its watcher task is cancelled (the worker going away —
    the job's detached process keeps running), and a second `JobMethods`
    over the same store takes over via `reattach_all` + `resume_reattached`.
    """

    @staticmethod
    def _job_cmd(go_file, exit_code=0):
        # Prints a line, waits for the test to say "go", prints another.
        return [
            sys.executable,
            "-c",
            "import os, sys, time\n"
            "print('line 1', flush=True)\n"
            f"while not os.path.exists({str(go_file)!r}): time.sleep(0.05)\n"
            "print('line 2', flush=True)\n"
            f"sys.exit({exit_code})",
        ]

    async def _start_then_restart_worker(self, store, tmp_path, spec, cmd):
        from sleap_rtc.jobs.process import reattach_all

        first = _make_methods(store, tmp_path, cmd)
        job_id = (await first.submit({"spec": spec.to_dict()}, conn=None))["job_id"]
        deadline = time.monotonic() + 10
        while not any(
            ev.topic == "job.log" for ev in await store.get_events_since(job_id, 0)
        ):
            assert time.monotonic() < deadline, "job never logged its first line"
            await asyncio.sleep(0.05)
        task = first._tasks[job_id]
        task.cancel()  # the first worker goes away; the job keeps running
        await asyncio.gather(task, return_exceptions=True)
        return job_id, reattach_all

    async def _log_lines(self, store, job_id):
        return [
            ev.data["line"]
            for ev in await store.get_events_since(job_id, 0)
            if ev.topic == "job.log"
        ]

    async def test_a_reattached_job_completes_with_its_log_resumed_exactly(
        self, store, spec, tmp_path
    ):
        go = tmp_path / "go"
        cmd = self._job_cmd(go)
        job_id, reattach_all = await self._start_then_restart_worker(
            store, tmp_path, spec, cmd
        )

        outcomes = await reattach_all(store)
        assert outcomes == {job_id: "reattached"}
        second = _make_methods(store, tmp_path, cmd)
        second.resume_reattached(outcomes)
        go.touch()

        record = await _wait_for_terminal(second, store, job_id)
        assert record.state == "completed"
        # No line lost or repeated across the restart.
        assert await self._log_lines(store, job_id) == ["line 1", "line 2"]
        topics = [ev.topic for ev in await store.get_events_since(job_id, 0)]
        assert topics[-2:] == ["job.result", "job.status"]

    async def test_a_reattached_job_that_fails_is_marked_failed_with_its_code(
        self, store, spec, tmp_path
    ):
        go = tmp_path / "go"
        cmd = self._job_cmd(go, exit_code=3)
        job_id, reattach_all = await self._start_then_restart_worker(
            store, tmp_path, spec, cmd
        )

        second = _make_methods(store, tmp_path, cmd)
        second.resume_reattached(await reattach_all(store))
        go.touch()

        record = await _wait_for_terminal(second, store, job_id)
        assert record.state == "failed"
        assert record.error == "exit code 3"

    async def test_a_job_that_finished_while_no_worker_ran_is_finalized(
        self, store, spec, tmp_path
    ):
        from sleap_rtc.jobs.process import is_alive

        go = tmp_path / "go"
        cmd = self._job_cmd(go)
        job_id, reattach_all = await self._start_then_restart_worker(
            store, tmp_path, spec, cmd
        )
        go.touch()  # finishes while "no worker is running"
        record = await store.get_job(job_id)
        deadline = time.monotonic() + 10
        while is_alive(record.pid, record.process_started_at):
            assert time.monotonic() < deadline
            await asyncio.sleep(0.05)

        outcomes = await reattach_all(store)
        assert outcomes == {job_id: "exited"}
        second = _make_methods(store, tmp_path, cmd)
        second.resume_reattached(outcomes)

        record = await _wait_for_terminal(second, store, job_id)
        assert record.state == "completed"
        # The line printed while no worker was up is still delivered.
        assert await self._log_lines(store, job_id) == ["line 1", "line 2"]

    async def test_a_reattached_job_holds_the_queue_slot(self, store, spec, tmp_path):
        go = tmp_path / "go"
        cmd = self._job_cmd(go)
        job_id, reattach_all = await self._start_then_restart_worker(
            store, tmp_path, spec, cmd
        )

        second = _make_methods(store, tmp_path, cmd)
        second.resume_reattached(await reattach_all(store))
        await asyncio.sleep(0.2)
        new_id = (await second.submit({"spec": spec.to_dict()}, conn=None))["job_id"]
        await asyncio.sleep(1.0)
        # Waits for the reattached job instead of sharing the GPU with it.
        assert (await store.get_job(new_id)).state == "queued"

        go.touch()
        await _wait_for_terminal(second, store, job_id)
        await _wait_for_terminal(second, store, new_id)


class TestProgressLines:
    """tqdm-style `\\r` redraws become throttled `progress` events."""

    async def test_progress_bar_is_throttled_then_finalized_as_one_line(
        self, store, spec, tmp_path
    ):
        # ~2.5 s of redraws, 20 per second, like a fast tqdm bar.
        script = (
            "import sys, time\n"
            "print('start', flush=True)\n"
            "for i in range(50):\n"
            "    sys.stdout.write(f'\\rEpoch 1: {i*2}%|bar| {i}/50')\n"
            "    sys.stdout.flush()\n"
            "    time.sleep(0.05)\n"
            "sys.stdout.write('\\rEpoch 1: 100%|bar| 50/50\\n')\n"
            "print('done', flush=True)\n"
        )
        methods = _make_methods(store, tmp_path, [sys.executable, "-c", script])
        job_id = (await methods.submit({"spec": spec.to_dict()}, conn=None))["job_id"]
        await _wait_for_terminal(methods, store, job_id)

        logs = [
            e.data for e in await store.get_events_since(job_id) if e.topic == "job.log"
        ]
        lines = [d["line"] for d in logs if not d.get("progress")]
        progress = [d["line"] for d in logs if d.get("progress")]

        assert lines == ["start", "Epoch 1: 100%|bar| 50/50", "done"]
        # Redraws are visible while running, but nowhere near one per redraw.
        assert 1 <= len(progress) <= 5
        assert all(p.startswith("Epoch 1: ") and "\r" not in p for p in progress)

    async def test_ansi_control_sequences_are_stripped(self, store, spec, tmp_path):
        script = (
            "import sys\n"
            "sys.stdout.write('\\x1b[2K\\x1b[1Aloading \\x1b[32mok\\x1b[0m\\n')\n"
        )
        methods = _make_methods(store, tmp_path, [sys.executable, "-c", script])
        job_id = (await methods.submit({"spec": spec.to_dict()}, conn=None))["job_id"]
        await _wait_for_terminal(methods, store, job_id)

        logs = [
            e.data for e in await store.get_events_since(job_id) if e.topic == "job.log"
        ]
        assert logs == [{"line": "loading ok"}]

    async def test_crlf_output_is_not_mistaken_for_a_redraw(
        self, store, spec, tmp_path
    ):
        script = "import sys; sys.stdout.write('a\\r\\nb\\r\\n')"
        methods = _make_methods(store, tmp_path, [sys.executable, "-c", script])
        job_id = (await methods.submit({"spec": spec.to_dict()}, conn=None))["job_id"]
        await _wait_for_terminal(methods, store, job_id)

        logs = [
            e.data for e in await store.get_events_since(job_id) if e.topic == "job.log"
        ]
        assert logs == [{"line": "a"}, {"line": "b"}]


class _CkptAwareBuilder:
    """Runs `script(ckpt_dir)`, mimicking sleap-nn writing into its ckpt_dir."""

    def __init__(self, script):
        self._script = script

    def build_command(self, spec):
        return [sys.executable, "-c", self._script(spec.ckpt_dir)]


def _make_ckpt_methods(store, tmp_path, script):
    return JobMethods(
        ProtocolV1Server(node_id="test-node"),
        store,
        JobQueue(max_concurrent=1),
        log_dir=tmp_path / "job-logs",
        command_builder=_CkptAwareBuilder(script),
    )


def _write_run(ckpt_dir, epochs=(), wait_for=None):
    """Script: create a sleap-nn-like run folder, optionally logging epochs."""
    return (
        "import os, time\n"
        f"run = os.path.join({ckpt_dir!r}, 'centroid')\n"
        "os.makedirs(run, exist_ok=True)\n"
        "open(os.path.join(run, 'training_config.yaml'), 'w').write('x: 1')\n"
        "print('line 1', flush=True)\n"
        + (
            f"while not os.path.exists({str(wait_for)!r}): time.sleep(0.05)\n"
            if wait_for
            else ""
        )
        + "with open(os.path.join(run, 'training_log.csv'), 'w') as f:\n"
        "    f.write('epoch,train_loss,val_loss,learning_rate\\n')\n"
        + "".join(
            f"    f.write('{e},{0.5 / (e + 1)},{0.6 / (e + 1)},0.001\\n')\n"
            for e in epochs
        )
    )


class TestTrainOutputs:
    async def test_result_reports_model_dir_and_materialized_labels(
        self, store, tmp_path
    ):
        import base64

        spec = TrainJobSpec(
            config_path="/data/centroid.yaml",
            labels_content=base64.b64encode(b"slp-bytes").decode(),
        )
        methods = _make_ckpt_methods(store, tmp_path, _write_run)
        job_id = (await methods.submit({"spec": spec.to_dict()}, conn=None))["job_id"]
        record = await _wait_for_terminal(methods, store, job_id)

        job_dir = tmp_path / "job-runs" / job_id
        assert record.state == "completed"
        assert record.result["model_dir"] == str(job_dir / "models" / "centroid")
        assert record.result["labels_path"] == str(job_dir / "labels.slp")
        assert (job_dir / "labels.slp").read_bytes() == b"slp-bytes"
        result_events = [
            e.data
            for e in await store.get_events_since(job_id)
            if e.topic == "job.result"
        ]
        assert result_events == [record.result]

    async def test_client_supplied_ckpt_dir_is_overridden(self, store, tmp_path):
        spec = TrainJobSpec(config_path="/data/c.yaml", ckpt_dir="/etc")
        methods = _make_ckpt_methods(store, tmp_path, _write_run)
        job_id = (await methods.submit({"spec": spec.to_dict()}, conn=None))["job_id"]
        record = await _wait_for_terminal(methods, store, job_id)
        assert record.result["model_dir"].startswith(str(tmp_path / "job-runs"))


class _TrainThenTrackBuilder:
    """A train job's command writes a fake model dir (`train_script`,
    typically `_write_run`); a track job's command just exits 0. Lets a
    single `JobMethods`/`CommandBuilder` pair run both legs of a chained
    post-inference run, like the real worker does.
    """

    def __init__(self, train_script=_write_run):
        self._train_script = train_script

    def build_command(self, spec):
        if isinstance(spec, TrainJobSpec):
            return [sys.executable, "-c", self._train_script(spec.ckpt_dir)]
        return [sys.executable, "-c", "pass"]


def _make_chaining_methods(store, tmp_path, train_script=_write_run):
    return JobMethods(
        ProtocolV1Server(node_id="test-node"),
        store,
        JobQueue(max_concurrent=1),
        log_dir=tmp_path / "job-logs",
        command_builder=_TrainThenTrackBuilder(train_script),
    )


class TestPostInferenceChaining:
    """A train job's `post_inference` chains as track job(s) once every job
    in its `run` (or just itself, run-less) has completed.
    """

    async def test_single_job_run_chains_one_track_job(self, store, tmp_path):
        spec = TrainJobSpec(
            config_path="/data/centroid.yaml",
            labels_path="/data/train.slp",
            project={"name": "flies.slp", "id": "p1"},
            post_inference=[{"peak_threshold": 0.3, "tracker": "simple"}],
        )
        methods = _make_chaining_methods(store, tmp_path)

        job_id = (await methods.submit({"spec": spec.to_dict()}, conn=None))["job_id"]
        record = await _wait_for_terminal(methods, store, job_id)

        assert record.state == "completed"
        assert len(record.result["chained_job_ids"]) == 1
        chained_id = record.result["chained_job_ids"][0]
        chained = await _wait_for_terminal(methods, store, chained_id)

        assert chained.state == "completed"
        assert isinstance(chained.spec, TrackJobSpec)
        assert (
            chained.spec.data_path == record.result["labels_path"] == "/data/train.slp"
        )
        assert chained.spec.model_paths == [record.result["model_dir"]]
        assert chained.spec.project == {"name": "flies.slp", "id": "p1"}
        assert chained.spec.peak_threshold == 0.3
        assert chained.spec.tracker == "simple"
        # A run-less train job's chained job is grouped under its job id.
        assert chained.spec.run == {
            "id": job_id,
            "index": 0,
            "count": 1,
            "stage": "inference",
        }

        # job.result for the train job carries the chained id, emitted once.
        result_events = [
            e.data
            for e in await store.get_events_since(job_id)
            if e.topic == "job.result"
        ]
        assert result_events == [record.result]

        jobs = {
            j["job_id"]: j for j in (await methods.list_jobs({}, conn=None))["jobs"]
        }
        assert jobs[job_id]["post_inference"] is True
        assert jobs[chained_id]["post_inference"] is False
        assert jobs[job_id]["model_name"] == Path(record.result["model_dir"]).name
        assert jobs[chained_id]["model_name"] is None

    async def test_two_job_run_chains_once_after_the_second_completes(
        self, store, tmp_path
    ):
        run_id = "run-abc"
        spec0 = TrainJobSpec(
            config_path="/data/centroid.yaml",
            labels_path="/data/train.slp",
            model_types=["centroid"],
            run={"id": run_id, "index": 0, "count": 2},
            post_inference=[{"frame_filter": "suggested"}],
        )
        spec1 = TrainJobSpec(
            config_path="/data/centered_instance.yaml",
            labels_path="/data/train.slp",
            model_types=["centered_instance"],
            run={"id": run_id, "index": 1, "count": 2},
            post_inference=[{"frame_filter": "suggested"}],
        )
        methods = _make_chaining_methods(store, tmp_path)

        job0 = (await methods.submit({"spec": spec0.to_dict()}, conn=None))["job_id"]
        record0 = await _wait_for_terminal(methods, store, job0)
        assert record0.state == "completed"
        assert "chained_job_ids" not in record0.result  # sibling not done yet

        job1 = (await methods.submit({"spec": spec1.to_dict()}, conn=None))["job_id"]
        record1 = await _wait_for_terminal(methods, store, job1)
        assert record1.state == "completed"
        assert len(record1.result["chained_job_ids"]) == 1
        chained_id = record1.result["chained_job_ids"][0]
        chained = await _wait_for_terminal(methods, store, chained_id)

        # model_paths ordered by run.index, not completion order.
        record0_again = await store.get_job(job0)
        assert chained.spec.model_paths == [
            record0_again.result["model_dir"],
            record1.result["model_dir"],
        ]
        assert "chained_job_ids" not in record0_again.result  # dedupe: only job1 got it
        # Grouped under the training run's id, outside its train siblings.
        assert chained.spec.run == {
            "id": run_id,
            "index": 0,
            "count": 1,
            "stage": "inference",
        }

    async def test_failed_sibling_never_chains(self, store, tmp_path):
        run_id = "run-fail"
        ok_spec = TrainJobSpec(
            config_path="/data/centroid.yaml",
            labels_path="/data/train.slp",
            run={"id": run_id, "index": 0, "count": 2},
            post_inference=[{"frame_filter": "suggested"}],
        )
        fail_spec = TrainJobSpec(
            config_path="/data/centered_instance.yaml",
            labels_path="/data/train.slp",
            run={"id": run_id, "index": 1, "count": 2},
            post_inference=[{"frame_filter": "suggested"}],
        )
        methods = _make_chaining_methods(store, tmp_path)
        ok_id = (await methods.submit({"spec": ok_spec.to_dict()}, conn=None))["job_id"]
        await _wait_for_terminal(methods, store, ok_id)

        # A second `JobMethods` sharing the same store, so the sibling's
        # command can fail outright instead of going through the ckpt-aware
        # builder (which only knows how to succeed).
        fail_methods = _make_methods(
            store, tmp_path, [sys.executable, "-c", "import sys; sys.exit(1)"]
        )
        fail_id = (await fail_methods.submit({"spec": fail_spec.to_dict()}, conn=None))[
            "job_id"
        ]
        record = await _wait_for_terminal(fail_methods, store, fail_id)

        assert record.state == "failed"
        jobs = (await methods.list_jobs({}, conn=None))["jobs"]
        assert len(jobs) == 2  # no chained track job was ever created

    async def test_no_post_inference_never_chains(self, store, tmp_path):
        spec = TrainJobSpec(
            config_path="/data/centroid.yaml", labels_path="/data/t.slp"
        )
        methods = _make_chaining_methods(store, tmp_path)

        job_id = (await methods.submit({"spec": spec.to_dict()}, conn=None))["job_id"]
        record = await _wait_for_terminal(methods, store, job_id)

        assert "chained_job_ids" not in record.result
        jobs = (await methods.list_jobs({}, conn=None))["jobs"]
        assert len(jobs) == 1

    async def test_missing_sibling_model_dir_records_chain_error(self, store, tmp_path):
        # A plain command builder (no `_write_run`) never produces a model
        # dir, so `_train_outputs` reports model_dir=None for both siblings.
        run_id = "run-no-model"
        spec0 = TrainJobSpec(
            config_path="/data/c0.yaml",
            labels_path="/data/t.slp",
            run={"id": run_id, "index": 0, "count": 2},
        )
        spec1 = TrainJobSpec(
            config_path="/data/c1.yaml",
            labels_path="/data/t.slp",
            run={"id": run_id, "index": 1, "count": 2},
            post_inference=[{"frame_filter": "suggested"}],
        )
        methods = _make_methods(store, tmp_path, [sys.executable, "-c", "pass"])

        job0 = (await methods.submit({"spec": spec0.to_dict()}, conn=None))["job_id"]
        await _wait_for_terminal(methods, store, job0)
        job1 = (await methods.submit({"spec": spec1.to_dict()}, conn=None))["job_id"]
        record1 = await _wait_for_terminal(methods, store, job1)

        assert record1.state == "completed"  # chaining failure doesn't fail the job
        assert "missing model_dir" in record1.result["chain_error"]
        assert "chained_job_ids" not in record1.result
        jobs = (await methods.list_jobs({}, conn=None))["jobs"]
        assert len(jobs) == 2  # no chained job was created

    async def test_chained_submit_error_is_recorded_without_failing_the_job(
        self, store, tmp_path
    ):
        spec = TrainJobSpec(
            config_path="/data/centroid.yaml",
            labels_path="/data/train.slp",
            post_inference=[{"frame_filter": "not-a-real-filter"}],
        )
        methods = _make_chaining_methods(store, tmp_path)

        job_id = (await methods.submit({"spec": spec.to_dict()}, conn=None))["job_id"]
        record = await _wait_for_terminal(methods, store, job_id)

        assert record.state == "completed"
        assert record.result["chained_job_ids"] == []
        assert "frame_filter" in record.result["chain_error"]

    async def test_concurrent_finishers_chain_the_run_exactly_once(
        self, store, tmp_path
    ):
        """Simulates two siblings finishing at the same instant (today's
        queue happens to fully serialize execution, but `_chain_post_inference`
        doesn't rely on that) by invoking the chaining step directly from two
        tasks at once: only one may actually submit the chained job.
        """
        spec0 = TrainJobSpec(
            config_path="/c0.yaml",
            labels_path="/data/a.slp",
            run={"id": "run-race", "index": 0, "count": 2},
            post_inference=[{"peak_threshold": 0.5}],
        )
        spec1 = TrainJobSpec(
            config_path="/c1.yaml",
            labels_path="/data/a.slp",
            run={"id": "run-race", "index": 1, "count": 2},
        )
        methods = _make_methods(store, tmp_path, [sys.executable, "-c", "pass"])
        result0 = {"model_dir": "/models/m0", "labels_path": "/data/a.slp"}
        result1 = {"model_dir": "/models/m1", "labels_path": "/data/a.slp"}
        await store.create_job("job0", spec0)
        await store.create_job("job1", spec1)
        await store.update_state("job0", "completed", result=result0)
        await store.update_state("job1", "completed", result=result1)

        outcomes = await asyncio.gather(
            methods._chain_post_inference("job0", spec0, result0),
            methods._chain_post_inference("job1", spec1, result1),
        )

        chained = [o for o in outcomes if o is not None]
        assert len(chained) == 1  # the other call saw the dedupe marker and no-oped
        assert len(chained[0]["chained_job_ids"]) == 1


class TestReattachedEpochBackfill:
    async def test_epochs_logged_while_no_worker_ran_are_backfilled_once(
        self, store, tmp_path
    ):
        from sleap_rtc.jobs.process import reattach_all

        go = tmp_path / "go"
        script = lambda ckpt_dir: _write_run(ckpt_dir, epochs=(0, 1), wait_for=go)
        spec = TrainJobSpec(config_path="/data/c.yaml")
        first = _make_ckpt_methods(store, tmp_path, script)
        job_id = (await first.submit({"spec": spec.to_dict()}, conn=None))["job_id"]
        deadline = time.monotonic() + 10
        while not any(
            e.topic == "job.log" for e in await store.get_events_since(job_id)
        ):
            assert time.monotonic() < deadline
            await asyncio.sleep(0.05)
        task = first._tasks[job_id]
        task.cancel()  # worker goes away
        await asyncio.gather(task, return_exceptions=True)
        go.touch()  # epochs get logged while no worker is running

        second = _make_ckpt_methods(store, tmp_path, script)
        second.resume_reattached(await reattach_all(store))
        record = await _wait_for_terminal(second, store, job_id)

        assert record.state == "completed"
        epochs = [
            e.data
            for e in await store.get_events_since(job_id)
            if e.topic == "job.epoch"
        ]
        assert [e["epoch"] for e in epochs] == [0, 1]
        assert epochs[0]["train_loss"] == 0.5


class TestQueuePosition:
    async def test_waiting_jobs_report_their_place_in_line(self, store, spec, tmp_path):
        go = tmp_path / "go"
        cmd = [
            sys.executable,
            "-c",
            f"import os, time\nwhile not os.path.exists({str(go)!r}): time.sleep(0.05)",
        ]
        methods = _make_methods(store, tmp_path, cmd)
        ids = [
            (await methods.submit({"spec": spec.to_dict()}, conn=None))["job_id"]
            for _ in range(3)
        ]
        await _wait_for_state(store, ids[0], "running")
        await asyncio.sleep(0.1)

        assert [methods.queue_position(i) for i in ids] == [None, 1, 2]

        go.touch()
        for job_id in ids:
            await _wait_for_terminal(methods, store, job_id)
        assert [methods.queue_position(i) for i in ids] == [None, None, None]


class TestJobSummaries:
    """jobs.list / jobs.status describe a job well enough to list and re-run it."""

    async def test_list_and_status_describe_the_job_without_inline_labels(
        self, store, tmp_path
    ):
        import base64

        run = {"id": "run-abc123", "index": 0, "count": 2}
        spec = TrainJobSpec(
            config_contents=["model: centroid"],
            model_types=["centroid"],
            labels_content=base64.b64encode(b"slp").decode(),
            project={"name": "flies.slp", "id": "p1"},
            run=run,
        )
        methods = _make_methods(store, tmp_path, [sys.executable, "-c", "pass"])
        job_id = (await methods.submit({"spec": spec.to_dict()}, conn=None))["job_id"]
        await _wait_for_terminal(methods, store, job_id)

        (listed,) = (await methods.list_jobs({}, conn=None))["jobs"]
        assert listed["job_id"] == job_id
        assert listed["kind"] == "train"
        assert listed["model_types"] == ["centroid"]
        assert listed["project"] == {"name": "flies.slp", "id": "p1"}
        assert listed["run"] == run
        assert listed["queue_position"] is None
        assert "spec" not in listed
        assert "labels_content" not in json.dumps(listed)

        status = await methods.status({"job_id": job_id}, conn=None)
        assert status["spec"]["config_contents"] == ["model: centroid"]
        assert "labels_content" not in status["spec"]
        assert status["kind"] == "train"
        assert status["project"] == {"name": "flies.slp", "id": "p1"}
        assert status["run"] == run

    async def test_track_job_lists_its_data_path(self, store, tmp_path):
        spec = TrackJobSpec(data_path="/data/v.slp", model_paths=["/m"])
        methods = _make_methods(store, tmp_path, [sys.executable, "-c", "pass"])
        job_id = (await methods.submit({"spec": spec.to_dict()}, conn=None))["job_id"]
        await _wait_for_terminal(methods, store, job_id)

        (listed,) = (await methods.list_jobs({}, conn=None))["jobs"]
        assert listed["kind"] == "track"
        assert listed["labels_path"] == "/data/v.slp"

    async def test_failed_job_lists_its_error(self, store, spec, tmp_path):
        methods = _make_methods(
            store, tmp_path, [sys.executable, "-c", "raise SystemExit(4)"]
        )
        job_id = (await methods.submit({"spec": spec.to_dict()}, conn=None))["job_id"]
        await _wait_for_terminal(methods, store, job_id)

        (listed,) = (await methods.list_jobs({}, conn=None))["jobs"]
        assert listed["state"] == "failed"
        assert listed["error"] == "exit code 4"


class TestPreflight:
    """A job whose input files are missing fails before anything is spawned."""

    async def test_missing_labels_fail_fast_without_spawning(self, store, tmp_path):
        marker = tmp_path / "spawned"
        config = tmp_path / "c.yaml"
        config.write_text("x: 1")
        spec = TrainJobSpec(config_paths=[str(config)], labels_path="/nope/labels.slp")
        methods = _make_methods(
            store, tmp_path, [sys.executable, "-c", f"open({str(marker)!r}, 'w')"]
        )
        job_id = (await methods.submit({"spec": spec.to_dict()}, conn=None))["job_id"]
        record = await _wait_for_terminal(methods, store, job_id)

        assert record.state == "failed"
        assert record.error == "not found on worker: /nope/labels.slp"
        assert not marker.exists()
        status = [
            e.data
            for e in await store.get_events_since(job_id)
            if e.topic == "job.status"
        ][-1]
        assert status["missing"] == ["/nope/labels.slp"]

    async def test_missing_model_fails_a_track_job(self, store, tmp_path):
        data = tmp_path / "v.slp"
        data.write_bytes(b"x")
        spec = TrackJobSpec(data_path=str(data), model_paths=["/nope/model"])
        methods = _make_methods(store, tmp_path, [sys.executable, "-c", "pass"])
        job_id = (await methods.submit({"spec": spec.to_dict()}, conn=None))["job_id"]
        record = await _wait_for_terminal(methods, store, job_id)

        assert record.state == "failed"
        assert "/nope/model" in record.error


class TestCancelBeforeSpawn:
    """Cancelling a job that hasn't started its process yet must stop it from
    ever starting (previously it was marked failed and then ran anyway)."""

    async def test_cancelling_a_queued_job_means_it_never_runs(
        self, store, spec, tmp_path
    ):
        go = tmp_path / "go"
        marker = tmp_path / "second-ran"
        blocker = [
            sys.executable,
            "-c",
            f"import os, time\nwhile not os.path.exists({str(go)!r}): time.sleep(0.05)",
        ]
        methods = _make_methods(store, tmp_path, blocker)
        first = (await methods.submit({"spec": spec.to_dict()}, conn=None))["job_id"]
        await _wait_for_state(store, first, "running")
        methods._builder = _FakeCommandBuilder(
            [sys.executable, "-c", f"open({str(marker)!r}, 'w')"]
        )
        second = (await methods.submit({"spec": spec.to_dict()}, conn=None))["job_id"]
        await asyncio.sleep(0.1)

        await methods.cancel({"job_id": second}, conn=None)
        go.touch()
        await _wait_for_terminal(methods, store, first)
        record = await _wait_for_terminal(methods, store, second)

        assert record.state == "canceled"
        assert not marker.exists()

    async def test_cancel_right_after_submit_never_spawns(self, store, spec, tmp_path):
        marker = tmp_path / "ran"
        methods = _make_methods(
            store, tmp_path, [sys.executable, "-c", f"open({str(marker)!r}, 'w')"]
        )
        job_id = (await methods.submit({"spec": spec.to_dict()}, conn=None))["job_id"]
        await methods.cancel({"job_id": job_id, "mode": "stop"}, conn=None)

        record = await _wait_for_terminal(methods, store, job_id)
        assert record.state == "canceled"
        assert not marker.exists()
