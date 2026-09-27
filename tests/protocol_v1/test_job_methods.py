"""Integration tests for the jobs.* / fs.* protocol v1 method handlers.

Uses a real JobStore (tmp-file SQLite), a real JobQueue, and real
subprocesses — swapping in a fake `CommandBuilder` that returns a plain
Python invocation instead of a real ``sleap-nn`` command, so the full
submit → spawn → tail-log → complete flow is exercised for real without
requiring sleap-nn to be installed.
"""

import asyncio
import hashlib
import sys
import time

import pytest

from sleap_rtc.jobs.queue import JobQueue
from sleap_rtc.jobs.spec import TrackJobSpec, TrainJobSpec
from sleap_rtc.jobs.store import JobStore
from sleap_rtc.protocol_v1.blobs import BlobIndex
from sleap_rtc.protocol_v1.errors import JOB_NOT_FOUND, ProtocolError
from sleap_rtc.protocol_v1.job_methods import JobMethods
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
    def __init__(self, mounts, listing):
        self._mounts = mounts
        self._listing = listing

    def get_mounts(self):
        return self._mounts

    def list_directory(self, path, offset=0):
        return self._listing


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


@pytest.fixture
def spec():
    return TrainJobSpec(config_path="/data/centroid.yaml")


@pytest.fixture
async def store(tmp_path):
    async with JobStore(tmp_path / "jobs.sqlite") as store:
        yield store


def _make_methods(store, tmp_path, cmd, file_manager=None, blob_index=None):
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

        assert record.result == {"blobs": {}}

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
