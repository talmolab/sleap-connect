"""Tests for detached subprocess spawning, liveness checks, and reattach.

Uses real subprocesses (short-lived Python interpreter invocations) rather
than mocks — the whole point of this module is real OS process-group and
start-time semantics, which mocks can't meaningfully verify.
"""

import asyncio
import sys

import pytest

from sleap_rtc.jobs.process import is_alive, reattach_all, spawn_detached
from sleap_rtc.jobs.spec import TrainJobSpec
from sleap_rtc.jobs.store import JobStore


@pytest.fixture
def spec():
    """A minimal valid TrainJobSpec for use in tests."""
    return TrainJobSpec(config_path="/data/centroid.yaml")


@pytest.fixture
async def store(tmp_path):
    """A JobStore backed by a temp-file SQLite database."""
    async with JobStore(tmp_path / "jobs.sqlite") as store:
        yield store


class TestSpawnDetached:
    """Tests for spawn_detached."""

    async def test_records_process_info_on_the_job(self, store, spec, tmp_path):
        await store.create_job("job-1", spec)
        log_path = tmp_path / "job-1.log"

        process = await spawn_detached(
            [sys.executable, "-c", "print('hello')"],
            job_id="job-1",
            store=store,
            log_path=log_path,
        )
        await process.wait()

        record = await store.get_job("job-1")
        assert record.pid == process.pid
        assert record.process_started_at is not None
        assert record.log_path == str(log_path)

    async def test_subprocess_output_lands_in_the_log_file(self, store, spec, tmp_path):
        await store.create_job("job-1", spec)
        log_path = tmp_path / "job-1.log"

        process = await spawn_detached(
            [sys.executable, "-c", "print('hello from subprocess')"],
            job_id="job-1",
            store=store,
            log_path=log_path,
        )
        await process.wait()

        assert "hello from subprocess" in log_path.read_text()

    async def test_creates_parent_directory_for_log_path(self, store, spec, tmp_path):
        await store.create_job("job-1", spec)
        log_path = tmp_path / "nested" / "dir" / "job-1.log"

        process = await spawn_detached(
            [sys.executable, "-c", "pass"],
            job_id="job-1",
            store=store,
            log_path=log_path,
        )
        await process.wait()

        assert log_path.exists()


class TestIsAlive:
    """Tests for is_alive."""

    async def test_true_for_a_running_process(self):
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(2)"
        )
        try:
            import psutil

            started_at = psutil.Process(process.pid).create_time()
            assert is_alive(process.pid, started_at) is True
        finally:
            process.kill()
            await process.wait()

    async def test_false_after_the_process_exits(self):
        process = await asyncio.create_subprocess_exec(sys.executable, "-c", "pass")
        import psutil

        started_at = psutil.Process(process.pid).create_time()
        await process.wait()

        assert is_alive(process.pid, started_at) is False

    async def test_false_when_start_time_does_not_match(self):
        """Guards against PID reuse: same PID, wrong recorded start time."""
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(2)"
        )
        try:
            wrong_started_at = 1.0  # clearly not this process's real start time
            assert is_alive(process.pid, wrong_started_at) is False
        finally:
            process.kill()
            await process.wait()


class TestReattachAll:
    """Tests for reattach_all."""

    async def test_leaves_a_genuinely_running_job_running(self, store, spec, tmp_path):
        await store.create_job("job-1", spec)
        await store.update_state("job-1", "running")
        process = await spawn_detached(
            [sys.executable, "-c", "import time; time.sleep(2)"],
            job_id="job-1",
            store=store,
            log_path=tmp_path / "job-1.log",
        )
        try:
            outcomes = await reattach_all(store)

            assert outcomes == {"job-1": "reattached"}
            record = await store.get_job("job-1")
            assert record.state == "running"
        finally:
            process.kill()
            await process.wait()

    async def test_marks_a_dead_job_failed(self, store, spec, tmp_path):
        await store.create_job("job-1", spec)
        await store.update_state("job-1", "running")
        process = await spawn_detached(
            [sys.executable, "-c", "pass"],
            job_id="job-1",
            store=store,
            log_path=tmp_path / "job-1.log",
        )
        await process.wait()  # let it actually exit before reattaching

        outcomes = await reattach_all(store)

        assert outcomes == {"job-1": "marked_failed"}
        record = await store.get_job("job-1")
        assert record.state == "failed"
        assert "worker restarted" in record.error

    async def test_ignores_jobs_that_are_not_running(self, store, spec):
        await store.create_job("job-1", spec)
        await store.update_state("job-1", "completed", result={"ok": True})
        await store.create_job("job-2", spec)  # stays "queued"

        outcomes = await reattach_all(store)

        assert outcomes == {}
        assert (await store.get_job("job-1")).state == "completed"
        assert (await store.get_job("job-2")).state == "queued"

    async def test_treats_a_job_with_no_recorded_process_as_dead(self, store, spec):
        # A job marked "running" that never got set_process_info called on
        # it (e.g. the worker crashed between create_job and spawn_detached)
        # has no pid to check — it must be treated as lost, not skipped.
        await store.create_job("job-1", spec)
        await store.update_state("job-1", "running")

        outcomes = await reattach_all(store)

        assert outcomes == {"job-1": "marked_failed"}
