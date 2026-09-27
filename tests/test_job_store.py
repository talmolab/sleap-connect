"""Tests for the durable SQLite job store."""

import asyncio

import pytest

from sleap_rtc.jobs.spec import TrainJobSpec
from sleap_rtc.jobs.store import JobStore, JobStoreError


@pytest.fixture
def spec():
    """A minimal valid TrainJobSpec for use in tests."""
    return TrainJobSpec(config_path="/data/centroid.yaml", max_epochs=10)


@pytest.fixture
async def store(tmp_path):
    """A JobStore backed by a temp-file SQLite database, connected and closed."""
    db_path = tmp_path / "jobs.sqlite"
    async with JobStore(db_path) as store:
        yield store


class TestCreateAndGetJob:
    """Tests for create_job / get_job / list_jobs."""

    async def test_create_job_returns_queued_record(self, store, spec):
        record = await store.create_job("job-1", spec)

        assert record.job_id == "job-1"
        assert record.state == "queued"
        assert record.spec.config_path == "/data/centroid.yaml"
        assert record.spec.max_epochs == 10
        assert record.result is None
        assert record.error is None

    async def test_get_job_roundtrips_spec(self, store, spec):
        await store.create_job("job-1", spec)

        fetched = await store.get_job("job-1")

        assert fetched is not None
        assert fetched.job_id == "job-1"
        assert isinstance(fetched.spec, TrainJobSpec)
        assert fetched.spec.config_path == "/data/centroid.yaml"

    async def test_get_job_returns_none_for_unknown_id(self, store):
        assert await store.get_job("does-not-exist") is None

    async def test_create_job_rejects_duplicate_id(self, store, spec):
        await store.create_job("job-1", spec)

        with pytest.raises(JobStoreError):
            await store.create_job("job-1", spec)

    async def test_list_jobs_orders_newest_first(self, store, spec):
        await store.create_job("job-1", spec)
        await store.create_job("job-2", spec)

        jobs = await store.list_jobs()

        assert [j.job_id for j in jobs] == ["job-2", "job-1"]


class TestUpdateState:
    """Tests for update_state."""

    async def test_update_state_transitions_job(self, store, spec):
        await store.create_job("job-1", spec)

        updated = await store.update_state("job-1", "running")

        assert updated.state == "running"
        assert updated.updated_at >= updated.created_at

    async def test_update_state_stores_terminal_result(self, store, spec):
        await store.create_job("job-1", spec)

        updated = await store.update_state(
            "job-1", "completed", result={"predictions": {"sha256": "abc", "size": 42}}
        )

        assert updated.state == "completed"
        assert updated.result == {"predictions": {"sha256": "abc", "size": 42}}

    async def test_update_state_stores_terminal_error(self, store, spec):
        await store.create_job("job-1", spec)

        updated = await store.update_state("job-1", "failed", error="OOM")

        assert updated.state == "failed"
        assert updated.error == "OOM"

    async def test_update_state_rejects_invalid_state(self, store, spec):
        await store.create_job("job-1", spec)

        with pytest.raises(JobStoreError):
            await store.update_state("job-1", "not-a-real-state")

    async def test_update_state_rejects_unknown_job(self, store):
        with pytest.raises(JobStoreError):
            await store.update_state("does-not-exist", "running")


class TestEvents:
    """Tests for append_event / get_events_since."""

    async def test_append_event_assigns_increasing_seq(self, store, spec):
        await store.create_job("job-1", spec)

        seq1 = await store.append_event("job-1", "job.log", {"line": "starting"})
        seq2 = await store.append_event("job-1", "job.log", {"line": "epoch 1"})

        assert seq1 == 1
        assert seq2 == 2

    async def test_append_event_rejects_unknown_job(self, store):
        with pytest.raises(JobStoreError):
            await store.append_event("does-not-exist", "job.log", {"line": "x"})

    async def test_concurrent_appends_to_the_same_job_get_distinct_seqs(
        self, store, spec
    ):
        # Regression test: seq assignment is a read-then-write
        # (SELECT MAX then INSERT), which isn't atomic at the SQL level.
        # Fire many concurrent appends for the same job and confirm every
        # one lands with a unique, gapless seq — no IntegrityError, no lost
        # writes from two calls computing the same next_seq.
        await store.create_job("job-1", spec)

        seqs = await asyncio.gather(
            *[
                store.append_event("job-1", "job.log", {"line": f"line-{i}"})
                for i in range(20)
            ]
        )

        assert sorted(seqs) == list(range(1, 21))

    async def test_get_events_since_zero_returns_full_history(self, store, spec):
        await store.create_job("job-1", spec)
        await store.append_event("job-1", "job.log", {"line": "a"})
        await store.append_event("job-1", "job.log", {"line": "b"})

        events = await store.get_events_since("job-1", since_seq=0)

        assert [e.data["line"] for e in events] == ["a", "b"]
        assert [e.seq for e in events] == [1, 2]

    async def test_get_events_since_n_returns_only_newer(self, store, spec):
        await store.create_job("job-1", spec)
        await store.append_event("job-1", "job.log", {"line": "a"})
        await store.append_event("job-1", "job.log", {"line": "b"})
        await store.append_event("job-1", "job.log", {"line": "c"})

        events = await store.get_events_since("job-1", since_seq=1)

        assert [e.data["line"] for e in events] == ["b", "c"]

    async def test_events_are_scoped_per_job(self, store, spec):
        await store.create_job("job-1", spec)
        await store.create_job("job-2", spec)
        await store.append_event("job-1", "job.log", {"line": "job1-a"})
        await store.append_event("job-2", "job.log", {"line": "job2-a"})

        job1_events = await store.get_events_since("job-1")
        job2_events = await store.get_events_since("job-2")

        assert [e.data["line"] for e in job1_events] == ["job1-a"]
        assert [e.data["line"] for e in job2_events] == ["job2-a"]


class TestPersistence:
    """Tests that state survives across separate JobStore instances (reattach)."""

    async def test_job_and_events_survive_reconnect(self, tmp_path, spec):
        db_path = tmp_path / "jobs.sqlite"

        async with JobStore(db_path) as store:
            await store.create_job("job-1", spec)
            await store.update_state("job-1", "running")
            await store.append_event("job-1", "job.log", {"line": "hello"})

        # Simulate a worker restart: brand new JobStore instance, same file.
        async with JobStore(db_path) as reopened:
            record = await reopened.get_job("job-1")
            events = await reopened.get_events_since("job-1")

        assert record is not None
        assert record.state == "running"
        assert [e.data["line"] for e in events] == ["hello"]
