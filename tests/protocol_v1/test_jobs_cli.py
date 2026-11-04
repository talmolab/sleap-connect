"""Tests for `sleap-connect jobs` (list, queue, show, logs, stop, cancel).

The read commands run against a seeded job store with no worker running —
that they work offline is the point. `stop`/`cancel` run against a real
worker (`start_worker_server`) with a fake command builder, as in
test_job_methods.py, and a pidfile pointing at it, as `serve` writes.
"""

import asyncio
import json
import sqlite3
import sys
import time

import pytest
from click.testing import CliRunner

from sleap_connect.jobs.spec import TrackJobSpec, TrainJobSpec
from sleap_connect.jobs.store import JobStore
from sleap_connect.protocol_v1 import daemon
from sleap_connect.protocol_v1 import job_methods as job_methods_module
from sleap_connect.protocol_v1.jobs_cli import jobs
from sleap_connect.protocol_v1.runner import (
    job_log_dir,
    job_store_path,
    open_trust_store,
    start_worker_server,
)

from .test_job_methods import (
    _FakeCommandBuilder,
    _wait_for_spawned,
    _wait_for_terminal,
)

DEVICE = "device-node-id-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def _invoke(data_dir, *args):
    return CliRunner().invoke(jobs, [*args, "--data-dir", str(data_dir)])


async def _seed_async(data_dir):
    log_dir = job_log_dir(data_dir)
    log_dir.mkdir(parents=True)
    async with JobStore(job_store_path(data_dir)) as store:
        # Completed training job, with a log full of progress-bar updates.
        await store.create_job(
            "job_done1",
            TrainJobSpec(
                config_path="/c.yaml",
                labels_path="/data/proj/labels.slp",
                model_types=["centroid"],
                labels_content="QUJD",
            ),
            submitted_by=DEVICE,
        )
        log = log_dir / "job_done1.log"
        log.write_text("start\nepoch 1\repoch 1 50%\repoch 1 100%\nsaved\ndone\n")
        await store.set_process_info("job_done1", 1, 1000.0, str(log))
        await store.update_state(
            "job_done1",
            "completed",
            result={"blobs": {}, "model_dir": "/data/proj/models/260922.centroid"},
        )
        # Failed inference job, from a device that has since unpaired.
        await store.create_job(
            "job_fail2",
            TrackJobSpec(
                data_path="/data/proj/video.mp4",
                model_paths=["/m/260922.centroid", "/m/260922.instance"],
            ),
            submitted_by="gone-device-zzzz",
        )
        await store.update_state("job_fail2", "failed", error="exit code 1")
        # Running, then two waiting.
        for job_id in ("job_run3", "job_wait4", "job_wait5"):
            await store.create_job(
                job_id,
                TrainJobSpec(
                    config_path="/c.yaml",
                    labels_path="/data/proj/labels.slp",
                    model_types=["centered_instance"],
                ),
                submitted_by=DEVICE,
            )
        await store.set_process_info("job_run3", 1, 2000.0, str(log_dir / "x.log"))
        await store.update_state("job_run3", "running")
    # Deterministic creation order (the store stamps wall-clock time).
    with sqlite3.connect(job_store_path(data_dir)) as db:
        for i, job_id in enumerate(
            ("job_done1", "job_fail2", "job_run3", "job_wait4", "job_wait5")
        ):
            db.execute(
                "UPDATE jobs SET created_at = ? WHERE job_id = ?",
                (time.time() - 1000 + i, job_id),
            )
    open_trust_store(data_dir).add_trusted(DEVICE, {"device_name": "Rig 4"})


@pytest.fixture
def seeded(tmp_path):
    asyncio.run(_seed_async(tmp_path))
    return tmp_path


class TestJobsList:
    def test_lists_jobs_newest_first_with_device_names(self, seeded):
        result = _invoke(seeded)

        assert result.exit_code == 0, result.output
        lines = result.output.splitlines()
        assert lines[0].split() == [
            "ID",
            "Kind",
            "Model",
            "State",
            "Device",
            "Started",
            "Labels",
        ]
        ids = [line.split()[0] for line in lines[1:] if line.startswith("job_")]
        assert ids == ["job_wait5", "job_wait4", "job_run3", "job_fail2", "job_done1"]
        done = next(line for line in lines if line.startswith("job_done1"))
        for text in ("train", "centroid", "completed", "Rig 4", "labels.slp"):
            assert text in done
        failed = next(line for line in lines if line.startswith("job_fail2"))
        assert "260922.centroid, 260922.instance" in failed
        assert "unpaired (gone-dev)" in failed

    def test_limit_says_how_many_are_hidden(self, seeded):
        result = _invoke(seeded, "-n", "2")

        assert "job_run3" not in result.output
        assert "Showing the 2 newest of 5 jobs" in result.output

    def test_json(self, seeded):
        result = _invoke(seeded, "--json")

        listing = json.loads(result.output)
        assert [j["job_id"] for j in listing][-1] == "job_done1"
        done = listing[-1]
        assert done["kind"] == "train"
        assert done["state"] == "completed"
        assert done["device"] == "Rig 4"
        assert done["submitted_by"] == DEVICE
        assert done["model_dir"] == "/data/proj/models/260922.centroid"
        assert "labels_content" not in json.dumps(listing)

    def test_no_jobs_yet_creates_nothing(self, tmp_path):
        result = _invoke(tmp_path)

        assert result.exit_code == 0
        assert "No jobs yet" in result.output
        assert not job_store_path(tmp_path).exists()


class TestJobsQueue:
    def test_running_then_waiting_in_run_order(self, seeded):
        result = _invoke(seeded, "queue")

        assert result.exit_code == 0, result.output
        rows = [line.split() for line in result.output.splitlines()[1:4]]
        assert [(r[0], r[1]) for r in rows] == [
            ("▶", "job_run3"),
            ("1", "job_wait4"),
            ("2", "job_wait5"),
        ]
        assert "worker isn't running" in result.output

    def test_json_has_queue_positions(self, seeded):
        result = _invoke(seeded, "queue", "--json")

        queue = json.loads(result.output)
        assert [(j["job_id"], j["queue_position"]) for j in queue] == [
            ("job_run3", None),
            ("job_wait4", 1),
            ("job_wait5", 2),
        ]

    def test_empty(self, tmp_path):
        assert "Nothing running or waiting" in _invoke(tmp_path, "queue").output


class TestJobsShow:
    def test_shows_output_folder_timing_and_log(self, seeded):
        result = _invoke(seeded, "show", "job_done")

        assert result.exit_code == 0, result.output
        out = result.output
        assert out.splitlines()[0] == "job_done1"
        assert "/data/proj/models/260922.centroid" in out
        assert "Rig 4" in out
        assert "Took:" in out
        assert "job_done1.log" in out

    def test_shows_the_error(self, seeded):
        assert "exit code 1" in _invoke(seeded, "show", "job_fail2").output

    def test_ambiguous_prefix(self, seeded):
        result = _invoke(seeded, "show", "job_wait")

        assert result.exit_code == 1
        assert "matches 2 jobs" in result.output

    def test_unknown(self, seeded):
        result = _invoke(seeded, "show", "nope")

        assert result.exit_code == 1
        assert 'No job matches "nope"' in result.output

    def test_json_has_the_spec_without_inline_labels(self, seeded):
        shown = json.loads(_invoke(seeded, "show", "job_done1", "--json").output)

        assert shown["spec"]["model_types"] == ["centroid"]
        assert "labels_content" not in shown["spec"]
        assert shown["log_path"].endswith("job_done1.log")
        assert shown["result"]["model_dir"] == "/data/proj/models/260922.centroid"


class TestJobsLogs:
    def test_tail_collapses_progress_bars(self, seeded):
        result = _invoke(seeded, "logs", "job_done1", "-n", "3")

        assert result.output.splitlines() == ["epoch 1 100%", "saved", "done"]

    def test_whole_log(self, seeded):
        result = _invoke(seeded, "logs", "job_done1", "-n", "0")

        assert result.output.splitlines()[0] == "start"

    def test_follow_a_finished_job_prints_and_exits(self, seeded):
        result = _invoke(seeded, "logs", "job_done1", "-f")

        assert result.exit_code == 0
        assert "done" in result.output
        assert "Job job_done1 completed." in result.output

    def test_queued_job_has_no_log_yet(self, seeded):
        result = _invoke(seeded, "logs", "job_wait4")

        assert result.exit_code == 0
        assert "hasn't started yet" in result.output


class TestStopCancelWithoutWorker:
    @pytest.mark.parametrize("args", [("stop", "job_run3"), ("cancel", "job_run3")])
    def test_says_the_worker_isnt_running(self, seeded, args):
        result = _invoke(seeded, *args)

        assert result.exit_code == 1
        assert "No worker is running" in result.output
        assert "sleap-connect serve" in result.output

    def test_already_finished_needs_no_worker(self, seeded):
        result = _invoke(seeded, "cancel", "job_done1")

        assert result.exit_code == 0
        assert "already completed" in result.output

    def test_needs_ids_or_queued(self, seeded):
        assert _invoke(seeded, "cancel").exit_code == 2
        assert _invoke(seeded, "cancel", "job_run3", "--queued").exit_code == 2


@pytest.fixture
async def live_worker(tmp_path, monkeypatch):
    """A real worker with a long-running fake job command, recorded in a
    pidfile the way `serve` records itself."""
    monkeypatch.setattr(job_methods_module, "_missing_inputs", lambda spec: [])
    worker = await start_worker_server(host="127.0.0.1", port=0, data_dir=tmp_path)
    worker.job_methods._builder = _FakeCommandBuilder(
        [sys.executable, "-c", "import time; time.sleep(30)"]
    )
    port = worker.ws_server.sockets[0].getsockname()[1]
    daemon.acquire_pidfile(tmp_path, {"host": "0.0.0.0", "port": port})
    try:
        yield worker
    finally:
        daemon.release_pidfile(tmp_path)
        await worker.close()


async def _invoke_async(data_dir, *args):
    # The CLI runs its own event loop; keep it off the worker's.
    return await asyncio.to_thread(_invoke, data_dir, *args)


class TestStopCancelThroughWorker:
    async def test_cancel_running_and_queued(self, live_worker, tmp_path):
        methods = live_worker.job_methods
        spec = TrainJobSpec(config_path="/c.yaml", model_types=["centroid"])
        first = await methods._submit_spec(spec)
        await _wait_for_spawned(live_worker.store, first)
        second = await methods._submit_spec(spec)
        third = await methods._submit_spec(spec)

        result = await _invoke_async(tmp_path, "cancel", "--queued", "-y")

        assert result.exit_code == 0, result.output
        assert f"Job {second} canceled." in result.output
        assert f"Job {third} canceled." in result.output
        assert (await live_worker.store.get_job(first)).state == "running"

        result = await _invoke_async(tmp_path, "cancel", first)

        assert result.exit_code == 0, result.output
        record = await _wait_for_terminal(methods, live_worker.store, first)
        assert record.state == "canceled"

    async def test_stop(self, live_worker, tmp_path):
        methods = live_worker.job_methods
        job_id = await methods._submit_spec(TrainJobSpec(config_path="/c.yaml"))
        await _wait_for_spawned(live_worker.store, job_id)

        result = await _invoke_async(tmp_path, "stop", job_id)

        assert result.exit_code == 0, result.output
        assert job_id in result.output
        # No ZMQ controller in this fake job, so it's signalled instead.
        record = await _wait_for_terminal(methods, live_worker.store, job_id, 15)
        assert record.error == "stopped early by request"

    async def test_cancel_queued_asks_first(self, live_worker, tmp_path):
        methods = live_worker.job_methods
        spec = TrainJobSpec(config_path="/c.yaml")
        first = await methods._submit_spec(spec)
        await _wait_for_spawned(live_worker.store, first)
        second = await methods._submit_spec(spec)

        result = await asyncio.to_thread(
            CliRunner().invoke,
            jobs,
            ["cancel", "--queued", "--data-dir", str(tmp_path)],
            input="n\n",
        )

        assert "Cancel 1 waiting job?" in result.output
        assert "Nothing canceled." in result.output
        assert (await live_worker.store.get_job(second)).state == "queued"
