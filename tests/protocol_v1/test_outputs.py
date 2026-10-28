"""Tests for SLEAP-style job outputs (folder access, stage 3)."""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

from sleap_rtc.jobs.queue import JobQueue
from sleap_rtc.jobs.spec import TrainJobSpec
from sleap_rtc.jobs.store import JobStore
from sleap_rtc.protocol_v1 import job_methods as job_methods_module
from sleap_rtc.protocol_v1.job_methods import JobMethods
from sleap_rtc.protocol_v1.outputs import (
    bare_run_name,
    predictions_path,
    read_model_dir,
    run_name_for,
    train_output_root,
    write_outputs_record,
)
from sleap_rtc.protocol_v1.server import ProtocolV1Server

NOW = datetime(2026, 10, 28, 10, 30, 0)


def Train(**kw) -> TrainJobSpec:
    kw.setdefault("config_content", "{}")
    return TrainJobSpec(**kw)


class TestOutputRoot:
    def test_labels_on_the_worker_use_their_folder(self, tmp_path):
        labels = tmp_path / "labels.slp"
        assert train_output_root(Train(labels_path=str(labels))) == tmp_path.resolve()

    def test_uploaded_labels_stay_in_the_job_folder(self, tmp_path):
        spec = Train(labels_path=str(tmp_path / "labels.slp"), labels_content="AAAA")
        assert train_output_root(spec) is None

    @pytest.mark.skipif(
        sys.platform == "win32" or os.geteuid() == 0,
        reason="needs a non-root POSIX user",
    )
    def test_a_read_only_folder_stays_in_the_job_folder(self, tmp_path):
        ro = tmp_path / "ro"
        ro.mkdir()
        ro.chmod(0o500)
        try:
            assert train_output_root(Train(labels_path=str(ro / "labels.slp"))) is None
        finally:
            ro.chmod(0o700)


class TestRunName:
    def test_default_is_sleaps_timestamp_and_model(self, tmp_path):
        assert (
            run_name_for(Train(model_types=["centroid"]), tmp_path, NOW)
            == "261028_103000.centroid"
        )

    def test_client_name_is_kept_quotes_and_all(self, tmp_path):
        assert run_name_for(Train(run_name="'my run'"), tmp_path) == "'my run'"
        assert bare_run_name("'my run'") == "my run"

    def test_a_taken_name_gets_a_suffix(self, tmp_path):
        (tmp_path / "centroid").mkdir()
        (tmp_path / "centroid_2").mkdir()
        assert run_name_for(Train(run_name="centroid"), tmp_path) == "centroid_3"
        (tmp_path / "my run").mkdir()
        assert run_name_for(Train(run_name="'my run'"), tmp_path) == "'my run_2'"


class TestPredictionsPath:
    def test_labels_file_gets_a_predictions_folder(self):
        assert predictions_path("/p/labels.v002.slp", NOW) == Path(
            "/p/predictions/labels.v002.slp.261028_103000.predictions.slp"
        )

    def test_video_gets_a_file_next_to_it(self):
        assert predictions_path("/p/mice.mp4", NOW) == Path(
            "/p/mice.mp4.261028_103000.predictions.slp"
        )


class TestOutputsRecord:
    def test_round_trip_once_trained(self, tmp_path):
        model = tmp_path / "models" / "run"
        write_outputs_record(tmp_path / "job", model)
        assert read_model_dir(tmp_path / "job") is None  # not trained yet
        model.mkdir(parents=True)
        (model / "training_config.yaml").write_text("{}")
        assert read_model_dir(tmp_path / "job") == model


class _TrainIntoCkptDir:
    """A fake sleap-nn train: creates <ckpt_dir>/<run name>/training_config.yaml."""

    def __init__(self):
        self.specs = []

    def build_command(self, spec, zmq_ports=None):
        self.specs.append((spec.ckpt_dir, spec.run_name))
        run_dir = Path(spec.ckpt_dir) / bare_run_name(spec.run_name)
        code = (
            f"import pathlib; d = pathlib.Path({str(run_dir)!r}); "
            "d.mkdir(parents=True); (d / 'training_config.yaml').write_text('{}')"
        )
        return [sys.executable, "-c", code]


async def _run(methods, store, spec):
    result = await methods.submit(
        {"spec": {"type": "train", **spec.to_dict()}}, conn=None
    )
    job_id = result["job_id"]
    await methods.wait_for_job(job_id, timeout=30)
    return job_id, await store.get_job(job_id)


class TestTrainJobLayout:
    async def test_models_land_next_to_the_labels_each_in_its_own_folder(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(job_methods_module, "_missing_inputs", lambda spec: [])
        project = tmp_path / "project"
        project.mkdir()
        labels = project / "labels.slp"
        builder = _TrainIntoCkptDir()
        async with JobStore(tmp_path / "jobs.sqlite") as store:
            methods = JobMethods(
                ProtocolV1Server(node_id="n"),
                store,
                JobQueue(max_concurrent=1),
                log_dir=tmp_path / "data" / "logs",
                command_builder=builder,
            )
            spec = Train(labels_path=str(labels), run_name="centroid")
            first_id, first = await _run(methods, store, spec)
            second_id, second = await _run(
                methods, store, Train(labels_path=str(labels), run_name="centroid")
            )

        assert builder.specs[0] == (str(project.resolve() / "models"), "centroid")
        assert builder.specs[1][1] == "centroid_2"
        assert first.result["model_dir"] == str(
            project.resolve() / "models" / "centroid"
        )
        assert second.result["model_dir"] == str(
            project.resolve() / "models" / "centroid_2"
        )
        # The submitted config is kept in the job folder (stage 2), not next to the labels.
        assert (tmp_path / "data" / "job-runs" / first_id / "config_0.yaml").exists()


class TestSnapshotOutputDir:
    def test_snapshot_with_output_dir_uses_it_and_creates_it(self, tmp_path):
        out = tmp_path / "serve" / "sleap-connect" / "mice"
        spec = Train(labels_content="AAAA", output_dir=str(out))
        assert train_output_root(spec) == out.resolve()
        assert out.is_dir()

    def test_snapshot_without_output_dir_keeps_the_job_folder(self):
        assert train_output_root(Train(labels_content="AAAA")) is None

    def test_worker_file_ignores_output_dir(self, tmp_path):
        spec = Train(
            labels_path=str(tmp_path / "labels.slp"), output_dir=str(tmp_path / "x")
        )
        assert train_output_root(spec) == tmp_path.resolve()

    def test_snapshot_predictions_are_named_after_the_project(self):
        assert predictions_path(
            "/w/job-runs/j/labels.slp", NOW, output_dir="/s/out", name="mice"
        ) == Path("/s/out/predictions/mice.261028_103000.predictions.slp")
