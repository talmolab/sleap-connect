"""Tests for job path checks (folder access, stage 2)."""

import sys
from pathlib import Path

import pytest

from sleap_rtc.config import MountConfig
from sleap_rtc.jobs.builder import CommandBuilder
from sleap_rtc.jobs.queue import JobQueue
from sleap_rtc.jobs.spec import TrackJobSpec, TrainJobSpec
from sleap_rtc.jobs.store import JobStore
from sleap_rtc.protocol_v1.errors import ProtocolError
from sleap_rtc.protocol_v1.job_methods import JobMethods
from sleap_rtc.protocol_v1.job_paths import JobPathPolicy, run_name_escapes
from sleap_rtc.protocol_v1.server import ProtocolV1Server
from sleap_rtc.protocol_v1.shared_folders import SharedFolderFileManager, SharedFolders


@pytest.fixture
def layout(tmp_path):
    """shared/ (shared, holds the worker's data dir), outside/, data dir with job-runs/."""
    shared = tmp_path / "shared"
    data_dir = shared / ".sleap-rtc"
    runs = data_dir / "job-runs"
    outside = tmp_path / "outside"
    for d in (shared, runs, outside):
        d.mkdir(parents=True)
    policy = JobPathPolicy(lambda: [str(shared)], runs, data_dir=data_dir)
    return (
        shared.resolve(),
        outside.resolve(),
        data_dir.resolve(),
        runs.resolve(),
        policy,
    )


def Train(**kw) -> TrainJobSpec:
    """A train spec; a placeholder inline config unless the test gives one."""
    if not any(
        k in kw
        for k in ("config_path", "config_paths", "config_content", "config_contents")
    ):
        kw["config_content"] = "{}"
    return TrainJobSpec(**kw)


def _refused(policy, spec) -> ProtocolError:
    with pytest.raises(ProtocolError) as exc:
        policy.check(spec)
    assert exc.value.code == "job.path_not_shared"
    return exc.value


class TestRunName:
    @pytest.mark.parametrize(
        "name", ["centroid", "251028_1030.centroid.n=120", "a/b", "'my run'"]
    )
    def test_names_that_stay_inside(self, name):
        assert run_name_escapes(name) is False

    @pytest.mark.parametrize(
        "name",
        [
            "../x",
            "../../../../root/vast/other/models",
            "/abs",
            "'../x'",
            "a/../../x",
            "..\\x",
            "C:\\x",
        ],
    )
    def test_names_that_escape(self, name):
        assert run_name_escapes(name) is True


class TestRunNameOnWindows:
    """The check must not depend on the worker's OS (Windows os.path uses
    backslashes, which once made every run name look like it escaped)."""

    def test_same_answers_with_windows_path_rules(self, monkeypatch):
        import ntpath

        from sleap_rtc.protocol_v1 import job_paths

        monkeypatch.setattr(job_paths.os, "path", ntpath)
        assert run_name_escapes("centroid") is False
        assert run_name_escapes("a/b") is False
        assert run_name_escapes("../x") is True


class TestTrain:
    def test_labels_inside_a_shared_folder_are_allowed(self, layout):
        shared, *_, policy = layout
        policy.check(Train(labels_path=str(shared / "labels.slp")))

    def test_labels_outside_are_refused_with_the_fix(self, layout):
        shared, outside, *_, policy = layout
        err = _refused(policy, Train(labels_path=str(outside / "labels.slp")))
        assert str(shared) in str(err)
        assert f'sleap-rtc add-folder "{outside}"' in str(err)
        assert err.data["field"] == "labels_path"

    def test_an_upload_without_a_path_is_allowed(self, layout):
        *_, policy = layout
        policy.check(Train(labels_content="AAAA"))

    def test_an_upload_naming_a_file_outside_is_refused(self, layout):
        # The Connect window's corrected copy of a worker file: the file
        # itself must still be in a shared folder.
        _, outside, *_, policy = layout
        err = _refused(
            policy, Train(labels_path=str(outside / "x.slp"), labels_content="AAAA")
        )
        assert err.data["field"] == "labels_path"

    @pytest.mark.parametrize("field", ["val_labels_path", "resume_ckpt_path"])
    def test_other_read_fields(self, layout, field):
        _, outside, *_, policy = layout
        _refused(policy, Train(**{field: str(outside / "f")}))

    def test_config_file_outside_is_refused(self, layout):
        _, outside, *_, policy = layout
        _refused(policy, Train(config_paths=[str(outside / "c.yaml")]))

    def test_worker_data_dir_is_refused_even_inside_a_shared_folder(self, layout):
        _, _, data_dir, _, policy = layout
        _refused(policy, Train(labels_path=str(data_dir / "devices.sqlite")))

    def test_run_name_and_model_types_are_checked(self, layout):
        *_, policy = layout
        _refused(policy, Train(run_name="../../x"))
        _refused(policy, Train(model_types=["centroid", "../x"]))


class TestConfigContents:
    def _cfg(self, **data):
        lines = ["data_config:"] + [f"  {k}: {v}" for k, v in data.items()]
        return "\n".join(lines) + "\n"

    def test_labels_in_config_checked_only_when_not_overridden(self, layout):
        shared, outside, *_, policy = layout
        cfg = self._cfg(train_labels_path=f"[{outside}/l.slp]")
        _refused(policy, Train(config_contents=[cfg]))
        policy.check(Train(config_contents=[cfg], labels_path=str(shared / "l.slp")))

    def test_test_file_and_pretrained_weights(self, layout):
        _, outside, *_, policy = layout
        _refused(
            policy,
            Train(config_contents=[self._cfg(test_file_path=f"{outside}/t.mp4")]),
        )
        weights = f"model_config:\n  pretrained_backbone_weights: {outside}/w.ckpt\n"
        _refused(policy, Train(config_contents=[weights]))

    def test_path_mappings_apply_before_checking(self, layout):
        shared, *_, policy = layout
        cfg = self._cfg(test_file_path="/Volumes/lab/t.mp4")
        policy.check(
            Train(config_contents=[cfg], path_mappings={"/Volumes/lab": str(shared)})
        )

    def test_config_file_contents_are_checked(self, layout):
        shared, outside, *_, policy = layout
        cfg = shared / "c.yaml"
        cfg.write_text(self._cfg(val_labels_path=f"[{outside}/v.slp]"))
        _refused(policy, Train(config_paths=[str(cfg)]))

    def test_unparseable_config_is_left_to_sleap_nn(self, layout):
        *_, policy = layout
        policy.check(Train(config_contents=["{not: [valid"]))


class TestTrack:
    def test_inputs(self, layout):
        shared, outside, _, runs, policy = layout
        policy.check(
            TrackJobSpec(
                data_path=str(shared / "v.mp4"), model_paths=[str(runs / "j1/models/m")]
            )
        )
        _refused(policy, TrackJobSpec(data_path=str(outside / "v.mp4")))
        _refused(
            policy,
            TrackJobSpec(
                data_path=str(shared / "v.mp4"), model_paths=[str(outside / "m")]
            ),
        )

    def test_output_path_inside_allowed_outside_refused(self, layout):
        shared, outside, data_dir, _, policy = layout
        data = str(shared / "v.mp4")
        policy.check(TrackJobSpec(data_path=data, output_path=str(shared / "out.slp")))
        err = _refused(
            policy, TrackJobSpec(data_path=data, output_path=str(outside / "out.slp"))
        )
        assert "write to" in str(err)
        _refused(
            policy,
            TrackJobSpec(data_path=data, output_path=str(data_dir / "devices.sqlite")),
        )


class _Mounts:
    def __init__(self, *paths):
        self.mounts = [
            MountConfig(path=str(p), label=Path(p).name or "root") for p in paths
        ]


class TestSubmit:
    async def test_submit_refuses_before_queueing(self, tmp_path, layout):
        shared, outside, *_ = layout
        async with JobStore(tmp_path / "jobs.sqlite") as store:
            methods = JobMethods(
                ProtocolV1Server(node_id="n"),
                store,
                JobQueue(max_concurrent=1),
                log_dir=tmp_path / "data" / "logs",
                file_manager=_Mounts(shared),
                command_builder=_NoopBuilder(),
                check_paths=True,
            )
            spec = TrackJobSpec(data_path=str(outside / "v.mp4"))
            with pytest.raises(ProtocolError) as exc:
                await methods.submit(
                    {"spec": {"type": "track", **spec.to_dict()}}, conn=None
                )
            assert exc.value.code == "job.path_not_shared"
            assert await store.list_jobs() == []

    async def test_checks_are_off_unless_asked(self, tmp_path, layout):
        _, outside, *_ = layout
        async with JobStore(tmp_path / "jobs.sqlite") as store:
            methods = JobMethods(
                ProtocolV1Server(node_id="n"),
                store,
                JobQueue(max_concurrent=1),
                log_dir=tmp_path / "data" / "logs",
                command_builder=_NoopBuilder(),
            )
            assert methods.path_policy is None


class _NoopBuilder:
    def build_command(self, spec, zmq_ports=None):
        return [sys.executable, "-c", "pass"]


class TestJobFolder:
    def test_inline_configs_are_saved_in_the_job_folder(self, tmp_path):
        spec = Train(config_contents=["a: 1\n", "b: 2\n"])
        JobMethods._materialize_config_contents(spec, tmp_path)
        assert spec.config_paths == [
            str(tmp_path / "config_0.yaml"),
            str(tmp_path / "config_1.yaml"),
        ]
        assert (tmp_path / "config_1.yaml").read_text() == "b: 2\n"

    def test_image_cache_is_forced_into_the_job_folder(self, tmp_path):
        cfg = tmp_path / "c.yaml"
        cfg.write_text("data_config:\n  cache_img_path: /etc/elsewhere\n")
        spec = Train(config_paths=[str(cfg)])
        assert JobMethods._cache_dir_for(spec, tmp_path) == str(tmp_path / "img_cache")
        spec.cache_img_path = JobMethods._cache_dir_for(spec, tmp_path)
        cmd = CommandBuilder().build_train_command(spec)
        assert f"data_config.cache_img_path={tmp_path / 'img_cache'}" in cmd

    def test_no_cache_setting_no_override(self, tmp_path):
        cfg = tmp_path / "c.yaml"
        cfg.write_text("data_config:\n  batch_size: 4\n")
        assert (
            JobMethods._cache_dir_for(Train(config_paths=[str(cfg)]), tmp_path) is None
        )


class TestStatOutsideSharedFolders:
    def test_existence_only(self, tmp_path):
        shared, outside, data_dir = (
            tmp_path / "s",
            tmp_path / "o",
            tmp_path / "s" / ".sleap-rtc",
        )
        data_dir.mkdir(parents=True)
        outside.mkdir()
        (outside / "v.mp4").write_bytes(b"x" * 100)
        (data_dir / "identity.json").write_text("{}")
        store = SharedFolders(tmp_path / "shared.json")
        store.add(shared.resolve())
        fm = SharedFolderFileManager(store, data_dir)

        assert fm.stat_path(str(outside / "v.mp4")) == {
            "path": str((outside / "v.mp4").resolve()),
            "type": "file",
            "shared": False,
        }
        assert (
            fm.stat_path(str(outside / "missing.mp4"))["error_code"] == "PATH_NOT_FOUND"
        )
        assert (
            fm.stat_path(str(data_dir / "identity.json"))["error_code"]
            == "ACCESS_DENIED"
        )
        assert "size" in fm.stat_path(str(shared))
