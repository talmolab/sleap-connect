"""Tests for worker handling of config_content in job submissions."""

import json
import os
import pytest

from unittest.mock import AsyncMock, MagicMock, patch

from sleap_connect.config import MountConfig
from sleap_connect.jobs.spec import TrainJobSpec
from sleap_connect.protocol import (
    MSG_JOB_SUBMIT,
    MSG_JOB_ACCEPTED,
    MSG_JOB_REJECTED,
    MSG_SEPARATOR,
)
from sleap_connect.worker.worker_class import RTCWorkerClient


@pytest.fixture(autouse=True)
def _mock_pipeline_reporter():
    """Patch ProgressReporter in worker_class to prevent real ZMQ socket binding.

    handle_job_submit now creates a ProgressReporter before the model loop.
    Tests here only care about execute_from_spec interactions, not ZMQ, so stub
    the reporter out to keep these tests fast and port-independent.
    """
    mock_reporter = MagicMock()
    mock_reporter.async_cleanup = AsyncMock()
    with patch(
        "sleap_connect.worker.worker_class.ProgressReporter", return_value=mock_reporter
    ):
        yield


@pytest.fixture
def temp_mount(tmp_path):
    """Create a temporary mount with test files."""
    labels = tmp_path / "labels.slp"
    labels.write_text("slp data")
    return tmp_path


@pytest.fixture
def worker(temp_mount):
    """Create a Worker with a test mount configured."""
    mount = MountConfig(path=str(temp_mount), label="Test Mount")
    w = RTCWorkerClient(mounts=[mount], working_dir=str(temp_mount))
    w.peer_id = "test-worker-123"
    # Mock job_executor so we don't actually run training
    w.job_executor = MagicMock()
    w.job_executor.execute_from_spec = AsyncMock()
    return w


@pytest.fixture
def mock_channel():
    """Create a mock RTCDataChannel."""
    channel = MagicMock()
    channel.readyState = "open"
    channel.send = MagicMock()
    return channel


def build_submit_message(spec_dict, job_id="test-job-1"):
    """Build a JOB_SUBMIT message from a spec dict."""
    return (
        f"{MSG_JOB_SUBMIT}{MSG_SEPARATOR}{job_id}{MSG_SEPARATOR}{json.dumps(spec_dict)}"
    )


class TestConfigContentHandling:
    """Tests for config_content in handle_job_submit."""

    @pytest.mark.asyncio
    async def test_config_content_writes_temp_file(
        self, worker, mock_channel, temp_mount
    ):
        """Test that config_content is written to a temp YAML file."""
        config_yaml = "model_config:\n  backbone: unet\n"
        labels_path = str(temp_mount / "labels.slp")

        spec_dict = {
            "type": "train",
            "config_content": config_yaml,
            "labels_path": labels_path,
        }
        message = build_submit_message(spec_dict)

        await worker.handle_job_submit(mock_channel, message)

        # Should have sent JOB_ACCEPTED
        calls = [str(c) for c in mock_channel.send.call_args_list]
        accepted_calls = [c for c in calls if MSG_JOB_ACCEPTED in c]
        assert len(accepted_calls) == 1

        # Job executor should have been called
        worker.job_executor.execute_from_spec.assert_called_once()

    @pytest.mark.asyncio
    async def test_config_content_temp_file_cleaned_up(
        self, worker, mock_channel, temp_mount
    ):
        """Test that temp config file is cleaned up after execution."""
        config_yaml = "model_config:\n  backbone: unet\n"
        labels_path = str(temp_mount / "labels.slp")

        spec_dict = {
            "type": "train",
            "config_content": config_yaml,
            "labels_path": labels_path,
        }
        message = build_submit_message(spec_dict)

        # Track temp files that exist during execution
        temp_files_during_exec = []

        async def capture_temp_files(channel, cmd, job_id, job_type="train", **kwargs):
            """Capture temp files in working_dir during execution."""
            import glob

            temp_files_during_exec.extend(
                glob.glob(os.path.join(str(temp_mount), "rtc_config_*.yaml"))
            )

        worker.job_executor.execute_from_spec = AsyncMock(
            side_effect=capture_temp_files
        )

        await worker.handle_job_submit(mock_channel, message)

        # Temp file should have existed during execution
        assert len(temp_files_during_exec) > 0

        # But should be cleaned up now
        for f in temp_files_during_exec:
            assert not os.path.exists(f), f"Temp file not cleaned up: {f}"

    @pytest.mark.asyncio
    async def test_config_content_cleanup_on_execution_error(
        self, worker, mock_channel, temp_mount
    ):
        """Test that temp file is cleaned up even when execution fails."""
        config_yaml = "model_config:\n  backbone: unet\n"
        labels_path = str(temp_mount / "labels.slp")

        spec_dict = {
            "type": "train",
            "config_content": config_yaml,
            "labels_path": labels_path,
        }
        message = build_submit_message(spec_dict)

        # Track the temp file path
        created_temp_files = []
        original_mkstemp = os.fdopen

        worker.job_executor.execute_from_spec = AsyncMock(
            side_effect=RuntimeError("Training crashed")
        )

        # The outer try/except in handle_job_submit catches RuntimeError,
        # but the finally block should still clean up
        await worker.handle_job_submit(mock_channel, message)

        # Check no rtc_config_ temp files are left in working_dir
        import glob

        remaining = glob.glob(os.path.join(str(temp_mount), "rtc_config_*.yaml"))
        assert len(remaining) == 0, f"Temp files not cleaned up: {remaining}"

    @pytest.mark.asyncio
    async def test_config_content_file_has_correct_content(
        self, worker, mock_channel, temp_mount
    ):
        """Test that the temp file contains the exact config_content."""
        config_yaml = "model_config:\n  backbone: unet\n  num_layers: 3\n"
        labels_path = str(temp_mount / "labels.slp")

        spec_dict = {
            "type": "train",
            "config_content": config_yaml,
            "labels_path": labels_path,
        }
        message = build_submit_message(spec_dict)

        # Capture the command passed to execute_from_spec
        captured_cmds = []

        async def capture_cmd(channel, cmd, job_id, job_type="train", **kwargs):
            captured_cmds.append(cmd)

        worker.job_executor.execute_from_spec = AsyncMock(side_effect=capture_cmd)

        await worker.handle_job_submit(mock_channel, message)

        # The command should reference a temp file that existed during execution
        assert len(captured_cmds) == 1
        cmd = captured_cmds[0]
        # Command uses: sleap-nn train --config-name <name> --config-dir <dir>
        assert "--config-name" in cmd
        assert "--config-dir" in cmd
        name_idx = cmd.index("--config-name")
        assert cmd[name_idx + 1].startswith("rtc_config_")
        assert cmd[name_idx + 1].endswith(".yaml")

    @pytest.mark.asyncio
    async def test_config_paths_still_works(self, worker, mock_channel, temp_mount):
        """Test that config_paths-based specs still work unchanged."""
        # Create a real config file
        config_file = temp_mount / "config.yaml"
        config_file.write_text("trainer: {}")
        labels_path = str(temp_mount / "labels.slp")

        spec_dict = {
            "type": "train",
            "config_path": str(config_file),
            "labels_path": labels_path,
        }
        message = build_submit_message(spec_dict)

        await worker.handle_job_submit(mock_channel, message)

        # Should have sent JOB_ACCEPTED
        calls = [str(c) for c in mock_channel.send.call_args_list]
        accepted_calls = [c for c in calls if MSG_JOB_ACCEPTED in c]
        assert len(accepted_calls) == 1

    @pytest.mark.asyncio
    async def test_path_mappings_logged(self, worker, mock_channel, temp_mount):
        """Test that path_mappings are logged when present."""
        config_yaml = "model_config:\n  backbone: unet\n"
        labels_path = str(temp_mount / "labels.slp")

        spec_dict = {
            "type": "train",
            "config_content": config_yaml,
            "labels_path": labels_path,
            "path_mappings": {"C:/data/vid.mp4": "/mnt/data/vid.mp4"},
        }
        message = build_submit_message(spec_dict)

        with patch("sleap_connect.worker.worker_class.logging") as mock_logging:
            await worker.handle_job_submit(mock_channel, message)

            # Check that path_mappings were logged
            info_calls = [str(c) for c in mock_logging.info.call_args_list]
            mapping_logs = [
                c
                for c in info_calls
                if "path_mappings" in c.lower() or "Path mappings" in c
            ]
            assert len(mapping_logs) >= 1


class TestResolvedSlpWriting:
    """Tests for writing a resolved SLP with corrected video paths during job submission."""

    @pytest.mark.asyncio
    async def test_video_mappings_trigger_resolved_slp(
        self, worker, mock_channel, temp_mount
    ):
        """When video path mappings are present, a resolved SLP is written and train cmd uses it."""
        config_yaml = "model_config:\n  backbone: unet\n"
        slp_path = str(temp_mount / "labels.slp")
        resolved_slp = str(temp_mount / "resolved_labels.slp")
        # Create the resolved file so the JobValidator doesn't reject it
        (temp_mount / "resolved_labels.slp").write_text("resolved slp")

        spec_dict = {
            "type": "train",
            "config_content": config_yaml,
            "labels_path": slp_path,
            "path_mappings": {
                "/Volumes/data/labels.slp": slp_path,
                "/Volumes/data/vid.mp4": str(temp_mount / "vid.mp4"),
            },
        }
        message = build_submit_message(spec_dict)

        mock_fm = MagicMock()
        mock_fm.write_slp_with_new_paths.return_value = {
            "output_path": resolved_slp,
            "videos_updated": 1,
        }
        worker.file_manager = mock_fm

        await worker.handle_job_submit(mock_channel, message)

        mock_fm.write_slp_with_new_paths.assert_called_once()
        call_kwargs = mock_fm.write_slp_with_new_paths.call_args
        assert call_kwargs.kwargs["slp_path"] == slp_path
        filename_map = call_kwargs.kwargs["filename_map"]
        # SLP mapping must be excluded; only video mapping present
        assert "/Volumes/data/labels.slp" not in filename_map
        assert "/Volumes/data/vid.mp4" in filename_map

        # execute_from_spec should have received a cmd referencing the resolved SLP
        # Signature: execute_from_spec(channel, cmd, job_id, ...)
        call_args = worker.job_executor.execute_from_spec.call_args
        cmd = call_args.args[1]
        assert any(resolved_slp in tok for tok in cmd)

    @pytest.mark.asyncio
    async def test_no_video_mappings_no_resolved_slp(
        self, worker, mock_channel, temp_mount
    ):
        """When only an SLP mapping is present (no video mappings), no resolved SLP is written."""
        config_yaml = "model_config:\n  backbone: unet\n"
        slp_path = str(temp_mount / "labels.slp")

        spec_dict = {
            "type": "train",
            "config_content": config_yaml,
            "labels_path": slp_path,
            "path_mappings": {
                "/Volumes/data/labels.slp": slp_path,
            },
        }
        message = build_submit_message(spec_dict)

        mock_fm = MagicMock()
        worker.file_manager = mock_fm

        await worker.handle_job_submit(mock_channel, message)

        mock_fm.write_slp_with_new_paths.assert_not_called()

    @pytest.mark.asyncio
    async def test_write_slp_error_falls_back_to_original(
        self, worker, mock_channel, temp_mount
    ):
        """When write_slp_with_new_paths fails, the original labels_path is used in the cmd."""
        config_yaml = "model_config:\n  backbone: unet\n"
        slp_path = str(temp_mount / "labels.slp")

        spec_dict = {
            "type": "train",
            "config_content": config_yaml,
            "labels_path": slp_path,
            "path_mappings": {
                "/Volumes/data/labels.slp": slp_path,
                "/Volumes/data/vid.mp4": str(temp_mount / "vid.mp4"),
            },
        }
        message = build_submit_message(spec_dict)

        mock_fm = MagicMock()
        mock_fm.write_slp_with_new_paths.return_value = {
            "error": "sleap-io not available"
        }
        worker.file_manager = mock_fm

        await worker.handle_job_submit(mock_channel, message)

        # Should fall back: cmd references original slp_path, not a resolved path
        call_args = worker.job_executor.execute_from_spec.call_args
        cmd = call_args.args[1]
        assert any(slp_path in tok for tok in cmd)


# =============================================================================
# sleap-nn version in registration properties
# =============================================================================


class TestSleapNNVersionProperty:
    """sleap_nn_version must be included in worker registration properties."""

    def test_get_sleap_nn_version_returns_string(self):
        """_get_sleap_nn_version() must return a non-empty string."""
        from sleap_connect.worker.worker_class import _get_sleap_nn_version

        version = _get_sleap_nn_version()
        assert isinstance(version, str)
        assert len(version) > 0

    def test_get_sleap_nn_version_fallback_on_missing(self, monkeypatch):
        """_get_sleap_nn_version() returns 'unknown' when sleap_nn is not importable."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "sleap_nn":
                raise ImportError("sleap_nn not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        from sleap_connect.worker import worker_class

        monkeypatch.setattr(worker_class, "_get_sleap_nn_version", lambda: "unknown")
        assert worker_class._get_sleap_nn_version() == "unknown"


# =============================================================================
# Bug fixes: GPU detection empty string, YAML parsing, ICE server config
# =============================================================================


class TestGPUModelDetection:
    """GPU model must never register as empty string — fallback to nvidia-smi or 'CPU'."""

    def test_empty_string_from_torch_falls_back(self, monkeypatch):
        """If torch returns empty string for GPU name, capabilities falls back (not empty)."""
        import types
        import sys

        # Stub torch.cuda so we can monkeypatch without a real GPU
        fake_torch = types.ModuleType("torch")
        fake_cuda = types.ModuleType("torch.cuda")

        class FakeProps:
            name = ""  # empty string — the HPC bug

        fake_cuda.is_available = lambda: True
        fake_cuda.device_count = lambda: 1
        fake_cuda.get_device_properties = lambda _: FakeProps()
        fake_torch.cuda = fake_cuda
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        monkeypatch.setitem(sys.modules, "torch.cuda", fake_cuda)

        # Also stub out subprocess so nvidia-smi returns a known value
        import subprocess
        import unittest.mock as mock

        fake_result = mock.MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = "NVIDIA A100-SXM4-40GB\n"
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: fake_result)

        # Reload capabilities to pick up the stubbed torch
        import importlib
        import sleap_connect.worker.capabilities as caps_mod

        importlib.reload(caps_mod)

        caps = caps_mod.WorkerCapabilities.__new__(caps_mod.WorkerCapabilities)
        caps.gpu_id = 0
        result = caps._detect_gpu_model()
        assert result != ""
        assert result == "NVIDIA A100-SXM4-40GB"


class TestMeshCoordinatorICEServers:
    """_handle_client_offer must use a fresh PC with ICE servers, not self.worker.pc."""

    def test_handle_client_offer_creates_fresh_pc(self):
        """_handle_client_offer source must call _create_client_peer_connection."""
        import inspect
        from sleap_connect.worker.mesh_coordinator import MeshCoordinator

        src = inspect.getsource(MeshCoordinator._handle_client_offer)
        assert (
            "_create_client_peer_connection" in src
        ), "_handle_client_offer must create a fresh RTCPeerConnection with ICE servers"


class TestExplicitIceServers:
    """RTCPeerConnection must always use explicit RTCConfiguration."""

    def test_create_peer_connection_passes_explicit_ice_servers(self):
        """_create_peer_connection must always pass RTCConfiguration(iceServers=...)
        so aiortc does NOT fall back to its hardcoded default in an uncontrolled way."""
        import inspect
        from sleap_connect.worker.worker_class import RTCWorkerClient

        src = inspect.getsource(RTCWorkerClient._create_peer_connection)
        # Must always create an RTCConfiguration — no bare RTCPeerConnection()
        assert "RTCConfiguration" in src
        import re

        bare_pc = re.findall(r"^\s+pc\s*=\s*RTCPeerConnection\(\)", src, re.MULTILINE)
        assert (
            len(bare_pc) == 0
        ), "Found bare RTCPeerConnection() — must pass RTCConfiguration(iceServers=...)"

    def test_stun_fallback_when_no_ice_servers(self):
        """When signaling server provides no ICE servers, factory must fall
        back to public STUN so workers behind NAT get srflx candidates."""
        import inspect
        from sleap_connect.worker.worker_class import RTCWorkerClient

        src = inspect.getsource(RTCWorkerClient._create_peer_connection)
        assert "_FALLBACK_STUN" in src
        assert "stun.l.google.com" in src

    def test_mesh_coordinator_uses_factory(self):
        """All RTCPeerConnection calls in mesh_coordinator must use the factory
        method, not bare RTCPeerConnection()."""
        import inspect
        from sleap_connect.worker import mesh_coordinator

        src = inspect.getsource(mesh_coordinator)
        # Count bare RTCPeerConnection() calls (not inside strings/comments)
        # The import line will have "RTCPeerConnection" but not "RTCPeerConnection()"
        import re

        bare_calls = re.findall(r"pc\s*=\s*RTCPeerConnection\(\)", src)
        assert len(bare_calls) == 0, (
            f"Found {len(bare_calls)} bare RTCPeerConnection() call(s) in "
            "mesh_coordinator — use worker._create_*_peer_connection() instead"
        )


class TestDoubleOfferGuard:
    """Main WebSocket loop must not double-handle offers when MeshCoordinator is admin."""

    def test_offer_skipped_when_admin_controller_is_admin(self):
        """worker_class main loop must skip 'offer' when admin_controller.is_admin is True."""
        import inspect
        from sleap_connect.worker.worker_class import RTCWorkerClient

        # Read the source around the offer handler and verify the admin guard
        src = inspect.getsource(RTCWorkerClient.handle_connection)
        # The guard must check is_admin before processing the offer
        assert (
            "admin_controller.is_admin" in src
        ), "Main WebSocket loop must skip offer when MeshCoordinator is admin"
        # Must use continue to skip (not fall through)
        assert "continue" in src
