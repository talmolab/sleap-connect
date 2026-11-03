"""Tests for persistent ProgressReporter across a multi-model pipeline.

These tests verify that:
- execute_from_spec accepts and honours an externally-provided ProgressReporter
  (no new instance created, no cleanup called on it)
- handle_job_submit creates ONE ProgressReporter before the model loop and
  passes the same instance to every execute_from_spec call
- handle_job_submit calls async_cleanup exactly once after all models finish
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from sleap_connect.config import MountConfig
from sleap_connect.worker.job_executor import JobExecutor
from sleap_connect.worker.worker_class import RTCWorkerClient
from sleap_connect.protocol import MSG_JOB_SUBMIT, MSG_SEPARATOR


# ── helpers ──────────────────────────────────────────────────────────────────

def _submit_msg(spec_dict, job_id="test-job-1"):
    import json
    return f"{MSG_JOB_SUBMIT}{MSG_SEPARATOR}{job_id}{MSG_SEPARATOR}{json.dumps(spec_dict)}"


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_mount(tmp_path):
    (tmp_path / "labels.slp").write_text("slp data")
    return tmp_path


@pytest.fixture
def mock_channel():
    ch = MagicMock()
    ch.readyState = "open"
    ch.send = MagicMock()
    return ch


@pytest.fixture
def mock_process():
    """Subprocess that exits immediately with return code 0."""
    proc = MagicMock()
    proc.pid = 12345
    proc.returncode = 0
    proc.stdout = AsyncMock()
    proc.stdout.read = AsyncMock(return_value=b"")  # empty → stream loop breaks
    proc.wait = AsyncMock()
    return proc


@pytest.fixture
def executor():
    """JobExecutor with stub worker/capabilities (no real subprocess needed)."""
    return JobExecutor(worker=MagicMock(), capabilities=MagicMock())


@pytest.fixture
def worker(temp_mount):
    """RTCWorkerClient whose job_executor.execute_from_spec is mocked out."""
    mount = MountConfig(path=str(temp_mount), label="Test Mount")
    w = RTCWorkerClient(mounts=[mount], working_dir=str(temp_mount))
    w.peer_id = "test-worker-123"
    w.job_executor = MagicMock()
    w.job_executor.execute_from_spec = AsyncMock()
    return w


# ── execute_from_spec reporter ownership ─────────────────────────────────────

class TestExecuteFromSpecReporterOwnership:
    """execute_from_spec must not touch a reporter it did not create."""

    @pytest.mark.asyncio
    async def test_does_not_create_reporter_when_one_is_provided(
        self, executor, mock_channel, mock_process
    ):
        """ProgressReporter() constructor must not be called when a reporter is passed in."""
        external_reporter = MagicMock()
        external_reporter.async_cleanup = AsyncMock()

        with patch("sleap_connect.worker.job_executor.ProgressReporter") as mock_cls, \
             patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=mock_process)):
            await executor.execute_from_spec(
                mock_channel, ["echo", "done"], "job_1",
                job_type="train",
                zmq_ports={"controller": 9000, "publish": 9001},
                progress_reporter=external_reporter,
            )

        mock_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_does_not_cleanup_external_reporter(
        self, executor, mock_channel, mock_process
    ):
        """async_cleanup must NOT be called on a reporter that was passed in externally."""
        external_reporter = MagicMock()
        external_reporter.async_cleanup = AsyncMock()

        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=mock_process)):
            await executor.execute_from_spec(
                mock_channel, ["echo", "done"], "job_1",
                job_type="train",
                zmq_ports={"controller": 9000, "publish": 9001},
                progress_reporter=external_reporter,
            )

        external_reporter.async_cleanup.assert_not_called()

    @pytest.mark.asyncio
    async def test_creates_and_cleans_up_own_reporter_when_none_provided(
        self, executor, mock_channel, mock_process
    ):
        """When no reporter is passed, execute_from_spec creates and cleans up its own."""
        owned_reporter = MagicMock()
        owned_reporter.async_cleanup = AsyncMock()

        with patch("sleap_connect.worker.job_executor.ProgressReporter", return_value=owned_reporter), \
             patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=mock_process)):
            await executor.execute_from_spec(
                mock_channel, ["echo", "done"], "job_1",
                job_type="train",
                zmq_ports={"controller": 9000, "publish": 9001},
            )

        owned_reporter.async_cleanup.assert_called_once()


# ── handle_job_submit pipeline reporter ──────────────────────────────────────

class TestPipelineReporter:
    """handle_job_submit must create one reporter for the whole pipeline."""

    @pytest.mark.asyncio
    async def test_passes_same_reporter_to_all_models(
        self, worker, mock_channel, temp_mount
    ):
        """The same ProgressReporter instance must be passed to every execute_from_spec call."""
        config_yaml = "model_config:\n  backbone: unet\n"
        labels_path = str(temp_mount / "labels.slp")

        spec_dict = {
            "type": "train",
            "config_contents": [config_yaml, config_yaml],
            "model_types": ["centroid", "centered_instance"],
            "labels_path": labels_path,
        }
        message = _submit_msg(spec_dict)

        captured_reporters = []

        async def capture_reporter(channel, cmd, job_id, job_type="train", **kwargs):
            captured_reporters.append(kwargs.get("progress_reporter"))

        worker.job_executor.execute_from_spec = AsyncMock(side_effect=capture_reporter)

        pipeline_reporter = MagicMock()
        pipeline_reporter.async_cleanup = AsyncMock()
        pipeline_reporter.restart_progress_listener = AsyncMock()

        with patch("sleap_connect.worker.worker_class.ProgressReporter", return_value=pipeline_reporter):
            await worker.handle_job_submit(mock_channel, message)

        assert len(captured_reporters) == 2
        assert captured_reporters[0] is pipeline_reporter
        assert captured_reporters[1] is pipeline_reporter

    @pytest.mark.asyncio
    async def test_cleans_up_reporter_once_after_pipeline(
        self, worker, mock_channel, temp_mount
    ):
        """async_cleanup must be called exactly once after all models complete."""
        config_yaml = "model_config:\n  backbone: unet\n"
        labels_path = str(temp_mount / "labels.slp")

        spec_dict = {
            "type": "train",
            "config_contents": [config_yaml, config_yaml],
            "model_types": ["centroid", "centered_instance"],
            "labels_path": labels_path,
        }
        message = _submit_msg(spec_dict)

        pipeline_reporter = MagicMock()
        pipeline_reporter.async_cleanup = AsyncMock()

        with patch("sleap_connect.worker.worker_class.ProgressReporter", return_value=pipeline_reporter):
            await worker.handle_job_submit(mock_channel, message)

        pipeline_reporter.async_cleanup.assert_called_once()
