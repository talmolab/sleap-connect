"""Tests for Stop Early SIGINT behaviour.

When a CONTROL_COMMAND with {"command": "stop"} is forwarded through
send_control_message, the entire subprocess process group must receive
SIGINT so that all distributed workers (rank 0, rank 1, …) exit together,
preventing rank 1 from keeping stdout open and blocking the pipeline.

The subprocess must be started in its own process group (start_new_session=True)
so that os.killpg is safe to call without affecting the worker process itself.
"""

import json
import signal
import sys
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from sleap_connect.worker.job_executor import JobExecutor


class TestStopEarlySendsSignal:

    def _make_executor(self, running=True):
        executor = JobExecutor(worker=MagicMock(), capabilities=MagicMock())
        executor._progress_reporter = MagicMock()
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.returncode = None if running else 0
        executor._running_process = mock_process
        return executor

    @pytest.mark.xfail(
        reason=(
            "Sending SIGINT from send_control_message races against DDP's "
            "reduce_boolean_decision barrier: rank 0 enters the barrier waiting "
            "for rank 1, but SIGINT kills rank 1 before it can participate, "
            "causing a 30s distributed timeout. ZMQ-only stop is correct for "
            "multi-GPU; SIGINT is only safe as a post-timeout fallback."
        )
    )
    def test_stop_command_kills_process_group_with_sigint(self):
        """send_control_message('{"command":"stop"}') must SIGINT the whole process group."""
        executor = self._make_executor()
        with patch("os.killpg") as mock_killpg, patch("os.getpgid", return_value=99999):
            executor.send_control_message('{"command": "stop"}')
            mock_killpg.assert_called_once_with(99999, signal.SIGINT)

    def test_stop_command_still_forwards_zmq_message(self):
        """ZMQ forwarding must not be skipped when the process group is signalled."""
        executor = self._make_executor()
        with (
            patch("os.killpg", create=True),
            patch("os.getpgid", return_value=99999, create=True),
        ):
            executor.send_control_message('{"command": "stop"}')
        executor._progress_reporter.send_control_message.assert_called_once_with(
            '{"command": "stop"}'
        )

    def test_non_stop_command_does_not_kill_process_group(self):
        """Non-stop ZMQ commands must not signal the process group."""
        executor = self._make_executor()
        with patch("os.killpg", create=True) as mock_killpg:
            executor.send_control_message('{"command": "pause"}')
        mock_killpg.assert_not_called()

    def test_stop_command_does_not_kill_already_exited_process(self):
        """Process group must not be signalled when the process has already exited."""
        executor = self._make_executor(running=False)
        with patch("os.killpg", create=True) as mock_killpg:
            executor.send_control_message('{"command": "stop"}')
        mock_killpg.assert_not_called()

    def test_stop_command_with_no_running_process_does_not_raise(self):
        """send_control_message must not raise when no subprocess is running."""
        executor = self._make_executor()
        executor._running_process = None
        executor.send_control_message('{"command": "stop"}')  # must not raise


class TestStopRunningJob:

    def _make_executor(self, pid=12345, running=True):
        executor = JobExecutor(worker=MagicMock(), capabilities=MagicMock())
        mock_process = MagicMock()
        mock_process.pid = pid
        mock_process.returncode = None if running else 0
        executor._running_process = mock_process
        return executor

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix process groups only")
    def test_stop_running_job_kills_process_group_with_sigint(self):
        """stop_running_job must send SIGINT to the process group, not just the PID."""
        executor = self._make_executor()
        with patch("os.killpg") as mock_killpg, patch("os.getpgid", return_value=99999):
            executor.stop_running_job()
        mock_killpg.assert_called_once_with(99999, signal.SIGINT)

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix process groups only")
    def test_cancel_running_job_kills_process_group_with_sigterm(self):
        """cancel_running_job must send SIGTERM to the process group, not just the PID."""
        executor = self._make_executor()
        with patch("os.killpg") as mock_killpg, patch("os.getpgid", return_value=99999):
            executor.cancel_running_job()
        mock_killpg.assert_called_once_with(99999, signal.SIGTERM)

    def test_stop_running_job_no_op_when_already_exited(self):
        """stop_running_job must not signal a process that has already exited."""
        executor = self._make_executor(running=False)
        with patch("os.killpg", create=True) as mock_killpg:
            executor.stop_running_job()
        mock_killpg.assert_not_called()


class TestStopRequestedFlag:
    """execute_from_spec must send JOB_COMPLETE (not JOB_FAILED) when PL exits
    with code 1 after handling SIGINT gracefully (i.e. a stop was requested)."""

    def _make_process(self, returncode):
        proc = MagicMock()
        proc.pid = 12345
        proc.returncode = returncode
        proc.stdout = AsyncMock()
        proc.stdout.read = AsyncMock(return_value=b"")
        proc.stderr = AsyncMock()
        proc.stderr.read = AsyncMock(return_value=b"")
        proc.wait = AsyncMock()
        return proc

    @pytest.mark.asyncio
    async def test_exit_code_1_after_stop_sends_job_complete(self):
        """When stop was requested and process exits with code 1 (PL graceful
        shutdown), JOB_COMPLETE must be sent — not JOB_FAILED."""
        from sleap_connect.protocol import MSG_JOB_COMPLETE, MSG_JOB_FAILED, MSG_SEPARATOR

        executor = JobExecutor(worker=MagicMock(), capabilities=MagicMock())
        mock_channel = MagicMock()
        mock_channel.readyState = "open"
        mock_reporter = MagicMock()

        # Simulate stop command arriving before process exits
        with (
            patch("os.killpg", create=True),
            patch("os.getpgid", return_value=99999, create=True),
        ):
            executor.send_control_message('{"command": "stop"}')

        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=self._make_process(1)),
        ):
            await executor.execute_from_spec(
                mock_channel,
                ["sleap-nn", "train"],
                "job_1",
                job_type="train",
                zmq_ports={"controller": 9000, "publish": 9001},
                progress_reporter=mock_reporter,
            )

        sent = [str(c) for c in mock_channel.send.call_args_list]
        assert any(
            MSG_JOB_COMPLETE in s for s in sent
        ), f"Expected JOB_COMPLETE but got: {sent}"
        assert not any(
            MSG_JOB_FAILED in s for s in sent
        ), f"JOB_FAILED must not be sent after a requested stop: {sent}"

    @pytest.mark.asyncio
    async def test_exit_code_1_without_stop_sends_job_failed(self):
        """Exit code 1 with no prior stop command must still send JOB_FAILED."""
        from sleap_connect.protocol import MSG_JOB_FAILED

        executor = JobExecutor(worker=MagicMock(), capabilities=MagicMock())
        mock_channel = MagicMock()
        mock_channel.readyState = "open"

        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=self._make_process(1)),
        ):
            await executor.execute_from_spec(
                mock_channel,
                ["sleap-nn", "train"],
                "job_1",
                job_type="train",
                zmq_ports={"controller": 9000, "publish": 9001},
                progress_reporter=MagicMock(),
            )

        sent = [str(c) for c in mock_channel.send.call_args_list]
        assert any(
            MSG_JOB_FAILED in s for s in sent
        ), f"Expected JOB_FAILED for genuine failure: {sent}"

    @pytest.mark.asyncio
    async def test_stop_flag_resets_for_next_job(self):
        """_stop_requested must be False at the start of each execute_from_spec
        so a stop for model 1 does not make model 2's genuine failure look like
        a clean stop."""
        from sleap_connect.protocol import MSG_JOB_FAILED

        executor = JobExecutor(worker=MagicMock(), capabilities=MagicMock())
        mock_channel = MagicMock()
        mock_channel.readyState = "open"
        mock_reporter = MagicMock()

        # Model 1: stop requested, exits with 1
        with (
            patch("os.killpg", create=True),
            patch("os.getpgid", return_value=99999, create=True),
        ):
            executor.send_control_message('{"command": "stop"}')

        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=self._make_process(1)),
        ):
            await executor.execute_from_spec(
                mock_channel,
                ["sleap-nn", "train"],
                "job_1",
                job_type="train",
                zmq_ports={"controller": 9000, "publish": 9001},
                progress_reporter=mock_reporter,
            )

        # Model 2: no stop requested, exits with 1 (genuine failure)
        mock_channel.send.reset_mock()
        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=self._make_process(1)),
        ):
            await executor.execute_from_spec(
                mock_channel,
                ["sleap-nn", "train"],
                "job_2",
                job_type="train",
                zmq_ports={"controller": 9000, "publish": 9001},
                progress_reporter=mock_reporter,
            )

        sent = [str(c) for c in mock_channel.send.call_args_list]
        assert any(
            MSG_JOB_FAILED in s for s in sent
        ), f"Model 2 genuine failure must send JOB_FAILED: {sent}"


class TestWatchdogSigintFallback:
    """Watchdog must send SIGINT after 30 s when ZMQ stop alone doesn't work."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform == "win32", reason="Unix process groups only")
    async def test_watchdog_sends_sigint_when_stop_requested_and_process_lingers(self):
        """When _stop_requested is True and the process is still alive after
        one watchdog cycle, the watchdog must send SIGINT to the process group."""
        executor = JobExecutor(worker=MagicMock(), capabilities=MagicMock())

        # Process that keeps stdout open until SIGINT is delivered, matching
        # real behaviour where a stuck process doesn't close its stdout pipe.
        killed_event = asyncio.Event()
        linger_process = MagicMock()
        linger_process.pid = 12345
        linger_process.returncode = None
        linger_process.stdout = MagicMock()

        async def fake_read(n):
            # Block until the process is killed, then return EOF.
            await killed_event.wait()
            return b""

        linger_process.stdout.read = fake_read

        async def fake_wait():
            await killed_event.wait()

        linger_process.wait = fake_wait

        sigint_calls = []

        def fake_killpg(pgid, sig):
            if sig == 0:
                # Signal 0 is a liveness probe; raise once the process is dead.
                if killed_event.is_set():
                    raise ProcessLookupError
                return
            sigint_calls.append((pgid, sig))
            linger_process.returncode = -signal.SIGINT
            killed_event.set()  # unblocks both fake_read (stdout EOF) and fake_wait

        mock_channel = MagicMock()
        mock_channel.readyState = "open"

        # Patch asyncio.sleep so the 30-second watchdog cycle fires immediately.
        with (
            patch("os.killpg", side_effect=fake_killpg),
            patch("os.getpgid", return_value=99999),
            patch("sleap_connect.worker.job_executor._WATCHDOG_INTERVAL_SECS", 0),
            patch(
                "asyncio.create_subprocess_exec",
                new=AsyncMock(return_value=linger_process),
            ),
        ):
            executor._stop_requested = True
            await executor.execute_from_spec(
                mock_channel,
                ["sleep", "999"],
                "job_1",
                job_type="train",
                zmq_ports={"controller": 9000, "publish": 9001},
                progress_reporter=MagicMock(),
            )

        assert any(
            sig == signal.SIGINT for _, sig in sigint_calls
        ), "Watchdog must send SIGINT when stop was requested and process lingers"


class TestSubprocessNewSession:

    @pytest.mark.asyncio
    async def test_subprocess_started_in_new_session(self):
        """execute_from_spec must pass start_new_session=True to create_subprocess_exec."""
        executor = JobExecutor(worker=MagicMock(), capabilities=MagicMock())

        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.returncode = 0
        mock_process.stdout = AsyncMock()
        mock_process.stdout.read = AsyncMock(return_value=b"")
        mock_process.wait = AsyncMock()

        mock_channel = MagicMock()
        mock_channel.readyState = "open"

        with patch(
            "asyncio.create_subprocess_exec", new=AsyncMock(return_value=mock_process)
        ) as mock_exec:
            await executor.execute_from_spec(
                mock_channel,
                ["echo", "done"],
                "job_1",
                job_type="train",
                zmq_ports={"controller": 9000, "publish": 9001},
                progress_reporter=MagicMock(),
            )
            _, kwargs = mock_exec.call_args
            assert (
                kwargs.get("start_new_session") is True
            ), "subprocess must be started in its own session so killpg is safe"
