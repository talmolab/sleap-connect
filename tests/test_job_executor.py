"""Tests for ``JobExecutor.run_inference``.

These tests verify that the worker streams the output ``predictions.slp``
file to the client over the RTC data channel before signalling
``INFERENCE_COMPLETE`` (Gap 1 of the prediction-streaming v1 design doc,
``2026-04-22-prediction-streaming-v1-design.md``).
"""

import inspect
import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sleap_connect.jobs.spec import TrackJobSpec
from sleap_connect.worker.job_executor import JobExecutor


def _make_process(returncode: int = 0):
    """Build a MagicMock subprocess whose stdout yields EOF immediately.

    ``run_inference`` uses ``async for raw_line in process.stdout:`` so
    ``process.stdout`` must be an async iterable.  A list yields its items
    then raises ``StopAsyncIteration``, matching real pipe EOF.
    """
    proc = MagicMock()
    proc.pid = 12345
    proc.returncode = returncode

    class _EmptyAsyncIter:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    proc.stdout = _EmptyAsyncIter()
    proc.wait = AsyncMock()
    return proc


@pytest.mark.asyncio
async def test_run_inference_streams_predictions_before_complete(tmp_path):
    """Worker must stream predictions via ``send_file`` before ``INFERENCE_COMPLETE``.

    Verifies the fix for Gap 1 — predictions must be streamed to the client,
    not merely referenced by a worker-side path.
    """
    predictions_path = tmp_path / "predictions.slp"
    predictions_path.write_bytes(b"fake slp content for test")

    channel = MagicMock()
    channel.readyState = "open"

    file_manager = MagicMock()
    file_manager.send_file = AsyncMock()

    worker = MagicMock()
    worker.file_manager = file_manager

    # Record call ordering across both send_file (async) and channel.send
    # (sync) using a shared counter on a wrapping container.
    call_order: list[str] = []

    async def _record_send_file(*args, **kwargs):
        call_order.append("send_file")

    file_manager.send_file.side_effect = _record_send_file

    def _record_channel_send(msg):
        call_order.append(f"channel.send:{msg}")

    channel.send = MagicMock(side_effect=_record_channel_send)

    executor = JobExecutor(worker=worker, capabilities=MagicMock())

    with patch(
        "asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=_make_process(returncode=0)),
    ):
        await executor.run_inference(
            channel,
            cmd=["true"],
            predictions_path=str(predictions_path),
        )

    # 1. send_file was awaited exactly once, with the channel and predictions path.
    file_manager.send_file.assert_awaited_once()
    send_file_args = file_manager.send_file.call_args
    assert send_file_args.args[0] is channel, (
        f"send_file must be called with the RTC channel as first arg; got "
        f"{send_file_args.args[0]!r}"
    )
    assert send_file_args.args[1] == str(predictions_path), (
        f"send_file must be called with the predictions path; got "
        f"{send_file_args.args[1]!r}"
    )

    # 2. INFERENCE_COMPLETE was sent.
    inf_complete_msgs = [
        entry
        for entry in call_order
        if entry.startswith("channel.send:INFERENCE_COMPLETE::")
    ]
    assert (
        len(inf_complete_msgs) == 1
    ), f"Exactly one INFERENCE_COMPLETE expected; call_order={call_order!r}"

    # 3. send_file was awaited BEFORE INFERENCE_COMPLETE was sent.
    send_file_idx = call_order.index("send_file")
    inf_complete_idx = call_order.index(inf_complete_msgs[0])
    assert send_file_idx < inf_complete_idx, (
        f"send_file must be awaited BEFORE INFERENCE_COMPLETE is sent; "
        f"call_order={call_order!r}"
    )

    # 4. INFERENCE_FAILED must NOT be sent on the happy path.
    assert not any(
        entry.startswith("channel.send:INFERENCE_FAILED::") for entry in call_order
    ), f"INFERENCE_FAILED must not be emitted on success; call_order={call_order!r}"


@pytest.mark.asyncio
async def test_run_inference_emits_failed_when_streaming_raises(tmp_path):
    """If ``send_file`` raises, the worker must send ``INFERENCE_FAILED`` and
    must NOT send ``INFERENCE_COMPLETE``.
    """
    predictions_path = tmp_path / "predictions.slp"
    predictions_path.write_bytes(b"fake slp content for test")

    channel = MagicMock()
    channel.readyState = "open"
    channel.send = MagicMock()

    file_manager = MagicMock()
    file_manager.send_file = AsyncMock(side_effect=RuntimeError("boom"))

    worker = MagicMock()
    worker.file_manager = file_manager

    executor = JobExecutor(worker=worker, capabilities=MagicMock())

    with patch(
        "asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=_make_process(returncode=0)),
    ):
        await executor.run_inference(
            channel,
            cmd=["true"],
            predictions_path=str(predictions_path),
        )

    sent = [c.args[0] for c in channel.send.call_args_list if c.args]
    assert any(
        isinstance(s, str) and s.startswith("INFERENCE_FAILED::") for s in sent
    ), f"Expected INFERENCE_FAILED after streaming failure; got {sent!r}"
    assert not any(
        isinstance(s, str) and s.startswith("INFERENCE_COMPLETE::") for s in sent
    ), f"INFERENCE_COMPLETE must not be sent after streaming failure; got {sent!r}"


@pytest.mark.asyncio
async def test_inference_failed_with_quote_in_error_is_valid_json(tmp_path):
    """When the worker emits INFERENCE_FAILED, the wire payload after the
    ``INFERENCE_FAILED::`` prefix must be valid JSON regardless of what
    characters appear in the interpolated error string. Real Python
    exception messages routinely contain ``"`` and ``\\``.
    """
    predictions_path = tmp_path / "predictions.slp"
    predictions_path.write_bytes(b"x")

    channel = MagicMock()
    channel.readyState = "open"
    channel.send = MagicMock()

    file_manager = MagicMock()
    # Force send_file to raise an exception whose str() contains a `"`.
    file_manager.send_file = AsyncMock(
        side_effect=RuntimeError('boom: file "/tmp/with quote".slp missing')
    )

    worker = MagicMock()
    worker.file_manager = file_manager
    executor = JobExecutor(worker=worker, capabilities=MagicMock())

    with patch(
        "asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=_make_process(returncode=0)),
    ):
        await executor.run_inference(
            channel,
            cmd=["true"],
            predictions_path=str(predictions_path),
        )

    failed_calls = [
        call.args[0]
        for call in channel.send.call_args_list
        if isinstance(call.args[0], str)
        and call.args[0].startswith("INFERENCE_FAILED::")
    ]
    assert (
        len(failed_calls) == 1
    ), f"Expected exactly one INFERENCE_FAILED; got {failed_calls!r}"

    payload_str = failed_calls[0].split("::", 1)[1]
    # MUST be parseable as JSON — this assertion is the whole point of the test.
    payload = json.loads(payload_str)
    assert "error" in payload
    assert "boom" in payload["error"]
    assert "with quote" in payload["error"]


@pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "Windows reserves '\"' as an invalid filename character "
        '(NTFS / FAT disallow < > : " | ? * in filenames), so we cannot '
        "create a real file at this path on Windows. The fix being verified "
        "(json.dumps safety) is platform-independent and is exercised by the "
        "macOS and Linux runs."
    ),
)
@pytest.mark.asyncio
async def test_inference_complete_with_quote_in_path_is_valid_json(tmp_path):
    """Same property for INFERENCE_COMPLETE — the predictions_path field
    must round-trip through JSON even if the path contains ``"`` or ``\\``.
    """
    weird_path = tmp_path / 'has"quote.slp'
    weird_path.write_bytes(b"x")

    channel = MagicMock()
    channel.readyState = "open"
    channel.send = MagicMock()

    file_manager = MagicMock()
    file_manager.send_file = AsyncMock()

    worker = MagicMock()
    worker.file_manager = file_manager
    executor = JobExecutor(worker=worker, capabilities=MagicMock())

    with patch(
        "asyncio.create_subprocess_exec",
        new=AsyncMock(return_value=_make_process(returncode=0)),
    ):
        await executor.run_inference(
            channel,
            cmd=["true"],
            predictions_path=str(weird_path),
        )

    complete_calls = [
        call.args[0]
        for call in channel.send.call_args_list
        if isinstance(call.args[0], str)
        and call.args[0].startswith("INFERENCE_COMPLETE::")
    ]
    assert len(complete_calls) == 1
    payload = json.loads(complete_calls[0].split("::", 1)[1])
    assert payload["predictions_path"] == str(weird_path)


class TestExecuteFromSpecAcceptsSpec:
    """Task 6 plumbing: ``execute_from_spec`` accepts an optional ``spec``
    parameter that Task 7 will use to stream predictions back to the client.

    Pure-signature test — no behavior change in this commit.
    """

    def test_signature_includes_spec_parameter(self):
        sig = inspect.signature(JobExecutor.execute_from_spec)
        assert "spec" in sig.parameters
        assert sig.parameters["spec"].default is None
        assert sig.parameters["spec"].kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )

    def test_spec_is_optional_so_existing_callers_still_work(self):
        """The new parameter must be optional — pre-Task-6 callers that omit
        ``spec=`` must still match the signature.
        """
        sig = inspect.signature(JobExecutor.execute_from_spec)
        # All required parameters (no default) must be the same as before.
        required = [
            name
            for name, p in sig.parameters.items()
            if p.default is inspect.Parameter.empty and name != "self"
        ]
        # Per the pre-Task-6 signature, only ``channel``, ``cmd``, ``job_id``
        # are required.  ``spec`` must NOT be in this list.
        assert "spec" not in required
        assert set(required) == {"channel", "cmd", "job_id"}


class TestExecuteFromSpecTrackStreaming:
    """Task 7: ``execute_from_spec`` must stream the output predictions file
    back to the client BEFORE emitting ``MSG_JOB_COMPLETE`` for track jobs,
    mirroring PR #79's ``run_inference`` streaming pattern.
    """

    @pytest.mark.asyncio
    async def test_track_success_streams_output_before_complete(self, tmp_path):
        """Worker must stream the output file via ``send_file`` before
        ``MSG_JOB_COMPLETE``, and the payload must include ``output_path``.
        """
        output_path = tmp_path / "predictions.slp"
        output_path.write_bytes(b"fake slp content for test")

        spec = TrackJobSpec(
            data_path=str(tmp_path / "video.mp4"),
            model_paths=[str(tmp_path / "model")],
            output_path=str(output_path),
        )

        channel = MagicMock()
        channel.readyState = "open"

        file_manager = MagicMock()
        file_manager.send_file = AsyncMock()

        worker = MagicMock()
        worker.file_manager = file_manager

        # Record call ordering across both send_file (async) and channel.send
        # (sync) using a shared list.
        call_order: list[str] = []

        async def _record_send_file(*args, **kwargs):
            call_order.append("send_file")

        file_manager.send_file.side_effect = _record_send_file

        def _record_channel_send(msg):
            call_order.append(f"channel.send:{msg}")

        channel.send = MagicMock(side_effect=_record_channel_send)

        executor = JobExecutor(worker=worker, capabilities=MagicMock())

        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=_make_process(returncode=0)),
        ):
            result = await executor.execute_from_spec(
                channel,
                cmd=["true"],
                job_id="job-track-success",
                job_type="track",
                spec=spec,
            )

        # 1. send_file was awaited exactly once with (channel, output_path).
        file_manager.send_file.assert_awaited_once()
        send_file_args = file_manager.send_file.call_args
        assert send_file_args.args[0] is channel, (
            f"send_file must be called with the RTC channel as first arg; got "
            f"{send_file_args.args[0]!r}"
        )
        assert send_file_args.args[1] == str(output_path), (
            f"send_file must be called with the spec.output_path; got "
            f"{send_file_args.args[1]!r}"
        )

        # 2. MSG_JOB_COMPLETE was sent exactly once.
        complete_msgs = [
            entry
            for entry in call_order
            if entry.startswith("channel.send:JOB_COMPLETE::")
        ]
        assert (
            len(complete_msgs) == 1
        ), f"Exactly one MSG_JOB_COMPLETE expected; call_order={call_order!r}"

        # 3. send_file was awaited BEFORE MSG_JOB_COMPLETE was sent.
        send_file_idx = call_order.index("send_file")
        complete_idx = call_order.index(complete_msgs[0])
        assert send_file_idx < complete_idx, (
            f"send_file must be awaited BEFORE MSG_JOB_COMPLETE is sent; "
            f"call_order={call_order!r}"
        )

        # 4. MSG_JOB_FAILED must NOT be sent on the happy path.
        assert not any(
            entry.startswith("channel.send:JOB_FAILED::") for entry in call_order
        ), f"MSG_JOB_FAILED must not be emitted on success; call_order={call_order!r}"

        # 5. Payload contains output_path == str(spec.output_path).
        payload_str = complete_msgs[0].split("::", 1)[1].removeprefix("channel.send:")
        # Defensive — the prefix `channel.send:JOB_COMPLETE::` was already
        # split off above; payload_str should now be just JSON.
        payload = json.loads(payload_str)
        assert payload.get("output_path") == str(spec.output_path), (
            f"MSG_JOB_COMPLETE payload must include output_path == "
            f"{str(spec.output_path)!r}; got {payload!r}"
        )

        # 6. The success result dict.
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_track_streaming_failure_emits_msg_job_failed(self, tmp_path):
        """If ``send_file`` raises during a track job, the worker must emit
        ``MSG_JOB_FAILED`` and must NOT emit ``MSG_JOB_COMPLETE``.
        """
        output_path = tmp_path / "predictions.slp"
        output_path.write_bytes(b"fake slp content for test")

        spec = TrackJobSpec(
            data_path=str(tmp_path / "video.mp4"),
            model_paths=[str(tmp_path / "model")],
            output_path=str(output_path),
        )

        channel = MagicMock()
        channel.readyState = "open"
        channel.send = MagicMock()

        file_manager = MagicMock()
        file_manager.send_file = AsyncMock(side_effect=RuntimeError("boom"))

        worker = MagicMock()
        worker.file_manager = file_manager

        executor = JobExecutor(worker=worker, capabilities=MagicMock())

        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=_make_process(returncode=0)),
        ):
            result = await executor.execute_from_spec(
                channel,
                cmd=["true"],
                job_id="job-track-fail",
                job_type="track",
                spec=spec,
            )

        sent = [c.args[0] for c in channel.send.call_args_list if c.args]
        assert any(
            isinstance(s, str) and s.startswith("JOB_FAILED::") for s in sent
        ), f"Expected MSG_JOB_FAILED after streaming failure; got {sent!r}"
        assert not any(
            isinstance(s, str) and s.startswith("JOB_COMPLETE::") for s in sent
        ), f"MSG_JOB_COMPLETE must not be sent after streaming failure; got {sent!r}"

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_train_branch_does_not_stream(self, tmp_path):
        """Training jobs must not trigger streaming and must not include
        ``output_path`` in the MSG_JOB_COMPLETE payload.
        """
        channel = MagicMock()
        channel.readyState = "open"
        channel.send = MagicMock()

        file_manager = MagicMock()
        file_manager.send_file = AsyncMock()

        worker = MagicMock()
        worker.file_manager = file_manager

        executor = JobExecutor(worker=worker, capabilities=MagicMock())

        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=_make_process(returncode=0)),
        ):
            await executor.execute_from_spec(
                channel,
                cmd=["true"],
                job_id="job-train",
                job_type="train",
            )

        # send_file must not be called for train jobs.
        file_manager.send_file.assert_not_awaited()

        sent = [c.args[0] for c in channel.send.call_args_list if c.args]
        complete_msgs = [
            s for s in sent if isinstance(s, str) and s.startswith("JOB_COMPLETE::")
        ]
        assert (
            len(complete_msgs) == 1
        ), f"Expected exactly one MSG_JOB_COMPLETE for train job; got {sent!r}"

        payload = json.loads(complete_msgs[0].split("::", 1)[1])
        assert "output_path" not in payload, (
            f"MSG_JOB_COMPLETE payload for train jobs must NOT contain "
            f"output_path; got {payload!r}"
        )

    @pytest.mark.asyncio
    async def test_track_with_no_spec_does_not_stream(self, tmp_path):
        """Compatibility: a track job invoked WITHOUT a spec (pre-Task-7 caller
        path) must not stream and must not add ``output_path`` to the payload.
        """
        channel = MagicMock()
        channel.readyState = "open"
        channel.send = MagicMock()

        file_manager = MagicMock()
        file_manager.send_file = AsyncMock()

        worker = MagicMock()
        worker.file_manager = file_manager

        executor = JobExecutor(worker=worker, capabilities=MagicMock())

        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=_make_process(returncode=0)),
        ):
            await executor.execute_from_spec(
                channel,
                cmd=["true"],
                job_id="job-track-no-spec",
                job_type="track",
                spec=None,
            )

        file_manager.send_file.assert_not_awaited()

        sent = [c.args[0] for c in channel.send.call_args_list if c.args]
        complete_msgs = [
            s for s in sent if isinstance(s, str) and s.startswith("JOB_COMPLETE::")
        ]
        assert len(complete_msgs) == 1
        payload = json.loads(complete_msgs[0].split("::", 1)[1])
        assert "output_path" not in payload


class TestCancelGraceConstant:
    """The SIGTERM→SIGKILL grace must be 5 seconds (brainstorm decision)."""

    def test_grace_constant_is_five_seconds(self):
        from sleap_connect.worker.job_executor import _CANCEL_GRACE_SECS

        assert _CANCEL_GRACE_SECS == 5, (
            "Brainstorm locked in 5 s grace before SIGKILL escalation. "
            "If you change this, update the brainstorm doc and the design "
            "doc together."
        )


class TestCancelDuringTrack:
    """Task 8: ``cancel_running_job`` must produce a SIGTERM-style
    return code (-signal.SIGTERM) so ``execute_from_spec`` classifies the
    job as ``cancelled`` and emits ``MSG_JOB_FAILED`` (not COMPLETE).
    """

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only signal semantics")
    @pytest.mark.asyncio
    async def test_cancel_during_track_emits_msg_job_failed(self, tmp_path):
        import signal as _signal

        from sleap_connect.jobs.spec import TrackJobSpec
        from sleap_connect.worker.job_executor import JobExecutor

        spec = TrackJobSpec(
            data_path=str(tmp_path / "video.mp4"),
            model_paths=[str(tmp_path / "model")],
            output_path=str(tmp_path / "predictions.slp"),
        )

        channel = MagicMock()
        channel.readyState = "open"
        channel.send = MagicMock()

        worker = MagicMock()
        worker.file_manager = MagicMock()
        worker.file_manager.send_file = AsyncMock()

        # _make_process(returncode=-signal.SIGTERM) simulates the kernel
        # reporting "killed by SIGTERM" to asyncio.subprocess.
        proc = _make_process(returncode=-_signal.SIGTERM)

        executor = JobExecutor(worker=worker, capabilities=MagicMock())

        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=proc),
        ):
            result = await executor.execute_from_spec(
                channel,
                cmd=["true"],
                job_id="job-track-cancel",
                job_type="track",
                spec=spec,
            )

        sent = [c.args[0] for c in channel.send.call_args_list if c.args]
        assert any(
            isinstance(s, str) and s.startswith("JOB_FAILED::")
            for s in sent
        ), f"Cancelled track job must emit MSG_JOB_FAILED; got {sent!r}"
        assert not any(
            isinstance(s, str) and s.startswith("JOB_COMPLETE::")
            for s in sent
        ), f"Cancelled track job must NOT emit MSG_JOB_COMPLETE; got {sent!r}"
        # The "cancelled by user" wording is a contract: the GUI uses it
        # to distinguish user-cancelled from worker-side error.
        failed_msgs = [s for s in sent if s.startswith("JOB_FAILED::")]
        payload = json.loads(failed_msgs[0].split("::", 1)[1])
        assert "cancel" in payload.get("message", "").lower()
        assert result["cancelled"] is True

    def test_cancel_running_job_no_op_when_no_process(self):
        """``cancel_running_job`` must be a safe no-op when no process is running."""
        from sleap_connect.worker.job_executor import JobExecutor

        executor = JobExecutor(worker=MagicMock(), capabilities=MagicMock())
        assert executor._running_process is None
        executor.cancel_running_job()  # must not raise


class TestExecuteFromSpecTrackJobLogEmission:
    """Task 9: track jobs must wrap subprocess output in MSG_JOB_LOG::
    so the client can dispatch through on_job_message instead of parsing
    raw strings.  Training jobs keep the legacy bare-line format.
    """

    @pytest.mark.asyncio
    async def test_track_lines_emitted_as_msg_job_log(self, tmp_path):
        from sleap_connect.jobs.spec import TrackJobSpec
        from sleap_connect.worker.job_executor import JobExecutor

        spec = TrackJobSpec(
            data_path=str(tmp_path / "video.mp4"),
            model_paths=[str(tmp_path / "model")],
            output_path=str(tmp_path / "predictions.slp"),
        )
        spec_output = tmp_path / "predictions.slp"
        spec_output.write_bytes(b"fake")

        channel = MagicMock()
        channel.readyState = "open"
        channel.send = MagicMock()

        worker = MagicMock()
        worker.file_manager = MagicMock()
        worker.file_manager.send_file = AsyncMock()

        # Build a process whose stdout yields two newline-terminated lines.
        proc = MagicMock()
        proc.pid = 4242
        proc.returncode = 0

        chunks = [
            b"Predicting... 50% 17/35 ETA: 0:00:01\n",
            b"Predicting... 100% 35/35 ETA: 0:00:00 47.8 FPS\n",
            b"",  # EOF
        ]

        async def _read(_n):
            return chunks.pop(0) if chunks else b""

        proc.stdout = MagicMock()
        proc.stdout.read = AsyncMock(side_effect=_read)
        proc.stderr = None  # track-mode merges stderr into stdout
        proc.wait = AsyncMock()

        executor = JobExecutor(worker=worker, capabilities=MagicMock())

        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=proc),
        ):
            await executor.execute_from_spec(
                channel,
                cmd=["true"],
                job_id="job-log-emit",
                job_type="track",
                spec=spec,
            )

        sent = [c.args[0] for c in channel.send.call_args_list if c.args]
        log_msgs = [s for s in sent if isinstance(s, str) and s.startswith("JOB_LOG::")]
        assert len(log_msgs) >= 2, (
            f"Expected at least 2 MSG_JOB_LOG messages from track stdout, "
            f"got: {sent!r}"
        )
        # Raw subprocess output must NOT be sent without the JOB_LOG:: prefix.
        bare_predicting = [
            s for s in sent
            if isinstance(s, str)
            and s.startswith("Predicting...")
            and not s.startswith("JOB_LOG::")
        ]
        assert not bare_predicting, (
            f"Track jobs must wrap subprocess output in JOB_LOG::; got "
            f"bare lines: {bare_predicting!r}"
        )

    @pytest.mark.asyncio
    async def test_train_lines_remain_bare(self, tmp_path):
        """Regression: training jobs must keep the legacy bare-line format
        because _run_training_async's on_log callback expects it.
        """
        from sleap_connect.worker.job_executor import JobExecutor

        channel = MagicMock()
        channel.readyState = "open"
        channel.send = MagicMock()

        proc = MagicMock()
        proc.pid = 4243
        proc.returncode = 0

        chunks = [b"Epoch 1 - train_loss=0.5\n", b""]

        async def _read(_n):
            return chunks.pop(0) if chunks else b""

        proc.stdout = MagicMock()
        proc.stdout.read = AsyncMock(side_effect=_read)
        proc.stderr = MagicMock()
        proc.stderr.read = AsyncMock(return_value=b"")
        proc.wait = AsyncMock()

        executor = JobExecutor(worker=MagicMock(), capabilities=MagicMock())

        with patch(
            "asyncio.create_subprocess_exec",
            new=AsyncMock(return_value=proc),
        ):
            await executor.execute_from_spec(
                channel,
                cmd=["true"],
                job_id="job-train-bare",
                job_type="train",
            )

        sent = [c.args[0] for c in channel.send.call_args_list if c.args]
        assert any(
            isinstance(s, str)
            and s.startswith("Epoch 1")
            and not s.startswith("JOB_LOG::")
            for s in sent
        ), (
            f"Training stdout must remain bare-line format for legacy "
            f"on_log compatibility; got {sent!r}"
        )
