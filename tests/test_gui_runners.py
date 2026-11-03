"""Tests for sleap_connect.gui.runners module."""

import sys
import pytest
from unittest.mock import MagicMock, patch

import jsonpickle

from sleap_connect.api import ProgressEvent
from sleap_connect.gui.runners import (
    RemoteProgressBridge,
    run_remote_training,
    format_progress_line,
)


# Create a mock zmq module for testing
@pytest.fixture
def mock_zmq():
    """Create and install mock zmq module.

    The mock context's socket() method returns different mocks for PUB and
    SUB socket types. The SUB socket's poll() returns 0 (no messages) so the
    RemoteProgressBridge poll thread stays idle during tests.
    """
    mock_module = MagicMock()
    mock_module.PUB = 1
    mock_module.SUB = 2
    mock_module.POLLIN = 1
    mock_module.NOBLOCK = 1

    mock_context = MagicMock()

    def _make_socket(socket_type):
        sock = MagicMock()
        if socket_type == mock_module.SUB:
            sock.poll.return_value = 0  # No messages — poll thread stays idle
        return sock

    mock_context.socket.side_effect = _make_socket
    mock_module.Context.return_value = mock_context

    # Patch zmq in sys.modules so imports find it
    with patch.dict(sys.modules, {"zmq": mock_module}):
        yield mock_module


def _setup_zmq_mocks(mock_zmq):
    """Set up ZMQ mocks for tests that directly interact with sockets.

    Returns the PUB socket mock (the one used for send_string).
    Tests that need the PUB socket should call this instead of setting up
    mock_context.socket.return_value manually.
    """
    mock_context = mock_zmq.Context.return_value
    # The fixture's side_effect creates fresh mocks for each socket() call.
    # Create PUB and SUB sockets to capture their mocks, then reset the
    # side_effect so the bridge's start() creates fresh ones with the same
    # behavior.
    pub_socket = MagicMock()
    sub_socket = MagicMock()
    sub_socket.poll.return_value = 0

    def _make_socket(socket_type):
        if socket_type == mock_zmq.PUB:
            return pub_socket
        return sub_socket

    mock_context.socket.side_effect = _make_socket
    return pub_socket


def _decode_zmq_message(mock_socket) -> dict:
    """Helper to decode the last ZMQ message sent on a mock socket."""
    encoded_str = mock_socket.send_string.call_args[0][0]
    return jsonpickle.decode(encoded_str)


def _decode_all_zmq_messages(mock_socket) -> list[dict]:
    """Helper to decode all ZMQ messages sent on a mock socket."""
    return [
        jsonpickle.decode(call[0][0])
        for call in mock_socket.send_string.call_args_list
    ]


# =============================================================================
# RemoteProgressBridge Tests
# =============================================================================


class TestRemoteProgressBridge:
    """Tests for RemoteProgressBridge class."""

    def test_init_default_port(self):
        """Should initialize with default port 9001."""
        bridge = RemoteProgressBridge()
        assert bridge._publish_port == 9001
        assert bridge._model_type == ""
        assert not bridge._started

    def test_init_custom_port_and_model_type(self):
        """Should accept custom port and model type."""
        bridge = RemoteProgressBridge(publish_port=9999, model_type="centroid")
        assert bridge._publish_port == 9999
        assert bridge._model_type == "centroid"

    def test_set_model_type(self):
        """Should update model type for subsequent messages."""
        bridge = RemoteProgressBridge(model_type="centroid")
        assert bridge._model_type == "centroid"
        bridge.set_model_type("centered_instance")
        assert bridge._model_type == "centered_instance"

    def test_start_connects_socket(self, mock_zmq):
        """Should create ZMQ PUB and SUB sockets on start."""
        mock_socket = _setup_zmq_mocks(mock_zmq)

        bridge = RemoteProgressBridge(publish_port=9001)
        bridge.start()

        # PUB socket must connect (not bind) — LossViewer owns the bind
        mock_socket.connect.assert_called_once_with("tcp://127.0.0.1:9001")
        mock_socket.bind.assert_not_called()
        # SUB socket also created and connected
        assert bridge._sub_socket is not None
        assert bridge._started

        bridge.stop()

    def test_stop_closes_socket(self, mock_zmq):
        """Should close both PUB and SUB sockets on stop."""
        mock_socket = _setup_zmq_mocks(mock_zmq)

        bridge = RemoteProgressBridge()
        bridge.start()
        sub_socket = bridge._sub_socket
        bridge.stop()

        mock_socket.close.assert_called_once()
        sub_socket.close.assert_called_once()
        assert not bridge._started

    def test_context_manager(self, mock_zmq):
        """Should work as context manager."""
        mock_socket = _setup_zmq_mocks(mock_zmq)

        with RemoteProgressBridge() as bridge:
            assert bridge._started
            sub_socket = bridge._sub_socket

        mock_socket.close.assert_called_once()
        sub_socket.close.assert_called_once()

    def test_on_progress_not_started(self):
        """Should not crash if bridge not started."""
        bridge = RemoteProgressBridge()
        event = ProgressEvent(event_type="train_begin")
        # Should not raise
        bridge.on_progress(event)

    def test_on_raw_zmq_message(self, mock_zmq):
        """on_raw_zmq_message should decode, inject 'what' if empty, and re-encode."""
        import jsonpickle

        mock_socket = _setup_zmq_mocks(mock_zmq)

        raw_msg = jsonpickle.encode({"event": "batch_end", "what": "centroid"})

        with RemoteProgressBridge() as bridge:
            bridge.on_raw_zmq_message(raw_msg)

        # Should preserve existing non-empty 'what'
        sent = jsonpickle.decode(mock_socket.send_string.call_args[0][0])
        assert sent["event"] == "batch_end"
        assert sent["what"] == "centroid"

    def test_on_raw_zmq_message_injects_what(self, mock_zmq):
        """on_raw_zmq_message should inject model_type when 'what' is empty."""
        import jsonpickle

        mock_socket = _setup_zmq_mocks(mock_zmq)

        # sleap-nn sends what="" by default
        raw_msg = jsonpickle.encode({"event": "batch_end", "what": ""})

        with RemoteProgressBridge(model_type="centroid") as bridge:
            bridge.on_raw_zmq_message(raw_msg)

        sent = jsonpickle.decode(mock_socket.send_string.call_args[0][0])
        assert sent["what"] == "centroid"

    def test_on_raw_zmq_message_not_started(self):
        """on_raw_zmq_message should not crash if bridge not started."""
        bridge = RemoteProgressBridge()
        # Should not raise
        bridge.on_raw_zmq_message('{"event": "batch_end"}')


class TestRemoteProgressBridgeCancelForwarding:
    """Regression: the existing LossViewer-cancel → MSG_JOB_CANCEL path
    must keep working after Task 8's refactor.

    The path is:
        LossViewer cancel cmd → bridge SUB socket
        → bridge._poll_commands sees `command == "cancel"`
        → calls `self._send_fn(MSG_JOB_CANCEL)`
        → that send_fn was registered via on_channel_ready

    We verify by stubbing the SUB socket to yield one `cancel` jsonpickle
    payload, registering a recording send_fn, starting the bridge,
    waiting briefly for the poll thread to consume the message, and
    asserting the recording captured `MSG_JOB_CANCEL`.
    """

    def test_loss_viewer_cancel_forwards_msg_job_cancel(self, mock_zmq):
        import time
        import jsonpickle
        from sleap_connect.gui.runners import RemoteProgressBridge
        from sleap_connect.protocol import MSG_JOB_CANCEL

        sent: list[str] = []

        def _record(msg: str) -> None:
            sent.append(msg)

        # Configure the SUB socket to yield ONE cancel command, then 0s.
        mock_context = mock_zmq.Context.return_value
        sub_socket = MagicMock()
        cancel_payload = jsonpickle.encode({"command": "cancel"})
        sub_socket.poll.side_effect = [1] + [0] * 100
        sub_socket.recv_string.return_value = cancel_payload

        pub_socket = MagicMock()

        def _make_socket(socket_type):
            return pub_socket if socket_type == mock_zmq.PUB else sub_socket

        mock_context.socket.side_effect = _make_socket

        bridge = RemoteProgressBridge()
        bridge.set_send_fn(_record)
        bridge.start()
        # The poll thread sleeps 100 ms per iteration; give it a few cycles.
        time.sleep(0.5)
        bridge.stop()

        assert MSG_JOB_CANCEL in sent, (
            f"Expected bridge to forward MSG_JOB_CANCEL after LossViewer "
            f"cancel command; got {sent!r}"
        )


# =============================================================================
# LossViewer Message Format Tests
# =============================================================================


class TestLossViewerMessageFormat:
    """Tests verifying message format matches SLEAP's LossViewer expectations.

    LossViewer (sleap/gui/widgets/monitor.py) reads messages with:
        msg = jsonpickle.decode(self.sub.recv_string())

    And expects:
    - "event": event type string
    - "what": model type for filtering
    - "logs": dict with "train/loss" and "val/loss" keys (for epoch_end)
    """

    def test_uses_send_string_not_multipart(self, mock_zmq):
        """Must use send_string, not send_multipart."""
        mock_socket = _setup_zmq_mocks(mock_zmq)

        event = ProgressEvent(event_type="train_begin")
        with RemoteProgressBridge() as bridge:
            bridge.on_progress(event)

        mock_socket.send_string.assert_called_once()
        mock_socket.send_multipart.assert_not_called()

    def test_uses_jsonpickle_encoding(self, mock_zmq):
        """Must use jsonpickle.encode, decodable by jsonpickle.decode."""
        mock_socket = _setup_zmq_mocks(mock_zmq)

        event = ProgressEvent(event_type="train_begin")
        with RemoteProgressBridge() as bridge:
            bridge.on_progress(event)

        encoded_str = mock_socket.send_string.call_args[0][0]
        # Must be decodable by jsonpickle (what LossViewer uses)
        decoded = jsonpickle.decode(encoded_str)
        assert isinstance(decoded, dict)
        assert decoded["event"] == "train_begin"

    def test_train_begin_format(self, mock_zmq):
        """train_begin must include what field and optional wandb_url."""
        mock_socket = _setup_zmq_mocks(mock_zmq)

        event = ProgressEvent(
            event_type="train_begin",
            wandb_url="https://wandb.ai/run/123",
        )

        with RemoteProgressBridge(model_type="centroid") as bridge:
            bridge.on_progress(event)

        msg = _decode_zmq_message(mock_socket)
        assert msg["event"] == "train_begin"
        assert msg["what"] == "centroid"
        assert msg["wandb_url"] == "https://wandb.ai/run/123"

    def test_epoch_begin_format(self, mock_zmq):
        """epoch_begin must include what and epoch fields."""
        mock_socket = _setup_zmq_mocks(mock_zmq)

        event = ProgressEvent(event_type="epoch_begin", epoch=5)

        with RemoteProgressBridge(model_type="centroid") as bridge:
            bridge.on_progress(event)

        msg = _decode_zmq_message(mock_socket)
        assert msg["event"] == "epoch_begin"
        assert msg["what"] == "centroid"
        assert msg["epoch"] == 5

    def test_epoch_end_format_with_logs(self, mock_zmq):
        """epoch_end must wrap loss data in logs dict with sleap-nn keys."""
        mock_socket = _setup_zmq_mocks(mock_zmq)

        event = ProgressEvent(
            event_type="epoch_end",
            epoch=10,
            train_loss=0.0045,
            val_loss=0.0051,
            metrics={"learning_rate": 0.001},
        )

        with RemoteProgressBridge(model_type="centroid") as bridge:
            bridge.on_progress(event)

        msg = _decode_zmq_message(mock_socket)
        assert msg["event"] == "epoch_end"
        assert msg["what"] == "centroid"
        # Loss data must be in "logs" dict with sleap-nn naming
        assert "logs" in msg
        assert msg["logs"]["train/loss"] == 0.0045
        assert msg["logs"]["val/loss"] == 0.0051
        assert msg["logs"]["learning_rate"] == 0.001
        # Loss must NOT be at top level (old format)
        assert "train_loss" not in msg
        assert "val_loss" not in msg

    def test_epoch_end_partial_loss(self, mock_zmq):
        """epoch_end with only train_loss should still work."""
        mock_socket = _setup_zmq_mocks(mock_zmq)

        event = ProgressEvent(
            event_type="epoch_end",
            epoch=1,
            train_loss=0.5,
        )

        with RemoteProgressBridge(model_type="centroid") as bridge:
            bridge.on_progress(event)

        msg = _decode_zmq_message(mock_socket)
        assert msg["logs"]["train/loss"] == 0.5
        assert "val/loss" not in msg["logs"]

    def test_train_end_format(self, mock_zmq):
        """train_end must include what field."""
        mock_socket = _setup_zmq_mocks(mock_zmq)

        event = ProgressEvent(event_type="train_end", success=True)

        with RemoteProgressBridge(model_type="centroid") as bridge:
            bridge.on_progress(event)

        msg = _decode_zmq_message(mock_socket)
        assert msg["event"] == "train_end"
        assert msg["what"] == "centroid"

    def test_model_type_in_all_events(self, mock_zmq):
        """All event types must include the what field."""
        mock_socket = _setup_zmq_mocks(mock_zmq)

        events = [
            ProgressEvent(event_type="train_begin"),
            ProgressEvent(event_type="epoch_begin", epoch=0),
            ProgressEvent(event_type="epoch_end", epoch=0, train_loss=0.5),
            ProgressEvent(event_type="train_end", success=True),
        ]

        with RemoteProgressBridge(model_type="centered_instance") as bridge:
            for event in events:
                bridge.on_progress(event)

        messages = _decode_all_zmq_messages(mock_socket)
        assert len(messages) == 4
        for msg in messages:
            assert msg["what"] == "centered_instance"

    def test_model_type_updates(self, mock_zmq):
        """set_model_type should affect subsequent messages."""
        mock_socket = _setup_zmq_mocks(mock_zmq)

        with RemoteProgressBridge(model_type="centroid") as bridge:
            bridge.on_progress(ProgressEvent(event_type="train_begin"))
            bridge.set_model_type("centered_instance")
            bridge.on_progress(ProgressEvent(event_type="train_begin"))

        messages = _decode_all_zmq_messages(mock_socket)
        assert messages[0]["what"] == "centroid"
        assert messages[1]["what"] == "centered_instance"

    def test_epoch_begin_synthesized_for_epoch_end(self, mock_zmq):
        """epoch_end should auto-emit epoch_begin so LossViewer tracks epoch."""
        mock_socket = _setup_zmq_mocks(mock_zmq)

        event = ProgressEvent(
            event_type="epoch_end", epoch=3, train_loss=0.1, val_loss=0.2
        )

        with RemoteProgressBridge(model_type="centroid") as bridge:
            bridge.on_progress(event)

        messages = _decode_all_zmq_messages(mock_socket)
        assert len(messages) == 2
        # First message: synthesized epoch_begin (0-indexed: 3 → 2)
        assert messages[0]["event"] == "epoch_begin"
        assert messages[0]["epoch"] == 2
        assert messages[0]["what"] == "centroid"
        # Second message: the actual epoch_end
        assert messages[1]["event"] == "epoch_end"
        assert messages[1]["logs"]["train/loss"] == 0.1

    def test_epoch_begin_not_duplicated(self, mock_zmq):
        """Explicit epoch_begin prevents duplicate synthesis."""
        mock_socket = _setup_zmq_mocks(mock_zmq)

        with RemoteProgressBridge(model_type="centroid") as bridge:
            # Explicit epoch_begin
            bridge.on_progress(ProgressEvent(event_type="epoch_begin", epoch=1))
            # epoch_end for same epoch — should NOT synthesize another epoch_begin
            bridge.on_progress(
                ProgressEvent(event_type="epoch_end", epoch=1, train_loss=0.5)
            )

        messages = _decode_all_zmq_messages(mock_socket)
        assert len(messages) == 2
        assert messages[0]["event"] == "epoch_begin"
        assert messages[1]["event"] == "epoch_end"

    def test_epoch_begin_only_on_new_epoch(self, mock_zmq):
        """Multiple epoch_end with same epoch should only synthesize once."""
        mock_socket = _setup_zmq_mocks(mock_zmq)

        with RemoteProgressBridge() as bridge:
            bridge.on_progress(
                ProgressEvent(event_type="epoch_end", epoch=1, train_loss=0.5)
            )
            bridge.on_progress(
                ProgressEvent(event_type="epoch_end", epoch=1, train_loss=0.4)
            )
            bridge.on_progress(
                ProgressEvent(event_type="epoch_end", epoch=2, train_loss=0.3)
            )

        messages = _decode_all_zmq_messages(mock_socket)
        events = [m["event"] for m in messages]
        # epoch_begin(1), epoch_end, epoch_end, epoch_begin(2), epoch_end
        assert events == [
            "epoch_begin", "epoch_end", "epoch_end",
            "epoch_begin", "epoch_end",
        ]

    def test_unknown_event_type(self, mock_zmq):
        """Unknown event types should be silently dropped."""
        mock_socket = _setup_zmq_mocks(mock_zmq)

        event = ProgressEvent(event_type="unknown_event")

        with RemoteProgressBridge() as bridge:
            bridge.on_progress(event)

        mock_socket.send_string.assert_not_called()


# =============================================================================
# format_progress_line Tests
# =============================================================================


class TestFormatProgressLine:
    """Tests for format_progress_line function."""

    def test_format_train_begin(self):
        """Should format train_begin with separators."""
        event = ProgressEvent(
            event_type="train_begin",
            total_epochs=100,
            wandb_url="https://wandb.ai/run/123",
        )
        line = format_progress_line(event)

        assert "Training started" in line
        assert "WandB: https://wandb.ai/run/123" in line
        assert "Total epochs: 100" in line
        assert "─" * 60 in line

    def test_format_epoch_end(self):
        """Should format epoch_end with metrics."""
        event = ProgressEvent(
            event_type="epoch_end",
            epoch=10,
            total_epochs=100,
            train_loss=0.5,
            val_loss=0.6,
        )
        line = format_progress_line(event)

        assert "Epoch 10/100" in line
        assert "train_loss=0.5000" in line
        assert "val_loss=0.6000" in line

    def test_format_epoch_end_no_total(self):
        """Should format epoch without total epochs."""
        event = ProgressEvent(
            event_type="epoch_end",
            epoch=10,
            train_loss=0.5,
        )
        line = format_progress_line(event)

        assert "Epoch 10 -" in line
        assert "/100" not in line

    def test_format_train_end_success(self):
        """Should format successful train_end."""
        event = ProgressEvent(
            event_type="train_end",
            success=True,
        )
        line = format_progress_line(event)

        assert "Training completed successfully" in line
        assert "─" * 60 in line

    def test_format_train_end_failure(self):
        """Should format failed train_end with error."""
        event = ProgressEvent(
            event_type="train_end",
            success=False,
            error_message="CUDA out of memory",
        )
        line = format_progress_line(event)

        assert "Training failed" in line
        assert "Error: CUDA out of memory" in line


# =============================================================================
# run_remote_training Tests
# =============================================================================


class TestRunRemoteTraining:
    """Tests for run_remote_training function."""

    @patch("sleap_connect.api.run_training")
    def test_basic_call(self, mock_run_training, mock_zmq):
        """Should call run_training with correct args."""
        mock_socket = _setup_zmq_mocks(mock_zmq)

        mock_result = MagicMock()
        mock_run_training.return_value = mock_result

        result = run_remote_training(
            config_path="/path/to/config.json",
            room_id="test-room",
            worker_id="worker-1",
        )

        mock_run_training.assert_called_once()
        call_kwargs = mock_run_training.call_args[1]
        assert call_kwargs["config_path"] == "/path/to/config.json"
        assert call_kwargs["room_id"] == "test-room"
        assert call_kwargs["worker_id"] == "worker-1"
        assert result == mock_result

    @patch("sleap_connect.api.run_training")
    def test_on_raw_progress_wired_to_bridge(self, mock_run_training, mock_zmq):
        """Should wire on_raw_progress to bridge.on_raw_zmq_message."""
        mock_socket = _setup_zmq_mocks(mock_zmq)

        mock_run_training.return_value = MagicMock()

        run_remote_training(
            config_path="/path/to/config.json",
            room_id="test-room",
            model_type="centroid",
        )

        call_kwargs = mock_run_training.call_args[1]
        assert call_kwargs["on_raw_progress"] is not None

    @patch("sleap_connect.api.run_training")
    def test_raw_zmq_passthrough(self, mock_run_training, mock_zmq):
        """Should forward raw ZMQ messages directly to LossViewer socket."""
        mock_socket = _setup_zmq_mocks(mock_zmq)

        def simulate_training(*args, **kwargs):
            on_raw = kwargs.get("on_raw_progress")
            if on_raw:
                on_raw('{"event": "batch_end", "what": "centroid"}')
                on_raw('{"event": "epoch_end", "what": "centroid"}')
            return MagicMock()

        mock_run_training.side_effect = simulate_training

        run_remote_training(
            config_path="/path/to/config.json",
            room_id="test-room",
        )

        assert mock_socket.send_string.call_count == 2
        # Raw strings should be passed through without re-encoding
        mock_socket.send_string.assert_any_call(
            '{"event": "batch_end", "what": "centroid"}'
        )
        mock_socket.send_string.assert_any_call(
            '{"event": "epoch_end", "what": "centroid"}'
        )

    @patch("sleap_connect.api.run_training")
    def test_progress_callback_no_longer_publishes_to_zmq(self, mock_run_training, mock_zmq):
        """Progress callback should NOT publish to ZMQ (raw passthrough handles that)."""
        mock_socket = _setup_zmq_mocks(mock_zmq)

        def simulate_training(*args, **kwargs):
            on_progress = kwargs.get("progress_callback")
            if on_progress:
                on_progress(ProgressEvent(event_type="train_begin"))
                on_progress(
                    ProgressEvent(event_type="epoch_end", epoch=1, train_loss=0.5)
                )
                on_progress(ProgressEvent(event_type="train_end", success=True))
            return MagicMock()

        mock_run_training.side_effect = simulate_training

        run_remote_training(
            config_path="/path/to/config.json",
            room_id="test-room",
        )

        # No ZMQ messages from progress_callback — LossViewer is fed by raw passthrough
        mock_socket.send_string.assert_not_called()

    @patch("sleap_connect.api.run_training")
    def test_custom_callback_called(self, mock_run_training, mock_zmq):
        """Should call custom progress callback."""
        mock_socket = _setup_zmq_mocks(mock_zmq)

        callback_events = []

        def my_callback(event):
            callback_events.append(event)

        def simulate_training(*args, **kwargs):
            on_progress = kwargs.get("progress_callback")
            if on_progress:
                on_progress(ProgressEvent(event_type="train_begin"))
            return MagicMock()

        mock_run_training.side_effect = simulate_training

        run_remote_training(
            config_path="/path/to/config.json",
            room_id="test-room",
            on_progress=my_callback,
        )

        assert len(callback_events) == 1
        assert callback_events[0].event_type == "train_begin"

    @patch("sleap_connect.api.run_training")
    def test_on_model_type_updates_bridge_and_calls_user_callback(
        self, mock_run_training, mock_zmq
    ):
        """Bridge model type must update even when a custom on_model_type is provided.

        Regression test: previously, providing on_model_type caused bridge.set_model_type
        to be skipped. This meant model 2's ZMQ messages had the wrong 'what' field and
        were filtered out by the LossViewer in multi-model pipelines.
        """
        import jsonpickle

        mock_socket = _setup_zmq_mocks(mock_zmq)
        user_callback_calls = []

        def simulate_training(*args, **kwargs):
            on_model_type = kwargs.get("on_model_type")
            on_raw = kwargs.get("on_raw_progress")
            # Simulate worker switching to model 2
            if on_model_type:
                on_model_type("centered_instance")
            # Simulate model 2's ZMQ progress (no 'what' field — sleap-nn doesn't set it)
            if on_raw:
                on_raw(jsonpickle.encode({"event": "epoch_end", "logs": {}}))
            return MagicMock()

        mock_run_training.side_effect = simulate_training

        run_remote_training(
            config_path="/path/to/config.json",
            room_id="test-room",
            model_type="centroid",
            on_model_type=lambda mt: user_callback_calls.append(mt),
        )

        # User callback must have been called
        assert user_callback_calls == ["centered_instance"]

        # The ZMQ message forwarded to LossViewer must use the updated model type
        messages = _decode_all_zmq_messages(mock_socket)
        assert len(messages) == 1
        assert messages[0]["what"] == "centered_instance", (
            "bridge._model_type was not updated — model 2 messages would be "
            "filtered out by LossViewer"
        )

    @patch("sleap_connect.api.run_training")
    def test_model_type_passed_to_run_training(self, mock_run_training, mock_zmq):
        """Should pass model_type through to run_training API call."""
        mock_socket = _setup_zmq_mocks(mock_zmq)

        mock_run_training.return_value = MagicMock()

        run_remote_training(
            config_path="/path/to/config.json",
            room_id="test-room",
            model_type="centroid",
        )

        call_kwargs = mock_run_training.call_args[1]
        assert call_kwargs["model_type"] == "centroid"

    @patch("sleap_connect.api.run_training")
    def test_on_log_passed_to_run_training(self, mock_run_training, mock_zmq):
        """Should pass on_log through to run_training API call."""
        mock_socket = _setup_zmq_mocks(mock_zmq)

        mock_run_training.return_value = MagicMock()

        log_lines = []
        run_remote_training(
            config_path="/path/to/config.json",
            room_id="test-room",
            on_log=lambda line: log_lines.append(line),
        )

        # on_log should be passed through as-is
        call_kwargs = mock_run_training.call_args[1]
        assert call_kwargs["on_log"] is not None

    @patch("sleap_connect.api.run_training")
    def test_default_on_log_prints_to_stdout(self, mock_run_training, mock_zmq, capsys):
        """When on_log is not provided, raw log lines should print to stdout."""
        mock_socket = _setup_zmq_mocks(mock_zmq)

        def simulate_training(*args, **kwargs):
            # Simulate the on_log callback being invoked with a raw log line
            on_log = kwargs.get("on_log")
            if on_log:
                on_log("Epoch 1/10: loss=0.5\n")
            return MagicMock()

        mock_run_training.side_effect = simulate_training

        run_remote_training(
            config_path="/path/to/config.json",
            room_id="test-room",
        )

        captured = capsys.readouterr()
        assert "Epoch 1/10: loss=0.5" in captured.out

    @patch("sleap_connect.api.run_training")
    def test_custom_on_log_receives_lines(self, mock_run_training, mock_zmq):
        """Custom on_log callback should receive raw log lines."""
        mock_socket = _setup_zmq_mocks(mock_zmq)

        log_lines = []

        def simulate_training(*args, **kwargs):
            on_log = kwargs.get("on_log")
            if on_log:
                on_log("line 1\n")
                on_log("line 2\n")
            return MagicMock()

        mock_run_training.side_effect = simulate_training

        run_remote_training(
            config_path="/path/to/config.json",
            room_id="test-room",
            on_log=lambda line: log_lines.append(line),
        )

        assert log_lines == ["line 1\n", "line 2\n"]

    @patch("sleap_connect.api.run_training")
    def test_progress_also_printed_to_terminal(self, mock_run_training, mock_zmq):
        """Structured progress events should also be formatted and sent to on_log."""
        mock_socket = _setup_zmq_mocks(mock_zmq)

        log_lines = []

        def simulate_training(*args, **kwargs):
            on_progress = kwargs.get("progress_callback")
            if on_progress:
                on_progress(
                    ProgressEvent(event_type="epoch_end", epoch=1, train_loss=0.5)
                )
            return MagicMock()

        mock_run_training.side_effect = simulate_training

        run_remote_training(
            config_path="/path/to/config.json",
            room_id="test-room",
            on_log=lambda line: log_lines.append(line),
        )

        # Should have received the formatted progress line
        assert len(log_lines) == 1
        assert "Epoch 1" in log_lines[0]
        assert "train_loss=0.5" in log_lines[0]

    @patch("sleap_connect.api.run_training")
    def test_on_channel_ready_wired_to_bridge(self, mock_run_training, mock_zmq):
        """Should pass bridge.set_send_fn as on_channel_ready."""
        mock_socket = _setup_zmq_mocks(mock_zmq)

        mock_run_training.return_value = MagicMock()

        run_remote_training(
            config_path="/path/to/config.json",
            room_id="test-room",
        )

        call_kwargs = mock_run_training.call_args[1]
        assert call_kwargs["on_channel_ready"] is not None


# =============================================================================
# Stop/Cancel Command Tests
# =============================================================================


class TestStopCancelCommands:
    """Tests for stop/cancel command forwarding from LossViewer to worker."""

    def test_set_send_fn(self):
        """set_send_fn should store the send function."""
        bridge = RemoteProgressBridge()
        send_fn = MagicMock()
        bridge.set_send_fn(send_fn)
        assert bridge._send_fn is send_fn

    def test_poll_commands_stop(self, mock_zmq):
        """Should forward 'stop' as CONTROL_COMMAND::{raw} via send_fn."""
        import jsonpickle

        from sleap_connect.protocol import MSG_CONTROL_COMMAND, MSG_SEPARATOR

        mock_socket = _setup_zmq_mocks(mock_zmq)
        mock_context = mock_zmq.Context.return_value

        # Get the SUB socket that will be created
        sub_sockets = []
        original_side_effect = mock_context.socket.side_effect

        def _capture_sockets(socket_type):
            sock = original_side_effect(socket_type)
            if socket_type == mock_zmq.SUB:
                sub_sockets.append(sock)
            return sock

        mock_context.socket.side_effect = _capture_sockets

        bridge = RemoteProgressBridge()
        send_fn = MagicMock()
        bridge.set_send_fn(send_fn)

        # Configure the SUB socket to return a stop command once, then nothing
        bridge.start()
        sub_sock = sub_sockets[0]
        sub_sock.poll.side_effect = [1, 0, 0, 0, 0]  # Message available once
        raw = jsonpickle.encode({"command": "stop"})
        sub_sock.recv_string.return_value = raw

        # Give the poll thread time to process
        import time

        time.sleep(0.3)
        bridge.stop()

        send_fn.assert_called_with(f"{MSG_CONTROL_COMMAND}{MSG_SEPARATOR}{raw}")

    def test_poll_commands_cancel(self, mock_zmq):
        """Should forward 'cancel' command as MSG_JOB_CANCEL via send_fn."""
        import jsonpickle

        from sleap_connect.protocol import MSG_JOB_CANCEL

        mock_socket = _setup_zmq_mocks(mock_zmq)
        mock_context = mock_zmq.Context.return_value

        sub_sockets = []
        original_side_effect = mock_context.socket.side_effect

        def _capture_sockets(socket_type):
            sock = original_side_effect(socket_type)
            if socket_type == mock_zmq.SUB:
                sub_sockets.append(sock)
            return sock

        mock_context.socket.side_effect = _capture_sockets

        bridge = RemoteProgressBridge()
        send_fn = MagicMock()
        bridge.set_send_fn(send_fn)

        bridge.start()
        sub_sock = sub_sockets[0]
        sub_sock.poll.side_effect = [1, 0, 0, 0, 0]
        sub_sock.recv_string.return_value = jsonpickle.encode({"command": "cancel"})

        import time

        time.sleep(0.3)
        bridge.stop()

        send_fn.assert_called_with(MSG_JOB_CANCEL)

    def test_poll_no_send_fn(self, mock_zmq):
        """Should not crash if stop received but no send_fn set."""
        import jsonpickle

        mock_socket = _setup_zmq_mocks(mock_zmq)
        mock_context = mock_zmq.Context.return_value

        sub_sockets = []
        original_side_effect = mock_context.socket.side_effect

        def _capture_sockets(socket_type):
            sock = original_side_effect(socket_type)
            if socket_type == mock_zmq.SUB:
                sub_sockets.append(sock)
            return sock

        mock_context.socket.side_effect = _capture_sockets

        bridge = RemoteProgressBridge()
        # Don't set send_fn

        bridge.start()
        sub_sock = sub_sockets[0]
        sub_sock.poll.side_effect = [1, 0, 0, 0, 0]
        sub_sock.recv_string.return_value = jsonpickle.encode({"command": "stop"})

        import time

        time.sleep(0.3)
        bridge.stop()
        # Should not raise — just log a warning

    def test_controller_port_configurable(self):
        """Should accept custom controller_port."""
        bridge = RemoteProgressBridge(controller_port=8888)
        assert bridge._controller_port == 8888


# =============================================================================
# JobExecutor Stop/Cancel Tests
# =============================================================================


class TestJobExecutorStopCancel:
    """Tests for JobExecutor stop/cancel methods."""

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix process groups only")
    def test_stop_running_job_sends_sigint(self):
        """stop_running_job should send SIGINT to the process group."""
        import signal
        from unittest.mock import patch
        from sleap_connect.worker.job_executor import JobExecutor

        executor = JobExecutor(worker=MagicMock(), capabilities=MagicMock())
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.returncode = None  # Still running
        executor._running_process = mock_process

        with patch("os.killpg") as mock_killpg, \
             patch("os.getpgid", return_value=99999):
            executor.stop_running_job()

        mock_killpg.assert_called_once_with(99999, signal.SIGINT)

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix process groups only")
    def test_cancel_running_job_sends_sigterm(self):
        """cancel_running_job should send SIGTERM to the process group."""
        import signal
        from unittest.mock import patch
        from sleap_connect.worker.job_executor import JobExecutor

        executor = JobExecutor(worker=MagicMock(), capabilities=MagicMock())
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.returncode = None
        executor._running_process = mock_process

        with patch("os.killpg") as mock_killpg, \
             patch("os.getpgid", return_value=99999):
            executor.cancel_running_job()

        mock_killpg.assert_called_once_with(99999, signal.SIGTERM)

    def test_stop_no_running_process(self):
        """stop_running_job should not crash when no process is running."""
        from sleap_connect.worker.job_executor import JobExecutor

        executor = JobExecutor(worker=MagicMock(), capabilities=MagicMock())
        executor._running_process = None

        # Should not raise
        executor.stop_running_job()

    def test_stop_already_exited_process(self):
        """stop_running_job should not signal an already-exited process."""
        from sleap_connect.worker.job_executor import JobExecutor

        executor = JobExecutor(worker=MagicMock(), capabilities=MagicMock())
        mock_process = MagicMock()
        mock_process.returncode = 0  # Already exited
        executor._running_process = mock_process

        executor.stop_running_job()

        mock_process.send_signal.assert_not_called()


# =============================================================================
# InferenceProgressDialog integration via RemoteProgressBridge
# =============================================================================


@pytest.fixture(scope="session")
def qapp():
    """QApplication instance required for Qt widget tests."""
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestHandleInferenceMessage:
    """Tests for RemoteProgressBridge.handle_inference_message."""

    def test_inference_begin_creates_dialog(self, qapp):
        """INFERENCE_BEGIN should create and show an InferenceProgressDialog."""
        bridge = RemoteProgressBridge()
        assert bridge._inference_dialog is None
        bridge._dispatch_inference_msg("INFERENCE_BEGIN", {})
        assert bridge._inference_dialog is not None
        bridge._inference_dialog.close()

    def test_inference_progress_updates_dialog(self, qapp):
        """INFERENCE_PROGRESS should call dialog.update with the payload."""
        bridge = RemoteProgressBridge()
        bridge._dispatch_inference_msg("INFERENCE_BEGIN", {})
        data = {"n_processed": 50, "n_total": 100, "rate": 10.0, "eta": 5}
        bridge._dispatch_inference_msg("INFERENCE_PROGRESS", data)
        assert bridge._inference_dialog._progress_bar.value() == 50
        bridge._inference_dialog.close()

    def test_inference_progress_without_begin_noop(self, qapp):
        """INFERENCE_PROGRESS before INFERENCE_BEGIN should not crash."""
        bridge = RemoteProgressBridge()
        # Should not raise — dialog is None so update is silently skipped
        bridge._dispatch_inference_msg(
            "INFERENCE_PROGRESS", {"n_processed": 5, "n_total": 10}
        )

    def test_inference_complete_calls_finish(self, qapp):
        """INFERENCE_COMPLETE should call dialog.finish and show summary."""
        bridge = RemoteProgressBridge()
        bridge._dispatch_inference_msg("INFERENCE_BEGIN", {})
        bridge._dispatch_inference_msg(
            "INFERENCE_COMPLETE",
            {
                "predictions_path": "/tmp/labels.predictions.slp",
                "n_frames": 100,
                "n_with_instances": 80,
                "n_empty": 20,
            },
        )
        from qtpy.QtWidgets import QDialogButtonBox

        ok_btn = bridge._inference_dialog._button_box.button(QDialogButtonBox.Ok)
        assert ok_btn.isEnabled(), "OK button should be enabled after finish"
        bridge._inference_dialog.close()

    def test_inference_complete_calls_predictions_callback(self, qapp):
        """INFERENCE_COMPLETE should invoke the on_predictions_ready callback."""
        bridge = RemoteProgressBridge()
        received = []
        bridge.set_predictions_ready_callback(received.append)
        bridge._dispatch_inference_msg("INFERENCE_BEGIN", {})
        bridge._dispatch_inference_msg(
            "INFERENCE_COMPLETE",
            {"predictions_path": "/tmp/labels.predictions.slp"},
        )
        assert received == ["/tmp/labels.predictions.slp"]
        bridge._inference_dialog.close()

    def test_inference_complete_no_callback_no_crash(self, qapp):
        """INFERENCE_COMPLETE without callback should not crash."""
        bridge = RemoteProgressBridge()
        bridge._dispatch_inference_msg("INFERENCE_BEGIN", {})
        bridge._dispatch_inference_msg(
            "INFERENCE_COMPLETE",
            {"predictions_path": "/tmp/labels.predictions.slp"},
        )
        bridge._inference_dialog.close()

    def test_inference_failed_calls_show_error(self, qapp):
        """INFERENCE_FAILED should call dialog.show_error with the message."""
        bridge = RemoteProgressBridge()
        bridge._dispatch_inference_msg("INFERENCE_BEGIN", {})
        bridge._dispatch_inference_msg(
            "INFERENCE_FAILED", {"error": "out of memory"}
        )
        log = bridge._inference_dialog._log_text.toPlainText()
        assert "out of memory" in log
        bridge._inference_dialog.close()

    def test_inference_failed_without_begin_opens_dialog(self, qapp):
        """INFERENCE_FAILED without a prior INFERENCE_BEGIN should open a dialog."""
        bridge = RemoteProgressBridge()
        assert bridge._inference_dialog is None
        bridge._dispatch_inference_msg(
            "INFERENCE_FAILED", {"error": "subprocess crashed"}
        )
        assert bridge._inference_dialog is not None
        bridge._inference_dialog.close()

    def test_inference_skipped_no_dialog(self, qapp):
        """INFERENCE_SKIPPED should log but not open a dialog."""
        bridge = RemoteProgressBridge()
        bridge._dispatch_inference_msg(
            "INFERENCE_SKIPPED", {"reason": "training_failed"}
        )
        assert bridge._inference_dialog is None

    def test_inference_log_appends_to_dialog(self, qapp):
        """INFERENCE_LOG should append text to the dialog's log area."""
        bridge = RemoteProgressBridge()
        bridge._dispatch_inference_msg("INFERENCE_BEGIN", {})
        bridge._dispatch_inference_msg(
            "INFERENCE_LOG", {"text": "2026-01-01 | Started inference at: ..."}
        )
        log = bridge._inference_dialog._log_text.toPlainText()
        assert "Started inference" in log
        bridge._inference_dialog.close()

    def test_inference_log_updates_progress_bar(self, qapp):
        """INFERENCE_LOG with a rich progress line should advance the progress bar."""
        bridge = RemoteProgressBridge()
        bridge._dispatch_inference_msg("INFERENCE_BEGIN", {})
        bridge._dispatch_inference_msg(
            "INFERENCE_LOG",
            {"text": "Predicting... 100% 35/35 ETA: 0:00:00 Elapsed: 0:00:01 47.8 FPS"},
        )
        assert bridge._inference_dialog._progress_bar.value() == 100
        bridge._inference_dialog.close()

    def test_inference_log_caches_n_frames(self, qapp):
        """INFERENCE_LOG parsing should cache n_frames for INFERENCE_COMPLETE."""
        bridge = RemoteProgressBridge()
        bridge._dispatch_inference_msg("INFERENCE_BEGIN", {})
        bridge._dispatch_inference_msg(
            "INFERENCE_LOG",
            {"text": "Predicting... 100% 35/35 ETA: 0:00:00 47.8 FPS"},
        )
        assert bridge._last_n_frames == 35
        bridge._inference_dialog.close()

    def test_inference_complete_uses_cached_n_frames(self, qapp):
        """INFERENCE_COMPLETE should use cached n_frames when payload lacks it."""
        bridge = RemoteProgressBridge()
        bridge._dispatch_inference_msg("INFERENCE_BEGIN", {})
        bridge._dispatch_inference_msg(
            "INFERENCE_LOG",
            {"text": "Predicting... 100% 35/35 ETA: 0:00:00 47.8 FPS"},
        )
        bridge._dispatch_inference_msg(
            "INFERENCE_COMPLETE", {"predictions_path": "/tmp/out.slp"}
        )
        # Frame count appears in the status label (not the log)
        label = bridge._inference_dialog._status_label.text()
        assert "35" in label
        bridge._inference_dialog.close()


class TestHandleJobMessage:
    """Task 9: ``RemoteProgressBridge.handle_job_message`` mirrors
    ``handle_inference_message`` for the standalone-track flow's JOB_*
    wire prefix.  Uses a patched ``InferenceProgressDialog`` so the tests
    run without a Qt event loop.
    """

    def test_job_log_creates_dialog_and_appends(self):
        bridge = RemoteProgressBridge()
        with patch("sleap_connect.gui.widgets.InferenceProgressDialog") as MockDialog:
            instance = MagicMock()
            MockDialog.return_value = instance
            bridge._dispatch_job_msg(
                "JOB_LOG",
                {"text": "Predicting... 50% 17/35 ETA: 0:00:01 47.8 FPS"},
            )
            MockDialog.assert_called_once()
            instance.show.assert_called_once()
            instance.append_log.assert_called_once_with(
                "Predicting... 50% 17/35 ETA: 0:00:01 47.8 FPS"
            )
            instance.set_progress.assert_called_once_with(17, 35, 47.8, 1)
        assert bridge._last_n_frames == 35

    def test_job_log_with_cr_prefix_strips_before_regex(self):
        bridge = RemoteProgressBridge()
        with patch("sleap_connect.gui.widgets.InferenceProgressDialog") as MockDialog:
            instance = MagicMock()
            MockDialog.return_value = instance
            bridge._dispatch_job_msg(
                "JOB_LOG",
                {"text": "CR::Predicting... 100% 35/35 ETA: 0:00:00 47.8 FPS"},
            )
            # CR:: prefix must be stripped before the line is shown / parsed.
            instance.append_log.assert_called_once_with(
                "Predicting... 100% 35/35 ETA: 0:00:00 47.8 FPS"
            )
            instance.set_progress.assert_called_once_with(35, 35, 47.8, 0)
        assert bridge._last_n_frames == 35

    def test_job_progress_calls_update(self):
        bridge = RemoteProgressBridge()
        with patch("sleap_connect.gui.widgets.InferenceProgressDialog") as MockDialog:
            instance = MagicMock()
            MockDialog.return_value = instance
            payload = {"n_processed": 10, "n_total": 50}
            bridge._dispatch_job_msg("JOB_PROGRESS", payload)
            instance.update.assert_called_once_with(payload)

    def test_job_complete_calls_finish_and_predictions_ready(self):
        bridge = RemoteProgressBridge()
        received: list[str] = []
        bridge.set_predictions_ready_callback(received.append)
        with patch("sleap_connect.gui.widgets.InferenceProgressDialog") as MockDialog:
            instance = MagicMock()
            MockDialog.return_value = instance
            # Open the dialog with an initial JOB_LOG so the COMPLETE branch
            # has something to call finish() on.
            bridge._dispatch_job_msg(
                "JOB_LOG",
                {"text": "Predicting... 100% 35/35 ETA: 0:00:00 47.8 FPS"},
            )
            bridge._dispatch_job_msg(
                "JOB_COMPLETE",
                {
                    "output_path": "/tmp/out.slp",
                    "n_with_instances": 30,
                    "n_empty": 5,
                },
            )
            instance.finish.assert_called_once_with(35, 30, 5)
        assert received == ["/tmp/out.slp"]

    def test_job_failed_with_no_dialog_creates_one_and_shows_error(self):
        bridge = RemoteProgressBridge()
        assert bridge._inference_dialog is None
        with patch("sleap_connect.gui.widgets.InferenceProgressDialog") as MockDialog:
            instance = MagicMock()
            MockDialog.return_value = instance
            bridge._dispatch_job_msg("JOB_FAILED", {"message": "subprocess crashed"})
            MockDialog.assert_called_once()
            instance.show.assert_called_once()
            instance.show_error.assert_called_once_with("subprocess crashed")

    def test_job_failed_with_existing_dialog_reuses_it(self):
        bridge = RemoteProgressBridge()
        with patch("sleap_connect.gui.widgets.InferenceProgressDialog") as MockDialog:
            instance = MagicMock()
            MockDialog.return_value = instance
            bridge._dispatch_job_msg("JOB_LOG", {"text": "Booting..."})
            assert MockDialog.call_count == 1
            bridge._dispatch_job_msg("JOB_FAILED", {"message": "GPU OOM"})
            # Dialog must NOT be reconstructed — show_error on the same one.
            assert MockDialog.call_count == 1
            instance.show_error.assert_called_once_with("GPU OOM")

    def test_show_eager_dialog_creates_waiting_state(self):
        bridge = RemoteProgressBridge()
        with patch("sleap_connect.gui.widgets.InferenceProgressDialog") as mock_cls:
            mock_dialog = MagicMock()
            mock_cls.return_value = mock_dialog

            bridge.show_eager_dialog()

            mock_cls.assert_called_once()
            mock_dialog.show_waiting.assert_called_once()

    def test_first_job_log_transitions_from_waiting_to_live(self):
        bridge = RemoteProgressBridge()
        with patch("sleap_connect.gui.widgets.InferenceProgressDialog") as mock_cls:
            mock_dialog = MagicMock()
            mock_dialog._progress_bar.maximum.return_value = 0  # indeterminate
            mock_cls.return_value = mock_dialog

            bridge.show_eager_dialog()
            bridge._dispatch_job_msg("JOB_LOG", {"text": "Predicting... 1/10"})

            # Should NOT create a second dialog
            assert mock_cls.call_count == 1
            # Should transition from waiting to live
            mock_dialog._status_label.setText.assert_called_with("Running inference…")
            mock_dialog._progress_bar.setRange.assert_called_with(0, 100)

    def test_inference_dispatch_unchanged(self, qapp):
        """Regression: existing INFERENCE_* dispatch path is unaffected."""
        bridge = RemoteProgressBridge()
        bridge._dispatch_inference_msg("INFERENCE_BEGIN", {})
        bridge._dispatch_inference_msg(
            "INFERENCE_LOG",
            {"text": "Predicting... 100% 35/35 ETA: 0:00:00 47.8 FPS"},
        )
        assert bridge._last_n_frames == 35
        assert bridge._inference_dialog is not None
        bridge._inference_dialog.close()


class TestDispatchJobMsgMultiSpec:
    """Verify _dispatch_job_msg reads job_index/jobs_total for multi-spec."""

    def test_first_job_log_shows_video_1_of_3(self):
        bridge = RemoteProgressBridge()
        with patch("sleap_connect.gui.widgets.InferenceProgressDialog") as MockDialog:
            instance = MagicMock()
            MockDialog.return_value = instance
            bridge._dispatch_job_msg("JOB_LOG", {
                "text": "Starting...", "job_index": 0, "jobs_total": 3,
            })
            MockDialog.assert_called_once()
            instance.show.assert_called_once()
            # After creating dialog with multi-spec, status should show video count
            instance._status_label.setText.assert_called_with(
                "Running inference… (video 1 of 3)"
            )

    def test_second_spec_log_updates_to_video_2_of_3(self):
        bridge = RemoteProgressBridge()
        with patch("sleap_connect.gui.widgets.InferenceProgressDialog") as MockDialog:
            instance = MagicMock()
            # First call: dialog just created, maximum returns default (not 0)
            # After first JOB_LOG creates dialog, subsequent calls check maximum()
            instance._progress_bar.maximum.return_value = 100
            MockDialog.return_value = instance
            # First spec
            bridge._dispatch_job_msg("JOB_LOG", {
                "text": "line1", "job_index": 0, "jobs_total": 3,
            })
            # Complete first spec (mid-batch)
            bridge._dispatch_job_msg("JOB_COMPLETE", {
                "output_path": "/tmp/pred0.slp", "job_index": 0, "jobs_total": 3,
            })
            # Second spec starts — after mid-batch complete, progress bar was
            # reset to indeterminate (max=0)
            instance._progress_bar.maximum.return_value = 0
            bridge._dispatch_job_msg("JOB_LOG", {
                "text": "line2", "job_index": 1, "jobs_total": 3,
            })
            # The last setText call should be the video 2 of 3 label
            calls = instance._status_label.setText.call_args_list
            last_label = calls[-1][0][0]
            assert "2 of 3" in last_label

    def test_single_spec_no_video_label(self):
        """When jobs_total=1, don't show 'video 1 of 1' — same as current behavior."""
        bridge = RemoteProgressBridge()
        with patch("sleap_connect.gui.widgets.InferenceProgressDialog") as MockDialog:
            instance = MagicMock()
            MockDialog.return_value = instance
            bridge._dispatch_job_msg("JOB_LOG", {
                "text": "line1", "job_index": 0, "jobs_total": 1,
            })
            # Status label should NOT contain "of" — no multi-spec label
            for call in instance._status_label.setText.call_args_list:
                assert "of" not in call[0][0].lower()

    def test_no_enrichment_fields_works_like_before(self):
        """Without job_index/jobs_total (pre-batch callers), no crash."""
        bridge = RemoteProgressBridge()
        with patch("sleap_connect.gui.widgets.InferenceProgressDialog") as MockDialog:
            instance = MagicMock()
            MockDialog.return_value = instance
            bridge._dispatch_job_msg("JOB_LOG", {"text": "hello"})
            MockDialog.assert_called_once()
            instance.show.assert_called_once()
            instance.append_log.assert_called_once_with("hello")

    def test_job_complete_mid_batch_resets_progress(self):
        """JOB_COMPLETE for a non-final spec resets progress bar to indeterminate."""
        bridge = RemoteProgressBridge()
        with patch("sleap_connect.gui.widgets.InferenceProgressDialog") as MockDialog:
            instance = MagicMock()
            MockDialog.return_value = instance
            # Open dialog
            bridge._dispatch_job_msg("JOB_LOG", {
                "text": "Predicting... 50% 5/10", "job_index": 0, "jobs_total": 3,
            })
            # Complete first spec (non-final)
            bridge._dispatch_job_msg("JOB_COMPLETE", {
                "output_path": "/tmp/pred.slp", "job_index": 0, "jobs_total": 3,
            })
            # Should NOT call finish() — mid-batch
            instance.finish.assert_not_called()
            # Should reset to indeterminate
            instance._progress_bar.setRange.assert_called_with(0, 0)
            instance._progress_bar.setValue.assert_called_with(0)

    def test_job_complete_final_spec_calls_finish(self):
        """JOB_COMPLETE for the final spec calls finish() as normal."""
        bridge = RemoteProgressBridge()
        with patch("sleap_connect.gui.widgets.InferenceProgressDialog") as MockDialog:
            instance = MagicMock()
            MockDialog.return_value = instance
            # Open dialog
            bridge._dispatch_job_msg("JOB_LOG", {
                "text": "Predicting... 100% 10/10", "job_index": 2, "jobs_total": 3,
            })
            bridge._dispatch_job_msg("JOB_COMPLETE", {
                "output_path": "/tmp/pred.slp", "job_index": 2, "jobs_total": 3,
            })
            # Final spec — should call finish()
            instance.finish.assert_called_once()

    def test_mid_batch_complete_still_fires_predictions_ready(self):
        """on_predictions_ready fires for every spec, not just the final one."""
        bridge = RemoteProgressBridge()
        received: list[str] = []
        bridge.set_predictions_ready_callback(received.append)
        with patch("sleap_connect.gui.widgets.InferenceProgressDialog") as MockDialog:
            instance = MagicMock()
            MockDialog.return_value = instance
            bridge._dispatch_job_msg("JOB_LOG", {
                "text": "line1", "job_index": 0, "jobs_total": 3,
            })
            bridge._dispatch_job_msg("JOB_COMPLETE", {
                "output_path": "/tmp/pred0.slp", "job_index": 0, "jobs_total": 3,
            })
        assert received == ["/tmp/pred0.slp"]
