"""Remote training runners for SLEAP GUI integration.

This module provides the bridge between sleap-connect's WebRTC-based progress
reporting and SLEAP's LossViewer which expects ZMQ messages.

The flow is:
    Worker (sleap-nn) → ZMQ → WebRTC DataChannel → Client → ZMQ → LossViewer
"""

from __future__ import annotations

import re
import threading
from typing import TYPE_CHECKING, Callable

from loguru import logger
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QApplication

if TYPE_CHECKING:
    from sleap_connect.api import ProgressEvent, TrainingResult
    from sleap_connect.jobs.spec import TrainJobSpec


class RemoteProgressBridge(QObject):
    """Bridge WebRTC progress messages to ZMQ for LossViewer compatibility.

    This class receives ProgressEvent objects from the sleap-connect training API
    and re-emits them via ZMQ in the exact format expected by SLEAP's
    LossViewer widget (sleap/gui/widgets/monitor.py).

    LossViewer parses messages with ``jsonpickle.decode(sub.recv_string())``
    and expects:
    - ``what``: model type string for filtering (e.g., "centroid")
    - ``logs``: dict with ``train/loss`` and ``val/loss`` keys
    - ``event``: one of train_begin, epoch_begin, epoch_end, train_end

    The LossViewer SUB socket **binds** to a port, so this bridge's PUB
    socket must **connect** to it (not bind).

    Bidirectional support:
    - **PUB socket** (publish_port): Sends progress updates TO LossViewer.
    - **SUB socket** (controller_port): Receives stop/cancel commands FROM
      LossViewer's controller PUB socket. Commands are forwarded over the
      RTC data channel via ``send_fn``.
    """

    # Signal used to marshal inference messages from the WebRTC background
    # thread to the main Qt thread.  Qt guarantees the connected slot runs
    # on the thread the QObject was created on (the main thread) even when
    # the signal is emitted from a different thread.
    _sig_inference_msg: Signal = Signal(str, object)
    # Mirror signal for JOB_* messages from the standalone-track flow.
    _sig_job_msg: Signal = Signal(str, object)

    def __init__(
        self,
        publish_port: int = 9001,
        controller_port: int = 9000,
        model_type: str = "",
    ):
        """Initialize the progress bridge.

        Args:
            publish_port: ZMQ port where LossViewer's SUB socket is bound.
            controller_port: ZMQ port where LossViewer's controller PUB
                socket is bound. The bridge connects a SUB socket here to
                receive stop/cancel commands.
            model_type: Model type string for the ``what`` field (e.g.,
                "centroid", "centered_instance"). LossViewer uses this to
                filter messages when training multiple models sequentially.
        """
        super().__init__()
        # SLEAP typically creates this object inside a background thread
        # (to keep the GUI responsive during training).  Without explicit
        # thread affinity, Qt's AutoConnection treats signal emissions from
        # that same thread as same-thread → DirectConnection → slot runs on
        # the background thread → NSWindow crash on macOS.
        # moveToThread pins affinity to the main Qt thread so AutoConnection
        # always queues slots there, regardless of where this object is created.
        app = QApplication.instance()
        if app is not None:
            self.moveToThread(app.thread())
        self._sig_inference_msg.connect(self._dispatch_inference_msg)
        self._sig_job_msg.connect(self._dispatch_job_msg)
        self._publish_port = publish_port
        self._controller_port = controller_port
        self._model_type = model_type
        self._socket = None
        self._sub_socket = None
        self._context = None
        self._started = False
        self._lock = threading.Lock()
        self._send_fn: Callable[[str], None] | None = None
        self._poll_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_epoch: int | None = None
        self._inference_dialog = None  # InferenceProgressDialog, created on demand
        self._on_predictions_ready: Callable[[str], None] | None = None
        self._last_n_frames: int | None = (
            None  # parsed from INFERENCE_LOG progress lines
        )

    def set_model_type(self, model_type: str):
        """Update the model type for subsequent messages.

        Call this when switching between training phases in multi-model
        training (e.g., centroid → centered_instance in top-down).

        Args:
            model_type: The new model type string.
        """
        self._model_type = model_type

    def set_send_fn(self, send_fn: Callable[[str], None]):
        """Store a thread-safe send function for the RTC data channel.

        Called by ``run_remote_training`` once the data channel is open.
        The bridge uses this to forward stop/cancel commands to the worker.

        Args:
            send_fn: Thread-safe function that sends a string message over
                the RTC data channel.
        """
        self._send_fn = send_fn

    def _connect_cancel_to_send_fn(self, dialog: "QDialog"):
        """Wire a dialog's Cancel button to send MSG_JOB_CANCEL.

        For training, cancel flows through ZMQ (LossViewer → _poll_commands).
        For standalone inference the bridge is NOT started (no ZMQ), so we
        connect the dialog's ``rejected`` signal directly to ``_send_fn``.
        """
        from sleap_connect.protocol import MSG_JOB_CANCEL

        def _on_cancel():
            if self._send_fn is not None:
                try:
                    self._send_fn(MSG_JOB_CANCEL)
                    logger.info("Sent MSG_JOB_CANCEL to worker")
                except Exception as e:
                    logger.warning(f"Failed to send MSG_JOB_CANCEL: {e}")

        dialog.rejected.connect(_on_cancel)

    def start(self):
        """Start the ZMQ publisher and subscriber sockets.

        - PUB connects to LossViewer's SUB socket (publish_port).
        - SUB connects to LossViewer's controller PUB socket
          (controller_port) and starts a background poll thread.
        """
        import zmq

        with self._lock:
            if self._started:
                return

            self._context = zmq.Context()

            # PUB socket for sending progress to LossViewer
            self._socket = self._context.socket(zmq.PUB)
            self._socket.connect(f"tcp://127.0.0.1:{self._publish_port}")

            # SUB socket for receiving commands from LossViewer
            self._sub_socket = self._context.socket(zmq.SUB)
            self._sub_socket.subscribe(b"")  # Subscribe to all messages
            self._sub_socket.connect(f"tcp://127.0.0.1:{self._controller_port}")

            self._started = True
            self._stop_event.clear()

            # Start background thread to poll for commands
            self._poll_thread = threading.Thread(
                target=self._poll_commands, daemon=True
            )
            self._poll_thread.start()

            logger.debug(
                f"RemoteProgressBridge connected to LossViewer "
                f"(publish={self._publish_port}, "
                f"controller={self._controller_port})"
            )

    def stop(self):
        """Stop the ZMQ sockets and poll thread."""
        with self._lock:
            if not self._started:
                return
            self._started = False

        # Signal the poll thread to stop and wait for it
        self._stop_event.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2.0)
            self._poll_thread = None

        with self._lock:
            if self._sub_socket:
                self._sub_socket.close()
                self._sub_socket = None
            if self._socket:
                self._socket.close()
                self._socket = None
            if self._context:
                self._context.term()
                self._context = None
            self._send_fn = None
            logger.debug("RemoteProgressBridge stopped")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False

    def on_progress(self, event: "ProgressEvent"):
        """Handle a progress event from the training job.

        This method converts the ProgressEvent to ZMQ message format and
        publishes it for LossViewer to consume.

        Args:
            event: The progress event from sleap-connect training.
        """
        if not self._started or not self._socket:
            logger.warning("Progress bridge not started, dropping event")
            return

        # Track explicit epoch_begin events to avoid duplicate synthesis.
        if event.event_type == "epoch_begin" and event.epoch is not None:
            self._last_epoch = event.epoch

        # Synthesize epoch_begin before epoch_end so LossViewer tracks
        # the epoch number (it only reads epoch from epoch_begin messages).
        # Subtract 1 to convert from 1-indexed (sleap-nn text output) to
        # 0-indexed (LossViewer internal convention).
        if (
            event.event_type == "epoch_end"
            and event.epoch is not None
            and event.epoch != self._last_epoch
        ):
            self._last_epoch = event.epoch
            self._publish(
                {
                    "event": "epoch_begin",
                    "what": self._model_type,
                    "epoch": max(0, event.epoch - 1),
                }
            )

        message = self._format_message(event)
        if message:
            self._publish(message)

    def on_raw_zmq_message(self, raw_msg: str):
        """Publish raw jsonpickle ZMQ message directly to LossViewer.

        This bypasses the ProgressEvent conversion and sends the original
        sleap-nn ZMQ message straight through, preserving all event types
        including batch_end (for scatter dots).

        Args:
            raw_msg: Raw jsonpickle-encoded string from the worker's
                ProgressReporter (received via ``PROGRESS_REPORT::``).
        """
        if not self._started or not self._socket:
            return
        try:
            import jsonpickle

            msg = jsonpickle.decode(raw_msg)
            if isinstance(msg, dict) and not msg.get("what"):
                msg["what"] = self._model_type
            self._socket.send_string(jsonpickle.encode(msg))
        except Exception as e:
            logger.error(f"Failed to publish raw ZMQ progress: {e}")

    def _poll_commands(self):
        """Background thread: poll ZMQ SUB for stop/cancel from LossViewer.

        Routes control messages based on command type:
        - ``"stop"`` → forwarded as raw ZMQ via ``CONTROL_COMMAND::`` so
          sleap-nn's ``TrainingControllerZMQ`` receives it identically to
          local training (transparent ZMQ bridge).
        - ``"cancel"`` → sent as ``MSG_JOB_CANCEL`` for process-level
          SIGTERM (no ZMQ equivalent in sleap-nn).
        """
        import zmq

        from sleap_connect.protocol import (
            MSG_CONTROL_COMMAND,
            MSG_JOB_CANCEL,
            MSG_SEPARATOR,
        )

        while not self._stop_event.is_set():
            try:
                with self._lock:
                    sub = self._sub_socket
                if sub is None:
                    break

                # Poll with 100ms timeout so we can check stop_event
                if sub.poll(100, zmq.POLLIN):
                    raw = sub.recv_string(zmq.NOBLOCK)
                    try:
                        import jsonpickle

                        msg = jsonpickle.decode(raw)
                    except Exception:
                        logger.warning(f"Could not decode controller message: {raw!r}")
                        continue

                    command = msg.get("command") if isinstance(msg, dict) else None
                    if command is None:
                        continue

                    send_fn = self._send_fn
                    if send_fn is None:
                        logger.warning(f"Received {command} but no send_fn available")
                        continue

                    try:
                        if command == "cancel":
                            # Cancel = process-level kill (SIGTERM)
                            logger.info(
                                "LossViewer sent cancel — sending MSG_JOB_CANCEL"
                            )
                            send_fn(MSG_JOB_CANCEL)
                        else:
                            # All other commands (including stop) = transparent
                            # ZMQ forwarding to sleap-nn's TrainingControllerZMQ
                            logger.info(
                                f"LossViewer sent command: {command} — "
                                f"forwarding via {MSG_CONTROL_COMMAND}"
                            )
                            send_fn(f"{MSG_CONTROL_COMMAND}{MSG_SEPARATOR}{raw}")
                    except Exception as e:
                        logger.error(f"Failed to forward command {command}: {e}")
            except Exception as e:
                if not self._stop_event.is_set():
                    logger.error(f"Error in command poll: {e}")
                break

    def _format_message(self, event: "ProgressEvent") -> dict | None:
        """Convert ProgressEvent to LossViewer-compatible ZMQ message format.

        The LossViewer expects messages with:
        - ``event``: event type string
        - ``what``: model type for filtering
        - ``logs``: dict with loss values using sleap-nn keys

        Args:
            event: The progress event to format.

        Returns:
            Dictionary matching LossViewer's expected format.
        """
        if event.event_type == "train_begin":
            msg = {"event": "train_begin", "what": self._model_type}
            if event.wandb_url:
                msg["wandb_url"] = event.wandb_url
            return msg

        elif event.event_type == "epoch_begin":
            return {
                "event": "epoch_begin",
                "what": self._model_type,
                "epoch": event.epoch,
            }

        elif event.event_type == "epoch_end":
            logs = {}
            if event.train_loss is not None:
                logs["train/loss"] = event.train_loss
            if event.val_loss is not None:
                logs["val/loss"] = event.val_loss
            if event.metrics:
                logs.update(event.metrics)
            return {
                "event": "epoch_end",
                "what": self._model_type,
                "logs": logs,
            }

        elif event.event_type == "train_end":
            return {
                "event": "train_end",
                "what": self._model_type,
            }

        else:
            logger.warning(f"Unknown event type: {event.event_type}")
            return None

    def _publish(self, message: dict):
        """Publish a message to the ZMQ socket.

        Uses ``jsonpickle.encode`` and ``send_string`` to match the format
        that LossViewer reads with ``jsonpickle.decode(sub.recv_string())``.

        Args:
            message: The message dictionary to publish.
        """
        try:
            import jsonpickle

            self._socket.send_string(jsonpickle.encode(message))
            logger.debug(f"Published progress: {message.get('event')}")
        except Exception as e:
            logger.error(f"Failed to publish progress: {e}")

    # ------------------------------------------------------------------
    # Post-training inference message handling
    # ------------------------------------------------------------------

    def set_predictions_ready_callback(self, callback: "Callable[[str], None] | None"):
        """Register a callback to be invoked when inference completes.

        The callback receives the ``predictions_path`` string from the worker's
        ``INFERENCE_COMPLETE`` message so the caller can load and merge the
        predictions into the open labels project.

        Args:
            callback: Callable taking a single ``predictions_path`` str argument,
                or ``None`` to clear the callback.
        """
        self._on_predictions_ready = callback

    def handle_inference_message(self, msg_type: str, data: dict):
        """Thread-safe entry point for post-training inference messages.

        This method is intended to be passed as the ``on_inference_message``
        callback to :func:`run_remote_training`.  It is called from the
        WebRTC background thread and marshals the message to the main Qt
        thread via a queued signal, where :meth:`_dispatch_inference_msg`
        creates and updates the :class:`InferenceProgressDialog`.

        Args:
            msg_type: One of ``"INFERENCE_BEGIN"``, ``"INFERENCE_PROGRESS"``,
                ``"INFERENCE_COMPLETE"``, ``"INFERENCE_FAILED"``,
                ``"INFERENCE_SKIPPED"``.
            data: Parsed JSON payload dict from the worker message.
        """
        self._sig_inference_msg.emit(msg_type, data)

    def _dispatch_inference_msg(self, msg_type: str, data: dict):
        """Handle an inference message on the main Qt thread.

        Called via the ``_sig_inference_msg`` signal so it always runs on
        the main thread regardless of which thread emitted the signal.
        """
        from sleap_connect.gui.widgets import InferenceProgressDialog

        if msg_type == "INFERENCE_BEGIN":
            logger.info("Inference started — opening InferenceProgressDialog")
            self._last_n_frames = None
            self._inference_dialog = InferenceProgressDialog()
            self._inference_dialog.show()

        elif msg_type == "INFERENCE_LOG":
            text = data.get("text", "")
            if self._inference_dialog is not None:
                self._inference_dialog.append_log(text)
                # Parse progress from rich progress bar lines, e.g.:
                #   "Predicting... 100% 35/35 ETA: 0:00:00 Elapsed: 0:00:01 47.8 FPS"
                m = re.search(r"(\d+)/(\d+)", text)
                fps_m = re.search(r"([\d.]+)\s*FPS", text, re.IGNORECASE)
                if m:
                    n_processed = int(m.group(1))
                    n_total = int(m.group(2))
                    rate = float(fps_m.group(1)) if fps_m else 0.0
                    eta_m = re.search(r"ETA:\s*(\d+):(\d+):(\d+)", text)
                    eta = 0
                    if eta_m:
                        h, mn, s = (
                            int(eta_m.group(1)),
                            int(eta_m.group(2)),
                            int(eta_m.group(3)),
                        )
                        eta = h * 3600 + mn * 60 + s
                    self._inference_dialog.set_progress(n_processed, n_total, rate, eta)
                    self._last_n_frames = n_total

        elif msg_type == "INFERENCE_PROGRESS":
            if self._inference_dialog is not None:
                self._inference_dialog.update(data)

        elif msg_type == "INFERENCE_COMPLETE":
            predictions_path = data.get("predictions_path", "")
            # Use frame count from payload if present, else fall back to value
            # parsed from the rich progress line (sleap-nn track doesn't report
            # per-frame instance stats, so n_with_instances/n_empty stay None).
            n_frames = data.get("n_frames") or self._last_n_frames
            n_with_instances = data.get("n_with_instances")
            n_empty = data.get("n_empty")
            logger.info(f"Inference complete — predictions at {predictions_path}")
            if self._inference_dialog is not None:
                self._inference_dialog.finish(n_frames, n_with_instances, n_empty)
            if self._on_predictions_ready and predictions_path:
                try:
                    self._on_predictions_ready(predictions_path)
                except Exception as e:
                    logger.error(f"on_predictions_ready callback failed: {e}")

        elif msg_type == "INFERENCE_FAILED":
            error = data.get("error", "Unknown inference error")
            logger.error(f"Inference failed: {error}")
            if self._inference_dialog is not None:
                self._inference_dialog.show_error(error)
            else:
                # No dialog open yet (e.g., BEGIN was missed) — show one now
                self._inference_dialog = InferenceProgressDialog()
                self._inference_dialog.show()
                self._inference_dialog.show_error(error)

        elif msg_type == "INFERENCE_SKIPPED":
            reason = data.get("reason", "unknown")
            logger.info(f"Inference skipped: {reason}")
            # No dialog shown for skipped inference.

    def show_eager_dialog(self, worker_name: str | None = None):
        """Create and show the InferenceProgressDialog immediately.

        Call this right after constructing the bridge so the user sees
        'Waiting to start remote inference on <worker> worker...' while
        the WebRTC connection and job submission happen in the background.
        """
        from sleap_connect.gui.widgets import InferenceProgressDialog

        if self._inference_dialog is None:
            self._inference_dialog = InferenceProgressDialog()
            self._connect_cancel_to_send_fn(self._inference_dialog)
            self._inference_dialog.show_waiting(worker_name=worker_name)

    def handle_job_message(self, msg_type: str, data: dict):
        """Thread-safe entry point for standalone-track JOB_* messages.

        This method is intended to be passed as the ``on_job_message``
        callback to :func:`run_inference` (or :func:`run_inference_batch`).
        It is called from the WebRTC background thread and marshals the
        message to the main Qt thread via a queued signal, where
        :meth:`_dispatch_job_msg` creates and updates the
        :class:`InferenceProgressDialog`.

        Mirrors :meth:`handle_inference_message` (which routes the
        post-training inference flow through the same dialog).

        Args:
            msg_type: One of ``"JOB_LOG"``, ``"JOB_PROGRESS"``,
                ``"JOB_COMPLETE"``, ``"JOB_FAILED"``.
            data: Parsed JSON payload dict from the worker message.
        """
        self._sig_job_msg.emit(msg_type, data)

    def _dispatch_job_msg(self, msg_type: str, data: dict):
        """Handle a standalone-track JOB_* message on the main Qt thread.

        Called via the ``_sig_job_msg`` signal so it always runs on the
        main thread regardless of which thread emitted the signal.

        Mirrors :meth:`_dispatch_inference_msg` exactly — same dialog, same
        regex parsing — but for the JOB_* wire prefix used by the
        standalone-track flow rather than the INFERENCE_* prefix used by
        post-training inference.
        """
        from sleap_connect.gui.widgets import InferenceProgressDialog

        if msg_type == "JOB_LOG":
            text = data.get("text", "")
            # The worker forwards tqdm batch updates as ``CR::<line>``;
            # strip that wire prefix before feeding the line to the dialog
            # / regex parser so we don't dirty the displayed log.
            clean = text[4:] if text.startswith("CR::") else text
            if self._inference_dialog is None:
                logger.info("Track job log received — opening InferenceProgressDialog")
                self._last_n_frames = None
                self._inference_dialog = InferenceProgressDialog()
                self._connect_cancel_to_send_fn(self._inference_dialog)
                self._inference_dialog.show()
            else:
                # Transition from "waiting" → live: reset to determinate progress
                if self._inference_dialog._progress_bar.maximum() == 0:
                    self._inference_dialog._status_label.setText("Running inference…")
                    self._inference_dialog._progress_bar.setRange(0, 100)
            # Multi-spec batch: show "video N of M" in the status label
            jobs_total = data.get("jobs_total", 1)
            job_index = data.get("job_index", 0)
            if jobs_total > 1:
                self._inference_dialog._status_label.setText(
                    f"Running inference… (video {job_index + 1} of {jobs_total})"
                )
            self._inference_dialog.append_log(clean)
            # Parse progress from rich progress bar lines, e.g.:
            #   "Predicting... 100% 35/35 ETA: 0:00:00 Elapsed: 0:00:01 47.8 FPS"
            m = re.search(r"(\d+)/(\d+)", clean)
            fps_m = re.search(r"([\d.]+)\s*FPS", clean, re.IGNORECASE)
            if m:
                n_processed = int(m.group(1))
                n_total = int(m.group(2))
                rate = float(fps_m.group(1)) if fps_m else 0.0
                eta_m = re.search(r"ETA:\s*(\d+):(\d+):(\d+)", clean)
                eta = 0
                if eta_m:
                    h, mn, s = (
                        int(eta_m.group(1)),
                        int(eta_m.group(2)),
                        int(eta_m.group(3)),
                    )
                    eta = h * 3600 + mn * 60 + s
                self._inference_dialog.set_progress(n_processed, n_total, rate, eta)
                self._last_n_frames = n_total

        elif msg_type == "JOB_PROGRESS":
            if self._inference_dialog is None:
                self._last_n_frames = None
                self._inference_dialog = InferenceProgressDialog()
                self._connect_cancel_to_send_fn(self._inference_dialog)
                self._inference_dialog.show()
            else:
                # Transition from "waiting" → live: reset to determinate progress
                if self._inference_dialog._progress_bar.maximum() == 0:
                    self._inference_dialog._status_label.setText("Running inference…")
                    self._inference_dialog._progress_bar.setRange(0, 100)
            self._inference_dialog.update(data)

        elif msg_type == "JOB_COMPLETE":
            predictions_path = data.get("output_path", "")
            n_frames = data.get("n_frames") or self._last_n_frames
            n_with_instances = data.get("n_with_instances")
            n_empty = data.get("n_empty")
            jobs_total = data.get("jobs_total", 1)
            job_index = data.get("job_index", 0)
            is_final = job_index >= jobs_total - 1
            logger.info(f"Track job complete — predictions at {predictions_path}")
            if self._inference_dialog is not None:
                if is_final:
                    self._inference_dialog.finish(n_frames, n_with_instances, n_empty)
                else:
                    # Mid-batch: reset for next spec
                    self._inference_dialog._status_label.setText(
                        f"Starting video {job_index + 2} of {jobs_total}…"
                    )
                    self._inference_dialog._progress_bar.setRange(0, 0)  # indeterminate
                    self._inference_dialog._progress_bar.setValue(0)
            if self._on_predictions_ready and predictions_path:
                try:
                    self._on_predictions_ready(predictions_path)
                except Exception as e:
                    logger.error(f"on_predictions_ready callback failed: {e}")

        elif msg_type == "JOB_FAILED":
            error = data.get("message", data.get("error", "Unknown job error"))
            logger.error(f"Track job failed: {error}")
            if self._inference_dialog is None:
                # No dialog open yet (failure before any log line) —
                # show one now so the error is visible.
                self._inference_dialog = InferenceProgressDialog()
                self._inference_dialog.show()
            self._inference_dialog.show_error(error)


def run_remote_training(
    config_path: str | None = None,
    room_id: str = "",
    worker_id: str | None = None,
    publish_port: int = 9001,
    controller_port: int = 9000,
    on_progress: Callable[["ProgressEvent"], None] | None = None,
    timeout: int | None = None,
    config_content: str | None = None,
    path_mappings: dict[str, str] | None = None,
    spec: "TrainJobSpec | None" = None,
    model_type: str = "",
    on_log: Callable[[str], None] | None = None,
    on_model_type: "Callable[[str], None] | None" = None,
    on_inference_message: "Callable[[str, dict], None] | None" = None,
) -> "TrainingResult":
    """Run remote training with progress forwarding to ZMQ.

    This is the main entry point for SLEAP GUI integration. It submits a
    training job to a remote worker and forwards progress events to a local
    ZMQ socket for LossViewer to consume.

    There are two ways to specify the job:
    1. Pass a pre-built ``spec`` (TrainJobSpec) — preferred for GUI integration.
    2. Pass individual parameters (``config_path`` or ``config_content``, etc.).

    Args:
        config_path: Path to the training configuration file.
        room_id: The room ID to connect to.
        worker_id: Optional specific worker ID (None for auto-select).
        publish_port: ZMQ port for LossViewer progress (default: 9001).
        controller_port: ZMQ port for LossViewer controller (default: 9000).
            The bridge listens here for stop/cancel commands.
        on_progress: Optional callback for progress events (in addition to ZMQ).
        timeout: Optional timeout in seconds.
        config_content: Serialized training config (YAML string) sent over
            datachannel. Alternative to config_path for GUI integration.
        path_mappings: Maps original client-side paths to resolved worker paths.
        spec: A pre-built TrainJobSpec. When provided, config_path,
            config_content, and other spec fields are ignored.
        model_type: Model type string for LossViewer filtering (e.g.,
            "centroid", "centered_instance"). LossViewer uses this to track
            the correct training job in multi-model pipelines.
        on_log: Optional callback invoked with each raw log line from the
            worker. When not provided, raw log lines are printed to stdout.
        on_model_type: Optional callback invoked when the worker switches
            model type during multi-model training. When not provided,
            defaults to calling ``bridge.set_model_type()``.
        on_inference_message: Optional callback invoked with ``(msg_type,
            data)`` for post-training inference messages.  When provided the
            data channel stays open after training completes until a terminal
            inference message is received.  Defaults to
            ``bridge.handle_inference_message``.

    Returns:
        TrainingResult with model paths and final status.

    Raises:
        AuthenticationError: If not logged in.
        RoomNotFoundError: If room doesn't exist or no access.
        JobError: If training fails.

    Example:
        # From SLEAP GUI with pre-built spec
        spec = TrainJobSpec(
            config_content=yaml_string,
            labels_path="/mnt/data/labels.slp",
            path_mappings=resolved_mappings,
        )
        result = run_remote_training(
            spec=spec,
            room_id="my-room",
            publish_port=9001,
            model_type="centroid",
        )
        print(f"Model saved to: {result.model_path}")
    """
    from sleap_connect.api import run_training, ProgressEvent

    # Create progress bridge for ZMQ forwarding (bidirectional)
    bridge = RemoteProgressBridge(
        publish_port=publish_port,
        controller_port=controller_port,
        model_type=model_type,
    )

    # Default on_log prints raw log lines to stdout
    log_fn = on_log if on_log is not None else lambda line: print(line, end="")

    def progress_handler(event: ProgressEvent):
        """Handle progress by forwarding to terminal and optional callback.

        LossViewer is now fed by raw ZMQ passthrough (on_raw_progress),
        so we no longer call bridge.on_progress() here.
        """
        # Print structured progress to terminal
        line = format_progress_line(event)
        if line:
            log_fn(line + "\n")
        if on_progress:
            on_progress(event)

    # Always update the bridge model type so on_raw_zmq_message sets the
    # correct 'what' field on ZMQ messages. Also call any user callback so
    # the LossViewer can reset its display for the new model.
    def _combined_model_type_fn(new_model_type: str) -> None:
        bridge.set_model_type(new_model_type)
        if on_model_type is not None:
            on_model_type(new_model_type)

    model_type_fn = _combined_model_type_fn

    # Default inference handler: bridge manages InferenceProgressDialog.
    inference_fn = (
        on_inference_message
        if on_inference_message is not None
        else bridge.handle_inference_message
    )

    # Run training with progress forwarding
    kwargs = dict(
        config_path=config_path,
        room_id=room_id,
        worker_id=worker_id,
        progress_callback=progress_handler,
        config_content=config_content,
        path_mappings=path_mappings,
        spec=spec,
        model_type=model_type,
        on_log=log_fn,
        on_channel_ready=bridge.set_send_fn,
        on_raw_progress=bridge.on_raw_zmq_message,
        on_model_type=model_type_fn,
        on_inference_message=inference_fn,
    )
    if timeout is not None:
        kwargs["timeout"] = timeout
    with bridge:
        result = run_training(**kwargs)

    return result


def format_progress_line(event: "ProgressEvent") -> str:
    """Format a progress event as a CLI-style line.

    This matches the formatting from the cleanup-cli-ux changes for
    consistency between CLI and GUI progress displays.

    Args:
        event: The progress event to format.

    Returns:
        Formatted progress line string.
    """
    if event.event_type == "train_begin":
        lines = ["─" * 60, "Training started"]
        if event.wandb_url:
            lines.append(f"WandB: {event.wandb_url}")
        if event.total_epochs:
            lines.append(f"Total epochs: {event.total_epochs}")
        lines.append("─" * 60)
        return "\n".join(lines)

    elif event.event_type == "epoch_end":
        parts = [f"Epoch {event.epoch}"]
        if event.total_epochs:
            parts[0] = f"Epoch {event.epoch}/{event.total_epochs}"

        metrics = []
        if event.train_loss is not None:
            metrics.append(f"train_loss={event.train_loss:.4f}")
        if event.val_loss is not None:
            metrics.append(f"val_loss={event.val_loss:.4f}")
        if event.metrics:
            for key, value in event.metrics.items():
                if isinstance(value, float):
                    metrics.append(f"{key}={value:.4f}")
                else:
                    metrics.append(f"{key}={value}")

        if metrics:
            parts.append(" | ".join(metrics))

        return " - ".join(parts)

    elif event.event_type == "train_end":
        lines = ["─" * 60]
        if event.success:
            lines.append("Training completed successfully")
        else:
            lines.append("Training failed")
            if event.error_message:
                lines.append(f"Error: {event.error_message}")
        lines.append("─" * 60)
        return "\n".join(lines)

    else:
        return f"Unknown event: {event.event_type}"
