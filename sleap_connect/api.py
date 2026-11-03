"""High-level Python API for sleap-connect GUI integration.

This module provides a clean, synchronous API for SLEAP GUI to interact with
sleap-connect functionality. It wraps the lower-level CLI and client functionality
into simple function calls.

Example usage:
    >>> import sleap_connect.api as rtc
    >>> if rtc.is_available() and rtc.is_logged_in():
    ...     rooms = rtc.list_rooms()
    ...     workers = rtc.list_workers(rooms[0].id)
"""

from __future__ import annotations

import atexit
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import requests

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from typing import Callable

__all__ = [
    # Availability
    "is_available",
    # Authentication
    "is_logged_in",
    "get_logged_in_user",
    "login",
    "logout",
    # Discovery
    "list_rooms",
    "list_workers",
    # Path checking
    "check_video_paths",
    "PathCheckResult",
    "VideoPathStatus",
    # Config validation
    "validate_config",
    "ValidationResult",
    "ValidationIssue",
    # Remote execution
    "run_training",
    "run_inference",
    "run_inference_batch",
    "ProgressEvent",
    "TrainingResult",
    "InferenceResult",
    "TrainingJob",
    # Data classes
    "Room",
    "Worker",
    "User",
    # Exceptions
    "AuthenticationError",
    "RoomNotFoundError",
    "ConfigurationError",
    "JobError",
]


# =============================================================================
# Exceptions
# =============================================================================


class AuthenticationError(Exception):
    """Raised when an operation requires authentication but user is not logged in."""

    pass


class RoomNotFoundError(Exception):
    """Raised when a room does not exist or user lacks access."""

    pass


class ConfigurationError(Exception):
    """Raised when a configuration file is invalid or cannot be read."""

    pass


class JobError(Exception):
    """Raised when a remote job fails.

    Attributes:
        job_id: The job ID that failed.
        exit_code: The process exit code (if available).
        message: Error message from the worker.
    """

    def __init__(
        self,
        message: str,
        job_id: str | None = None,
        exit_code: int | None = None,
    ):
        super().__init__(message)
        self.job_id = job_id
        self.exit_code = exit_code


# =============================================================================
# Streamed file reception (Task 2 — Gap 1)
# =============================================================================
#
# The worker streams predictions.slp back to the client over the data
# channel using the bidirectional file-transfer protocol:
#
#     FILE_META::<filename>:<size>:<hint>   (string)
#     <binary chunk 1>                      (bytes)
#     <binary chunk 2>                      (bytes)
#     ...
#     END_OF_FILE                           (string)
#     INFERENCE_COMPLETE::{"predictions_path": "<worker path>"}  (string)
#
# The state machine below captures the FILE_META + chunks + END_OF_FILE
# sub-sequence and writes the bytes to a local temp file, so that the
# GUI-path message handler in _run_training_async can substitute the
# local path for the worker-side path in the INFERENCE_COMPLETE payload.


# -----------------------------------------------------------------------------
# Atexit safety net for temp prediction files (Task 3).
#
# _StreamedFileReceiver writes predictions.slp to a NamedTemporaryFile with
# delete=False so the caller (SLEAP GUI dialog.py) can load, merge, and then
# unlink. If the process exits abnormally between receive and caller-side
# unlink, the tempfile would leak in /tmp. We register each successfully
# received path in a module-level set and install an atexit handler that
# deletes any still-tracked paths on exit. Callers should call
# untrack_temp_prediction() once they have consumed and unlinked the file.
# -----------------------------------------------------------------------------

_temp_prediction_paths: set[str] = set()


def _cleanup_temp_prediction_files() -> None:
    """Atexit handler: delete any tracked temp prediction files.

    Called when the process exits. Silently ignores missing files
    (they were cleaned up by the caller, which is the happy path).
    """
    for p in list(_temp_prediction_paths):
        try:
            if os.path.exists(p):
                os.unlink(p)
        except OSError:
            pass  # best-effort cleanup; nothing we can do at exit time
        _temp_prediction_paths.discard(p)


atexit.register(_cleanup_temp_prediction_files)


def track_temp_prediction(path: str) -> None:
    """Register a temp prediction file for atexit cleanup.

    Used by `_StreamedFileReceiver` on successful predictions.slp
    reception, so that if the caller never consumes the path or
    fails to unlink it, the file is still removed at process exit.
    """
    _temp_prediction_paths.add(path)


def untrack_temp_prediction(path: str) -> None:
    """Deregister a temp prediction file from atexit cleanup.

    Callers should invoke this after they have successfully consumed
    the prediction file (e.g. after merging and unlinking in the
    SLEAP GUI). Removes the path from the atexit tracking set.
    Idempotent — safe to call on a path that was never tracked.
    """
    _temp_prediction_paths.discard(path)


class _StreamedFileReceiver:
    """State machine for receiving FILE_META + bytes + END_OF_FILE.

    Only the ``predictions.slp`` filename is retained for the caller; any
    other filename is drained to a tempfile and then immediately unlinked.

    This class is deliberately minimal (no logging, no retries, no
    concurrent transfers). The worker is expected to stream files one at a
    time over the inference channel.
    """

    def __init__(self) -> None:
        # Info about the currently-in-flight transfer, or None.
        # Keys: fh (open file handle), local_path (str), filename (str),
        #       expected_size (int), bytes_written (int).
        self._pending: dict | None = None
        # Path of the most recently completed predictions.slp transfer,
        # awaiting retrieval via take_predictions_path().
        self._received_predictions_local_path: str | None = None
        # Sticky failure flag for the most recent transfer attempt; consumed
        # by take_transfer_error(). Set on tempfile-open failure, malformed
        # FILE_META, or close() failure during END_OF_FILE. Cleared on a
        # successful FILE_META → tempfile open.
        self._transfer_failed_reason: str | None = None
        # One-shot rate limit for "dropped bytes" warnings. Reset on every
        # FILE_META so a later failed transfer can log once.
        self._warned_on_dropped_bytes: bool = False

    def handle_string(self, message: str) -> bool:
        """Process a string message.

        Returns True if the message was consumed by the file-transfer state
        machine (caller should not forward it further); False otherwise.
        """
        if message.startswith("FILE_META::"):
            # Format: FILE_META::<filename>:<size>:<hint>
            _, meta = message.split("FILE_META::", 1)
            parts = meta.split(":")
            filename = parts[0] if parts else ""

            # If a prior transfer is somehow still open, close it defensively
            # so we don't leak a file descriptor.
            self._abort_pending()
            # New transfer — reset the one-shot dropped-bytes warning.
            self._warned_on_dropped_bytes = False

            # Malformed FILE_META (missing size subfield) is an explicit
            # failure: we can't safely complete the transfer, so don't even
            # open a tempfile. Any bytes that follow drop silently (with a
            # rate-limited warning). END_OF_FILE will find _pending is None
            # and do nothing.
            if len(parts) < 2:
                self._transfer_failed_reason = (
                    f"malformed FILE_META for {filename!r}: missing size field"
                )
                return True
            try:
                expected_size = int(parts[1])
            except ValueError:
                self._transfer_failed_reason = (
                    f"malformed FILE_META for {filename!r}: "
                    f"non-integer size {parts[1]!r}"
                )
                return True

            suffix = os.path.splitext(filename)[1]
            try:
                fh = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            except OSError as exc:
                logger.exception("Failed to open tempfile for streamed transfer")
                self._transfer_failed_reason = (
                    f"failed to open tempfile for {filename}: {exc}"
                )
                return True
            # Successful open — clear any sticky failure from a prior attempt.
            self._transfer_failed_reason = None
            self._pending = {
                "fh": fh,
                "local_path": fh.name,
                "filename": filename,
                "expected_size": expected_size,
                "bytes_written": 0,
            }
            return True

        if message == "END_OF_FILE":
            if self._pending is None:
                # Stray terminator with no active transfer — ignore.
                return True
            pending = self._pending
            self._pending = None
            try:
                pending["fh"].close()
            except OSError as exc:
                logger.warning(f"Error closing streamed transfer: {exc}")
                self._transfer_failed_reason = (
                    f"close failed for {pending['filename']}: {exc}"
                )
                # Attempt to remove the partial file; best-effort.
                try:
                    os.unlink(pending["local_path"])
                except OSError:
                    logger.exception(
                        f"Could not remove partial tempfile {pending['local_path']}"
                    )
                return True

            if pending["filename"].endswith("predictions.slp"):
                # Retain for the caller to consume via take_predictions_path().
                # Worker constructs the output filename as
                # ``<input_data_path>.predictions.slp`` (job_executor builds
                # this), so the basename is e.g.
                # ``resolved_20260427_labels.v003.predictions.slp``, NOT the
                # bare string ``predictions.slp``. Match by suffix (without a
                # leading dot, so the bare ``predictions.slp`` literal also
                # matches) rather than equality.
                self._received_predictions_local_path = pending["local_path"]
                # Register for atexit cleanup so the file doesn't leak in
                # /tmp if the process dies before the caller consumes and
                # unlinks it. The caller (SLEAP GUI) is expected to call
                # untrack_temp_prediction() after a successful merge.
                track_temp_prediction(pending["local_path"])
                logger.info(
                    f"Received predictions stream: {pending['local_path']} "
                    f"({pending['bytes_written']} bytes)"
                )
            else:
                # v1 only expects ``[*.]predictions.slp`` over this channel.
                # Any other filename is a worker misbehavior; don't leave it
                # lingering in /tmp.
                try:
                    os.unlink(pending["local_path"])
                except OSError:
                    pass
            return True

        return False

    def handle_bytes(self, message: bytes) -> None:
        """Process a binary message (a file chunk).

        If no FILE_META has been received (or the transfer failed to open),
        the bytes are dropped. A single WARNING is logged per failed
        transfer to avoid spamming logs with per-chunk messages.
        """
        if self._pending is None:
            if (
                self._transfer_failed_reason is not None
                and not self._warned_on_dropped_bytes
            ):
                logger.warning(
                    f"Dropping streamed bytes: {self._transfer_failed_reason}"
                )
                self._warned_on_dropped_bytes = True
            return
        try:
            self._pending["fh"].write(message)
            self._pending["bytes_written"] += len(message)
        except OSError as exc:
            pending = self._pending
            self._pending = None
            logger.warning(
                f"Error writing streamed transfer chunk for "
                f"{pending['filename']}: {exc}"
            )
            self._transfer_failed_reason = (
                f"write failed for {pending['filename']}: {exc}"
            )
            # Close + unlink the partial tempfile; best-effort.
            try:
                pending["fh"].close()
            except OSError:
                pass
            try:
                os.unlink(pending["local_path"])
            except OSError:
                logger.exception(
                    f"Could not remove partial tempfile {pending['local_path']}"
                )

    def take_predictions_path(self) -> str | None:
        """Return and clear the local path of the most-recently-received
        predictions.slp, or None if no transfer has completed since the
        last call.
        """
        path = self._received_predictions_local_path
        self._received_predictions_local_path = None
        return path

    def take_transfer_error(self) -> str | None:
        """Return and clear the most recent transfer failure reason, or
        None if the last transfer attempt succeeded (or no transfer has
        been attempted). Consumed-once, like take_predictions_path().
        """
        reason = self._transfer_failed_reason
        self._transfer_failed_reason = None
        return reason

    def _abort_pending(self) -> None:
        """Close and unlink any in-flight transfer. Defensive — normally the
        worker streams one file at a time.
        """
        if self._pending is None:
            return
        pending = self._pending
        self._pending = None
        try:
            pending["fh"].close()
        except OSError:
            pass
        try:
            os.unlink(pending["local_path"])
        except OSError:
            pass


def _apply_received_predictions(
    receiver: _StreamedFileReceiver, data: dict, field_name: str
) -> None:
    """Rewrite ``data[field_name]`` in-place to the local tempfile path if
    the receiver has captured a streamed predictions.slp.

    Used to rewrite worker-side path references in protocol messages
    (``INFERENCE_COMPLETE.predictions_path``, ``MSG_JOB_COMPLETE.output_path``)
    to the local tempfile the receiver wrote during streaming. The original
    (worker-side) path is preserved as ``data['worker_<field_name>']`` for
    v2 dual-mode fallback.

    If the local stream failed (tempfile open failed, close failed, or
    malformed FILE_META), we log a warning and leave ``data[field_name]``
    as the worker-side path so downstream code can fall back to it.
    """
    local_path = receiver.take_predictions_path()
    if local_path is None:
        # Surface any latched transfer failure so callers know the local
        # stream was attempted but did not complete.
        error = receiver.take_transfer_error()
        if error is not None:
            logger.warning(
                f"Predictions stream was not received locally: {error}; "
                f"falling back to worker-side path"
            )
        return
    data[f"worker_{field_name}"] = data.get(field_name)
    data[field_name] = local_path


def _dispatch_inference_response(
    response: str,
    on_job_message: "Callable[[str, dict], None] | None",
    on_log: "Callable[[str], None] | None",
) -> bool:
    """Dispatch a single MSG_JOB_PROGRESS / MSG_JOB_LOG wire message.

    Returns True if the response was handled (matched a known prefix),
    False otherwise so the caller's chain can keep checking
    ACCEPTED/REJECTED/COMPLETE/FAILED itself.
    """
    import json

    from sleap_connect.protocol import (
        MSG_JOB_PROGRESS,
        MSG_JOB_LOG,
        MSG_SEPARATOR,
    )

    if response.startswith(MSG_JOB_PROGRESS):
        parts = response.split(MSG_SEPARATOR, 1)
        raw = parts[1] if len(parts) > 1 else ""
        if on_job_message is not None:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {"raw": raw}
            on_job_message("JOB_PROGRESS", data)
        return True

    if response.startswith(MSG_JOB_LOG):
        parts = response.split(MSG_SEPARATOR, 1)
        text = parts[1] if len(parts) > 1 else ""
        if on_log is not None:
            on_log(text)
        if on_job_message is not None:
            on_job_message("JOB_LOG", {"text": text})
        return True

    return False


def _enriched_job_message_wrapper(
    on_job_message: "Callable[[str, dict], None] | None",
    job_index: int,
    jobs_total: int,
) -> "Callable[[str, dict], None]":
    """Wrap on_job_message to inject job_index and jobs_total into payloads."""

    def _wrapper(msg_type: str, data: dict):
        data["job_index"] = job_index
        data["jobs_total"] = jobs_total
        if on_job_message is not None:
            on_job_message(msg_type, data)

    return _wrapper


async def _run_single_spec_async(
    spec: "TrackJobSpec",
    job_id: str,
    data_channel,
    response_queue: "asyncio.Queue",
    file_receiver: "_StreamedFileReceiver",
    timeout: float,
    on_job_message: "Callable[[str, dict], None] | None" = None,
    on_log: "Callable[[str], None] | None" = None,
) -> "InferenceResult":
    """Submit a single TrackJobSpec and process responses until completion.

    This is the job-specific portion extracted from ``_run_inference_batch_async``.
    It sends the spec over *data_channel*, reads responses from
    *response_queue*, and returns an ``InferenceResult`` (success **or**
    failure).  Callers decide whether to raise on failure.

    Raises:
        JobError: If the overall timeout is exceeded.
        ConfigurationError: If the worker rejects the job.
    """
    import asyncio
    import json

    from sleap_connect.protocol import (
        MSG_JOB_SUBMIT,
        MSG_JOB_ACCEPTED,
        MSG_JOB_REJECTED,
        MSG_JOB_COMPLETE,
        MSG_JOB_FAILED,
        MSG_SEPARATOR,
    )

    # Submit job
    spec_json = spec.to_json()
    submit_msg = f"{MSG_JOB_SUBMIT}{MSG_SEPARATOR}{job_id}{MSG_SEPARATOR}{spec_json}"
    data_channel.send(submit_msg)

    # Process responses
    start_time = asyncio.get_event_loop().time()
    server_job_id = None

    while True:
        elapsed = asyncio.get_event_loop().time() - start_time
        remaining = timeout - elapsed
        if remaining <= 0:
            raise JobError("Inference timed out", job_id=job_id)

        try:
            response = await asyncio.wait_for(
                response_queue.get(),
                timeout=min(remaining, 60.0),
            )
        except asyncio.TimeoutError:
            continue

        if _dispatch_inference_response(response, on_job_message, on_log):
            continue

        if response.startswith(MSG_JOB_ACCEPTED):
            parts = response.split(MSG_SEPARATOR)
            server_job_id = parts[1] if len(parts) > 1 else job_id

        elif response.startswith(MSG_JOB_REJECTED):
            parts = response.split(MSG_SEPARATOR, 2)
            error_json = parts[2] if len(parts) > 2 else "{}"
            try:
                error_data = json.loads(error_json)
                errors = error_data.get("errors", [])
                error_msgs = [e.get("message", "Unknown") for e in errors]
                raise ConfigurationError(f"Job rejected: {'; '.join(error_msgs)}")
            except json.JSONDecodeError:
                raise ConfigurationError(f"Job rejected: {error_json}")

        elif response.startswith(MSG_JOB_COMPLETE):
            parts = response.split(MSG_SEPARATOR, 1)
            result_json = parts[1] if len(parts) > 1 else "{}"
            try:
                result_data = json.loads(result_json)
                # If the worker streamed predictions.slp to us
                # (Tasks 6-7), replace the worker-side path with
                # our local temp path. The worker path is
                # preserved as worker_output_path for v2 dual-mode.
                _apply_received_predictions(file_receiver, result_data, "output_path")
                # Notify the GUI dispatcher BEFORE we build the
                # InferenceResult so the progress dialog can finish
                # its "complete" animation before this function
                # returns. (Task 9)
                if on_job_message is not None:
                    on_job_message("JOB_COMPLETE", result_data)
                return InferenceResult(
                    job_id=server_job_id or job_id,
                    success=True,
                    duration_seconds=result_data.get("duration_seconds"),
                    predictions_path=result_data.get("output_path"),
                )
            except json.JSONDecodeError:
                if on_job_message is not None:
                    on_job_message("JOB_COMPLETE", {})
                return InferenceResult(
                    job_id=server_job_id or job_id,
                    success=True,
                )

        elif response.startswith(MSG_JOB_FAILED):
            parts = response.split(MSG_SEPARATOR, 2)
            error_json = parts[2] if len(parts) > 2 else "{}"
            try:
                error_data = json.loads(error_json)
                error_msg = error_data.get("message", "Job failed")
                duration = error_data.get("duration_seconds")
            except json.JSONDecodeError:
                error_data = {}
                error_msg = "Job failed"
                duration = None

            if on_job_message is not None:
                on_job_message(
                    "JOB_FAILED",
                    error_data if error_data else {"message": error_msg},
                )

            return InferenceResult(
                job_id=server_job_id or job_id,
                success=False,
                duration_seconds=duration,
                error_message=error_msg,
            )


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class VideoPathStatus:
    """Status of a single video path on the worker.

    Attributes:
        filename: The video filename (basename).
        original_path: The original path in the SLP file.
        worker_path: The resolved path on the worker (if found).
        found: Whether the video was found on the worker.
        suggestions: List of suggested paths if video was not found.
    """

    filename: str
    original_path: str
    worker_path: str | None = None
    found: bool = False
    suggestions: list[str] | None = None


@dataclass
class PathCheckResult:
    """Result of checking video paths on a worker.

    Attributes:
        all_found: True if all videos were found on the worker.
        total_videos: Total number of videos in the SLP file.
        found_count: Number of videos found on the worker.
        missing_count: Number of videos not found on the worker.
        embedded_count: Number of videos with frames embedded in the SLP file
            (no external video file required).
        videos: List of VideoPathStatus for each video.
        slp_path: The SLP path that was checked.
        path_mappings: User-resolved path mappings from interactive dialogs
            (e.g., ``{original_path: worker_path}``). Populated when the
            ``on_videos_missing`` callback resolves missing video paths.
    """

    all_found: bool
    total_videos: int
    found_count: int
    missing_count: int
    videos: list[VideoPathStatus]
    slp_path: str
    embedded_count: int = 0
    path_mappings: dict[str, str] = field(default_factory=dict)


@dataclass
class ValidationIssue:
    """A validation issue (error or warning) for a configuration.

    Attributes:
        field: Name of the field that has the issue.
        message: Human-readable description of the issue.
        code: Machine-readable error code (e.g., "PATH_NOT_FOUND").
        is_error: True for errors (blocking), False for warnings.
        path: The path that caused the issue (if path-related).
    """

    field: str
    message: str
    code: str | None = None
    is_error: bool = True
    path: str | None = None


@dataclass
class ValidationResult:
    """Result of validating a configuration file.

    Attributes:
        valid: True if no errors (warnings are allowed).
        errors: List of blocking validation errors.
        warnings: List of non-blocking validation warnings.
        config_path: Path to the config file that was validated.
    """

    valid: bool
    errors: list[ValidationIssue]
    warnings: list[ValidationIssue]
    config_path: str


@dataclass
class ProgressEvent:
    """A progress event from a remote training job.

    Attributes:
        event_type: Type of event ("train_begin", "epoch_end", "train_end").
        epoch: Current epoch number (for epoch_end events).
        total_epochs: Total number of epochs (if known).
        train_loss: Training loss for this epoch (if available).
        val_loss: Validation loss for this epoch (if available).
        metrics: Additional metrics dict (learning_rate, etc.).
        wandb_url: WandB run URL (for train_begin events).
        error_message: Error message (for train_end with failure).
        success: Whether training succeeded (for train_end events).
        model_type: Model type string for LossViewer filtering (e.g.,
            "centroid", "centered_instance"). Used by RemoteProgressBridge
            to set the ``what`` field in ZMQ messages.
    """

    event_type: str  # "train_begin", "epoch_end", "train_end"
    epoch: int | None = None
    total_epochs: int | None = None
    train_loss: float | None = None
    val_loss: float | None = None
    metrics: dict | None = None
    wandb_url: str | None = None
    error_message: str | None = None
    success: bool | None = None
    model_type: str | None = None


@dataclass
class TrainingResult:
    """Result of a completed training job.

    Attributes:
        job_id: The job ID assigned by the worker.
        success: Whether training completed successfully.
        duration_seconds: Total training duration in seconds.
        model_path: Path to the trained model directory on the worker.
        checkpoint_path: Path to the best checkpoint (if available).
        final_epoch: The final epoch number reached.
        final_train_loss: Final training loss.
        final_val_loss: Final validation loss.
        wandb_url: WandB run URL (if WandB was enabled).
        error_message: Error message if training failed.
    """

    job_id: str
    success: bool
    duration_seconds: float | None = None
    model_path: str | None = None
    checkpoint_path: str | None = None
    final_epoch: int | None = None
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    wandb_url: str | None = None
    error_message: str | None = None


@dataclass
class InferenceResult:
    """Result of a completed inference job.

    Attributes:
        job_id: The job ID assigned by the worker.
        success: Whether inference completed successfully.
        duration_seconds: Total inference duration in seconds.
        predictions_path: Path to the predictions file on the worker.
        frames_processed: Number of frames processed.
        error_message: Error message if inference failed.
    """

    job_id: str
    success: bool
    duration_seconds: float | None = None
    predictions_path: str | None = None
    frames_processed: int | None = None
    error_message: str | None = None


@dataclass
class User:
    """Represents an authenticated user."""

    id: str
    username: str
    avatar_url: str | None = None


@dataclass
class Room:
    """Represents a sleap-connect room."""

    id: str
    name: str
    role: str  # "owner" or "member"
    created_by: str | None = None
    joined_at: int | None = None  # Unix timestamp
    expires_at: int | None = None  # Unix timestamp


@dataclass
class Worker:
    """Represents a worker in a room."""

    id: str
    name: str
    status: str  # "available", "busy", etc.
    gpu_name: str | None = None
    gpu_memory_mb: int | None = None
    metadata: dict | None = None


# =============================================================================
# Availability
# =============================================================================


def is_available() -> bool:
    """Check if sleap-connect is available and properly configured.

    Returns:
        True if sleap-connect is installed and can connect to the signaling server.
        False if configuration is missing or connection fails.
    """
    try:
        from sleap_connect.config import get_config

        config = get_config()
        # Check that we have signaling server URLs configured
        return bool(config.signaling_websocket and config.signaling_http)
    except Exception:
        return False


# =============================================================================
# Authentication
# =============================================================================


def is_logged_in() -> bool:
    """Check if the user is currently authenticated.

    Returns:
        True if valid JWT credentials exist, False otherwise.
    """
    from sleap_connect.auth.credentials import get_valid_jwt

    jwt = get_valid_jwt()
    return jwt is not None


def get_logged_in_user() -> User | None:
    """Get the currently logged in user.

    Returns:
        User object if logged in, None otherwise.
    """
    from sleap_connect.auth.credentials import get_user, get_valid_jwt

    # Check JWT is valid first
    if get_valid_jwt() is None:
        return None

    user_data = get_user()
    if user_data is None:
        return None

    return User(
        id=user_data.get("id", ""),
        username=user_data.get("username", ""),
        avatar_url=user_data.get("avatar_url"),
    )


def _ensure_keypair_registered(
    auth_token: str,
    device_name: str = "gui",
) -> None:
    """Ensure an Ed25519 keypair exists locally and is registered with the server.

    This is idempotent and safe to call on every login. It will:
    1. Generate a new Ed25519 keypair if none exists locally.
    2. Register the public key with the signaling server if not yet registered.

    Does not raise on failure — registration is retried on the next login.

    Args:
        auth_token: Bearer token (JWT or account key) for the server request.
        device_name: Label for this key on the server (e.g., "gui", "cli").
    """
    from sleap_connect.auth.credentials import (
        get_private_key_b64,
        get_public_key_registered,
        save_private_key_b64,
        set_public_key_registered,
    )
    from sleap_connect.auth.keypair import (
        generate_keypair,
        private_key_from_b64,
        private_key_to_b64,
        public_key_to_b64,
    )
    from sleap_connect.config import get_config

    # Step 1: Ensure local keypair exists
    pub_b64: str | None = None
    priv_b64 = get_private_key_b64()
    if priv_b64 is None:
        private_key, public_key = generate_keypair()
        priv_b64 = private_key_to_b64(private_key)
        pub_b64 = public_key_to_b64(public_key)
        save_private_key_b64(priv_b64)
        logger.info("Generated new Ed25519 keypair for P2P authentication")

    # Step 2: Register public key with server (if not already done)
    if get_public_key_registered():
        return

    # Derive public key from existing private key if we didn't just generate it
    if pub_b64 is None:
        private_key = private_key_from_b64(priv_b64)
        pub_b64 = public_key_to_b64(private_key.public_key())

    try:
        config = get_config()
        resp = requests.post(
            f"{config.get_http_url()}/api/auth/public-keys",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"public_key": pub_b64, "device_name": device_name},
            timeout=10,
        )
        if resp.ok:
            set_public_key_registered(True)
            logger.info("Public key registered with server")
        else:
            logger.warning(
                f"Public key registration failed ({resp.status_code}), "
                "will retry on next login"
            )
    except Exception as e:
        logger.warning(f"Public key registration failed: {e}, will retry on next login")


def login(
    timeout: int = 120,
    on_url_ready: Callable[[str], None] | None = None,
) -> User:
    """Initiate the login flow via browser-based OAuth.

    Opens the default browser to the authentication page and waits for
    the user to complete authentication. After saving the JWT, ensures
    an Ed25519 keypair exists and is registered for P2P authentication.

    Args:
        timeout: Maximum time to wait for authentication in seconds.
        on_url_ready: Optional callback called with the auth URL when ready.
            Useful for GUI to display the URL to the user.

    Returns:
        User object for the authenticated user.

    Raises:
        AuthenticationError: If authentication fails or times out.
    """
    from sleap_connect.auth.github import github_login
    from sleap_connect.auth.credentials import save_jwt

    try:
        result = github_login(timeout=timeout, on_url_ready=on_url_ready, silent=True)
        save_jwt(result["jwt"], result["user"])

        # Ensure Ed25519 keypair exists and is registered for P2P auth.
        # Non-fatal: if registration fails, it will retry on next login.
        try:
            _ensure_keypair_registered(auth_token=result["jwt"])
        except Exception:
            logger.warning("Keypair registration failed, will retry on next login")

        return User(
            id=result["user"].get("id", ""),
            username=result["user"].get("username", ""),
            avatar_url=result["user"].get("avatar_url"),
        )
    except AuthenticationError:
        raise
    except Exception as e:
        raise AuthenticationError(f"Login failed: {e}") from e


def logout() -> None:
    """Clear stored credentials and log out.

    After calling this, is_logged_in() will return False.
    """
    from sleap_connect.auth.credentials import clear_jwt

    clear_jwt()


# =============================================================================
# Discovery
# =============================================================================


def list_rooms(
    role: str | None = None,
    sort_by: str = "name",
    sort_order: str = "asc",
    search: str | None = None,
) -> list[Room]:
    """List available rooms for the authenticated user.

    Args:
        role: Filter by role ("owner", "member", or None for all).
        sort_by: Field to sort by ("name", "joined_at", "expires_at", "role").
        sort_order: Sort order ("asc" or "desc").
        search: Optional search string to filter rooms.

    Returns:
        List of Room objects.

    Raises:
        AuthenticationError: If user is not logged in.
    """
    import requests
    from sleap_connect.auth.credentials import get_valid_jwt
    from sleap_connect.config import get_config

    jwt = get_valid_jwt()
    if jwt is None:
        raise AuthenticationError("Not logged in. Call login() first.")

    config = get_config()
    endpoint = f"{config.get_http_url()}/api/auth/rooms"

    params: dict[str, str] = {
        "sort_by": sort_by,
        "sort_order": sort_order,
    }
    if role:
        params["role"] = role
    if search:
        params["search"] = search

    response = requests.get(
        endpoint,
        headers={"Authorization": f"Bearer {jwt}"},
        params=params,
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    rooms = []
    for room_data in data.get("rooms", []):
        rooms.append(
            Room(
                id=room_data.get("room_id", ""),
                name=room_data.get("name", ""),
                role=room_data.get("role", ""),
                created_by=room_data.get("created_by"),
                joined_at=room_data.get("joined_at"),
                expires_at=room_data.get("expires_at"),
            )
        )
    return rooms


def list_workers(room_id: str) -> list[Worker]:
    """List workers available in a room.

    This connects briefly to the signaling server via WebSocket to discover
    workers in the specified room.

    Args:
        room_id: The room ID to list workers for.

    Returns:
        List of Worker objects.

    Raises:
        AuthenticationError: If user is not logged in.
        RoomNotFoundError: If room does not exist or user lacks access.
    """
    import asyncio
    from sleap_connect.auth.credentials import get_valid_jwt
    from sleap_connect.config import get_config

    jwt = get_valid_jwt()
    if jwt is None:
        raise AuthenticationError("Not logged in. Call login() first.")

    config = get_config()

    # Run async worker discovery in sync context
    return asyncio.run(_discover_workers_async(config, jwt, room_id))


async def _discover_workers_async(config, jwt: str, room_id: str) -> list[Worker]:
    """Async implementation of worker discovery.

    Connects to signaling server, registers, requests peer list, and disconnects.
    """
    import uuid
    import websockets
    import json

    peer_id = f"api-client-{uuid.uuid4().hex[:8]}"
    workers = []

    try:
        async with websockets.connect(
            config.signaling_websocket,
            additional_headers={"Authorization": f"Bearer {jwt}"},
        ) as ws:
            # Register with room
            register_msg = {
                "type": "register",
                "peer_id": peer_id,
                "room_id": room_id,
                "role": "client",
                "jwt": jwt,
                "metadata": {
                    "tags": ["sleap-connect", "api-discovery"],
                    "properties": {"purpose": "worker-discovery"},
                },
            }
            await ws.send(json.dumps(register_msg))

            # Wait for registration response
            while True:
                response = json.loads(await ws.recv())
                if response.get("type") == "registered_auth":
                    break
                if response.get("type") == "error":
                    raise RoomNotFoundError(
                        f"Failed to join room: {response.get('message', 'Unknown error')}"
                    )

            # Request worker list
            discover_msg = {
                "type": "discover_peers",
                "from_peer_id": peer_id,
                "filters": {
                    "role": "worker",
                    "room_id": room_id,
                    "tags": ["sleap-connect"],
                },
            }
            await ws.send(json.dumps(discover_msg))

            # Wait for peer list response
            while True:
                response = json.loads(await ws.recv())
                if response.get("type") == "peer_list":
                    for peer in response.get("peers", []):
                        metadata = peer.get("metadata", {})
                        properties = metadata.get("properties", {})
                        workers.append(
                            Worker(
                                id=peer.get("peer_id", ""),
                                name=properties.get(
                                    "worker_name", peer.get("peer_id", "")
                                ),
                                status=properties.get("status", "unknown"),
                                gpu_name=properties.get("gpu_name"),
                                gpu_memory_mb=properties.get("gpu_memory_mb"),
                                metadata=metadata,
                            )
                        )
                    break
                if response.get("type") == "error":
                    break

    except Exception as e:
        if "401" in str(e) or "403" in str(e):
            raise AuthenticationError(f"Authentication failed: {e}") from e
        raise RoomNotFoundError(f"Failed to discover workers: {e}") from e

    return workers


# =============================================================================
# Path Checking
# =============================================================================


def check_video_paths(
    slp_path: str,
    room_id: str,
    worker_id: str | None = None,
    timeout: float = 30.0,
    on_path_rejected: "Callable[[str, str, Callable], str | None] | None" = None,
    on_fs_response: "Callable[[str], None] | None" = None,
    on_videos_missing: "Callable[[list, Callable], dict[str, str] | None] | None" = None,
    on_upload_response: "Callable[[str], None] | None" = None,
) -> PathCheckResult:
    """Check if video paths in an SLP file are accessible on a worker.

    Connects to a worker in the specified room and checks whether the video
    files referenced in the SLP file exist on the worker's filesystem.

    Args:
        slp_path: Path to the SLP file (as it should exist on the worker).
        room_id: The room ID containing the worker.
        worker_id: Specific worker ID to check against. If None, connects
            to the first available worker.
        timeout: Timeout in seconds for the video check operation.
        on_path_rejected: Optional callback invoked when the worker rejects
            the SLP path (e.g., not within mounts, file not found). Called
            with (attempted_path, error_message, send_fn). ``send_fn`` is
            the data channel's send method, allowing the callback to send
            FS_* messages for remote file browsing. Should return a corrected
            path to retry, or None to abort. The retry reuses the same
            WebRTC connection, avoiding reconnection delays.
        on_fs_response: Optional callback for routing FS_* response messages
            from the data channel. When provided, incoming browser-specific
            messages (``FS_MOUNTS_RESPONSE``, ``FS_LIST_RESPONSE``,
            ``FS_ERROR``) are dispatched to this callback instead of the
            internal response queue. Used to feed
            ``RemoteFileBrowser.on_response()`` for interactive file browsing
            during path resolution dialogs.
        on_videos_missing: Optional callback invoked when the worker reports
            missing video paths. Called with (videos, send_fn) where
            ``videos`` is a list of ``VideoPathStatus`` objects and
            ``send_fn`` is a thread-safe wrapper around the data channel's
            send method. The callback should return a dict mapping
            ``{original_path: resolved_worker_path}`` or None to cancel.
            Called while the data channel is alive so ``send_fn`` can be
            used for remote file browsing.

    Returns:
        PathCheckResult with information about each video path.

    Raises:
        AuthenticationError: If user is not logged in.
        RoomNotFoundError: If room does not exist or has no workers.
        ConfigurationError: If the SLP file cannot be read or has no videos.
    """
    import asyncio

    return asyncio.run(
        _check_video_paths_async(
            slp_path,
            room_id,
            worker_id,
            timeout,
            on_path_rejected,
            on_fs_response,
            on_videos_missing,
            on_upload_response,
        )
    )


async def _authenticate_channel(
    data_channel,
    response_queue: "asyncio.Queue",
    room_secret: str | None = None,
    timeout: float = 15.0,
) -> None:
    """Complete P2P authentication handshake on a data channel.

    After a data channel opens, the worker sends an AUTH_CHALLENGE with a nonce.
    This function handles the full handshake using Ed25519-first auth:
    1. Receives AUTH_CHALLENGE with nonce
    2. Signs nonce with stored Ed25519 private key (or falls back to PSK HMAC)
    3. Waits for AUTH_SUCCESS

    Args:
        data_channel: The open WebRTC data channel.
        response_queue: Queue that receives all messages from the channel.
        room_secret: Optional PSK for HMAC fallback (legacy, backward compat).
        timeout: Max seconds to wait for each auth message.

    Raises:
        ConfigurationError: If authentication fails or times out.
    """
    import asyncio
    from sleap_connect.protocol import (
        MSG_AUTH_CHALLENGE,
        MSG_AUTH_RESPONSE,
        MSG_AUTH_SUCCESS,
        MSG_AUTH_FAILURE,
        MSG_SEPARATOR,
    )

    # Wait for AUTH_CHALLENGE from worker
    try:
        challenge_msg = await asyncio.wait_for(response_queue.get(), timeout=timeout)
    except asyncio.TimeoutError:
        raise ConfigurationError("Authentication timed out waiting for challenge")

    if not challenge_msg.startswith(MSG_AUTH_CHALLENGE):
        raise ConfigurationError(f"Expected AUTH_CHALLENGE, got: {challenge_msg[:50]}")

    # Extract nonce
    parts = challenge_msg.split(MSG_SEPARATOR, 1)
    if len(parts) != 2:
        raise ConfigurationError("Invalid AUTH_CHALLENGE format")
    nonce = parts[1]

    # Build auth response: Ed25519 first, PSK fallback, then "missing"
    auth_payload = None

    # Path A: Ed25519 — sign with stored private key
    from sleap_connect.auth.credentials import get_private_key_b64

    priv_b64 = get_private_key_b64()
    if priv_b64:
        try:
            from sleap_connect.auth.keypair import private_key_from_b64, sign_nonce

            private_key = private_key_from_b64(priv_b64)
            auth_payload = sign_nonce(private_key, nonce)
        except Exception:
            pass  # Fall through to PSK

    # Path B: PSK (room_secret configured) — backward compat
    if auth_payload is None and room_secret:
        from sleap_connect.auth.psk import compute_hmac

        auth_payload = compute_hmac(room_secret, nonce)

    # Path C: No credentials
    if auth_payload is None:
        auth_payload = "missing"

    data_channel.send(f"{MSG_AUTH_RESPONSE}{MSG_SEPARATOR}{auth_payload}")

    # Wait for AUTH_SUCCESS or AUTH_FAILURE
    try:
        auth_result = await asyncio.wait_for(response_queue.get(), timeout=timeout)
    except asyncio.TimeoutError:
        raise ConfigurationError("Authentication timed out waiting for result")

    if auth_result == MSG_AUTH_SUCCESS:
        return
    elif auth_result.startswith(MSG_AUTH_FAILURE):
        parts = auth_result.split(MSG_SEPARATOR, 1)
        reason = parts[1] if len(parts) > 1 else "unknown"
        raise ConfigurationError(f"Authentication failed: {reason}")
    else:
        raise ConfigurationError(f"Unexpected response during auth: {auth_result[:50]}")


async def _check_video_paths_async(
    slp_path: str,
    room_id: str,
    worker_id: str | None,
    timeout: float,
    on_path_rejected: "Callable[[str, str, Callable], str | None] | None" = None,
    on_fs_response: "Callable[[str], None] | None" = None,
    on_videos_missing: "Callable[[list, Callable], dict[str, str] | None] | None" = None,
    on_upload_response: "Callable[[str], None] | None" = None,
) -> PathCheckResult:
    """Async implementation of video path checking."""
    import json
    import uuid
    import websockets
    from aiortc import RTCPeerConnection, RTCSessionDescription
    from sleap_connect.auth.credentials import get_valid_jwt
    from sleap_connect.config import get_config
    from sleap_connect.protocol import (
        MSG_USE_WORKER_PATH,
        MSG_WORKER_PATH_OK,
        MSG_WORKER_PATH_ERROR,
        MSG_FS_CHECK_VIDEOS_RESPONSE,
        MSG_SEPARATOR,
    )

    jwt = get_valid_jwt()
    if jwt is None:
        raise AuthenticationError("Not logged in. Call login() first.")

    config = get_config()
    peer_id = f"api-pathcheck-{uuid.uuid4().hex[:8]}"

    # Response queues
    import asyncio

    response_queue: asyncio.Queue = asyncio.Queue()
    data_channel = None
    pc = None

    try:
        async with websockets.connect(config.signaling_websocket) as ws:
            # Register with room
            register_msg = {
                "type": "register",
                "peer_id": peer_id,
                "room_id": room_id,
                "role": "client",
                "jwt": jwt,
                "metadata": {
                    "tags": ["sleap-connect", "api-pathcheck"],
                    "properties": {"purpose": "video-path-check"},
                },
            }
            await ws.send(json.dumps(register_msg))

            # Wait for registration
            while True:
                response = json.loads(await ws.recv())
                if response.get("type") == "registered_auth":
                    break
                if response.get("type") == "error":
                    raise RoomNotFoundError(
                        f"Failed to join room: {response.get('message', 'Unknown error')}"
                    )

            # Discover workers if worker_id not specified
            if worker_id is None:
                discover_msg = {
                    "type": "discover_peers",
                    "from_peer_id": peer_id,
                    "filters": {
                        "role": "worker",
                        "room_id": room_id,
                        "tags": ["sleap-connect"],
                    },
                }
                await ws.send(json.dumps(discover_msg))

                while True:
                    response = json.loads(await ws.recv())
                    if response.get("type") == "peer_list":
                        peers = response.get("peers", [])
                        if not peers:
                            raise RoomNotFoundError("No workers available in room")
                        worker_id = peers[0].get("peer_id")
                        break

            # Create WebRTC connection
            pc = RTCPeerConnection()
            data_channel = pc.createDataChannel("pathcheck")

            channel_open = asyncio.Event()
            channel_closed = asyncio.Event()

            @data_channel.on("open")
            def on_open():
                channel_open.set()

            @data_channel.on("close")
            def on_close():
                channel_closed.set()

            # FS_* prefixes that belong to the RemoteFileBrowser widget.
            # Other FS_* messages (e.g. FS_CHECK_VIDEOS_RESPONSE) must go to
            # the response queue so the path-check logic can read them.
            _BROWSER_FS_PREFIXES = (
                "FS_MOUNTS_RESPONSE",
                "FS_LIST_RESPONSE",
                "FS_ERROR",
            )

            @data_channel.on("message")
            async def on_message(message):
                if isinstance(message, str):
                    # Route browser-specific FS_* responses to the widget
                    if on_fs_response is not None and any(
                        message.startswith(p) for p in _BROWSER_FS_PREFIXES
                    ):
                        on_fs_response(message)
                    # Route FILE_UPLOAD_* responses to the upload dialog
                    elif on_upload_response is not None and message.startswith(
                        "FILE_UPLOAD_"
                    ):
                        on_upload_response(message)
                    else:
                        await response_queue.put(message)

            # Send offer
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)

            offer_msg = json.dumps(
                {
                    "type": pc.localDescription.type,
                    "sender": peer_id,
                    "target": worker_id,
                    "sdp": pc.localDescription.sdp,
                }
            )
            await ws.send(offer_msg)

            # Wait for answer
            while True:
                response = json.loads(await ws.recv())
                if response.get("type") == "answer":
                    answer = RTCSessionDescription(
                        sdp=response.get("sdp"),
                        type="answer",
                    )
                    await pc.setRemoteDescription(answer)
                    break
                elif response.get("type") == "candidate":
                    candidate = response.get("candidate")
                    if candidate:
                        await pc.addIceCandidate(candidate)

            # Wait for channel to open
            await asyncio.wait_for(channel_open.wait(), timeout=10.0)

            # Authenticate with worker via PSK
            await _authenticate_channel(data_channel, response_queue)

            # Thread-safe send wrapper for callbacks that may run in
            # executor threads (Qt dialogs).  Shared by on_path_rejected
            # and on_videos_missing.
            loop = asyncio.get_running_loop()

            def _thread_safe_send(msg: str) -> None:
                """Schedule data_channel.send on the event loop."""
                loop.call_soon_threadsafe(data_channel.send, msg)

            # Send SLP path to worker, with retry via callback on rejection
            current_path = slp_path
            max_path_retries = 3

            for _attempt in range(max_path_retries):
                data_channel.send(f"{MSG_USE_WORKER_PATH}{MSG_SEPARATOR}{current_path}")

                # Wait for path OK/error response
                path_response = await asyncio.wait_for(
                    response_queue.get(), timeout=timeout
                )

                if path_response.startswith(MSG_WORKER_PATH_ERROR):
                    parts = path_response.split(MSG_SEPARATOR)
                    error_msg = parts[1] if len(parts) > 1 else "Unknown error"

                    # If we have a callback, ask for a corrected path
                    if on_path_rejected is not None:
                        # Run callback in a thread-pool executor so it can
                        # block (e.g. waiting for a Qt dialog) without
                        # freezing the asyncio event loop.  ICE keepalives
                        # and FS_* response routing keep working.
                        corrected = await loop.run_in_executor(
                            None,
                            on_path_rejected,
                            current_path,
                            error_msg,
                            _thread_safe_send,
                        )
                        if corrected is not None:
                            current_path = corrected
                            continue  # Retry with corrected path
                        else:
                            # User cancelled
                            raise ConfigurationError(
                                f"Worker rejected SLP path: {error_msg}"
                            )
                    else:
                        raise ConfigurationError(
                            f"Worker rejected SLP path: {error_msg}"
                        )

                # Path accepted — update slp_path for the result
                slp_path = current_path
                break
            else:
                raise ConfigurationError(
                    "SLP path could not be resolved after multiple attempts."
                )

            # Wait for video check response
            video_response = await asyncio.wait_for(
                response_queue.get(), timeout=timeout
            )
            if not video_response.startswith(MSG_FS_CHECK_VIDEOS_RESPONSE):
                raise ConfigurationError(f"Unexpected response: {video_response[:50]}")

            # Parse video check data
            json_str = video_response.split(MSG_SEPARATOR, 1)[1]
            video_data = json.loads(json_str)

            # Build result
            videos = []
            for v in video_data.get("found", []):
                videos.append(
                    VideoPathStatus(
                        filename=v.get("filename", ""),
                        original_path=v.get("original_path", ""),
                        worker_path=v.get("worker_path"),
                        found=True,
                    )
                )
            for v in video_data.get("missing", []):
                videos.append(
                    VideoPathStatus(
                        filename=v.get("filename", ""),
                        original_path=v.get("original_path", ""),
                        found=False,
                        suggestions=v.get("suggestions"),
                    )
                )

            total = video_data.get("total_videos", len(videos))
            found_count = len(video_data.get("found", []))
            missing_count = len(video_data.get("missing", []))
            embedded_count = video_data.get("embedded", 0)

            # If videos are missing and we have a callback, let the caller
            # resolve them interactively while the data channel is alive.
            resolved_mappings: dict[str, str] = {}
            if missing_count > 0 and on_videos_missing is not None:
                resolved_mappings_or_none = await loop.run_in_executor(
                    None,
                    on_videos_missing,
                    videos,
                    _thread_safe_send,
                )
                if resolved_mappings_or_none is None:
                    raise ConfigurationError("Video path resolution cancelled by user.")
                resolved_mappings = resolved_mappings_or_none

            return PathCheckResult(
                all_found=missing_count == 0,
                total_videos=total,
                found_count=found_count,
                missing_count=missing_count,
                embedded_count=embedded_count,
                videos=videos,
                slp_path=slp_path,
                path_mappings=resolved_mappings,
            )

    finally:
        if pc:
            await pc.close()


# =============================================================================
# Config Validation
# =============================================================================


def validate_config(config_path: str) -> ValidationResult:
    """Validate a training configuration file.

    Checks the configuration YAML file for:
    - Valid YAML syntax
    - Required fields present
    - Valid field values (numeric ranges, etc.)
    - Path fields reference existing files (locally - not on worker)

    Note: This validates the config file structure and local paths only.
    It does NOT validate paths on the remote worker - use check_video_paths()
    for that.

    Args:
        config_path: Path to the training config YAML file.

    Returns:
        ValidationResult with errors and warnings.

    Raises:
        ConfigurationError: If the config file cannot be read.
    """
    from pathlib import Path
    import yaml

    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    config_file = Path(config_path)

    # Check file exists
    if not config_file.exists():
        errors.append(
            ValidationIssue(
                field="config_path",
                message=f"Config file not found: {config_path}",
                code="FILE_NOT_FOUND",
                path=config_path,
            )
        )
        return ValidationResult(
            valid=False, errors=errors, warnings=warnings, config_path=config_path
        )

    # Try to parse YAML
    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        errors.append(
            ValidationIssue(
                field="config_path",
                message=f"Invalid YAML syntax: {e}",
                code="YAML_PARSE_ERROR",
                path=config_path,
            )
        )
        return ValidationResult(
            valid=False, errors=errors, warnings=warnings, config_path=config_path
        )
    except OSError as e:
        raise ConfigurationError(f"Cannot read config file: {e}") from e

    if not isinstance(config, dict):
        errors.append(
            ValidationIssue(
                field="config_path",
                message="Config file must be a YAML mapping (dictionary)",
                code="INVALID_FORMAT",
                path=config_path,
            )
        )
        return ValidationResult(
            valid=False, errors=errors, warnings=warnings, config_path=config_path
        )

    # Validate data_config section
    data_config = config.get("data_config", {})
    if not isinstance(data_config, dict):
        data_config = {}

    # Check train_labels_path
    train_labels = data_config.get("train_labels_path")
    if train_labels is None:
        warnings.append(
            ValidationIssue(
                field="data_config.train_labels_path",
                message="No train_labels_path specified in config",
                code="MISSING_FIELD",
                is_error=False,
            )
        )
    elif isinstance(train_labels, str):
        if not Path(train_labels).exists():
            warnings.append(
                ValidationIssue(
                    field="data_config.train_labels_path",
                    message=f"File not found locally (may exist on worker): {train_labels}",
                    code="PATH_NOT_FOUND_LOCAL",
                    is_error=False,
                    path=train_labels,
                )
            )
    elif isinstance(train_labels, list):
        for i, path in enumerate(train_labels):
            if isinstance(path, str) and not Path(path).exists():
                warnings.append(
                    ValidationIssue(
                        field=f"data_config.train_labels_path[{i}]",
                        message=f"File not found locally (may exist on worker): {path}",
                        code="PATH_NOT_FOUND_LOCAL",
                        is_error=False,
                        path=path,
                    )
                )

    # Check val_labels_path
    val_labels = data_config.get("val_labels_path")
    if val_labels is not None:
        if isinstance(val_labels, str):
            if not Path(val_labels).exists():
                warnings.append(
                    ValidationIssue(
                        field="data_config.val_labels_path",
                        message=f"File not found locally (may exist on worker): {val_labels}",
                        code="PATH_NOT_FOUND_LOCAL",
                        is_error=False,
                        path=val_labels,
                    )
                )
        elif isinstance(val_labels, list):
            for i, path in enumerate(val_labels):
                if isinstance(path, str) and not Path(path).exists():
                    warnings.append(
                        ValidationIssue(
                            field=f"data_config.val_labels_path[{i}]",
                            message=f"File not found locally (may exist on worker): {path}",
                            code="PATH_NOT_FOUND_LOCAL",
                            is_error=False,
                            path=path,
                        )
                    )

    # Validate trainer_config section
    trainer_config = config.get("trainer_config", {})
    if isinstance(trainer_config, dict):
        # Check max_epochs
        max_epochs = trainer_config.get("max_epochs")
        if max_epochs is not None:
            if not isinstance(max_epochs, (int, float)) or max_epochs < 1:
                errors.append(
                    ValidationIssue(
                        field="trainer_config.max_epochs",
                        message=f"max_epochs must be a positive integer, got: {max_epochs}",
                        code="INVALID_VALUE",
                    )
                )
            elif max_epochs > 10000:
                warnings.append(
                    ValidationIssue(
                        field="trainer_config.max_epochs",
                        message=f"max_epochs={max_epochs} is unusually high",
                        code="VALUE_WARNING",
                        is_error=False,
                    )
                )

    # Validate model_config section
    model_config = config.get("model_config", {})
    if isinstance(model_config, dict):
        # Check batch_size
        batch_size = model_config.get("batch_size")
        if batch_size is not None:
            if not isinstance(batch_size, int) or batch_size < 1:
                errors.append(
                    ValidationIssue(
                        field="model_config.batch_size",
                        message=f"batch_size must be a positive integer, got: {batch_size}",
                        code="INVALID_VALUE",
                    )
                )
            elif batch_size > 256:
                warnings.append(
                    ValidationIssue(
                        field="model_config.batch_size",
                        message=f"batch_size={batch_size} may cause memory issues",
                        code="VALUE_WARNING",
                        is_error=False,
                    )
                )

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        config_path=config_path,
    )


# =============================================================================
# Remote Execution
# =============================================================================


class TrainingJob:
    """Handle for a running training job.

    This class allows monitoring and cancelling a training job that was
    started with run_training().

    Attributes:
        job_id: The job ID assigned by the worker.
        room_id: The room ID where the job is running.
        worker_id: The worker ID running the job.
        status: Current status ("pending", "running", "completed", "failed", "cancelled").
    """

    def __init__(
        self,
        job_id: str,
        room_id: str,
        worker_id: str,
        _cancel_func: "Callable[[], None] | None" = None,
    ):
        self.job_id = job_id
        self.room_id = room_id
        self.worker_id = worker_id
        self.status = "running"
        self._cancel_func = _cancel_func
        self._result: TrainingResult | None = None

    def cancel(self) -> None:
        """Request cancellation of the training job.

        Note: Cancellation is best-effort. The job may complete before
        the cancellation request is processed.
        """
        if self._cancel_func:
            self._cancel_func()
        self.status = "cancelled"

    @property
    def result(self) -> TrainingResult | None:
        """Get the training result if the job has completed."""
        return self._result


def run_training(
    config_path: str | None = None,
    room_id: str = "",
    worker_id: str | None = None,
    labels_path: str | None = None,
    val_labels_path: str | None = None,
    max_epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    run_name: str | None = None,
    resume_ckpt_path: str | None = None,
    progress_callback: "Callable[[ProgressEvent], None] | None" = None,
    timeout: float = 86400.0,  # 24 hours default
    config_content: str | None = None,
    path_mappings: dict[str, str] | None = None,
    spec: "TrainJobSpec | None" = None,
    model_type: str = "",
    on_log: "Callable[[str], None] | None" = None,
    on_channel_ready: "Callable[[Callable[[str], None]], None] | None" = None,
    on_raw_progress: "Callable[[str], None] | None" = None,
    on_model_type: "Callable[[str], None] | None" = None,
    on_inference_message: "Callable[[str, dict], None] | None" = None,
) -> TrainingResult:
    """Run training remotely on a worker.

    Submits a training job to a worker in the specified room and waits
    for completion. Progress events are forwarded to the callback if provided.

    There are two ways to provide the training config:
    1. ``config_path``: Path to a config YAML file on the shared filesystem.
    2. ``config_content``: Serialized config YAML string sent over datachannel.
    3. ``spec``: A pre-built TrainJobSpec (overrides all other spec parameters).

    Args:
        config_path: Path to training config YAML file (on worker filesystem).
        room_id: The room ID containing the worker.
        worker_id: Specific worker ID. If None, auto-selects best available.
        labels_path: Override for data_config.train_labels_path.
        val_labels_path: Override for data_config.val_labels_path.
        max_epochs: Override max training epochs.
        batch_size: Override batch size.
        learning_rate: Override learning rate.
        run_name: Name for the training run/checkpoint directory.
        resume_ckpt_path: Path to checkpoint to resume from.
        progress_callback: Function called with ProgressEvent for each update.
        timeout: Maximum time to wait for job completion in seconds.
        config_content: Serialized training config (YAML string) sent over
            datachannel. Alternative to config_path for GUI integration.
        path_mappings: Maps original client-side paths to resolved worker paths.
        spec: A pre-built TrainJobSpec. When provided, config_path,
            config_content, labels_path, and other spec fields are ignored.
        model_type: Model type string for LossViewer filtering (e.g.,
            "centroid", "centered_instance"). Included in all ProgressEvent
            objects emitted to the progress_callback.
        on_log: Optional callback invoked with each raw log line from the
            worker that doesn't match a known protocol prefix. Use this to
            display live training output in a terminal or text widget.
        on_channel_ready: Optional callback invoked with a thread-safe send
            function once the data channel is open. Use this to enable
            bidirectional communication (e.g., sending stop/cancel commands
            to the worker during training).
        on_raw_progress: Optional callback invoked with the raw jsonpickle
            payload string from ``PROGRESS_REPORT::`` messages. Used to
            forward sleap-nn's native ZMQ progress directly to LossViewer.
        on_model_type: Optional callback invoked when the worker sends a
            ``MODEL_TYPE::`` message indicating a switch between models
            in multi-model pipeline training.
        on_inference_message: Optional callback invoked with ``(msg_type,
            data)`` for post-training inference messages sent by the worker
            after training completes.  ``msg_type`` is one of
            ``"INFERENCE_BEGIN"``, ``"INFERENCE_PROGRESS"``,
            ``"INFERENCE_COMPLETE"``, ``"INFERENCE_FAILED"``,
            ``"INFERENCE_SKIPPED"``.  When provided, the data channel stays
            open after the last ``JOB_COMPLETE`` until a terminal inference
            message is received.

    Returns:
        TrainingResult with job outcome and model paths.

    Raises:
        AuthenticationError: If user is not logged in.
        RoomNotFoundError: If room does not exist or has no workers.
        ConfigurationError: If config is invalid.
        JobError: If the training job fails.
    """
    import asyncio

    return asyncio.run(
        _run_training_async(
            config_path=config_path,
            room_id=room_id,
            worker_id=worker_id,
            labels_path=labels_path,
            val_labels_path=val_labels_path,
            max_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            run_name=run_name,
            resume_ckpt_path=resume_ckpt_path,
            progress_callback=progress_callback,
            timeout=timeout,
            config_content=config_content,
            path_mappings=path_mappings,
            spec=spec,
            model_type=model_type,
            on_log=on_log,
            on_channel_ready=on_channel_ready,
            on_raw_progress=on_raw_progress,
            on_model_type=on_model_type,
            on_inference_message=on_inference_message,
        )
    )


async def _run_training_async(
    config_path: str | None,
    room_id: str,
    worker_id: str | None,
    labels_path: str | None,
    val_labels_path: str | None,
    max_epochs: int | None,
    batch_size: int | None,
    learning_rate: float | None,
    run_name: str | None,
    resume_ckpt_path: str | None,
    progress_callback: "Callable[[ProgressEvent], None] | None",
    timeout: float,
    config_content: str | None = None,
    path_mappings: dict[str, str] | None = None,
    spec: "TrainJobSpec | None" = None,
    model_type: str = "",
    on_log: "Callable[[str], None] | None" = None,
    on_channel_ready: "Callable[[Callable[[str], None]], None] | None" = None,
    on_raw_progress: "Callable[[str], None] | None" = None,
    on_model_type: "Callable[[str], None] | None" = None,
    on_inference_message: "Callable[[str, dict], None] | None" = None,
) -> TrainingResult:
    """Async implementation of run_training."""
    import json
    import uuid
    import websockets
    from aiortc import RTCPeerConnection, RTCSessionDescription
    from sleap_connect.auth.credentials import get_valid_jwt
    from sleap_connect.config import get_config
    from sleap_connect.protocol import (
        MSG_JOB_SUBMIT,
        MSG_JOB_ACCEPTED,
        MSG_JOB_REJECTED,
        MSG_JOB_PROGRESS,
        MSG_JOB_COMPLETE,
        MSG_JOB_FAILED,
        MSG_SEPARATOR,
    )
    from sleap_connect.jobs.spec import TrainJobSpec

    jwt = get_valid_jwt()
    if jwt is None:
        raise AuthenticationError("Not logged in. Call login() first.")

    config = get_config()
    peer_id = f"api-train-{uuid.uuid4().hex[:8]}"
    job_id = str(uuid.uuid4())[:8]

    # Build job spec (use pre-built spec if provided)
    if spec is None:
        if config_content is not None:
            spec = TrainJobSpec(
                config_content=config_content,
                labels_path=labels_path,
                val_labels_path=val_labels_path,
                max_epochs=max_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                run_name=run_name,
                resume_ckpt_path=resume_ckpt_path,
                path_mappings=path_mappings or {},
            )
        else:
            spec = TrainJobSpec(
                config_paths=[config_path] if config_path else [],
                labels_path=labels_path,
                val_labels_path=val_labels_path,
                max_epochs=max_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                run_name=run_name,
                resume_ckpt_path=resume_ckpt_path,
                path_mappings=path_mappings or {},
            )

    # Response handling
    import asyncio

    response_queue: asyncio.Queue = asyncio.Queue()
    result: TrainingResult | None = None
    final_epoch = 0
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    # Track completions for multi-model pipelines
    expected_completions = max(1, len(getattr(spec, "config_contents", []) or []))
    completions_received = 0
    pc = None

    try:
        async with websockets.connect(config.signaling_websocket) as ws:
            # Register with room
            register_msg = {
                "type": "register",
                "peer_id": peer_id,
                "room_id": room_id,
                "role": "client",
                "jwt": jwt,
                "metadata": {
                    "tags": ["sleap-connect", "api-training"],
                    "properties": {"purpose": "remote-training"},
                },
            }
            await ws.send(json.dumps(register_msg))

            # Wait for registration
            while True:
                response = json.loads(await ws.recv())
                if response.get("type") == "registered_auth":
                    break
                if response.get("type") == "error":
                    raise RoomNotFoundError(
                        f"Failed to join room: {response.get('message', 'Unknown error')}"
                    )

            # Discover workers if not specified
            if worker_id is None:
                discover_msg = {
                    "type": "discover_peers",
                    "from_peer_id": peer_id,
                    "filters": {
                        "role": "worker",
                        "room_id": room_id,
                        "tags": ["sleap-connect"],
                    },
                }
                await ws.send(json.dumps(discover_msg))

                while True:
                    response = json.loads(await ws.recv())
                    if response.get("type") == "peer_list":
                        peers = response.get("peers", [])
                        if not peers:
                            raise RoomNotFoundError("No workers available in room")
                        # Select first available worker
                        worker_id = peers[0].get("peer_id")
                        break

            # Create WebRTC connection
            pc = RTCPeerConnection()
            data_channel = pc.createDataChannel("training")

            channel_open = asyncio.Event()

            # State machine for incoming FILE_META/bytes/END_OF_FILE
            # transfers (currently only the post-inference predictions.slp
            # stream from the worker).
            file_receiver = _StreamedFileReceiver()

            @data_channel.on("open")
            def on_open():
                channel_open.set()

            @data_channel.on("message")
            async def on_message(message):
                # Filter out the worker's KEEP_ALIVE heartbeat (10-byte
                # binary) BEFORE any other handling. If we let it fall
                # through to file_receiver.handle_bytes(), an in-flight
                # transfer would have those 10 bytes appended to its
                # tempfile, corrupting the predictions.slp by 10 bytes.
                if isinstance(message, bytes) and message == b"KEEP_ALIVE":
                    return

                if isinstance(message, bytes):
                    # Binary message — treat as a file-transfer chunk.
                    # If no FILE_META is active, the receiver silently
                    # drops the bytes (backward-compatible).
                    file_receiver.handle_bytes(message)
                    return

                if isinstance(message, str):
                    # File-transfer control messages (FILE_META, END_OF_FILE)
                    # are consumed by the receiver and not forwarded.
                    if file_receiver.handle_string(message):
                        return

                    if (
                        message.startswith("PROGRESS_REPORT::")
                        and on_raw_progress is not None
                    ):
                        payload = message.split("PROGRESS_REPORT::", 1)[1]
                        on_raw_progress(payload)
                    else:
                        await response_queue.put(message)

            # Send offer
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)

            offer_msg = json.dumps(
                {
                    "type": pc.localDescription.type,
                    "sender": peer_id,
                    "target": worker_id,
                    "sdp": pc.localDescription.sdp,
                }
            )
            await ws.send(offer_msg)

            # Wait for answer
            while True:
                response = json.loads(await ws.recv())
                if response.get("type") == "answer":
                    answer = RTCSessionDescription(
                        sdp=response.get("sdp"),
                        type="answer",
                    )
                    await pc.setRemoteDescription(answer)
                    break
                elif response.get("type") == "candidate":
                    candidate = response.get("candidate")
                    if candidate:
                        await pc.addIceCandidate(candidate)

            # Wait for channel
            await asyncio.wait_for(channel_open.wait(), timeout=30.0)

            # Authenticate with worker via PSK
            await _authenticate_channel(data_channel, response_queue)

            # Expose thread-safe send function for bidirectional communication
            if on_channel_ready:
                loop = asyncio.get_running_loop()

                def _thread_safe_send(msg: str) -> None:
                    loop.call_soon_threadsafe(data_channel.send, msg)

                on_channel_ready(_thread_safe_send)

            # Notify train_begin
            if progress_callback:
                progress_callback(
                    ProgressEvent(
                        event_type="train_begin",
                        model_type=model_type or None,
                    )
                )

            # Submit job
            spec_json = spec.to_json()
            submit_msg = (
                f"{MSG_JOB_SUBMIT}{MSG_SEPARATOR}{job_id}{MSG_SEPARATOR}{spec_json}"
            )
            data_channel.send(submit_msg)

            # Process responses until completion or failure
            start_time = asyncio.get_event_loop().time()
            server_job_id = None

            while True:
                elapsed = asyncio.get_event_loop().time() - start_time
                remaining = timeout - elapsed
                if remaining <= 0:
                    raise JobError("Training timed out", job_id=job_id)

                try:
                    response = await asyncio.wait_for(
                        response_queue.get(),
                        timeout=min(remaining, 60.0),
                    )
                except asyncio.TimeoutError:
                    continue  # Keep waiting

                if response.startswith(MSG_JOB_ACCEPTED):
                    parts = response.split(MSG_SEPARATOR)
                    server_job_id = parts[1] if len(parts) > 1 else job_id

                elif response.startswith(MSG_JOB_REJECTED):
                    parts = response.split(MSG_SEPARATOR, 2)
                    error_json = parts[2] if len(parts) > 2 else "{}"
                    try:
                        error_data = json.loads(error_json)
                        errors = error_data.get("errors", [])
                        error_msgs = [e.get("message", "Unknown") for e in errors]
                        raise ConfigurationError(
                            f"Job rejected: {'; '.join(error_msgs)}"
                        )
                    except json.JSONDecodeError:
                        raise ConfigurationError(f"Job rejected: {error_json}")

                elif response.startswith(MSG_JOB_PROGRESS):
                    parts = response.split(MSG_SEPARATOR, 1)
                    progress_data = parts[1] if len(parts) > 1 else ""

                    # Try to parse as JSON progress
                    try:
                        data = json.loads(progress_data)
                        epoch = data.get("epoch")
                        train_loss = data.get("loss")
                        val_loss = data.get("val_loss")

                        if epoch is not None:
                            final_epoch = epoch
                        if train_loss is not None:
                            final_train_loss = train_loss
                        if val_loss is not None:
                            final_val_loss = val_loss
                        # Don't call progress_callback here — MSG_JOB_PROGRESS
                        # fires once per stdout pattern match (including per-batch
                        # \r lines), so calling it would spam the terminal with
                        # ~100 "Epoch N - train_loss=X" lines per epoch.  The
                        # terminal already receives raw tqdm output via on_log,
                        # and LossViewer gets epoch data via on_raw_progress (ZMQ).
                    except json.JSONDecodeError:
                        # Raw progress output - just forward as-is
                        pass

                elif response.startswith(MSG_JOB_COMPLETE):
                    completions_received += 1
                    parts = response.split(MSG_SEPARATOR, 1)
                    result_json = parts[1] if len(parts) > 1 else "{}"
                    try:
                        result_data = json.loads(result_json)
                        result = TrainingResult(
                            job_id=server_job_id or job_id,
                            success=True,
                            duration_seconds=result_data.get("duration_seconds"),
                            model_path=result_data.get("output_path"),
                            final_epoch=final_epoch,
                            final_train_loss=final_train_loss,
                            final_val_loss=final_val_loss,
                        )
                    except json.JSONDecodeError:
                        result = TrainingResult(
                            job_id=server_job_id or job_id,
                            success=True,
                            final_epoch=final_epoch,
                            final_train_loss=final_train_loss,
                            final_val_loss=final_val_loss,
                        )

                    if completions_received >= expected_completions:
                        if progress_callback:
                            progress_callback(
                                ProgressEvent(
                                    event_type="train_end",
                                    success=True,
                                    model_type=model_type or None,
                                )
                            )
                        if on_inference_message is not None:
                            # Keep loop alive to receive post-training inference
                            # messages from the worker.
                            pass
                        else:
                            break

                elif response.startswith(MSG_JOB_FAILED):
                    parts = response.split(MSG_SEPARATOR, 2)
                    error_json = parts[2] if len(parts) > 2 else "{}"
                    try:
                        error_data = json.loads(error_json)
                        error_msg = error_data.get("message", "Job failed")
                        exit_code = error_data.get("exit_code")
                        duration = error_data.get("duration_seconds")
                    except json.JSONDecodeError:
                        error_msg = "Job failed"
                        exit_code = None
                        duration = None

                    if progress_callback:
                        progress_callback(
                            ProgressEvent(
                                event_type="train_end",
                                success=False,
                                error_message=error_msg,
                                model_type=model_type or None,
                            )
                        )

                    result = TrainingResult(
                        job_id=server_job_id or job_id,
                        success=False,
                        duration_seconds=duration,
                        final_epoch=final_epoch,
                        final_train_loss=final_train_loss,
                        final_val_loss=final_val_loss,
                        error_message=error_msg,
                    )
                    if on_inference_message is not None:
                        # Keep loop alive — worker will send INFERENCE_SKIPPED
                        # even after a training failure.
                        pass
                    else:
                        break

                elif response.startswith("MODEL_TYPE::"):
                    new_model_type = response.split("::", 1)[1]
                    model_type = new_model_type
                    if on_model_type:
                        on_model_type(new_model_type)

                elif response.startswith("INFERENCE_BEGIN::"):
                    if on_inference_message is not None:
                        try:
                            data = json.loads(response.split("::", 1)[1] or "{}")
                        except json.JSONDecodeError:
                            data = {}
                        on_inference_message("INFERENCE_BEGIN", data)

                elif response.startswith("INFERENCE_LOG::"):
                    if on_inference_message is not None:
                        text = response.split("::", 1)[1]
                        on_inference_message("INFERENCE_LOG", {"text": text})

                elif response.startswith("INFERENCE_PROGRESS::"):
                    if on_inference_message is not None:
                        try:
                            data = json.loads(response.split("::", 1)[1])
                        except json.JSONDecodeError:
                            data = {}
                        on_inference_message("INFERENCE_PROGRESS", data)

                elif response.startswith("INFERENCE_COMPLETE::"):
                    if on_inference_message is not None:
                        try:
                            data = json.loads(response.split("::", 1)[1])
                        except json.JSONDecodeError:
                            data = {}
                        # If the worker streamed predictions.slp to us
                        # (Task 1), replace the worker-side path with our
                        # local temp path. The worker path is preserved
                        # as worker_predictions_path for v2 dual-mode.
                        _apply_received_predictions(
                            file_receiver, data, "predictions_path"
                        )
                        on_inference_message("INFERENCE_COMPLETE", data)
                    break  # Terminal inference message — close channel

                elif response.startswith("INFERENCE_FAILED::"):
                    if on_inference_message is not None:
                        try:
                            data = json.loads(response.split("::", 1)[1])
                        except json.JSONDecodeError:
                            data = {}
                        on_inference_message("INFERENCE_FAILED", data)
                    break  # Terminal inference message — close channel

                elif response.startswith("INFERENCE_SKIPPED::"):
                    if on_inference_message is not None:
                        try:
                            data = json.loads(response.split("::", 1)[1])
                        except json.JSONDecodeError:
                            data = {}
                        on_inference_message("INFERENCE_SKIPPED", data)
                    break  # Terminal inference message — close channel

                else:
                    if response.startswith("CR::"):
                        # \r-terminated progress update — overwrite the current
                        # terminal line to emulate tqdm's animated progress bar.
                        if on_log:
                            on_log("\r" + response[4:])
                    else:
                        # Raw training log line from worker
                        if on_log:
                            on_log(response)

    finally:
        if pc:
            await pc.close()

    if result is None:
        raise JobError("Training ended unexpectedly", job_id=job_id)

    if not result.success:
        raise JobError(
            result.error_message or "Training failed",
            job_id=result.job_id,
        )

    return result


def run_inference(
    data_path: str,
    model_paths: list[str],
    room_id: str,
    worker_id: str | None = None,
    output_path: str | None = None,
    batch_size: int | None = None,
    peak_threshold: float | None = None,
    only_suggested_frames: bool = False,
    frames: str | None = None,
    frame_filter: str | None = None,
    video_index: int | None = None,
    exclude_user_labeled: bool = False,
    path_mappings: dict[str, str] | None = None,
    progress_callback: "Callable[[ProgressEvent], None] | None" = None,
    timeout: float = 3600.0,  # 1 hour default
    on_channel_ready: "Callable[[Callable[[str], None]], None] | None" = None,
    on_log: "Callable[[str], None] | None" = None,
    on_job_message: "Callable[[str, dict], None] | None" = None,
) -> InferenceResult:
    """Run inference remotely on a worker.

    Submits an inference job to a worker in the specified room and waits
    for completion.

    Args:
        data_path: Path to data file (SLP or video) on worker filesystem.
        model_paths: List of model directory paths on worker filesystem.
        room_id: The room ID containing the worker.
        worker_id: Specific worker ID. If None, auto-selects best available.
        output_path: Output predictions file path on worker.
        batch_size: Inference batch size.
        peak_threshold: Peak detection threshold.
        only_suggested_frames: Only run on suggested frames.
        frames: Frame range string (e.g., "0-100,200-300").
        frame_filter: Which subset of frames to run inference on. One of
            "suggested", "user", "predicted", "random", or None (all frames).
        video_index: Restrict inference to a single video by index. None means
            all videos in the labels file.
        path_mappings: Maps original client-side paths to resolved worker paths.
            The worker uses these to rewrite video paths inside the SLP before
            running sleap-nn. Mirrors the same parameter on run_training.
        progress_callback: Function called with progress updates.
        timeout: Maximum time to wait for job completion in seconds.
        on_channel_ready: Optional callback invoked once with a thread-safe
            ``send_fn(str)`` after the data channel is authenticated. The
            GUI's Cancel button uses this to send ``MSG_JOB_CANCEL`` from
            the Qt thread without touching the asyncio loop directly.
        on_log: Optional callback invoked with raw log lines from the
            worker (Task 9 wiring; accepted in Task 8 for forward
            compatibility).
        on_job_message: Optional callback invoked with ``(msg_type, payload)``
            for typed ``JOB_*`` dispatches (Task 9 wiring; accepted in
            Task 8 for forward compatibility).

    Returns:
        InferenceResult with job outcome and predictions path.

    Raises:
        AuthenticationError: If user is not logged in.
        RoomNotFoundError: If room does not exist or has no workers.
        ConfigurationError: If job spec is invalid.
        JobError: If the inference job fails.
    """
    from sleap_connect.jobs.spec import TrackJobSpec

    spec = TrackJobSpec(
        data_path=data_path,
        model_paths=model_paths,
        output_path=output_path,
        batch_size=batch_size,
        peak_threshold=peak_threshold,
        only_suggested_frames=only_suggested_frames,
        exclude_user_labeled=exclude_user_labeled,
        frames=frames,
        frame_filter=frame_filter,
        video_index=video_index,
        path_mappings=path_mappings or {},
    )

    results = run_inference_batch(
        specs=[spec],
        room_id=room_id,
        worker_id=worker_id,
        on_channel_ready=on_channel_ready,
        on_job_message=on_job_message,
        on_log=on_log,
        timeout=timeout,
    )
    return results[0]


def run_inference_batch(
    specs: list["TrackJobSpec"],
    room_id: str,
    worker_id: str | None = None,
    on_channel_ready: "Callable[[Callable[[str], None]], None] | None" = None,
    on_job_message: "Callable[[str, dict], None] | None" = None,
    on_log: "Callable[[str], None] | None" = None,
    on_spec_complete: "Callable[[InferenceResult], None] | None" = None,
    timeout: float = 3600.0,
) -> list[InferenceResult]:
    """Run multiple inference specs sequentially over a single WebRTC connection.

    Opens one WebRTC connection to the worker and submits each spec in order.
    If a spec fails the batch stops early (fail-fast).

    Args:
        specs: List of TrackJobSpec instances to run sequentially.
        room_id: The room ID containing the worker.
        worker_id: Specific worker ID. If None, auto-selects best available.
        on_channel_ready: Optional callback invoked once with a thread-safe
            ``send_fn(str)`` after the data channel is authenticated.
        on_job_message: Optional callback invoked with ``(msg_type, payload)``
            for typed ``JOB_*`` dispatches. Payloads are enriched with
            ``job_index`` and ``jobs_total`` fields.
        on_log: Optional callback invoked with raw log lines from the worker.
        on_spec_complete: Optional callback invoked after each spec completes
            (success or failure) with the corresponding ``InferenceResult``.
        timeout: Maximum time to wait for each individual spec in seconds.

    Returns:
        List of InferenceResult, one per spec that was attempted. On fail-fast
        the list will be shorter than *specs*.

    Raises:
        AuthenticationError: If user is not logged in.
        RoomNotFoundError: If room does not exist or has no workers.
    """
    import asyncio

    return asyncio.run(
        _run_inference_batch_async(
            specs=specs,
            room_id=room_id,
            worker_id=worker_id,
            on_channel_ready=on_channel_ready,
            on_job_message=on_job_message,
            on_log=on_log,
            on_spec_complete=on_spec_complete,
            timeout=timeout,
        )
    )


async def _run_inference_batch_async(
    specs: list["TrackJobSpec"],
    room_id: str,
    worker_id: str | None = None,
    on_channel_ready: "Callable[[Callable[[str], None]], None] | None" = None,
    on_job_message: "Callable[[str, dict], None] | None" = None,
    on_log: "Callable[[str], None] | None" = None,
    on_spec_complete: "Callable[[InferenceResult], None] | None" = None,
    timeout: float = 3600.0,
) -> list["InferenceResult"]:
    """Async implementation of run_inference_batch."""
    import asyncio
    import json
    import uuid

    import websockets
    from aiortc import RTCPeerConnection, RTCSessionDescription

    from sleap_connect.auth.credentials import get_valid_jwt
    from sleap_connect.config import get_config

    jwt = get_valid_jwt()
    if jwt is None:
        raise AuthenticationError("Not logged in. Call login() first.")

    config = get_config()
    batch_id = str(uuid.uuid4())[:8]
    peer_id = f"api-infer-{uuid.uuid4().hex[:8]}"

    # Response handling
    response_queue: asyncio.Queue = asyncio.Queue()
    pc = None

    try:
        async with websockets.connect(config.signaling_websocket) as ws:
            # Register with room
            register_msg = {
                "type": "register",
                "peer_id": peer_id,
                "room_id": room_id,
                "role": "client",
                "jwt": jwt,
                "metadata": {
                    "tags": ["sleap-connect", "api-inference"],
                    "properties": {"purpose": "remote-inference-batch"},
                },
            }
            await ws.send(json.dumps(register_msg))

            # Wait for registration
            while True:
                response = json.loads(await ws.recv())
                if response.get("type") == "registered_auth":
                    break
                if response.get("type") == "error":
                    raise RoomNotFoundError(
                        f"Failed to join room: {response.get('message', 'Unknown error')}"
                    )

            # Discover workers if not specified
            if worker_id is None:
                discover_msg = {
                    "type": "discover_peers",
                    "from_peer_id": peer_id,
                    "filters": {
                        "role": "worker",
                        "room_id": room_id,
                        "tags": ["sleap-connect"],
                    },
                }
                await ws.send(json.dumps(discover_msg))

                while True:
                    response = json.loads(await ws.recv())
                    if response.get("type") == "peer_list":
                        peers = response.get("peers", [])
                        if not peers:
                            raise RoomNotFoundError("No workers available in room")
                        worker_id = peers[0].get("peer_id")
                        break

            # Create WebRTC connection
            pc = RTCPeerConnection()
            data_channel = pc.createDataChannel("inference")

            channel_open = asyncio.Event()

            # State machine for incoming FILE_META/bytes/END_OF_FILE
            # transfers (the post-track predictions.slp stream from the
            # worker).
            file_receiver = _StreamedFileReceiver()

            @data_channel.on("open")
            def on_open():
                channel_open.set()

            @data_channel.on("message")
            async def on_message(message):
                if isinstance(message, bytes) and message == b"KEEP_ALIVE":
                    return

                if isinstance(message, bytes):
                    file_receiver.handle_bytes(message)
                    return

                if isinstance(message, str):
                    if file_receiver.handle_string(message):
                        return

                    await response_queue.put(message)

            # Send offer
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)

            offer_msg = json.dumps(
                {
                    "type": pc.localDescription.type,
                    "sender": peer_id,
                    "target": worker_id,
                    "sdp": pc.localDescription.sdp,
                }
            )
            await ws.send(offer_msg)

            # Wait for answer
            while True:
                response = json.loads(await ws.recv())
                if response.get("type") == "answer":
                    answer = RTCSessionDescription(
                        sdp=response.get("sdp"),
                        type="answer",
                    )
                    await pc.setRemoteDescription(answer)
                    break
                elif response.get("type") == "candidate":
                    candidate = response.get("candidate")
                    if candidate:
                        await pc.addIceCandidate(candidate)

            # Wait for channel
            await asyncio.wait_for(channel_open.wait(), timeout=30.0)

            # Authenticate with worker via PSK
            await _authenticate_channel(data_channel, response_queue)

            # Expose thread-safe send function for bidirectional communication.
            if on_channel_ready:
                loop = asyncio.get_running_loop()

                def _thread_safe_send(msg: str) -> None:
                    loop.call_soon_threadsafe(data_channel.send, msg)

                on_channel_ready(_thread_safe_send)

            # Run specs sequentially over the single connection
            results: list[InferenceResult] = []
            for i, spec in enumerate(specs):
                job_id = f"{batch_id}-{i}"
                enriched = _enriched_job_message_wrapper(on_job_message, i, len(specs))
                try:
                    result = await _run_single_spec_async(
                        spec=spec,
                        job_id=job_id,
                        data_channel=data_channel,
                        response_queue=response_queue,
                        file_receiver=file_receiver,
                        timeout=timeout,
                        on_job_message=enriched,
                        on_log=on_log,
                    )
                except (ConfigurationError, JobError) as e:
                    result = InferenceResult(
                        job_id=job_id,
                        success=False,
                        error_message=str(e),
                    )
                results.append(result)
                if on_spec_complete is not None:
                    on_spec_complete(result)
                if not result.success:
                    break

            return results

    finally:
        if pc:
            await pc.close()
