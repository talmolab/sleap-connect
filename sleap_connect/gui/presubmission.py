"""Pre-submission validation flow for remote training.

This module provides the validation sequence that runs before submitting a
training job to a remote worker. It checks:
1. Authentication status (prompts login if needed)
2. Video paths on worker (shows PathResolutionDialog if needed)
3. Config validation (shows ConfigValidationDialog if needed)

Only if all checks pass does the training proceed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from loguru import logger

if TYPE_CHECKING:
    from sleap_connect.api import PathCheckResult, ValidationResult


@dataclass
class PresubmissionResult:
    """Result of the pre-submission validation sequence.

    Attributes:
        success: Whether all checks passed and training can proceed.
        cancelled: Whether the user cancelled during any dialog.
        error: Error message if validation failed.
        path_mappings: Resolved path mappings (original -> worker path).
        validation_result: Config validation result.
    """

    success: bool
    cancelled: bool = False
    error: str | None = None
    path_mappings: dict[str, str] | None = None
    validation_result: "ValidationResult | None" = None


def run_presubmission_checks(
    slp_path: str,
    room_id: str,
    worker_id: str | None = None,
    config_path: str | None = None,
    config_content: str | None = None,
    parent_widget=None,
    on_login_required: Callable[[], bool] | None = None,
    send_fn: "Callable[[str], None] | None" = None,
    convert_fn: "Callable[[str], None] | None" = None,
    save_fn: "Callable[[str], None] | None" = None,
) -> PresubmissionResult:
    """Run the complete pre-submission validation sequence.

    This function orchestrates the validation flow:
    1. Check authentication - if not logged in, call on_login_required callback
    2. Validate config - show ConfigValidationDialog if errors/warnings
       (skipped when config_content is provided, since config was built in-memory)
    3. Check video paths - show PathResolutionDialog if paths need resolution

    Args:
        slp_path: Path to the SLP file (for video path checking).
        room_id: The room ID to use for path checking.
        worker_id: Optional specific worker ID.
        config_path: Path to the training configuration file.
        config_content: Serialized config YAML string (alternative to
            config_path). When provided, config validation is skipped since
            the config was built from the dialog form.
        parent_widget: Parent Qt widget for dialogs.
        on_login_required: Callback called when login is required.
            Should return True if login succeeded, False otherwise.
            If not provided, authentication check will fail if not logged in.
        send_fn: Optional callable to send FS_* messages to a worker.
            When provided, path resolution dialogs will embed a remote
            file browser panel for navigating the worker's filesystem.

    Returns:
        PresubmissionResult indicating whether training can proceed.

    Example:
        result = run_presubmission_checks(
            slp_path="/path/to/labels.slp",
            room_id="my-room",
            config_content=yaml_string,
            parent_widget=self,
            on_login_required=self._handle_login,
        )
        if result.success:
            spec = TrainJobSpec(
                config_content=yaml_string,
                labels_path=slp_path,
                path_mappings=result.path_mappings,
            )
            run_remote_training(spec=spec, room_id=room_id)
    """
    # Step 1: Check authentication
    auth_result = check_authentication(on_login_required)
    if not auth_result.success:
        return auth_result

    # Step 2: Validate config (skipped when config_content is provided,
    # since the config was built from the dialog form and doesn't need
    # file-based validation)
    if config_path and not config_content:
        config_result = check_config_validation(config_path, parent_widget)
        if not config_result.success:
            return config_result
    else:
        config_result = PresubmissionResult(success=True)

    # Step 3: Check video paths
    path_result = check_video_paths(
        slp_path,
        room_id,
        worker_id,
        parent_widget,
        send_fn=send_fn,
        convert_fn=convert_fn,
        save_fn=save_fn,
    )
    if not path_result.success:
        return path_result

    # All checks passed
    return PresubmissionResult(
        success=True,
        path_mappings=path_result.path_mappings,
        validation_result=config_result.validation_result,
    )


def check_authentication(
    on_login_required: Callable[[], bool] | None = None,
) -> PresubmissionResult:
    """Check if user is authenticated, prompt for login if needed.

    Args:
        on_login_required: Callback to handle login. Returns True if login
            succeeded, False if cancelled or failed.

    Returns:
        PresubmissionResult indicating whether authentication check passed.
    """
    from sleap_connect.api import is_logged_in

    if is_logged_in():
        logger.debug("Authentication check passed: already logged in")
        return PresubmissionResult(success=True)

    logger.info("User not logged in, prompting for login")

    if on_login_required is None:
        return PresubmissionResult(
            success=False,
            error="Not logged in. Please log in to continue.",
        )

    # Call the login callback
    login_success = on_login_required()

    if login_success:
        logger.info("Login successful")
        return PresubmissionResult(success=True)
    else:
        logger.info("Login cancelled or failed")
        return PresubmissionResult(
            success=False,
            cancelled=True,
            error="Login required to submit remote training job.",
        )


def check_config_validation(
    config_path: str,
    parent_widget=None,
) -> PresubmissionResult:
    """Validate the training configuration file.

    Shows ConfigValidationDialog if there are errors or warnings.
    - Errors block submission (user must fix config)
    - Warnings allow proceeding (user can click "Continue Anyway")

    Args:
        config_path: Path to the training configuration file.
        parent_widget: Parent Qt widget for the dialog.

    Returns:
        PresubmissionResult indicating whether validation passed.
    """
    from sleap_connect.api import validate_config, ConfigurationError

    try:
        validation_result = validate_config(config_path)
    except ConfigurationError as e:
        return PresubmissionResult(
            success=False,
            error=f"Cannot read config file: {e}",
        )

    # If no errors and no warnings, we're good
    if validation_result.valid and not validation_result.warnings:
        logger.debug("Config validation passed: no errors or warnings")
        return PresubmissionResult(
            success=True,
            validation_result=validation_result,
        )

    # Show dialog for errors or warnings
    logger.info(
        f"Config validation: {len(validation_result.errors)} errors, "
        f"{len(validation_result.warnings)} warnings"
    )

    if parent_widget is not None:
        from sleap_connect.gui.widgets import ConfigValidationDialog

        dialog = ConfigValidationDialog(validation_result, parent=parent_widget)
        result = dialog.exec()

        if validation_result.errors:
            # Errors: dialog only has OK button, always reject
            return PresubmissionResult(
                success=False,
                error="Configuration has validation errors that must be fixed.",
                validation_result=validation_result,
            )
        else:
            # Warnings only: check if user clicked "Continue Anyway"
            if result:  # accepted
                logger.info("User chose to continue despite warnings")
                return PresubmissionResult(
                    success=True,
                    validation_result=validation_result,
                )
            else:  # rejected/cancelled
                logger.info("User cancelled due to warnings")
                return PresubmissionResult(
                    success=False,
                    cancelled=True,
                    validation_result=validation_result,
                )
    else:
        # No parent widget, can't show dialog
        if validation_result.errors:
            error_msgs = [e.message for e in validation_result.errors]
            return PresubmissionResult(
                success=False,
                error=f"Configuration errors: {'; '.join(error_msgs)}",
                validation_result=validation_result,
            )
        else:
            # Warnings only, proceed without dialog
            logger.info("Config has warnings but no dialog to show, proceeding")
            return PresubmissionResult(
                success=True,
                validation_result=validation_result,
            )


def check_video_paths(
    slp_path: str,
    room_id: str,
    worker_id: str | None = None,
    parent_widget=None,
    send_fn: "Callable[[str], None] | None" = None,
    convert_fn: "Callable[[str], None] | None" = None,
    save_fn: "Callable[[str], None] | None" = None,
) -> PresubmissionResult:
    """Check if video paths exist on the worker.

    Sends the SLP path to the worker for validation. If the worker rejects the
    path (e.g., not within mounts or file not found), shows SlpPathDialog to
    let the user provide the correct worker-side path — the retry happens on
    the same WebRTC connection so there's no reconnection delay.

    The API call runs in a background thread so that the asyncio event loop
    (processing data channel messages) stays alive while Qt dialogs are shown
    on the main thread. FS_* responses from the data channel are routed to
    the ``RemoteFileBrowser`` widget via its thread-safe ``on_response()``
    method, enabling interactive filesystem browsing during path resolution.

    Both SLP path resolution and video path resolution happen while the data
    channel is alive, so the ``RemoteFileBrowser`` can send FS_* requests
    and receive responses in real time.

    Args:
        slp_path: Path to the SLP file (may be a local path).
        room_id: The room ID to check paths against.
        worker_id: Optional specific worker ID.
        parent_widget: Parent Qt widget for the dialog.
        send_fn: Unused (kept for API compatibility). The data channel's
            thread-safe send wrapper is provided by the API callbacks.

    Returns:
        PresubmissionResult with path mappings if successful.
    """
    import queue
    import threading

    from sleap_connect.api import (
        check_video_paths as api_check_video_paths,
        AuthenticationError,
        RoomNotFoundError,
        ConfigurationError,
    )

    # Typed dialog request tags
    _SLP_PATH_REQUEST = "slp_path"
    _VIDEOS_MISSING_REQUEST = "videos_missing"

    # Cross-thread dialog bridging queues (shared by both dialog types)
    dialog_request_q: queue.Queue = queue.Queue()
    dialog_response_q: queue.Queue = queue.Queue()

    # Active RemoteFileBrowser for FS_* response routing (set by main thread)
    browser_ref: list = [None]
    # Active SlpPathDialog for FILE_UPLOAD_* response routing
    upload_dialog_ref: list = [None]

    def _on_fs_response(message: str):
        """Route FS_* data channel responses to the active browser widget.

        Called from the asyncio thread. ``RemoteFileBrowser.on_response()``
        is thread-safe (emits a ``QueuedConnection`` Qt signal).
        """
        if browser_ref[0] is not None:
            browser_ref[0].on_response(message)

    def _on_upload_response(message: str):
        """Route FILE_UPLOAD_* responses to the active upload dialog.

        Called from the asyncio thread. ``SlpPathDialog.on_upload_response()``
        is thread-safe (emits a ``QueuedConnection`` Qt signal).
        """
        if upload_dialog_ref[0] is not None:
            upload_dialog_ref[0].on_upload_response(message)

    def _set_browser(b):
        """Switch which RemoteFileBrowser receives FS_* responses."""
        browser_ref[0] = b

    def _on_path_rejected(
        attempted_path: str, error_msg: str, send_fn_dc: Callable
    ) -> str | None:
        """Callback invoked from the background thread when the worker
        rejects the SLP path.

        Posts a request to the main thread to show ``SlpPathDialog``, then
        blocks until the main thread puts the user's response (corrected
        path or None) into ``dialog_response_q``.
        """
        dialog_request_q.put((_SLP_PATH_REQUEST, attempted_path, error_msg, send_fn_dc))
        return dialog_response_q.get()

    def _on_videos_missing(videos: list, send_fn_dc: Callable) -> dict[str, str] | None:
        """Callback invoked from the background thread when videos are missing.

        Posts a request to the main thread to show ``PathResolutionDialog``
        with a live ``send_fn`` for remote file browsing, then blocks until
        the main thread puts the user's response (resolved mappings dict
        or None) into ``dialog_response_q``.
        """
        dialog_request_q.put((_VIDEOS_MISSING_REQUEST, videos, send_fn_dc))
        return dialog_response_q.get()

    # Holders for background thread result / exception
    result_holder: list = [None]
    error_holder: list = [None]

    def _run_api():
        """Background thread: runs asyncio.run() with WebRTC connection."""
        try:
            result_holder[0] = api_check_video_paths(
                slp_path=slp_path,
                room_id=room_id,
                worker_id=worker_id,
                on_path_rejected=_on_path_rejected,
                on_fs_response=_on_fs_response,
                on_upload_response=_on_upload_response,
                on_videos_missing=(
                    _on_videos_missing if parent_widget is not None else None
                ),
            )
        except Exception as e:
            error_holder[0] = e

    thread = threading.Thread(target=_run_api, daemon=True)
    thread.start()

    # Main thread: process Qt events and handle dialog requests while
    # the background thread runs the API call.
    if parent_widget is not None:
        from qtpy.QtWidgets import QApplication

        while thread.is_alive():
            try:
                request = dialog_request_q.get(timeout=0.05)
            except queue.Empty:
                QApplication.processEvents()
                continue

            request_type = request[0]

            if request_type == _SLP_PATH_REQUEST:
                _, attempted_path, error_msg, send_fn_dc = request

                from sleap_connect.gui.widgets import SlpPathDialog

                logger.info(
                    f"SLP path rejected by worker: {error_msg}. "
                    f"Showing path resolution dialog."
                )
                dialog = SlpPathDialog(
                    local_path=attempted_path,
                    error_message=error_msg,
                    send_fn=send_fn_dc,
                    on_browser_changed=_set_browser,
                    convert_fn=convert_fn,
                    save_fn=save_fn,
                    parent=parent_widget,
                )
                upload_dialog_ref[0] = dialog
                if dialog._browser is not None:
                    browser_ref[0] = dialog._browser

                if dialog.exec():
                    corrected = dialog.get_worker_path()
                    logger.info(f"User provided worker SLP path: {corrected}")
                    dialog_response_q.put(corrected)
                else:
                    logger.info("User cancelled SLP path resolution")
                    dialog_response_q.put(None)

                upload_dialog_ref[0] = None
                browser_ref[0] = None

            elif request_type == _VIDEOS_MISSING_REQUEST:
                _, videos, send_fn_dc = request

                from sleap_connect.gui.widgets import PathResolutionDialog

                logger.info("Video path resolution required. Showing dialog.")
                dialog = PathResolutionDialog(
                    videos, send_fn=send_fn_dc, parent=parent_widget
                )
                if dialog._browser is not None:
                    browser_ref[0] = dialog._browser

                if dialog.exec():
                    resolved_paths = dialog.get_resolved_paths()
                    logger.info(f"User resolved {len(resolved_paths)} video paths")
                    # Offer to save directory prefix mappings for each resolved pair
                    from pathlib import Path
                    from sleap_connect.config import get_config
                    from sleap_connect.gui.widgets import show_save_mapping_prompt

                    _cfg = get_config()
                    for orig, worker in resolved_paths.items():
                        local_dir = str(Path(orig).parent)
                        worker_dir = str(Path(worker).parent)
                        if local_dir != worker_dir:
                            show_save_mapping_prompt(
                                local_dir, worker_dir, _cfg, parent_widget
                            )
                    dialog_response_q.put(resolved_paths)
                else:
                    logger.info("User cancelled video path resolution")
                    dialog_response_q.put(None)

                browser_ref[0] = None

    thread.join()

    # Re-raise exceptions from background thread as PresubmissionResult
    if error_holder[0] is not None:
        exc = error_holder[0]
        if isinstance(exc, AuthenticationError):
            return PresubmissionResult(
                success=False,
                error=f"Authentication error: {exc}",
            )
        if isinstance(exc, RoomNotFoundError):
            return PresubmissionResult(
                success=False,
                error=f"Room error: {exc}",
            )
        if isinstance(exc, ConfigurationError):
            error_str = str(exc)
            if any(
                s in error_str.lower()
                for s in ("rejected slp path", "cancelled by user")
            ):
                return PresubmissionResult(
                    success=False,
                    cancelled=True,
                    error=error_str,
                )
            return PresubmissionResult(
                success=False,
                error=error_str,
            )
        logger.warning(f"Video path check failed: {exc}")
        return PresubmissionResult(
            success=True,
            path_mappings={},
        )

    path_result = result_holder[0]

    # Build path mappings: local SLP path -> worker SLP path
    mappings: dict[str, str] = {}
    if path_result.slp_path != slp_path:
        mappings[slp_path] = path_result.slp_path

    # Merge found video paths
    for v in path_result.videos:
        if v.found and v.worker_path:
            mappings[v.original_path] = v.worker_path

    # Merge user-resolved video path mappings (from on_videos_missing)
    if path_result.path_mappings:
        mappings.update(path_result.path_mappings)

    # Fully-embedded pkg.slp: all frames are stored inside the SLP file,
    # no external video files needed — skip PathResolutionDialog entirely.
    if path_result.missing_count == 0 and path_result.embedded_count > 0:
        logger.debug(
            f"All {path_result.embedded_count} video(s) are embedded in the "
            "SLP file — skipping video path resolution"
        )
        return PresubmissionResult(
            success=True,
            path_mappings=mappings,
        )

    if path_result.all_found or path_result.path_mappings:
        logger.debug(f"Video paths resolved: {len(mappings)} mappings")
        return PresubmissionResult(
            success=True,
            path_mappings=mappings,
        )

    # Fallback: missing videos but no callback was available (headless mode)
    missing_paths = [v.filename for v in path_result.videos if not v.found]
    return PresubmissionResult(
        success=False,
        error=f"Missing video paths on worker: {', '.join(missing_paths)}",
    )


class PresubmissionFlow:
    """Class-based interface for pre-submission validation.

    This provides a more flexible interface for GUI integration, allowing
    step-by-step validation with callbacks for each stage.

    Example:
        flow = PresubmissionFlow(
            config_path="/path/to/config.yaml",
            slp_path="/path/to/labels.slp",
            room_id="my-room",
        )
        flow.on_auth_required = self._handle_auth
        flow.on_validation_issues = self._handle_validation
        flow.on_path_resolution = self._handle_paths

        if flow.run(parent=self):
            # All checks passed, proceed with training
            run_remote_training(...)
    """

    def __init__(
        self,
        slp_path: str,
        room_id: str,
        worker_id: str | None = None,
        config_path: str | None = None,
        config_content: str | None = None,
    ):
        """Initialize the pre-submission flow.

        Args:
            slp_path: Path to the SLP file (for video path checking).
            room_id: The room ID to use for path checking.
            worker_id: Optional specific worker ID.
            config_path: Path to the training configuration file.
            config_content: Serialized config YAML string (alternative to
                config_path).
        """
        self.config_path = config_path
        self.config_content = config_content
        self.slp_path = slp_path
        self.room_id = room_id
        self.worker_id = worker_id

        # Callbacks (set by caller)
        self.on_auth_required: Callable[[], bool] | None = None
        self.on_validation_issues: Callable[["ValidationResult"], bool] | None = None
        self.on_path_resolution: Callable[["PathCheckResult"], dict | None] | None = (
            None
        )

        # Results
        self.result: PresubmissionResult | None = None
        self.path_mappings: dict[str, str] = {}
        self.validation_result: "ValidationResult | None" = None

    def run(self, parent=None) -> bool:
        """Run the complete pre-submission flow.

        Args:
            parent: Parent Qt widget for any dialogs.

        Returns:
            True if all checks passed and training can proceed.
        """
        self.result = run_presubmission_checks(
            slp_path=self.slp_path,
            room_id=self.room_id,
            worker_id=self.worker_id,
            config_path=self.config_path,
            config_content=self.config_content,
            parent_widget=parent,
            on_login_required=self.on_auth_required,
        )

        if self.result.success:
            self.path_mappings = self.result.path_mappings or {}
            self.validation_result = self.result.validation_result

        return self.result.success

    def get_error_message(self) -> str | None:
        """Get the error message if validation failed.

        Returns:
            Error message string, or None if no error.
        """
        if self.result:
            return self.result.error
        return None

    def was_cancelled(self) -> bool:
        """Check if the user cancelled during validation.

        Returns:
            True if user cancelled a dialog.
        """
        if self.result:
            return self.result.cancelled
        return False
