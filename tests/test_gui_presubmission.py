"""Tests for sleap_connect.gui.presubmission module."""

import pytest
from unittest.mock import MagicMock, patch

from sleap_connect.api import (
    ValidationResult,
    ValidationIssue,
    PathCheckResult,
    VideoPathStatus,
)
from sleap_connect.gui.presubmission import (
    PresubmissionResult,
    PresubmissionFlow,
    run_presubmission_checks,
    check_authentication,
    check_config_validation,
    check_video_paths,
)


# =============================================================================
# check_authentication Tests
# =============================================================================


class TestCheckAuthentication:
    """Tests for check_authentication function."""

    @patch("sleap_connect.api.is_logged_in")
    def test_already_logged_in(self, mock_is_logged_in):
        """Should succeed if already logged in."""
        mock_is_logged_in.return_value = True

        result = check_authentication()

        assert result.success is True
        assert result.cancelled is False
        assert result.error is None

    @patch("sleap_connect.api.is_logged_in")
    def test_not_logged_in_no_callback(self, mock_is_logged_in):
        """Should fail if not logged in and no callback provided."""
        mock_is_logged_in.return_value = False

        result = check_authentication()

        assert result.success is False
        assert result.cancelled is False
        assert "Not logged in" in result.error

    @patch("sleap_connect.api.is_logged_in")
    def test_not_logged_in_login_succeeds(self, mock_is_logged_in):
        """Should succeed if login callback returns True."""
        mock_is_logged_in.return_value = False
        callback = MagicMock(return_value=True)

        result = check_authentication(on_login_required=callback)

        assert result.success is True
        assert result.cancelled is False
        callback.assert_called_once()

    @patch("sleap_connect.api.is_logged_in")
    def test_not_logged_in_login_cancelled(self, mock_is_logged_in):
        """Should fail with cancelled=True if login callback returns False."""
        mock_is_logged_in.return_value = False
        callback = MagicMock(return_value=False)

        result = check_authentication(on_login_required=callback)

        assert result.success is False
        assert result.cancelled is True
        callback.assert_called_once()


# =============================================================================
# check_config_validation Tests
# =============================================================================


class TestCheckConfigValidation:
    """Tests for check_config_validation function."""

    @patch("sleap_connect.api.validate_config")
    def test_valid_config_no_warnings(self, mock_validate):
        """Should succeed if config is valid with no warnings."""
        mock_validate.return_value = ValidationResult(
            valid=True,
            errors=[],
            warnings=[],
            config_path="/path/to/config.yaml",
        )

        result = check_config_validation("/path/to/config.yaml")

        assert result.success is True
        assert result.validation_result is not None
        assert result.validation_result.valid is True

    @patch("sleap_connect.api.validate_config")
    def test_config_with_errors_no_parent(self, mock_validate):
        """Should fail with errors when no parent widget provided."""
        mock_validate.return_value = ValidationResult(
            valid=False,
            errors=[
                ValidationIssue(
                    field="max_epochs",
                    message="Invalid value",
                    code="INVALID_VALUE",
                )
            ],
            warnings=[],
            config_path="/path/to/config.yaml",
        )

        result = check_config_validation("/path/to/config.yaml", parent_widget=None)

        assert result.success is False
        assert "Invalid value" in result.error
        assert result.validation_result is not None

    @patch("sleap_connect.api.validate_config")
    def test_config_with_warnings_only_no_parent(self, mock_validate):
        """Should succeed with warnings when no parent widget provided."""
        mock_validate.return_value = ValidationResult(
            valid=True,
            errors=[],
            warnings=[
                ValidationIssue(
                    field="batch_size",
                    message="Value is high",
                    code="VALUE_WARNING",
                    is_error=False,
                )
            ],
            config_path="/path/to/config.yaml",
        )

        result = check_config_validation("/path/to/config.yaml", parent_widget=None)

        assert result.success is True
        assert result.validation_result is not None

    @patch("sleap_connect.api.validate_config")
    def test_config_read_error(self, mock_validate):
        """Should fail with error if config cannot be read."""
        from sleap_connect.api import ConfigurationError

        mock_validate.side_effect = ConfigurationError("File not found")

        result = check_config_validation("/nonexistent.yaml")

        assert result.success is False
        assert "Cannot read config file" in result.error

    @patch("sleap_connect.gui.widgets.ConfigValidationDialog")
    @patch("sleap_connect.api.validate_config")
    def test_config_with_errors_dialog_shown(self, mock_validate, mock_dialog_class):
        """Should show dialog and fail when config has errors."""
        mock_validate.return_value = ValidationResult(
            valid=False,
            errors=[
                ValidationIssue(
                    field="max_epochs",
                    message="Invalid value",
                    code="INVALID_VALUE",
                )
            ],
            warnings=[],
            config_path="/path/to/config.yaml",
        )

        mock_dialog = MagicMock()
        mock_dialog.exec.return_value = False  # Dialog rejected (OK button)
        mock_dialog_class.return_value = mock_dialog

        parent = MagicMock()
        result = check_config_validation("/path/to/config.yaml", parent_widget=parent)

        assert result.success is False
        mock_dialog_class.assert_called_once()
        mock_dialog.exec.assert_called_once()

    @patch("sleap_connect.gui.widgets.ConfigValidationDialog")
    @patch("sleap_connect.api.validate_config")
    def test_config_with_warnings_user_continues(self, mock_validate, mock_dialog_class):
        """Should succeed when user clicks Continue Anyway for warnings."""
        mock_validate.return_value = ValidationResult(
            valid=True,
            errors=[],
            warnings=[
                ValidationIssue(
                    field="batch_size",
                    message="Value is high",
                    code="VALUE_WARNING",
                    is_error=False,
                )
            ],
            config_path="/path/to/config.yaml",
        )

        mock_dialog = MagicMock()
        mock_dialog.exec.return_value = True  # User clicked "Continue Anyway"
        mock_dialog_class.return_value = mock_dialog

        parent = MagicMock()
        result = check_config_validation("/path/to/config.yaml", parent_widget=parent)

        assert result.success is True
        mock_dialog_class.assert_called_once()

    @patch("sleap_connect.gui.widgets.ConfigValidationDialog")
    @patch("sleap_connect.api.validate_config")
    def test_config_with_warnings_user_cancels(self, mock_validate, mock_dialog_class):
        """Should fail with cancelled=True when user cancels on warnings."""
        mock_validate.return_value = ValidationResult(
            valid=True,
            errors=[],
            warnings=[
                ValidationIssue(
                    field="batch_size",
                    message="Value is high",
                    code="VALUE_WARNING",
                    is_error=False,
                )
            ],
            config_path="/path/to/config.yaml",
        )

        mock_dialog = MagicMock()
        mock_dialog.exec.return_value = False  # User clicked "Cancel"
        mock_dialog_class.return_value = mock_dialog

        parent = MagicMock()
        result = check_config_validation("/path/to/config.yaml", parent_widget=parent)

        assert result.success is False
        assert result.cancelled is True


# =============================================================================
# check_video_paths Tests
# =============================================================================


class TestCheckVideoPaths:
    """Tests for check_video_paths function."""

    @patch("sleap_connect.api.check_video_paths")
    def test_all_paths_found(self, mock_check):
        """Should succeed if all video paths are found."""
        mock_check.return_value = PathCheckResult(
            all_found=True,
            total_videos=2,
            found_count=2,
            missing_count=0,
            videos=[
                VideoPathStatus(
                    filename="video1.mp4",
                    original_path="/local/video1.mp4",
                    worker_path="/data/video1.mp4",
                    found=True,
                ),
                VideoPathStatus(
                    filename="video2.mp4",
                    original_path="/local/video2.mp4",
                    worker_path="/data/video2.mp4",
                    found=True,
                ),
            ],
            slp_path="/data/labels.slp",
        )

        result = check_video_paths("/data/labels.slp", "test-room")

        assert result.success is True
        assert result.path_mappings == {
            "/local/video1.mp4": "/data/video1.mp4",
            "/local/video2.mp4": "/data/video2.mp4",
        }

    @patch("sleap_connect.api.check_video_paths")
    def test_missing_paths_no_parent(self, mock_check):
        """Should fail with error if paths missing and no parent widget."""
        mock_check.return_value = PathCheckResult(
            all_found=False,
            total_videos=2,
            found_count=1,
            missing_count=1,
            videos=[
                VideoPathStatus(
                    filename="video1.mp4",
                    original_path="/local/video1.mp4",
                    worker_path="/data/video1.mp4",
                    found=True,
                ),
                VideoPathStatus(
                    filename="video2.mp4",
                    original_path="/local/video2.mp4",
                    found=False,
                ),
            ],
            slp_path="/data/labels.slp",
        )

        result = check_video_paths("/data/labels.slp", "test-room", parent_widget=None)

        assert result.success is False
        assert "video2.mp4" in result.error

    @patch("sleap_connect.api.check_video_paths")
    def test_auth_error(self, mock_check):
        """Should fail with auth error."""
        from sleap_connect.api import AuthenticationError

        mock_check.side_effect = AuthenticationError("Not logged in")

        result = check_video_paths("/data/labels.slp", "test-room")

        assert result.success is False
        assert "Authentication error" in result.error

    @patch("sleap_connect.api.check_video_paths")
    def test_room_not_found(self, mock_check):
        """Should fail with room not found error."""
        from sleap_connect.api import RoomNotFoundError

        mock_check.side_effect = RoomNotFoundError("Room not found")

        result = check_video_paths("/data/labels.slp", "test-room")

        assert result.success is False
        assert "Room error" in result.error

    @patch("sleap_connect.api.check_video_paths")
    def test_network_error_allows_proceeding(self, mock_check):
        """Should allow proceeding if network error occurs."""
        mock_check.side_effect = Exception("Network timeout")

        result = check_video_paths("/data/labels.slp", "test-room")

        # Network errors allow proceeding with empty mappings
        assert result.success is True
        assert result.path_mappings == {}

    @patch("sleap_connect.api.check_video_paths")
    def test_missing_paths_user_resolves(self, mock_check):
        """Should succeed when user resolves missing paths via on_videos_missing."""
        # Video resolution now happens inside the API callback, so the
        # PathCheckResult comes back with path_mappings already populated.
        mock_check.return_value = PathCheckResult(
            all_found=False,
            total_videos=1,
            found_count=0,
            missing_count=1,
            videos=[
                VideoPathStatus(
                    filename="video.mp4",
                    original_path="/local/video.mp4",
                    found=False,
                ),
            ],
            slp_path="/data/labels.slp",
            path_mappings={"/local/video.mp4": "/worker/video.mp4"},
        )

        parent = MagicMock()
        result = check_video_paths("/data/labels.slp", "test-room", parent_widget=parent)

        assert result.success is True
        assert result.path_mappings == {"/local/video.mp4": "/worker/video.mp4"}

    @patch("sleap_connect.api.check_video_paths")
    def test_missing_paths_user_cancels(self, mock_check):
        """Should fail with cancelled=True when user cancels path resolution."""
        from sleap_connect.api import ConfigurationError

        # When user cancels, the on_videos_missing callback returns None,
        # causing the API to raise ConfigurationError.
        mock_check.side_effect = ConfigurationError(
            "Video path resolution cancelled by user."
        )

        parent = MagicMock()
        result = check_video_paths("/data/labels.slp", "test-room", parent_widget=parent)

        assert result.success is False
        assert result.cancelled is True

    @patch("sleap_connect.api.check_video_paths")
    def test_fully_embedded_pkg_slp_skips_dialog(self, mock_check):
        """Should succeed immediately for fully-embedded pkg.slp without showing
        PathResolutionDialog (missing == 0 and embedded > 0)."""
        mock_check.return_value = PathCheckResult(
            all_found=True,
            total_videos=3,
            found_count=0,
            missing_count=0,
            embedded_count=3,
            videos=[],
            slp_path="/data/labels.pkg.slp",
        )

        parent = MagicMock()
        result = check_video_paths(
            "/data/labels.pkg.slp", "test-room", parent_widget=parent
        )

        assert result.success is True
        # No video path mappings needed — all embedded
        assert result.path_mappings == {}

    @patch("sleap_connect.api.check_video_paths")
    def test_embedded_zero_falls_through_to_normal_path(self, mock_check):
        """Should not trigger the embedded-skip path when embedded_count == 0."""
        mock_check.return_value = PathCheckResult(
            all_found=True,
            total_videos=2,
            found_count=2,
            missing_count=0,
            embedded_count=0,
            videos=[
                VideoPathStatus(
                    filename="video1.mp4",
                    original_path="/local/video1.mp4",
                    worker_path="/data/video1.mp4",
                    found=True,
                ),
                VideoPathStatus(
                    filename="video2.mp4",
                    original_path="/local/video2.mp4",
                    worker_path="/data/video2.mp4",
                    found=True,
                ),
            ],
            slp_path="/data/labels.slp",
        )

        result = check_video_paths("/data/labels.slp", "test-room")

        # Falls through to all_found path — still succeeds
        assert result.success is True
        assert result.path_mappings == {
            "/local/video1.mp4": "/data/video1.mp4",
            "/local/video2.mp4": "/data/video2.mp4",
        }


# =============================================================================
# run_presubmission_checks Tests
# =============================================================================


class TestRunPresubmissionChecks:
    """Tests for run_presubmission_checks function."""

    @patch("sleap_connect.gui.presubmission.check_video_paths")
    @patch("sleap_connect.gui.presubmission.check_config_validation")
    @patch("sleap_connect.gui.presubmission.check_authentication")
    def test_all_checks_pass(self, mock_auth, mock_config, mock_paths):
        """Should succeed when all checks pass."""
        mock_auth.return_value = PresubmissionResult(success=True)
        mock_config.return_value = PresubmissionResult(
            success=True,
            validation_result=ValidationResult(
                valid=True, errors=[], warnings=[], config_path="/path/config.yaml"
            ),
        )
        mock_paths.return_value = PresubmissionResult(
            success=True,
            path_mappings={"/local/video.mp4": "/worker/video.mp4"},
        )

        result = run_presubmission_checks(
            config_path="/path/config.yaml",
            slp_path="/path/labels.slp",
            room_id="test-room",
        )

        assert result.success is True
        assert result.path_mappings == {"/local/video.mp4": "/worker/video.mp4"}
        mock_auth.assert_called_once()
        mock_config.assert_called_once()
        mock_paths.assert_called_once()

    @patch("sleap_connect.gui.presubmission.check_config_validation")
    @patch("sleap_connect.gui.presubmission.check_authentication")
    def test_auth_fails_stops_flow(self, mock_auth, mock_config):
        """Should stop at auth if it fails."""
        mock_auth.return_value = PresubmissionResult(
            success=False,
            error="Not logged in",
        )

        result = run_presubmission_checks(
            config_path="/path/config.yaml",
            slp_path="/path/labels.slp",
            room_id="test-room",
        )

        assert result.success is False
        assert "Not logged in" in result.error
        mock_auth.assert_called_once()
        mock_config.assert_not_called()

    @patch("sleap_connect.gui.presubmission.check_video_paths")
    @patch("sleap_connect.gui.presubmission.check_config_validation")
    @patch("sleap_connect.gui.presubmission.check_authentication")
    def test_config_validation_fails_stops_flow(self, mock_auth, mock_config, mock_paths):
        """Should stop at config validation if it fails."""
        mock_auth.return_value = PresubmissionResult(success=True)
        mock_config.return_value = PresubmissionResult(
            success=False,
            error="Invalid config",
        )

        result = run_presubmission_checks(
            config_path="/path/config.yaml",
            slp_path="/path/labels.slp",
            room_id="test-room",
        )

        assert result.success is False
        assert "Invalid config" in result.error
        mock_auth.assert_called_once()
        mock_config.assert_called_once()
        mock_paths.assert_not_called()

    @patch("sleap_connect.gui.presubmission.check_video_paths")
    @patch("sleap_connect.gui.presubmission.check_config_validation")
    @patch("sleap_connect.gui.presubmission.check_authentication")
    def test_path_check_fails_stops_flow(self, mock_auth, mock_config, mock_paths):
        """Should fail if path check fails."""
        mock_auth.return_value = PresubmissionResult(success=True)
        mock_config.return_value = PresubmissionResult(success=True)
        mock_paths.return_value = PresubmissionResult(
            success=False,
            cancelled=True,
            error="Path resolution cancelled",
        )

        result = run_presubmission_checks(
            config_path="/path/config.yaml",
            slp_path="/path/labels.slp",
            room_id="test-room",
        )

        assert result.success is False
        assert result.cancelled is True

    @patch("sleap_connect.gui.presubmission.check_authentication")
    def test_login_callback_passed(self, mock_auth):
        """Should pass login callback to check_authentication."""
        mock_auth.return_value = PresubmissionResult(success=False)
        login_callback = MagicMock()

        run_presubmission_checks(
            config_path="/path/config.yaml",
            slp_path="/path/labels.slp",
            room_id="test-room",
            on_login_required=login_callback,
        )

        mock_auth.assert_called_once_with(login_callback)


# =============================================================================
# PresubmissionFlow Tests
# =============================================================================


class TestPresubmissionFlow:
    """Tests for PresubmissionFlow class."""

    @patch("sleap_connect.gui.presubmission.run_presubmission_checks")
    def test_run_success(self, mock_checks):
        """Should run checks and store results."""
        mock_checks.return_value = PresubmissionResult(
            success=True,
            path_mappings={"/local/video.mp4": "/worker/video.mp4"},
            validation_result=ValidationResult(
                valid=True, errors=[], warnings=[], config_path="/path/config.yaml"
            ),
        )

        flow = PresubmissionFlow(
            config_path="/path/config.yaml",
            slp_path="/path/labels.slp",
            room_id="test-room",
        )

        result = flow.run()

        assert result is True
        assert flow.path_mappings == {"/local/video.mp4": "/worker/video.mp4"}
        assert flow.validation_result is not None
        assert flow.result.success is True

    @patch("sleap_connect.gui.presubmission.run_presubmission_checks")
    def test_run_failure(self, mock_checks):
        """Should handle failure and provide error message."""
        mock_checks.return_value = PresubmissionResult(
            success=False,
            error="Validation failed",
        )

        flow = PresubmissionFlow(
            config_path="/path/config.yaml",
            slp_path="/path/labels.slp",
            room_id="test-room",
        )

        result = flow.run()

        assert result is False
        assert flow.get_error_message() == "Validation failed"

    @patch("sleap_connect.gui.presubmission.run_presubmission_checks")
    def test_was_cancelled(self, mock_checks):
        """Should report if user cancelled."""
        mock_checks.return_value = PresubmissionResult(
            success=False,
            cancelled=True,
        )

        flow = PresubmissionFlow(
            config_path="/path/config.yaml",
            slp_path="/path/labels.slp",
            room_id="test-room",
        )

        flow.run()

        assert flow.was_cancelled() is True

    @patch("sleap_connect.gui.presubmission.run_presubmission_checks")
    def test_on_auth_required_callback_passed(self, mock_checks):
        """Should pass on_auth_required callback to run_presubmission_checks."""
        mock_checks.return_value = PresubmissionResult(success=True)
        auth_callback = MagicMock(return_value=True)

        flow = PresubmissionFlow(
            config_path="/path/config.yaml",
            slp_path="/path/labels.slp",
            room_id="test-room",
        )
        flow.on_auth_required = auth_callback

        parent = MagicMock()
        flow.run(parent=parent)

        mock_checks.assert_called_once()
        call_kwargs = mock_checks.call_args[1]
        assert call_kwargs["on_login_required"] == auth_callback
        assert call_kwargs["parent_widget"] == parent


# =============================================================================
# send_fn Threading Tests
# =============================================================================


class TestSendFnThreading:
    """Tests for send_fn parameter threading through presubmission flow."""

    @patch("sleap_connect.api.check_video_paths")
    @patch("sleap_connect.api.is_logged_in")
    def test_send_fn_passed_to_check_video_paths(self, mock_logged_in, mock_check):
        """send_fn should be passed through run_presubmission_checks."""
        mock_logged_in.return_value = True
        mock_check.return_value = PathCheckResult(
            all_found=True,
            total_videos=0,
            found_count=0,
            missing_count=0,
            videos=[],
            slp_path="/data/labels.slp",
        )

        send_fn = MagicMock()
        result = run_presubmission_checks(
            slp_path="/data/labels.slp",
            room_id="test-room",
            send_fn=send_fn,
        )

        assert result.success is True

    @patch("sleap_connect.gui.widgets.SlpPathDialog")
    @patch("sleap_connect.api.check_video_paths")
    @patch("sleap_connect.api.is_logged_in")
    def test_send_fn_passed_to_slp_dialog(
        self, mock_logged_in, mock_check, mock_dialog_cls
    ):
        """send_fn should be passed to SlpPathDialog when path is rejected."""
        mock_logged_in.return_value = True
        send_fn = MagicMock()

        # Configure the mock dialog
        mock_dialog = MagicMock()
        mock_dialog.exec.return_value = True
        mock_dialog.get_worker_path.return_value = "/corrected/labels.slp"
        mock_dialog_cls.return_value = mock_dialog

        # Simulate path rejection: on_path_rejected callback is called
        # with (path, error, send_fn_dc) — the third arg is the thread-safe
        # data channel send wrapper.
        mock_dc_send = MagicMock()

        def check_side_effect(*args, **kwargs):
            on_rejected = kwargs.get("on_path_rejected")
            if on_rejected:
                # Simulate worker rejecting the path
                corrected = on_rejected(
                    "/data/labels.slp", "File not found", mock_dc_send
                )
                if corrected:
                    return PathCheckResult(
                        all_found=True,
                        total_videos=0,
                        found_count=0,
                        missing_count=0,
                        videos=[],
                        slp_path=corrected,
                    )
            return PathCheckResult(
                all_found=True,
                total_videos=0,
                found_count=0,
                missing_count=0,
                videos=[],
                slp_path="/data/labels.slp",
            )

        mock_check.side_effect = check_side_effect

        parent = MagicMock()
        result = run_presubmission_checks(
            slp_path="/data/labels.slp",
            room_id="test-room",
            parent_widget=parent,
            send_fn=send_fn,
        )

        # Verify SlpPathDialog was created with the data channel's send_fn
        mock_dialog_cls.assert_called_once()
        call_kwargs = mock_dialog_cls.call_args[1]
        assert call_kwargs.get("send_fn") == mock_dc_send
        assert result.success is True

    @pytest.mark.xfail(reason="QDialog rejects MagicMock parent; needs real QWidget fixture")
    @patch("sleap_connect.gui.widgets.PathResolutionDialog")
    @patch("sleap_connect.api.check_video_paths")
    @patch("sleap_connect.api.is_logged_in")
    def test_send_fn_passed_to_path_resolution_dialog(
        self, mock_logged_in, mock_check, mock_dialog_cls
    ):
        """send_fn should be passed to PathResolutionDialog via on_videos_missing."""
        mock_logged_in.return_value = True

        # Configure the mock dialog
        mock_dialog = MagicMock()
        mock_dialog.exec.return_value = True
        mock_dialog._browser = None
        mock_dialog.get_resolved_paths.return_value = {
            "/local/video.mp4": "/mnt/data/video.mp4"
        }
        mock_dialog_cls.return_value = mock_dialog

        # Simulate on_videos_missing being called with a send_fn_dc
        mock_dc_send = MagicMock()
        missing_videos = [
            VideoPathStatus(
                filename="video.mp4",
                original_path="/local/video.mp4",
                found=False,
            ),
        ]

        def check_side_effect(*args, **kwargs):
            on_videos_missing = kwargs.get("on_videos_missing")
            if on_videos_missing:
                resolved = on_videos_missing(missing_videos, mock_dc_send)
                return PathCheckResult(
                    all_found=False,
                    total_videos=1,
                    found_count=0,
                    missing_count=1,
                    videos=missing_videos,
                    slp_path="/data/labels.slp",
                    path_mappings=resolved or {},
                )
            return PathCheckResult(
                all_found=False,
                total_videos=1,
                found_count=0,
                missing_count=1,
                videos=missing_videos,
                slp_path="/data/labels.slp",
            )

        mock_check.side_effect = check_side_effect

        parent = MagicMock()
        result = run_presubmission_checks(
            slp_path="/data/labels.slp",
            room_id="test-room",
            parent_widget=parent,
        )

        # Verify PathResolutionDialog was created with the data channel send_fn
        mock_dialog_cls.assert_called_once()
        call_args = mock_dialog_cls.call_args
        assert call_args[1].get("send_fn") == mock_dc_send
        assert result.success is True
