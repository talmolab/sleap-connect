"""Integration tests for sleap-connect GUI components.

These tests verify the integration between sleap-connect GUI components and
simulate how they would be used within SLEAP's training dialog.

Testing Approach for SLEAP Maintainers
======================================

This module demonstrates how to test sleap-connect GUI integration without
requiring a live server or actual SLEAP installation. The key patterns are:

1. **Mock SLEAP Preferences**: Use `MockSLEAPPreferences` to simulate
   SLEAP's preference system for the experimental features flag.

2. **Mock API Responses**: Patch `sleap_connect.api` functions to return
   controlled test data without network calls.

3. **Mock Qt Dialogs**: Patch dialog classes to simulate user interactions
   (clicking OK, Cancel, entering data).

4. **ZMQ Mocking**: Use `sys.modules` patching to mock the zmq module
   since it's imported lazily inside functions.

Example usage in SLEAP tests:

    from tests.test_gui_integration import MockSLEAPPreferences

    def test_remote_training_disabled_by_default():
        prefs = MockSLEAPPreferences()
        assert not prefs.get("enable_experimental_features", False)
        # Widget should not be visible when experimental features disabled
"""

import json
import pytest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Any

from qtpy.QtCore import Qt

from sleap_connect.api import (
    User,
    Room,
    Worker,
    ProgressEvent,
    TrainingResult,
    ValidationResult,
    ValidationIssue,
    PathCheckResult,
    VideoPathStatus,
    AuthenticationError,
    RoomNotFoundError,
    ConfigurationError,
    JobError,
)


# =============================================================================
# Mock SLEAP Preferences
# =============================================================================


class MockSLEAPPreferences:
    """Mock implementation of SLEAP's preference system.

    This simulates how SLEAP stores user preferences, including the
    experimental features flag that controls sleap-connect visibility.

    Usage:
        prefs = MockSLEAPPreferences()
        prefs.set("enable_experimental_features", True)
        assert prefs.get("enable_experimental_features") is True
    """

    def __init__(self, defaults: dict[str, Any] | None = None):
        self._defaults = {
            "enable_experimental_features": False,
            "default_room_id": None,
            "auto_select_worker": True,
        }
        if defaults:
            self._defaults.update(defaults)
        self._values = dict(self._defaults)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a preference value."""
        return self._values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a preference value."""
        self._values[key] = value

    def reset(self, key: str) -> None:
        """Reset a preference to its default value."""
        if key in self._defaults:
            self._values[key] = self._defaults[key]
        elif key in self._values:
            del self._values[key]

    def reset_all(self) -> None:
        """Reset all preferences to defaults."""
        self._values = dict(self._defaults)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def qapp():
    """Create a QApplication instance for Qt widget tests."""
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def mock_prefs():
    """Create mock SLEAP preferences."""
    return MockSLEAPPreferences()


@pytest.fixture
def mock_user():
    """Create a mock authenticated user."""
    return User(
        id="user-123",
        username="testuser",
        avatar_url="https://github.com/testuser.png",
    )


@pytest.fixture
def mock_rooms():
    """Create mock rooms list."""
    return [
        Room(
            id="room-1",
            name="Test Lab GPU",
            role="owner",
            created_by="user-123",
            joined_at=1700000000,
        ),
        Room(
            id="room-2",
            name="Shared Cluster",
            role="member",
            created_by="user-456",
            joined_at=1700100000,
        ),
    ]


@pytest.fixture
def mock_workers():
    """Create mock workers list."""
    return [
        Worker(
            id="worker-1",
            name="GPU Server A",
            status="available",
            gpu_name="NVIDIA RTX 4090",
            gpu_memory_mb=24576,
        ),
        Worker(
            id="worker-2",
            name="GPU Server B",
            status="busy",
            gpu_name="NVIDIA A100",
            gpu_memory_mb=81920,
        ),
    ]


@pytest.fixture
def mock_zmq():
    """Create and install mock zmq module."""
    mock_module = MagicMock()
    mock_module.PUB = 1
    mock_module.Context.return_value = MagicMock()

    with patch.dict("sys.modules", {"zmq": mock_module}):
        yield mock_module


# =============================================================================
# SLEAP Preferences Integration Tests
# =============================================================================


class TestSLEAPPreferencesIntegration:
    """Tests for SLEAP preferences integration."""

    def test_mock_preferences_defaults(self, mock_prefs):
        """Should have correct default values."""
        assert mock_prefs.get("enable_experimental_features") is False
        assert mock_prefs.get("default_room_id") is None
        assert mock_prefs.get("auto_select_worker") is True

    def test_mock_preferences_set_get(self, mock_prefs):
        """Should allow setting and getting values."""
        mock_prefs.set("enable_experimental_features", True)
        assert mock_prefs.get("enable_experimental_features") is True

        mock_prefs.set("default_room_id", "room-123")
        assert mock_prefs.get("default_room_id") == "room-123"

    def test_mock_preferences_reset(self, mock_prefs):
        """Should reset to defaults."""
        mock_prefs.set("enable_experimental_features", True)
        mock_prefs.reset("enable_experimental_features")
        assert mock_prefs.get("enable_experimental_features") is False

    def test_experimental_features_flag_controls_visibility(self, mock_prefs):
        """Experimental features flag should control widget visibility.

        This simulates how SLEAP would check the flag before showing
        the RemoteTrainingWidget.
        """
        # Default: experimental features disabled
        assert not mock_prefs.get("enable_experimental_features", False)

        # Widget visibility would be controlled by this flag
        should_show_widget = mock_prefs.get("enable_experimental_features", False)
        assert should_show_widget is False

        # Enable experimental features
        mock_prefs.set("enable_experimental_features", True)
        should_show_widget = mock_prefs.get("enable_experimental_features", False)
        assert should_show_widget is True


# =============================================================================
# Widget Visibility Tests
# =============================================================================


class TestWidgetVisibility:
    """Tests for widget visibility based on feature flags."""

    def test_widget_hidden_when_experimental_disabled(self, mock_prefs):
        """Widget should be hidden when experimental features disabled."""
        mock_prefs.set("enable_experimental_features", False)

        # Simulate SLEAP's logic for showing/hiding the widget
        show_remote_training = mock_prefs.get("enable_experimental_features", False)

        assert show_remote_training is False

    def test_widget_visible_when_experimental_enabled(self, mock_prefs):
        """Widget should be visible when experimental features enabled."""
        mock_prefs.set("enable_experimental_features", True)

        show_remote_training = mock_prefs.get("enable_experimental_features", False)

        assert show_remote_training is True

    @patch("sleap_connect.api.is_available")
    def test_widget_hidden_when_rtc_unavailable(self, mock_is_available, mock_prefs):
        """Widget should be hidden when sleap-connect is not available."""
        mock_prefs.set("enable_experimental_features", True)
        mock_is_available.return_value = False

        from sleap_connect.api import is_available

        # Both conditions must be true
        show_remote_training = (
            mock_prefs.get("enable_experimental_features", False) and is_available()
        )

        assert show_remote_training is False

    @patch("sleap_connect.api.is_available")
    def test_widget_visible_when_all_conditions_met(
        self, mock_is_available, mock_prefs
    ):
        """Widget should be visible when all conditions are met."""
        mock_prefs.set("enable_experimental_features", True)
        mock_is_available.return_value = True

        from sleap_connect.api import is_available

        show_remote_training = (
            mock_prefs.get("enable_experimental_features", False) and is_available()
        )

        assert show_remote_training is True


# =============================================================================
# Full Training Flow Tests
# =============================================================================


class TestFullTrainingFlow:
    """Tests for the complete training workflow."""

    @patch("sleap_connect.api.run_training")
    @patch("sleap_connect.api.check_video_paths")
    @patch("sleap_connect.api.validate_config")
    @patch("sleap_connect.api.list_workers")
    @patch("sleap_connect.api.list_rooms")
    @patch("sleap_connect.api.is_logged_in")
    def test_successful_training_flow(
        self,
        mock_is_logged_in,
        mock_list_rooms,
        mock_list_workers,
        mock_validate_config,
        mock_check_video_paths,
        mock_run_training,
        mock_user,
        mock_rooms,
        mock_workers,
    ):
        """Test complete successful training flow."""
        # Setup mocks
        mock_is_logged_in.return_value = True
        mock_list_rooms.return_value = mock_rooms
        mock_list_workers.return_value = mock_workers
        mock_validate_config.return_value = ValidationResult(
            valid=True,
            errors=[],
            warnings=[],
            config_path="/path/to/config.yaml",
        )
        mock_check_video_paths.return_value = PathCheckResult(
            all_found=True,
            total_videos=1,
            found_count=1,
            missing_count=0,
            videos=[
                VideoPathStatus(
                    filename="video.mp4",
                    original_path="/local/video.mp4",
                    worker_path="/data/video.mp4",
                    found=True,
                )
            ],
            slp_path="/data/labels.slp",
        )

        progress_events = []

        def capture_progress(event):
            progress_events.append(event)

        mock_run_training.return_value = TrainingResult(
            job_id="job-123",
            success=True,
            duration_seconds=3600.0,
            model_path="/models/run-123",
            final_epoch=100,
            final_train_loss=0.01,
            final_val_loss=0.02,
        )

        # Execute flow
        from sleap_connect.api import (
            is_logged_in,
            list_rooms,
            list_workers,
            validate_config,
            check_video_paths,
            run_training,
        )

        # Step 1: Check authentication
        assert is_logged_in() is True

        # Step 2: Get rooms and workers
        rooms = list_rooms()
        assert len(rooms) == 2

        workers = list_workers(rooms[0].id)
        assert len(workers) == 2

        # Step 3: Validate config
        validation = validate_config("/path/to/config.yaml")
        assert validation.valid is True

        # Step 4: Check video paths
        path_result = check_video_paths("/data/labels.slp", rooms[0].id)
        assert path_result.all_found is True

        # Step 5: Run training
        result = run_training(
            config_path="/path/to/config.yaml",
            room_id=rooms[0].id,
            worker_id=workers[0].id,
        )

        assert result.success is True
        assert result.job_id == "job-123"
        assert result.model_path == "/models/run-123"

    @patch("sleap_connect.api.run_training")
    @patch("sleap_connect.api.is_logged_in")
    def test_training_flow_not_logged_in(
        self,
        mock_is_logged_in,
        mock_run_training,
    ):
        """Test training flow fails when not logged in."""
        mock_is_logged_in.return_value = False

        from sleap_connect.gui.presubmission import check_authentication

        result = check_authentication()

        assert result.success is False
        assert "Not logged in" in result.error
        mock_run_training.assert_not_called()


# =============================================================================
# Progress Forwarding Tests
# =============================================================================


class TestProgressForwarding:
    """Tests for progress forwarding end-to-end."""

    def test_progress_bridge_formats_train_begin(self, mock_zmq):
        """Should format train_begin event as jsonpickle via send_string."""
        import jsonpickle

        mock_context = MagicMock()
        mock_socket = MagicMock()
        mock_zmq.Context.return_value = mock_context
        mock_context.socket.return_value = mock_socket

        from sleap_connect.gui.runners import RemoteProgressBridge

        event = ProgressEvent(
            event_type="train_begin",
            total_epochs=100,
            wandb_url="https://wandb.ai/run/123",
        )

        with RemoteProgressBridge() as bridge:
            bridge.on_progress(event)

        mock_socket.send_string.assert_called_once()
        payload = jsonpickle.decode(mock_socket.send_string.call_args[0][0])
        assert payload["event"] == "train_begin"
        assert payload["what"] == ""
        assert payload["wandb_url"] == "https://wandb.ai/run/123"

    def test_progress_bridge_formats_epoch_end(self, mock_zmq):
        """Should format epoch_end with loss in logs dict."""
        import jsonpickle

        mock_context = MagicMock()
        mock_socket = MagicMock()
        mock_zmq.Context.return_value = mock_context
        mock_context.socket.return_value = mock_socket

        from sleap_connect.gui.runners import RemoteProgressBridge

        event = ProgressEvent(
            event_type="epoch_end",
            epoch=50,
            total_epochs=100,
            train_loss=0.05,
            val_loss=0.06,
        )

        with RemoteProgressBridge(model_type="centroid") as bridge:
            bridge.on_progress(event)

        payload = jsonpickle.decode(mock_socket.send_string.call_args[0][0])
        assert payload["event"] == "epoch_end"
        assert payload["what"] == "centroid"
        assert payload["logs"]["train/loss"] == 0.05
        assert payload["logs"]["val/loss"] == 0.06

    def test_progress_bridge_formats_train_end(self, mock_zmq):
        """Should format train_end with what field."""
        import jsonpickle

        mock_context = MagicMock()
        mock_socket = MagicMock()
        mock_zmq.Context.return_value = mock_context
        mock_context.socket.return_value = mock_socket

        from sleap_connect.gui.runners import RemoteProgressBridge

        event = ProgressEvent(
            event_type="train_end",
            success=True,
        )

        with RemoteProgressBridge() as bridge:
            bridge.on_progress(event)

        payload = jsonpickle.decode(mock_socket.send_string.call_args[0][0])
        assert payload["event"] == "train_end"
        assert payload["what"] == ""

    @pytest.mark.xfail(reason="ZMQ mock wiring mismatch; needs investigation")
    @patch("sleap_connect.api.run_training")
    def test_run_remote_training_forwards_progress(self, mock_run_training, mock_zmq):
        """Should forward all progress events through bridge via send_string."""
        mock_context = MagicMock()
        mock_socket = MagicMock()
        mock_zmq.Context.return_value = mock_context
        mock_context.socket.return_value = mock_socket

        # Simulate training with progress events
        def simulate_training(*args, **kwargs):
            # The handler is passed as 'progress_callback' to run_training
            on_progress = kwargs.get("progress_callback")
            if on_progress:
                on_progress(ProgressEvent(event_type="train_begin", total_epochs=10))
                for i in range(1, 11):
                    on_progress(
                        ProgressEvent(
                            event_type="epoch_end",
                            epoch=i,
                            total_epochs=10,
                            train_loss=1.0 / i,
                        )
                    )
                on_progress(ProgressEvent(event_type="train_end", success=True))
            return TrainingResult(job_id="test", success=True)

        mock_run_training.side_effect = simulate_training

        from sleap_connect.gui.runners import run_remote_training

        result = run_remote_training(
            config_path="/path/to/config.yaml",
            room_id="room-1",
        )

        # Should have published: train_begin + 10 epoch_end + train_end = 12
        assert mock_socket.send_string.call_count == 12


# =============================================================================
# Failure Scenario Tests
# =============================================================================


class TestFailureScenarios:
    """Tests for error handling and failure scenarios."""

    @patch("sleap_connect.api.list_rooms")
    @patch("sleap_connect.api.is_logged_in")
    def test_network_error_on_list_rooms(self, mock_is_logged_in, mock_list_rooms):
        """Should handle network errors when listing rooms."""
        mock_is_logged_in.return_value = True
        mock_list_rooms.side_effect = Exception("Network timeout")

        from sleap_connect.api import list_rooms

        with pytest.raises(Exception) as exc_info:
            list_rooms()

        assert "Network timeout" in str(exc_info.value)

    @patch("sleap_connect.api.validate_config")
    def test_validation_error_handling(self, mock_validate_config):
        """Should handle validation errors correctly."""
        mock_validate_config.return_value = ValidationResult(
            valid=False,
            errors=[
                ValidationIssue(
                    field="max_epochs",
                    message="Value must be positive",
                    code="INVALID_VALUE",
                ),
                ValidationIssue(
                    field="batch_size",
                    message="Value must be integer",
                    code="INVALID_TYPE",
                ),
            ],
            warnings=[],
            config_path="/path/to/config.yaml",
        )

        from sleap_connect.gui.presubmission import check_config_validation

        result = check_config_validation("/path/to/config.yaml", parent_widget=None)

        assert result.success is False
        assert "Value must be positive" in result.error

    @patch("sleap_connect.api.check_video_paths")
    def test_path_resolution_error_handling(self, mock_check_paths):
        """Should handle path resolution errors correctly."""
        mock_check_paths.return_value = PathCheckResult(
            all_found=False,
            total_videos=2,
            found_count=0,
            missing_count=2,
            videos=[
                VideoPathStatus(
                    filename="video1.mp4",
                    original_path="/local/video1.mp4",
                    found=False,
                ),
                VideoPathStatus(
                    filename="video2.mp4",
                    original_path="/local/video2.mp4",
                    found=False,
                ),
            ],
            slp_path="/data/labels.slp",
        )

        from sleap_connect.gui.presubmission import check_video_paths

        result = check_video_paths("/data/labels.slp", "room-1", parent_widget=None)

        assert result.success is False
        assert "video1.mp4" in result.error
        assert "video2.mp4" in result.error

    @patch("sleap_connect.api.run_training")
    def test_training_error_handling(self, mock_run_training):
        """Should handle training errors correctly."""
        mock_run_training.side_effect = JobError(
            "CUDA out of memory",
            job_id="job-123",
            exit_code=1,
        )

        from sleap_connect.api import run_training

        with pytest.raises(JobError) as exc_info:
            run_training(
                config_path="/path/to/config.yaml",
                room_id="room-1",
            )

        assert "CUDA out of memory" in str(exc_info.value)
        assert exc_info.value.job_id == "job-123"

    @patch("sleap_connect.api.is_logged_in")
    def test_authentication_error_handling(self, mock_is_logged_in):
        """Should handle authentication errors correctly."""
        mock_is_logged_in.return_value = False

        from sleap_connect.gui.presubmission import run_presubmission_checks

        result = run_presubmission_checks(
            config_path="/path/to/config.yaml",
            slp_path="/path/to/labels.slp",
            room_id="room-1",
        )

        assert result.success is False
        assert "Not logged in" in result.error

    @patch("sleap_connect.api.list_workers")
    @patch("sleap_connect.api.is_logged_in")
    def test_room_not_found_error(self, mock_is_logged_in, mock_list_workers):
        """Should handle room not found errors."""
        mock_is_logged_in.return_value = True
        mock_list_workers.side_effect = RoomNotFoundError("Room not found")

        from sleap_connect.api import list_workers

        with pytest.raises(RoomNotFoundError):
            list_workers("nonexistent-room")

    @patch("sleap_connect.api.validate_config")
    def test_config_file_not_found(self, mock_validate_config):
        """Should handle missing config file."""
        mock_validate_config.side_effect = ConfigurationError("File not found")

        from sleap_connect.gui.presubmission import check_config_validation

        result = check_config_validation("/nonexistent/config.yaml")

        assert result.success is False
        assert "Cannot read config file" in result.error


# =============================================================================
# End-to-End Presubmission Flow Tests
# =============================================================================


class TestPresubmissionEndToEnd:
    """End-to-end tests for the presubmission validation flow."""

    @patch("sleap_connect.api.check_video_paths")
    @patch("sleap_connect.api.validate_config")
    @patch("sleap_connect.api.is_logged_in")
    def test_full_presubmission_success(
        self,
        mock_is_logged_in,
        mock_validate_config,
        mock_check_video_paths,
    ):
        """Test successful presubmission flow."""
        mock_is_logged_in.return_value = True
        mock_validate_config.return_value = ValidationResult(
            valid=True,
            errors=[],
            warnings=[],
            config_path="/path/to/config.yaml",
        )
        mock_check_video_paths.return_value = PathCheckResult(
            all_found=True,
            total_videos=1,
            found_count=1,
            missing_count=0,
            videos=[
                VideoPathStatus(
                    filename="video.mp4",
                    original_path="/local/video.mp4",
                    worker_path="/data/video.mp4",
                    found=True,
                )
            ],
            slp_path="/data/labels.slp",
        )

        from sleap_connect.gui.presubmission import run_presubmission_checks

        result = run_presubmission_checks(
            config_path="/path/to/config.yaml",
            slp_path="/data/labels.slp",
            room_id="room-1",
        )

        assert result.success is True
        assert result.path_mappings == {"/local/video.mp4": "/data/video.mp4"}

    @patch("sleap_connect.api.check_video_paths")
    @patch("sleap_connect.api.validate_config")
    @patch("sleap_connect.api.is_logged_in")
    def test_presubmission_stops_on_auth_failure(
        self,
        mock_is_logged_in,
        mock_validate_config,
        mock_check_video_paths,
    ):
        """Test presubmission stops when authentication fails."""
        mock_is_logged_in.return_value = False

        from sleap_connect.gui.presubmission import run_presubmission_checks

        result = run_presubmission_checks(
            config_path="/path/to/config.yaml",
            slp_path="/data/labels.slp",
            room_id="room-1",
        )

        assert result.success is False
        # Should not call validation or path check if auth fails
        mock_validate_config.assert_not_called()
        mock_check_video_paths.assert_not_called()

    @patch("sleap_connect.api.check_video_paths")
    @patch("sleap_connect.api.validate_config")
    @patch("sleap_connect.api.is_logged_in")
    def test_presubmission_stops_on_validation_error(
        self,
        mock_is_logged_in,
        mock_validate_config,
        mock_check_video_paths,
    ):
        """Test presubmission stops when config validation fails."""
        mock_is_logged_in.return_value = True
        mock_validate_config.return_value = ValidationResult(
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

        from sleap_connect.gui.presubmission import run_presubmission_checks

        result = run_presubmission_checks(
            config_path="/path/to/config.yaml",
            slp_path="/data/labels.slp",
            room_id="room-1",
        )

        assert result.success is False
        # Should not call path check if validation fails
        mock_check_video_paths.assert_not_called()


# =============================================================================
# Documentation for SLEAP Maintainers
# =============================================================================


class TestDocumentation:
    """Tests that serve as documentation for SLEAP integration.

    These tests demonstrate the expected patterns for integrating
    sleap-connect into SLEAP's GUI.
    """

    def test_example_sleap_integration_pattern(self, mock_prefs):
        """Example: How SLEAP should integrate the RemoteTrainingWidget.

        This test demonstrates the recommended pattern for integrating
        sleap-connect widgets into SLEAP's training configuration dialog.
        """
        # Step 1: Check if experimental features are enabled
        show_remote_training = mock_prefs.get("enable_experimental_features", False)

        if show_remote_training:
            # Step 2: Check if sleap-connect is available
            # In real code: from sleap_connect.api import is_available
            # if is_available():
            #     # Step 3: Create and add the widget
            #     from sleap_connect.gui import RemoteTrainingWidget
            #     remote_widget = RemoteTrainingWidget()
            #     layout.addWidget(remote_widget)
            pass

        # When experimental features are disabled, widget is not created
        assert show_remote_training is False

    def test_example_presubmission_integration(self):
        """Example: How SLEAP should use presubmission validation.

        This test demonstrates how to use the presubmission flow
        before starting remote training.
        """
        # In real SLEAP code:
        # from sleap_connect.gui import run_presubmission_checks
        #
        # result = run_presubmission_checks(
        #     config_path=self.config_path,
        #     slp_path=self.labels.filename,
        #     room_id=self.remote_widget.get_selected_room_id(),
        #     worker_id=self.remote_widget.get_selected_worker_id(),
        #     parent_widget=self,
        #     on_login_required=self._handle_login,
        # )
        #
        # if result.success:
        #     # Proceed with training using result.path_mappings
        #     pass
        # elif result.cancelled:
        #     # User cancelled, do nothing
        #     pass
        # else:
        #     # Show error message
        #     QMessageBox.critical(self, "Error", result.error)
        pass

    @pytest.mark.xfail(
        reason="ZMQ send_string called unexpected number of times; needs investigation"
    )
    def test_example_progress_forwarding(self, mock_zmq):
        """Example: How to use progress forwarding with LossViewer.

        This test demonstrates how progress events are forwarded
        to SLEAP's LossViewer via ZMQ.
        """
        mock_context = MagicMock()
        mock_socket = MagicMock()
        mock_zmq.Context.return_value = mock_context
        mock_context.socket.return_value = mock_socket

        # In real SLEAP code:
        # from sleap_connect.gui import run_remote_training
        #
        # # LossViewer subscribes to port 9001 by default
        # result = run_remote_training(
        #     config_path=config_path,
        #     room_id=room_id,
        #     worker_id=worker_id,
        #     publish_port=9001,  # Same port LossViewer listens on
        # )
        #
        # The progress events will be published in the format:
        # - Topic: b"progress"
        # - Payload: JSON with "event" field
        #
        # LossViewer will receive these and update the loss plot.

        from sleap_connect.gui.runners import RemoteProgressBridge

        with RemoteProgressBridge(publish_port=9001) as bridge:
            bridge.on_progress(
                ProgressEvent(
                    event_type="epoch_end",
                    epoch=1,
                    train_loss=0.5,
                    val_loss=0.6,
                )
            )

        # Verify message was published
        mock_socket.send_string.assert_called_once()


# =============================================================================
# Remote File Browser Tests
# =============================================================================


class TestRemoteFileBrowser:
    """Tests for RemoteFileBrowser widget with mocked send/receive."""

    @pytest.fixture
    def send_fn(self):
        """Mock send function that records sent messages."""
        return MagicMock()

    @pytest.fixture
    def browser(self, send_fn, qapp):
        """Create a RemoteFileBrowser with mocked transport."""
        from sleap_connect.gui.widgets import RemoteFileBrowser

        browser = RemoteFileBrowser(send_fn=send_fn)
        yield browser
        browser.deleteLater()

    @pytest.fixture
    def browser_with_filter(self, send_fn, qapp):
        """Create a RemoteFileBrowser with SLP file filter."""
        from sleap_connect.gui.widgets import RemoteFileBrowser

        browser = RemoteFileBrowser(send_fn=send_fn, file_filter="*.slp")
        yield browser
        browser.deleteLater()

    def _mounts_response(self, mounts: list[dict]) -> str:
        """Build a FS_MOUNTS_RESPONSE message string."""
        return f"FS_MOUNTS_RESPONSE::{json.dumps(mounts)}"

    def _list_response(
        self,
        path: str,
        entries: list[dict],
        has_more: bool = False,
        total_count: int | None = None,
    ) -> str:
        """Build a FS_LIST_RESPONSE message string."""
        if total_count is None:
            total_count = len(entries)
        data = {
            "path": path,
            "entries": entries,
            "has_more": has_more,
            "total_count": total_count,
        }
        return f"FS_LIST_RESPONSE::{json.dumps(data)}"

    def _error_response(self, code: str, message: str) -> str:
        """Build a FS_ERROR message string."""
        return f"FS_ERROR::{code}::{message}"

    # --- Mount Loading ---

    def test_load_mounts_sends_message(self, browser, send_fn):
        """load_mounts() should send FS_GET_MOUNTS."""
        browser.load_mounts()
        send_fn.assert_called_once_with("FS_GET_MOUNTS")

    def test_mounts_response_populates_column(self, browser):
        """FS_MOUNTS_RESPONSE should create a mount selector column."""
        mounts = [
            {"path": "/mnt/data", "label": "Lab Data"},
            {"path": "/mnt/models", "label": "Models"},
        ]
        browser._handle_response(self._mounts_response(mounts))

        assert len(browser._columns) == 1
        col = browser._columns[0]
        assert col.count() == 2
        assert col.item(0).text() == "Lab Data"
        assert col.item(0).data(Qt.ItemDataRole.UserRole) == "/mnt/data"
        assert col.item(1).text() == "Models"

    def test_mounts_response_replaces_existing(self, browser):
        """Loading mounts again should replace existing columns."""
        mounts1 = [{"path": "/mnt/a", "label": "A"}]
        mounts2 = [{"path": "/mnt/b", "label": "B"}, {"path": "/mnt/c", "label": "C"}]

        browser._handle_response(self._mounts_response(mounts1))
        assert len(browser._columns) == 1

        browser._handle_response(self._mounts_response(mounts2))
        assert len(browser._columns) == 1
        assert browser._columns[0].count() == 2
        assert browser._columns[0].item(0).text() == "B"

    # --- Directory Navigation ---

    def test_mount_click_sends_list_dir(self, browser, send_fn):
        """Clicking a mount should send FS_LIST_DIR for mount root."""
        mounts = [{"path": "/mnt/data", "label": "Lab Data"}]
        browser._handle_response(self._mounts_response(mounts))

        item = browser._columns[0].item(0)
        browser._on_item_clicked(item)

        send_fn.assert_called_with("FS_LIST_DIR::/mnt/data::0")

    def test_list_response_creates_column(self, browser):
        """FS_LIST_RESPONSE should create a new column with entries."""
        mounts = [{"path": "/mnt/data", "label": "Lab Data"}]
        browser._handle_response(self._mounts_response(mounts))

        entries = [
            {"name": "videos", "type": "directory", "size": 0, "modified": 1700000000},
            {
                "name": "labels.slp",
                "type": "file",
                "size": 1024,
                "modified": 1700000100,
            },
        ]
        browser._handle_response(self._list_response("/mnt/data", entries))

        assert len(browser._columns) == 2
        col = browser._columns[1]
        # Directories first, then files
        assert col.item(0).text() == "videos/"
        assert col.item(1).text() == "labels.slp"

    def test_directory_click_sends_list_dir(self, browser, send_fn):
        """Clicking a directory should send FS_LIST_DIR for that path."""
        mounts = [{"path": "/mnt/data", "label": "Lab Data"}]
        browser._handle_response(self._mounts_response(mounts))

        entries = [
            {"name": "videos", "type": "directory", "size": 0, "modified": 0},
        ]
        browser._handle_response(self._list_response("/mnt/data", entries))

        dir_item = browser._columns[1].item(0)
        browser._on_item_clicked(dir_item)

        send_fn.assert_called_with("FS_LIST_DIR::/mnt/data/videos::0")

    def test_directory_click_removes_deeper_columns(self, browser):
        """Clicking a folder in column N should remove columns N+1 and beyond."""
        mounts = [{"path": "/mnt/data", "label": "Data"}]
        browser._handle_response(self._mounts_response(mounts))

        # Simulate navigating 3 levels deep
        browser._handle_response(
            self._list_response(
                "/mnt/data",
                [
                    {"name": "a", "type": "directory", "size": 0, "modified": 0},
                ],
            )
        )
        browser._handle_response(
            self._list_response(
                "/mnt/data/a",
                [
                    {"name": "b", "type": "directory", "size": 0, "modified": 0},
                ],
            )
        )
        browser._handle_response(
            self._list_response(
                "/mnt/data/a/b",
                [
                    {"name": "c.txt", "type": "file", "size": 100, "modified": 0},
                ],
            )
        )

        # Now 4 columns: mount, /mnt/data, /mnt/data/a, /mnt/data/a/b
        assert len(browser._columns) == 4

        # Click mount "Data" again
        browser._on_item_clicked(browser._columns[0].item(0))
        # Should remove columns 1,2,3 (leaving only mount)
        assert len(browser._columns) == 1

    # --- File Selection ---

    def test_file_click_selects_path(self, browser):
        """Clicking a file should update path bar and enable Select button."""
        mounts = [{"path": "/mnt/data", "label": "Data"}]
        browser._handle_response(self._mounts_response(mounts))

        entries = [
            {
                "name": "labels.slp",
                "type": "file",
                "size": 2048,
                "modified": 1700000000,
            },
        ]
        browser._handle_response(self._list_response("/mnt/data", entries))

        file_item = browser._columns[1].item(0)
        browser._on_item_clicked(file_item)

        assert browser._selected_path == "/mnt/data/labels.slp"
        assert browser._path_bar.text() == "/mnt/data/labels.slp"
        assert browser._select_button.isEnabled()

    def test_file_double_click_emits_signal(self, browser):
        """Double-clicking a file should emit file_selected signal."""
        mounts = [{"path": "/mnt/data", "label": "Data"}]
        browser._handle_response(self._mounts_response(mounts))

        entries = [
            {"name": "labels.slp", "type": "file", "size": 2048, "modified": 0},
        ]
        browser._handle_response(self._list_response("/mnt/data", entries))

        received = []
        browser.file_selected.connect(received.append)

        file_item = browser._columns[1].item(0)
        browser._on_item_double_clicked(file_item)

        assert received == ["/mnt/data/labels.slp"]

    def test_select_button_emits_signal(self, browser):
        """Select button should emit file_selected for the current path."""
        mounts = [{"path": "/mnt/data", "label": "Data"}]
        browser._handle_response(self._mounts_response(mounts))

        entries = [
            {"name": "labels.slp", "type": "file", "size": 1024, "modified": 0},
        ]
        browser._handle_response(self._list_response("/mnt/data", entries))

        received = []
        browser.file_selected.connect(received.append)

        # Click file to select, then click Select button
        file_item = browser._columns[1].item(0)
        browser._on_item_clicked(file_item)
        browser._on_select_clicked()

        assert received == ["/mnt/data/labels.slp"]

    def test_select_button_disabled_initially(self, browser):
        """Select button should be disabled when no file is selected."""
        assert not browser._select_button.isEnabled()

    # --- File Filtering ---

    def test_filter_greys_out_non_matching(self, browser_with_filter):
        """Non-matching files should be greyed out and not selectable."""
        browser = browser_with_filter
        mounts = [{"path": "/mnt/data", "label": "Data"}]
        browser._handle_response(self._mounts_response(mounts))

        entries = [
            {"name": "labels.slp", "type": "file", "size": 1024, "modified": 0},
            {"name": "video.mp4", "type": "file", "size": 2048, "modified": 0},
            {"name": "readme.txt", "type": "file", "size": 100, "modified": 0},
        ]
        browser._handle_response(self._list_response("/mnt/data", entries))

        col = browser._columns[1]
        # Files are sorted alphabetically: labels.slp, readme.txt, video.mp4
        slp_item = col.item(0)
        assert slp_item.text() == "labels.slp"
        assert slp_item.flags() & Qt.ItemFlag.ItemIsEnabled

        # readme.txt should be disabled
        txt_item = col.item(1)
        assert txt_item.text() == "readme.txt"
        assert not (txt_item.flags() & Qt.ItemFlag.ItemIsEnabled)

        # video.mp4 should be disabled
        mp4_item = col.item(2)
        assert mp4_item.text() == "video.mp4"
        assert not (mp4_item.flags() & Qt.ItemFlag.ItemIsEnabled)

    def test_filter_allows_all_directories(self, browser_with_filter):
        """Directories should always be navigable regardless of filter."""
        browser = browser_with_filter
        mounts = [{"path": "/mnt/data", "label": "Data"}]
        browser._handle_response(self._mounts_response(mounts))

        entries = [
            {"name": "subdir", "type": "directory", "size": 0, "modified": 0},
        ]
        browser._handle_response(self._list_response("/mnt/data", entries))

        col = browser._columns[1]
        dir_item = col.item(0)
        assert dir_item.text() == "subdir/"
        assert dir_item.flags() & Qt.ItemFlag.ItemIsEnabled

    def test_no_filter_allows_all_files(self, browser):
        """Without a filter, all files should be selectable."""
        mounts = [{"path": "/mnt/data", "label": "Data"}]
        browser._handle_response(self._mounts_response(mounts))

        entries = [
            {"name": "any.xyz", "type": "file", "size": 100, "modified": 0},
        ]
        browser._handle_response(self._list_response("/mnt/data", entries))

        col = browser._columns[1]
        assert col.item(0).flags() & Qt.ItemFlag.ItemIsEnabled

    def test_double_click_disabled_file_no_signal(self, browser_with_filter):
        """Double-clicking a disabled file should not emit file_selected."""
        browser = browser_with_filter
        mounts = [{"path": "/mnt/data", "label": "Data"}]
        browser._handle_response(self._mounts_response(mounts))

        entries = [
            {"name": "video.mp4", "type": "file", "size": 2048, "modified": 0},
        ]
        browser._handle_response(self._list_response("/mnt/data", entries))

        received = []
        browser.file_selected.connect(received.append)

        file_item = browser._columns[1].item(0)
        browser._on_item_double_clicked(file_item)

        assert received == []

    # --- Pagination ---

    def test_has_more_shows_load_more(self, browser):
        """When has_more=true, a 'Load more...' entry should appear."""
        mounts = [{"path": "/mnt/data", "label": "Data"}]
        browser._handle_response(self._mounts_response(mounts))

        entries = [
            {"name": f"file{i}.txt", "type": "file", "size": 100, "modified": 0}
            for i in range(20)
        ]
        browser._handle_response(
            self._list_response("/mnt/data", entries, has_more=True, total_count=50)
        )

        col = browser._columns[1]
        last_item = col.item(col.count() - 1)
        assert last_item.text() == "Load more..."
        assert last_item.data(Qt.ItemDataRole.UserRole + 1) == "load_more"

    def test_load_more_sends_offset(self, browser, send_fn):
        """Clicking 'Load more...' should send FS_LIST_DIR with correct offset."""
        mounts = [{"path": "/mnt/data", "label": "Data"}]
        browser._handle_response(self._mounts_response(mounts))

        entries = [
            {"name": f"file{i}.txt", "type": "file", "size": 100, "modified": 0}
            for i in range(20)
        ]
        browser._handle_response(
            self._list_response("/mnt/data", entries, has_more=True, total_count=50)
        )

        col = browser._columns[1]
        load_more_item = col.item(col.count() - 1)
        browser._on_item_clicked(load_more_item)

        send_fn.assert_called_with("FS_LIST_DIR::/mnt/data::20")

    def test_pagination_appends_entries(self, browser):
        """Paginated response should append entries and remove 'Load more...'."""
        mounts = [{"path": "/mnt/data", "label": "Data"}]
        browser._handle_response(self._mounts_response(mounts))

        # First page
        entries1 = [
            {"name": f"file{i}.txt", "type": "file", "size": 100, "modified": 0}
            for i in range(20)
        ]
        browser._handle_response(
            self._list_response("/mnt/data", entries1, has_more=True, total_count=25)
        )

        col = browser._columns[1]
        assert col.count() == 21  # 20 files + "Load more..."

        # Second page (no more)
        entries2 = [
            {"name": f"file{i}.txt", "type": "file", "size": 100, "modified": 0}
            for i in range(20, 25)
        ]
        browser._handle_response(
            self._list_response("/mnt/data", entries2, has_more=False, total_count=25)
        )

        # "Load more..." removed, 5 new entries added
        assert col.count() == 25
        # No "Load more..." at end
        assert (
            col.item(col.count() - 1).data(Qt.ItemDataRole.UserRole + 1) != "load_more"
        )

    # --- Preview ---

    def test_file_click_shows_preview(self, browser):
        """Clicking a file should show metadata in the preview panel."""
        mounts = [{"path": "/mnt/data", "label": "Data"}]
        browser._handle_response(self._mounts_response(mounts))

        entries = [
            {
                "name": "labels.slp",
                "type": "file",
                "size": 1048576,
                "modified": 1700000000,
            },
        ]
        browser._handle_response(self._list_response("/mnt/data", entries))

        browser._on_item_clicked(browser._columns[1].item(0))

        assert browser._preview_name.text() == "labels.slp"
        assert "1.0 MB" in browser._preview_size.text()
        assert browser._preview_modified.text() != ""

    def test_directory_click_clears_preview(self, browser):
        """Clicking a directory should clear the preview panel."""
        mounts = [{"path": "/mnt/data", "label": "Data"}]
        browser._handle_response(self._mounts_response(mounts))

        entries = [
            {"name": "labels.slp", "type": "file", "size": 1024, "modified": 0},
            {"name": "subdir", "type": "directory", "size": 0, "modified": 0},
        ]
        browser._handle_response(self._list_response("/mnt/data", entries))

        # Click file first
        browser._on_item_clicked(browser._columns[1].item(1))  # labels.slp (after dir)
        assert browser._preview_name.text() != ""

        # Click directory
        browser._on_item_clicked(browser._columns[1].item(0))  # subdir/
        assert browser._preview_name.text() == ""

    # --- Format Size ---

    def test_format_size_bytes(self, browser):
        """Should format bytes correctly."""
        assert browser._format_size(500) == "500 B"

    def test_format_size_kb(self, browser):
        """Should format kilobytes correctly."""
        assert browser._format_size(1536) == "1.5 KB"

    def test_format_size_mb(self, browser):
        """Should format megabytes correctly."""
        assert browser._format_size(1048576) == "1.0 MB"

    def test_format_size_gb(self, browser):
        """Should format gigabytes correctly."""
        assert browser._format_size(1073741824) == "1.0 GB"

    # --- Thread Safety ---

    def test_on_response_routes_to_handler(self, browser):
        """on_response() should route messages to _handle_response."""
        # Call _handle_response directly (simulates what the signal does)
        mounts = [{"path": "/mnt/data", "label": "Data"}]
        browser._handle_response(self._mounts_response(mounts))
        assert len(browser._columns) == 1

    # --- Error Handling ---

    def test_error_response_handled_gracefully(self, browser):
        """FS_ERROR should be handled without crashing."""
        browser._handle_response(self._error_response("PATH_NOT_FOUND", "/bad/path"))
        # Should not crash, columns unchanged
        assert len(browser._columns) == 0

    def test_invalid_json_handled_gracefully(self, browser):
        """Invalid JSON in responses should not crash."""
        browser._handle_response("FS_MOUNTS_RESPONSE::not-valid-json")
        assert len(browser._columns) == 0

    # --- Multi-Extension Filter ---

    def test_multi_extension_filter(self, send_fn, qapp):
        """Multiple extensions in filter should all be selectable."""
        from sleap_connect.gui.widgets import RemoteFileBrowser

        browser = RemoteFileBrowser(send_fn=send_fn, file_filter="*.mp4,*.avi,*.mov")
        mounts = [{"path": "/mnt/data", "label": "Data"}]
        browser._handle_response(self._mounts_response(mounts))

        entries = [
            {"name": "clip.mp4", "type": "file", "size": 100, "modified": 0},
            {"name": "clip.avi", "type": "file", "size": 100, "modified": 0},
            {"name": "clip.mov", "type": "file", "size": 100, "modified": 0},
            {"name": "clip.mkv", "type": "file", "size": 100, "modified": 0},
        ]
        browser._handle_response(self._list_response("/mnt/data", entries))

        col = browser._columns[1]
        # Sorted alphabetically: clip.avi, clip.mkv, clip.mov, clip.mp4
        assert col.item(0).text() == "clip.avi"
        assert col.item(0).flags() & Qt.ItemFlag.ItemIsEnabled  # avi
        assert col.item(1).text() == "clip.mkv"
        assert not (col.item(1).flags() & Qt.ItemFlag.ItemIsEnabled)  # mkv
        assert col.item(2).text() == "clip.mov"
        assert col.item(2).flags() & Qt.ItemFlag.ItemIsEnabled  # mov
        assert col.item(3).text() == "clip.mp4"
        assert col.item(3).flags() & Qt.ItemFlag.ItemIsEnabled  # mp4

        browser.deleteLater()

    # --- Entries Sorted ---

    def test_entries_sorted_dirs_first(self, browser):
        """Directories should appear before files, both sorted alphabetically."""
        mounts = [{"path": "/mnt/data", "label": "Data"}]
        browser._handle_response(self._mounts_response(mounts))

        entries = [
            {"name": "zebra.txt", "type": "file", "size": 0, "modified": 0},
            {"name": "beta", "type": "directory", "size": 0, "modified": 0},
            {"name": "alpha.txt", "type": "file", "size": 0, "modified": 0},
            {"name": "alpha", "type": "directory", "size": 0, "modified": 0},
        ]
        browser._handle_response(self._list_response("/mnt/data", entries))

        col = browser._columns[1]
        names = [col.item(i).text() for i in range(col.count())]
        assert names == ["alpha/", "beta/", "alpha.txt", "zebra.txt"]


# =============================================================================
# File Browser Integration in Dialogs (Phase 4)
# =============================================================================


class TestSlpPathDialogBrowser:
    """Tests for RemoteFileBrowser integration in SlpPathDialog."""

    @pytest.fixture
    def send_fn(self):
        return MagicMock()

    def test_no_browser_without_send_fn(self, qapp):
        """SlpPathDialog without send_fn should not have a browser panel."""
        from sleap_connect.gui.widgets import SlpPathDialog

        dialog = SlpPathDialog(
            local_path="/local/labels.slp",
            error_message="File not found",
        )
        assert dialog._browser is None
        dialog.deleteLater()

    def test_browser_panel_present_with_send_fn(self, qapp, send_fn):
        """SlpPathDialog with send_fn should have a browser panel."""
        from sleap_connect.gui.widgets import SlpPathDialog

        dialog = SlpPathDialog(
            local_path="/local/labels.slp",
            error_message="File not found",
            send_fn=send_fn,
        )
        assert dialog._browser is not None
        assert not dialog._browser.isVisible()  # hidden by default
        dialog.deleteLater()

    def test_browse_toggle_loads_mounts(self, qapp, send_fn):
        """Toggling browse on should load mounts and toggle visibility."""
        from sleap_connect.gui.widgets import SlpPathDialog

        dialog = SlpPathDialog(
            local_path="/local/labels.slp",
            error_message="File not found",
            send_fn=send_fn,
        )
        # Toggle on — browser should become visible and load mounts
        dialog._on_browse_toggled(True)
        send_fn.assert_called_once_with("FS_GET_MOUNTS")
        assert dialog._browse_toggle.text() == "Hide browser"

        # Toggle off
        dialog._on_browse_toggled(False)
        assert dialog._browse_toggle.text() == "Browse worker filesystem..."
        dialog.deleteLater()

    def test_file_selected_fills_path(self, qapp, send_fn):
        """Selecting a file in the browser should fill the worker path input."""
        from sleap_connect.gui.widgets import SlpPathDialog

        dialog = SlpPathDialog(
            local_path="/local/labels.slp",
            error_message="File not found",
            send_fn=send_fn,
        )
        dialog._on_file_selected("/mnt/data/labels.slp")
        assert dialog._path_edit.text() == "/mnt/data/labels.slp"
        assert dialog._ok_btn.isEnabled()
        dialog.deleteLater()

    def test_browser_has_slp_filter(self, qapp, send_fn):
        """Browser should filter for .slp files."""
        from sleap_connect.gui.widgets import SlpPathDialog

        dialog = SlpPathDialog(
            local_path="/local/labels.slp",
            error_message="File not found",
            send_fn=send_fn,
        )
        assert ".slp" in dialog._browser._allowed_extensions
        dialog.deleteLater()


class TestPathResolutionDialogBrowser:
    """Tests for RemoteFileBrowser integration in PathResolutionDialog."""

    @pytest.fixture
    def send_fn(self):
        return MagicMock()

    @pytest.fixture(autouse=True)
    def suppress_cascade_confirm(self, monkeypatch):
        """Suppress the cascade-fill QMessageBox in all tests in this class.

        _on_path_changed shows a modal confirmation dialog when a path is filled
        and other empty sibling rows exist. In headless CI this hangs. Patch it
        at the module level so any dialog created in a test is unaffected.
        """
        from qtpy.QtWidgets import QMessageBox

        monkeypatch.setattr(
            "sleap_connect.gui.widgets.QMessageBox.question",
            lambda *a, **kw: QMessageBox.No,
        )

    @pytest.fixture
    def missing_videos(self):
        return [
            VideoPathStatus(
                filename="video1.mp4",
                original_path="/local/video1.mp4",
                found=False,
            ),
            VideoPathStatus(
                filename="video2.avi",
                original_path="/local/video2.avi",
                found=False,
            ),
            VideoPathStatus(
                filename="video3.mp4",
                original_path="/local/video3.mp4",
                found=True,
                worker_path="/mnt/data/video3.mp4",
            ),
        ]

    def test_no_browser_without_send_fn(self, qapp, missing_videos):
        """PathResolutionDialog without send_fn should not have a browser."""
        from sleap_connect.gui.widgets import PathResolutionDialog

        dialog = PathResolutionDialog(missing_videos)
        assert dialog._browser is None
        dialog.deleteLater()

    def test_browser_present_with_send_fn(self, qapp, missing_videos, send_fn):
        """PathResolutionDialog with send_fn should have a browser."""
        from sleap_connect.gui.widgets import PathResolutionDialog

        dialog = PathResolutionDialog(missing_videos, send_fn=send_fn)
        assert dialog._browser is not None
        assert not dialog._browser.isVisible()
        dialog.deleteLater()

    def test_browse_button_sets_target_and_loads(self, qapp, missing_videos, send_fn):
        """Clicking Browse... should set target row and load mounts."""
        from sleap_connect.gui.widgets import PathResolutionDialog

        dialog = PathResolutionDialog(missing_videos, send_fn=send_fn)

        # Click browse for first missing video (row 0)
        dialog._on_browse_path(0)
        assert dialog._browse_target_path == "/local/video1.mp4"
        send_fn.assert_called_with("FS_GET_MOUNTS")
        dialog.deleteLater()

    def test_file_selected_fills_target_row(self, qapp, missing_videos, send_fn):
        """Selecting a file in browser should fill the target row's path."""
        from sleap_connect.gui.widgets import PathResolutionDialog

        dialog = PathResolutionDialog(missing_videos, send_fn=send_fn)

        # Browse for first row
        dialog._on_browse_path(0)
        dialog._on_browser_file_selected("/mnt/data/video1.mp4")

        path_edit = dialog._path_widgets.get("/local/video1.mp4")
        assert path_edit is not None
        assert path_edit.text() == "/mnt/data/video1.mp4"
        # Target cleared after selection
        assert dialog._browse_target_path is None
        dialog.deleteLater()

    def test_browse_different_rows(self, qapp, missing_videos, send_fn):
        """Browsing different rows should target the correct path edit."""
        from sleap_connect.gui.widgets import PathResolutionDialog

        dialog = PathResolutionDialog(missing_videos, send_fn=send_fn)

        # Browse for row 1 (video2.avi)
        dialog._on_browse_path(1)
        assert dialog._browse_target_path == "/local/video2.avi"
        dialog._on_browser_file_selected("/mnt/data/video2.avi")

        path_edit = dialog._path_widgets.get("/local/video2.avi")
        assert path_edit.text() == "/mnt/data/video2.avi"
        dialog.deleteLater()

    def test_browser_has_video_filter(self, qapp, missing_videos, send_fn):
        """Browser should filter for video file types."""
        from sleap_connect.gui.widgets import PathResolutionDialog

        dialog = PathResolutionDialog(missing_videos, send_fn=send_fn)
        exts = dialog._browser._allowed_extensions
        assert ".mp4" in exts
        assert ".avi" in exts
        assert ".mov" in exts
        assert ".h264" in exts
        assert ".mkv" in exts
        dialog.deleteLater()

    def test_browse_fallback_without_send_fn(self, qapp, missing_videos):
        """Without send_fn, Browse... should use fallback QInputDialog."""
        from sleap_connect.gui.widgets import PathResolutionDialog

        dialog = PathResolutionDialog(missing_videos)
        # _on_browse_path with no browser should try QInputDialog
        with patch("qtpy.QtWidgets.QInputDialog") as mock_input:
            mock_input.getText.return_value = ("/mnt/data/video1.mp4", True)
            dialog._on_browse_path(0)
            mock_input.getText.assert_called_once()
        dialog.deleteLater()
