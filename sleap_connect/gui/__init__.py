"""GUI widgets for sleap-connect integration with SLEAP.

This package provides Qt widgets for integrating remote training and inference
functionality into the SLEAP GUI. The widgets are designed to be embedded in
SLEAP's Training Configuration dialog.

Example usage (in SLEAP):
    from sleap_connect.gui.widgets import RemoteTrainingWidget

    # Add to dialog layout
    remote_widget = RemoteTrainingWidget()
    layout.addWidget(remote_widget)

    # Check if remote training is enabled
    if remote_widget.is_enabled():
        room_id = remote_widget.get_selected_room_id()
        worker_id = remote_widget.get_selected_worker_id()
"""

from sleap_connect.gui.widgets import (
    RemoteTrainingWidget,
    WorkerSetupDialog,
    RoomBrowserDialog,
    PathResolutionDialog,
    ConfigValidationDialog,
    TrainingFailureDialog,
)
from sleap_connect.gui.runners import (
    RemoteProgressBridge,
    run_remote_training,
    format_progress_line,
)
from sleap_connect.gui.presubmission import (
    PresubmissionResult,
    PresubmissionFlow,
    run_presubmission_checks,
    check_authentication,
    check_config_validation,
    check_video_paths,
)

__all__ = [
    # Widgets
    "RemoteTrainingWidget",
    "WorkerSetupDialog",
    "RoomBrowserDialog",
    "PathResolutionDialog",
    "ConfigValidationDialog",
    "TrainingFailureDialog",
    # Runners
    "RemoteProgressBridge",
    "run_remote_training",
    "format_progress_line",
    # Pre-submission
    "PresubmissionResult",
    "PresubmissionFlow",
    "run_presubmission_checks",
    "check_authentication",
    "check_config_validation",
    "check_video_paths",
]
