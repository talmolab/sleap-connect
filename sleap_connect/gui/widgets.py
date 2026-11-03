"""Qt widgets for sleap-connect remote training integration.

This module provides the RemoteTrainingWidget for embedding in SLEAP's
Training Configuration dialog.
"""

from __future__ import annotations

import queue as _queue_module
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt, Signal, QThread
from qtpy.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QCheckBox,
    QComboBox,
    QPushButton,
    QRadioButton,
    QButtonGroup,
    QLabel,
    QWidget,
    QFrame,
    QDialog,
    QDialogButtonBox,
    QTextEdit,
    QPlainTextEdit,
    QApplication,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QLineEdit,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QSplitter,
    QSizePolicy,
    QProgressBar,
    QMessageBox,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from sleap_connect.api import Room, Worker, User


def _is_mock_mode() -> bool:
    """Check if mock mode is enabled via environment variable."""
    import os

    return os.environ.get("SLEAP_RTC_MOCK", "").lower() in ("1", "true", "yes")


def _get_mock_rooms():
    """Get mock rooms for testing."""
    from sleap_connect.api import Room

    return [
        Room(
            id="room-1",
            name="Lab GPU Server",
            owner="mockuser",
            created_at="2024-01-15",
        ),
        Room(
            id="room-2",
            name="Cloud Training",
            owner="mockuser",
            created_at="2024-02-01",
        ),
    ]


def _get_mock_workers(room_id: str):
    """Get mock workers for testing."""
    from sleap_connect.api import Worker

    if room_id == "room-1":
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
    elif room_id == "room-2":
        return [
            Worker(
                id="worker-3",
                name="Cluster Node 1",
                status="available",
                gpu_name="NVIDIA V100",
                gpu_memory_mb=32768,
            ),
        ]
    return []


def _filter_active_rooms(rooms: list) -> list:
    """Filter rooms to only include active (non-expired) ones.

    Args:
        rooms: List of Room objects.

    Returns:
        List of Room objects that have not expired.
    """
    import time

    current_time = int(time.time())
    active_rooms = []
    for room in rooms:
        # Room is active if expires_at is None (never expires) or in the future
        if room.expires_at is None or room.expires_at > current_time:
            active_rooms.append(room)
    return active_rooms


def _has_room_secret(room_id: str) -> bool:
    """Check if a room has a local secret stored.

    Args:
        room_id: The room ID to check.

    Returns:
        True if the room has a locally stored secret.
    """
    from sleap_connect.auth.credentials import get_room_secret

    return get_room_secret(room_id) is not None


class WorkerDiscoveryThread(QThread):
    """Background thread for discovering workers in a room."""

    workers_loaded = Signal(list)  # List[Worker]
    error = Signal(str)

    def __init__(self, room_id: str, parent=None):
        super().__init__(parent)
        self.room_id = room_id

    def run(self):
        """Fetch workers from the room."""
        try:
            if _is_mock_mode():
                import time

                time.sleep(0.3)  # Simulate network delay
                self.workers_loaded.emit(_get_mock_workers(self.room_id))
                return

            from sleap_connect.api import list_workers

            workers = list_workers(self.room_id)
            self.workers_loaded.emit(workers)
        except Exception as e:
            self.error.emit(str(e))


class RoomLoadThread(QThread):
    """Background thread for loading available rooms."""

    rooms_loaded = Signal(list)  # List[Room]
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        """Fetch available rooms."""
        try:
            if _is_mock_mode():
                import time

                time.sleep(0.3)  # Simulate network delay
                self.rooms_loaded.emit(_get_mock_rooms())
                return

            from sleap_connect.api import list_rooms

            rooms = list_rooms()
            self.rooms_loaded.emit(rooms)
        except Exception as e:
            self.error.emit(str(e))


class LoginThread(QThread):
    """Background thread for login flow."""

    login_success = Signal(object)  # User
    login_failed = Signal(str)
    url_ready = Signal(str)

    def __init__(self, timeout: int = 120, parent=None):
        super().__init__(parent)
        self.timeout = timeout

    def run(self):
        """Run the login flow."""
        try:
            # Check for mock mode (for development/testing)
            if _is_mock_mode():
                from sleap_connect.api import User
                import time

                time.sleep(0.5)  # Simulate brief delay
                mock_user = User(
                    id="mock-user-123",
                    username="mockuser",
                    avatar_url=None,
                )
                self.login_success.emit(mock_user)
                return

            import webbrowser
            from sleap_connect.api import login

            def on_url_ready(url: str):
                # Emit signal for UI update AND open browser
                self.url_ready.emit(url)
                webbrowser.open(url)

            user = login(timeout=self.timeout, on_url_ready=on_url_ready)
            self.login_success.emit(user)
        except Exception as e:
            self.login_failed.emit(str(e))


class WorkerSetupDialog(QDialog):
    """Dialog to help users set up their first worker.

    This dialog is shown when a room has 0 workers available. It provides
    step-by-step instructions for setting up a worker on a GPU machine.
    """

    def __init__(
        self,
        room_name: str = "",
        room_id: str = "",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("No Workers Available")
        self.setMinimumWidth(600)
        self._room_name = room_name
        self._room_id = room_id
        self._setup_ui()

    def _setup_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Header explanation
        header_text = (
            "This room has no workers connected. You need a worker running on "
            "a GPU machine to train remotely."
        )
        if self._room_name:
            header_text = (
                f'The room "{self._room_name}" has no workers connected. You need '
                "a worker running on a GPU machine to train remotely."
            )

        header_label = QLabel(header_text)
        header_label.setWordWrap(True)
        layout.addWidget(header_label)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        # Quick Setup section
        setup_label = QLabel("<b>Quick Setup (on your GPU machine):</b>")
        layout.addWidget(setup_label)

        # Step 1: Install sleap-connect
        step1_layout = QVBoxLayout()
        step1_label = QLabel("1. Install sleap-connect (one time):")
        step1_layout.addWidget(step1_label)

        install_cmd = QLabel(
            "   <code>uv tool install --python 3.11 sleap-connect"
            " --with &quot;sleap-nn[torch]&quot;"
            " --with-executables-from sleap-nn"
            " --torch-backend auto</code>"
        )
        install_cmd.setTextFormat(Qt.RichText)
        install_cmd.setStyleSheet("background-color: #f0f0f0; padding: 4px;")
        install_cmd.setTextInteractionFlags(Qt.TextSelectableByMouse)
        install_cmd.setWordWrap(True)
        step1_layout.addWidget(install_cmd)
        layout.addLayout(step1_layout)

        # Step 2: Set up an account key
        step2_layout = QVBoxLayout()
        step2_label = QLabel(
            "2. Set up an account key (create in the Dashboard "
            "<b>Account Keys</b> tab, or via CLI):"
        )
        step2_label.setWordWrap(True)
        step2_layout.addWidget(step2_label)

        key_new_cmd = QLabel(
            "   <code>sleap-connect login</code> then "
            "<code>sleap-connect key create --save</code>"
            "&nbsp;&nbsp;<span style='color:#666;'>(new users)</span>"
        )
        key_new_cmd.setTextFormat(Qt.RichText)
        key_new_cmd.setStyleSheet("background-color: #f0f0f0; padding: 4px;")
        key_new_cmd.setTextInteractionFlags(Qt.TextSelectableByMouse)
        key_new_cmd.setWordWrap(True)
        step2_layout.addWidget(key_new_cmd)

        key_existing_cmd = QLabel(
            "   <code>sleap-connect key use slp_acct_xxx...</code>"
            "&nbsp;&nbsp;<span style='color:#666;'>"
            "(if you already have a key)</span>"
        )
        key_existing_cmd.setTextFormat(Qt.RichText)
        key_existing_cmd.setStyleSheet("background-color: #f0f0f0; padding: 4px;")
        key_existing_cmd.setTextInteractionFlags(Qt.TextSelectableByMouse)
        key_existing_cmd.setWordWrap(True)
        step2_layout.addWidget(key_existing_cmd)

        mount_desc = QLabel(
            "   Register the directories where your training data is stored:"
        )
        mount_desc.setWordWrap(True)
        step2_layout.addWidget(mount_desc)

        mount_cmd = QLabel(
            "   <code>sleap-connect config add-mount /path/to/your/data/"
            " &quot;Your Mount Name&quot; --global</code>"
        )
        mount_cmd.setTextFormat(Qt.RichText)
        mount_cmd.setStyleSheet("background-color: #f0f0f0; padding: 4px;")
        mount_cmd.setTextInteractionFlags(Qt.TextSelectableByMouse)
        mount_cmd.setWordWrap(True)
        step2_layout.addWidget(mount_cmd)
        layout.addLayout(step2_layout)

        # Step 3: Start the worker
        step3_layout = QVBoxLayout()
        step3_label = QLabel("3. Start the worker:")
        step3_layout.addWidget(step3_label)

        # Build worker command, substituting room_id if available
        room_arg = self._room_id if self._room_id else "ROOM_ID"
        worker_cmd_str = (
            f"sleap-connect worker --account-key slp_acct_xxx..."
            f' --room {room_arg} --name "My GPU Server"'
        )
        self._worker_cmd = QLabel(f"   <code>{worker_cmd_str}</code>")
        self._worker_cmd.setTextFormat(Qt.RichText)
        self._worker_cmd.setStyleSheet("background-color: #f0f0f0; padding: 4px;")
        self._worker_cmd.setWordWrap(True)
        self._worker_cmd.setTextInteractionFlags(Qt.TextSelectableByMouse)
        step3_layout.addWidget(self._worker_cmd)

        layout.addLayout(step3_layout)

        layout.addStretch()

        # Separator before buttons
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        separator2.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator2)

        # Button row
        button_layout = QHBoxLayout()

        self._copy_commands_button = QPushButton("Copy Commands")
        self._copy_commands_button.clicked.connect(self._on_copy_commands)
        button_layout.addWidget(self._copy_commands_button)

        self._open_docs_button = QPushButton("Open Documentation")
        self._open_docs_button.clicked.connect(self._on_open_documentation)
        button_layout.addWidget(self._open_docs_button)

        button_layout.addStretch()

        self._close_button = QPushButton("Close")
        self._close_button.clicked.connect(self.accept)
        button_layout.addWidget(self._close_button)

        layout.addLayout(button_layout)

    def _on_copy_commands(self):
        """Copy setup commands to clipboard."""
        room_arg = self._room_id if self._room_id else "ROOM_ID"
        worker_cmd = (
            f"sleap-connect worker --account-key slp_acct_xxx..."
            f' --room {room_arg} --name "My GPU Server"'
        )

        commands = f"""# Step 1: Install sleap-connect (one time)
uv tool install --python 3.11 sleap-connect --with "sleap-nn[torch]" --with-executables-from sleap-nn --torch-backend auto

# Step 2: Set up an account key
# Option A — new users:
sleap-connect login
sleap-connect key create --save
# Option B — if you already have a key:
# sleap-connect key use slp_acct_xxx...

# Register directories where your training data is stored
sleap-connect config add-mount /path/to/your/data/ "Your Mount Name" --global

# Step 3: Start the worker
{worker_cmd}
"""
        clipboard = QApplication.clipboard()
        clipboard.setText(commands)

        # Briefly show feedback
        original_text = self._copy_commands_button.text()
        self._copy_commands_button.setText("Copied!")
        self._copy_commands_button.setEnabled(False)

        # Reset button after delay using QTimer
        from qtpy.QtCore import QTimer

        QTimer.singleShot(
            1500,
            lambda: (
                self._copy_commands_button.setText(original_text),
                self._copy_commands_button.setEnabled(True),
            ),
        )

    def _on_open_dashboard(self):
        """Open the sleap-connect dashboard in a browser."""
        import webbrowser

        from sleap_connect.auth.github import get_dashboard_url

        url = get_dashboard_url()
        webbrowser.open(url)

    def _on_open_documentation(self):
        """Open the sleap-connect documentation in a browser."""
        import webbrowser

        # Link to the worker setup documentation
        docs_url = "https://github.com/talmolab/sleap-connect#worker-setup"
        webbrowser.open(docs_url)


class RoomSecretSetupDialog(QDialog):
    """Dialog to set up a room secret for P2P communication.

    This dialog is shown when a room doesn't have a local secret stored.
    It allows users to generate a new secret or enter an existing one.
    """

    secret_saved = Signal(str, str)  # room_id, secret

    def __init__(
        self,
        room_id: str,
        room_name: str = "",
        is_owner: bool = True,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Room Setup Required")
        self.setMinimumWidth(500)
        self._room_id = room_id
        self._room_name = room_name
        self._is_owner = is_owner
        self._setup_ui()

    def _setup_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Header explanation
        room_display = (
            f'"{self._room_name}"' if self._room_name else f"ID: {self._room_id}"
        )
        header_text = (
            f"Room {room_display} needs a secret for secure P2P communication. "
            "This secret must be shared with any workers that will connect to this room."
        )
        header_label = QLabel(header_text)
        header_label.setWordWrap(True)
        layout.addWidget(header_label)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        if self._is_owner:
            # Owner can generate a new secret
            generate_label = QLabel("<b>Option 1: Generate a new secret</b>")
            layout.addWidget(generate_label)

            generate_desc = QLabel(
                "Generate a new secret and share it with your workers. "
                "Use this if you're setting up this room for the first time."
            )
            generate_desc.setWordWrap(True)
            layout.addWidget(generate_desc)

            self._generate_button = QPushButton("Generate New Secret")
            self._generate_button.clicked.connect(self._on_generate_secret)
            layout.addWidget(self._generate_button)

            layout.addSpacing(10)

        # Enter existing secret
        enter_label = QLabel(
            "<b>Option 2: Enter existing secret</b>"
            if self._is_owner
            else "<b>Enter room secret</b>"
        )
        layout.addWidget(enter_label)

        if self._is_owner:
            enter_desc = QLabel(
                "If workers are already using a secret for this room, enter it here."
            )
        else:
            enter_desc = QLabel("Ask the room owner for the secret and enter it here.")
        enter_desc.setWordWrap(True)
        layout.addWidget(enter_desc)

        secret_layout = QHBoxLayout()
        self._secret_input = QLineEdit()
        self._secret_input.setPlaceholderText("Paste room secret here...")
        secret_layout.addWidget(self._secret_input)

        self._save_button = QPushButton("Save")
        self._save_button.clicked.connect(self._on_save_secret)
        self._save_button.setEnabled(False)
        secret_layout.addWidget(self._save_button)
        layout.addLayout(secret_layout)

        self._secret_input.textChanged.connect(
            lambda text: self._save_button.setEnabled(len(text.strip()) > 0)
        )

        # Status label
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        layout.addStretch()

        # Cancel button
        button_box = QDialogButtonBox(QDialogButtonBox.Cancel)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _on_generate_secret(self):
        """Generate a new room secret."""
        from sleap_connect.auth.psk import generate_secret
        from sleap_connect.auth.credentials import save_room_secret

        secret = generate_secret()
        save_room_secret(self._room_id, secret)

        self._status_label.setText(
            f"<span style='color: green;'>✓ Secret generated and saved!</span><br><br>"
            f"<b>Important:</b> Share this secret with your workers:<br>"
            f"<code style='background: #f0f0f0; padding: 4px;'>{secret}</code><br><br>"
            f"Workers can set this secret with:<br>"
            f"<code>sleap-connect secret set {self._room_id} {secret}</code>"
        )
        self._status_label.setTextFormat(Qt.RichText)

        # Copy to clipboard
        QApplication.clipboard().setText(secret)
        self._generate_button.setText("Generated! (copied to clipboard)")
        self._generate_button.setEnabled(False)

        self.secret_saved.emit(self._room_id, secret)

    def _on_save_secret(self):
        """Save the entered secret."""
        from sleap_connect.auth.credentials import save_room_secret

        secret = self._secret_input.text().strip()
        if not secret:
            return

        save_room_secret(self._room_id, secret)

        self._status_label.setText(
            "<span style='color: green;'>✓ Secret saved successfully!</span>"
        )
        self._status_label.setTextFormat(Qt.RichText)

        self._secret_input.setEnabled(False)
        self._save_button.setEnabled(False)

        self.secret_saved.emit(self._room_id, secret)


class RoomBrowserDialog(QDialog):
    """Dialog for browsing and selecting rooms.

    This dialog provides a table view of available rooms with details
    including name, creation date, and worker count.
    """

    room_selected = Signal(str, str)  # room_id, room_name

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Browse Rooms")
        self.setMinimumSize(600, 400)
        self._rooms: list = []
        self._selected_room_id: str | None = None
        self._selected_room_name: str | None = None
        self._load_thread: RoomLoadThread | None = None
        self._setup_ui()
        self._load_rooms()

    def _setup_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Header with refresh button
        header_layout = QHBoxLayout()
        header_label = QLabel("Select a room to use for remote training:")
        header_layout.addWidget(header_label, 1)

        self._refresh_button = QPushButton("Refresh")
        self._refresh_button.clicked.connect(self._load_rooms)
        header_layout.addWidget(self._refresh_button)

        layout.addLayout(header_layout)

        # Room table
        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["Name", "Role", "Created", "Workers"])
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.itemDoubleClicked.connect(self._on_double_click)

        # Set column stretch
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # Name stretches
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Role
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # Created
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # Workers

        layout.addWidget(self._table)

        # Status label
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: gray;")
        layout.addWidget(self._status_label)

        # Button row
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self._select_button = QPushButton("Select")
        self._select_button.setEnabled(False)
        self._select_button.clicked.connect(self._on_select)
        button_layout.addWidget(self._select_button)

        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self._cancel_button)

        layout.addLayout(button_layout)

    def _load_rooms(self):
        """Load rooms from the server."""
        if self._load_thread and self._load_thread.isRunning():
            return

        self._refresh_button.setEnabled(False)
        self._refresh_button.setText("Loading...")
        self._table.setRowCount(0)
        self._status_label.setText("Loading rooms...")
        self._status_label.setStyleSheet("color: gray;")

        self._load_thread = RoomLoadThread(self)
        self._load_thread.rooms_loaded.connect(self._on_rooms_loaded)
        self._load_thread.error.connect(self._on_rooms_error)
        self._load_thread.start()

    def _on_rooms_loaded(self, rooms: list):
        """Handle rooms loaded from server."""
        # Filter to only show active (non-expired) rooms
        rooms = _filter_active_rooms(rooms)
        self._rooms = rooms
        self._refresh_button.setEnabled(True)
        self._refresh_button.setText("Refresh")

        self._table.setRowCount(len(rooms))

        for row, room in enumerate(rooms):
            # Name
            name_item = QTableWidgetItem(room.name)
            name_item.setData(Qt.UserRole, room.id)
            self._table.setItem(row, 0, name_item)

            # Role
            role_item = QTableWidgetItem(room.role)
            self._table.setItem(row, 1, role_item)

            # Created date
            created_str = ""
            if room.joined_at:
                try:
                    from datetime import datetime

                    dt = datetime.fromisoformat(room.joined_at.replace("Z", "+00:00"))
                    created_str = dt.strftime("%Y-%m-%d")
                except Exception:
                    created_str = room.joined_at[:10] if room.joined_at else ""
            created_item = QTableWidgetItem(created_str)
            self._table.setItem(row, 2, created_item)

            # Worker count (we don't have this in Room, so show "-")
            workers_item = QTableWidgetItem("-")
            workers_item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, 3, workers_item)

        if rooms:
            self._status_label.setText(
                f"{len(rooms)} room{'s' if len(rooms) != 1 else ''} available"
            )
            self._status_label.setStyleSheet("color: green;")
        else:
            self._status_label.setText("No rooms available")
            self._status_label.setStyleSheet("color: orange;")

    def _on_rooms_error(self, error: str):
        """Handle room loading error."""
        self._refresh_button.setEnabled(True)
        self._refresh_button.setText("Refresh")
        self._status_label.setText(f"Error: {error}")
        self._status_label.setStyleSheet("color: red;")

    def _on_selection_changed(self):
        """Handle table selection change."""
        selected_items = self._table.selectedItems()
        if selected_items:
            row = selected_items[0].row()
            name_item = self._table.item(row, 0)
            self._selected_room_id = name_item.data(Qt.UserRole)
            self._selected_room_name = name_item.text()
            self._select_button.setEnabled(True)
        else:
            self._selected_room_id = None
            self._selected_room_name = None
            self._select_button.setEnabled(False)

    def _on_double_click(self, item: QTableWidgetItem):
        """Handle double-click on table row."""
        self._on_selection_changed()
        if self._selected_room_id:
            self._on_select()

    def _on_select(self):
        """Handle select button click."""
        if self._selected_room_id:
            self.room_selected.emit(
                self._selected_room_id, self._selected_room_name or ""
            )
            self.accept()

    def get_selected_room(self) -> tuple[str | None, str | None]:
        """Get the selected room ID and name.

        Returns:
            Tuple of (room_id, room_name), or (None, None) if no selection.
        """
        return self._selected_room_id, self._selected_room_name


def show_save_mapping_prompt(
    local_dir: str,
    worker_dir: str,
    config,
    parent_widget=None,
) -> None:
    """Show a Save / Skip prompt to persist a directory prefix mapping.

    Silently returns without showing the dialog when the mapping already
    exists in *config*.

    Args:
        local_dir: Local directory prefix (e.g. ``/Users/alice/data``).
        worker_dir: Corresponding worker directory prefix.
        config: ``Config`` instance used to check/save mappings.
        parent_widget: Optional Qt parent for the dialog.
    """
    for m in config.get_path_mappings():
        if m.local == local_dir and m.worker == worker_dir:
            return

    from qtpy.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout, QLineEdit
    from qtpy.QtGui import QFont

    dlg = QDialog(parent_widget)
    dlg.setWindowTitle("Save path mapping?")
    dlg.setMinimumWidth(500)
    layout = QVBoxLayout(dlg)
    layout.setSpacing(6)

    layout.addWidget(QLabel("Save path mapping for future use?"))

    mono = QFont("Courier")
    mono.setStyleHint(QFont.StyleHint.Monospace)

    def _code_block(text: str) -> QLineEdit:
        box = QLineEdit(text)
        box.setReadOnly(True)
        box.setFont(mono)
        return box

    layout.addWidget(_code_block(local_dir))

    arrow = QLabel("↓")
    arrow.setAlignment(Qt.AlignCenter)
    layout.addWidget(arrow)

    layout.addWidget(_code_block(worker_dir))

    buttons = QDialogButtonBox()
    buttons.addButton("Save", QDialogButtonBox.ButtonRole.AcceptRole)
    buttons.addButton("Skip", QDialogButtonBox.ButtonRole.RejectRole)
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    layout.addWidget(buttons)

    if dlg.exec() == QDialog.DialogCode.Accepted:
        config.save_path_mapping(local_dir, worker_dir)


class _ExportThread(QThread):
    """Background thread that exports a Labels object to a temp .pkg.slp file.

    Calls ``convert_fn(output_path)`` on a worker thread so the Qt main thread
    stays responsive during what can be a slow embed-and-write operation.

    Signals:
        complete: Emitted with the path of the written temp file on success.
        error: Emitted with a human-readable reason on failure.
    """

    complete = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        convert_fn: "Callable[[str], None]",
        temp_path: str,
        parent=None,
    ):
        super().__init__(parent)
        self._convert_fn = convert_fn
        self._temp_path = temp_path

    def run(self):
        try:
            self._convert_fn(self._temp_path)
            self.complete.emit(self._temp_path)
        except Exception as exc:
            self.error.emit(str(exc))


class _UploadThread(QThread):
    """Background thread that executes the client-to-worker upload protocol.

    Sends FILE_UPLOAD_CHECK / FILE_UPLOAD_START / binary chunks /
    FILE_UPLOAD_END via ``send_fn`` and reads worker responses from
    ``response_queue`` (a ``queue.Queue`` fed by the main thread via
    ``SlpPathDialog.on_upload_response()``).

    Signals:
        progress: Emitted with (bytes_sent, total_bytes) for each
            FILE_UPLOAD_PROGRESS message from the worker.
        complete: Emitted with the worker-side absolute path on success
            (FILE_UPLOAD_COMPLETE or FILE_UPLOAD_CACHE_HIT).
        error: Emitted with a human-readable reason on failure.
    """

    progress = Signal(int, int)
    complete = Signal(str)
    error = Signal(str)

    _CHUNK_SIZE = 64 * 1024
    _RESPONSE_TIMEOUT = 30.0
    _END_TIMEOUT = 120.0  # wait longer for large-file writes on the worker

    def __init__(
        self,
        send_fn: "Callable",
        local_path: str,
        dest_dir: str,
        create_subdir: str,
        response_queue: "_queue_module.Queue",
        parent=None,
    ):
        super().__init__(parent)
        self._send_fn = send_fn
        self._local_path = local_path
        self._dest_dir = dest_dir
        self._create_subdir = create_subdir
        self._response_queue = response_queue

    def _wait_for(self, timeout: float | None = None) -> str:
        t = timeout if timeout is not None else self._RESPONSE_TIMEOUT
        try:
            return self._response_queue.get(timeout=t)
        except _queue_module.Empty:
            raise RuntimeError(f"Timed out waiting for worker response ({t:.0f} s)")

    def run(self):
        import hashlib
        from pathlib import Path
        from sleap_connect.protocol import (
            MSG_FILE_UPLOAD_CACHE_HIT,
            MSG_FILE_UPLOAD_CHECK,
            MSG_FILE_UPLOAD_COMPLETE,
            MSG_FILE_UPLOAD_END,
            MSG_FILE_UPLOAD_ERROR,
            MSG_FILE_UPLOAD_PROGRESS,
            MSG_FILE_UPLOAD_READY,
            MSG_FILE_UPLOAD_START,
            MSG_SEPARATOR,
        )

        try:
            path = Path(self._local_path)
            filename = path.name
            total_bytes = path.stat().st_size

            # Step 1: SHA-256 (streaming)
            sha256_ctx = hashlib.sha256()
            with open(self._local_path, "rb") as fh:
                while chunk := fh.read(self._CHUNK_SIZE):
                    sha256_ctx.update(chunk)
            sha256 = sha256_ctx.hexdigest()

            # Step 2: FILE_UPLOAD_CHECK
            self._send_fn(
                f"{MSG_FILE_UPLOAD_CHECK}{MSG_SEPARATOR}{sha256}{MSG_SEPARATOR}{filename}"
            )
            resp = self._wait_for()

            if resp.startswith(MSG_FILE_UPLOAD_CACHE_HIT + MSG_SEPARATOR):
                self.complete.emit(resp.split(MSG_SEPARATOR, 1)[1])
                return

            if resp.startswith(MSG_FILE_UPLOAD_ERROR + MSG_SEPARATOR):
                self.error.emit(resp.split(MSG_SEPARATOR, 1)[1])
                return

            if resp != MSG_FILE_UPLOAD_READY:
                self.error.emit(f"Unexpected response to CHECK: {resp[:80]}")
                return

            # Step 3: FILE_UPLOAD_START
            self._send_fn(
                f"{MSG_FILE_UPLOAD_START}{MSG_SEPARATOR}{filename}{MSG_SEPARATOR}"
                f"{total_bytes}{MSG_SEPARATOR}{self._dest_dir}{MSG_SEPARATOR}"
                f"{self._create_subdir}"
            )
            resp = self._wait_for()

            if resp.startswith(MSG_FILE_UPLOAD_ERROR + MSG_SEPARATOR):
                self.error.emit(resp.split(MSG_SEPARATOR, 1)[1])
                return

            if resp != MSG_FILE_UPLOAD_READY:
                self.error.emit(f"Unexpected response to START: {resp[:80]}")
                return

            # Step 4: Send binary chunks
            with open(self._local_path, "rb") as fh:
                while chunk := fh.read(self._CHUNK_SIZE):
                    self._send_fn(chunk)

            # Step 5: FILE_UPLOAD_END; drain progress, await COMPLETE/ERROR
            self._send_fn(MSG_FILE_UPLOAD_END)

            while True:
                resp = self._wait_for(timeout=self._END_TIMEOUT)

                if resp.startswith(MSG_FILE_UPLOAD_PROGRESS + MSG_SEPARATOR):
                    parts = resp.split(MSG_SEPARATOR)
                    if len(parts) >= 3:
                        try:
                            self.progress.emit(int(parts[1]), int(parts[2]))
                        except ValueError:
                            pass
                    continue

                if resp.startswith(MSG_FILE_UPLOAD_COMPLETE + MSG_SEPARATOR):
                    self.complete.emit(resp.split(MSG_SEPARATOR, 1)[1])
                    return

                if resp.startswith(MSG_FILE_UPLOAD_ERROR + MSG_SEPARATOR):
                    self.error.emit(resp.split(MSG_SEPARATOR, 1)[1])
                    return

        except Exception as exc:
            self.error.emit(str(exc))


class UploadDestDialog(QDialog):
    """Modal dialog for picking an upload destination directory on the worker.

    Shows a ``RemoteFileBrowser`` for navigating the worker filesystem.
    Navigating into any directory sets it as the destination. A checkbox
    lets the user opt in to a ``sleap-connect-downloads/`` subfolder.

    Args:
        send_fn: Callable that sends FS_* messages to the worker.
        parent: Optional parent widget.
    """

    def __init__(
        self,
        send_fn: "Callable[[str], None]",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Choose Upload Destination")
        self.setMinimumSize(720, 520)
        self._dest_dir: str | None = None
        self._browser: RemoteFileBrowser | None = None
        self._setup_ui(send_fn)

    def _setup_ui(self, send_fn):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        info = QLabel(
            "Navigate to the directory where the file should be uploaded.\n"
            "Click any folder to select it as the destination."
        )
        info.setWordWrap(True)
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info)

        self._browser = RemoteFileBrowser(
            send_fn=send_fn, file_filter="", directory_mode=True
        )
        self._browser.directory_entered.connect(self._on_directory_entered)
        self._browser.setMinimumHeight(300)
        layout.addWidget(self._browser)

        # Selected destination path
        layout.addWidget(QLabel("Destination:"))
        self._dir_label = QLineEdit()
        self._dir_label.setReadOnly(True)
        self._dir_label.setPlaceholderText("No directory selected")
        self._dir_label.setStyleSheet("color: gray;")
        layout.addWidget(self._dir_label)

        # Subfolder option
        self._subdir_check = QCheckBox(
            "Create `sleap_connect_downloads` subfolder inside the selected directory"
        )
        self._subdir_check.setToolTip(
            "When checked, a 'sleap_connect_downloads' folder will be created inside "
            "the selected directory and the file will be saved there."
        )
        self._subdir_check.toggled.connect(lambda _: self._update_dest_label())
        layout.addWidget(self._subdir_check)

        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        self._ok_btn = QPushButton("Use This Directory")
        self._ok_btn.setEnabled(False)
        self._ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self._ok_btn)

        layout.addLayout(btn_layout)

    _SUBDIR_NAME = "sleap_connect_downloads"

    def _on_directory_entered(self, path: str):
        self._dest_dir = path
        self._update_dest_label()
        self._ok_btn.setEnabled(True)

    def _update_dest_label(self, _=None):
        if self._dest_dir is None:
            return
        display = self._dest_dir
        if self._subdir_check.isChecked():
            sep = "" if self._dest_dir.endswith("/") else "/"
            display = f"{self._dest_dir}{sep}{self._SUBDIR_NAME}"
        self._dir_label.setText(display)
        self._dir_label.setStyleSheet("color: green; font-weight: bold;")
        self._dir_label.setCursorPosition(len(display))

    def get_dest_dir(self) -> str | None:
        """Return the selected destination directory path."""
        return self._dest_dir

    def get_create_subdir(self) -> bool:
        """Return whether the user wants a sleap-connect-downloads/ subfolder."""
        return self._subdir_check.isChecked()


class SlpPathDialog(QDialog):
    """Dialog for resolving the SLP file path on the worker.

    Shown when the local SLP path doesn't exist on the remote worker,
    prompting the user to provide the correct worker-side path.

    For ``.pkg.slp`` files an additional "Upload file to worker..." button
    is shown, letting the user transfer the file directly over the data
    channel.

    Args:
        local_path: The local SLP path that was rejected by the worker.
        error_message: Error message from the worker explaining the rejection.
        send_fn: Optional callable to send messages to the worker.
            When provided, a remote file browser panel and upload button
            are shown.
        on_browser_changed: Optional callback invoked with the currently
            active ``RemoteFileBrowser`` (or ``None``) whenever the active
            browser changes (e.g. when the upload dest dialog opens/closes).
            Used by ``presubmission.py`` to keep FS_* response routing
            pointed at the right widget.
        parent: Optional parent widget.
    """

    # Thread-safe signal for routing FILE_UPLOAD_* messages from asyncio thread
    _upload_response_signal = Signal(str)

    def __init__(
        self,
        local_path: str,
        error_message: str,
        send_fn: "Callable[[str], None] | None" = None,
        on_browser_changed: "Callable[[RemoteFileBrowser | None], None] | None" = None,
        convert_fn: "Callable[[str], None] | None" = None,
        save_fn: "Callable[[str], None] | None" = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("SLP File Path Resolution")
        self.setMinimumSize(700, 500)
        self._worker_path: str | None = None
        self._send_fn = send_fn
        self._convert_fn = convert_fn
        self._save_fn = save_fn
        self._browser: RemoteFileBrowser | None = None
        self._error_message = error_message
        self._local_path = local_path
        self._on_browser_changed = on_browser_changed

        # Upload state
        self._upload_queue: _queue_module.Queue = _queue_module.Queue()
        self._upload_thread: _UploadThread | None = None
        self._export_thread: _ExportThread | None = None
        self._upload_btn: QPushButton | None = None
        self._export_btn: QPushButton | None = None
        self._progress_bar: QProgressBar | None = None
        self._temp_pkg_path: str | None = None
        # Saved from dest dialog for use after async export completes
        self._pending_dest_dir: str | None = None
        self._pending_create_subdir: str = "0"

        self._upload_response_signal.connect(
            self._on_upload_response_received,
            Qt.ConnectionType.QueuedConnection,
        )

        self._setup_ui(local_path, error_message)

        # Auto-fill worker path from saved path mappings (if any match)
        from sleap_connect.config import get_config

        translated = get_config().translate_path(local_path)
        if translated:
            self._path_edit.setText(translated)
            self._error_label.setText(
                "<b>Note:</b> Using saved path mapping. You can edit below if needed."
            )
            self._error_label.setStyleSheet("color: #27ae60;")

    def _setup_ui(self, local_path: str, error_message: str):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Explanation
        info_label = QLabel(
            "The SLP file could not be found on the worker at the path "
            "used locally. Please provide the correct path on the worker."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Error details
        self._error_label = QLabel(f"<b>Error:</b> {error_message}")
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("color: #c0392b;")
        layout.addWidget(self._error_label)

        # Local path (read-only)
        form = QFormLayout()
        local_edit = QLineEdit(local_path)
        local_edit.setReadOnly(True)
        local_edit.setStyleSheet("color: gray;")
        local_edit.setMinimumWidth(400)
        form.addRow("Local path:", local_edit)

        # Worker path input
        worker_layout = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setMinimumWidth(400)
        self._path_edit.setPlaceholderText("e.g. /root/vast/data/labels.v002.slp")
        self._path_edit.textChanged.connect(self._on_text_changed)
        worker_layout.addWidget(self._path_edit)
        form.addRow("Worker path:", worker_layout)

        layout.addLayout(form)

        # ------------------------------------------------------------------
        # Save-to-folder section (shown above upload/browse when callables
        # are provided so the user can copy to a shared filesystem first)
        # ------------------------------------------------------------------
        self._save_status_label: QLabel | None = None
        _any_save_btn = self._save_fn is not None or self._convert_fn is not None
        if _any_save_btn:
            save_row = QHBoxLayout()
            save_row.addWidget(QLabel("Save to mounted filesystem:"))

            if self._save_fn is not None:
                self._save_slp_btn = QPushButton("Save as .slp")
                self._save_slp_btn.clicked.connect(self._on_save_slp_clicked)
                save_row.addWidget(self._save_slp_btn)

            if self._convert_fn is not None:
                self._save_pkg_slp_btn = QPushButton("Save as .pkg.slp")
                self._save_pkg_slp_btn.clicked.connect(self._on_save_pkg_slp_clicked)
                save_row.addWidget(self._save_pkg_slp_btn)

            save_row.addStretch()

            layout.addLayout(save_row)

            self._save_status_label = QLabel("")
            self._save_status_label.setWordWrap(True)
            self._save_status_label.setAlignment(Qt.AlignCenter)
            self._save_status_label.setVisible(False)
            layout.addWidget(self._save_status_label)

        # Upload button — only for .pkg.slp with an active connection
        is_pkg_slp = local_path.lower().endswith(".pkg.slp")
        is_plain_slp = not is_pkg_slp and local_path.lower().endswith(".slp")
        if is_pkg_slp and self._send_fn is not None:
            self._upload_btn = QPushButton("Upload file to worker...")
            self._upload_btn.setToolTip(
                "Transfer this .pkg.slp file directly to the worker over the "
                "data channel."
            )
            self._upload_btn.clicked.connect(self._on_upload_clicked)
            layout.addWidget(self._upload_btn)

        # Export+upload button — for plain .slp when convert_fn is provided
        if is_plain_slp and self._convert_fn is not None and self._send_fn is not None:
            self._export_btn = QPushButton("Export as .pkg.slp and upload to worker...")
            self._export_btn.setToolTip(
                "Embed video frames locally and transfer the packaged file "
                "directly to the worker over the data channel. "
                "No shared filesystem required."
            )
            self._export_btn.clicked.connect(self._on_export_upload_clicked)
            layout.addWidget(self._export_btn)

        # Shared progress bar (used by both upload and export→upload flows)
        if (is_pkg_slp and self._send_fn is not None) or (
            is_plain_slp and self._convert_fn is not None and self._send_fn is not None
        ):
            self._progress_bar = QProgressBar()
            self._progress_bar.setVisible(False)
            self._progress_bar.setMinimum(0)
            self._progress_bar.setMaximum(100)
            layout.addWidget(self._progress_bar)

        # Collapsible file browser panel (only when send_fn is provided)
        if self._send_fn is not None:
            self._browse_toggle = QPushButton("Browse worker filesystem...")
            self._browse_toggle.setCheckable(True)
            self._browse_toggle.toggled.connect(self._on_browse_toggled)
            layout.addWidget(self._browse_toggle)

            self._browser = RemoteFileBrowser(
                send_fn=self._send_fn,
                file_filter="*.slp",
            )
            self._browser.file_selected.connect(self._on_file_selected)
            self._browser.setVisible(False)
            self._browser.setMinimumHeight(300)
            layout.addWidget(self._browser)

        # Separator + Buttons (visually distinct from browser's Select bar)
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        self._ok_btn = QPushButton("Continue")
        self._ok_btn.setEnabled(False)
        self._ok_btn.clicked.connect(self._on_accept)
        button_layout.addWidget(self._ok_btn)

        layout.addLayout(button_layout)

    def _on_browse_toggled(self, checked: bool):
        """Toggle the file browser panel visibility."""
        if self._browser is not None:
            self._browser.setVisible(checked)
            if checked:
                self._browser.load_mounts()
                self._browse_toggle.setText("Hide browser")
            else:
                self._browse_toggle.setText("Browse worker filesystem...")
            self.adjustSize()

    def _on_file_selected(self, path: str):
        """Handle file selection from the browser."""
        self._path_edit.setText(path)
        self._error_label.setText("<b>Status:</b> Path found")
        self._error_label.setStyleSheet("color: green;")

    def _on_text_changed(self, text: str):
        """Enable Continue button only when a path is entered."""
        self._ok_btn.setEnabled(bool(text.strip()))
        if not text.strip():
            self._error_label.setText(f"<b>Error:</b> {self._error_message}")
            self._error_label.setStyleSheet("color: #c0392b;")
        else:
            # Hide the error once the user has provided a path
            self._error_label.setText("")
            self._error_label.setStyleSheet("")

    def _on_accept(self):
        """Accept the dialog with the entered path."""
        self._worker_path = self._path_edit.text().strip()
        self._prompt_save_mapping(self._local_path, self._worker_path)
        self.accept()

    def _prompt_save_mapping(self, local_path: str, worker_path: str) -> None:
        """Offer to save a directory prefix mapping after the user hits Continue.

        Shows a small dialog only when the local and worker directories differ
        and no existing mapping already covers this exact prefix pair.
        """
        from pathlib import Path
        from sleap_connect.config import get_config

        local_dir = str(Path(local_path).parent)
        worker_dir = str(Path(worker_path).parent)
        if local_dir == worker_dir:
            return
        show_save_mapping_prompt(local_dir, worker_dir, get_config(), self)

    def get_worker_path(self) -> str | None:
        """Get the worker-side SLP path entered by the user."""
        return self._worker_path

    # ------------------------------------------------------------------
    # Save-to-folder handlers
    # ------------------------------------------------------------------

    def _on_save_slp_clicked(self) -> None:
        """Save the labels as a plain .slp to a user-chosen folder."""
        from pathlib import Path
        from qtpy.QtWidgets import QFileDialog

        chosen_dir = QFileDialog.getExistingDirectory(self, "Choose save folder", "")
        if not chosen_dir:
            return

        stem = Path(self._local_path).stem
        output_path = str(Path(chosen_dir) / f"{stem}.slp")

        if hasattr(self, "_save_slp_btn"):
            self._save_slp_btn.setEnabled(False)
        try:
            self._save_fn(output_path)
            self._show_save_status(
                f"Saved to {output_path}. "
                "Now use Browse worker filesystem to locate this file on the worker.",
                error=False,
            )
        except Exception as exc:
            self._show_save_status(f"Save error: {exc}", error=True)
            if hasattr(self, "_save_slp_btn"):
                self._save_slp_btn.setEnabled(True)

    def _on_save_pkg_slp_clicked(self) -> None:
        """Export labels as a .pkg.slp (embedded frames) to a user-chosen folder."""
        from pathlib import Path
        from qtpy.QtWidgets import QFileDialog

        chosen_dir = QFileDialog.getExistingDirectory(self, "Choose save folder", "")
        if not chosen_dir:
            return

        stem = Path(self._local_path).stem
        output_path = str(Path(chosen_dir) / f"{stem}.pkg.slp")

        if hasattr(self, "_save_pkg_slp_btn"):
            self._save_pkg_slp_btn.setEnabled(False)
        try:
            self._convert_fn(output_path)
            self._show_save_status(
                f"Saved to {output_path}. "
                "Now use Browse worker filesystem to locate this file on the worker.",
                error=False,
            )
        except Exception as exc:
            self._show_save_status(f"Save error: {exc}", error=True)
            if hasattr(self, "_save_pkg_slp_btn"):
                self._save_pkg_slp_btn.setEnabled(True)

    def _show_save_status(self, message: str, error: bool = False) -> None:
        """Update the save status label below the save buttons."""
        if self._save_status_label is None:
            return
        self._save_status_label.setText(message)
        color = "#c0392b" if error else "#27ae60"
        self._save_status_label.setStyleSheet(f"color: {color};")
        self._save_status_label.setVisible(True)

    # ------------------------------------------------------------------
    # Upload — thread-safe response routing
    # ------------------------------------------------------------------

    def on_upload_response(self, message: str) -> None:
        """Route a FILE_UPLOAD_* response from any thread.

        May be called from the asyncio/WebRTC thread.  Emits a Qt signal
        so the actual handling runs on the Qt main thread.
        """
        self._upload_response_signal.emit(message)

    def _on_upload_response_received(self, message: str) -> None:
        """Qt slot: feed the response into the upload thread's queue."""
        self._upload_queue.put(message)

    # ------------------------------------------------------------------
    # Upload — UI actions
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Export → Upload flow  (plain .slp → temp .pkg.slp → worker)
    # ------------------------------------------------------------------

    def _on_export_upload_clicked(self) -> None:
        """Open destination picker, confirm, then export and upload."""
        dest_dialog = UploadDestDialog(send_fn=self._send_fn, parent=self)

        if self._on_browser_changed is not None:
            self._on_browser_changed(dest_dialog._browser)

        dest_dialog._browser.load_mounts()
        accepted = dest_dialog.exec()

        if self._on_browser_changed is not None:
            self._on_browser_changed(self._browser)

        if not accepted or dest_dialog.get_dest_dir() is None:
            return

        dest_dir = dest_dialog.get_dest_dir()
        create_subdir = dest_dialog.get_create_subdir()

        # Use an approximate size estimate for the confirm dialog.
        # The actual .pkg.slp may differ but this gives a reasonable bound.
        if not self._confirm_upload(dest_dir, create_subdir):
            return

        self._pending_dest_dir = dest_dir
        self._pending_create_subdir = "1" if create_subdir else "0"

        # Write the temp file into an isolated tmpdir so cleanup is atomic
        import tempfile
        from pathlib import Path

        tmp_dir = tempfile.mkdtemp(prefix="sleap_connect_export_")
        filename = Path(self._local_path).stem + ".pkg.slp"
        self._temp_pkg_path = str(Path(tmp_dir) / filename)

        # Disable button; show indeterminate progress while exporting
        if self._export_btn is not None:
            self._export_btn.setEnabled(False)
        if self._progress_bar is not None:
            self._progress_bar.setMinimum(0)
            self._progress_bar.setMaximum(0)  # indeterminate
            self._progress_bar.setVisible(True)

        self._error_label.setText("<b>Status:</b> Exporting .pkg.slp…")
        self._error_label.setStyleSheet("color: #2980b9;")

        self._export_thread = _ExportThread(
            convert_fn=self._convert_fn,
            temp_path=self._temp_pkg_path,
            parent=self,
        )
        self._export_thread.complete.connect(self._on_export_complete)
        self._export_thread.error.connect(self._on_export_error)
        self._export_thread.start()

    def _on_export_complete(self, temp_path: str) -> None:
        """Export finished — switch to determinate progress and start upload."""
        self._error_label.setText("<b>Status:</b> Uploading…")
        self._start_upload(
            dest_dir=self._pending_dest_dir,
            create_subdir=self._pending_create_subdir,
            path_override=temp_path,
        )

    def _on_export_error(self, reason: str) -> None:
        """Export failed — show error and re-enable the export button."""
        if self._progress_bar is not None:
            self._progress_bar.setMaximum(100)
            self._progress_bar.setVisible(False)

        self._error_label.setText(f"<b>Export error:</b> {reason}")
        self._error_label.setStyleSheet("color: #c0392b;")

        if self._export_btn is not None:
            self._export_btn.setEnabled(True)

        self._cleanup_temp_pkg()

    # ------------------------------------------------------------------
    # Upload — UI actions
    # ------------------------------------------------------------------

    def _on_upload_clicked(self) -> None:
        """Open the destination-picker and, if confirmed, start the upload."""
        dest_dialog = UploadDestDialog(send_fn=self._send_fn, parent=self)

        # Redirect FS_* routing to the dest dialog's browser while it's open
        if self._on_browser_changed is not None:
            self._on_browser_changed(dest_dialog._browser)

        dest_dialog._browser.load_mounts()
        accepted = dest_dialog.exec()

        # Restore FS_* routing to the main browser (or None)
        if self._on_browser_changed is not None:
            self._on_browser_changed(self._browser)

        if not accepted or dest_dialog.get_dest_dir() is None:
            return

        dest_dir = dest_dialog.get_dest_dir()
        create_subdir = dest_dialog.get_create_subdir()

        # Show tiered time-estimate warning before starting
        if not self._confirm_upload(dest_dir, create_subdir):
            return

        self._start_upload(dest_dir, "1" if create_subdir else "0")

    def _confirm_upload(self, dest_dir: str, create_subdir: bool) -> bool:
        """Compute estimated transfer time, show a tiered warning, and ask
        the user to confirm or cancel before the upload starts.

        Returns:
            True if the user confirmed, False if they cancelled.
        """
        from pathlib import Path

        try:
            file_size = Path(self._local_path).stat().st_size
        except OSError:
            file_size = 0

        # Estimate at 10 Mbps (1.25 MB/s)
        est_sec = file_size / (10_000_000 / 8) if file_size else 0
        est_min = max(1, round(est_sec / 60))

        mb = file_size / (1024 * 1024)
        size_str = f"{mb:.0f} MB" if mb < 1024 else f"{mb / 1024:.1f} GB"

        if file_size < 500 * 1024 * 1024:
            suffix = ""
        elif file_size < 2 * 1024 * 1024 * 1024:
            suffix = " — this may take a while"
        else:
            suffix = " — consider using a shared filesystem if available"

        time_str = f"~{est_min} min{suffix}"

        dest_display = dest_dir
        if create_subdir:
            dest_display = dest_dir.rstrip("/") + "/sleap_connect_downloads/"

        from qtpy.QtWidgets import (
            QDialog,
            QDialogButtonBox,
            QLabel,
            QTextEdit,
            QVBoxLayout,
        )
        from qtpy.QtGui import QFont

        dlg = QDialog(self)
        dlg.setWindowTitle("Confirm Upload")
        dlg.setMinimumWidth(480)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(8)

        header = QLabel(
            f"Upload <b>{Path(self._local_path).name}</b> ({size_str}) to worker?"
        )
        header.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(header)

        dest_label = QLabel("<b>Destination:</b>")
        dest_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(dest_label)

        path_box = QTextEdit()
        path_box.setReadOnly(True)
        path_box.setPlainText(dest_display)
        mono = QFont("Courier")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        path_box.setFont(mono)
        path_box.setFixedHeight(60)
        layout.addWidget(path_box)

        time_label = QLabel(f"<b>Estimated time:</b> {time_str}")
        time_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(time_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.rejected.connect(dlg.reject)
        buttons.accepted.connect(dlg.accept)
        layout.addWidget(buttons)

        return dlg.exec() == QDialog.DialogCode.Accepted

    def _start_upload(
        self,
        dest_dir: str,
        create_subdir: str,
        path_override: "str | None" = None,
    ) -> None:
        """Create and start the upload thread.

        Args:
            dest_dir: Worker-side destination directory.
            create_subdir: ``"1"`` to create a ``sleap_connect_downloads``
                subdirectory inside *dest_dir*, ``"0"`` otherwise.
            path_override: If provided, upload this local path instead of
                ``self._local_path``.  Used by the export flow to upload the
                freshly-written temp ``.pkg.slp``.
        """
        local_path = path_override if path_override is not None else self._local_path

        # Disable buttons and show progress bar
        if self._upload_btn is not None:
            self._upload_btn.setEnabled(False)
        if self._progress_bar is not None:
            self._progress_bar.setMinimum(0)
            self._progress_bar.setMaximum(100)
            self._progress_bar.setValue(0)
            self._progress_bar.setVisible(True)

        self._error_label.setText("<b>Status:</b> Uploading…")
        self._error_label.setStyleSheet("color: #2980b9;")

        # Drain any stale messages from a previous attempt
        while not self._upload_queue.empty():
            try:
                self._upload_queue.get_nowait()
            except _queue_module.Empty:
                break

        self._upload_thread = _UploadThread(
            send_fn=self._send_fn,
            local_path=local_path,
            dest_dir=dest_dir,
            create_subdir=create_subdir,
            response_queue=self._upload_queue,
            parent=self,
        )
        self._upload_thread.progress.connect(self._on_upload_progress)
        self._upload_thread.complete.connect(self._on_upload_complete)
        self._upload_thread.error.connect(self._on_upload_error)
        self._upload_thread.start()

    def _on_upload_progress(self, bytes_sent: int, total: int) -> None:
        """Update the progress bar."""
        if self._progress_bar is not None and total > 0:
            pct = min(100, int(bytes_sent * 100 / total))
            self._progress_bar.setValue(pct)

    def _on_upload_complete(self, worker_path: str) -> None:
        """Auto-fill the worker path field and enable Continue."""
        if self._progress_bar is not None:
            self._progress_bar.setValue(100)
            self._progress_bar.setVisible(False)

        self._path_edit.setText(worker_path)
        self._error_label.setText("<b>Status:</b> Upload complete")
        self._error_label.setStyleSheet("color: green;")

        # Upload button stays disabled — path is filled, Continue is enabled
        if self._upload_btn is not None:
            self._upload_btn.setEnabled(False)

        self._cleanup_temp_pkg()

    def _on_upload_error(self, reason: str) -> None:
        """Show error message and re-enable the upload/export button for retry."""
        if self._progress_bar is not None:
            self._progress_bar.setVisible(False)

        self._error_label.setText(f"<b>Upload error:</b> {reason}")
        self._error_label.setStyleSheet("color: #c0392b;")

        if self._upload_btn is not None:
            self._upload_btn.setEnabled(True)
        if self._export_btn is not None:
            self._export_btn.setEnabled(True)

        self._cleanup_temp_pkg()

    def _cleanup_temp_pkg(self) -> None:
        """Delete the temp .pkg.slp and its parent directory if they exist."""
        import shutil
        from pathlib import Path

        if self._temp_pkg_path is not None:
            try:
                tmp = Path(self._temp_pkg_path)
                if tmp.exists():
                    shutil.rmtree(tmp.parent, ignore_errors=True)
            except Exception:
                pass
            finally:
                self._temp_pkg_path = None

    def closeEvent(self, event) -> None:
        """Clean up any in-progress temp file when the dialog is closed."""
        self._cleanup_temp_pkg()
        super().closeEvent(event)


class PathResolutionDialog(QDialog):
    """Dialog for resolving video paths that differ between client and worker.

    This dialog is shown when video files in the SLP cannot be found on the
    worker machine, allowing the user to specify correct paths.

    Args:
        path_results: List of VideoPathStatus objects from path checking.
        send_fn: Optional callable to send FS_* messages to the worker.
            When provided, a shared remote file browser panel is shown
            allowing the user to browse the worker's filesystem.
        parent: Optional parent widget.
    """

    paths_resolved = Signal(dict)  # {original_path: resolved_path}

    def __init__(
        self,
        path_results: list,  # List of VideoPathStatus
        send_fn: "Callable[[str], None] | None" = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Video Path Resolution Required")
        self.setMinimumSize(700, 700)
        self._path_results = path_results
        self._resolved_paths: dict[str, str] = {}
        self._path_widgets: dict[str, QLineEdit] = {}
        self._send_fn = send_fn
        self._browser: RemoteFileBrowser | None = None
        self._browse_target_path: str | None = (
            None  # original_path of row being browsed
        )
        self._cascading = False  # guard against recursive cascade fills
        # Injectable confirm callable for cascade fill — override in tests to
        # avoid modal dialogs in headless environments.
        self._cascade_confirm = QMessageBox.question
        self._setup_ui()

    def _setup_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Header explanation
        header_label = QLabel(
            "Some video files cannot be found on the worker. "
            "Please provide the correct paths."
        )
        header_label.setWordWrap(True)
        layout.addWidget(header_label)

        # Path table
        self._table = QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Video", "Status", "Worker Path"])
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.verticalHeader().setVisible(False)

        # Set column stretch
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)

        # Set initial column widths
        self._table.setColumnWidth(0, 200)

        self._populate_table()
        layout.addWidget(self._table)

        # Auto-detect button
        auto_layout = QHBoxLayout()
        self._auto_detect_button = QPushButton("Auto-detect in folder...")
        self._auto_detect_button.clicked.connect(self._on_auto_detect)
        auto_layout.addWidget(self._auto_detect_button)
        auto_layout.addStretch()
        layout.addLayout(auto_layout)

        # Shared file browser panel (only when send_fn is provided)
        if self._send_fn is not None:
            _VIDEO_FILTER = "*.mp4,*.avi,*.mov,*.h264,*.mkv"
            self._browser = RemoteFileBrowser(
                send_fn=self._send_fn,
                file_filter=_VIDEO_FILTER,
            )
            self._browser.file_selected.connect(self._on_browser_file_selected)
            self._browser.setVisible(False)
            self._browser.setFixedHeight(300)
            layout.addWidget(self._browser)

        # Status label — add top margin to separate from the browser panel
        self._status_label = QLabel("")
        self._status_label.setContentsMargins(0, 8, 0, 0)
        layout.addWidget(self._status_label)

        # Button row
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self._cancel_button)

        self._continue_button = QPushButton("Continue with Resolved")
        self._continue_button.clicked.connect(self._on_continue)
        button_layout.addWidget(self._continue_button)

        layout.addLayout(button_layout)

        # Update status after all widgets are created
        self._update_status()

    def _populate_table(self):
        """Populate the table with path results."""
        self._path_edit_to_row: dict[QLineEdit, int] = {}
        self._table.setRowCount(len(self._path_results))

        for row, video in enumerate(self._path_results):
            # Video filename (read-only)
            name_item = QTableWidgetItem(video.filename)
            name_item.setToolTip(video.original_path)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, 0, name_item)

            # Status (read-only)
            if video.found:
                status_item = QTableWidgetItem("✓ Found")
                status_item.setForeground(Qt.darkGreen)
                status_item.setFlags(status_item.flags() & ~Qt.ItemIsEditable)
                self._table.setItem(row, 1, status_item)

                # Worker path (read-only for found videos)
                path_item = QTableWidgetItem(video.worker_path or "")
                path_item.setToolTip(video.worker_path or "")
                path_item.setFlags(path_item.flags() & ~Qt.ItemIsEditable)
                self._table.setItem(row, 2, path_item)
            else:
                status_item = QTableWidgetItem("✗ Missing")
                status_item.setForeground(Qt.red)
                status_item.setFlags(status_item.flags() & ~Qt.ItemIsEditable)
                self._table.setItem(row, 1, status_item)

                # Create editable path widget for missing videos
                path_widget = QWidget()
                path_layout = QHBoxLayout(path_widget)
                path_layout.setContentsMargins(2, 2, 2, 2)
                path_layout.setSpacing(4)

                path_edit = QLineEdit()
                path_edit.setPlaceholderText("Enter worker path...")

                # Priority: saved mapping > worker suggestion > empty
                from sleap_connect.config import get_config as _get_cfg

                prefill = _get_cfg().translate_path(video.original_path)
                if prefill:
                    path_edit.setText(prefill)
                elif video.suggestions:
                    path_edit.setText(video.suggestions[0])

                if path_edit.text().strip():
                    status_item.setText("✓ Resolved")
                    status_item.setForeground(Qt.darkGreen)

                path_edit.textChanged.connect(self._on_path_changed)
                self._path_edit_to_row[path_edit] = row
                path_layout.addWidget(path_edit)

                browse_button = QPushButton("Browse...")
                browse_button.setMaximumWidth(70)
                browse_button.clicked.connect(
                    lambda checked, r=row: self._on_browse_path(r)
                )
                path_layout.addWidget(browse_button)

                self._table.setCellWidget(row, 2, path_widget)
                self._path_widgets[video.original_path] = path_edit

        # Adjust row heights
        self._table.resizeRowsToContents()

    def _on_path_changed(self):
        """Handle path text changes, update row status, and cascade-fill siblings."""
        edit = self.sender()
        if edit is not None:
            row = self._path_edit_to_row.get(edit)
            if row is not None:
                has_path = bool(edit.text().strip())
                status_item = self._table.item(row, 1)
                if status_item is not None:
                    if has_path:
                        status_item.setText("✓ Resolved")
                        status_item.setForeground(Qt.darkGreen)
                    else:
                        status_item.setText("✗ Missing")
                        status_item.setForeground(Qt.red)

            # Cascade: when one path is entered, offer to fill other empty
            # paths using the same worker directory (mirrors SLEAP's missing-
            # files dialog which confirms before applying prefix changes).
            if not self._cascading:
                path = edit.text().strip()
                if path:
                    from pathlib import Path

                    worker_dir = str(Path(path).parent)

                    # Collect other missing videos with no path yet
                    candidates = [
                        (video, other_edit)
                        for video in self._path_results
                        if not video.found
                        for other_edit in [self._path_widgets.get(video.original_path)]
                        if other_edit is not None
                        and other_edit is not edit
                        and not other_edit.text().strip()
                    ]

                    if candidates:
                        names = "\n".join(
                            f"  \u2022 {v.filename}" for v, _ in candidates
                        )
                        response = self._cascade_confirm(
                            self,
                            "Auto-fill other paths?",
                            f"Other missing videos can be resolved using the "
                            f"same folder:\n\n  {worker_dir}\n\n"
                            f"Apply to:\n{names}",
                            QMessageBox.Yes | QMessageBox.No,
                            QMessageBox.Yes,
                        )
                        if response == QMessageBox.Yes:
                            self._cascading = True
                            try:
                                for video, other_edit in candidates:
                                    other_edit.setText(f"{worker_dir}/{video.filename}")
                            finally:
                                self._cascading = False

        self._update_status()

    def _on_browse_path(self, row: int):
        """Handle browse button click for a specific row."""
        video = self._path_results[row]

        if self._browser is not None:
            # Use the shared remote file browser
            self._browse_target_path = video.original_path
            self._browser.setVisible(True)
            if not self._browser._columns:
                self._browser.load_mounts()
        else:
            # Fallback: simple text input dialog
            from qtpy.QtWidgets import QInputDialog

            path, ok = QInputDialog.getText(
                self,
                "Enter Worker Path",
                f"Enter the path to '{video.filename}' on the worker:",
                QLineEdit.Normal,
                self._path_widgets.get(video.original_path, QLineEdit()).text(),
            )

            if ok and path:
                path_edit = self._path_widgets.get(video.original_path)
                if path_edit:
                    path_edit.setText(path)

    def _on_browser_file_selected(self, path: str):
        """Handle file selection from the shared browser."""
        if self._browse_target_path is not None:
            path_edit = self._path_widgets.get(self._browse_target_path)
            if path_edit:
                path_edit.setText(path)
            self._browse_target_path = None

    def _on_auto_detect(self):
        """Handle auto-detect button click."""
        # Ask for a folder path
        from qtpy.QtWidgets import QInputDialog

        folder, ok = QInputDialog.getText(
            self,
            "Auto-detect Videos",
            "Enter a folder path on the worker to search for videos:",
            QLineEdit.Normal,
            "/data/videos",
        )

        if ok and folder:
            folder = folder.rstrip("/")
            # Auto-fill paths based on folder + filename
            for video in self._path_results:
                if not video.found:
                    path_edit = self._path_widgets.get(video.original_path)
                    if path_edit:
                        suggested_path = f"{folder}/{video.filename}"
                        path_edit.setText(suggested_path)

            self._update_status()

    def _update_status(self):
        """Update the status label based on resolved paths."""
        total_missing = sum(1 for v in self._path_results if not v.found)
        resolved_count = 0

        for video in self._path_results:
            if not video.found:
                path_edit = self._path_widgets.get(video.original_path)
                if path_edit and path_edit.text().strip():
                    resolved_count += 1

        if resolved_count == total_missing:
            self._status_label.setText(
                f"All {total_missing} missing path{'s' if total_missing != 1 else ''} resolved"
            )
            self._status_label.setStyleSheet("color: green;")
            self._continue_button.setEnabled(True)
        else:
            remaining = total_missing - resolved_count
            self._status_label.setText(
                f"{remaining} of {total_missing} missing path{'s' if total_missing != 1 else ''} still need resolution"
            )
            self._status_label.setStyleSheet("color: orange;")
            self._continue_button.setEnabled(False)

    def _on_continue(self):
        """Handle continue button click."""
        # Collect resolved paths
        resolved = {}
        for video in self._path_results:
            if video.found:
                resolved[video.original_path] = video.worker_path
            else:
                path_edit = self._path_widgets.get(video.original_path)
                if path_edit:
                    resolved[video.original_path] = path_edit.text().strip()

        self._resolved_paths = resolved
        self.paths_resolved.emit(resolved)
        self.accept()

    def get_resolved_paths(self) -> dict[str, str]:
        """Get the resolved path mappings.

        Returns:
            Dictionary mapping original paths to resolved worker paths.
        """
        return self._resolved_paths


class ConfigValidationDialog(QDialog):
    """Dialog for displaying configuration validation errors and warnings.

    This dialog is shown when config validation fails before job submission.
    Errors block submission, warnings allow proceeding.
    """

    def __init__(
        self,
        validation_result,  # ValidationResult
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._validation_result = validation_result
        self._has_errors = len(validation_result.errors) > 0
        self._setup_window_title()
        self.setMinimumWidth(550)
        self._setup_ui()

    def _setup_window_title(self):
        """Set window title based on validation result."""
        if self._has_errors:
            self.setWindowTitle("Configuration Validation Failed")
        else:
            self.setWindowTitle("Configuration Warnings")

    def _setup_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Header message
        if self._has_errors:
            header_text = "The training configuration has errors that must be fixed:"
        else:
            header_text = "The training configuration has warnings:"
        header_label = QLabel(header_text)
        header_label.setWordWrap(True)
        layout.addWidget(header_label)

        # Issues list
        issues_widget = QWidget()
        issues_layout = QVBoxLayout(issues_widget)
        issues_layout.setContentsMargins(10, 5, 10, 5)
        issues_layout.setSpacing(8)

        # Add errors first
        for error in self._validation_result.errors:
            issue_label = QLabel(f"✗ <b>{error.field}</b>: {error.message}")
            issue_label.setTextFormat(Qt.RichText)
            issue_label.setStyleSheet("color: #c0392b;")  # Dark red
            issue_label.setWordWrap(True)
            issues_layout.addWidget(issue_label)

        # Add warnings
        for warning in self._validation_result.warnings:
            issue_label = QLabel(f"⚠ <b>{warning.field}</b>: {warning.message}")
            issue_label.setTextFormat(Qt.RichText)
            issue_label.setStyleSheet("color: #d68910;")  # Dark orange/yellow
            issue_label.setWordWrap(True)
            issues_layout.addWidget(issue_label)

        issues_layout.addStretch()
        layout.addWidget(issues_widget, 1)

        # Tip section (only for errors)
        if self._has_errors:
            tip_label = QLabel(
                "<i>Tip: Review the config in the Model Configuration tabs.</i>"
            )
            tip_label.setTextFormat(Qt.RichText)
            tip_label.setStyleSheet("color: gray;")
            layout.addWidget(tip_label)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        # Button row
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        if self._has_errors:
            # Only OK button for errors
            ok_button = QPushButton("OK")
            ok_button.clicked.connect(self.reject)
            button_layout.addWidget(ok_button)
        else:
            # Cancel and Continue buttons for warnings only
            cancel_button = QPushButton("Cancel")
            cancel_button.clicked.connect(self.reject)
            button_layout.addWidget(cancel_button)

            continue_button = QPushButton("Continue Anyway")
            continue_button.clicked.connect(self.accept)
            button_layout.addWidget(continue_button)

        layout.addLayout(button_layout)

    def has_errors(self) -> bool:
        """Check if the validation result has errors.

        Returns:
            True if there are blocking errors.
        """
        return self._has_errors


class TrainingFailureDialog(QDialog):
    """Dialog for displaying training failure details and recovery options.

    This dialog is shown when remote training fails, providing error details
    and instructions for resuming from checkpoints.
    """

    def __init__(
        self,
        error_message: str,
        epoch: int | None = None,
        checkpoint_path: str | None = None,
        room_id: str | None = None,
        config_path: str | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Remote Training Failed")
        self.setMinimumWidth(550)
        self._error_message = error_message
        self._epoch = epoch
        self._checkpoint_path = checkpoint_path
        self._room_id = room_id
        self._config_path = config_path
        self._resume_command = self._build_resume_command()
        self._setup_ui()

    def _build_resume_command(self) -> str | None:
        """Build the CLI command to resume training."""
        if not self._checkpoint_path:
            return None

        parts = ["sleap-connect train"]
        if self._room_id:
            parts.append(f"--room {self._room_id}")
        if self._config_path:
            parts.append(f"--config {self._config_path}")
        parts.append(f"--resume-ckpt {self._checkpoint_path}")

        return " \\\n    ".join(parts)

    def _setup_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Error icon and message header
        header_layout = QHBoxLayout()
        error_icon = QLabel("❌")
        error_icon.setStyleSheet("font-size: 24px;")
        header_layout.addWidget(error_icon)

        header_text = "Training failed"
        if self._epoch is not None:
            header_text += f" at epoch {self._epoch}"
        header_label = QLabel(f"<b>{header_text}</b>")
        header_label.setTextFormat(Qt.RichText)
        header_label.setStyleSheet("font-size: 14px;")
        header_layout.addWidget(header_label, 1)
        layout.addLayout(header_layout)

        # Error message
        error_box = QFrame()
        error_box.setFrameShape(QFrame.StyledPanel)
        error_box.setStyleSheet(
            "background-color: #fdecea; border: 1px solid #f5c6cb; border-radius: 4px;"
        )
        error_layout = QVBoxLayout(error_box)
        error_layout.setContentsMargins(10, 10, 10, 10)

        error_label = QLabel(self._error_message)
        error_label.setWordWrap(True)
        error_label.setStyleSheet("color: #721c24;")
        error_layout.addWidget(error_label)

        layout.addWidget(error_box)

        # Checkpoint information (if available)
        if self._checkpoint_path:
            checkpoint_section = QWidget()
            checkpoint_layout = QVBoxLayout(checkpoint_section)
            checkpoint_layout.setContentsMargins(0, 10, 0, 0)
            checkpoint_layout.setSpacing(8)

            checkpoint_header = QLabel("<b>Checkpoints saved on worker:</b>")
            checkpoint_header.setTextFormat(Qt.RichText)
            checkpoint_layout.addWidget(checkpoint_header)

            checkpoint_path_label = QLabel(f"  {self._checkpoint_path}")
            checkpoint_path_label.setStyleSheet(
                "font-family: monospace; background-color: #f5f5f5; "
                "padding: 4px 8px; border-radius: 2px;"
            )
            checkpoint_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            checkpoint_layout.addWidget(checkpoint_path_label)

            layout.addWidget(checkpoint_section)

        # Resume command (if available)
        if self._resume_command:
            resume_section = QWidget()
            resume_layout = QVBoxLayout(resume_section)
            resume_layout.setContentsMargins(0, 10, 0, 0)
            resume_layout.setSpacing(8)

            resume_header = QLabel("<b>To resume training, use the CLI:</b>")
            resume_header.setTextFormat(Qt.RichText)
            resume_layout.addWidget(resume_header)

            # Command display
            command_text = QTextEdit()
            command_text.setPlainText(self._resume_command)
            command_text.setReadOnly(True)
            command_text.setMaximumHeight(80)
            command_text.setStyleSheet(
                "font-family: monospace; background-color: #2d2d2d; "
                "color: #f8f8f2; padding: 8px; border-radius: 4px;"
            )
            resume_layout.addWidget(command_text)

            # Copy button
            copy_layout = QHBoxLayout()
            copy_layout.addStretch()
            self._copy_button = QPushButton("Copy Command")
            self._copy_button.clicked.connect(self._on_copy_command)
            copy_layout.addWidget(self._copy_button)
            resume_layout.addLayout(copy_layout)

            layout.addWidget(resume_section)

        layout.addStretch()

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        # OK button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)
        layout.addLayout(button_layout)

    def _on_copy_command(self):
        """Copy the resume command to clipboard."""
        if self._resume_command:
            clipboard = QApplication.clipboard()
            clipboard.setText(self._resume_command.replace(" \\\n    ", " "))

            # Show feedback
            original_text = self._copy_button.text()
            self._copy_button.setText("Copied!")
            self._copy_button.setEnabled(False)

            from qtpy.QtCore import QTimer

            QTimer.singleShot(
                1500,
                lambda: (
                    self._copy_button.setText(original_text),
                    self._copy_button.setEnabled(True),
                ),
            )

    def get_resume_command(self) -> str | None:
        """Get the CLI command to resume training.

        Returns:
            The resume command string, or None if no checkpoint available.
        """
        return self._resume_command


class RemoteFileBrowser(QWidget):
    """Column-view file browser for navigating a remote worker's filesystem.

    Provides a macOS Finder-style column view where clicking a folder opens
    a new column to the right showing its contents. Uses the FS_* protocol
    messages over an injected ``send_fn`` callable for transport-agnostic
    communication.

    Args:
        send_fn: Callable that sends a string message to the worker.
        file_filter: Comma-separated glob patterns for selectable files
            (e.g., ``"*.slp"`` or ``"*.mp4,*.avi"``). Files not matching
            are shown greyed out and cannot be selected.
        parent: Optional parent widget.

    Signals:
        file_selected: Emitted with the full path when a file is confirmed
            via double-click or the Select button.
        response_received: Internal signal for thread-safe delivery of
            FS_* response strings from async callbacks.
    """

    file_selected = Signal(str)
    response_received = Signal(str)
    directory_entered = Signal(str)  # emitted when user navigates into a directory

    _SEPARATOR = "::"
    _COLUMN_WIDTH = 200
    _PREVIEW_WIDTH = 180

    def __init__(
        self,
        send_fn: "Callable[[str], None]",
        file_filter: str = "",
        directory_mode: bool = False,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self._send_fn = send_fn
        self._file_filter = file_filter
        self._directory_mode = directory_mode
        self._allowed_extensions: set[str] = set()
        if file_filter:
            for pattern in file_filter.split(","):
                ext = pattern.strip().lstrip("*").lower()
                if ext:
                    self._allowed_extensions.add(ext)

        # State
        self._columns: list[QListWidget] = []
        self._column_paths: list[str] = []  # path each column represents
        self._selected_path: str = ""

        self._build_ui()

        # Thread-safe response routing
        self.response_received.connect(
            self._handle_response, Qt.ConnectionType.QueuedConnection
        )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        """Build the column-view layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Main area: columns + preview
        self._splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # Column scroll area (holds mount selector + directory columns).
        # Horizontal scrolling navigates columns; each QListWidget column
        # handles its own vertical scrolling internally.
        self._column_scroll = QScrollArea()
        self._column_scroll.setWidgetResizable(True)
        self._column_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._column_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._column_container = QWidget()
        self._column_layout = QHBoxLayout(self._column_container)
        self._column_layout.setContentsMargins(0, 0, 0, 0)
        self._column_layout.setSpacing(0)
        self._column_layout.addStretch()
        self._column_scroll.setWidget(self._column_container)

        # File preview panel
        self._preview = QWidget()
        self._preview.setFixedWidth(self._PREVIEW_WIDTH)
        preview_layout = QVBoxLayout(self._preview)
        preview_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._preview_name = QLabel("")
        self._preview_name.setWordWrap(True)
        self._preview_name.setStyleSheet("font-weight: bold;")
        self._preview_size = QLabel("")
        self._preview_modified = QLabel("")
        preview_layout.addWidget(self._preview_name)
        preview_layout.addWidget(self._preview_size)
        preview_layout.addWidget(self._preview_modified)
        preview_layout.addStretch()

        self._splitter.addWidget(self._column_scroll)
        self._splitter.addWidget(self._preview)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)

        layout.addWidget(self._splitter, 1)

        # Bottom bar: path + select button
        bottom = QHBoxLayout()
        self._path_bar = QLineEdit()
        placeholder = (
            "No directory selected" if self._directory_mode else "No file selected"
        )
        self._path_bar.setPlaceholderText(placeholder)
        self._path_bar.setReadOnly(True)
        self._select_button = QPushButton("Select")
        self._select_button.setEnabled(False)
        self._select_button.clicked.connect(self._on_select_clicked)
        bottom.addWidget(self._path_bar, 1)
        bottom.addWidget(self._select_button)
        layout.addLayout(bottom)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_mounts(self):
        """Request mount points from the worker.

        Sends ``FS_GET_MOUNTS`` via the configured ``send_fn``.
        """
        self._send_fn("FS_GET_MOUNTS")

    def on_response(self, message: str):
        """Route an incoming FS_* response to the widget thread-safely.

        Call this from any thread (e.g. a WebRTC data channel callback).
        The message is delivered to the Qt thread via ``response_received``.
        """
        self.response_received.emit(message)

    # ------------------------------------------------------------------
    # Internal response handling
    # ------------------------------------------------------------------

    def _handle_response(self, message: str):
        """Dispatch an FS_* response to the appropriate handler."""
        if message.startswith("FS_MOUNTS_RESPONSE" + self._SEPARATOR):
            payload = message.split(self._SEPARATOR, 1)[1]
            self._handle_mounts_response(payload)
        elif message.startswith("FS_LIST_RESPONSE" + self._SEPARATOR):
            payload = message.split(self._SEPARATOR, 1)[1]
            self._handle_list_response(payload)
        elif message.startswith("FS_ERROR" + self._SEPARATOR):
            parts = message.split(self._SEPARATOR, 2)
            error_code = parts[1] if len(parts) > 1 else "UNKNOWN"
            error_msg = parts[2] if len(parts) > 2 else ""
            self._handle_error(error_code, error_msg)

    def _handle_mounts_response(self, payload: str):
        """Populate the mount selector column from a FS_MOUNTS_RESPONSE."""
        import json

        try:
            mounts = json.loads(payload)
        except json.JSONDecodeError:
            return

        # Clear existing columns
        self._clear_columns()

        # Create mount selector column
        mount_list = self._create_column()
        for mount in mounts:
            label = mount.get("label", mount.get("path", ""))
            path = mount.get("path", "")
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setData(Qt.ItemDataRole.UserRole + 1, "mount")
            mount_list.addItem(item)

        self._column_paths.append("")  # mount column has no path

    def _handle_list_response(self, payload: str):
        """Populate or extend a directory column from a FS_LIST_RESPONSE."""
        import json

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return

        dir_path = data.get("path", "")
        entries = data.get("entries", [])
        has_more = data.get("has_more", False)
        total_count = data.get("total_count", 0)

        # Find if we already have a column for this path (pagination append)
        existing_idx = None
        for i, p in enumerate(self._column_paths):
            if p == dir_path:
                existing_idx = i
                break

        if existing_idx is not None:
            # Pagination: remove "Load more..." item and append new entries
            col = self._columns[existing_idx]
            # Remove the last item if it's a "Load more..." sentinel
            for row in range(col.count() - 1, -1, -1):
                item = col.item(row)
                if item and item.data(Qt.ItemDataRole.UserRole + 1) == "load_more":
                    col.takeItem(row)
                    break
            self._populate_column(col, entries, dir_path, has_more, total_count)
        else:
            # New column for this directory
            col = self._create_column()
            self._column_paths.append(dir_path)
            self._populate_column(col, entries, dir_path, has_more, total_count)

    def _handle_error(self, error_code: str, error_msg: str):
        """Handle an FS_ERROR response."""
        from loguru import logger

        logger.warning(f"RemoteFileBrowser FS error: {error_code}: {error_msg}")

    # ------------------------------------------------------------------
    # Column management
    # ------------------------------------------------------------------

    def _create_column(self) -> QListWidget:
        """Create and add a new column to the column container."""
        col = QListWidget()
        col.setFixedWidth(self._COLUMN_WIDTH)
        col.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        col.itemClicked.connect(self._on_item_clicked)
        col.itemDoubleClicked.connect(self._on_item_double_clicked)
        # Insert before the stretch
        idx = self._column_layout.count() - 1  # before stretch
        self._column_layout.insertWidget(idx, col)
        self._columns.append(col)
        return col

    def _clear_columns(self):
        """Remove all columns."""
        for col in self._columns:
            self._column_layout.removeWidget(col)
            col.deleteLater()
        self._columns.clear()
        self._column_paths.clear()
        self._clear_preview()
        self._selected_path = ""
        self._path_bar.clear()
        self._select_button.setEnabled(False)

    def _remove_columns_after(self, index: int):
        """Remove all columns after the given index."""
        while len(self._columns) > index + 1:
            col = self._columns.pop()
            self._column_layout.removeWidget(col)
            col.deleteLater()
            self._column_paths.pop()

    def _populate_column(
        self,
        col: QListWidget,
        entries: list[dict],
        dir_path: str,
        has_more: bool,
        total_count: int,
    ):
        """Add entries to a column list widget."""
        # Sort: directories first, then files, alphabetically within each
        dirs = sorted(
            [e for e in entries if e.get("type") == "directory"],
            key=lambda e: e["name"].lower(),
        )
        files = sorted(
            [e for e in entries if e.get("type") != "directory"],
            key=lambda e: e["name"].lower(),
        )

        for entry in dirs:
            name = entry["name"]
            # Trailing indicator for folders
            item = QListWidgetItem(name + "/")
            full_path = dir_path.rstrip("/") + "/" + name
            item.setData(Qt.ItemDataRole.UserRole, full_path)
            item.setData(Qt.ItemDataRole.UserRole + 1, "directory")
            col.addItem(item)

        for entry in files:
            name = entry["name"]
            item = QListWidgetItem(name)
            full_path = dir_path.rstrip("/") + "/" + name
            item.setData(Qt.ItemDataRole.UserRole, full_path)
            item.setData(Qt.ItemDataRole.UserRole + 1, "file")
            # Store metadata for preview
            item.setData(Qt.ItemDataRole.UserRole + 2, entry.get("size", 0))
            item.setData(Qt.ItemDataRole.UserRole + 3, entry.get("modified", 0))

            # Apply file filter
            if self._allowed_extensions and not self._matches_filter(name):
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                item.setForeground(
                    col.palette().color(col.palette().ColorRole.PlaceholderText)
                )

            col.addItem(item)

        # Pagination sentinel
        if has_more:
            load_more = QListWidgetItem("Load more...")
            load_more.setData(Qt.ItemDataRole.UserRole, dir_path)
            load_more.setData(Qt.ItemDataRole.UserRole + 1, "load_more")
            load_more.setData(
                Qt.ItemDataRole.UserRole + 4, col.count()
            )  # offset = current count
            load_more.setForeground(col.palette().color(col.palette().ColorRole.Link))
            col.addItem(load_more)

        # Scroll column container to show new column
        self._column_scroll.ensureWidgetVisible(col)

    def _matches_filter(self, filename: str) -> bool:
        """Check if a filename matches the file filter."""
        lower = filename.lower()
        return any(lower.endswith(ext) for ext in self._allowed_extensions)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_item_clicked(self, item: QListWidgetItem):
        """Handle single click on a column item."""
        item_type = item.data(Qt.ItemDataRole.UserRole + 1)
        path = item.data(Qt.ItemDataRole.UserRole)

        if item_type == "load_more":
            # Pagination: request next page
            offset = item.data(Qt.ItemDataRole.UserRole + 4)
            self._send_fn(
                f"FS_LIST_DIR{self._SEPARATOR}{path}{self._SEPARATOR}{offset}"
            )
            return

        # Find which column this item belongs to
        source_col = item.listWidget()
        col_idx = self._columns.index(source_col) if source_col in self._columns else -1

        if item_type == "mount":
            # Mount click: clear columns after mount selector, request root listing
            self._remove_columns_after(0)
            self._send_fn(f"FS_LIST_DIR{self._SEPARATOR}{path}{self._SEPARATOR}0")
            self._clear_preview()
            self.directory_entered.emit(path)
            if self._directory_mode:
                self._selected_path = path
                self._path_bar.setText(path)
                self._select_button.setEnabled(True)
            else:
                self._selected_path = ""
                self._path_bar.clear()
                self._select_button.setEnabled(False)

        elif item_type == "directory":
            # Directory click: remove deeper columns, request listing
            if col_idx >= 0:
                self._remove_columns_after(col_idx)
            self._send_fn(f"FS_LIST_DIR{self._SEPARATOR}{path}{self._SEPARATOR}0")
            self._clear_preview()
            self.directory_entered.emit(path)
            if self._directory_mode:
                # In directory-pick mode, selecting a directory IS the action
                self._selected_path = path
                self._path_bar.setText(path)
                self._select_button.setEnabled(True)
            else:
                self._selected_path = ""
                self._path_bar.clear()
                self._select_button.setEnabled(False)

        elif item_type == "file":
            # File click: highlight, show preview, update path bar
            if self._directory_mode:
                # In directory-pick mode, file clicks don't constitute a selection
                return
            if col_idx >= 0:
                self._remove_columns_after(col_idx)
            self._selected_path = path
            self._path_bar.setText(path)
            self._select_button.setEnabled(True)
            self._show_preview(item)

    def _on_item_double_clicked(self, item: QListWidgetItem):
        """Handle double click — confirms file selection."""
        item_type = item.data(Qt.ItemDataRole.UserRole + 1)
        if item_type == "file" and item.flags() & Qt.ItemFlag.ItemIsEnabled:
            path = item.data(Qt.ItemDataRole.UserRole)
            self._selected_path = path
            self._path_bar.setText(path)
            self.file_selected.emit(path)

    def _on_select_clicked(self):
        """Handle Select button click."""
        if self._selected_path:
            self.file_selected.emit(self._selected_path)

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _show_preview(self, item: QListWidgetItem):
        """Show file metadata in the preview panel."""
        import datetime

        name = item.text()
        size = item.data(Qt.ItemDataRole.UserRole + 2) or 0
        modified = item.data(Qt.ItemDataRole.UserRole + 3) or 0

        self._preview_name.setText(name)
        self._preview_size.setText(self._format_size(size))
        if modified:
            dt = datetime.datetime.fromtimestamp(modified)
            self._preview_modified.setText(dt.strftime("%Y-%m-%d %H:%M"))
        else:
            self._preview_modified.setText("")

    def _clear_preview(self):
        """Clear the preview panel."""
        self._preview_name.setText("")
        self._preview_size.setText("")
        self._preview_modified.setText("")

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format byte count as human-readable string."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


class RemoteTrainingWidget(QGroupBox):
    """Widget for configuring remote training via sleap-connect.

    This widget provides UI elements for:
    - Enabling/disabling remote training
    - Selecting a room to connect to
    - Selecting a worker (auto or manual)
    - Viewing authentication status
    - Viewing connection status

    Signals:
        enabled_changed: Emitted when remote training is enabled/disabled.
        room_changed: Emitted when the selected room changes.
        worker_changed: Emitted when the selected worker changes.
    """

    enabled_changed = Signal(bool)
    room_changed = Signal(str)  # room_id
    worker_changed = Signal(str)  # worker_id or empty for auto

    def __init__(self, parent: QWidget | None = None, mode: str = "training"):
        title = (
            "Remote Training (Experimental)"
            if mode == "training"
            else "Remote Inference (Experimental)"
        )
        super().__init__(title, parent)

        # State
        self._rooms: list[Room] = []
        self._workers: list[Worker] = []
        self._current_user: User | None = None
        self._worker_thread: WorkerDiscoveryThread | None = None
        self._room_thread: RoomLoadThread | None = None
        self._login_thread: LoginThread | None = None
        self._pending_room_selection: str | None = None

        # Build UI
        self._setup_ui()

        # Initial state
        self._update_ui_state()

        # Check initial auth status
        self._check_auth_status()

    def _setup_ui(self):
        """Build the widget UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Enable checkbox - controls whether Run trains locally or remotely
        self._enable_checkbox = QCheckBox("Enable Remote Training")
        self._enable_checkbox.stateChanged.connect(self._on_enable_changed)
        layout.addWidget(self._enable_checkbox)

        # Main content widget (disabled when checkbox unchecked)
        self._content_widget = QWidget()
        content_layout = QVBoxLayout(self._content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)

        # Authentication section
        auth_layout = QHBoxLayout()
        self._auth_status_label = QLabel("Status: Checking...")
        self._auth_status_label.setWordWrap(True)
        auth_layout.addWidget(self._auth_status_label, 1)

        self._login_button = QPushButton("Login...")
        self._login_button.clicked.connect(self._on_login_clicked)
        auth_layout.addWidget(self._login_button)

        self._logout_button = QPushButton("Logout")
        self._logout_button.clicked.connect(self._on_logout_clicked)
        auth_layout.addWidget(self._logout_button)

        content_layout.addLayout(auth_layout)

        # Room selection section
        room_layout = QHBoxLayout()
        room_layout.addWidget(QLabel("Room:"))

        self._room_combo = QComboBox()
        self._room_combo.setMinimumWidth(200)
        self._room_combo.currentIndexChanged.connect(self._on_room_changed)
        room_layout.addWidget(self._room_combo, 1)

        self._browse_rooms_button = QPushButton("Browse...")
        self._browse_rooms_button.clicked.connect(self._on_browse_rooms_clicked)
        room_layout.addWidget(self._browse_rooms_button)

        content_layout.addLayout(room_layout)

        # Connection status
        self._connection_status_label = QLabel("Status: Not connected")
        self._connection_status_label.setStyleSheet("color: gray;")
        content_layout.addWidget(self._connection_status_label)

        # Worker selection section
        worker_group = QWidget()
        worker_layout = QVBoxLayout(worker_group)
        worker_layout.setContentsMargins(0, 0, 0, 0)
        worker_layout.setSpacing(4)

        worker_layout.addWidget(QLabel("Worker Selection:"))

        # Auto-select radio
        self._worker_button_group = QButtonGroup(self)

        self._auto_worker_radio = QRadioButton("Auto-select (best available GPU)")
        self._auto_worker_radio.setChecked(True)
        self._worker_button_group.addButton(self._auto_worker_radio)
        worker_layout.addWidget(self._auto_worker_radio)

        # Manual select radio + dropdown
        manual_layout = QHBoxLayout()
        self._manual_worker_radio = QRadioButton("Choose worker:")
        self._worker_button_group.addButton(self._manual_worker_radio)
        manual_layout.addWidget(self._manual_worker_radio)

        self._worker_combo = QComboBox()
        self._worker_combo.setMinimumWidth(250)
        self._worker_combo.setEnabled(False)
        manual_layout.addWidget(self._worker_combo, 1)

        self._refresh_workers_button = QPushButton("Refresh")
        self._refresh_workers_button.setMaximumWidth(70)
        self._refresh_workers_button.clicked.connect(self._on_refresh_workers_clicked)
        manual_layout.addWidget(self._refresh_workers_button)

        worker_layout.addLayout(manual_layout)

        content_layout.addWidget(worker_group)

        # Connect radio button changes
        self._auto_worker_radio.toggled.connect(self._on_worker_selection_changed)
        self._manual_worker_radio.toggled.connect(self._on_worker_selection_changed)
        self._worker_combo.currentIndexChanged.connect(self._on_worker_combo_changed)

        layout.addWidget(self._content_widget)

        # Initially disable content
        self._content_widget.setEnabled(False)

    def _update_ui_state(self):
        """Update UI element states based on current state."""
        is_enabled = self._enable_checkbox.isChecked()
        is_logged_in = self._current_user is not None
        has_room = self._room_combo.currentIndex() >= 0

        # Content widget enabled only when checkbox is checked
        self._content_widget.setEnabled(is_enabled)

        # Auth buttons
        self._login_button.setVisible(not is_logged_in)
        self._logout_button.setVisible(is_logged_in)

        # Room combo enabled only when logged in
        self._room_combo.setEnabled(is_logged_in)
        self._browse_rooms_button.setEnabled(is_logged_in)

        # Worker controls enabled only when room selected
        self._worker_combo.setEnabled(
            self._manual_worker_radio.isChecked() and has_room
        )
        self._refresh_workers_button.setEnabled(has_room)

        # Update auth status label
        if is_logged_in:
            self._auth_status_label.setText(
                f"Status: Logged in as {self._current_user.username}"
            )
            self._auth_status_label.setStyleSheet("color: green;")
        else:
            self._auth_status_label.setText("Status: Not logged in")
            self._auth_status_label.setStyleSheet("color: orange;")

    def _check_auth_status(self):
        """Check current authentication status."""
        try:
            # In mock mode, start not logged in (user can click Login to mock login)
            if _is_mock_mode():
                self._current_user = None
                self._update_ui_state()
                return

            from sleap_connect.api import is_logged_in, get_logged_in_user

            if is_logged_in():
                self._current_user = get_logged_in_user()
                if self._current_user:
                    self._load_rooms()
            else:
                self._current_user = None
        except Exception:
            self._current_user = None

        self._update_ui_state()

    def _load_rooms(self):
        """Load available rooms in background."""
        if self._room_thread and self._room_thread.isRunning():
            return

        self._room_combo.clear()
        self._room_combo.addItem("Loading rooms...", None)
        self._room_combo.setEnabled(False)

        self._room_thread = RoomLoadThread(self)
        self._room_thread.rooms_loaded.connect(self._on_rooms_loaded)
        self._room_thread.error.connect(self._on_rooms_error)
        self._room_thread.start()

    def _on_rooms_loaded(self, rooms: list):
        """Handle rooms loaded from background thread."""
        # Filter to only show active (non-expired) rooms
        rooms = _filter_active_rooms(rooms)
        self._rooms = rooms
        self._room_combo.clear()

        select_index = 0  # Default to first room with a secret
        first_ready_room_index = -1

        if not rooms:
            self._room_combo.addItem("No rooms available", None)
        else:
            for i, room in enumerate(rooms):
                display_text = f"{room.name} ({room.role})"
                if first_ready_room_index < 0:
                    first_ready_room_index = i
                self._room_combo.addItem(display_text, room.id)
                # Check if this is the pending selection
                if (
                    self._pending_room_selection
                    and room.id == self._pending_room_selection
                ):
                    select_index = i

            # Default to first room
            if first_ready_room_index >= 0 and not self._pending_room_selection:
                select_index = first_ready_room_index

        # Apply pending selection if any
        if self._pending_room_selection and rooms:
            self._room_combo.setCurrentIndex(select_index)
            self._pending_room_selection = None
        elif rooms and first_ready_room_index >= 0:
            self._room_combo.setCurrentIndex(first_ready_room_index)

        self._room_combo.setEnabled(True)
        self._update_ui_state()

    def _on_rooms_error(self, error: str):
        """Handle room loading error."""
        self._room_combo.clear()
        self._room_combo.addItem(f"Error: {error[:30]}...", None)
        self._room_combo.setEnabled(True)
        self._pending_room_selection = None
        self._update_ui_state()

    def _load_workers(self, room_id: str):
        """Load workers for the selected room."""
        if self._worker_thread and self._worker_thread.isRunning():
            return

        self._worker_combo.clear()
        self._worker_combo.addItem("Loading workers...", None)
        self._refresh_workers_button.setEnabled(False)

        self._connection_status_label.setText("Status: Connecting...")
        self._connection_status_label.setStyleSheet("color: orange;")

        self._worker_thread = WorkerDiscoveryThread(room_id, self)
        self._worker_thread.workers_loaded.connect(self._on_workers_loaded)
        self._worker_thread.error.connect(self._on_workers_error)
        self._worker_thread.start()

    def _on_workers_loaded(self, workers: list):
        """Handle workers loaded from background thread."""
        self._workers = workers
        self._worker_combo.clear()

        if not workers:
            self._worker_combo.addItem("No workers available", None)
            self._connection_status_label.setText("Status: Connected (0 workers)")
            self._connection_status_label.setStyleSheet("color: orange;")

            # Show worker setup dialog
            self._show_worker_setup_dialog()
        else:
            for worker in workers:
                # Format: "worker-name (GPU Name, XXX GB) - status"
                gpu_info = ""
                if worker.gpu_name:
                    gpu_info = f"{worker.gpu_name}"
                    if worker.gpu_memory_mb:
                        gpu_info += f", {worker.gpu_memory_mb // 1024}GB"
                    gpu_info = f" ({gpu_info})"

                display_text = f"{worker.name}{gpu_info} - {worker.status}"
                self._worker_combo.addItem(display_text, worker.id)

            self._connection_status_label.setText(
                f"Status: Connected ({len(workers)} worker{'s' if len(workers) != 1 else ''} available)"
            )
            self._connection_status_label.setStyleSheet("color: green;")

        self._refresh_workers_button.setEnabled(True)
        self._update_ui_state()

    def _on_workers_error(self, error: str):
        """Handle worker loading error."""
        self._workers = []
        self._worker_combo.clear()
        self._worker_combo.addItem("Error loading workers", None)
        self._refresh_workers_button.setEnabled(True)

        # Provide more helpful error messages
        error_lower = error.lower()
        if "no access" in error_lower or "room not found" in error_lower:
            display_error = "Could not access room - check login and room membership"
        elif (
            "authentication" in error_lower
            or "401" in error_lower
            or "403" in error_lower
        ):
            display_error = "Authentication failed - try logging in again"
        elif "timeout" in error_lower or "connection" in error_lower:
            display_error = "Connection timeout - check network"
        else:
            # Truncate long errors but show more context
            display_error = error[:60] + "..." if len(error) > 60 else error

        self._connection_status_label.setText(f"Status: Error - {display_error}")
        self._connection_status_label.setStyleSheet("color: red;")

        self._update_ui_state()

    def _show_worker_setup_dialog(self):
        """Show the worker setup dialog to help user set up a worker."""
        room_name = ""
        room_id = ""
        current_index = self._room_combo.currentIndex()
        if current_index >= 0:
            display_text = self._room_combo.currentText()
            room_id = self._room_combo.itemData(current_index) or ""

            if " (" in display_text:
                room_name = display_text.rsplit(" (", 1)[0]
            else:
                room_name = display_text

        dialog = WorkerSetupDialog(
            room_name=room_name,
            room_id=room_id,
            parent=self,
        )
        dialog.exec()

    def _show_room_setup_dialog(self, room_id: str):
        """Show the room setup dialog to generate/enter a room secret."""
        # Get room info from the stored rooms
        room_name = ""
        is_owner = False
        for room in self._rooms:
            if room.id == room_id:
                room_name = room.name
                is_owner = room.role == "owner"
                break

        # Clear worker UI while showing setup dialog
        self._workers = []
        self._worker_combo.clear()
        self._worker_combo.addItem("Room needs setup", None)
        self._connection_status_label.setText("Status: Room secret required")
        self._connection_status_label.setStyleSheet("color: orange;")

        dialog = RoomSecretSetupDialog(
            room_id=room_id, room_name=room_name, is_owner=is_owner, parent=self
        )
        dialog.secret_saved.connect(self._on_room_secret_saved)
        dialog.exec()

    def _on_room_secret_saved(self, room_id: str, secret: str):
        """Handle room secret being saved."""
        # Update the room combo to remove "[needs setup]" indicator
        for i in range(self._room_combo.count()):
            if self._room_combo.itemData(i) == room_id:
                display_text = self._room_combo.itemText(i)
                if " [needs setup]" in display_text:
                    new_text = display_text.replace(" [needs setup]", "")
                    self._room_combo.setItemText(i, new_text)
                break

        # Now try to load workers
        self._load_workers(room_id)

    # Event handlers

    def _on_enable_changed(self, state: int):
        """Handle enable checkbox state change."""
        is_enabled = state == Qt.Checked
        self._content_widget.setEnabled(is_enabled)
        self.enabled_changed.emit(is_enabled)
        self._update_ui_state()

    def _on_login_clicked(self):
        """Handle login button click."""
        if self._login_thread and self._login_thread.isRunning():
            return

        self._login_button.setEnabled(False)
        self._login_button.setText("Logging in...")
        self._auth_status_label.setText("Status: Opening browser...")
        self._auth_status_label.setStyleSheet("color: orange;")

        self._login_thread = LoginThread(parent=self)
        self._login_thread.login_success.connect(self._on_login_success)
        self._login_thread.login_failed.connect(self._on_login_failed)
        self._login_thread.url_ready.connect(self._on_login_url_ready)
        self._login_thread.start()

    def _on_login_url_ready(self, url: str):
        """Handle login URL ready."""
        self._auth_status_label.setText("Status: Waiting for browser login...")

    def _on_login_success(self, user):
        """Handle successful login."""
        self._current_user = user
        self._login_button.setEnabled(True)
        self._login_button.setText("Login...")
        self._update_ui_state()
        self._load_rooms()

    def _on_login_failed(self, error: str):
        """Handle login failure."""
        self._login_button.setEnabled(True)
        self._login_button.setText("Login...")
        self._auth_status_label.setText(f"Status: Login failed - {error[:30]}")
        self._auth_status_label.setStyleSheet("color: red;")

    def _on_logout_clicked(self):
        """Handle logout button click."""
        try:
            from sleap_connect.api import logout

            logout()
        except Exception:
            pass

        self._current_user = None
        self._rooms = []
        self._workers = []
        self._room_combo.clear()
        self._worker_combo.clear()
        self._connection_status_label.setText("Status: Not connected")
        self._connection_status_label.setStyleSheet("color: gray;")
        self._update_ui_state()

    def _on_room_changed(self, index: int):
        """Handle room selection change."""
        room_id = self._room_combo.itemData(index)
        if room_id:
            self.room_changed.emit(room_id)
            self._load_workers(room_id)
        else:
            self._workers = []
            self._worker_combo.clear()
            self._connection_status_label.setText("Status: Not connected")
            self._connection_status_label.setStyleSheet("color: gray;")

        self._update_ui_state()

    def _on_browse_rooms_clicked(self):
        """Handle browse rooms button click."""
        dialog = RoomBrowserDialog(parent=self)
        dialog.room_selected.connect(self._on_room_selected_from_browser)
        dialog.exec()

    def _on_room_selected_from_browser(self, room_id: str, room_name: str):
        """Handle room selection from browser dialog."""
        # Find and select the room in the combo box, or add it if not present
        for i in range(self._room_combo.count()):
            if self._room_combo.itemData(i) == room_id:
                self._room_combo.setCurrentIndex(i)
                return

        # Room not in combo, reload rooms and try to select it
        self._load_rooms()
        # The selection will be handled after rooms are loaded
        # Store the pending selection
        self._pending_room_selection = room_id

    def _on_refresh_workers_clicked(self):
        """Handle refresh workers button click."""
        room_id = self._room_combo.itemData(self._room_combo.currentIndex())
        if room_id:
            self._load_workers(room_id)

    def _on_worker_selection_changed(self, checked: bool):
        """Handle worker selection mode change."""
        if self._manual_worker_radio.isChecked():
            self._worker_combo.setEnabled(True)
        else:
            self._worker_combo.setEnabled(False)

        # Emit worker changed signal
        if self._auto_worker_radio.isChecked():
            self.worker_changed.emit("")  # Empty = auto-select
        else:
            worker_id = self._worker_combo.itemData(self._worker_combo.currentIndex())
            self.worker_changed.emit(worker_id or "")

    def _on_worker_combo_changed(self, index: int):
        """Handle worker combo selection change."""
        if self._manual_worker_radio.isChecked():
            worker_id = self._worker_combo.itemData(index)
            self.worker_changed.emit(worker_id or "")

    # Public API

    def is_enabled(self) -> bool:
        """Check if remote training is enabled.

        Returns:
            True if the "Enable Remote Training" checkbox is checked.
        """
        return self._enable_checkbox.isChecked()

    def set_enabled(self, enabled: bool):
        """Set whether remote training is enabled.

        Args:
            enabled: True to enable remote training.
        """
        self._enable_checkbox.setChecked(enabled)

    def get_selected_room_id(self) -> str | None:
        """Get the currently selected room ID.

        Returns:
            Room ID string, or None if no room is selected.
        """
        return self._room_combo.itemData(self._room_combo.currentIndex())

    def get_selected_worker_id(self) -> str | None:
        """Get the currently selected worker ID.

        Returns:
            Worker ID string, None for auto-select, or None if no selection.
        """
        if self._auto_worker_radio.isChecked():
            return None  # Auto-select
        return self._worker_combo.itemData(self._worker_combo.currentIndex())

    def get_selected_worker_name(self) -> str | None:
        """Get the display name of the currently selected worker.

        Returns:
            Worker name string, or None if auto-select or no selection.
        """
        if self._auto_worker_radio.isChecked():
            return None
        text = self._worker_combo.currentText()
        if not text or text in ("No workers available", "Loading workers..."):
            return None
        # Display format is "worker-name (GPU, XXG) - status"; extract just the name
        return text.split(" (")[0].split(" - ")[0].strip()

    def is_auto_worker_selection(self) -> bool:
        """Check if auto worker selection is enabled.

        Returns:
            True if auto-select is chosen, False if manual selection.
        """
        return self._auto_worker_radio.isChecked()

    def get_workers(self) -> list:
        """Get the list of available workers.

        Returns:
            List of Worker objects.
        """
        return self._workers.copy()

    def refresh_workers(self):
        """Refresh the worker list for the current room."""
        self._on_refresh_workers_clicked()

    def refresh_rooms(self):
        """Refresh the room list."""
        self._load_rooms()


class InferenceProgressDialog(QDialog):
    """Dialog shown during post-training inference.

    Displays a progress bar, status label, and scrolling log area while
    ``sleap track`` runs on the worker.  Matches SLEAP's local post-training
    inference window.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Running Inference")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Status label — centered, word-wrapped; bold on completion
        self._status_label = QLabel("Running inference…")
        self._status_label.setWordWrap(True)
        self._status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._status_label)

        # Progress bar — taller for visibility
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setMinimumHeight(25)
        self._progress_bar.setTextVisible(True)
        layout.addWidget(self._progress_bar)

        # "Log output:" label
        log_label = QLabel("Log output:")
        layout.addWidget(log_label)

        # Scrolling log area — dark monospace, matches SLEAP's local dialog
        self._log_text = QPlainTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMinimumHeight(250)
        self._log_text.setStyleSheet(
            "QPlainTextEdit { background-color: #1e1e1e; color: #a0a0a0; }"
        )
        font = self._log_text.font()
        font.setFamily("Consolas, Monaco, monospace")
        font.setPointSize(8)
        self._log_text.setFont(font)
        layout.addWidget(self._log_text, 1)

        # OK / Cancel buttons — Cancel is enabled during inference, OK after
        self._button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self._button_box.button(QDialogButtonBox.Ok).setEnabled(False)
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)
        layout.addWidget(self._button_box)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_waiting(self, worker_name: str | None = None):
        """Show a 'waiting for worker' state with indeterminate progress."""
        if worker_name:
            self._status_label.setText(
                f"Waiting to start remote inference on "
                f'<span style="color: green;">{worker_name}</span> worker…'
            )
            self._status_label.setTextFormat(Qt.RichText)
        else:
            self._status_label.setText("Waiting to start remote inference…")
        self._progress_bar.setRange(0, 0)  # indeterminate
        self.show()

    def update(self, data: dict):
        """Update the dialog with a progress payload from INFERENCE_PROGRESS.

        Expected keys: ``n_processed``, ``n_total``, ``rate``, ``eta``.
        """
        n_processed = data.get("n_processed", 0)
        n_total = data.get("n_total", 0)
        rate = data.get("rate", 0.0)
        eta = data.get("eta", 0)

        self.set_progress(n_processed, n_total, rate, eta)

        # Append a log line
        self._log_text.appendPlainText(
            f"Frame {n_processed}/{n_total} — {rate:.1f} fps"
        )

    def set_progress(self, n_processed: int, n_total: int, rate: float, eta: int):
        """Update the progress bar and status label without appending to the log.

        Used when progress is parsed from raw ``INFERENCE_LOG`` lines rather
        than structured ``INFERENCE_PROGRESS`` JSON.
        """
        if n_total:
            pct = int(100 * n_processed / n_total)
            self._progress_bar.setValue(pct)
            eta_str = self._format_eta(eta)
            self._status_label.setText(
                f"Predicted: {n_processed}/{n_total}  "
                f"FPS: {rate:.1f}  "
                f"ETA: {eta_str}"
            )
        else:
            self._status_label.setText(f"Predicted: {n_processed}  FPS: {rate:.1f}")

    def append_log(self, text: str):
        """Append a raw log line to the scrolling log area.

        Handles ``\\r``-terminated rich progress bar output by taking the last
        segment so only the final state of an overwriting progress line is shown.
        """
        # Rich uses \r to overwrite progress lines; take the last non-empty part
        parts = text.split("\r")
        clean = parts[-1].strip()
        if clean:
            self._log_text.appendPlainText(clean)
            scrollbar = self._log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def finish(
        self,
        n_frames: int | None,
        n_with_instances: int | None,
        n_empty: int | None,
    ):
        """Show the completion summary and enable OK."""
        self._progress_bar.setValue(100)
        if n_frames is not None:
            no_instances = (
                max(0, n_frames - n_with_instances)
                if n_with_instances is not None
                else None
            )
            if n_with_instances is not None:
                self._status_label.setText(
                    f"<b>Inference complete!</b><br><br>"
                    f"Inference ran on {n_frames:,} frames.<br>"
                    f"Instances were predicted on {n_with_instances:,} frames "
                    f"({no_instances:,} frame{'s' if no_instances != 1 else ''} "
                    f"with no instances found)."
                )
            else:
                self._status_label.setText(
                    f"<b>Inference complete!</b><br><br>"
                    f"Inference ran on {n_frames:,} frames."
                )
        else:
            self._status_label.setText("<b>Inference complete!</b>")
        self._log_text.appendPlainText("")
        self._log_text.appendPlainText("Predictions saved.")
        self._button_box.button(QDialogButtonBox.Ok).setEnabled(True)
        self._button_box.button(QDialogButtonBox.Cancel).setEnabled(False)

    def show_error(self, msg: str):
        """Display an error message and enable OK so the user can dismiss."""
        self._status_label.setText("<b>Inference failed.</b>")
        self._log_text.appendPlainText(f"\nError: {msg}")
        self._button_box.button(QDialogButtonBox.Ok).setEnabled(True)
        self._button_box.button(QDialogButtonBox.Cancel).setEnabled(False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_eta(seconds: float) -> str:
        """Format seconds into a human-readable ETA string."""
        try:
            secs = int(seconds)
        except (TypeError, ValueError):
            return "?"
        if secs < 60:
            return f"{secs}s"
        m, s = divmod(secs, 60)
        if m < 60:
            return f"{m}m {s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h {m:02d}m"
