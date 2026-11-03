"""Worker tabs widget for switching between workers.

This module provides a tab bar for displaying and switching between
multiple workers in a room.
"""

from dataclasses import dataclass
from typing import Optional, Callable

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static, Button
from textual.widget import Widget
from textual.reactive import reactive
from textual.message import Message
from textual.binding import Binding


@dataclass
class WorkerInfo:
    """Information about a worker."""

    peer_id: str
    hostname: str = ""
    worker_name: str = ""
    status: str = "unknown"
    gpu_info: str = ""
    mounts: list = None

    def __post_init__(self):
        if self.mounts is None:
            self.mounts = []
        if not self.hostname:
            self.hostname = self.peer_id[:15]

    @classmethod
    def from_dict(cls, data: dict) -> "WorkerInfo":
        """Create WorkerInfo from worker discovery response."""
        peer_id = data.get("peer_id", "unknown")
        metadata = data.get("metadata", {})
        props = metadata.get("properties", {})

        return cls(
            peer_id=peer_id,
            hostname=props.get("hostname", peer_id[:15]),
            worker_name=props.get("worker_name", ""),
            status=props.get("status", "unknown"),
            gpu_info=props.get("gpu_name", ""),
            mounts=props.get("mounts", []),
        )

    @property
    def display_name(self) -> str:
        """Get display name for the tab."""
        if self.worker_name:
            return self.worker_name[:20]
        return self.hostname[:20] if self.hostname else self.peer_id[:20]

    @property
    def status_icon(self) -> str:
        """Get status indicator icon."""
        icons = {
            "available": "",
            "busy": "",
            "connected": "",
            "disconnected": "",
            "unknown": "",
        }
        return icons.get(self.status, "")


class WorkerTab(Button):
    """A single worker tab button."""

    DEFAULT_CSS = """
    WorkerTab {
        min-width: 20;
        height: 3;
        margin: 0 1 0 0;
        border: none;
        background: $panel;
        color: $foreground-muted;
    }

    WorkerTab:hover {
        background: $primary-darken-1;
        color: $foreground;
    }

    WorkerTab:focus {
        background: $primary;
    }

    WorkerTab.active {
        background: $accent;
        color: $foreground;
        text-style: bold;
    }

    WorkerTab.connected {
        border-bottom: solid $success;
    }

    WorkerTab.disconnected {
        border-bottom: solid $error;
    }
    """

    def __init__(self, worker: WorkerInfo, index: int, **kwargs):
        label = f"{worker.status_icon} {worker.display_name}"
        super().__init__(label, **kwargs)
        self.worker = worker
        self.index = index

    def set_active(self, active: bool):
        """Set whether this tab is active."""
        if active:
            self.add_class("active")
        else:
            self.remove_class("active")

    def set_connected(self, connected: bool):
        """Set connection status indicator."""
        self.remove_class("connected", "disconnected")
        if connected:
            self.add_class("connected")
        else:
            self.add_class("disconnected")

    def update_status(self, status: str):
        """Update the worker status."""
        self.worker.status = status
        self.label = f"{self.worker.status_icon} {self.worker.display_name}"


class WorkerTabs(Widget):
    """Tab bar for switching between workers.

    Displays a horizontal row of tabs, one per worker in the room.
    Supports keyboard navigation with number keys and Tab.
    """

    DEFAULT_CSS = """
    WorkerTabs {
        height: 3;
        width: 100%;
        background: $background;
        padding: 0 1;
    }

    #tabs-container {
        height: 100%;
        width: auto;
    }

    #no-workers {
        padding: 0 2;
        color: $warning;
    }
    """

    BINDINGS = [
        Binding("1", "select_tab_1", "Worker 1", show=False),
        Binding("2", "select_tab_2", "Worker 2", show=False),
        Binding("3", "select_tab_3", "Worker 3", show=False),
        Binding("4", "select_tab_4", "Worker 4", show=False),
        Binding("5", "select_tab_5", "Worker 5", show=False),
        Binding("6", "select_tab_6", "Worker 6", show=False),
        Binding("7", "select_tab_7", "Worker 7", show=False),
        Binding("8", "select_tab_8", "Worker 8", show=False),
        Binding("9", "select_tab_9", "Worker 9", show=False),
        Binding("tab", "next_tab", "Next Worker", show=False),
        Binding("shift+tab", "prev_tab", "Previous Worker", show=False),
    ]

    # Messages
    class WorkerSelected(Message):
        """Emitted when a worker tab is selected."""

        def __init__(self, worker: WorkerInfo, index: int):
            super().__init__()
            self.worker = worker
            self.index = index

    # Reactive properties
    active_index = reactive(0)

    def __init__(
        self,
        workers: Optional[list[WorkerInfo]] = None,
        on_worker_selected: Optional[Callable[[WorkerInfo, int], None]] = None,
        **kwargs,
    ):
        """Initialize worker tabs.

        Args:
            workers: List of WorkerInfo objects.
            on_worker_selected: Callback when worker is selected.
        """
        super().__init__(**kwargs)
        self._workers = workers or []
        self.on_worker_selected = on_worker_selected
        self.tabs: list[WorkerTab] = []

    def compose(self) -> ComposeResult:
        with Horizontal(id="tabs-container"):
            if not self._workers:
                yield Static("No workers available", id="no-workers")
            else:
                for i, worker in enumerate(self._workers):
                    tab = WorkerTab(worker, i, id=f"worker-tab-{i}")
                    self.tabs.append(tab)
                    yield tab

    def on_mount(self):
        """Set initial active tab."""
        if self.tabs:
            self.tabs[0].set_active(True)

    def watch_active_index(self, index: int):
        """Update tab styles when active index changes."""
        for i, tab in enumerate(self.tabs):
            tab.set_active(i == index)

    def set_workers(self, workers: list[WorkerInfo]):
        """Update the list of workers.

        Args:
            workers: New list of WorkerInfo objects.
        """
        self._workers = workers
        self.tabs = []

        # Clear and rebuild tabs
        container = self.query_one("#tabs-container", Horizontal)
        container.remove_children()

        if not workers:
            container.mount(Static("No workers available", id="no-workers"))
        else:
            for i, worker in enumerate(workers):
                tab = WorkerTab(worker, i, id=f"worker-tab-{i}")
                self.tabs.append(tab)
                container.mount(tab)

            # Reset active index
            self.active_index = 0
            if self.tabs:
                self.tabs[0].set_active(True)

    def set_worker_connected(self, index: int, connected: bool):
        """Set connection status for a worker tab.

        Args:
            index: Worker index.
            connected: Whether the worker is connected.
        """
        if 0 <= index < len(self.tabs):
            self.tabs[index].set_connected(connected)

    def select_worker(self, index: int):
        """Select a worker by index.

        Args:
            index: Worker index to select.
        """
        if 0 <= index < len(self._workers):
            self.active_index = index
            worker = self._workers[index]

            # Post message
            self.post_message(self.WorkerSelected(worker, index))

            # Call callback
            if self.on_worker_selected:
                self.on_worker_selected(worker, index)

    def get_active_worker(self) -> Optional[WorkerInfo]:
        """Get the currently active worker."""
        if 0 <= self.active_index < len(self._workers):
            return self._workers[self.active_index]
        return None

    def on_button_pressed(self, event: Button.Pressed):
        """Handle tab button press."""
        if isinstance(event.button, WorkerTab):
            self.select_worker(event.button.index)

    # Action handlers for number key bindings
    def action_select_tab_1(self):
        self.select_worker(0)

    def action_select_tab_2(self):
        self.select_worker(1)

    def action_select_tab_3(self):
        self.select_worker(2)

    def action_select_tab_4(self):
        self.select_worker(3)

    def action_select_tab_5(self):
        self.select_worker(4)

    def action_select_tab_6(self):
        self.select_worker(5)

    def action_select_tab_7(self):
        self.select_worker(6)

    def action_select_tab_8(self):
        self.select_worker(7)

    def action_select_tab_9(self):
        self.select_worker(8)

    def action_next_tab(self):
        """Select next worker tab."""
        if self._workers:
            next_index = (self.active_index + 1) % len(self._workers)
            self.select_worker(next_index)

    def action_prev_tab(self):
        """Select previous worker tab."""
        if self._workers:
            prev_index = (self.active_index - 1) % len(self._workers)
            self.select_worker(prev_index)
