"""Tree-style file browser widget (VSCode-like).

This module provides a single-column file browser with parent directory
navigation, similar to VSCode's file explorer.
"""

import asyncio
from typing import Optional, Callable, Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static, ListView, ListItem, Label
from textual.widget import Widget
from textual.reactive import reactive
from textual.message import Message
from textual.binding import Binding

from sleap_connect.tui.widgets.miller import FileEntry, FileListItem, LoadMoreItem


class ParentDirItem(ListItem):
    """A list item for navigating to parent directory."""

    DEFAULT_CSS = """
    ParentDirItem {
        color: $accent;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compose(self) -> ComposeResult:
        yield Label("..  [parent directory]")


class TreeBrowser(Widget):
    """Single-column file browser with parent directory navigation.

    This widget displays a single list showing the current directory contents
    with a '..' entry to navigate to the parent directory.
    """

    DEFAULT_CSS = """
    TreeBrowser {
        height: 100%;
        width: 100%;
    }

    #path-bar {
        display: none;
    }

    #file-list {
        height: 1fr;
        width: 100%;
        border: solid $primary-darken-2;
    }

    #file-list:focus {
        border: solid $primary;
    }

    #loading-indicator {
        height: 1;
        background: $warning;
        padding: 0 1;
        display: none;
    }

    #loading-indicator.visible {
        display: block;
    }
    """

    BINDINGS = [
        Binding("enter", "select_entry", "Open", show=False),
        Binding("backspace", "go_parent", "Parent Dir", show=False),
    ]

    # Messages
    class PathChanged(Message):
        """Emitted when the current path changes."""

        def __init__(self, path: str, entry: Optional[FileEntry] = None):
            super().__init__()
            self.path = path
            self.entry = entry

    class FileSelected(Message):
        """Emitted when a file is selected (Enter on a file)."""

        def __init__(self, entry: FileEntry, path: str):
            super().__init__()
            self.entry = entry
            self.path = path

    # Reactive properties
    current_path = reactive("/")

    def __init__(
        self,
        fetch_directory: Optional[Callable[[str, int], Any]] = None,
        **kwargs,
    ):
        """Initialize tree browser.

        Args:
            fetch_directory: Async callback to fetch directory contents.
                Should return dict with entries, total_count, has_more.
        """
        super().__init__(**kwargs)
        self.fetch_directory = fetch_directory
        self.entries: list[FileEntry] = []
        self.total_count = 0
        self.has_more = False
        self.is_loading = False
        self.path_history: list[str] = ["/"]  # Stack for back navigation
        self.mount_points: set[str] = set()  # Track mount point paths

    def compose(self) -> ComposeResult:
        yield Static(self.current_path, id="path-bar")
        yield Static("Loading...", id="loading-indicator")
        yield ListView(id="file-list")

    def on_mount(self):
        """Initialize - don't auto-load, wait for explicit load_root() call."""
        pass

    def load_root(self):
        """Load the root directory. Call this after connection is ready."""
        if self.fetch_directory:
            self._load_directory("/")

    def watch_current_path(self, path: str):
        """Update path bar when path changes."""
        if not self.is_mounted:
            return
        try:
            path_bar = self.query_one("#path-bar", Static)
            path_bar.update(path)
        except Exception:
            pass

    def _load_directory(self, path: str):
        """Load directory contents.

        Args:
            path: Directory path to load.
        """
        self.current_path = path
        self.is_loading = True
        self.entries = []

        # Show loading state
        file_list = self.query_one("#file-list", ListView)
        file_list.clear()
        file_list.append(ListItem(Label("Loading...")))

        # Fetch directory contents
        if self.fetch_directory:
            asyncio.create_task(self._fetch_and_populate(path))

    async def _fetch_and_populate(self, path: str):
        """Fetch directory contents and populate list."""
        file_list = self.query_one("#file-list", ListView)

        try:
            result = await self.fetch_directory(path, 0)

            if result is None:
                file_list.clear()
                file_list.append(ListItem(Label("Error: Failed to load directory")))
                return

            if "error" in result:
                file_list.clear()
                file_list.append(ListItem(Label(f"Error: {result['error']}")))
                return

            self.entries = [
                FileEntry(
                    name=e["name"],
                    type=e["type"],
                    size=e.get("size", 0),
                    modified=e.get("modified", 0),
                    path=e.get("actual_path") or f"{path.rstrip('/')}/{e['name']}",
                )
                for e in result.get("entries", [])
            ]

            # Track mount points when loading root
            if path == "/":
                self.mount_points = {e.path for e in self.entries if e.is_dir}

            self.total_count = result.get("total_count", len(self.entries))
            self.has_more = result.get("has_more", False)
            self.is_loading = False

            self._populate()

        except Exception as e:
            file_list.clear()
            file_list.append(ListItem(Label(f"Error: {e}")))

    def _populate(self):
        """Populate the list with entries."""
        file_list = self.query_one("#file-list", ListView)
        file_list.clear()

        # Add parent directory entry if not at root
        if self.current_path != "/":
            file_list.append(ParentDirItem())

        # Add file/directory entries
        for entry in self.entries:
            item = FileListItem(entry)
            file_list.append(item)

        # Add "Load more" if there are more entries
        if self.has_more:
            remaining = self.total_count - len(self.entries)
            file_list.append(LoadMoreItem(remaining))

        # Focus the list
        file_list.focus()

    def _load_more(self):
        """Load more entries (pagination)."""
        if not self.fetch_directory or not self.has_more:
            return

        offset = len(self.entries)
        asyncio.create_task(self._fetch_more(offset))

    async def _fetch_more(self, offset: int):
        """Fetch more entries for pagination."""
        try:
            result = await self.fetch_directory(self.current_path, offset)

            if result is None or "error" in result:
                return

            new_entries = [
                FileEntry(
                    name=e["name"],
                    type=e["type"],
                    size=e.get("size", 0),
                    modified=e.get("modified", 0),
                    path=e.get("actual_path")
                    or f"{self.current_path.rstrip('/')}/{e['name']}",
                )
                for e in result.get("entries", [])
            ]

            # Remove "Load more" item
            file_list = self.query_one("#file-list", ListView)
            if file_list.children and isinstance(file_list.children[-1], LoadMoreItem):
                file_list.children[-1].remove()

            # Add new entries
            for entry in new_entries:
                self.entries.append(entry)
                file_list.append(FileListItem(entry))

            self.has_more = result.get("has_more", False)
            if self.has_more:
                remaining = self.total_count - len(self.entries)
                file_list.append(LoadMoreItem(remaining))

        except Exception:
            pass

    def on_list_view_selected(self, event: ListView.Selected):
        """Handle item selection in the list."""
        item = event.item

        # Handle parent directory
        if isinstance(item, ParentDirItem):
            self._go_to_parent()
            return

        # Handle "Load more" item
        if isinstance(item, LoadMoreItem):
            self._load_more()
            return

        if not isinstance(item, FileListItem):
            return

        entry = item.entry

        if entry.is_dir:
            # Navigate into directory
            self.path_history.append(self.current_path)
            self._load_directory(entry.path)
            self.post_message(self.PathChanged(entry.path, entry))
        else:
            # File selected
            self.post_message(self.FileSelected(entry, entry.path))

    def on_list_view_highlighted(self, event: ListView.Highlighted):
        """Handle item highlight (cursor movement)."""
        item = event.item
        if isinstance(item, FileListItem):
            entry = item.entry
            self.post_message(self.PathChanged(entry.path, entry))

    def _go_to_parent(self):
        """Navigate to parent directory."""
        if self.current_path == "/":
            return

        # Check if current path is a mount point - go to "/" (mounts list)
        if self.current_path in self.mount_points:
            self._load_directory("/")
            self.post_message(self.PathChanged("/"))
            return

        # Get parent path
        parts = self.current_path.rstrip("/").rsplit("/", 1)
        parent_path = parts[0] if parts[0] else "/"

        # Check if parent would be above a mount point - go to "/" instead
        # This happens when parent_path is not "/" and not within any mount
        if parent_path != "/" and self.mount_points:
            is_within_mount = any(
                parent_path.startswith(mp) or parent_path == mp
                for mp in self.mount_points
            )
            if not is_within_mount:
                self._load_directory("/")
                self.post_message(self.PathChanged("/"))
                return

        self._load_directory(parent_path)
        self.post_message(self.PathChanged(parent_path))

    def action_select_entry(self):
        """Select the current entry (same as clicking)."""
        file_list = self.query_one("#file-list", ListView)
        if file_list.index is not None:
            # Trigger selection event
            item = file_list.children[file_list.index]
            if isinstance(item, ParentDirItem):
                self._go_to_parent()
            elif isinstance(item, LoadMoreItem):
                self._load_more()
            elif isinstance(item, FileListItem):
                entry = item.entry
                if entry.is_dir:
                    self.path_history.append(self.current_path)
                    self._load_directory(entry.path)
                    self.post_message(self.PathChanged(entry.path, entry))
                else:
                    self.post_message(self.FileSelected(entry, entry.path))

    def action_go_parent(self):
        """Go to parent directory (backspace)."""
        self._go_to_parent()

    def refresh_current(self):
        """Refresh the current directory."""
        self._load_directory(self.current_path)

    def get_selected_entry(self) -> Optional[FileEntry]:
        """Get the currently selected entry."""
        file_list = self.query_one("#file-list", ListView)
        if file_list.index is not None:
            item = file_list.children[file_list.index]
            if isinstance(item, FileListItem):
                return item.entry
        return None
