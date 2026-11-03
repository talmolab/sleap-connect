"""Miller columns widget for file browsing.

This module provides a Miller columns interface (like macOS Finder) for
navigating directory hierarchies.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional, Callable, Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, ListView, ListItem, Label
from textual.widget import Widget
from textual.reactive import reactive
from textual.message import Message
from textual.binding import Binding


@dataclass
class FileEntry:
    """Represents a file or directory entry."""

    name: str
    type: str  # "file" or "directory"
    size: int = 0
    modified: float = 0.0
    path: str = ""

    @property
    def is_dir(self) -> bool:
        return self.type == "directory"

    @property
    def icon(self) -> str:
        if self.is_dir:
            return ""
        # File type icons based on extension
        ext = self.name.lower().split(".")[-1] if "." in self.name else ""
        icons = {
            "slp": "",
            "mp4": "",
            "avi": "",
            "mov": "",
            "h5": "",
            "json": "",
            "yaml": "",
            "yml": "",
            "py": "",
            "txt": "",
            "md": "",
        }
        return icons.get(ext, "")


class FileListItem(ListItem):
    """A list item representing a file or directory."""

    DEFAULT_CSS = """
    FileListItem .dir-label {
        color: $primary;
    }

    FileListItem .file-label {
        color: $text;
    }
    """

    def __init__(self, entry: FileEntry, **kwargs):
        super().__init__(**kwargs)
        self.entry = entry

    def compose(self) -> ComposeResult:
        css_class = "dir-label" if self.entry.is_dir else "file-label"
        prefix = "▸ " if self.entry.is_dir else "  "
        yield Label(f"{prefix}{self.entry.name}", classes=css_class)


class LoadMoreItem(ListItem):
    """A list item that triggers loading more entries."""

    DEFAULT_CSS = """
    LoadMoreItem {
        color: $accent;
        text-style: italic;
    }
    """

    def __init__(self, remaining: int, **kwargs):
        super().__init__(**kwargs)
        self.remaining = remaining

    def compose(self) -> ComposeResult:
        yield Label(f"  Load more... ({self.remaining} more items)")


class MillerColumn(ListView):
    """A single column in the Miller columns view."""

    DEFAULT_CSS = """
    MillerColumn {
        width: 1fr;
        min-width: 25;
        max-width: 40;
        height: 100%;
        border: solid $primary-darken-2;
        margin: 0 1 0 0;
    }

    MillerColumn:focus {
        border: solid $primary;
    }

    MillerColumn.loading {
        opacity: 0.6;
    }
    """

    def __init__(
        self,
        path: str = "",
        column_index: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.path = path
        self.column_index = column_index
        self.entries: list[FileEntry] = []
        self.total_count = 0
        self.has_more = False
        self.is_loading = False
        self.error: Optional[str] = None

    def set_entries(self, entries: list[FileEntry], total: int, has_more: bool):
        """Set the entries for this column."""
        self.entries = entries
        self.total_count = total
        self.has_more = has_more
        self.is_loading = False
        self.remove_class("loading")
        self._populate()

    def set_loading(self):
        """Set column to loading state."""
        self.is_loading = True
        self.add_class("loading")
        self.clear()
        self.append(ListItem(Label("Loading...")))

    def set_error(self, error: str):
        """Set column to error state."""
        self.error = error
        self.is_loading = False
        self.remove_class("loading")
        self.clear()
        self.append(ListItem(Label(f"Error: {error}")))

    def _populate(self):
        """Populate the list with entries."""
        self.clear()
        for entry in self.entries:
            item = FileListItem(entry)
            self.append(item)

        if self.has_more:
            remaining = self.total_count - len(self.entries)
            self.append(LoadMoreItem(remaining))

    def append_entries(self, new_entries: list[FileEntry], has_more: bool):
        """Append more entries to the column (for pagination)."""
        # Remove "Load more" item if present
        if self.children and isinstance(self.children[-1], LoadMoreItem):
            self.children[-1].remove()

        # Add new entries
        for entry in new_entries:
            self.entries.append(entry)
            item = FileListItem(entry)
            self.append(item)

        self.has_more = has_more
        if has_more:
            remaining = self.total_count - len(self.entries)
            self.append(LoadMoreItem(remaining))

    def get_selected_entry(self) -> Optional[FileEntry]:
        """Get the currently selected entry."""
        if self.index is not None and 0 <= self.index < len(self.entries):
            return self.entries[self.index]
        return None


class MillerColumns(Widget):
    """Miller columns widget for hierarchical file browsing.

    This widget displays multiple columns side-by-side, where selecting
    a directory in one column shows its contents in the next column.
    """

    DEFAULT_CSS = """
    MillerColumns {
        height: 100%;
        width: 100%;
    }

    #columns-container {
        height: 1fr;
        width: 100%;
        padding: 0;
    }

    #path-bar {
        display: none;
    }

    #loading-indicator {
        height: 1;
        background: $warning;
        padding: 0 1;
        display: none;
        color: $background;
    }

    #loading-indicator.visible {
        display: block;
    }
    """

    BINDINGS = [
        Binding("left", "column_left", "Previous column", show=False),
        Binding("right", "column_right", "Next column", show=False),
        Binding("enter", "select_entry", "Open", show=False),
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
    active_column = reactive(0)

    def __init__(
        self,
        fetch_directory: Optional[Callable[[str, int], Any]] = None,
        num_columns: int = 4,
        **kwargs,
    ):
        """Initialize Miller columns.

        Args:
            fetch_directory: Async callback to fetch directory contents.
                Should return dict with entries, total_count, has_more.
            num_columns: Number of visible columns.
        """
        super().__init__(**kwargs)
        self.fetch_directory = fetch_directory
        self.num_columns = num_columns
        self.columns: list[MillerColumn] = []
        self.path_stack: list[str] = ["/"]  # Stack of paths for each column
        self.full_path_history: list[str] = ["/"]  # Complete history of all paths
        self.history_offset: int = 0  # How many paths have been shifted off the left

    def compose(self) -> ComposeResult:
        yield Static(self.current_path, id="path-bar")
        yield Static("Loading...", id="loading-indicator")
        with Horizontal(id="columns-container"):
            for i in range(self.num_columns):
                col = MillerColumn(column_index=i, id=f"col-{i}")
                self.columns.append(col)
                yield col

    def on_mount(self):
        """Initialize the first column with root directory."""
        # Hide columns initially except first
        for i, col in enumerate(self.columns):
            col.display = i == 0

        # Don't auto-load here - wait for explicit load_root() call
        # This allows the parent to establish connections first

    def load_root(self):
        """Load the root directory. Call this after connection is ready."""
        if self.fetch_directory:
            # Reset history state
            self.full_path_history = ["/"]
            self.history_offset = 0
            self.path_stack = ["/"]
            self._load_column(0, "/")

    def watch_current_path(self, path: str):
        """Update path bar when path changes."""
        if not self.is_mounted:
            return
        try:
            path_bar = self.query_one("#path-bar", Static)
            path_bar.update(path)
        except Exception:
            pass

    def _shift_columns_left(self, new_path: str):
        """Shift all columns left to make room for a new directory.

        This is called when navigating deeper than the available columns.
        Like macOS Finder, columns shift left and the new directory appears
        on the right.

        Args:
            new_path: Path of the new directory to display in the last column.
        """
        # Add to full history
        self.full_path_history.append(new_path)

        # Shift path stack left
        if self.path_stack:
            self.path_stack.pop(0)
        self.path_stack.append(new_path)

        # Increment offset to track how many have shifted off
        self.history_offset += 1

        # Shift column data left: col[i] gets col[i+1]'s data
        for i in range(len(self.columns) - 1):
            src_col = self.columns[i + 1]
            dst_col = self.columns[i]

            # Copy data from source to destination
            dst_col.path = src_col.path
            dst_col.entries = src_col.entries.copy()
            dst_col.total_count = src_col.total_count
            dst_col.has_more = src_col.has_more
            dst_col.error = src_col.error

            # Repopulate the destination column with the copied data
            dst_col._populate()
            dst_col.display = True

        # Load new directory into the last column
        last_col_index = len(self.columns) - 1
        last_col = self.columns[last_col_index]
        last_col.path = new_path
        last_col.set_loading()
        last_col.display = True

        # Update active column to the last one
        self.active_column = last_col_index

        # Fetch the new directory contents
        if self.fetch_directory:
            asyncio.create_task(self._fetch_and_populate(last_col_index, new_path))

        # Focus the new column
        last_col.focus()

    def _shift_columns_right(self):
        """Shift all columns right to reveal a parent directory.

        This is called when navigating left at column 0 and there are
        hidden parent directories to reveal.
        """
        if self.history_offset <= 0:
            # Nothing to shift back to
            return

        # Decrement offset
        self.history_offset -= 1

        # Get the path that was shifted off
        parent_path = self.full_path_history[self.history_offset]

        # Update path stack
        self.path_stack.insert(0, parent_path)
        if len(self.path_stack) > self.num_columns:
            self.path_stack.pop()

        # Shift column data right: col[i+1] gets col[i]'s data
        for i in range(len(self.columns) - 1, 0, -1):
            src_col = self.columns[i - 1]
            dst_col = self.columns[i]

            # Copy data from source to destination
            dst_col.path = src_col.path
            dst_col.entries = src_col.entries.copy()
            dst_col.total_count = src_col.total_count
            dst_col.has_more = src_col.has_more
            dst_col.error = src_col.error

            # Repopulate the destination column with the copied data
            dst_col._populate()
            dst_col.display = True

        # Load parent directory into the first column
        first_col = self.columns[0]
        first_col.path = parent_path
        first_col.set_loading()
        first_col.display = True

        # Keep active column at 0
        self.active_column = 0

        # Fetch the parent directory contents
        if self.fetch_directory:
            asyncio.create_task(self._fetch_and_populate(0, parent_path))

        # Focus the first column
        first_col.focus()

    def _load_column(self, col_index: int, path: str):
        """Load directory contents into a column.

        Args:
            col_index: Index of the column to load into.
            path: Directory path to load.
        """
        if col_index >= len(self.columns):
            return

        col = self.columns[col_index]
        col.path = path
        col.set_loading()
        col.display = True

        # Update path stack
        while len(self.path_stack) > col_index:
            self.path_stack.pop()
        self.path_stack.append(path)

        # Update full path history - truncate to current position and add new path
        history_index = self.history_offset + col_index
        # Truncate history to this point
        self.full_path_history = self.full_path_history[:history_index]
        self.full_path_history.append(path)

        # Hide columns after this one
        for i in range(col_index + 1, len(self.columns)):
            self.columns[i].display = False

        # Fetch directory contents
        if self.fetch_directory:
            asyncio.create_task(self._fetch_and_populate(col_index, path))

    async def _fetch_and_populate(self, col_index: int, path: str):
        """Fetch directory contents and populate column."""
        col = self.columns[col_index]

        try:
            result = await self.fetch_directory(path, 0)

            if result is None:
                col.set_error("Failed to load directory")
                return

            if "error" in result:
                col.set_error(result["error"])
                return

            entries = [
                FileEntry(
                    name=e["name"],
                    type=e["type"],
                    size=e.get("size", 0),
                    modified=e.get("modified", 0),
                    # Use actual_path if provided (for mounts), otherwise construct path
                    path=e.get("actual_path") or f"{path.rstrip('/')}/{e['name']}",
                )
                for e in result.get("entries", [])
            ]

            total_count = result.get("total_count", len(entries))
            has_more = result.get("has_more", False)

            col.set_entries(entries, total_count, has_more)

            # Focus this column if it's the active one
            if col_index == self.active_column:
                col.focus()

        except Exception as e:
            col.set_error(str(e))

    def _load_more(self, col: MillerColumn):
        """Load more entries for a column (pagination).

        Args:
            col: The column to load more entries for.
        """
        if not self.fetch_directory or not col.has_more:
            return

        # Current offset is the number of entries already loaded
        offset = len(col.entries)
        asyncio.create_task(self._fetch_more(col, offset))

    async def _fetch_more(self, col: MillerColumn, offset: int):
        """Fetch more entries for pagination."""
        try:
            result = await self.fetch_directory(col.path, offset)

            if result is None:
                return

            if "error" in result:
                return

            new_entries = [
                FileEntry(
                    name=e["name"],
                    type=e["type"],
                    size=e.get("size", 0),
                    modified=e.get("modified", 0),
                    path=e.get("actual_path") or f"{col.path.rstrip('/')}/{e['name']}",
                )
                for e in result.get("entries", [])
            ]

            col.append_entries(new_entries, result.get("has_more", False))

        except Exception as e:
            pass  # Silently fail for pagination errors

    def on_list_view_selected(self, event: ListView.Selected):
        """Handle item selection in a column."""
        # Find which column this is from
        col = event.list_view
        if not isinstance(col, MillerColumn):
            return

        item = event.item

        # Handle "Load more" item
        if isinstance(item, LoadMoreItem):
            self._load_more(col)
            return

        if not isinstance(item, FileListItem):
            return

        entry = item.entry
        col_index = col.column_index

        # Update current path
        self.current_path = entry.path
        self.post_message(self.PathChanged(entry.path, entry))

        if entry.is_dir:
            # Open directory in next column
            next_col = col_index + 1
            if next_col < len(self.columns):
                self._load_column(next_col, entry.path)
                self.active_column = next_col
                self.columns[next_col].focus()
            else:
                # Shift columns left to make room for new directory
                self._shift_columns_left(entry.path)
        else:
            # File selected
            self.post_message(self.FileSelected(entry, entry.path))

    def on_list_view_highlighted(self, event: ListView.Highlighted):
        """Handle item highlight (cursor movement) in a column."""
        col = event.list_view
        if not isinstance(col, MillerColumn):
            return

        item = event.item
        if isinstance(item, FileListItem):
            entry = item.entry
            self.current_path = entry.path
            self.post_message(self.PathChanged(entry.path, entry))

    def action_column_left(self):
        """Move focus to the previous column."""
        if self.active_column > 0:
            self.active_column -= 1
            col = self.columns[self.active_column]
            if col.display:
                col.focus()
                # Update path to selected item in this column
                entry = col.get_selected_entry()
                if entry:
                    self.current_path = entry.path
        elif self.history_offset > 0:
            # At column 0 but have hidden parent directories - shift right
            self._shift_columns_right()

    def action_column_right(self):
        """Move focus to the next column."""
        next_col = self.active_column + 1
        if next_col < len(self.columns) and self.columns[next_col].display:
            self.active_column = next_col
            self.columns[next_col].focus()
            # Update path to selected item in this column
            entry = self.columns[next_col].get_selected_entry()
            if entry:
                self.current_path = entry.path

    def action_select_entry(self):
        """Select the current entry (same as clicking)."""
        col = self.columns[self.active_column]
        entry = col.get_selected_entry()
        if entry:
            if entry.is_dir:
                # Open directory in next column
                next_col = self.active_column + 1
                if next_col < len(self.columns):
                    self._load_column(next_col, entry.path)
                    self.active_column = next_col
                    self.columns[next_col].focus()
            else:
                self.post_message(self.FileSelected(entry, entry.path))

    async def load_path(self, path: str):
        """Load a specific path, expanding columns as needed.

        Args:
            path: The path to navigate to.
        """
        # Split path into components and load each
        parts = [p for p in path.split("/") if p]
        current = "/"

        for i, part in enumerate(parts):
            if i >= len(self.columns):
                break

            self._load_column(i, current)
            current = f"{current.rstrip('/')}/{part}"

            # Wait for column to load
            await asyncio.sleep(0.1)

        self.current_path = path

    def refresh_current(self):
        """Refresh the current column."""
        col = self.columns[self.active_column]
        if col.path:
            self._load_column(self.active_column, col.path)

    def get_selected_entry(self) -> Optional[FileEntry]:
        """Get the currently selected entry in the active column."""
        col = self.columns[self.active_column]
        return col.get_selected_entry()
