"""Interactive directory browser for remote filesystem navigation.

This module provides a terminal-based directory browser for navigating
the worker filesystem and selecting files. Used for path correction
when job submission fails due to invalid paths.
"""

import asyncio
import json
import logging
import os
import posixpath
import sys
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Callable, Any


@contextmanager
def suppress_logging():
    """Temporarily suppress all logging output.

    Used during interactive UI sessions to prevent log messages
    from interfering with the terminal display.
    """
    # Get the root logger and save its current level
    root_logger = logging.getLogger()
    original_level = root_logger.level

    # Disable all logging
    root_logger.setLevel(logging.CRITICAL + 1)

    try:
        yield
    finally:
        # Restore original level
        root_logger.setLevel(original_level)


from sleap_connect.protocol import (
    MSG_FS_LIST_DIR,
    MSG_FS_LIST_RESPONSE,
    MSG_FS_GET_MOUNTS,
    MSG_FS_MOUNTS_RESPONSE,
    MSG_FS_ERROR,
    MSG_SEPARATOR,
    FS_ERROR_ACCESS_DENIED,
    format_message,
)


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def _is_interactive_terminal() -> bool:
    """Check if we're running in an interactive terminal."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return False

    term = os.environ.get("TERM", "")
    if term in ("dumb", "", "unknown"):
        return False

    # Check for CI environments
    ci_env_vars = ["CI", "GITHUB_ACTIONS", "GITLAB_CI", "JENKINS_URL"]
    for env_var in ci_env_vars:
        if os.environ.get(env_var):
            return False

    return True


class DirectoryEntry:
    """Represents a directory entry (file or folder)."""

    def __init__(self, name: str, entry_type: str, size: int, modified: float):
        self.name = name
        self.type = entry_type  # "directory" or "file"
        self.size = size
        self.modified = modified

    @property
    def is_directory(self) -> bool:
        return self.type == "directory"

    @classmethod
    def from_dict(cls, data: dict) -> "DirectoryEntry":
        return cls(
            name=data.get("name", ""),
            entry_type=data.get("type", "file"),
            size=data.get("size", 0),
            modified=data.get("modified", 0),
        )


class DirectoryBrowser:
    """Interactive directory browser for remote filesystem navigation.

    Uses prompt_toolkit for arrow-key navigation with async directory
    listing refresh from the worker.

    Usage:
        browser = DirectoryBrowser(
            send_message=channel.send,
            receive_response=response_queue.get,
            start_path="/vast/data",
            file_filter=".slp",
        )
        selected = await browser.run()
        if selected:
            print(f"Selected: {selected}")
    """

    def __init__(
        self,
        send_message: Callable[[str], None],
        receive_response: Callable[[], Any],
        start_path: str = "/",
        file_filter: Optional[str] = None,
        title: str = "Browse remote filesystem:",
    ):
        """Initialize directory browser.

        Args:
            send_message: Function to send messages to worker (data_channel.send).
            receive_response: Async function to receive FS responses.
            start_path: Starting directory path.
            file_filter: Optional file extension filter. Can be a single extension
                (e.g., ".slp") or comma-separated extensions (e.g., ".yaml,.json").
            title: Title to display above the browser.
        """
        self.send_message = send_message
        self.receive_response = receive_response
        self.current_path = start_path
        self.title = title

        # Parse file filter - support comma-separated extensions
        if file_filter:
            self.file_filters = [ext.strip().lower() for ext in file_filter.split(",")]
        else:
            self.file_filters = None

        self.entries: List[DirectoryEntry] = []
        self.selected_index = 0
        self.cancelled = False
        self.selected_path: Optional[str] = None
        self.error_message: Optional[str] = None
        self.loading = False
        self.showing_mounts = False  # True when showing mount selection

    async def _fetch_mounts(self) -> List[dict]:
        """Fetch available mounts from worker.

        Returns:
            List of mount dicts with 'path' and 'label' keys.
        """
        self.send_message(MSG_FS_GET_MOUNTS)

        try:
            response = await asyncio.wait_for(self.receive_response(), timeout=30.0)
        except asyncio.TimeoutError:
            return []

        if response.startswith(MSG_FS_MOUNTS_RESPONSE):
            try:
                json_str = response.split(MSG_SEPARATOR, 1)[1]
                return json.loads(json_str)
            except (json.JSONDecodeError, IndexError):
                return []

        return []

    async def _show_mounts(self) -> None:
        """Show available mounts as directory entries."""
        self.loading = True
        self.error_message = None
        self.showing_mounts = True

        mounts = await self._fetch_mounts()

        if not mounts:
            self.error_message = "No accessible mounts available"
            self.entries = []
        else:
            # Convert mounts to directory entries
            self.entries = []
            for mount in mounts:
                path = mount.get("path", "")
                label = mount.get("label", "")
                name = f"{label} ({path})" if label else path
                self.entries.append(
                    DirectoryEntry(
                        name=name,
                        entry_type="directory",
                        size=0,
                        modified=0,
                    )
                )
            # Store mount paths for navigation
            self._mount_paths = [m.get("path", "") for m in mounts]

        self.current_path = "/ (Select a mount)"
        self.selected_index = 0
        self.loading = False

    async def _refresh_listing(self) -> None:
        """Fetch directory listing from worker."""
        self.loading = True
        self.error_message = None
        self.showing_mounts = False

        # Send FS_LIST_DIR request
        message = format_message(MSG_FS_LIST_DIR, self.current_path, "0")
        self.send_message(message)

        # Wait for response
        try:
            response = await asyncio.wait_for(self.receive_response(), timeout=30.0)
        except asyncio.TimeoutError:
            self.error_message = "Timeout waiting for directory listing"
            self.entries = []
            self.loading = False
            return

        # Parse response
        if response.startswith(MSG_FS_ERROR):
            parts = response.split(MSG_SEPARATOR)
            error_code = parts[1] if len(parts) > 1 else ""
            error_msg = parts[2] if len(parts) > 2 else "Unknown error"

            # If access denied, fall back to showing mounts
            if error_code == FS_ERROR_ACCESS_DENIED:
                self.loading = False
                await self._show_mounts()
                return

            self.error_message = error_msg
            self.entries = []
        elif response.startswith(MSG_FS_LIST_RESPONSE):
            try:
                json_str = response.split(MSG_SEPARATOR, 1)[1]
                data = json.loads(json_str)

                if "error" in data:
                    self.error_message = data["error"]
                    self.entries = []
                else:
                    self.current_path = data.get("path", self.current_path)
                    raw_entries = data.get("entries", [])

                    # Convert to DirectoryEntry objects
                    self.entries = [DirectoryEntry.from_dict(e) for e in raw_entries]

                    # Apply file filter (keep all directories, filter files)
                    if self.file_filters:
                        self.entries = [
                            e
                            for e in self.entries
                            if e.is_directory
                            or any(
                                e.name.lower().endswith(ext)
                                for ext in self.file_filters
                            )
                        ]

                    # Reset selection if out of bounds
                    if self.selected_index >= len(self.entries):
                        self.selected_index = max(0, len(self.entries) - 1)

            except (json.JSONDecodeError, IndexError) as e:
                self.error_message = f"Failed to parse response: {e}"
                self.entries = []
        else:
            self.error_message = f"Unexpected response: {response[:50]}"
            self.entries = []

        self.loading = False

    def _navigate_up(self) -> bool:
        """Navigate to parent directory. Returns True if path changed."""
        if self.current_path in ("/", ""):
            return False

        # Get parent path (use posixpath: remote worker paths are always Linux)
        parent = posixpath.dirname(self.current_path.rstrip("/"))
        if parent == self.current_path:
            return False

        self.current_path = parent if parent else "/"
        self.selected_index = 0
        return True

    def _navigate_into(self, entry: DirectoryEntry, index: int = None) -> bool:
        """Navigate into a directory. Returns True if path changed.

        Args:
            entry: The directory entry to navigate into.
            index: The index of the entry (used when showing mounts).
        """
        if not entry.is_directory:
            return False

        # If showing mounts, use the stored mount path
        if self.showing_mounts and hasattr(self, "_mount_paths") and index is not None:
            self.current_path = self._mount_paths[index]
        else:
            new_path = posixpath.join(self.current_path, entry.name)
            self.current_path = new_path

        self.selected_index = 0
        self.showing_mounts = False
        return True

    def _select_file(self, entry: DirectoryEntry) -> bool:
        """Select a file. Returns True if file was selected."""
        if entry.is_directory:
            return False

        self.selected_path = posixpath.join(self.current_path, entry.name)
        return True

    async def run(self) -> Optional[str]:
        """Run the directory browser and return selected path.

        Returns:
            Selected file path, or None if cancelled.
        """
        if not _is_interactive_terminal():
            return await self._run_simple()

        try:
            return await self._run_interactive()
        except ImportError:
            return await self._run_simple()

    async def _run_interactive(self) -> Optional[str]:
        """Run interactive prompt_toolkit-based browser."""
        from prompt_toolkit import Application
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.formatted_text import FormattedText

        # Suppress logging during the entire interactive session to prevent
        # log messages from cluttering the terminal display
        root_logger = logging.getLogger()
        original_level = root_logger.level
        root_logger.setLevel(logging.CRITICAL + 1)

        try:
            return await self._run_interactive_impl(
                Application,
                KeyBindings,
                Layout,
                Window,
                FormattedTextControl,
                FormattedText,
            )
        finally:
            # Restore logging
            root_logger.setLevel(original_level)

    async def _run_interactive_impl(
        self,
        Application,
        KeyBindings,
        Layout,
        Window,
        FormattedTextControl,
        FormattedText,
    ) -> Optional[str]:
        """Implementation of interactive browser with logging suppressed."""
        # Initial directory listing
        await self._refresh_listing()

        kb = KeyBindings()
        pending_refresh = [False]  # Flag to trigger refresh after app.run() returns
        stay_in_app = [True]  # Flag to keep the main loop running

        @kb.add("up")
        @kb.add("k")
        def move_up(event):
            if self.entries:
                self.selected_index = (self.selected_index - 1) % len(self.entries)

        @kb.add("down")
        @kb.add("j")
        def move_down(event):
            if self.entries:
                self.selected_index = (self.selected_index + 1) % len(self.entries)

        @kb.add("enter")
        @kb.add("right")
        @kb.add("l")
        def select_entry(event):
            if not self.entries:
                return

            entry = self.entries[self.selected_index]
            if entry.is_directory:
                self._navigate_into(entry, index=self.selected_index)
                # Mark for refresh but don't exit fullscreen
                pending_refresh[0] = True
                event.app.exit()
            else:
                self._select_file(entry)
                stay_in_app[0] = False
                event.app.exit()

        @kb.add("backspace")
        @kb.add("left")
        @kb.add("h")
        def go_back(event):
            # Can't go back when showing mount selection
            if self.showing_mounts:
                return
            if self._navigate_up():
                # Mark for refresh but don't exit fullscreen
                pending_refresh[0] = True
                event.app.exit()

        @kb.add("escape")
        @kb.add("q")
        def cancel(event):
            self.cancelled = True
            stay_in_app[0] = False
            event.app.exit()

        @kb.add("c-c")
        def ctrl_c(event):
            self.cancelled = True
            stay_in_app[0] = False
            event.app.exit()

        @kb.add("d")
        def select_as_path(event):
            """Select the highlighted entry as a path (even if it's a folder).

            This allows selecting folders without navigating into them,
            useful for model paths that are directories.
            """
            if self.showing_mounts:
                # Select mount point as path
                if self.mounts:
                    selected_mount = self.mounts[self.selected_index]
                    self.selected_path = selected_mount["path"]
                    stay_in_app[0] = False
                    event.app.exit()
                return

            if not self.entries:
                return

            entry = self.entries[self.selected_index]
            # Select whether it's a file or folder - construct full path
            self.selected_path = posixpath.join(self.current_path, entry.name)
            stay_in_app[0] = False
            event.app.exit()

        def get_formatted_text():
            """Generate formatted text for display."""
            lines = []

            # Title
            lines.append(("bold", f"\n{self.title}\n"))
            lines.append(("fg:ansibrightblack", f"Path: {self.current_path}\n\n"))

            if self.loading:
                lines.append(("fg:ansiyellow", "Loading...\n"))
            elif self.error_message:
                lines.append(("fg:ansired", f"Error: {self.error_message}\n"))
            elif not self.entries:
                lines.append(("fg:ansibrightblack", "(empty directory)\n"))
            else:
                # Show entries (max 15 visible)
                visible_start = max(0, self.selected_index - 7)
                visible_end = min(len(self.entries), visible_start + 15)

                if visible_start > 0:
                    lines.append(
                        ("fg:ansibrightblack", f"  ... {visible_start} more above\n")
                    )

                for i in range(visible_start, visible_end):
                    entry = self.entries[i]
                    selected = i == self.selected_index

                    if entry.is_directory:
                        if selected:
                            lines.append(("bold fg:ansicyan", f"> {entry.name}/\n"))
                        else:
                            lines.append(("fg:ansicyan", f"  {entry.name}/\n"))
                    else:
                        size_str = _format_size(entry.size)
                        if selected:
                            lines.append(("bold fg:ansiwhite", f"> {entry.name}"))
                            lines.append(("fg:ansibrightblack", f"  ({size_str})\n"))
                        else:
                            lines.append(("", f"  {entry.name}"))
                            lines.append(("fg:ansibrightblack", f"  ({size_str})\n"))

                remaining = len(self.entries) - visible_end
                if remaining > 0:
                    lines.append(
                        ("fg:ansibrightblack", f"  ... {remaining} more below\n")
                    )

            # Help text
            lines.append(("", "\n"))
            lines.append(("fg:ansibrightblack", "["))
            lines.append(("bold fg:ansiyellow", "↑/↓"))
            lines.append(("fg:ansibrightblack", "] Navigate  ["))
            lines.append(("bold fg:ansigreen", "Enter"))
            lines.append(("fg:ansibrightblack", "] Select  ["))
            lines.append(("bold fg:ansigreen", "d"))
            lines.append(("fg:ansibrightblack", "] Select Folder  ["))
            lines.append(("bold fg:ansiyellow", "←"))
            lines.append(("fg:ansibrightblack", "] Back  ["))
            lines.append(("bold fg:ansiyellow", "Esc"))
            lines.append(("fg:ansibrightblack", "] Cancel\n"))

            if self.file_filters:
                filter_str = ", ".join(f"*{ext}" for ext in self.file_filters)
                lines.append(("fg:ansibrightblack", f"Filter: {filter_str}\n"))

            return FormattedText(lines)

        # Create layout once
        layout = Layout(Window(content=FormattedTextControl(get_formatted_text)))

        # Create app with erase_when_done=False to prevent screen clearing between runs
        app = Application(
            layout=layout,
            key_bindings=kb,
            full_screen=True,
            mouse_support=False,
            erase_when_done=False,  # Keep content on screen between runs
        )

        # Run app loop - stay in fullscreen mode until done
        while stay_in_app[0] and not self.cancelled and not self.selected_path:

            def run_app():
                app.run()

            # Run prompt_toolkit in thread to avoid event loop conflicts
            try:
                loop = asyncio.get_running_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    await loop.run_in_executor(executor, run_app)
            except RuntimeError:
                app.run()

            # Handle pending refresh (fetch new directory contents)
            if pending_refresh[0]:
                pending_refresh[0] = False
                await self._refresh_listing()

        return self.selected_path

    async def _run_simple(self) -> Optional[str]:
        """Run simple numbered selection for non-interactive terminals."""
        await self._refresh_listing()

        while not self.cancelled and not self.selected_path:
            print(f"\nPath: {self.current_path}")

            if self.error_message:
                print(f"Error: {self.error_message}")
            elif not self.entries:
                print("(empty directory)")
            else:
                for i, entry in enumerate(self.entries):
                    if entry.is_directory:
                        print(f"  [{i + 1}] {entry.name}/")
                    else:
                        size_str = _format_size(entry.size)
                        print(f"  [{i + 1}] {entry.name} ({size_str})")

            print("\nEnter number to select, '..' to go up, or 'q' to cancel:")

            try:
                choice = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                self.cancelled = True
                break

            if choice.lower() == "q":
                self.cancelled = True
                break
            elif choice == "..":
                if self._navigate_up():
                    await self._refresh_listing()
            else:
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(self.entries):
                        entry = self.entries[idx]
                        if entry.is_directory:
                            self._navigate_into(entry, index=idx)
                            await self._refresh_listing()
                        else:
                            self._select_file(entry)
                    else:
                        print("Invalid number")
                except ValueError:
                    print("Invalid input")

        return self.selected_path


async def browse_for_file(
    send_message: Callable[[str], None],
    receive_response: Callable[[], Any],
    start_path: str = "/",
    file_filter: Optional[str] = None,
    title: str = "Select a file:",
) -> Optional[str]:
    """Convenience function to browse and select a file.

    Args:
        send_message: Function to send messages to worker.
        receive_response: Async function to receive FS responses.
        start_path: Starting directory path.
        file_filter: Optional file extension filter.
        title: Title for the browser.

    Returns:
        Selected file path, or None if cancelled.
    """
    browser = DirectoryBrowser(
        send_message=send_message,
        receive_response=receive_response,
        start_path=start_path,
        file_filter=file_filter,
        title=title,
    )
    return await browser.run()
