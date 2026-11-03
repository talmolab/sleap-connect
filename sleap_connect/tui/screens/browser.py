"""Browser screen for file navigation.

This screen provides the main file browsing interface using Miller columns
or tree view, and connects to workers via WebRTC.
"""

import asyncio
import os
import platform
import subprocess
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import Header, Footer, Static
from textual.reactive import reactive
from textual.binding import Binding

from sleap_connect.tui.widgets.miller import MillerColumns, FileEntry
from sleap_connect.tui.widgets.tree_browser import TreeBrowser
from sleap_connect.tui.widgets.worker_tabs import WorkerTabs, WorkerInfo
from sleap_connect.tui.widgets.slp_panel import SLPContextPanel, SLPInfo, VideoInfo
from sleap_connect.tui.bridge import WebRTCBridge
from sleap_connect.tui.screens.resolve_confirm import ResolveConfirmScreen
from sleap_connect.tui.screens.secret_input import SecretInputScreen


class BrowserScreen(Screen):
    """Main file browser screen with Miller columns and worker tabs.

    This screen connects to workers via WebRTC and displays the filesystem
    using Miller columns navigation. Multiple workers can be accessed via tabs.
    """

    DEFAULT_CSS = """
    BrowserScreen {
        height: 100%;
    }

    /* ===== Header ===== */
    #app-header {
        width: 100%;
        height: 1;
        padding: 0 1;
        background: $panel;
    }

    #header-title {
        text-style: bold;
    }

    #header-room {
        color: $text-muted;
        margin-left: 2;
    }

    #header-user {
        dock: right;
        color: $primary;
        margin-right: 2;
    }

    #header-status {
        dock: right;
    }

    #header-status.connected {
        color: $success;
    }

    /* ===== Main Layout ===== */
    #browser-main {
        height: 1fr;
    }

    #main-layout {
        height: 100%;
        width: 100%;
    }

    /* ===== Worker Tabs ===== */
    WorkerTabs {
        height: 3;
    }

    /* ===== Content Area ===== */
    #content-area {
        height: 100%;
        width: 1fr;
    }

    /* Path breadcrumb bar */
    #path-breadcrumb {
        height: 1;
        padding: 0 1;
    }

    #browser-content {
        height: 1fr;
    }

    #miller-container {
        height: 1fr;
        padding: 1;
    }

    #tree-container {
        height: 1fr;
        padding: 1;
    }

    /* ===== Connecting Overlay ===== */
    #connecting-overlay {
        align: center middle;
        width: 100%;
        height: 100%;
    }

    #connecting-box {
        width: 50;
        height: auto;
        padding: 2 4;
        border: solid $primary;
        text-align: center;
    }

    #connecting-box .title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }

    #connecting-status {
        color: $text-muted;
    }

    /* ===== SLP Panel ===== */
    SLPContextPanel {
        dock: bottom;
    }

    /* Legacy status bar (hidden in new design) */
    #status-bar {
        display: none;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("b", "back", "Back"),
        Binding("r", "refresh", "Refresh"),
        Binding("v", "toggle_view", "Toggle View"),
        Binding("y", "copy_path", "Copy Path"),
        Binding("Y", "copy_dir_path", "Copy Dir", show=False),
        Binding("f", "fix_video", "Fix Video"),
        Binding("s", "save_slp", "Save SLP"),
        Binding("c", "close_panel", "Close Panel"),
        Binding("escape", "back", "Back", show=False),
        # Number keys for worker tabs (handled by WorkerTabs widget)
        Binding("1", "select_worker_1", "Worker 1", show=False),
        Binding("2", "select_worker_2", "Worker 2", show=False),
        Binding("3", "select_worker_3", "Worker 3", show=False),
        Binding("4", "select_worker_4", "Worker 4", show=False),
        Binding("5", "select_worker_5", "Worker 5", show=False),
        Binding("tab", "next_worker", "Next Worker", show=False),
        Binding("shift+tab", "prev_worker", "Prev Worker", show=False),
    ]

    # Reactive properties
    connection_status = reactive("Disconnected")
    current_worker_name = reactive("")
    view_mode = reactive("miller")  # "miller" or "tree"

    def __init__(
        self,
        room_id: str,
        token: str,
        name: Optional[str] = None,
        room_secret: Optional[str] = None,
    ):
        """Initialize browser screen.

        Args:
            room_id: Room ID to connect to.
            token: Room token for authentication.
            name: Screen name.
            room_secret: Optional room secret for PSK authentication (CLI override).
        """
        super().__init__(name=name)
        self.room_id = room_id
        self.token = token
        self.room_secret = room_secret

        # WebRTC bridge (one per worker connection)
        self.bridge: Optional[WebRTCBridge] = None

        # Available workers (raw data from discovery)
        self.workers_data: list[dict] = []

        # WorkerInfo objects for tabs
        self.worker_infos: list[WorkerInfo] = []

        # Current worker index
        self.current_worker_index: int = 0

        # Connection state
        self._connecting = False
        self._connected = False

        # Track which views have been loaded
        self._miller_loaded = False
        self._tree_loaded = False

    def compose(self) -> ComposeResult:
        # Get user info for profile display
        from sleap_connect.auth.credentials import get_user

        user = get_user()
        username = user.get("username", "unknown") if user else "unknown"

        # Custom header
        with Horizontal(id="app-header"):
            yield Static("sleap-connect", id="header-title")
            yield Static(
                (
                    f"[{self.room_id[:12]}...]"
                    if len(self.room_id) > 15
                    else f"[{self.room_id}]"
                ),
                id="header-room",
            )
            yield Static(f"👤 {username}", id="header-user")
            yield Static("○ Disconnected", id="header-status")

        with Container(id="browser-main"):
            # Connecting overlay (shown while connecting)
            with Container(id="connecting-overlay"):
                with Vertical(id="connecting-box"):
                    yield Static("Connecting to Room...", classes="title")
                    yield Static("Please wait...", id="connecting-status")

            # Main layout
            with Vertical(id="main-layout"):
                # Worker tabs at top
                yield WorkerTabs(id="worker-tabs")

                # Content area below
                with Vertical(id="content-area"):
                    # Path breadcrumb
                    yield Static("/", id="path-breadcrumb")

                    # Browser content
                    with Vertical(id="browser-content"):
                        # Miller columns (default view)
                        with Container(id="miller-container"):
                            yield MillerColumns(
                                fetch_directory=self._fetch_directory,
                                id="miller-columns",
                            )

                        # Tree browser (alternate view, hidden by default)
                        with Container(id="tree-container"):
                            yield TreeBrowser(
                                fetch_directory=self._fetch_directory,
                                id="tree-browser",
                            )

                    # SLP context panel (hidden by default, docked to bottom)
                    yield SLPContextPanel(id="slp-panel")

        # Footer with keybindings
        yield Footer()

    def on_mount(self):
        """Start connecting when screen mounts."""
        # Hide main layout until connected
        self.query_one("#main-layout").display = False
        self.query_one("#connecting-overlay").display = True

        # Hide tree browser initially (miller is default)
        self.query_one("#tree-container").display = False

        # Start connection process
        asyncio.create_task(self._connect())

    def watch_view_mode(self, mode: str):
        """Update view when mode changes."""
        if not self.is_mounted:
            return
        try:
            miller_container = self.query_one("#miller-container")
            tree_container = self.query_one("#tree-container")

            if mode == "miller":
                miller_container.display = True
                tree_container.display = False
                # Focus miller columns
                self.query_one("#miller-columns", MillerColumns).focus()
            else:
                miller_container.display = False
                tree_container.display = True
                # Focus tree browser
                self.query_one("#tree-browser", TreeBrowser).focus()
        except Exception:
            pass

    def on_unmount(self):
        """Disconnect when screen unmounts."""
        if self.bridge:
            asyncio.create_task(self.bridge.disconnect())

    def watch_connection_status(self, status: str):
        """Update connection status display in header."""
        if not self.is_mounted:
            return
        try:
            header_status = self.query_one("#header-status", Static)
            if status == "Connected":
                header_status.update("● Connected")
                header_status.add_class("connected")
            else:
                header_status.update(f"○ {status}")
                header_status.remove_class("connected")
        except Exception:
            pass

    def watch_current_worker_name(self, worker: str):
        """Update worker info display."""
        # Worker name is now shown in the sidebar, not in the header
        pass

    async def _connect(self):
        """Connect to signaling server and discover workers."""
        self._connecting = True
        self.connection_status = "Connecting..."

        try:
            # Create bridge
            self.bridge = WebRTCBridge(
                room_id=self.room_id,
                token=self.token,
                on_connected=self._on_connected,
                on_disconnected=self._on_disconnected,
                on_auth_status=self._on_auth_status,
                room_secret=self.room_secret,
            )

            # Connect to signaling server
            connecting_status = self.query_one("#connecting-status", Static)
            connecting_status.update("Connecting to signaling server...")

            if not await self.bridge.connect_signaling():
                self.connection_status = "Failed to connect"
                connecting_status.update("Failed to connect to signaling server")
                return

            # Discover workers
            connecting_status.update("Discovering workers...")
            self.workers_data = await self.bridge.discover_workers()

            if not self.workers_data:
                self.connection_status = "No workers found"
                connecting_status.update("No workers found in room")
                return

            # Convert to WorkerInfo objects
            self.worker_infos = [WorkerInfo.from_dict(w) for w in self.workers_data]

            # Update worker tabs
            worker_tabs = self.query_one("#worker-tabs", WorkerTabs)
            worker_tabs.set_workers(self.worker_infos)

            # Connect to first worker
            await self._connect_to_worker(0)

        except Exception as e:
            self.connection_status = f"Error: {e}"
            self.query_one("#connecting-status", Static).update(f"Error: {e}")

    async def _connect_to_worker(self, index: int):
        """Connect to a worker by index.

        Args:
            index: Index of the worker in workers_data list.
        """
        if index < 0 or index >= len(self.workers_data):
            return

        worker = self.workers_data[index]
        worker_info = self.worker_infos[index]
        worker_id = worker.get("peer_id")

        if not worker_id:
            return

        # Disconnect from current worker if connected
        if self._connected and self.bridge:
            await self.bridge.disconnect()
            self._connected = False

            # Recreate bridge for new connection
            self.bridge = WebRTCBridge(
                room_id=self.room_id,
                token=self.token,
                on_connected=self._on_connected,
                on_disconnected=self._on_disconnected,
                on_auth_status=self._on_auth_status,
                room_secret=self.room_secret,
            )
            await self.bridge.connect_signaling()

        connecting_status = self.query_one("#connecting-status", Static)
        connecting_status.update(f"Connecting to {worker_info.display_name}...")

        if await self.bridge.connect(worker_id):
            self._connected = True
            self.current_worker_index = index
            self.connection_status = "Connected"
            self.current_worker_name = worker_info.display_name

            # Update tab connection status
            worker_tabs = self.query_one("#worker-tabs", WorkerTabs)
            worker_tabs.set_worker_connected(index, True)

            # Switch to browser view
            self.query_one("#connecting-overlay").display = False
            self.query_one("#main-layout").display = True

            # Focus miller columns
            miller = self.query_one("#miller-columns", MillerColumns)
            miller.focus()

            # Wait a moment for connection to stabilize, then load root
            self.set_timer(0.5, self._try_load_root)

        else:
            # Check if authentication failed
            if self.bridge and self.bridge.auth_failed_reason:
                reason = self.bridge.auth_failed_reason
                self.connection_status = f"Auth failed: {reason}"
                connecting_status.update(f"Authentication failed: {reason}")

                # Check if we need to prompt for room secret
                if "No room secret configured" in reason:
                    # Show secret input screen
                    await self._prompt_for_secret(index, reason)
                    return
                else:
                    # Show helpful error message for other auth failures
                    self.notify(
                        f"PSK authentication failed: {reason}\n"
                        "Check that your room secret matches the worker's secret.\n"
                        "Use 'sleap-connect room create-secret --room ROOM' to generate a new secret.",
                        severity="error",
                        timeout=15,
                    )
            else:
                self.connection_status = "Connection failed"
                connecting_status.update(
                    f"Failed to connect to {worker_info.display_name}"
                )

            # Update tab connection status
            worker_tabs = self.query_one("#worker-tabs", WorkerTabs)
            worker_tabs.set_worker_connected(index, False)

    async def _prompt_for_secret(self, worker_index: int, error_message: str = None):
        """Prompt user for room secret and retry connection.

        Args:
            worker_index: Index of worker to connect to after getting secret.
            error_message: Optional error message to display.
        """

        def on_secret_result(secret: Optional[str]) -> None:
            """Handle secret input result."""
            if secret:
                # Update room_secret and retry connection
                self.room_secret = secret
                asyncio.create_task(
                    self._retry_connection_with_secret(worker_index, secret)
                )
            else:
                # User cancelled - update status
                self.connection_status = "Authentication cancelled"
                connecting_status = self.query_one("#connecting-status", Static)
                connecting_status.update(
                    "Authentication cancelled - no secret provided"
                )

        # Show secret input screen
        secret_screen = SecretInputScreen(
            room_id=self.room_id,
            error_message=(
                error_message if "No room secret" not in (error_message or "") else None
            ),
        )
        self.app.push_screen(secret_screen, on_secret_result)

    async def _retry_connection_with_secret(self, worker_index: int, secret: str):
        """Retry connection to worker with provided secret.

        Args:
            worker_index: Index of worker to connect to.
            secret: Room secret to use for authentication.
        """
        worker = self.workers_data[worker_index]
        worker_info = self.worker_infos[worker_index]
        worker_id = worker.get("peer_id")

        # Disconnect and recreate bridge with new secret
        if self.bridge:
            await self.bridge.disconnect()

        self.bridge = WebRTCBridge(
            room_id=self.room_id,
            token=self.token,
            on_connected=self._on_connected,
            on_disconnected=self._on_disconnected,
            on_auth_status=self._on_auth_status,
            room_secret=secret,
        )

        connecting_status = self.query_one("#connecting-status", Static)
        connecting_status.update("Reconnecting to signaling server...")

        if not await self.bridge.connect_signaling():
            self.connection_status = "Failed to reconnect"
            connecting_status.update("Failed to reconnect to signaling server")
            return

        connecting_status.update(f"Connecting to {worker_info.display_name}...")

        if await self.bridge.connect(worker_id):
            self._connected = True
            self.current_worker_index = worker_index
            self.connection_status = "Connected"
            self.current_worker_name = worker_info.display_name

            # Save secret to credentials for future use
            self._save_room_secret(secret)

            # Update tab connection status
            worker_tabs = self.query_one("#worker-tabs", WorkerTabs)
            worker_tabs.set_worker_connected(worker_index, True)

            # Switch to browser view
            self.query_one("#connecting-overlay").display = False
            self.query_one("#main-layout").display = True

            # Focus miller columns and load root
            miller = self.query_one("#miller-columns", MillerColumns)
            miller.focus()
            self.set_timer(0.5, self._try_load_root)

        else:
            # Still failed - check reason
            if self.bridge and self.bridge.auth_failed_reason:
                reason = self.bridge.auth_failed_reason
                # Show secret input again with error
                await self._prompt_for_secret(worker_index, reason)
            else:
                self.connection_status = "Connection failed"
                connecting_status.update(
                    f"Failed to connect to {worker_info.display_name}"
                )

    def _save_room_secret(self, secret: str):
        """Save room secret to credentials for future use.

        Args:
            secret: Room secret to save.
        """
        try:
            from sleap_connect.auth.credentials import save_room_secret

            save_room_secret(self.room_id, secret)
            self.app.notify(
                f"Room secret saved for future connections",
                severity="information",
                timeout=3,
            )
        except Exception as e:
            # Non-fatal - just notify
            self.app.notify(
                f"Could not save room secret: {e}",
                severity="warning",
                timeout=3,
            )

    async def _switch_to_worker(self, index: int):
        """Switch to a different worker.

        Args:
            index: Worker index to switch to.
        """
        if index == self.current_worker_index:
            return

        if index < 0 or index >= len(self.workers_data):
            return

        # Show connecting overlay while switching
        self.query_one("#connecting-overlay").display = True
        self.query_one("#main-layout").display = False

        # Update tab active state
        worker_tabs = self.query_one("#worker-tabs", WorkerTabs)
        worker_tabs.active_index = index

        # Connect to new worker
        await self._connect_to_worker(index)

    async def _fetch_directory(self, path: str, offset: int = 0) -> Optional[dict]:
        """Fetch directory contents from worker.

        Args:
            path: Directory path to list.
            offset: Pagination offset.

        Returns:
            Dict with entries, total_count, has_more, or None on error.
        """
        if not self.bridge:
            return {"error": "No bridge connection"}

        if not self.bridge.is_connected:
            return {"error": "Data channel not connected"}

        # Handle root path specially - fetch mounts
        if path == "/":
            mounts = await self.bridge.get_mounts()
            if mounts:
                return {
                    "path": "/",
                    "entries": [
                        {
                            "name": m.get("label", m.get("path")),
                            "type": "directory",
                            "size": 0,
                            # Store the actual path for navigation
                            "actual_path": m.get("path"),
                        }
                        for m in mounts
                    ],
                    "total_count": len(mounts),
                    "has_more": False,
                }
            return {"error": "No mounts returned (timeout or auth required)"}

        result = await self.bridge.list_dir(path, offset)
        if result is None:
            return {"error": "Failed to list directory (timeout after 60s)"}

        return result

    def _try_load_root(self):
        """Try to load root directory."""
        if not self.bridge:
            return

        # Only load the active view to avoid race conditions
        if self.view_mode == "miller":
            miller = self.query_one("#miller-columns", MillerColumns)
            miller.load_root()
            self._miller_loaded = True
        else:
            tree = self.query_one("#tree-browser", TreeBrowser)
            tree.load_root()
            self._tree_loaded = True

    def _on_connected(self):
        """Called when WebRTC data channel opens."""
        self.connection_status = "Connected"

    def _on_disconnected(self):
        """Called when connection is lost."""
        self.connection_status = "Disconnected"
        self._connected = False

        # Update tab status (widget may not be mounted)
        try:
            worker_tabs = self.query_one("#worker-tabs", WorkerTabs)
            worker_tabs.set_worker_connected(self.current_worker_index, False)
        except Exception:
            pass

        self.notify("Connection to worker lost", severity="warning")

    def _on_auth_status(self, status: str):
        """Called when PSK authentication status changes.

        Args:
            status: Status message (e.g., "Authenticating...", "Authenticated", "Auth failed: reason")
        """
        # Update the connecting status display
        try:
            connecting_status = self.query_one("#connecting-status", Static)
            connecting_status.update(status)
        except Exception:
            pass

        # Also update header status during auth
        if status == "Authenticated":
            self.connection_status = "Connected"
        elif status.startswith("Auth failed"):
            self.connection_status = status
        else:
            self.connection_status = status

    def on_worker_tabs_worker_selected(self, event: WorkerTabs.WorkerSelected):
        """Handle worker selection from sidebar."""
        asyncio.create_task(self._switch_to_worker(event.index))

    def on_miller_columns_path_changed(self, event: MillerColumns.PathChanged):
        """Handle path changes in Miller columns."""
        # Update path breadcrumb
        self._update_path_breadcrumb(event.path)
        # Don't hide SLP panel - user may be browsing to find a video file to fix

    def on_tree_browser_path_changed(self, event: TreeBrowser.PathChanged):
        """Handle path changes in tree browser."""
        # Update path breadcrumb
        self._update_path_breadcrumb(event.path)
        # Don't hide SLP panel - user may be browsing to find a video file to fix

    def _update_path_breadcrumb(self, path: str):
        """Update the path breadcrumb display.

        Args:
            path: Current path to display.
        """
        if not self.is_mounted:
            return
        try:
            breadcrumb = self.query_one("#path-breadcrumb", Static)
            breadcrumb.update(path)
        except Exception:
            pass

    def on_tree_browser_file_selected(self, event: TreeBrowser.FileSelected):
        """Handle file selection in tree browser."""
        entry = event.entry
        if entry.name.endswith(".slp"):
            asyncio.create_task(self._check_slp_videos(entry.path))
        # Don't hide panel for non-SLP files - user may be selecting a video to fix

    def on_slp_context_panel_fix_requested(self, event: SLPContextPanel.FixRequested):
        """Handle fix request for a missing video."""
        video = event.video
        slp_info = event.slp_info

        # Get the currently selected file from active view
        if self.view_mode == "miller":
            miller = self.query_one("#miller-columns", MillerColumns)
            selected = miller.get_selected_entry()
        else:
            tree = self.query_one("#tree-browser", TreeBrowser)
            selected = tree.get_selected_entry()

        if not selected:
            self.notify("Select a video file to use as the fix", severity="warning")
            return

        # Use the selected file's path
        new_path = selected.path

        # Get other missing videos for batch resolution
        other_missing = [
            v.original_path
            for v in slp_info.videos
            if v.is_missing and v.original_path != video.original_path
        ]

        asyncio.create_task(
            self._resolve_video_path(video, slp_info, new_path, other_missing)
        )

    async def _resolve_video_path(
        self,
        video: VideoInfo,
        slp_info: SLPInfo,
        new_path: str,
        other_missing: list[str],
    ):
        """Resolve a missing video path using prefix matching.

        Shows a confirmation dialog, then marks videos as resolved_pending
        in memory. User must press 's' to save.

        Args:
            video: The missing video to fix.
            slp_info: SLP file information.
            new_path: The new path selected by the user.
            other_missing: List of other missing video paths.
        """
        if not self.bridge or not self.bridge.is_connected:
            self.notify("Not connected to worker", severity="warning")
            return

        self.notify(f"Computing prefix resolution...")

        # Compute prefix-based resolution
        result = await self.bridge.compute_prefix_resolution(
            original_path=video.original_path,
            new_path=new_path,
            other_missing=other_missing,
        )

        if result is None:
            self.notify("Failed to compute resolution", severity="error")
            return

        if result.get("error"):
            self.notify(f"Error: {result['error']}", severity="error")
            return

        # Extract resolution info
        old_prefix = result.get("old_prefix", "")
        new_prefix = result.get("new_prefix", "")
        additional_resolved = result.get("would_resolve", [])
        still_missing_paths = result.get("would_not_resolve", [])

        # Build the list of videos to resolve (including the selected one)
        videos_to_resolve = [{"original": video.original_path, "resolved": new_path}]
        videos_to_resolve.extend(additional_resolved)

        # Show confirmation dialog
        def on_confirm():
            # Apply resolutions when user confirms
            asyncio.create_task(
                self._apply_resolutions(
                    slp_info, videos_to_resolve, still_missing_paths
                )
            )

        confirm_screen = ResolveConfirmScreen(
            old_prefix=old_prefix,
            new_prefix=new_prefix,
            videos_to_resolve=videos_to_resolve,
            still_missing=still_missing_paths,
            on_confirm=on_confirm,
        )
        self.app.push_screen(confirm_screen)

    async def _apply_resolutions(
        self,
        slp_info: SLPInfo,
        videos_to_resolve: list[dict],
        still_missing: list[str],
    ):
        """Apply the confirmed video resolutions.

        Args:
            slp_info: SLP info to update.
            videos_to_resolve: List of {"original": ..., "resolved": ...} dicts.
            still_missing: List of paths that won't be resolved.
        """
        # Mark all videos as resolved
        for video in videos_to_resolve:
            slp_info.mark_video_resolved(video["original"], video["resolved"])

        # Refresh the SLP panel to show updated status
        slp_panel = self.query_one("#slp-panel", SLPContextPanel)
        slp_panel.refresh_display()

        # Show status message
        total_resolved = len(videos_to_resolve)
        if slp_info.missing == 0:
            self.notify(
                f"All {total_resolved} video(s) resolved! Press 's' to save new SLP.",
                severity="information",
            )
        else:
            self.notify(
                f"Resolved {total_resolved} video(s). {slp_info.missing} still missing.",
                severity="information",
            )

    def on_miller_columns_file_selected(self, event: MillerColumns.FileSelected):
        """Handle file selection in Miller columns."""
        entry = event.entry
        if entry.name.endswith(".slp"):
            # Use the entry's path directly
            asyncio.create_task(self._check_slp_videos(entry.path))
        # Don't hide panel for non-SLP files - user may be selecting a video to fix

    async def _check_slp_videos(self, slp_path: str):
        """Check video status for an SLP file and show panel.

        Args:
            slp_path: Full path to the SLP file.
        """
        # Reset save-with-missing flag when loading a new SLP
        self._save_with_missing = False

        if not self.bridge or not self.bridge.is_connected:
            self.notify("Not connected to worker", severity="warning")
            return

        self.notify(f"Checking videos in {slp_path.split('/')[-1]}...")

        # Check video status via WebRTC
        result = await self.bridge.check_slp_videos(slp_path)

        if result is None:
            self.notify("Failed to check video status", severity="error")
            return

        # Create SLPInfo from response
        slp_info = SLPInfo.from_check_response(slp_path, result)

        # Show the panel
        slp_panel = self.query_one("#slp-panel", SLPContextPanel)
        slp_panel.show(slp_info)

        # Notify user of status
        if slp_info.error:
            self.notify(f"Error: {slp_info.error}", severity="error")
        elif slp_info.missing > 0:
            self.notify(
                f"{slp_info.missing} video(s) missing - navigate to video, then press 'f'",
                severity="warning",
                timeout=5,
            )
        else:
            self.notify(
                f"All {slp_info.total_videos} videos found!", severity="information"
            )

    def action_quit(self):
        """Quit the application."""
        self.app.exit()

    def action_back(self):
        """Go back to room selection."""
        self.app.pop_screen()

    def action_refresh(self):
        """Refresh current directory in active view."""
        if self.view_mode == "miller":
            miller = self.query_one("#miller-columns", MillerColumns)
            miller.refresh_current()
        else:
            tree = self.query_one("#tree-browser", TreeBrowser)
            tree.refresh_current()

    def action_toggle_view(self):
        """Toggle between Miller columns and tree view."""
        if self.view_mode == "miller":
            self.view_mode = "tree"
            # Load tree view if not already loaded
            if not self._tree_loaded and self._connected:
                tree = self.query_one("#tree-browser", TreeBrowser)
                tree.load_root()
                self._tree_loaded = True
            self.notify("Switched to Tree View", timeout=2)
        else:
            self.view_mode = "miller"
            # Load miller view if not already loaded
            if not self._miller_loaded and self._connected:
                miller = self.query_one("#miller-columns", MillerColumns)
                miller.load_root()
                self._miller_loaded = True
            self.notify("Switched to Miller Columns", timeout=2)

    def action_fix_video(self):
        """Fix a missing video path using the currently selected file."""
        slp_panel = self.query_one("#slp-panel", SLPContextPanel)
        if not slp_panel.slp_info:
            self.notify(
                "No SLP file loaded - select an SLP file first", severity="warning"
            )
            return

        if slp_panel.slp_info.missing == 0:
            self.notify("No missing videos to fix", severity="information")
            return

        # Get the currently selected file from active view
        if self.view_mode == "miller":
            miller = self.query_one("#miller-columns", MillerColumns)
            selected = miller.get_selected_entry()
        else:
            tree = self.query_one("#tree-browser", TreeBrowser)
            selected = tree.get_selected_entry()

        if not selected:
            self.notify("Select a video file first, then press 'f'", severity="warning")
            return

        if selected.is_dir:
            self.notify("Select a video file, not a directory", severity="warning")
            return

        # Get the first missing video
        missing_video = next(
            (v for v in slp_panel.slp_info.videos if v.is_missing), None
        )
        if not missing_video:
            self.notify("No missing videos to fix", severity="information")
            return

        # Get other missing videos for batch resolution
        other_missing = [
            v.original_path
            for v in slp_panel.slp_info.videos
            if v.is_missing and v.original_path != missing_video.original_path
        ]

        asyncio.create_task(
            self._resolve_video_path(
                missing_video, slp_panel.slp_info, selected.path, other_missing
            )
        )

    def action_close_panel(self):
        """Close the SLP context panel."""
        slp_panel = self.query_one("#slp-panel", SLPContextPanel)
        slp_panel.hide()

    def action_copy_path(self):
        """Copy the currently selected file/directory path to clipboard."""
        # Get selected entry from active view
        if self.view_mode == "miller":
            miller = self.query_one("#miller-columns", MillerColumns)
            selected = miller.get_selected_entry()
        else:
            tree = self.query_one("#tree-browser", TreeBrowser)
            selected = tree.get_selected_entry()

        if not selected:
            # If nothing selected, copy current directory instead
            self.action_copy_dir_path()
            return

        path = selected.path
        if self._copy_to_clipboard(path):
            # Show truncated path in notification if too long
            display_path = path if len(path) < 50 else f"...{path[-47:]}"
            self.notify(f"Copied: {display_path}", severity="information")
        else:
            self.notify("Failed to copy to clipboard", severity="error")

    def action_copy_dir_path(self):
        """Copy the current directory path to clipboard."""
        # Get current directory from active view
        if self.view_mode == "miller":
            miller = self.query_one("#miller-columns", MillerColumns)
            path = miller.current_path
        else:
            tree = self.query_one("#tree-browser", TreeBrowser)
            path = tree.current_path

        if not path:
            self.notify("No directory path available", severity="warning")
            return

        if self._copy_to_clipboard(path):
            # Show truncated path in notification if too long
            display_path = path if len(path) < 50 else f"...{path[-47:]}"
            self.notify(f"Copied dir: {display_path}", severity="information")
        else:
            self.notify("Failed to copy to clipboard", severity="error")

    def _copy_to_clipboard(self, text: str) -> bool:
        """Copy text to system clipboard.

        Args:
            text: Text to copy.

        Returns:
            True if successful, False otherwise.
        """
        try:
            _sys = platform.system()
            if _sys == "Darwin":
                _clip_cmd = ["pbcopy"]
            elif _sys == "Windows":
                _clip_cmd = ["clip"]
            else:
                _clip_cmd = ["xclip", "-selection", "clipboard"]
            subprocess.run(
                _clip_cmd,
                input=text.encode(),
                check=True,
                capture_output=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return False

    def action_save_slp(self):
        """Save the SLP file with resolved video paths."""
        slp_panel = self.query_one("#slp-panel", SLPContextPanel)

        if not slp_panel.slp_info:
            self.notify("No SLP file loaded", severity="warning")
            return

        if not slp_panel.slp_info.has_pending_resolutions:
            self.notify("No pending resolutions to save", severity="information")
            return

        if slp_panel.slp_info.missing > 0:
            self.notify(
                f"Warning: {slp_panel.slp_info.missing} video(s) still missing. "
                "Fix all videos or press 's' again to save anyway.",
                severity="warning",
            )
            # Set a flag to allow saving with missing videos on second press
            if not hasattr(self, "_save_with_missing"):
                self._save_with_missing = True
                return
            self._save_with_missing = False

        asyncio.create_task(self._save_slp_with_resolutions(slp_panel.slp_info))

    async def _save_slp_with_resolutions(self, slp_info: SLPInfo):
        """Save SLP file with all pending video path resolutions.

        Args:
            slp_info: SLP info with pending resolutions.
        """
        if not self.bridge or not self.bridge.is_connected:
            self.notify("Not connected to worker", severity="warning")
            return

        # Get filename map from pending resolutions
        filename_map = slp_info.get_filename_map()

        if not filename_map:
            self.notify("No resolutions to save", severity="warning")
            return

        self.notify(f"Saving SLP with {len(filename_map)} resolved path(s)...")

        # Write the new SLP file
        result = await self.bridge.write_slp_with_new_paths(
            slp_path=slp_info.path,
            filename_map=filename_map,
        )

        if result and result.get("output_path"):
            output_path = result["output_path"]
            output_name = output_path.split("/")[-1]
            self.notify(
                f"Saved: {output_name}",
                severity="information",
            )

            # Refresh the SLP panel with the new file
            await self._check_slp_videos(output_path)
        else:
            error = result.get("error", "Unknown error") if result else "No response"
            self.notify(f"Failed to save: {error}", severity="error")

    # Worker selection actions (number keys)
    def action_select_worker_1(self):
        asyncio.create_task(self._switch_to_worker(0))

    def action_select_worker_2(self):
        asyncio.create_task(self._switch_to_worker(1))

    def action_select_worker_3(self):
        asyncio.create_task(self._switch_to_worker(2))

    def action_select_worker_4(self):
        asyncio.create_task(self._switch_to_worker(3))

    def action_select_worker_5(self):
        asyncio.create_task(self._switch_to_worker(4))

    def action_next_worker(self):
        """Switch to next worker."""
        if self.worker_infos:
            next_index = (self.current_worker_index + 1) % len(self.worker_infos)
            asyncio.create_task(self._switch_to_worker(next_index))

    def action_prev_worker(self):
        """Switch to previous worker."""
        if self.worker_infos:
            prev_index = (self.current_worker_index - 1) % len(self.worker_infos)
            asyncio.create_task(self._switch_to_worker(prev_index))
