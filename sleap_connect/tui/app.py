"""Main TUI application for sleap-connect.

This module provides the TUIApp class that manages the overall TUI experience,
including login flow, room selection, and file browsing.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from textual.app import App


def _configure_tui_logging():
    """Configure logging to write to file instead of stderr.

    Textual apps need logging redirected to a file, otherwise log messages
    will corrupt the terminal UI.
    """
    # Remove any existing handlers from root logger
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Set up file handler
    log_file = Path.home() / ".sleap-connect" / "tui.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    root.addHandler(file_handler)
    root.setLevel(logging.DEBUG)

    # Also silence some noisy libraries
    logging.getLogger("aiortc").setLevel(logging.WARNING)
    logging.getLogger("aioice").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)


from textual.binding import Binding

from sleap_connect.tui.screens.login import LoginScreen
from sleap_connect.tui.screens.room_select import RoomSelectScreen
from sleap_connect.tui.screens.browser import BrowserScreen


class TUIApp(App):
    """Main TUI application for sleap-connect file browser.

    This app manages:
    - Login flow (if not authenticated)
    - Room selection (if logged in)
    - File browser with Miller columns
    - SLP path resolution
    """

    TITLE = "sleap-connect"
    SUB_TITLE = "File Browser"

    CSS = """
    /* Basic styling */
    .status-connected {
        color: $success;
    }

    .status-disconnected {
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("?", "help", "Help"),
    ]

    def __init__(
        self,
        room_id: Optional[str] = None,
        token: Optional[str] = None,
        room_secret: Optional[str] = None,
    ):
        """Initialize the TUI app.

        Args:
            room_id: Optional room ID (bypasses room selection).
            token: Optional room token (not required for JWT auth, but kept for compatibility).
            room_secret: Optional room secret for PSK authentication (CLI override).
        """
        super().__init__()
        self.room_id = room_id
        self.token = token
        self.room_secret = room_secret

        # Will be set during connection
        self.bridge = None

    def on_mount(self) -> None:
        """Called when app is mounted. Determines initial screen based on state."""
        # Check if room ID provided directly (direct mode)
        if self.room_id:
            # Skip login/room selection, go straight to browser
            # Token is optional for JWT auth (server validates via membership)
            self._show_browser(self.room_id, self.token or "", self.room_secret)
            return

        # Check if user is logged in
        from sleap_connect.auth.credentials import is_logged_in

        if is_logged_in():
            # Show room selection
            self._show_room_select()
        else:
            # Show login screen
            self._show_login()

    def _show_login(self) -> None:
        """Show the login screen."""

        def on_login_success(jwt: str, user: dict) -> None:
            """Called when login succeeds."""
            self.notify(f"Logged in as {user.get('username', 'unknown')}")
            # After login, switch to room selection
            # Use switch_screen for clean transition (pops current, pushes new)
            self.switch_screen(
                RoomSelectScreen(on_room_selected=self._on_room_selected)
            )

        login_screen = LoginScreen(on_login_success=on_login_success)
        self.push_screen(login_screen)

    def _on_room_selected(self, room_data: dict) -> None:
        """Handle room selection from RoomSelectScreen."""
        room_id = room_data.get("room_id")
        token = room_data.get("token", "")

        # Resolve room_secret from credentials/env/filesystem
        from sleap_connect.auth.secret_resolver import resolve_secret

        room_secret = resolve_secret(room_id)

        self._show_browser(room_id, token, room_secret=room_secret)

    def _show_room_select(self) -> None:
        """Show the room selection screen."""
        room_screen = RoomSelectScreen(on_room_selected=self._on_room_selected)
        self.push_screen(room_screen)

    def _show_browser(
        self,
        room_id: str,
        token: str,
        room_secret: Optional[str] = None,
    ) -> None:
        """Show the file browser screen.

        Args:
            room_id: Room ID to browse.
            token: Room token for authentication.
            room_secret: Optional room secret for PSK authentication (CLI override).
        """
        browser_screen = BrowserScreen(
            room_id=room_id,
            token=token,
            room_secret=room_secret,
        )
        self.push_screen(browser_screen)

    def action_help(self) -> None:
        """Show help information."""
        self.notify(
            "Navigation: Arrow keys | Enter: Select | q: Quit | ?: Help",
            timeout=5,
        )


def run_tui(
    room_id: Optional[str] = None,
    token: Optional[str] = None,
    room_secret: Optional[str] = None,
) -> None:
    """Run the TUI application.

    Args:
        room_id: Optional room ID to connect to directly.
        token: Optional room token (not required for JWT auth).
        room_secret: Optional room secret for PSK authentication (CLI override).
    """
    # Redirect logging to file to avoid corrupting TUI display
    _configure_tui_logging()

    app = TUIApp(room_id=room_id, token=token, room_secret=room_secret)
    app.run()
