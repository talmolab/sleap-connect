"""Login screen for TUI authentication.

This screen handles the GitHub OAuth login flow by displaying a URL
and polling for JWT completion.
"""

import asyncio
import os
import platform
import secrets
from typing import Optional

import requests
from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, Button
from textual.reactive import reactive

from sleap_connect.config import get_config

# Default dashboard URL (GitHub Pages)
DEFAULT_DASHBOARD_URL = "https://talmolab.github.io/sleap-connect/dashboard/"

# Polling settings
POLL_INTERVAL = 2  # seconds
DEFAULT_TIMEOUT = 120  # seconds


def get_dashboard_url() -> str:
    """Get dashboard URL, allowing env override for development."""
    return os.environ.get("SLEAP_DASHBOARD_URL", DEFAULT_DASHBOARD_URL)


class LoginScreen(Screen):
    """Screen for handling login via GitHub OAuth.

    Displays a URL for the user to open and polls for JWT completion.
    """

    CSS = """
    LoginScreen {
        background: $background;
    }

    #login-container {
        align: center middle;
        height: 100%;
    }

    #login-box {
        width: 70;
        height: auto;
        padding: 2 4;
        border: solid $primary;
    }

    .title {
        text-style: bold;
        text-align: center;
        color: $primary;
        margin-bottom: 1;
    }

    .subtitle {
        text-align: center;
        margin-bottom: 1;
    }

    #url-box {
        border: solid $primary-darken-2;
        padding: 1 2;
        margin: 1 0;
        text-align: center;
    }

    #url-text {
        color: $primary;
    }

    .hint {
        text-align: center;
        color: $text-muted;
        margin-top: 1;
    }

    #status {
        text-align: center;
        margin-top: 1;
    }

    #countdown {
        text-align: center;
        color: $warning;
    }

    #cancel-btn {
        margin-top: 1;
        width: 100%;
    }
    """

    BINDINGS = [
        ("q", "cancel", "Cancel"),
        ("escape", "cancel", "Cancel"),
    ]

    # Reactive properties
    remaining_time = reactive(DEFAULT_TIMEOUT, init=False)
    status_message = reactive("Waiting for login...", init=False)

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        on_login_success: Optional[callable] = None,
        name: Optional[str] = None,
    ):
        """Initialize login screen.

        Args:
            timeout: Maximum seconds to wait for login.
            on_login_success: Callback when login succeeds (receives jwt, user).
            name: Screen name.
        """
        super().__init__(name=name)
        self.timeout = timeout
        self.on_login_success = on_login_success

        # Generate state token for this login attempt
        self.state = secrets.token_urlsafe(32)

        # Build login URL
        dashboard_url = get_dashboard_url().rstrip("/")
        self.login_url = f"{dashboard_url}?cli=true&cli_state={self.state}"

        # Polling state
        self._polling = False
        self._poll_task: Optional[asyncio.Task] = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Vertical(
                Static("Login Required", classes="title"),
                Static("Open this URL in your browser to login:", classes="subtitle"),
                Container(
                    Static(self.login_url, id="url-text"),
                    id="url-box",
                ),
                Static(
                    "(The URL has been copied to your clipboard if supported)",
                    classes="hint",
                ),
                Static(self.status_message, id="status"),
                Static(f"Time remaining: {self.remaining_time}s", id="countdown"),
                Button("Cancel", id="cancel-btn", variant="default"),
                id="login-box",
            ),
            id="login-container",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Start polling when screen is mounted."""
        # Try to copy URL to clipboard
        try:
            import subprocess

            _sys = platform.system()
            if _sys == "Darwin":
                _clip_cmd = ["pbcopy"]
            elif _sys == "Windows":
                _clip_cmd = ["clip"]
            else:
                _clip_cmd = ["xclip", "-selection", "clipboard"]
            subprocess.run(
                _clip_cmd,
                input=self.login_url.encode(),
                check=False,
                capture_output=True,
            )
        except Exception:
            pass  # Clipboard copy is best-effort

        # Try to open browser
        try:
            import webbrowser

            webbrowser.open(self.login_url)
        except Exception:
            pass  # Browser open is best-effort

        # Start polling
        self._polling = True
        self._poll_task = asyncio.create_task(self._poll_for_jwt())

    def on_unmount(self) -> None:
        """Stop polling when screen is unmounted."""
        self._polling = False
        if self._poll_task:
            self._poll_task.cancel()

    def watch_remaining_time(self, time: int) -> None:
        """Update countdown display when time changes."""
        try:
            countdown = self.query_one("#countdown", Static)
            if time > 0:
                countdown.update(f"Time remaining: {time}s")
            else:
                countdown.update("Login timed out")
        except Exception:
            pass  # Widget not mounted yet

    def watch_status_message(self, message: str) -> None:
        """Update status display when message changes."""
        try:
            status = self.query_one("#status", Static)
            status.update(message)
        except Exception:
            pass  # Widget not mounted yet

    async def _poll_for_jwt(self) -> None:
        """Poll signaling server for JWT."""
        config = get_config()
        server_url = config.get_http_url()
        poll_url = f"{server_url}/api/auth/cli/poll"

        start_time = asyncio.get_event_loop().time()

        while self._polling and self.remaining_time > 0:
            # Update countdown
            elapsed = asyncio.get_event_loop().time() - start_time
            self.remaining_time = max(0, int(self.timeout - elapsed))

            try:
                # Use asyncio-friendly request
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: requests.get(
                        poll_url, params={"state": self.state}, timeout=5
                    ),
                )

                if response.status_code == 200:
                    data = response.json()
                    if "jwt" in data and "user" in data:
                        # Login successful!
                        self.status_message = "Login successful!"
                        self._polling = False

                        # Save credentials
                        from sleap_connect.auth.credentials import save_jwt

                        save_jwt(data["jwt"], data["user"])

                        # Notify success and let callback handle navigation
                        if self.on_login_success:
                            self.on_login_success(data["jwt"], data["user"])
                            # Don't pop here - callback handles navigation
                        else:
                            # No callback, just dismiss
                            self.app.pop_screen()
                        return

            except requests.RequestException:
                pass  # Network error, keep trying

            # Wait before next poll
            await asyncio.sleep(POLL_INTERVAL)

        # Timeout reached
        if self.remaining_time <= 0:
            self.status_message = "Login timed out. Press 'q' to cancel and try again."

    def action_cancel(self) -> None:
        """Cancel login and return to previous screen or quit."""
        self._polling = False
        self.app.pop_screen()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "cancel-btn":
            self.action_cancel()
