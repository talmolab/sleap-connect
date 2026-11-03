"""GitHub OAuth flow for sleap-connect CLI.

Implements dashboard-based OAuth flow:
1. Open browser to dashboard with CLI mode flag
2. User completes OAuth in dashboard
3. Dashboard deposits JWT to signaling server
4. CLI polls signaling server until JWT available
"""

import os
import secrets
import time
import webbrowser

import rich_click as click
import requests
from loguru import logger

from sleap_connect.config import get_config

# Default dashboard URL (GitHub Pages)
DEFAULT_DASHBOARD_URL = "https://talmolab.github.io/sleap-connect/dashboard/"

# Polling settings
POLL_INTERVAL = 2  # seconds


def get_dashboard_url() -> str:
    """Get dashboard URL, allowing env override for development.

    Returns:
        Dashboard URL string.
    """
    return os.environ.get("SLEAP_DASHBOARD_URL", DEFAULT_DASHBOARD_URL)


def github_login(
    timeout: int = 120,
    on_url_ready: "Callable[[str], None] | None" = None,
    on_progress: "Callable[[int], None] | None" = None,
    silent: bool = False,
) -> dict:
    """Perform GitHub OAuth login via dashboard.

    Opens browser to dashboard which handles the OAuth flow. CLI polls
    the signaling server until the JWT is deposited by the dashboard.

    Args:
        timeout: Maximum seconds to wait for login (default: 120).
        on_url_ready: Optional callback called with the login URL when ready.
            If provided, browser is NOT opened automatically.
        on_progress: Optional callback called with remaining seconds during polling.
        silent: If True, suppress CLI output (for API use).

    Returns:
        Dictionary with jwt and user info.

    Raises:
        RuntimeError: If login fails or times out.
    """
    from typing import Callable  # Local import for type hint

    config = get_config()

    # Generate cryptographic state token
    state = secrets.token_urlsafe(32)

    # Build dashboard URL with CLI mode flag
    dashboard_url = get_dashboard_url().rstrip("/")
    login_url = f"{dashboard_url}?cli=true&cli_state={state}"

    # Notify caller of URL or print/open browser
    if on_url_ready is not None:
        on_url_ready(login_url)
    else:
        # Print URL for headless environments AND try to open browser
        if not silent:
            click.echo(f"\nOpen this URL to login:\n{login_url}\n")
        try:
            webbrowser.open(login_url)
        except Exception:
            pass  # Browser open is best-effort

    # Poll signaling server for token
    server_url = config.get_http_url()
    poll_url = f"{server_url}/api/auth/cli/poll"
    start_time = time.time()

    while time.time() - start_time < timeout:
        remaining = int(timeout - (time.time() - start_time))
        if on_progress is not None:
            on_progress(remaining)
        elif not silent:
            click.echo(f"\rWaiting for login... ({remaining}s) ", nl=False)

        try:
            response = requests.get(poll_url, params={"state": state}, timeout=10)

            if response.status_code == 200:
                if not silent:
                    click.echo("\rLogin successful!                    ")
                data = response.json()

                if "jwt" not in data or "user" not in data:
                    raise RuntimeError("Invalid response from server")

                return data

            # 202 = pending, keep polling
            # 404 = expired/not found, keep polling (might not be deposited yet)

        except requests.RequestException as e:
            logger.debug(f"Poll request failed: {e}")
            # Network error, keep trying

        time.sleep(POLL_INTERVAL)

    if not silent:
        click.echo("\r                                     ")
    raise RuntimeError(f"Login timed out after {timeout} seconds. Please try again.")
