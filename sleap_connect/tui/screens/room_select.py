"""Room selection screen for TUI.

This screen displays a list of rooms the user has access to and allows
them to select one to connect to.
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional, Callable

import requests
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, ListView, ListItem, Label
from textual.reactive import reactive

from sleap_connect.config import get_config
from sleap_connect.auth.credentials import get_user


def parse_expiration(expires_at) -> Optional[datetime]:
    """Parse expiration timestamp to datetime."""
    if not expires_at:
        return None
    if isinstance(expires_at, (int, float)):
        # Unix timestamp (seconds)
        if expires_at < 10000000000:
            return datetime.fromtimestamp(expires_at, tz=timezone.utc)
        return datetime.fromtimestamp(expires_at / 1000, tz=timezone.utc)
    if isinstance(expires_at, str):
        try:
            # Try ISO format
            return datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        except ValueError:
            pass
    return None


def format_days_remaining(expires_at) -> str:
    """Format expiration as days remaining."""
    if not expires_at:
        return "never"

    exp_dt = parse_expiration(expires_at)
    if not exp_dt:
        return "?"

    now = datetime.now(timezone.utc)
    delta = exp_dt - now
    days = delta.days

    if days < 0:
        return "expired"
    elif days == 0:
        return "<1d"
    else:
        return f"{days}d"


class RoomListItem(ListItem):
    """A list item representing a room with badge-style indicators."""

    # Reactive worker count: None = loading, -1 = error, >= 0 = count
    worker_count: reactive[Optional[int]] = reactive(None, init=False)

    def __init__(self, room_data: dict, *args, **kwargs):
        """Initialize room list item.

        Args:
            room_data: Dict with room_id, role, joined_at, expires_at, etc.
        """
        super().__init__(*args, **kwargs)
        self.room_data = room_data

    def compose(self) -> ComposeResult:
        role = self.room_data.get("role", "member")
        name = self.room_data.get("name") or self.room_data.get("room_id", "unknown")
        expires_at = self.room_data.get("expires_at")

        # Format expiration
        exp_str = format_days_remaining(expires_at)

        # Build the display with horizontal layout
        with Horizontal(classes="room-item-row"):
            # Role badge
            if role == "owner":
                yield Static("[OWNER]", classes="role-badge owner")
            else:
                yield Static("[MEMBER]", classes="role-badge member")

            # Room name
            yield Static(name, classes="room-name")

            # Worker indicator (updated reactively)
            yield Static(
                "— online", id=f"workers-{id(self)}", classes="worker-indicator"
            )

            # Expiration
            yield Static(f"⏱ {exp_str}", classes="expiration")

    def watch_worker_count(self, count: Optional[int]) -> None:
        """Update worker indicator when count changes."""
        try:
            indicator = self.query_one(f"#workers-{id(self)}", Static)
            if count is None:
                indicator.update("— online")
            elif count < 0:
                indicator.update("○ ? online")
            elif count == 0:
                indicator.update("○ 0 online")
            else:
                indicator.update(f"● {count} online")
        except Exception:
            pass  # Widget not mounted yet


class RoomSelectScreen(Screen):
    """Screen for selecting a room to connect to.

    Fetches rooms from the API and displays them in a list for selection.
    """

    CSS = """
    RoomSelectScreen {
        background: $background;
    }

    /* Custom header bar */
    #app-header {
        width: 100%;
        height: 1;
        padding: 0 1;
        background: $panel;
    }

    #header-title {
        text-style: bold;
    }

    #header-user {
        dock: right;
        color: $primary;
    }

    #room-container {
        align: center middle;
        height: 1fr;
    }

    #room-box {
        width: 60;
        height: auto;
        max-height: 80%;
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
        color: $text-muted;
        margin-bottom: 1;
    }

    #room-list {
        height: auto;
        max-height: 20;
        margin: 1 0;
        border: solid $primary-darken-2;
    }

    #status {
        text-align: center;
        color: $text-muted;
        margin-top: 1;
    }

    #error {
        text-align: center;
        color: $error;
        margin-top: 1;
    }

    .hint {
        text-align: center;
        color: $text-muted;
        margin-top: 1;
    }

    #no-rooms {
        text-align: center;
        padding: 2;
        color: $warning;
    }

    /* Room list item styling */
    .room-item-row {
        width: 100%;
        height: 1;
    }

    .role-badge {
        width: 8;
        text-style: bold;
    }

    .role-badge.owner {
        color: $primary;
    }

    .role-badge.member {
        color: $text-muted;
    }

    .room-name {
        width: 1fr;
        padding-left: 1;
    }

    .worker-indicator {
        width: 12;
        text-align: right;
        color: $text-muted;
    }

    .expiration {
        width: 10;
        text-align: right;
        color: $text-muted;
        padding-right: 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("escape", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
    ]

    # Reactive properties (init=False to prevent watchers firing before mount)
    loading = reactive(True, init=False)
    error_message = reactive("", init=False)

    def __init__(
        self,
        on_room_selected: Optional[Callable[[dict], None]] = None,
        name: Optional[str] = None,
    ):
        """Initialize room select screen.

        Args:
            on_room_selected: Callback when room is selected (receives room dict).
            name: Screen name.
        """
        super().__init__(name=name)
        self.on_room_selected = on_room_selected
        self.rooms: list[dict] = []

    def compose(self) -> ComposeResult:
        # Get user info for profile display
        user = get_user()
        username = user.get("username", "unknown") if user else "unknown"

        # Custom header with profile
        with Horizontal(id="app-header"):
            yield Static("sleap-connect", id="header-title")
            yield Static(f"👤 {username}", id="header-user")

        yield Container(
            Vertical(
                Static("Select a Room", classes="title"),
                Static("Choose a room to browse its workers", classes="subtitle"),
                ListView(id="room-list"),
                Static("Loading rooms...", id="status"),
                Static("", id="error"),
                Static(
                    "Press Enter to select, 'r' to refresh, 'q' to quit", classes="hint"
                ),
                id="room-box",
            ),
            id="room-container",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Fetch rooms when screen is mounted."""
        self.fetch_rooms()

    def watch_loading(self, loading: bool) -> None:
        """Update UI when loading state changes."""
        try:
            status = self.query_one("#status", Static)
            if loading:
                status.update("Loading rooms...")
                status.display = True
            else:
                status.display = False
        except Exception:
            pass  # Widget not mounted yet

    def watch_error_message(self, message: str) -> None:
        """Update error display."""
        try:
            error = self.query_one("#error", Static)
            if message:
                error.update(f"Error: {message}")
                error.display = True
            else:
                error.display = False
        except Exception:
            pass  # Widget not mounted yet

    def fetch_rooms(self) -> None:
        """Fetch rooms from API."""
        self.loading = True
        self.error_message = ""
        asyncio.create_task(self._fetch_rooms_async())

    async def _fetch_rooms_async(self) -> None:
        """Async room fetching."""
        from sleap_connect.auth.credentials import get_valid_jwt

        jwt_token = get_valid_jwt()
        if not jwt_token:
            self.loading = False
            self.error_message = (
                "Not logged in or token expired. Run: sleap-connect auth login"
            )
            return

        config = get_config()
        endpoint = f"{config.get_http_url()}/api/auth/rooms"

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: requests.get(
                    endpoint,
                    headers={"Authorization": f"Bearer {jwt_token}"},
                    timeout=30,
                ),
            )

            if response.status_code != 200:
                error = response.json().get("error", response.text)
                self.loading = False
                self.error_message = f"Failed to fetch rooms: {error}"
                return

            data = response.json()
            self.rooms = data.get("rooms", [])
            self.loading = False

            # Update the list view
            self._populate_room_list()

        except requests.RequestException as e:
            self.loading = False
            self.error_message = f"Request failed: {e}"

    def _populate_room_list(self) -> None:
        """Populate the room list with fetched rooms (filtered, non-expired only)."""
        room_list = self.query_one("#room-list", ListView)
        room_list.clear()

        # Filter out expired rooms
        now = datetime.now(timezone.utc)
        active_rooms = []
        for room in self.rooms:
            expires_at = room.get("expires_at")
            if expires_at:
                exp_dt = parse_expiration(expires_at)
                if exp_dt and exp_dt <= now:
                    continue  # Skip expired rooms
            active_rooms.append(room)

        if not active_rooms:
            # Show "no rooms" message
            item = ListItem(
                Label("No active rooms. Create one with: sleap-connect room create")
            )
            room_list.append(item)
            return

        # Create list items and track them for worker discovery
        self._room_items: list[tuple[dict, RoomListItem]] = []
        for room in active_rooms:
            item = RoomListItem(room)
            room_list.append(item)
            self._room_items.append((room, item))

        # Focus the list
        room_list.focus()

        # Start lazy loading worker counts
        asyncio.create_task(self._fetch_worker_counts())

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle room selection."""
        if not self.rooms:
            return

        item = event.item
        if isinstance(item, RoomListItem):
            room_data = item.room_data

            # Call the selection callback
            if self.on_room_selected:
                self.on_room_selected(room_data)

    def action_refresh(self) -> None:
        """Refresh the room list."""
        self.fetch_rooms()

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()

    async def _fetch_worker_counts(self) -> None:
        """Fetch worker counts for all rooms in background."""
        if not hasattr(self, "_room_items"):
            return

        # Fetch worker counts concurrently
        tasks = []
        for room_data, item in self._room_items:
            task = asyncio.create_task(self._discover_workers_for_room(room_data, item))
            tasks.append(task)

        # Wait for all to complete (with individual error handling)
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _discover_workers_for_room(
        self, room_data: dict, item: RoomListItem
    ) -> None:
        """Discover workers for a single room and update the list item.

        Args:
            room_data: Room data dict with room_id.
            item: The RoomListItem to update.
        """
        import json
        import websockets
        from sleap_connect.auth.credentials import get_valid_jwt, get_user

        room_id = room_data.get("room_id")
        if not room_id:
            item.worker_count = -1
            return

        config = get_config()
        ws_url = config.signaling_websocket

        try:
            jwt_token = get_valid_jwt()
            user = get_user()
            peer_id = user.get("username", "tui-probe") if user else "tui-probe"

            async with websockets.connect(ws_url) as ws:
                # Register with room (minimal registration for discovery)
                register_msg = json.dumps(
                    {
                        "type": "register",
                        "peer_id": f"{peer_id}-probe-{room_id[:8]}",
                        "room_id": room_id,
                        "token": "",  # JWT auth, no room token needed
                        "role": "client",
                        "jwt": jwt_token,
                        "metadata": {"tags": ["probe"]},
                    }
                )
                await ws.send(register_msg)

                # Wait for registration response
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(response)

                if data.get("type") != "registered_auth":
                    item.worker_count = -1
                    return

                # Discover workers
                discover_msg = json.dumps(
                    {
                        "type": "discover_peers",
                        "from_peer_id": f"{peer_id}-probe-{room_id[:8]}",
                        "filters": {
                            "role": "worker",
                            "room_id": room_id,
                            "tags": ["sleap-connect"],
                        },
                    }
                )
                await ws.send(discover_msg)

                # Get worker list
                response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(response)

                if data.get("type") == "peer_list":
                    workers = data.get("peers", [])
                    item.worker_count = len(workers)
                else:
                    item.worker_count = -1

        except Exception:
            # Discovery failed - show error state
            item.worker_count = -1
