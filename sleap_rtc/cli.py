"""Unified CLI for sleap-RTC using rich-click."""

import json
import logging
import os
import sys
import tomllib
from datetime import datetime
from pathlib import Path

import rich_click as click
import requests
from loguru import logger

# =============================================================================
# rich-click Configuration
# =============================================================================

# Use Rich markup in help text
click.rich_click.USE_RICH_MARKUP = True

# Style configuration
click.rich_click.STYLE_OPTION = "bold cyan"
click.rich_click.STYLE_ARGUMENT = "bold cyan"
click.rich_click.STYLE_COMMAND = "bold green"
click.rich_click.STYLE_SWITCH = "bold yellow"
click.rich_click.STYLE_METAVAR = "bold magenta"
click.rich_click.STYLE_USAGE = "bold"
click.rich_click.STYLE_HELPTEXT = ""

# Panel configuration
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.SHOW_METAVARS_COLUMN = True
click.rich_click.APPEND_METAVARS_HELP = True

# Command groups for organized help output
click.rich_click.COMMAND_GROUPS = {
    "sleap-rtc": [
        {
            "name": "Authentication",
            "commands": ["login", "logout", "whoami", "key"],
        },
        {
            "name": "Rooms & Tokens",
            "commands": ["room", "token"],
        },
        {
            "name": "Worker & Client",
            "commands": ["worker", "train", "track"],
        },
        {
            "name": "Protocol v1 Worker (no signaling server)",
            "commands": ["serve", "pair"],
        },
        {
            "name": "Utilities",
            "commands": ["tui", "status", "doctor", "credentials", "config"],
        },
        {
            "name": "Experimental",
            "commands": ["test"],
        },
    ],
    "sleap-rtc test": [
        {
            "name": "File Browser Tools",
            "commands": ["browse", "resolve-paths"],
        },
    ],
}

from sleap_rtc.rtc_worker import run_RTCworker
from sleap_rtc.rtc_client import run_RTCclient, run_job_submit
from sleap_rtc.rtc_client_track import run_RTCclient_track
from sleap_rtc.jobs.spec import TrainJobSpec, TrackJobSpec

# =============================================================================
# Authentication Helpers
# =============================================================================


def require_login() -> str:
    """Require user to be logged in and return valid JWT.

    Checks if user has a valid (non-expired) JWT token. If not, prints
    a helpful error message and exits.

    Returns:
        Valid JWT token string.

    Raises:
        SystemExit: If user is not logged in or JWT is expired.
    """
    from sleap_rtc.auth.credentials import get_jwt, get_user, is_logged_in

    if not is_logged_in():
        click.echo(click.style("Error: ", fg="red", bold=True) + "Not logged in.")
        click.echo("")
        click.echo("This command requires authentication. Please log in first:")
        click.echo(click.style("  sleap-rtc login", fg="cyan"))
        click.echo("")
        sys.exit(1)

    jwt_token = get_jwt()
    if not jwt_token:
        click.echo(click.style("Error: ", fg="red", bold=True) + "No JWT token found.")
        click.echo("")
        click.echo("Please log in again:")
        click.echo(click.style("  sleap-rtc login", fg="cyan"))
        click.echo("")
        sys.exit(1)

    # Check JWT expiration
    try:
        import jwt

        claims = jwt.decode(jwt_token, options={"verify_signature": False})
        exp = claims.get("exp")
        if exp:
            from datetime import datetime

            exp_dt = datetime.fromtimestamp(exp)
            if exp_dt < datetime.now():
                user = get_user()
                username = user.get("username", "unknown") if user else "unknown"
                click.echo(
                    click.style("Error: ", fg="red", bold=True)
                    + "JWT token has expired."
                )
                click.echo(f"  Last logged in as: {username}")
                click.echo("")
                click.echo("Please log in again:")
                click.echo(click.style("  sleap-rtc login", fg="cyan"))
                click.echo("")
                sys.exit(1)
    except Exception:
        # If we can't decode JWT, let the server handle validation
        pass

    return jwt_token


def _parse_duration(s: str) -> int:
    """Parse a duration string like '30m', '2h', '1h30m', '90s' into seconds.

    Also accepts plain integer strings as seconds (e.g. '3600').

    Args:
        s: Duration string.

    Returns:
        Duration in seconds (must be positive).

    Raises:
        click.BadParameter: If the string cannot be parsed or is zero/negative.
    """
    import re

    if s.isdigit():
        value = int(s)
        if value <= 0:
            raise click.BadParameter(
                "Duration must be positive. Use e.g. '30m', '2h', '3600'."
            )
        return value
    total = 0
    for match in re.finditer(r"(\d+)\s*([hms])", s.lower()):
        value, unit = int(match.group(1)), match.group(2)
        multiplier = {"h": 3600, "m": 60, "s": 1}[unit]
        total += value * multiplier
    if total <= 0:
        raise click.BadParameter(
            f"Cannot parse duration: {s!r}. Use e.g. '30m', '2h', '1h30m', '90s'."
        )
    return total


@click.group()
def cli():
    pass


# =============================================================================
# Auth Commands
# =============================================================================


@cli.command()
@click.option("--timeout", default=120, help="Login timeout in seconds.")
def login(timeout):
    """Log in to SLEAP-RTC via GitHub OAuth.

    Opens your browser to authenticate with GitHub via the SLEAP-RTC dashboard.
    After authorization, your credentials are saved locally for future CLI commands.
    On first login, an account key and Ed25519 keypair are generated automatically.

    Example:
        sleap-rtc login
    """
    from sleap_rtc.auth.credentials import (
        get_account_key,
        get_private_key_b64,
        get_public_key_registered,
        get_user,
        is_logged_in,
        save_account_key,
        save_jwt,
    )
    from sleap_rtc.auth.github import github_login

    if is_logged_in():
        user = get_user()
        if user:
            click.echo(f"Already logged in as {user.get('username', 'unknown')}")
            if not click.confirm("Log in as a different user?"):
                return

    try:
        result = github_login(timeout=timeout)
        save_jwt(result["jwt"], result["user"])

        # Store account key if server returned one (only on first login)
        if result.get("account_key"):
            save_account_key(result["account_key"])
            click.echo("\nAccount key (save this — shown only once):")
            click.echo(f"  {result['account_key']}")
            click.echo(
                f"\nStore it as: export SLEAP_RTC_ACCOUNT_KEY={result['account_key']}"
            )

        # Ensure Ed25519 keypair exists and is registered for P2P auth
        from sleap_rtc.api import _ensure_keypair_registered

        had_keypair = bool(get_private_key_b64())
        had_registered = get_public_key_registered()
        auth = get_account_key() or result.get("jwt")
        _ensure_keypair_registered(auth_token=auth, device_name="cli")
        if not had_keypair:
            click.echo("Generated Ed25519 keypair for P2P authentication.")
        if not had_registered and get_public_key_registered():
            click.echo("Public key registered with server.")

        user = result["user"]
        click.echo(f"\nLogged in as {user.get('username', 'unknown')}")
        logger.info("Credentials saved to ~/.sleap-rtc/credentials.json")

    except Exception as e:
        logger.error(f"Login failed: {e}")
        sys.exit(1)


@cli.command()
def logout():
    """Log out and clear stored credentials.

    Removes the JWT and user info from ~/.sleap-rtc/credentials.json.
    Worker tokens (API keys) are also removed.
    """
    from sleap_rtc.auth.credentials import clear_credentials, is_logged_in

    if not is_logged_in():
        click.echo("Not currently logged in")
        return

    clear_credentials()
    click.echo("Logged out successfully")


@cli.command()
def whoami():
    """Display current logged-in user info.

    Shows your GitHub username and user ID from the stored JWT.
    """
    from sleap_rtc.auth.credentials import get_user, get_jwt, is_logged_in

    if not is_logged_in():
        click.echo("Not logged in. Run: sleap-rtc login")
        sys.exit(1)

    user = get_user()
    if not user:
        click.echo("No user info found. Try logging in again.")
        sys.exit(1)

    click.echo(f"Username: {user.get('username', 'unknown')}")
    click.echo(f"User ID:  {user.get('user_id', user.get('id', 'unknown'))}")

    # Show JWT expiry if we can decode it
    jwt_token = get_jwt()
    if jwt_token:
        try:
            import jwt

            # Decode without verification just to read claims
            claims = jwt.decode(jwt_token, options={"verify_signature": False})
            exp = claims.get("exp")
            if exp:
                exp_dt = datetime.fromtimestamp(exp)
                click.echo(f"JWT expires: {exp_dt.isoformat()}")
        except Exception:
            pass


# =============================================================================
# Token Commands (subgroup)
# =============================================================================


@cli.group()
def token():
    """Manage worker API tokens.

    Worker tokens are used to authenticate workers with the signaling server.
    Each token is associated with a specific room.
    """
    pass


@token.command(name="create")
@click.option(
    "--room",
    "-r",
    required=True,
    help="Room ID to create token for.",
)
@click.option(
    "--name",
    "-n",
    required=True,
    help="Human-readable name for this worker (e.g., 'lab-gpu-1').",
)
@click.option(
    "--expires",
    "-e",
    type=int,
    default=None,
    help="Token expiration in days (default: 7).",
)
@click.option(
    "--save/--no-save",
    default=True,
    help="Save token to credentials file (default: yes).",
)
def token_create(room, name, expires, save):
    """Create a new worker API token.

    Creates an API key for a worker to join a specific room.
    The room must already exist and you must be a member.

    Example:
        sleap-rtc token create --room abc123 --name lab-gpu-1
    """
    from sleap_rtc.auth.credentials import get_jwt, save_token
    from sleap_rtc.config import get_config

    jwt_token = get_jwt()
    if not jwt_token:
        logger.error("Not logged in. Run: sleap-rtc login")
        sys.exit(1)

    config = get_config()
    endpoint = f"{config.get_http_url()}/api/auth/token"

    payload = {
        "room_id": room,
        "worker_name": name,
    }
    if expires:
        payload["expires_days"] = expires

    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers={"Authorization": f"Bearer {jwt_token}"},
            timeout=30,
        )

        if response.status_code != 200:
            error = response.json().get("error", response.text)
            logger.error(f"Failed to create token: {error}")
            sys.exit(1)

        data = response.json()

        click.echo("")
        click.echo("Worker token created successfully!")
        click.echo("")
        click.echo(f"  API Key:     {data['token_id']}")
        click.echo(f"  Room:        {data['room_id']}")
        click.echo(f"  Worker Name: {name}")
        if data.get("expires_at"):
            click.echo(f"  Expires:     {data['expires_at']}")
        click.echo("")
        click.echo("To start a worker with this token:")
        click.echo(f"  sleap-rtc worker --api-key {data['token_id']}")
        click.echo("")

        if save:
            save_token(room, data["token_id"], name)
            logger.info("Token saved to ~/.sleap-rtc/credentials.json")

    except requests.RequestException as e:
        logger.error(f"Request failed: {e}")
        sys.exit(1)


@token.command(name="list")
@click.option(
    "--room",
    "-r",
    help="Filter by room ID.",
)
@click.option(
    "--sort",
    "-s",
    type=click.Choice(["name", "created", "expires", "room"]),
    default="created",
    help="Sort by field.",
)
@click.option(
    "--reverse",
    is_flag=True,
    help="Reverse sort order.",
)
@click.option(
    "--active-only",
    "-a",
    is_flag=True,
    help="Hide revoked and expired tokens.",
)
def token_list(room, sort, reverse, active_only):
    """List your API tokens.

    Shows all tokens you've created, their status, and expiration.

    Examples:
        sleap-rtc token list --room abc123
        sleap-rtc token list --active-only
        sleap-rtc token list --sort expires
    """
    from sleap_rtc.auth.credentials import get_jwt
    from sleap_rtc.config import get_config

    jwt_token = get_jwt()
    if not jwt_token:
        logger.error("Not logged in. Run: sleap-rtc login")
        sys.exit(1)

    config = get_config()
    endpoint = f"{config.get_http_url()}/api/auth/tokens"

    # Build query params
    params = {}
    if room:
        params["room_id"] = room
    if active_only:
        params["active_only"] = "true"
    sort_map = {
        "name": "worker_name",
        "created": "created_at",
        "expires": "expires_at",
        "room": "room_name",
    }
    params["sort_by"] = sort_map.get(sort, "created_at")
    params["sort_order"] = "asc" if reverse else "desc"

    try:
        response = requests.get(
            endpoint,
            headers={"Authorization": f"Bearer {jwt_token}"},
            params=params,
            timeout=30,
        )

        if response.status_code != 200:
            error = response.json().get("error", response.text)
            logger.error(f"Failed to list tokens: {error}")
            sys.exit(1)

        data = response.json()
        tokens = data.get("tokens", [])

        if not tokens:
            click.echo("No tokens found")
            return

        click.echo("")
        click.echo(f"{'NAME':<20} {'ROOM':<12} {'STATUS':<10} {'EXPIRES':<20}")
        click.echo("-" * 62)

        for t in tokens:
            name = t.get("worker_name", "unknown")[:20]
            room_display = (t.get("room_name") or t.get("room_id", "?"))[:12]
            if t.get("revoked_at"):
                status = "revoked"
            elif not t.get("is_active"):
                status = "expired"
            else:
                status = "active"
            expires = t.get("expires_at", "never")
            if expires and expires != "never":
                expires = expires[:20]
            click.echo(f"{name:<20} {room_display:<12} {status:<10} {expires:<20}")

        click.echo("")

    except requests.RequestException as e:
        logger.error(f"Request failed: {e}")
        sys.exit(1)


@token.command(name="revoke")
@click.argument("token_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def token_revoke(token_id, yes):
    """Revoke an API token.

    Immediately invalidates the token. Workers using this token
    will no longer be able to connect.

    Example:
        sleap-rtc token revoke slp_abc123...
    """
    from sleap_rtc.auth.credentials import get_jwt
    from sleap_rtc.config import get_config

    jwt_token = get_jwt()
    if not jwt_token:
        logger.error("Not logged in. Run: sleap-rtc login")
        sys.exit(1)

    if not yes:
        if not click.confirm(f"Revoke token {token_id[:20]}...?"):
            click.echo("Cancelled")
            return

    config = get_config()
    endpoint = f"{config.get_http_url()}/api/auth/token/{token_id}"

    try:
        response = requests.delete(
            endpoint,
            headers={"Authorization": f"Bearer {jwt_token}"},
            timeout=30,
        )

        if response.status_code != 200:
            error = response.json().get("error", response.text)
            logger.error(f"Failed to revoke token: {error}")
            sys.exit(1)

        click.echo("Token revoked successfully")

    except requests.RequestException as e:
        logger.error(f"Request failed: {e}")
        sys.exit(1)


@token.command(name="delete")
@click.argument("token_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def token_delete(token_id, yes):
    """Permanently delete a revoked or expired token.

    Only inactive tokens (revoked or expired) can be deleted.
    Active tokens must be revoked first using 'sleap-rtc token revoke'.

    Example:
        sleap-rtc token delete slp_abc123...
    """
    from sleap_rtc.auth.credentials import get_jwt
    from sleap_rtc.config import get_config

    jwt_token = get_jwt()
    if not jwt_token:
        logger.error("Not logged in. Run: sleap-rtc login")
        sys.exit(1)

    if not yes:
        if not click.confirm(
            f"Permanently delete token {token_id[:20]}...? This cannot be undone."
        ):
            click.echo("Cancelled")
            return

    config = get_config()
    endpoint = f"{config.get_http_url()}/api/auth/tokens/{token_id}"

    try:
        response = requests.delete(
            endpoint,
            headers={"Authorization": f"Bearer {jwt_token}"},
            timeout=30,
        )

        if response.status_code != 200:
            error = response.json().get("detail", response.text)
            logger.error(f"Failed to delete token: {error}")
            sys.exit(1)

        click.echo("Token deleted permanently")

    except requests.RequestException as e:
        logger.error(f"Request failed: {e}")
        sys.exit(1)


@token.command(name="cleanup")
@click.option(
    "--room", "-r", help="Filter by room ID (delete only tokens for this room)."
)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
def token_cleanup(room, yes):
    """Delete all revoked and expired tokens.

    Permanently removes all inactive tokens to clean up your token list.
    Only deletes tokens you created or tokens for rooms you own.

    Examples:
        sleap-rtc token cleanup
        sleap-rtc token cleanup --room abc123
        sleap-rtc token cleanup --yes
    """
    from sleap_rtc.auth.credentials import get_jwt
    from sleap_rtc.config import get_config

    jwt_token = get_jwt()
    if not jwt_token:
        logger.error("Not logged in. Run: sleap-rtc login")
        sys.exit(1)

    config = get_config()

    # First, count inactive tokens
    list_endpoint = f"{config.get_http_url()}/api/auth/tokens"
    try:
        response = requests.get(
            list_endpoint,
            headers={"Authorization": f"Bearer {jwt_token}"},
            params={"room_id": room} if room else {},
            timeout=30,
        )
        if response.status_code != 200:
            logger.error("Failed to fetch tokens")
            sys.exit(1)

        tokens = response.json().get("tokens", [])
        inactive_tokens = [t for t in tokens if not t.get("is_active")]

        if not inactive_tokens:
            click.echo("No inactive tokens to delete.")
            return

        click.echo(f"Found {len(inactive_tokens)} inactive token(s) to delete:")
        for t in inactive_tokens[:5]:  # Show first 5
            status = "revoked" if t.get("revoked_at") else "expired"
            click.echo(f"  - {t['worker_name']} ({status})")
        if len(inactive_tokens) > 5:
            click.echo(f"  ... and {len(inactive_tokens) - 5} more")

    except requests.RequestException as e:
        logger.error(f"Request failed: {e}")
        sys.exit(1)

    if not yes:
        if not click.confirm(
            f"Delete {len(inactive_tokens)} inactive token(s)? This cannot be undone."
        ):
            click.echo("Cancelled")
            return

    # Delete all inactive tokens
    delete_endpoint = f"{config.get_http_url()}/api/auth/tokens"
    try:
        response = requests.delete(
            delete_endpoint,
            headers={"Authorization": f"Bearer {jwt_token}"},
            params={"room_id": room} if room else {},
            timeout=30,
        )

        if response.status_code != 200:
            error = response.json().get("detail", response.text)
            logger.error(f"Failed to delete tokens: {error}")
            sys.exit(1)

        deleted_count = response.json().get("deleted_count", 0)
        click.echo(f"Deleted {deleted_count} inactive token(s)")

    except requests.RequestException as e:
        logger.error(f"Request failed: {e}")
        sys.exit(1)


# =============================================================================
# Room Commands (subgroup)
# =============================================================================


@cli.group()
def room():
    """Manage rooms.

    Rooms are workspaces where workers and clients connect.
    Room owners can invite other users and create worker tokens.
    """
    pass


@room.command(name="list")
@click.option(
    "--filter",
    "-f",
    "role_filter",
    type=click.Choice(["all", "owned", "member"]),
    default="all",
    help="Filter by ownership role.",
)
@click.option(
    "--sort",
    "-s",
    type=click.Choice(["name", "created", "expires", "role"]),
    default="created",
    help="Sort by field.",
)
@click.option(
    "--reverse",
    "-r",
    is_flag=True,
    help="Reverse sort order.",
)
@click.option(
    "--search",
    help="Search by room name (case-insensitive substring match).",
)
def room_list(role_filter, sort, reverse, search):
    """List rooms you have access to.

    Shows all rooms where you are an owner or member.

    Examples:
        sleap-rtc room list --filter owned
        sleap-rtc room list --sort name
        sleap-rtc room list --search "lab"
    """
    from sleap_rtc.config import get_config

    config = get_config()
    endpoint = f"{config.get_http_url()}/api/auth/rooms"

    # Build query params
    params = {}
    if role_filter != "all":
        params["role"] = role_filter.replace("owned", "owner")  # Map 'owned' to 'owner'
    sort_map = {
        "name": "name",
        "created": "joined_at",
        "expires": "expires_at",
        "role": "role",
    }
    params["sort_by"] = sort_map.get(sort, "joined_at")
    params["sort_order"] = "asc" if reverse else "desc"
    if search:
        params["search"] = search

    response = _key_request("GET", endpoint, params=params, timeout=30)

    if response.status_code != 200:
        error = response.json().get("error", response.text)
        logger.error(f"Failed to list rooms: {error}")
        sys.exit(1)

    data = response.json()
    rooms = data.get("rooms", [])

    if not rooms:
        click.echo("No rooms found")
        click.echo("Create one with: sleap-rtc room create")
        return

    click.echo("")
    click.echo(f"{'ROOM ID':<12} {'NAME':<16} {'ROLE':<8} {'EXPIRES':<20}")
    click.echo("-" * 60)

    for r in rooms:
        room_id = r.get("room_id", "?")[:12]
        name = (r.get("name") or room_id)[:16]
        role = r.get("role", "?")[:8]

        # Format expiration
        expires_at = r.get("expires_at")
        if expires_at:
            expires_dt = datetime.fromtimestamp(expires_at)
            now = datetime.now()
            if expires_dt < now:
                expires_str = "EXPIRED"
            elif (expires_dt - now).days > 0:
                expires_str = f"{(expires_dt - now).days}d left"
            else:
                hours = (expires_dt - now).seconds // 3600
                expires_str = f"{hours}h left"
        else:
            expires_str = "Never"

        click.echo(f"{room_id:<12} {name:<16} {role:<8} {expires_str:<20}")

    click.echo("")


@room.command(name="create")
@click.option(
    "--name",
    "-n",
    default=None,
    help="Optional name for the room.",
)
@click.option(
    "--expires",
    "-e",
    type=click.Choice(["1d", "5d", "10d", "15d", "30d", "never"]),
    default="30d",
    help="Room expiration time (default: 30d). Use 'never' for no expiration.",
)
def room_create(name, expires):
    """Create a new room.

    Creates a room where you can add workers and invite collaborators.
    You will be the owner of the created room.

    Example:
        sleap-rtc room create --name "my-training-room"
        sleap-rtc room create --name "persistent-room" --expires never
    """
    from sleap_rtc.config import get_config

    config = get_config()
    endpoint = f"{config.get_http_url()}/api/auth/rooms"

    # Map expires to days (None = never expires)
    expires_map = {"1d": 1, "5d": 5, "10d": 10, "15d": 15, "30d": 30, "never": None}
    expires_in_days = expires_map[expires]

    payload = {"expires_in_days": expires_in_days}
    if name:
        payload["name"] = name

    response = _key_request("POST", endpoint, json=payload, timeout=30)

    if response.status_code != 200:
        error = response.json().get("error", response.text)
        logger.error(f"Failed to create room: {error}")
        sys.exit(1)

    data = response.json()

    # Format expiration display
    if data.get("expires_at"):
        expires_dt = datetime.fromtimestamp(data["expires_at"])
        expires_str = expires_dt.strftime("%Y-%m-%d %H:%M")
    else:
        expires_str = "Never"

    click.echo("")
    click.echo("Room created successfully!")
    click.echo("")
    click.echo(f"  Room ID:  {data['room_id']}")
    click.echo(f"  Expires:  {expires_str}")

    # Offer to set as default room
    from sleap_rtc.auth.credentials import get_default_room, save_default_room

    current_default = get_default_room()
    if current_default is None:
        # No default set — offer to make this the default
        if click.confirm(
            "\nSet this as your default room? "
            "(workers will use it when --room is not specified)",
            default=True,
        ):
            save_default_room(data["room_id"])
            click.echo(f"Default room set to: {data['room_id']}")
    elif current_default != data["room_id"]:
        # Different default exists — offer to replace
        if click.confirm(
            f"\nReplace your current default room ({current_default}) with this one?"
        ):
            save_default_room(data["room_id"])
            click.echo(f"Default room updated to: {data['room_id']}")

    click.echo("")
    click.echo("Next steps:")
    click.echo(f"  1. Set up an account key: sleap-rtc key create --save")
    click.echo(f'  2. Start a worker: sleap-rtc worker --name "My GPU Server"')
    click.echo("")


@room.command(name="info")
@click.argument("room_id")
def room_info(room_id):
    """Show details for a room.

    Displays room information including name, expiration, members, and role.

    Example:
        sleap-rtc room info abc123
    """
    from sleap_rtc.config import get_config

    config = get_config()
    endpoint = f"{config.get_http_url()}/api/auth/rooms/{room_id}"

    response = _key_request("GET", endpoint, timeout=30)

    if response.status_code != 200:
        error = response.json().get(
            "detail", response.json().get("error", response.text)
        )
        logger.error(f"Failed to get room info: {error}")
        sys.exit(1)

    data = response.json()

    # Format expiration
    expires_at = data.get("expires_at")
    if expires_at:
        expires_dt = datetime.fromtimestamp(expires_at)
        now = datetime.now()
        time_left = expires_dt - now

        if time_left.total_seconds() < 0:
            expires_str = f"{expires_dt.strftime('%Y-%m-%d %H:%M')} (EXPIRED)"
        elif time_left.days > 0:
            expires_str = (
                f"{expires_dt.strftime('%Y-%m-%d %H:%M')} ({time_left.days} days left)"
            )
        elif time_left.seconds > 3600:
            hours_left = time_left.seconds // 3600
            expires_str = (
                f"{expires_dt.strftime('%Y-%m-%d %H:%M')} ({hours_left} hours left)"
            )
        else:
            minutes_left = time_left.seconds // 60
            expires_str = (
                f"{expires_dt.strftime('%Y-%m-%d %H:%M')} ({minutes_left} minutes left)"
            )
    else:
        expires_str = "Never"

    click.echo("")
    click.echo(f"Room: {data.get('name', room_id)}")
    click.echo("")
    click.echo(f"  Room ID:    {room_id}")
    click.echo(f"  Your Role:  {data.get('role', 'unknown')}")
    click.echo(f"  Expires:    {expires_str}")

    # Show members if available
    members = data.get("members", [])
    if members:
        click.echo("")
        click.echo("  Members:")
        for member in members:
            role_icon = "👑" if member.get("role") == "owner" else "👤"
            click.echo(
                f"    {role_icon} {member.get('username', member.get('user_id', 'unknown'))} ({member.get('role', 'member')})"
            )

    click.echo("")


@room.command(name="delete")
@click.argument("room_id")
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Skip confirmation prompt.",
)
def room_delete(room_id, force):
    """Delete a room.

    Permanently deletes a room and all associated data including tokens
    and memberships. Only room owners can delete rooms.

    Example:
        sleap-rtc room delete abc123
        sleap-rtc room delete abc123 --force
    """
    from sleap_rtc.config import get_config

    if not force:
        if not click.confirm(
            f"Are you sure you want to delete room '{room_id}'? This cannot be undone."
        ):
            click.echo("Cancelled")
            return

    config = get_config()
    endpoint = f"{config.get_http_url()}/api/auth/rooms/{room_id}"

    response = _key_request("DELETE", endpoint, timeout=30)

    if response.status_code != 200:
        error = response.json().get(
            "detail", response.json().get("error", response.text)
        )
        logger.error(f"Failed to delete room: {error}")
        sys.exit(1)

    click.echo(f"Room '{room_id}' deleted successfully")


@room.command(name="invite")
@click.argument("room_id")
def room_invite(room_id):
    """Generate an invite code for a room.

    Creates a short-lived invite code that others can use to join the room.
    You must be the owner of the room to generate invites.

    Example:
        sleap-rtc room invite abc123
    """
    from sleap_rtc.config import get_config

    config = get_config()
    endpoint = f"{config.get_http_url()}/api/auth/rooms/{room_id}/invite"

    response = _key_request("POST", endpoint, timeout=30)

    if response.status_code != 200:
        error = response.json().get("error", response.text)
        logger.error(f"Failed to create invite: {error}")
        sys.exit(1)

    data = response.json()

    click.echo("")
    click.echo("Invite code created!")
    click.echo("")
    click.echo(f"  Code:    {data['invite_code']}")
    click.echo(f"  Expires: {data.get('expires_at', 'in 24 hours')}")
    click.echo("")
    click.echo("Share this command with your collaborator:")
    click.echo(f"  sleap-rtc room join --code {data['invite_code']}")
    click.echo("")


@room.command(name="create-secret")
@click.option(
    "--room",
    "--room-id",
    "-r",
    type=str,
    required=True,
    help="Room ID to create secret for.",
)
@click.option(
    "--save/--no-save",
    default=True,
    help="Save secret to credentials file (default: yes).",
)
def room_create_secret(room, save):
    """Generate a new room secret for legacy PSK authentication.

    [bold yellow]LEGACY[/bold yellow]: Account-key auth is now preferred for new setups.
    Use 'sleap-rtc key create --save' and 'sleap-rtc worker --account-key ...' instead.

    This command creates a cryptographically secure 256-bit pre-shared secret
    for authenticating P2P connections between workers and clients in the room.
    Room secrets are an older auth mechanism — account keys (Ed25519 keypairs)
    provide stronger, per-identity auth without manual secret distribution.

    Preferred auth flow:
        sleap-rtc login                     # log in and generate a keypair
        sleap-rtc key create --save         # create and save an account key
        sleap-rtc worker --account-key slp_acct_xxx...

    Legacy PSK configuration options (in priority order):
    1. CLI flag: --room-secret SECRET
    2. Environment variable: SLEAP_ROOM_SECRET=SECRET
    3. Filesystem: ~/.sleap-rtc/room-secrets/<room-id>
    4. Credentials file: ~/.sleap-rtc/credentials.json

    Examples:
        # Generate and save secret for a room (legacy)
        sleap-rtc room create-secret --room my-room

        # Generate without saving (just display)
        sleap-rtc room create-secret --room my-room --no-save
    """
    from sleap_rtc.auth.psk import generate_secret
    from sleap_rtc.auth.credentials import save_room_secret, get_room_secret

    # Check if secret already exists
    existing = get_room_secret(room)
    if existing:
        if not click.confirm(
            f"Room '{room}' already has a secret. Generate a new one?"
        ):
            click.echo("Cancelled")
            return

    # Generate new secret
    secret = generate_secret()

    click.echo("")
    click.echo("Room secret generated!")
    click.echo("")
    click.echo(f"  Room ID: {room}")
    click.echo(f"  Secret:  {secret}")
    click.echo("")

    if save:
        save_room_secret(room, secret)
        click.echo("Secret saved to ~/.sleap-rtc/credentials.json")
        click.echo("")

    click.echo("To use this secret (legacy PSK auth):")
    click.echo("")
    click.echo("  Worker:")
    click.echo(f"    sleap-rtc worker --room-secret {secret}")
    click.echo("    # Or set SLEAP_ROOM_SECRET environment variable")
    click.echo("")
    click.echo("  Client:")
    click.echo(
        f"    sleap-rtc train --room {room} --config /path/to/config.yaml --room-secret {secret}"
    )
    click.echo("    # Or set SLEAP_ROOM_SECRET environment variable")
    click.echo("")
    click.echo("Tip: For new setups, account-key auth is recommended instead:")
    click.echo("    sleap-rtc key create --save")
    click.echo("    sleap-rtc worker --account-key slp_acct_xxx...")
    click.echo("")


@room.command(name="join")
@click.option(
    "--code",
    "-c",
    required=True,
    help="Invite code from room owner.",
)
def room_join(code):
    """Join a room using an invite code.

    Use an invite code shared by the room owner to join their room.

    Example:
        sleap-rtc room join --code ABCD1234
    """
    from sleap_rtc.config import get_config

    config = get_config()
    endpoint = f"{config.get_http_url()}/api/auth/rooms/join"

    response = _key_request("POST", endpoint, json={"invite_code": code}, timeout=30)

    if response.status_code != 200:
        error = response.json().get("error", response.text)
        logger.error(f"Failed to join room: {error}")
        sys.exit(1)

    data = response.json()

    click.echo("")
    click.echo("Successfully joined room!")
    click.echo("")
    click.echo(f"  Room ID: {data['room_id']}")
    click.echo(f"  Role:    {data.get('role', 'member')}")

    # Offer to set as default room
    from sleap_rtc.auth.credentials import get_default_room, save_default_room

    current_default = get_default_room()
    if current_default is None:
        if click.confirm(
            "\nSet this as your default room? "
            "(workers will use it when --room is not specified)",
            default=True,
        ):
            save_default_room(data["room_id"])
            click.echo(f"Default room set to: {data['room_id']}")
    elif current_default != data["room_id"]:
        if click.confirm(
            f"\nReplace your current default room ({current_default}) with this one?"
        ):
            save_default_room(data["room_id"])
            click.echo(f"Default room updated to: {data['room_id']}")

    click.echo("")


@room.command(name="set-default")
@click.argument("room_id")
def room_set_default(room_id):
    """Set a room as your default (used by worker when --room is not specified).

    Example:
        sleap-rtc room set-default room-abc123
    """
    from sleap_rtc.auth.credentials import save_default_room

    save_default_room(room_id)
    click.echo(f"Default room set to: {room_id}")
    click.echo("Tip: workers started without --room will use this room.")


# =============================================================================
# Key Management
# =============================================================================


def _key_request(method, url, **kwargs):
    """Make an authenticated request for key commands.

    Tries the stored account key first. If the server returns 401 (e.g. the
    key was revoked externally), clears the bad key, warns the user, and
    retries with the JWT so the command still succeeds.

    Returns the Response, or raises SystemExit on unrecoverable failure.
    """
    from sleap_rtc.auth.credentials import get_account_key, get_jwt, remove_account_key

    account_key = get_account_key()
    jwt = get_jwt()
    auth = account_key or jwt

    if not auth:
        click.echo("Not logged in. Run: sleap-rtc login")
        sys.exit(1)

    headers = kwargs.pop("headers", {})
    headers["Authorization"] = f"Bearer {auth}"

    try:
        resp = requests.request(method, url, headers=headers, **kwargs)
    except requests.RequestException as e:
        logger.error(f"Request failed: {e}")
        sys.exit(1)

    if resp.status_code == 401 and account_key and jwt:
        click.echo(
            f"Warning: saved account key ({account_key[:20]}...) was rejected "
            "(revoked or expired). Falling back to JWT and clearing the bad key."
        )
        remove_account_key()
        headers["Authorization"] = f"Bearer {jwt}"
        try:
            resp = requests.request(method, url, headers=headers, **kwargs)
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            sys.exit(1)

    return resp


@cli.group()
def key():
    """Manage account keys for headless/container authentication.

    Account keys (slp_acct_...) replace the JWT for all API calls.
    They are long-lived, named, and revocable from the dashboard or CLI.

    Examples:
        sleap-rtc key list
        sleap-rtc key create --name hpc-cluster
        sleap-rtc key revoke slp_acct_xxx...
        sleap-rtc key show
    """
    pass


@key.command(name="list")
@click.option("--full", is_flag=True, help="Show full key IDs.")
def key_list(full):
    """List all account keys for your account."""
    from sleap_rtc.config import get_config

    config = get_config()
    resp = _key_request(
        "GET", f"{config.get_http_url()}/api/auth/account-keys", timeout=10
    )
    try:
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Request failed: {e}")
        sys.exit(1)

    keys = resp.json().get("keys", [])
    if not keys:
        click.echo("No account keys found.")
        return

    for k in keys:
        key_id = k["key_id"] if full else f"{k['key_id'][:30]}..."
        click.echo(f"  {key_id}  [{k.get('name', 'unnamed')}]")


@key.command(name="create")
@click.option(
    "--name", "-n", default="unnamed", help="Name for this key (e.g. 'hpc-cluster')."
)
@click.option(
    "--save",
    is_flag=True,
    help="Save the new key to ~/.sleap-rtc/credentials.json for automatic use.",
)
def key_create(name, save):
    """Create a new account key."""
    from sleap_rtc.auth.credentials import get_account_key, save_account_key
    from sleap_rtc.config import get_config

    config = get_config()
    resp = _key_request(
        "POST",
        f"{config.get_http_url()}/api/auth/account-keys",
        json={"name": name},
        timeout=10,
    )
    try:
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Request failed: {e}")
        sys.exit(1)

    data = resp.json()
    click.echo("\nNew account key created (save this — shown only once):")
    click.echo(f"  {data['key_id']}")

    if save:
        existing = get_account_key()
        if existing:
            if not click.confirm(
                f"\nYou already have a saved key ({existing[:20]}...). Replace it?"
            ):
                click.echo("Key created but not saved. Copy it from above.")
                return
            if click.confirm(f"Also revoke the old key ({existing[:20]}...)?"):
                try:
                    resp_revoke = requests.delete(
                        f"{config.get_http_url()}/api/auth/account-keys/{existing}",
                        headers={"Authorization": f"Bearer {auth}"},
                        timeout=10,
                    )
                    if resp_revoke.ok:
                        click.echo(f"Old key ({existing[:20]}...) revoked.")
                    else:
                        click.echo(
                            f"Warning: could not revoke old key: {resp_revoke.text}"
                        )
                except requests.RequestException as e:
                    click.echo(f"Warning: could not revoke old key: {e}")
        save_account_key(data["key_id"])
        click.echo("\nKey saved to ~/.sleap-rtc/credentials.json")
    else:
        click.echo(f"\nStore as: export SLEAP_RTC_ACCOUNT_KEY={data['key_id']}")


@key.command(name="revoke")
@click.argument("key_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation.")
def key_revoke(key_id, yes):
    """Revoke an account key by its ID."""
    from sleap_rtc.auth.credentials import get_account_key, remove_account_key
    from sleap_rtc.config import get_config

    is_saved_key = get_account_key() == key_id
    if is_saved_key and not yes:
        click.echo(
            f"Warning: {key_id[:20]}... is your currently saved key. "
            "After revoking it, automated commands (worker, key list, etc.) "
            "will fall back to your JWT until you save a new key with "
            "'sleap-rtc key create --save'."
        )

    if not yes and not click.confirm(
        f"Revoke key {key_id[:20]}...? This cannot be undone."
    ):
        click.echo("Cancelled.")
        return

    config = get_config()
    resp = _key_request(
        "DELETE",
        f"{config.get_http_url()}/api/auth/account-keys/{key_id}",
        timeout=10,
    )
    try:
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Request failed: {e}")
        sys.exit(1)

    click.echo(f"Key {key_id[:20]}... revoked.")

    if get_account_key() == key_id:
        remove_account_key()
        click.echo("Removed from local credentials (was your saved key).")


@key.command(name="use")
@click.argument("key_id")
def key_use(key_id):
    """Save an existing account key as the local default."""
    from sleap_rtc.auth.credentials import get_account_key, save_account_key

    existing = get_account_key()
    if existing:
        if existing == key_id:
            click.echo(f"Key {key_id[:20]}... is already your saved key.")
            return
        if not click.confirm(
            f"This will replace your current saved key ({existing[:20]}...). Continue?"
        ):
            click.echo("Cancelled.")
            return

    save_account_key(key_id)
    click.echo(f"Key {key_id[:20]}... saved to ~/.sleap-rtc/credentials.json")


@key.command(name="show")
def key_show():
    """Show the currently stored account key (with confirmation)."""
    from sleap_rtc.auth.credentials import get_account_key

    stored = get_account_key()
    if not stored:
        click.echo("No account key stored locally.")
        from sleap_rtc.auth.credentials import get_jwt

        if get_jwt():
            click.echo(
                "You're logged in. To set a key:\n"
                "  sleap-rtc key list --full          # find an existing key ID\n"
                "  sleap-rtc key use <key_id>         # save it as your default\n"
                "  sleap-rtc key create --save        # or create a new one"
            )
        else:
            click.echo(
                "Run 'sleap-rtc login' to log in, then:\n"
                "  sleap-rtc key create --save        # create and save a key\n"
                "Or set SLEAP_RTC_ACCOUNT_KEY=<key_id> as an environment variable."
            )
        return

    if click.confirm("Show your account key? (treat like a password)"):
        click.echo(f"\n  {stored}\n")
    else:
        click.echo("Cancelled.")


def show_worker_help():
    """Display"""
    help_text = """
    sleap-rtc worker - Set this machine as a sleap-RTC worker node.

    Usage:
      sleap-rtc worker

    Tips:
      - This machine should have a GPU available for optimal model inference.
      - Ensure that the sleap-RTC Client is running and accessible.
      - Make sure to copy the session-string after connecting to the signaling
        server.
    """
    click.echo(help_text)


@cli.command()
@click.option(
    "--account-key",
    type=str,
    envvar="SLEAP_RTC_ACCOUNT_KEY",
    required=False,
    help="Account key (slp_acct_...) for authentication. Can also use SLEAP_RTC_ACCOUNT_KEY env var.",
)
@click.option(
    "--api-key",
    "-k",
    type=str,
    envvar="SLEAP_RTC_API_KEY",
    required=False,
    help="[Legacy] Per-room API key (slp_xxx...). Use --account-key (slp_acct_xxx...) for new setups.",
)
@click.option(
    "--room",
    "--room-id",  # Alias for backward compatibility
    "-r",
    type=str,
    required=False,
    help="[Legacy] Room ID to join (if not provided, a new room will be created).",
)
@click.option(
    "--working-dir",
    "-w",
    type=str,
    required=False,
    help="Working directory for the worker. Overrides config file value.",
)
@click.option(
    "--name",
    "-n",
    type=str,
    required=False,
    help="Human-readable name for this worker (e.g. 'lab-gpu-1'). Shown in TUI and client discovery.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show detailed logs including ICE state, keep-alive, and file transfers.",
)
@click.option(
    "--room-secret",
    type=str,
    envvar="SLEAP_ROOM_SECRET",
    required=False,
    help="Room secret for P2P authentication. Can also use SLEAP_ROOM_SECRET env var.",
)
@click.option(
    "--max-reconnect-time",
    type=str,
    required=False,
    default=None,
    help="Max time to keep retrying signaling server before exiting. Examples: '30m', '2h'. Default: no limit (retry forever).",
)
def worker(
    api_key,
    account_key,
    room,
    working_dir,
    name,
    verbose,
    room_secret,
    max_reconnect_time,
):
    """Start the sleap-RTC worker node.

    Authentication (recommended setup):

        1. sleap-rtc login              # log in and generate a keypair
        2. sleap-rtc key create --save  # create and save an account key
        3. sleap-rtc worker             # starts automatically using saved key

    Or pass the key explicitly:

        sleap-rtc worker --account-key slp_acct_xxx...
        # or: export SLEAP_RTC_ACCOUNT_KEY=slp_acct_xxx...

    Legacy (per-room API key):

        sleap-rtc worker --api-key slp_xxx...  # deprecated, use --account-key
    """
    # Check for credential file if no explicit auth provided
    if not api_key and not account_key and not room:
        from sleap_rtc.auth.credentials import (
            get_account_key,
            get_credentials,
            get_default_room,
        )

        # Try account key first (new architecture)
        stored_account_key = get_account_key()
        if stored_account_key:
            account_key = stored_account_key
            room = get_default_room()
            if room:
                logger.info(f"Using account key + default room: {room}")
            else:
                logger.info(
                    "Using account key (no default room — use --room or "
                    "'sleap-rtc room set-default')"
                )
        else:
            # Fall back to legacy API key from credentials
            creds = get_credentials()
            tokens = creds.get("tokens", {})
            if tokens:
                first_room = next(iter(tokens))
                api_key = tokens[first_room].get("api_key")
                if api_key:
                    logger.info(
                        f"Using API key from credentials for room: {first_room}"
                    )
    elif account_key and not room:
        from sleap_rtc.auth.credentials import get_default_room

        room = get_default_room()

    # Validate authentication options
    has_api_key = api_key is not None
    has_room = room is not None

    if has_api_key and has_room:
        logger.error("Cannot use both --api-key and --room")
        logger.error("Choose one authentication method")
        sys.exit(1)

    # Configure logging based on verbosity
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate working directory if provided
    if working_dir:
        working_dir_path = Path(working_dir)
        if not working_dir_path.exists():
            logger.error(f"Working directory does not exist: {working_dir}")
            sys.exit(1)
        if not working_dir_path.is_dir():
            logger.error(f"Working directory is not a directory: {working_dir}")
            sys.exit(1)
        logger.info(f"Using working directory: {working_dir}")

    # Parse max reconnect time if provided
    max_reconnect_seconds = None
    if max_reconnect_time:
        max_reconnect_seconds = _parse_duration(max_reconnect_time)

    run_RTCworker(
        api_key=account_key or api_key,
        room_id=room,
        token=None,
        working_dir=working_dir,
        name=name,
        room_secret=room_secret,
        max_reconnect_time=max_reconnect_seconds,
    )


@cli.command(name="train")
@click.option(
    "--session-string",
    "--session_string",
    "-s",
    type=str,
    required=False,
    help="Session string for direct connection to a specific worker.",
)
@click.option(
    "--room",
    "--room-id",
    "-r",
    type=str,
    required=False,
    help="Room ID for room-based worker discovery. Requires login (sleap-rtc login).",
)
@click.option(
    "--worker-id",
    "-w",
    type=str,
    required=False,
    help="Specific worker peer-id to connect to (skips discovery).",
)
@click.option(
    "--auto-select",
    "-a",
    is_flag=True,
    default=False,
    help="Automatically select best worker by GPU memory (use with --room).",
)
@click.option(
    "--pkg-path",
    "--pkg_path",
    "-p",
    type=str,
    required=False,
    help="[DEPRECATED] Path to SLEAP training package. Use --config instead.",
)
@click.option(
    "--config",
    "-c",
    type=str,
    required=False,
    multiple=True,
    help="Path to sleap-nn config YAML file on worker filesystem. Can be specified multiple times for multi-model training (e.g., top-down: centroid + centered_instance).",
)
@click.option(
    "--labels",
    type=str,
    required=False,
    help="Override training labels path (data_config.train_labels_path).",
)
@click.option(
    "--val-labels",
    type=str,
    required=False,
    help="Override validation labels path (data_config.val_labels_path).",
)
@click.option(
    "--max-epochs",
    type=int,
    required=False,
    help="Maximum training epochs (trainer_config.max_epochs).",
)
@click.option(
    "--batch-size",
    type=int,
    required=False,
    help="Batch size for training and validation.",
)
@click.option(
    "--learning-rate",
    type=float,
    required=False,
    help="Learning rate for optimizer.",
)
@click.option(
    "--run-name",
    type=str,
    required=False,
    help="Name for the training run (used in checkpoint directory).",
)
@click.option(
    "--resume",
    type=str,
    required=False,
    help="Path to checkpoint for resuming training.",
)
@click.option(
    "--controller-port",
    "--controller_port",
    type=int,
    required=False,
    default=9000,
    help="ZMQ port for controller communication with SLEAP.",
)
@click.option(
    "--publish-port",
    "--publish_port",
    type=int,
    required=False,
    default=9001,
    help="ZMQ port for publish communication with SLEAP.",
)
@click.option(
    "--min-gpu-memory",
    type=int,
    required=False,
    default=None,
    help="Minimum GPU memory in MB required for training.",
)
@click.option(
    "--worker-path",
    type=str,
    required=False,
    default=None,
    help="Path on worker filesystem to use directly (skips path resolution).",
)
@click.option(
    "--non-interactive",
    is_flag=True,
    default=False,
    help="Non-interactive mode: auto-select best match without prompting (for CI/scripts).",
)
@click.option(
    "--mount",
    type=str,
    required=False,
    default=None,
    help="Mount label to search (skips mount selection prompt). Use 'all' to search all mounts.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show detailed logs including keep-alive, ICE state, and file transfers.",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    default=False,
    help="Show only errors and final status (hide progress bars and status messages).",
)
@click.option(
    "--room-secret",
    type=str,
    envvar="SLEAP_ROOM_SECRET",
    required=False,
    help="Room secret for P2P authentication. Can also use SLEAP_ROOM_SECRET env var.",
)
def train(**kwargs):
    """Run remote training on a worker.

    Connection modes (mutually exclusive):

    1. Session string (direct): --session-string SESSION
       Connect directly to a specific worker using its session string.

    2. Room-based discovery: --room ROOM
       Join a room and discover available workers. Requires login first
       (sleap-rtc login). Supports:
       - Interactive selection (default)
       - Auto-select: --auto-select
       - Direct worker: --worker-id PEER_ID
       - GPU filter: --min-gpu-memory MB

    Job specification options (mutually exclusive):

    1. Config-based (recommended): --config PATH
       Use a sleap-nn config YAML file with optional overrides.

    2. Package-based (deprecated): --pkg-path PATH
       Legacy workflow using training package files.

    All paths refer to locations on the worker filesystem. Use shared storage
    (e.g., /vast) or ensure files exist on the worker.

    Examples:
    \b
    # Basic training with config file
    sleap-rtc train --room my-room --config /vast/project/centroid.yaml

    \b
    # Training with labels override
    sleap-rtc train --room my-room --config /vast/project/centroid.yaml \\
        --labels /vast/data/labels.slp

    \b
    # Training with all overrides
    sleap-rtc train --room my-room --config /vast/project/centroid.yaml \\
        --labels /vast/data/labels.slp --max-epochs 100 --batch-size 8

    \b
    # Auto-select worker with GPU filter
    sleap-rtc train --room my-room --auto-select --min-gpu-memory 8000 \\
        --config /vast/project/centroid.yaml
    """
    # Extract connection options
    session_string = kwargs.pop("session_string", None)
    room_id = kwargs.pop("room", None)
    token = kwargs.pop("token", None)
    worker_id = kwargs.pop("worker_id", None)
    auto_select = kwargs.pop("auto_select", False)
    min_gpu_memory = kwargs.pop("min_gpu_memory", None)

    # Extract path resolution options
    worker_path = kwargs.pop("worker_path", None)
    non_interactive = kwargs.pop("non_interactive", False)
    mount_label = kwargs.pop("mount", None)

    # Extract P2P authentication options
    room_secret = kwargs.pop("room_secret", None)

    # Extract verbosity options
    verbose = kwargs.pop("verbose", False)
    quiet = kwargs.pop("quiet", False)

    # Validate: --verbose and --quiet are mutually exclusive
    if verbose and quiet:
        logger.error("--verbose and --quiet are mutually exclusive.")
        sys.exit(1)

    # Configure logging based on verbosity
    if quiet:
        verbosity = "quiet"
        logging.getLogger().setLevel(logging.ERROR)
    elif verbose:
        verbosity = "verbose"
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        verbosity = "default"
        logging.getLogger().setLevel(logging.INFO)

    # Extract job specification options (new structured flow)
    # --config is multiple=True, so it's a tuple
    config_paths = kwargs.pop("config", ())
    labels_path = kwargs.pop("labels", None)
    val_labels_path = kwargs.pop("val_labels", None)
    max_epochs = kwargs.pop("max_epochs", None)
    batch_size = kwargs.pop("batch_size", None)
    learning_rate = kwargs.pop("learning_rate", None)
    run_name = kwargs.pop("run_name", None)
    resume_ckpt_path = kwargs.pop("resume", None)

    # Extract legacy pkg-path option
    pkg_path = kwargs.pop("pkg_path", None)

    # Validation: --config and --pkg-path are mutually exclusive
    if config_paths and pkg_path:
        logger.error("--config and --pkg-path are mutually exclusive.")
        logger.error("Use --config for the new workflow (recommended).")
        sys.exit(1)

    # Validation: Must provide either --config or --pkg-path
    if not config_paths and not pkg_path:
        logger.error("Must provide a job specification:")
        logger.error("  --config PATH (recommended: sleap-nn config YAML)")
        logger.error("  --pkg-path PATH (deprecated: training package)")
        sys.exit(1)

    # Deprecation warning for --pkg-path
    if pkg_path:
        logger.warning("=" * 60)
        logger.warning("DEPRECATION WARNING: --pkg-path is deprecated.")
        logger.warning("Use --config with --labels for the new workflow:")
        logger.warning(
            "  sleap-rtc train --room ROOM --config /path/to/config.yaml --labels /path/to/labels.slp"
        )
        logger.warning("=" * 60)

    # Validation: Must provide either session string OR room credentials
    has_session = session_string is not None
    has_room = room_id is not None

    if has_session and has_room:
        logger.error("Connection modes are mutually exclusive. Use only one of:")
        logger.error("  --session-string (direct connection)")
        logger.error("  --room (room-based discovery)")
        sys.exit(1)

    if not has_session and not has_room:
        logger.error("Must provide a connection method:")
        logger.error("  --session-string SESSION (direct connection)")
        logger.error("  --room ROOM (room-based discovery, requires login)")
        sys.exit(1)

    # Room-based connection requires JWT authentication
    jwt_token = None
    if has_room:
        jwt_token = require_login()

    # Validation: worker selection options require room-id
    if (worker_id or auto_select) and not room_id:
        logger.error("--worker-id and --auto-select require --room")
        sys.exit(1)

    # Validation: worker-id and auto-select are mutually exclusive
    if worker_id and auto_select:
        logger.error("Cannot use both --worker-id and --auto-select")
        sys.exit(1)

    # Setup ZMQ ports
    logger.info(f"Using controller port: {kwargs['controller_port']}")
    logger.info(f"Using publish port: {kwargs['publish_port']}")
    kwargs["zmq_ports"] = dict()
    kwargs["zmq_ports"]["controller"] = kwargs.pop("controller_port")
    kwargs["zmq_ports"]["publish"] = kwargs.pop("publish_port")

    # Handle room-based connection
    if room_id:
        logger.info(f"Room-based connection: room_id={room_id}")
        kwargs["room_id"] = room_id
        kwargs["token"] = token or ""  # Token optional with JWT auth
        kwargs["jwt_token"] = jwt_token

        if worker_id:
            logger.info(f"Direct worker connection: worker_id={worker_id}")
            kwargs["worker_id"] = worker_id
        elif auto_select:
            logger.info("Auto-select mode enabled")
            kwargs["auto_select"] = True
        else:
            logger.info("Interactive worker selection mode")

        if min_gpu_memory:
            logger.info(f"Minimum GPU memory filter: {min_gpu_memory}MB")
            kwargs["min_gpu_memory"] = min_gpu_memory

    # Log path resolution options
    if worker_path:
        logger.info(f"Using explicit worker path: {worker_path}")
    if non_interactive:
        logger.info("Non-interactive mode: will auto-select best match")
    if mount_label:
        logger.info(f"Using mount filter: {mount_label}")

    # Branch based on job specification type
    if config_paths:
        # New structured job submission flow
        if len(config_paths) == 1:
            logger.info(f"Using structured job submission with 1 config")
        else:
            logger.info(
                f"Using structured job submission with {len(config_paths)} configs (multi-model training)"
            )
        job_spec = TrainJobSpec(
            config_paths=list(config_paths),
            labels_path=labels_path,
            val_labels_path=val_labels_path,
            max_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            run_name=run_name,
            resume_ckpt_path=resume_ckpt_path,
        )
        return run_job_submit(
            job_spec=job_spec,
            session_string=session_string,
            room_id=room_id,
            token=token,
            worker_id=worker_id,
            auto_select=auto_select,
            min_gpu_memory=min_gpu_memory,
            room_secret=room_secret,
            jwt_token=jwt_token,
            verbosity=verbosity,
        )
    else:
        # Legacy pkg-path flow
        zmq_ports = kwargs.pop("zmq_ports")
        return run_RTCclient(
            session_string=session_string,
            pkg_path=pkg_path,
            zmq_ports=zmq_ports,
            worker_path=worker_path,
            non_interactive=non_interactive,
            mount_label=mount_label,
            room_secret=room_secret,
            room_id=room_id,
            token=token,
            worker_id=worker_id,
            auto_select=auto_select,
            min_gpu_memory=min_gpu_memory,
            jwt_token=jwt_token,
            verbosity=verbosity,
        )


@cli.command(name="track")
@click.option(
    "--session-string",
    "--session_string",
    "-s",
    type=str,
    required=False,
    help="Session string for direct connection to a specific worker.",
)
@click.option(
    "--room",
    "--room-id",
    "-r",
    type=str,
    required=False,
    help="Room ID for room-based worker discovery. Requires login (sleap-rtc login).",
)
@click.option(
    "--worker-id",
    "-w",
    type=str,
    required=False,
    help="Specific worker peer-id to connect to (skips discovery).",
)
@click.option(
    "--auto-select",
    "-a",
    is_flag=True,
    default=False,
    help="Automatically select best worker by GPU memory (use with --room).",
)
@click.option(
    "--data-path",
    "--data_path",
    "-d",
    type=str,
    required=True,
    help="Path to .slp file on worker filesystem for inference.",
)
@click.option(
    "--model-paths",
    "--model_paths",
    "-m",
    multiple=True,
    required=True,
    help="Paths to trained model directories on worker filesystem.",
)
@click.option(
    "--output",
    "-o",
    type=str,
    default="predictions.slp",
    help="Output predictions filename.",
)
@click.option(
    "--only-suggested-frames",
    "--only_suggested_frames",
    is_flag=True,
    default=False,
    help="Track only suggested frames.",
)
@click.option(
    "--batch-size",
    type=int,
    required=False,
    help="Batch size for inference.",
)
@click.option(
    "--peak-threshold",
    type=float,
    required=False,
    help="Peak detection threshold (0.0-1.0).",
)
@click.option(
    "--frames",
    type=str,
    required=False,
    help="Frame range string (e.g., '0-100' or '0-100,200-300').",
)
@click.option(
    "--min-gpu-memory",
    type=int,
    required=False,
    default=None,
    help="Minimum GPU memory in MB required for inference.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show detailed logs including keep-alive, ICE state, and file transfers.",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    default=False,
    help="Show only errors and final status (hide progress bars and status messages).",
)
@click.option(
    "--room-secret",
    type=str,
    envvar="SLEAP_ROOM_SECRET",
    required=False,
    help="Room secret for P2P authentication. Can also use SLEAP_ROOM_SECRET env var.",
)
def track(**kwargs):
    """Run remote inference on a worker with pre-trained models.

    Connection modes (mutually exclusive):

    1. Session string (direct): --session-string SESSION
       Connect directly to a specific worker using its session string.

    2. Room-based discovery: --room ROOM
       Join a room and discover available workers. Supports:
       - Interactive selection (default)
       - Auto-select: --auto-select
       - Direct worker: --worker-id PEER_ID
       - GPU filter: --min-gpu-memory MB

    Note: Room-based discovery requires authentication. Run 'sleap-rtc login' first.

    All paths (--data-path, --model-paths, --output) refer to locations on the
    worker filesystem. Use shared storage or ensure files exist on the worker.

    Examples:
    \b
    # Basic inference with single model
    sleap-rtc track --room my-room \\
        --data-path /vast/data/labels.slp \\
        --model-paths /vast/models/centroid

    \b
    # Multi-model inference (top-down pipeline)
    sleap-rtc track --room my-room \\
        --data-path /vast/data/labels.slp \\
        --model-paths /vast/models/centroid \\
        --model-paths /vast/models/centered_instance \\
        --output /vast/output/predictions.slp

    \b
    # Inference with options
    sleap-rtc track --room my-room \\
        --data-path /vast/data/labels.slp \\
        --model-paths /vast/models/topdown \\
        --batch-size 16 --peak-threshold 0.3

    \b
    # Only process specific frames
    sleap-rtc track --room my-room \\
        --data-path /vast/data/labels.slp \\
        --model-paths /vast/models/centroid \\
        --frames "0-100,500-600" --only-suggested-frames
    """
    # Extract connection options
    session_string = kwargs.pop("session_string", None)
    room_id = kwargs.pop("room", None)
    token = kwargs.pop("token", None)
    worker_id = kwargs.pop("worker_id", None)
    auto_select = kwargs.pop("auto_select", False)
    min_gpu_memory = kwargs.pop("min_gpu_memory", None)

    # Extract P2P authentication options
    room_secret = kwargs.pop("room_secret", None)

    # Extract verbosity options
    verbose = kwargs.pop("verbose", False)
    quiet = kwargs.pop("quiet", False)

    # Validate: --verbose and --quiet are mutually exclusive
    if verbose and quiet:
        logger.error("--verbose and --quiet are mutually exclusive.")
        sys.exit(1)

    # Configure logging based on verbosity
    if quiet:
        verbosity = "quiet"
        logging.getLogger().setLevel(logging.ERROR)
    elif verbose:
        verbosity = "verbose"
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        verbosity = "default"
        logging.getLogger().setLevel(logging.INFO)

    # Extract job specification options
    data_path = kwargs.pop("data_path")
    model_paths = list(kwargs.pop("model_paths"))
    output_path = kwargs.pop("output")
    batch_size = kwargs.pop("batch_size", None)
    peak_threshold = kwargs.pop("peak_threshold", None)
    only_suggested_frames = kwargs.pop("only_suggested_frames", False)
    frames = kwargs.pop("frames", None)

    # Validation: Must provide either session string OR room credentials
    has_session = session_string is not None
    has_room = room_id is not None

    if has_session and has_room:
        logger.error("Connection modes are mutually exclusive. Use only one of:")
        logger.error("  --session-string (direct connection)")
        logger.error("  --room (room-based discovery)")
        sys.exit(1)

    if not has_session and not has_room:
        logger.error("Must provide a connection method:")
        logger.error("  --session-string SESSION (direct connection)")
        logger.error("  --room ROOM (room-based discovery, requires login)")
        sys.exit(1)

    # Room-based connection requires JWT authentication
    jwt_token = None
    if has_room:
        jwt_token = require_login()

    # Validation: worker selection options require room-id
    if (worker_id or auto_select) and not room_id:
        logger.error("--worker-id and --auto-select require --room")
        sys.exit(1)

    # Validation: worker-id and auto-select are mutually exclusive
    if worker_id and auto_select:
        logger.error("Cannot use both --worker-id and --auto-select")
        sys.exit(1)

    logger.info(f"Running inference with data: {data_path}")
    logger.info(f"Using models: {model_paths}")

    # Build TrackJobSpec for structured job submission
    job_spec = TrackJobSpec(
        data_path=data_path,
        model_paths=model_paths,
        output_path=output_path,
        batch_size=batch_size,
        peak_threshold=peak_threshold,
        only_suggested_frames=only_suggested_frames,
        frames=frames,
    )

    return run_job_submit(
        job_spec=job_spec,
        session_string=session_string,
        room_id=room_id,
        token=token,
        worker_id=worker_id,
        auto_select=auto_select,
        min_gpu_memory=min_gpu_memory,
        room_secret=room_secret,
        jwt_token=jwt_token,
        verbosity=verbosity,
    )


# =============================================================================
# Deprecated Command Aliases
# =============================================================================


@cli.command(name="client-train", hidden=True)
@click.option("--session-string", "--session_string", "-s", type=str, required=False)
@click.option("--room", "--room-id", "-r", type=str, required=False)
@click.option("--worker-id", "-w", type=str, required=False)
@click.option("--auto-select", "-a", is_flag=True, default=False)
@click.option(
    "--pkg-path",
    "--pkg_path",
    "-p",
    type=str,
    required=True,
    help="Path resolved on worker",
)
@click.option(
    "--controller-port", "--controller_port", type=int, required=False, default=9000
)
@click.option(
    "--publish-port", "--publish_port", type=int, required=False, default=9001
)
@click.option("--min-gpu-memory", type=int, required=False, default=None)
@click.option("--worker-path", type=str, required=False, default=None)
@click.option("--non-interactive", is_flag=True, default=False)
@click.option("--mount", type=str, required=False, default=None)
@click.option("--room-secret", type=str, envvar="SLEAP_ROOM_SECRET", required=False)
@click.pass_context
def client_train_deprecated(ctx, **kwargs):
    """[DEPRECATED] Use 'sleap-rtc train' instead."""
    click.echo(
        click.style("Warning: ", fg="yellow", bold=True)
        + "'sleap-rtc client-train' is deprecated. Use 'sleap-rtc train' instead."
    )
    ctx.invoke(train, **kwargs)


@cli.command(name="client", hidden=True)
@click.option("--session-string", "--session_string", "-s", type=str, required=False)
@click.option("--room", "--room-id", "-r", type=str, required=False)
@click.option("--worker-id", "-w", type=str, required=False)
@click.option("--auto-select", "-a", is_flag=True, default=False)
@click.option(
    "--pkg-path",
    "--pkg_path",
    "-p",
    type=str,
    required=True,
    help="Path resolved on worker",
)
@click.option(
    "--controller-port", "--controller_port", type=int, required=False, default=9000
)
@click.option(
    "--publish-port", "--publish_port", type=int, required=False, default=9001
)
@click.option("--min-gpu-memory", type=int, required=False, default=None)
@click.option("--worker-path", type=str, required=False, default=None)
@click.option("--non-interactive", is_flag=True, default=False)
@click.option("--mount", type=str, required=False, default=None)
@click.option("--room-secret", type=str, envvar="SLEAP_ROOM_SECRET", required=False)
@click.pass_context
def client_deprecated(ctx, **kwargs):
    """[DEPRECATED] Use 'sleap-rtc train' instead."""
    click.echo(
        click.style("Warning: ", fg="yellow", bold=True)
        + "'sleap-rtc client' is deprecated. Use 'sleap-rtc train' instead."
    )
    ctx.invoke(train, **kwargs)


@cli.command(name="client-track", hidden=True)
@click.option("--session-string", "--session_string", "-s", type=str, required=False)
@click.option("--room", "--room-id", "-r", type=str, required=False)
@click.option("--worker-id", "-w", type=str, required=False)
@click.option("--auto-select", "-a", is_flag=True, default=False)
@click.option(
    "--data-path",
    "--data_path",
    "-d",
    type=str,
    required=True,
    help="Local path, transferred to worker",
)
@click.option(
    "--model-paths",
    "--model_paths",
    "-m",
    multiple=True,
    required=True,
    help="Local paths, transferred to worker",
)
@click.option("--output", "-o", type=str, default="predictions.slp")
@click.option(
    "--only-suggested-frames", "--only_suggested_frames", is_flag=True, default=True
)
@click.option("--min-gpu-memory", type=int, required=False, default=None)
@click.option("--room-secret", type=str, envvar="SLEAP_ROOM_SECRET", required=False)
@click.pass_context
def client_track_deprecated(ctx, **kwargs):
    """[DEPRECATED] Use 'sleap-rtc track' instead."""
    click.echo(
        click.style("Warning: ", fg="yellow", bold=True)
        + "'sleap-rtc client-track' is deprecated. Use 'sleap-rtc track' instead."
    )
    ctx.invoke(track, **kwargs)


# =============================================================================
# Test Command Group (Experimental Features)
# =============================================================================


@cli.group()
def test():
    """Experimental file browser tools.

    These commands provide web-based tools for browsing worker filesystems
    and resolving file paths. They are considered experimental and may change.

    Available commands:
        browse         - Browse a worker's filesystem via web UI
        resolve-paths  - Resolve missing video paths in SLP files
    """
    pass


@test.command(name="browse")
@click.option(
    "--room",
    "--room-id",  # Alias for backward compatibility
    "-r",
    type=str,
    required=True,
    help="Room ID to connect to (required).",
)
@click.option(
    "--port",
    "-p",
    type=int,
    default=8765,
    help="Local port for the file browser server (default: 8765).",
)
@click.option(
    "--no-browser",
    is_flag=True,
    default=False,
    help="Don't auto-open browser (just print URL).",
)
@click.option(
    "--room-secret",
    type=str,
    envvar="SLEAP_ROOM_SECRET",
    required=False,
    help="[Legacy] Room secret for PSK authentication. Account-key auth is now preferred.",
)
def test_browse(room, port, no_browser, room_secret):
    """Browse a Worker's filesystem via web UI.

    [bold yellow]EXPERIMENTAL[/bold yellow]: This feature is under development.

    This command connects to a Worker in the specified room and starts
    a local web server that provides a browser-based file explorer.

    The browser UI allows you to:
    - Browse mount points and directories on the Worker
    - View file information (name, size, type)
    - Copy file paths for use with --worker-path

    Requires authentication. Run 'sleap-rtc login' first.
    P2P auth uses your account key (Ed25519) automatically if saved via
    'sleap-rtc key create --save' or 'sleap-rtc key use'.

    Examples:
        # Connect to a Worker and open browser
        sleap-rtc test browse --room my-room

        # Use a different port
        sleap-rtc test browse --room my-room --port 9000

        # Print URL without opening browser (for remote access)
        sleap-rtc test browse --room my-room --no-browser
    """
    import asyncio
    from sleap_rtc.rtc_browse import run_browse_client

    # Require JWT authentication (client will fetch JWT from credentials)
    require_login()

    logger.info(f"Starting filesystem browser for room: {room}")
    logger.info(f"Local server will run on port: {port}")

    try:
        asyncio.run(
            run_browse_client(
                room_id=room,
                token="",
                port=port,
                open_browser=not no_browser,
                room_secret=room_secret,
            )
        )
    except KeyboardInterrupt:
        logger.info("Browse session ended by user")
    except Exception as e:
        logger.error(f"Browse error: {e}")
        sys.exit(1)


@test.command(name="resolve-paths")
@click.option(
    "--room",
    "--room-id",  # Alias for backward compatibility
    "-r",
    type=str,
    required=True,
    help="Room ID to connect to (required).",
)
@click.option(
    "--slp",
    "-s",
    type=str,
    required=True,
    help="Path to SLP file on Worker filesystem (required).",
)
@click.option(
    "--port",
    "-p",
    type=int,
    default=8765,
    help="Local port for the resolution UI server (default: 8765).",
)
@click.option(
    "--no-browser",
    is_flag=True,
    default=False,
    help="Don't auto-open browser (just print URL).",
)
@click.option(
    "--room-secret",
    type=str,
    envvar="SLEAP_ROOM_SECRET",
    required=False,
    help="[Legacy] Room secret for PSK authentication. Account-key auth is now preferred.",
)
@click.option(
    "--use-jwt/--no-jwt",
    default=True,
    help="Use JWT authentication (default: yes). Disable for testing.",
)
def test_resolve_paths(room, slp, port, no_browser, room_secret, use_jwt):
    """Resolve missing video paths in an SLP file on a Worker.

    [bold yellow]EXPERIMENTAL[/bold yellow]: This feature is under development.

    This command connects to a Worker and checks if the video paths in an SLP
    file are accessible. If any videos are missing, it launches a web UI that
    allows you to browse the Worker's filesystem and resolve the paths.

    Requires JWT authentication. Run 'sleap-rtc login' first.
    P2P auth uses your account key (Ed25519) automatically if saved via
    'sleap-rtc key create --save' or 'sleap-rtc key use'.

    The resolution process:
    1. Worker checks if videos in the SLP are accessible
    2. If all accessible, prints success message
    3. If videos are missing, launches resolution UI
    4. In UI, you can browse and locate the correct video files
    5. Auto-detection finds other videos in the same directory
    6. Save creates a new SLP with corrected paths

    Examples:
        # Check video paths in an SLP file
        sleap-rtc test resolve-paths --room my-room --slp /mnt/data/project.slp

        # Use a different port
        sleap-rtc test resolve-paths -r my-room -s /mnt/data/project.slp -p 9000

        # Print URL without opening browser
        sleap-rtc test resolve-paths -r my-room -s /mnt/data/project.slp --no-browser
    """
    import asyncio
    from sleap_rtc.rtc_resolve import run_resolve_client

    # Require JWT authentication (unless --no-jwt for testing)
    # The client will fetch JWT from credentials internally
    if use_jwt:
        require_login()

    logger.info(f"Starting video path resolution for: {slp}")
    logger.info(f"Connecting to room: {room}")

    try:
        result = asyncio.run(
            run_resolve_client(
                room_id=room,
                token="",
                slp_path=slp,
                port=port,
                open_browser=not no_browser,
                room_secret=room_secret,
                use_jwt=use_jwt,
            )
        )
        if result:
            logger.info(f"Resolution complete: {result}")
        else:
            logger.warning("Resolution cancelled or failed")
    except KeyboardInterrupt:
        logger.info("Resolution cancelled by user")
    except Exception as e:
        logger.error(f"Resolution error: {e}")
        sys.exit(1)


# =============================================================================
# Utility Commands
# =============================================================================


@cli.command(name="tui")
@click.option(
    "--room",
    "--room-id",
    "-r",
    type=str,
    required=False,
    help="Room ID to connect to directly (bypasses room selection).",
)
@click.option(
    "--room-secret",
    type=str,
    envvar="SLEAP_ROOM_SECRET",
    required=False,
    help="[Legacy] Room secret for PSK authentication. Account-key auth is now preferred.",
)
def tui(room, room_secret):
    """Launch the interactive TUI file browser.

    The TUI (Text User Interface) provides an interactive way to:
    - Log in to SLEAP-RTC (if not already logged in)
    - Select from rooms you have access to
    - Browse worker filesystems with Miller columns navigation
    - Resolve video paths in SLP files

    If you provide --room, it will connect directly to that room,
    bypassing the room selection screen.

    P2P auth uses your account key (Ed25519) automatically if saved via
    'sleap-rtc key create --save' or 'sleap-rtc key use'.

    Examples:
        # Launch TUI with room selection
        sleap-rtc tui

        # Connect directly to a specific room
        sleap-rtc tui --room my-room
    """
    from sleap_rtc.tui.app import TUIApp

    app = TUIApp(
        room_id=room,
        room_secret=room_secret,
    )
    app.run()


@cli.command(name="status")
def status():
    """Display current authentication and configuration status.

    Shows:
    - Current login status and user info
    - JWT token expiration
    - Saved room secrets
    - Credential file location
    - API tokens (if any)

    Example:
        sleap-rtc status
    """
    from sleap_rtc.auth.credentials import (
        get_credentials,
        get_jwt,
        get_user,
        is_logged_in,
        CREDENTIALS_PATH,
    )

    click.echo("")
    click.echo(click.style("SLEAP-RTC Status", bold=True))
    click.echo("=" * 40)

    # Login status
    click.echo("")
    click.echo(click.style("Authentication:", bold=True))
    if is_logged_in():
        user = get_user()
        click.echo(f"  Status:   {click.style('Logged in', fg='green')}")
        click.echo(f"  Username: {user.get('username', 'unknown')}")
        click.echo(f"  User ID:  {user.get('user_id', user.get('id', 'unknown'))}")

        # JWT expiry
        jwt_token = get_jwt()
        if jwt_token:
            try:
                import jwt

                claims = jwt.decode(jwt_token, options={"verify_signature": False})
                exp = claims.get("exp")
                if exp:
                    from datetime import datetime

                    exp_dt = datetime.fromtimestamp(exp)
                    now = datetime.now()
                    if exp_dt > now:
                        remaining = exp_dt - now
                        hours = remaining.total_seconds() / 3600
                        click.echo(
                            f"  JWT expires: {exp_dt.strftime('%Y-%m-%d %H:%M')} ({hours:.1f}h remaining)"
                        )
                    else:
                        click.echo(f"  JWT expires: {click.style('EXPIRED', fg='red')}")
            except Exception:
                pass
    else:
        click.echo(f"  Status: {click.style('Not logged in', fg='yellow')}")
        click.echo("  Run 'sleap-rtc login' to authenticate")

    # Credential file
    click.echo("")
    click.echo(click.style("Credentials File:", bold=True))
    click.echo(f"  Location: {CREDENTIALS_PATH}")
    if CREDENTIALS_PATH.exists():
        click.echo(f"  Status:   {click.style('exists', fg='green')}")
    else:
        click.echo(f"  Status:   {click.style('not found', fg='yellow')}")

    # Room secrets
    creds = get_credentials()
    room_secrets = creds.get("room_secrets", {})
    click.echo("")
    click.echo(click.style("Room Secrets:", bold=True))
    if room_secrets:
        for room_id, secret in room_secrets.items():
            masked = secret[:4] + "..." + secret[-4:] if len(secret) > 8 else "****"
            click.echo(f"  {room_id}: {masked}")
    else:
        click.echo("  No room secrets saved")

    # API tokens
    tokens = creds.get("tokens", {})
    click.echo("")
    click.echo(click.style("API Tokens:", bold=True))
    if tokens:
        for room_id, token_info in tokens.items():
            name = token_info.get("worker_name", "unknown")
            api_key = token_info.get("api_key", "")
            masked_key = api_key[:8] + "..." if len(api_key) > 8 else api_key
            click.echo(f"  {room_id}: {name} ({masked_key})")
    else:
        click.echo("  No API tokens saved")

    # Protocol v1 worker (item 1.5/1.6 — pairing/jobs, no signaling server)
    from sleap_rtc.protocol_v1.cli import print_status_section

    print_status_section()

    click.echo("")


@cli.command(name="doctor")
def doctor():
    """Check system configuration and connectivity.

    Runs diagnostic checks to verify:
    - Python version and environment
    - Required dependencies
    - Network connectivity to signaling server
    - Credential file permissions
    - Configuration file status

    Use this command to troubleshoot connection issues.

    Example:
        sleap-rtc doctor
    """
    import platform

    click.echo("")
    click.echo(click.style("SLEAP-RTC Doctor", bold=True))
    click.echo("=" * 40)

    all_ok = True

    # Python version
    click.echo("")
    click.echo(click.style("Python Environment:", bold=True))
    py_version = platform.python_version()
    py_major, py_minor = map(int, py_version.split(".")[:2])
    if py_major >= 3 and py_minor >= 11:
        click.echo(f"  Python version: {py_version} {click.style('✓', fg='green')}")
    else:
        click.echo(
            f"  Python version: {py_version} {click.style('✗ (requires 3.11+)', fg='red')}"
        )
        all_ok = False

    click.echo(f"  Platform: {platform.system()} {platform.release()}")

    # Check key dependencies
    click.echo("")
    click.echo(click.style("Dependencies:", bold=True))
    deps_to_check = ["aiortc", "websockets", "textual", "rich_click", "requests"]
    for dep in deps_to_check:
        try:
            __import__(dep.replace("-", "_"))
            click.echo(f"  {dep}: {click.style('✓', fg='green')}")
        except ImportError:
            click.echo(f"  {dep}: {click.style('✗ missing', fg='red')}")
            all_ok = False

    # Network connectivity
    click.echo("")
    click.echo(click.style("Network Connectivity:", bold=True))
    from sleap_rtc.config import get_config

    config = get_config()
    server_url = config.get_http_url()
    click.echo(f"  Signaling server: {server_url}")

    try:
        response = requests.get(f"{server_url}/health", timeout=10)
        if response.status_code == 200:
            click.echo(f"  Health check: {click.style('✓ reachable', fg='green')}")
        else:
            click.echo(
                f"  Health check: {click.style(f'✗ HTTP {response.status_code}', fg='yellow')}"
            )
            all_ok = False
    except requests.exceptions.Timeout:
        click.echo(f"  Health check: {click.style('✗ timeout', fg='red')}")
        all_ok = False
    except requests.exceptions.ConnectionError:
        click.echo(f"  Health check: {click.style('✗ connection failed', fg='red')}")
        all_ok = False
    except Exception as e:
        click.echo(f"  Health check: {click.style(f'✗ {e}', fg='red')}")
        all_ok = False

    # Credential file
    click.echo("")
    click.echo(click.style("Credentials:", bold=True))
    from sleap_rtc.auth.credentials import CREDENTIALS_PATH, is_logged_in

    click.echo(f"  File: {CREDENTIALS_PATH}")

    if CREDENTIALS_PATH.exists():
        click.echo(f"  Exists: {click.style('✓', fg='green')}")
        # Check permissions
        import stat

        file_stat = CREDENTIALS_PATH.stat()
        mode = file_stat.st_mode
        if mode & stat.S_IRWXO:  # Others have any permissions
            click.echo(
                f"  Permissions: {click.style('✗ world-readable (insecure)', fg='yellow')}"
            )
        else:
            click.echo(f"  Permissions: {click.style('✓ secure', fg='green')}")

        if is_logged_in():
            click.echo(f"  Logged in: {click.style('✓', fg='green')}")
        else:
            click.echo(f"  Logged in: {click.style('✗ no valid JWT', fg='yellow')}")
    else:
        click.echo(f"  Exists: {click.style('✗ not found', fg='yellow')}")
        click.echo("  Run 'sleap-rtc login' to create credentials")

    # Account key
    from sleap_rtc.auth.credentials import get_account_key

    env_key = os.environ.get("SLEAP_RTC_ACCOUNT_KEY")
    stored_key = get_account_key()
    if env_key:
        masked = env_key[:16] + "..." if len(env_key) > 16 else env_key
        click.echo(
            f"  Account key: {masked} {click.style('✓ (from env var)', fg='green')}"
        )
    elif stored_key:
        masked = stored_key[:16] + "..." if len(stored_key) > 16 else stored_key
        click.echo(
            f"  Account key: {masked} {click.style('✓ (from credentials file)', fg='green')}"
        )
    else:
        click.echo(f"  Account key: {click.style('✗ not found', fg='yellow')}")
        click.echo(
            "  Run 'sleap-rtc login' then 'sleap-rtc key create --save', "
            "or set SLEAP_RTC_ACCOUNT_KEY env var"
        )

    # Config file
    click.echo("")
    click.echo(click.style("Configuration:", bold=True))
    config_file = Path("sleap-rtc.toml")
    if config_file.exists():
        click.echo(
            f"  Config file: {config_file.absolute()} {click.style('✓', fg='green')}"
        )
    else:
        click.echo(f"  Config file: {click.style('using defaults', fg='cyan')}")

    # GPU
    click.echo("")
    click.echo(click.style("GPU:", bold=True))
    try:
        import torch

        if torch.cuda.is_available():
            try:
                from sleap_rtc.worker.capabilities import WorkerCapabilities

                caps = WorkerCapabilities()
                click.echo(f"  GPU: {caps.gpu_model} {click.style('✓', fg='green')}")
                click.echo(f"  Memory: {caps.gpu_memory_mb} MB")
                click.echo(f"  CUDA: {caps.cuda_version}")
            except Exception:
                gpu_name = torch.cuda.get_device_name(0)
                click.echo(f"  GPU: {gpu_name} {click.style('✓', fg='green')}")
        else:
            click.echo(
                f"  GPU: {click.style('⚠ PyTorch installed but CUDA not available', fg='yellow')}"
            )
            click.echo(
                "  Training will use CPU. For GPU support, install PyTorch with CUDA."
            )
    except ImportError:
        click.echo(f"  GPU: {click.style('⚠ PyTorch not installed', fg='yellow')}")
        click.echo(
            "  Install with: pip install torch (or use the sleap-rtc Docker image)"
        )

    # Training dependencies
    click.echo("")
    click.echo(click.style("Training Dependencies:", bold=True))
    try:
        import sleap_nn

        version = getattr(sleap_nn, "__version__", "unknown")
        click.echo(f"  sleap-nn: {version} {click.style('✓', fg='green')}")
    except ImportError:
        click.echo(f"  sleap-nn: {click.style('⚠ not installed', fg='yellow')}")
        click.echo("  Required for training jobs. Install with: pip install sleap-nn")

    # Data mounts & working directory
    click.echo("")
    click.echo(click.style("Data Mounts & Working Directory:", bold=True))
    worker_io = config.get_worker_io_config()

    if worker_io.working_dir:
        wd = Path(worker_io.working_dir)
        if wd.exists() and wd.is_dir():
            click.echo(f"  Working dir: {wd} {click.style('✓', fg='green')}")
        else:
            click.echo(f"  Working dir: {wd} {click.style('✗ not found', fg='red')}")
            all_ok = False
    else:
        click.echo(
            f"  Working dir: {click.style('not configured (using cwd)', fg='cyan')}"
        )

    mounts = worker_io.mounts
    if mounts:
        for mount in mounts:
            mount_path = Path(mount.path)
            label = mount.label or mount.path
            if mount_path.exists() and mount_path.is_dir():
                click.echo(
                    f"  Mount '{label}': {mount.path} {click.style('✓', fg='green')}"
                )
            else:
                click.echo(
                    f"  Mount '{label}': {mount.path} {click.style('✗ not found', fg='red')}"
                )
                all_ok = False
    else:
        click.echo(f"  Mounts: {click.style('none configured', fg='cyan')}")
        click.echo(
            "  Use 'sleap-rtc config add-mount /path \"Label\"' to add data directories"
        )

    # Protocol v1 worker (item 1.5/1.6 — pairing/jobs, no signaling server)
    from sleap_rtc.protocol_v1.cli import print_doctor_section

    if not print_doctor_section():
        all_ok = False

    # Summary
    click.echo("")
    click.echo("=" * 40)
    if all_ok:
        click.echo(click.style("All checks passed! ✓", fg="green", bold=True))
    else:
        click.echo(
            click.style("Some checks failed. Review above.", fg="yellow", bold=True)
        )
    click.echo("")


# =============================================================================
# Credentials Command Group
# =============================================================================


@cli.group()
def credentials():
    """Manage local credentials and secrets.

    View, inspect, and remove locally stored authentication credentials,
    room secrets, and API tokens.

    Unlike 'sleap-rtc token' commands (which manage server-side tokens),
    these commands manage credentials stored in ~/.sleap-rtc/credentials.json.
    """
    pass


@credentials.command(name="list")
def credentials_list():
    """List stored credentials summary.

    Shows a summary of locally stored credentials:
    - Logged-in user (if any)
    - Account key (primary auth credential)
    - Room secrets (redacted, legacy)
    - API tokens (redacted, legacy)

    Use 'sleap-rtc credentials show --reveal' to see full values.

    Example:
        sleap-rtc credentials list
    """
    from sleap_rtc.auth.credentials import (
        get_credentials,
        get_account_key,
        get_user,
        is_logged_in,
        CREDENTIALS_PATH,
    )

    creds = get_credentials()

    click.echo("")
    click.echo(click.style("Stored Credentials", bold=True))
    click.echo("=" * 50)

    # Login status
    click.echo("")
    click.echo(click.style("User:", bold=True))
    if is_logged_in():
        user = get_user()
        click.echo(f"  Username: {user.get('username', 'unknown')}")
        click.echo(f"  User ID:  {user.get('user_id', user.get('id', 'unknown'))}")
    else:
        click.echo("  Not logged in")

    # Account key (primary credential)
    click.echo("")
    click.echo(click.style("Account Key (primary):", bold=True))
    account_key = get_account_key()
    if account_key:
        masked = account_key[:20] + "..." if len(account_key) > 20 else "****"
        click.echo(f"  {masked}")
    else:
        click.echo(
            "  (none) — run 'sleap-rtc key create --save' to set up, or 'sleap-rtc key use slp_acct_xxx...' if you already have one"
        )

    # Room secrets (legacy)
    room_secrets = creds.get("room_secrets", {})
    click.echo("")
    click.echo(click.style("Room Secrets (legacy):", bold=True))
    if room_secrets:
        for room_id in room_secrets:
            click.echo(f"  {room_id}: ****")
    else:
        click.echo("  (none)")

    # API tokens (legacy)
    tokens = creds.get("tokens", {})
    click.echo("")
    click.echo(click.style("API Tokens (legacy):", bold=True))
    if tokens:
        for room_id, token_info in tokens.items():
            name = token_info.get("worker_name", "unknown")
            click.echo(f"  {room_id}: {name} (****)")
    else:
        click.echo("  (none)")

    click.echo("")
    click.echo(f"Credentials file: {CREDENTIALS_PATH}")
    click.echo("")


@credentials.command(name="show")
@click.option(
    "--reveal",
    is_flag=True,
    default=False,
    help="Show full secret values (use with caution).",
)
def credentials_show(reveal):
    """Show detailed credentials.

    By default, secret values are redacted. Use --reveal to show
    the full values (be careful when screen-sharing!).

    Examples:
        sleap-rtc credentials show
        sleap-rtc credentials show --reveal
    """
    from sleap_rtc.auth.credentials import (
        get_credentials,
        get_account_key,
        get_jwt,
        get_user,
        is_logged_in,
        CREDENTIALS_PATH,
    )

    creds = get_credentials()

    click.echo("")
    click.echo(click.style("Stored Credentials (Detailed)", bold=True))
    click.echo("=" * 50)

    # Login status and JWT
    click.echo("")
    click.echo(click.style("User & JWT:", bold=True))
    if is_logged_in():
        user = get_user()
        click.echo(f"  Username: {user.get('username', 'unknown')}")
        click.echo(f"  User ID:  {user.get('user_id', user.get('id', 'unknown'))}")

        jwt_token = get_jwt()
        if jwt_token:
            if reveal:
                click.echo(f"  JWT:      {jwt_token}")
            else:
                # Show first and last few chars
                masked = (
                    jwt_token[:20] + "..." + jwt_token[-10:]
                    if len(jwt_token) > 30
                    else "****"
                )
                click.echo(f"  JWT:      {masked}")

            # Show JWT expiry
            try:
                import jwt

                claims = jwt.decode(jwt_token, options={"verify_signature": False})
                exp = claims.get("exp")
                if exp:
                    exp_dt = datetime.fromtimestamp(exp)
                    now = datetime.now()
                    if exp_dt > now:
                        remaining = exp_dt - now
                        hours = remaining.total_seconds() / 3600
                        click.echo(
                            f"  Expires:  {exp_dt.strftime('%Y-%m-%d %H:%M')} ({hours:.1f}h remaining)"
                        )
                    else:
                        click.echo(f"  Expires:  {click.style('EXPIRED', fg='red')}")
            except Exception:
                pass
    else:
        click.echo("  Not logged in")

    # Account key (primary credential)
    click.echo("")
    click.echo(click.style("Account Key (primary):", bold=True))
    account_key = get_account_key()
    if account_key:
        if reveal:
            click.echo(f"  {account_key}")
        else:
            masked = account_key[:20] + "..." if len(account_key) > 20 else "****"
            click.echo(f"  {masked}")
    else:
        click.echo(
            "  (none) — run 'sleap-rtc key create --save' to set up, or 'sleap-rtc key use slp_acct_xxx...' if you already have one"
        )

    # Room secrets (legacy)
    room_secrets = creds.get("room_secrets", {})
    click.echo("")
    click.echo(click.style("Room Secrets (legacy):", bold=True))
    if room_secrets:
        for room_id, secret in room_secrets.items():
            if reveal:
                click.echo(f"  {room_id}: {secret}")
            else:
                masked = secret[:4] + "..." + secret[-4:] if len(secret) > 8 else "****"
                click.echo(f"  {room_id}: {masked}")
    else:
        click.echo("  (none)")

    # API tokens (legacy)
    tokens = creds.get("tokens", {})
    click.echo("")
    click.echo(click.style("API Tokens (legacy):", bold=True))
    if tokens:
        for room_id, token_info in tokens.items():
            name = token_info.get("worker_name", "unknown")
            api_key = token_info.get("api_key", "")
            if reveal:
                click.echo(f"  {room_id}:")
                click.echo(f"    Worker: {name}")
                click.echo(f"    API Key: {api_key}")
            else:
                masked_key = api_key[:8] + "..." if len(api_key) > 8 else "****"
                click.echo(f"  {room_id}: {name} ({masked_key})")
    else:
        click.echo("  (none)")

    click.echo("")
    click.echo(f"Credentials file: {CREDENTIALS_PATH}")
    if not reveal:
        click.echo("")
        click.echo("Use --reveal to show full secret values")
    click.echo("")


@credentials.command(name="clear")
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Skip confirmation prompt.",
)
def credentials_clear(yes):
    """Clear all stored credentials.

    Removes the entire credentials file, including:
    - JWT and user info (logs you out)
    - All room secrets
    - All API tokens

    This is equivalent to 'sleap-rtc logout' but also removes
    room secrets and API tokens.

    Examples:
        sleap-rtc credentials clear
        sleap-rtc credentials clear --yes
    """
    from sleap_rtc.auth.credentials import (
        clear_credentials,
        get_credentials,
        CREDENTIALS_PATH,
    )

    if not CREDENTIALS_PATH.exists():
        click.echo("No credentials file found")
        return

    # Show what will be deleted
    creds = get_credentials()
    room_secrets = creds.get("room_secrets", {})
    tokens = creds.get("tokens", {})

    click.echo("")
    click.echo("This will remove:")
    click.echo(f"  - JWT and user info")
    click.echo(f"  - {len(room_secrets)} room secret(s)")
    click.echo(f"  - {len(tokens)} API token(s)")
    click.echo("")

    if not yes:
        if not click.confirm("Are you sure you want to clear all credentials?"):
            click.echo("Cancelled")
            return

    clear_credentials()
    click.echo("All credentials cleared")


@credentials.command(name="remove-secret")
@click.option(
    "--room",
    "-r",
    required=True,
    help="Room ID to remove secret for.",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Skip confirmation prompt.",
)
def credentials_remove_secret(room, yes):
    """Remove a stored room secret.

    Removes the P2P authentication secret for a specific room.
    This does not affect the room itself, only the local credential.

    Example:
        sleap-rtc credentials remove-secret --room my-room
    """
    from sleap_rtc.auth.credentials import remove_room_secret, get_room_secret

    # Check if secret exists
    if not get_room_secret(room):
        click.echo(f"No secret found for room: {room}")
        return

    if not yes:
        if not click.confirm(f"Remove secret for room '{room}'?"):
            click.echo("Cancelled")
            return

    if remove_room_secret(room):
        click.echo(f"Removed secret for room: {room}")
    else:
        click.echo(f"Failed to remove secret for room: {room}")


@credentials.command(name="remove-token")
@click.option(
    "--room",
    "-r",
    required=True,
    help="Room ID to remove token for.",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Skip confirmation prompt.",
)
def credentials_remove_token(room, yes):
    """Remove a stored API token.

    Removes the locally stored API token for a specific room.
    This does NOT revoke the token on the server - use
    'sleap-rtc token revoke' for that.

    Example:
        sleap-rtc credentials remove-token --room my-room
    """
    from sleap_rtc.auth.credentials import remove_token, get_credentials

    # Check if token exists
    creds = get_credentials()
    tokens = creds.get("tokens", {})
    if room not in tokens:
        click.echo(f"No token found for room: {room}")
        return

    token_info = tokens[room]
    name = token_info.get("worker_name", "unknown")

    if not yes:
        click.echo(f"Token: {name}")
        click.echo("")
        click.echo(
            click.style("Note:", fg="yellow") + " This only removes the local copy."
        )
        click.echo("To revoke the token on the server, use: sleap-rtc token revoke")
        click.echo("")
        if not click.confirm(f"Remove local token for room '{room}'?"):
            click.echo("Cancelled")
            return

    if remove_token(room):
        click.echo(f"Removed token for room: {room}")
    else:
        click.echo(f"Failed to remove token for room: {room}")


@credentials.command(name="remove-account-key")
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Skip confirmation prompt.",
)
def credentials_remove_account_key(yes):
    """Remove the stored account key.

    Removes the locally stored account key (slp_acct_xxx...) from
    ~/.sleap-rtc/credentials.json. This does NOT revoke the key on the
    server — use 'sleap-rtc key revoke' for that.

    Example:
        sleap-rtc credentials remove-account-key
    """
    from sleap_rtc.auth.credentials import get_account_key, remove_account_key

    account_key = get_account_key()
    if not account_key:
        click.echo("No account key stored in credentials")
        return

    masked = account_key[:20] + "..." if len(account_key) > 20 else "****"
    if not yes:
        click.echo(f"Account key: {masked}")
        click.echo("")
        click.echo(
            click.style("Note:", fg="yellow") + " This only removes the local copy."
        )
        click.echo("To revoke the key on the server, use: sleap-rtc key revoke")
        click.echo("")
        if not click.confirm("Remove stored account key?"):
            click.echo("Cancelled")
            return

    remove_account_key()
    click.echo("Removed account key from credentials")


# =============================================================================
# Config Command Group
# =============================================================================


@cli.group(name="config")
def config_group():
    """Manage SLEAP-RTC configuration.

    View and modify configuration settings including signaling server URLs
    and filesystem mount points for workers.

    Configuration is loaded from (in priority order):
    1. Environment variables (SLEAP_RTC_SIGNALING_WS, SLEAP_RTC_SIGNALING_HTTP)
    2. ./sleap-rtc.toml (current working directory)
    3. ~/.sleap-rtc/config.toml (home directory)
    4. Default values
    """
    pass


@config_group.command(name="show")
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Output as JSON for machine parsing.",
)
def config_show(as_json):
    """Show current configuration.

    Displays the merged configuration from all sources, including:
    - Signaling server URLs
    - Environment settings
    - Worker I/O configuration (mounts, working directory)

    Examples:
        sleap-rtc config show
        sleap-rtc config show --json
    """
    from sleap_rtc.config import get_config, reload_config

    # Force reload to get fresh config
    config = reload_config()

    if as_json:
        # JSON output
        config_dict = {
            "environment": config.environment,
            "signaling_websocket": config.signaling_websocket,
            "signaling_http": config.signaling_http,
            "worker_io": {
                "mounts": [
                    {"path": m.path, "label": m.label}
                    for m in config.get_worker_io_config().mounts
                ],
                "working_dir": config.get_worker_io_config().working_dir,
            },
            "sources": {
                "config_file": (
                    str(config._find_config_file())
                    if config._find_config_file()
                    else None
                ),
                "env_vars": {
                    "SLEAP_RTC_ENV": os.environ.get("SLEAP_RTC_ENV"),
                    "SLEAP_RTC_SIGNALING_WS": os.environ.get("SLEAP_RTC_SIGNALING_WS"),
                    "SLEAP_RTC_SIGNALING_HTTP": os.environ.get(
                        "SLEAP_RTC_SIGNALING_HTTP"
                    ),
                },
            },
        }
        click.echo(json.dumps(config_dict, indent=2))
        return

    # Human-readable output
    click.echo("")
    click.echo(click.style("SLEAP-RTC Configuration", bold=True))
    click.echo("=" * 50)

    # Environment
    click.echo("")
    click.echo(click.style("Environment:", bold=True))
    click.echo(f"  SLEAP_RTC_ENV: {config.environment}")

    # Signaling server
    click.echo("")
    click.echo(click.style("Signaling Server:", bold=True))
    click.echo(f"  WebSocket: {config.signaling_websocket}")
    click.echo(f"  HTTP:      {config.signaling_http}")

    # Check for env var overrides
    ws_env = os.environ.get("SLEAP_RTC_SIGNALING_WS")
    http_env = os.environ.get("SLEAP_RTC_SIGNALING_HTTP")
    if ws_env or http_env:
        click.echo("")
        click.echo(click.style("  (overridden by environment variables)", fg="yellow"))

    # Worker I/O config
    io_config = config.get_worker_io_config()
    click.echo("")
    click.echo(click.style("Worker I/O:", bold=True))

    if io_config.working_dir:
        click.echo(f"  Working directory: {io_config.working_dir}")
    else:
        click.echo("  Working directory: (not set)")

    click.echo("")
    click.echo("  Mounts:")
    if io_config.mounts:
        for mount in io_config.mounts:
            valid = mount.validate()
            status = (
                click.style("✓", fg="green") if valid else click.style("✗", fg="red")
            )
            click.echo(f"    {status} {mount.label}: {mount.path}")
    else:
        click.echo("    (none configured)")

    # Source info
    config_file = config._find_config_file()
    click.echo("")
    click.echo(click.style("Source:", bold=True))
    if config_file:
        click.echo(f"  Config file: {config_file}")
    else:
        click.echo("  Config file: (using defaults)")

    click.echo("")


@config_group.command(name="path")
def config_path():
    """Show configuration file locations.

    Displays the paths where configuration files are searched for
    and indicates which ones exist.

    Configuration files are loaded in priority order:
    1. ./sleap-rtc.toml (current working directory)
    2. ~/.sleap-rtc/config.toml (home directory)

    Example:
        sleap-rtc config path
    """
    cwd_config = Path.cwd() / "sleap-rtc.toml"
    home_config = Path.home() / ".sleap-rtc" / "config.toml"

    click.echo("")
    click.echo(click.style("Configuration File Locations", bold=True))
    click.echo("=" * 50)
    click.echo("")

    # CWD config
    click.echo(click.style("Current directory:", bold=True))
    click.echo(f"  Path: {cwd_config}")
    if cwd_config.exists():
        click.echo(f"  Status: {click.style('exists', fg='green')} (will be used)")
    else:
        click.echo(f"  Status: {click.style('not found', fg='yellow')}")

    click.echo("")

    # Home config
    click.echo(click.style("Home directory:", bold=True))
    click.echo(f"  Path: {home_config}")
    if home_config.exists():
        if cwd_config.exists():
            click.echo(
                f"  Status: {click.style('exists', fg='green')} (shadowed by CWD config)"
            )
        else:
            click.echo(f"  Status: {click.style('exists', fg='green')} (will be used)")
    else:
        click.echo(f"  Status: {click.style('not found', fg='yellow')}")

    click.echo("")
    click.echo(click.style("To create a config file:", bold=True))
    click.echo("  sleap-rtc config init")
    click.echo("  sleap-rtc config init --global")
    click.echo("")


@config_group.command(name="add-mount")
@click.argument("path", type=str)
@click.argument("label", type=str)
@click.option(
    "--global",
    "use_global",
    is_flag=True,
    default=False,
    help="Add to ~/.sleap-rtc/config.toml instead of ./sleap-rtc.toml",
)
def config_add_mount(path, label, use_global):
    """Add a filesystem mount point.

    Adds a mount point to the configuration file for Worker filesystem
    browsing. Workers use mounts to expose specific directories to clients.

    PATH is the absolute path to the directory.
    LABEL is a human-readable name for the mount.

    Examples:
        sleap-rtc config add-mount /vast/data "Data Storage"
        sleap-rtc config add-mount /mnt/nfs "NFS Share" --global
    """
    import tomli_w

    # Determine which config file to modify
    if use_global:
        config_file = Path.home() / ".sleap-rtc" / "config.toml"
    else:
        config_file = Path.cwd() / "sleap-rtc.toml"

    # Load existing config or create empty
    if config_file.exists():
        with open(config_file, "rb") as f:
            config_data = tomllib.load(f)
    else:
        config_data = {}

    # Ensure worker.io.mounts structure exists
    if "worker" not in config_data:
        config_data["worker"] = {}
    if "io" not in config_data["worker"]:
        config_data["worker"]["io"] = {}
    if "mounts" not in config_data["worker"]["io"]:
        config_data["worker"]["io"]["mounts"] = []

    # Check if label already exists
    for mount in config_data["worker"]["io"]["mounts"]:
        if mount.get("label") == label:
            click.echo(f"Mount with label '{label}' already exists")
            click.echo(f"  Current path: {mount.get('path')}")
            if not click.confirm("Replace it?"):
                click.echo("Cancelled")
                return
            mount["path"] = path
            break
    else:
        # Add new mount
        config_data["worker"]["io"]["mounts"].append(
            {
                "path": path,
                "label": label,
            }
        )

    # Ensure directory exists
    config_file.parent.mkdir(parents=True, exist_ok=True)

    # Write config file
    with open(config_file, "wb") as f:
        tomli_w.dump(config_data, f)

    click.echo(f"Added mount: {label} -> {path}")
    click.echo(f"Config file: {config_file}")


@config_group.command(name="remove-mount")
@click.argument("label", type=str)
@click.option(
    "--global",
    "use_global",
    is_flag=True,
    default=False,
    help="Remove from ~/.sleap-rtc/config.toml instead of ./sleap-rtc.toml",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Skip confirmation prompt.",
)
def config_remove_mount(label, use_global, yes):
    """Remove a filesystem mount point.

    Removes a mount point from the configuration file by its label.

    Examples:
        sleap-rtc config remove-mount "Data Storage"
        sleap-rtc config remove-mount "NFS Share" --global
    """
    import tomli_w

    # Determine which config file to modify
    if use_global:
        config_file = Path.home() / ".sleap-rtc" / "config.toml"
    else:
        config_file = Path.cwd() / "sleap-rtc.toml"

    if not config_file.exists():
        click.echo(f"Config file not found: {config_file}")
        return

    # Load config
    with open(config_file, "rb") as f:
        config_data = tomllib.load(f)

    # Find and remove mount
    mounts = config_data.get("worker", {}).get("io", {}).get("mounts", [])
    original_count = len(mounts)

    mount_to_remove = None
    for mount in mounts:
        if mount.get("label") == label:
            mount_to_remove = mount
            break

    if not mount_to_remove:
        click.echo(f"Mount with label '{label}' not found")
        return

    if not yes:
        click.echo(f"Mount: {label}")
        click.echo(f"  Path: {mount_to_remove.get('path')}")
        if not click.confirm("Remove this mount?"):
            click.echo("Cancelled")
            return

    mounts.remove(mount_to_remove)
    config_data["worker"]["io"]["mounts"] = mounts

    # Write config file
    with open(config_file, "wb") as f:
        tomli_w.dump(config_data, f)

    click.echo(f"Removed mount: {label}")
    click.echo(f"Config file: {config_file}")


@config_group.command(name="init")
@click.option(
    "--global",
    "use_global",
    is_flag=True,
    default=False,
    help="Create ~/.sleap-rtc/config.toml instead of ./sleap-rtc.toml",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    help="Overwrite existing config file.",
)
def config_init(use_global, force):
    """Create a new configuration file with defaults.

    Creates a sample configuration file that you can customize.

    Examples:
        sleap-rtc config init
        sleap-rtc config init --global
        sleap-rtc config init --force
    """
    import tomli_w

    # Determine which config file to create
    if use_global:
        config_file = Path.home() / ".sleap-rtc" / "config.toml"
    else:
        config_file = Path.cwd() / "sleap-rtc.toml"

    if config_file.exists() and not force:
        click.echo(f"Config file already exists: {config_file}")
        click.echo("Use --force to overwrite")
        return

    # Create sample config
    # Note: TOML doesn't support None, so we omit optional fields
    sample_config = {
        "worker": {
            "io": {
                "mounts": [
                    {"path": "/data", "label": "Data"},
                ],
            },
        },
        "environments": {
            "production": {
                "signaling_websocket": "ws://ec2-52-9-213-137.us-west-1.compute.amazonaws.com:8080",
                "signaling_http": "http://ec2-52-9-213-137.us-west-1.compute.amazonaws.com:8001",
            },
            "development": {
                "signaling_websocket": "ws://localhost:8080",
                "signaling_http": "http://localhost:8001",
            },
        },
    }

    # Ensure directory exists
    config_file.parent.mkdir(parents=True, exist_ok=True)

    # Write config file
    with open(config_file, "wb") as f:
        tomli_w.dump(sample_config, f)

    click.echo(f"Created config file: {config_file}")
    click.echo("")
    click.echo("Edit this file to customize your configuration.")
    click.echo("See 'sleap-rtc config show' to view current settings.")


# Path Mapping subcommands


@config_group.command(name="add-path-mapping")
@click.option("--local", "local_path", required=True, help="Local directory prefix.")
@click.option("--worker", "worker_path", required=True, help="Worker directory prefix.")
def config_add_path_mapping(local_path, worker_path):
    """Add a local→worker directory prefix mapping.

    Saves a path mapping to ~/.sleap-rtc/config.toml so that the worker path
    field is auto-filled in future job submissions.

    Examples:
        sleap-rtc config add-path-mapping --local /Users/alice/data --worker /root/vast/data
    """
    from sleap_rtc.config import get_config

    config = get_config()
    config.save_path_mapping(local_path, worker_path)
    click.echo(f"Saved path mapping: {local_path} → {worker_path}")


@config_group.command(name="remove-path-mapping")
@click.option("--local", "local_path", required=True, help="Local directory prefix.")
@click.option("--worker", "worker_path", required=True, help="Worker directory prefix.")
def config_remove_path_mapping(local_path, worker_path):
    """Remove a local→worker directory prefix mapping.

    Removes a matching mapping from ~/.sleap-rtc/config.toml.

    Examples:
        sleap-rtc config remove-path-mapping --local /Users/alice/data --worker /root/vast/data
    """
    from sleap_rtc.config import get_config

    config = get_config()
    mappings_before = config.get_path_mappings()
    config.remove_path_mapping(local_path, worker_path)
    mappings_after = config.get_path_mappings()
    if len(mappings_after) < len(mappings_before):
        click.echo(f"Removed path mapping: {local_path} → {worker_path}")
    else:
        click.echo(
            click.style("Warning: ", fg="yellow")
            + f"No matching mapping found for {local_path} → {worker_path}"
        )


@config_group.command(name="list-path-mappings")
def config_list_path_mappings():
    """List all saved local→worker directory prefix mappings.

    Reads mappings from ~/.sleap-rtc/config.toml and prints them.

    Example:
        sleap-rtc config list-path-mappings
    """
    from sleap_rtc.config import get_config

    config = get_config()
    mappings = config.get_path_mappings()
    if not mappings:
        click.echo("No path mappings configured.")
        return
    for m in mappings:
        click.echo(f"{m.local} → {m.worker}")


# =============================================================================
# Deprecated Top-Level Aliases for Test Commands
# =============================================================================


@cli.command(name="browse", hidden=True)
@click.option("--room", "--room-id", "-r", type=str, required=True)
@click.option("--port", "-p", type=int, default=8765)
@click.option("--no-browser", is_flag=True, default=False)
@click.option("--room-secret", type=str, envvar="SLEAP_ROOM_SECRET", required=False)
@click.pass_context
def browse_deprecated(ctx, **kwargs):
    """[DEPRECATED] Use 'sleap-rtc test browse' instead."""
    click.echo(
        click.style("Warning: ", fg="yellow", bold=True)
        + "'sleap-rtc browse' is deprecated. Use 'sleap-rtc test browse' instead."
    )
    ctx.invoke(test_browse, **kwargs)


@cli.command(name="resolve-paths", hidden=True)
@click.option("--room", "--room-id", "-r", type=str, required=True)
@click.option("--slp", "-s", type=str, required=True)
@click.option("--port", "-p", type=int, default=8765)
@click.option("--no-browser", is_flag=True, default=False)
@click.option("--room-secret", type=str, envvar="SLEAP_ROOM_SECRET", required=False)
@click.option("--use-jwt/--no-jwt", default=True)
@click.pass_context
def resolve_paths_deprecated(ctx, **kwargs):
    """[DEPRECATED] Use 'sleap-rtc test resolve-paths' instead."""
    click.echo(
        click.style("Warning: ", fg="yellow", bold=True)
        + "'sleap-rtc resolve-paths' is deprecated. Use 'sleap-rtc test resolve-paths' instead."
    )
    ctx.invoke(test_resolve_paths, **kwargs)


# =============================================================================
# Protocol v1 Worker Commands (items 1.5/1.6 — pairing/jobs, no signaling)
# =============================================================================

from sleap_rtc.protocol_v1.cli import pair as protocol_v1_pair
from sleap_rtc.protocol_v1.cli import serve as protocol_v1_serve

cli.add_command(protocol_v1_serve)
cli.add_command(protocol_v1_pair)


if __name__ == "__main__":
    cli()
