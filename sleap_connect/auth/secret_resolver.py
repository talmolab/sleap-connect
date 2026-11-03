"""Secret resolution for P2P PSK authentication.

This module provides a unified interface for resolving room secrets from
multiple sources in priority order:

1. CLI flag value (explicit override)
2. SLEAP_ROOM_SECRET environment variable
3. Filesystem path (shared storage for HPC setups)
4. Credentials file (~/.sleap-connect/credentials.json)

The filesystem path is configurable via SLEAP_SECRET_PATH environment variable,
defaulting to ~/.sleap-connect/room-secrets/.
"""

import os
from pathlib import Path
from typing import Optional

from loguru import logger

from sleap_connect.auth.credentials import get_room_secret as get_credential_secret

# Environment variable names
ENV_ROOM_SECRET = "SLEAP_ROOM_SECRET"
ENV_SECRET_PATH = "SLEAP_SECRET_PATH"

# Default filesystem path for room secrets
DEFAULT_SECRET_PATH = Path.home() / ".sleap-connect" / "room-secrets"


def get_secret_base_path() -> Path:
    """Get the base path for filesystem-based room secrets.

    Returns:
        Path from SLEAP_SECRET_PATH env var, or default ~/.sleap-connect/room-secrets/
    """
    env_path = os.environ.get(ENV_SECRET_PATH)
    if env_path:
        return Path(env_path).expanduser()
    return DEFAULT_SECRET_PATH


def resolve_secret(
    room_id: str,
    cli_secret: Optional[str] = None,
) -> Optional[str]:
    """Resolve a room secret from multiple sources in priority order.

    Lookup order:
    1. cli_secret parameter (explicit CLI flag)
    2. SLEAP_ROOM_SECRET environment variable
    3. Filesystem: {SLEAP_SECRET_PATH}/{room_id} or ~/.sleap-connect/room-secrets/{room_id}
    4. Credentials file: ~/.sleap-connect/credentials.json → room_secrets.{room_id}

    Args:
        room_id: The room ID to resolve the secret for.
        cli_secret: Optional secret provided via CLI flag (highest priority).

    Returns:
        The room secret if found from any source, None otherwise.

    Example:
        >>> # CLI flag takes priority
        >>> resolve_secret("room-123", cli_secret="my_secret")
        'my_secret'

        >>> # Falls back to env var, filesystem, then credentials
        >>> resolve_secret("room-123")
        None  # or secret if found
    """
    # 1. CLI flag (highest priority)
    if cli_secret:
        logger.debug(f"Using room secret from CLI flag for room {room_id}")
        return cli_secret

    # 2. Environment variable
    env_secret = os.environ.get(ENV_ROOM_SECRET)
    if env_secret:
        logger.debug(
            f"Using room secret from {ENV_ROOM_SECRET} env var for room {room_id}"
        )
        return env_secret

    # 3. Filesystem path
    fs_secret = _read_filesystem_secret(room_id)
    if fs_secret:
        logger.debug(f"Using room secret from filesystem for room {room_id}")
        return fs_secret

    # 4. Credentials file
    cred_secret = get_credential_secret(room_id)
    if cred_secret:
        logger.debug(f"Using room secret from credentials file for room {room_id}")
        return cred_secret

    # No secret found
    logger.debug(f"No room secret found for room {room_id}")
    return None


def _read_filesystem_secret(room_id: str) -> Optional[str]:
    """Read a room secret from the filesystem.

    The secret file path is: {base_path}/{room_id}
    where base_path is from SLEAP_SECRET_PATH env var or ~/.sleap-connect/room-secrets/

    The file should contain only the base64-encoded secret (whitespace trimmed).

    Args:
        room_id: The room ID to read the secret for.

    Returns:
        The secret string if file exists and is readable, None otherwise.
    """
    base_path = get_secret_base_path()
    secret_file = base_path / room_id

    if not secret_file.exists():
        return None

    try:
        content = secret_file.read_text().strip()
        if content:
            return content
        return None
    except (IOError, PermissionError) as e:
        logger.warning(f"Failed to read secret file {secret_file}: {e}")
        return None


def get_secret_sources(
    room_id: str, cli_secret: Optional[str] = None
) -> dict[str, Optional[str]]:
    """Get all potential secret sources for debugging/diagnostics.

    This is useful for showing users where secrets could be configured.

    Args:
        room_id: The room ID to check sources for.
        cli_secret: Optional CLI-provided secret.

    Returns:
        Dictionary mapping source names to their values (or None if not set).
    """
    base_path = get_secret_base_path()
    fs_path = base_path / room_id

    return {
        "cli_flag": cli_secret,
        "env_var": os.environ.get(ENV_ROOM_SECRET),
        "filesystem": str(fs_path) if fs_path.exists() else None,
        "credentials": get_credential_secret(room_id),
    }
