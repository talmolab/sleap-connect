"""Credential file management for sleap-connect.

Stores credentials in ~/.sleap-connect/credentials.json with the schema:
{
    "jwt": "eyJ...",                   # Short-lived, browser-obtained (legacy)
    "account_key": "slp_acct_xxx...", # Long-lived, used for all API calls
    "private_key": "base64...",       # Ed25519 private key (for P2P auth)
    "user": {
        "id": "12345678",
        "username": "researcher1",
        "avatar_url": "https://..."
    },
    "default_room": "room-abc123",    # Optional default room
    "tokens": {                        # Legacy per-room API keys (backward compat)
        "<room_id>": {
            "api_key": "slp_xxx...",
            "worker_name": "lab-gpu-1"
        }
    },
    "room_secrets": {                  # Legacy PSK secrets (backward compat)
        "<room_id>": "base64_encoded_secret..."
    }
}
"""

import json
import os
import stat
from pathlib import Path
from typing import Any, Optional

from loguru import logger

# Credentials file location
CREDENTIALS_DIR = Path.home() / ".sleap-connect"
CREDENTIALS_PATH = CREDENTIALS_DIR / "credentials.json"


def _ensure_credentials_dir() -> None:
    """Ensure the credentials directory exists with proper permissions."""
    if not CREDENTIALS_DIR.exists():
        CREDENTIALS_DIR.mkdir(parents=True, mode=0o700)
        logger.debug(f"Created credentials directory: {CREDENTIALS_DIR}")


def get_credentials() -> dict[str, Any]:
    """Load credentials from file.

    Returns:
        Credentials dictionary, or empty dict if file doesn't exist.
    """
    if not CREDENTIALS_PATH.exists():
        return {}

    try:
        with open(CREDENTIALS_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load credentials: {e}")
        return {}


def save_credentials(credentials: dict[str, Any]) -> None:
    """Save credentials to file with restrictive permissions.

    Args:
        credentials: Credentials dictionary to save.
    """
    _ensure_credentials_dir()

    # Write to temp file first, then rename (atomic)
    temp_path = CREDENTIALS_PATH.with_suffix(".tmp")

    try:
        with open(temp_path, "w") as f:
            json.dump(credentials, f, indent=2)

        # Set restrictive permissions (owner read/write only)
        os.chmod(temp_path, stat.S_IRUSR | stat.S_IWUSR)

        # Atomic replace (os.replace works even if destination exists, unlike
        # Path.rename which raises FileExistsError [WinError 183] on Windows)
        os.replace(temp_path, CREDENTIALS_PATH)
        logger.debug(f"Saved credentials to {CREDENTIALS_PATH}")

    except IOError as e:
        logger.error(f"Failed to save credentials: {e}")
        if temp_path.exists():
            temp_path.unlink()
        raise


def clear_credentials() -> None:
    """Remove the credentials file."""
    if CREDENTIALS_PATH.exists():
        CREDENTIALS_PATH.unlink()
        logger.info("Credentials cleared")
    else:
        logger.debug("No credentials to clear")


def clear_jwt() -> None:
    """Clear only JWT and user info, keeping room secrets and tokens.

    This is useful for logging out while preserving room access credentials.
    """
    creds = get_credentials()
    if not creds:
        logger.debug("No credentials to clear")
        return

    # Remove JWT and user, keep everything else
    creds.pop("jwt", None)
    creds.pop("user", None)

    if creds:
        # Still have other credentials (room secrets, tokens)
        save_credentials(creds)
        logger.info("JWT cleared, other credentials preserved")
    else:
        # Nothing left, remove the file
        clear_credentials()


def get_jwt() -> Optional[str]:
    """Get the stored JWT token.

    Returns:
        JWT string if stored, None otherwise.
    """
    creds = get_credentials()
    return creds.get("jwt")


def get_user() -> Optional[dict[str, Any]]:
    """Get the stored user info.

    Returns:
        User dictionary with id, username, avatar_url if stored, None otherwise.
    """
    creds = get_credentials()
    return creds.get("user")


def get_api_key(room_id: str) -> Optional[str]:
    """Get the API key for a specific room.

    Args:
        room_id: The room ID to get the API key for.

    Returns:
        API key string if stored, None otherwise.
    """
    creds = get_credentials()
    tokens = creds.get("tokens", {})
    room_token = tokens.get(room_id, {})
    return room_token.get("api_key")


def save_jwt(jwt: str, user: dict[str, Any]) -> None:
    """Save JWT and user info to credentials.

    Args:
        jwt: JWT token string.
        user: User info dictionary.
    """
    creds = get_credentials()
    creds["jwt"] = jwt
    creds["user"] = user
    save_credentials(creds)


def save_token(room_id: str, api_key: str, worker_name: str) -> None:
    """Save a worker token for a room.

    Args:
        room_id: Room ID the token is for.
        api_key: API key (slp_xxx...).
        worker_name: Human-readable name for the worker.
    """
    creds = get_credentials()
    if "tokens" not in creds:
        creds["tokens"] = {}

    creds["tokens"][room_id] = {
        "api_key": api_key,
        "worker_name": worker_name,
    }
    save_credentials(creds)


def remove_token(room_id: str) -> bool:
    """Remove a stored token for a room.

    Args:
        room_id: Room ID to remove token for.

    Returns:
        True if token was removed, False if not found.
    """
    creds = get_credentials()
    tokens = creds.get("tokens", {})

    if room_id in tokens:
        del tokens[room_id]
        creds["tokens"] = tokens
        save_credentials(creds)
        return True
    return False


def is_logged_in() -> bool:
    """Check if user is currently logged in.

    Returns:
        True if JWT exists in credentials.
    """
    return get_jwt() is not None


def get_valid_jwt() -> Optional[str]:
    """Get the stored JWT token if it exists and is not expired.

    Returns:
        JWT string if stored and valid, None if missing or expired.
    """
    jwt_token = get_jwt()
    if not jwt_token:
        return None

    try:
        import jwt

        # Decode without verification just to read expiration claim
        claims = jwt.decode(jwt_token, options={"verify_signature": False})
        exp = claims.get("exp")
        if exp:
            import time

            if time.time() >= exp:
                logger.warning("Stored JWT has expired")
                return None
        return jwt_token
    except Exception as e:
        logger.warning(f"Failed to decode JWT: {e}")
        return None


# =============================================================================
# Room Secret Management (PSK P2P Authentication)
# =============================================================================


def get_room_secret(room_id: str) -> Optional[str]:
    """Get the stored room secret for P2P authentication.

    Args:
        room_id: The room ID to get the secret for.

    Returns:
        Base64-encoded secret string if stored, None otherwise.
    """
    creds = get_credentials()
    room_secrets = creds.get("room_secrets", {})
    return room_secrets.get(room_id)


def save_room_secret(room_id: str, secret: str) -> None:
    """Save a room secret for P2P authentication.

    Args:
        room_id: Room ID the secret is for.
        secret: Base64-encoded secret string.
    """
    creds = get_credentials()
    if "room_secrets" not in creds:
        creds["room_secrets"] = {}

    creds["room_secrets"][room_id] = secret
    save_credentials(creds)
    logger.debug(f"Saved room secret for room {room_id}")


# =============================================================================
# Account Key Management (new auth architecture)
# =============================================================================


def get_account_key() -> Optional[str]:
    """Get the account key, preferring SLEAP_RTC_ACCOUNT_KEY env var.

    Returns:
        Account key string (slp_acct_...) if available, None otherwise.
    """
    env_key = os.environ.get("SLEAP_RTC_ACCOUNT_KEY")
    if env_key:
        return env_key
    creds = get_credentials()
    return creds.get("account_key")


def save_account_key(key: str) -> None:
    """Save an account key to credentials.

    Args:
        key: Account key string (slp_acct_...).
    """
    creds = get_credentials()
    creds["account_key"] = key
    save_credentials(creds)
    logger.debug("Saved account key to credentials")


def remove_account_key() -> None:
    """Remove the stored account key from credentials."""
    creds = get_credentials()
    creds.pop("account_key", None)
    save_credentials(creds)
    logger.debug("Removed account key from credentials")


def get_default_room() -> Optional[str]:
    """Get the default room ID, preferring SLEAP_RTC_DEFAULT_ROOM env var.

    Returns:
        Room ID string if set, None otherwise.
    """
    env_room = os.environ.get("SLEAP_RTC_DEFAULT_ROOM")
    if env_room:
        return env_room
    creds = get_credentials()
    return creds.get("default_room")


def save_default_room(room_id: str) -> None:
    """Save a default room ID to credentials.

    Args:
        room_id: Room ID to set as default.
    """
    creds = get_credentials()
    creds["default_room"] = room_id
    save_credentials(creds)
    logger.debug(f"Saved default room: {room_id}")


def get_private_key_b64() -> Optional[str]:
    """Get the stored Ed25519 private key as base64.

    Returns:
        Base64-encoded private key bytes if stored, None otherwise.
    """
    creds = get_credentials()
    return creds.get("private_key")


def save_private_key_b64(private_key_b64: str) -> None:
    """Save an Ed25519 private key (base64-encoded) to credentials.

    Also resets ``public_key_registered`` to False since the new key needs
    registration with the server.

    Args:
        private_key_b64: URL-safe base64-encoded private key bytes.
    """
    creds = get_credentials()
    creds["private_key"] = private_key_b64
    creds["public_key_registered"] = False
    save_credentials(creds)
    logger.debug("Saved private key to credentials")


def get_public_key_registered() -> bool:
    """Check if the local Ed25519 public key has been registered with the server.

    Returns:
        True if registered, False if not registered or unknown.
    """
    creds = get_credentials()
    return creds.get("public_key_registered", False)


def set_public_key_registered(value: bool) -> None:
    """Mark whether the local Ed25519 public key is registered with the server.

    Args:
        value: True if registered, False if not.
    """
    creds = get_credentials()
    creds["public_key_registered"] = value
    save_credentials(creds)
    logger.debug(f"Set public_key_registered = {value}")


def remove_room_secret(room_id: str) -> bool:
    """Remove a stored room secret.

    Args:
        room_id: Room ID to remove secret for.

    Returns:
        True if secret was removed, False if not found.
    """
    creds = get_credentials()
    room_secrets = creds.get("room_secrets", {})

    if room_id in room_secrets:
        del room_secrets[room_id]
        creds["room_secrets"] = room_secrets
        save_credentials(creds)
        logger.debug(f"Removed room secret for room {room_id}")
        return True
    return False
