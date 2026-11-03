"""Configuration management for sleap-connect.

This module handles loading configuration from multiple sources with the following priority:
1. CLI arguments (handled by caller)
2. Environment variables (SLEAP_RTC_SIGNALING_WS, SLEAP_RTC_SIGNALING_HTTP)
3. TOML configuration file
4. Default values (production environment)

Configuration files are loaded from:
- sleap-rtc.toml in current working directory
- ~/.sleap-connect/config.toml

Environment selection via SLEAP_RTC_ENV (development, staging, production).
Defaults to production if not set.
"""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


from loguru import logger


@dataclass
class PathMapping:
    """A local→worker directory prefix mapping.

    Attributes:
        local: Absolute path prefix on the local machine.
        worker: Corresponding absolute path prefix on the worker.
    """

    local: str
    worker: str


@dataclass
class MountConfig:
    """Configuration for a single filesystem mount point.

    Attributes:
        path: Absolute path to the mount point directory.
        label: Human-readable display name for the mount.
    """

    path: str
    label: str

    def __post_init__(self):
        """Validate mount configuration after initialization."""
        if not self.path:
            raise ValueError("Mount path cannot be empty")
        if not self.label:
            raise ValueError("Mount label cannot be empty")

    def validate(self) -> bool:
        """Validate that the mount path exists and is a directory.

        Returns:
            True if valid, False otherwise.
        """
        mount_path = Path(self.path)
        if not mount_path.exists():
            logger.warning(f"Mount path does not exist: {self.path}")
            return False
        if not mount_path.is_dir():
            logger.warning(f"Mount path is not a directory: {self.path}")
            return False
        return True


@dataclass
class WorkerIOConfig:
    """Configuration for Worker I/O settings.

    Attributes:
        mounts: List of configured mount points for filesystem browsing.
        working_dir: Optional working directory for the worker.
    """

    mounts: List[MountConfig] = field(default_factory=list)
    working_dir: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "WorkerIOConfig":
        """Create WorkerIOConfig from TOML dictionary.

        Args:
            data: Dictionary from TOML [worker.io] section.

        Returns:
            WorkerIOConfig instance.
        """
        mounts = []
        mount_data = data.get("mounts", [])

        for mount_entry in mount_data:
            if "path" not in mount_entry or "label" not in mount_entry:
                logger.warning(
                    f"Skipping invalid mount entry (missing path or label): {mount_entry}"
                )
                continue
            try:
                mount = MountConfig(
                    path=mount_entry["path"], label=mount_entry["label"]
                )
                mounts.append(mount)
            except ValueError as e:
                logger.warning(f"Skipping invalid mount entry: {e}")

        working_dir = data.get("working_dir")

        return cls(mounts=mounts, working_dir=working_dir)

    def get_valid_mounts(self) -> List[MountConfig]:
        """Get list of mounts that pass validation.

        Returns:
            List of MountConfig instances with valid paths.
        """
        valid_mounts = []
        for mount in self.mounts:
            if mount.validate():
                valid_mounts.append(mount)
        return valid_mounts


# Default production signaling server URLs
DEFAULT_SIGNALING_WEBSOCKET = "wss://signaling.sleap.ai"
DEFAULT_SIGNALING_HTTP = "https://signaling.sleap.ai"

# Valid environment names
VALID_ENVIRONMENTS = {"development", "staging", "production"}


class Config:
    """Configuration manager for sleap-connect."""

    def __init__(self):
        """Initialize configuration with defaults."""
        self.signaling_websocket: str = DEFAULT_SIGNALING_WEBSOCKET
        self.signaling_http: str = DEFAULT_SIGNALING_HTTP
        self.environment: str = "production"
        self._config_data: dict = {}

    def load(self) -> None:
        """Load configuration from all sources.

        Priority order:
        1. Environment variables (SLEAP_RTC_SIGNALING_WS, SLEAP_RTC_SIGNALING_HTTP)
        2. TOML configuration file
        3. Default values
        """
        # Determine environment
        self.environment = self._get_environment()

        # Try to load config file
        config_file = self._find_config_file()
        if config_file:
            self._load_config_file(config_file)

        # Apply environment variable overrides
        self._apply_env_overrides()

    def _get_environment(self) -> str:
        """Get the current environment from SLEAP_RTC_ENV.

        Returns:
            Environment name (development, staging, or production).
            Defaults to production if not set or invalid.
        """
        env = os.getenv("SLEAP_RTC_ENV", "production").lower()
        if env not in VALID_ENVIRONMENTS:
            logger.warning(
                f"Invalid SLEAP_RTC_ENV value '{env}'. "
                f"Valid values are: {', '.join(VALID_ENVIRONMENTS)}. "
                f"Defaulting to 'production'."
            )
            env = "production"
        return env

    def _find_config_file(self) -> Optional[Path]:
        """Find the configuration file.

        Searches in order:
        1. sleap-rtc.toml in current working directory
        2. ~/.sleap-connect/config.toml

        Returns:
            Path to config file if found, None otherwise.
        """
        # Check current working directory
        cwd_config = Path.cwd() / "sleap-rtc.toml"
        if cwd_config.exists():
            logger.info(f"Loading config from {cwd_config}")
            return cwd_config

        # Check home directory
        home_config = Path.home() / ".sleap-connect" / "config.toml"
        if home_config.exists():
            logger.info(f"Loading config from {home_config}")
            return home_config

        logger.debug("No config file found, using defaults")
        return None

    def _load_config_file(self, config_file: Path) -> None:
        """Load configuration from TOML file.

        Args:
            config_file: Path to the TOML configuration file.
        """
        try:
            with open(config_file, "rb") as f:
                self._config_data = tomllib.load(f)

            # Apply default section settings
            if "default" in self._config_data:
                # Default section for shared settings (future use)
                pass

            # Apply environment-specific settings
            environments = self._config_data.get("environments", {})
            env_config = environments.get(self.environment, {})

            if not env_config:
                logger.debug(
                    f"No configuration found for environment '{self.environment}' "
                    f"in {config_file}, using defaults"
                )
                return

            # Load signaling server URLs
            if "signaling_websocket" in env_config:
                self.signaling_websocket = env_config["signaling_websocket"]
                logger.debug(
                    f"Loaded signaling_websocket from config: {self.signaling_websocket}"
                )

            if "signaling_http" in env_config:
                self.signaling_http = env_config["signaling_http"]
                logger.debug(
                    f"Loaded signaling_http from config: {self.signaling_http}"
                )

        except Exception as e:
            logger.warning(
                f"Failed to load config file {config_file}: {e}. Using defaults."
            )

    def get_worker_io_config(self) -> WorkerIOConfig:
        """Get Worker I/O configuration from loaded config data.

        Returns:
            WorkerIOConfig instance with mounts and working_dir.
        """
        worker_config = self._config_data.get("worker", {})
        io_config = worker_config.get("io", {})
        return WorkerIOConfig.from_dict(io_config)

    # ------------------------------------------------------------------
    # Path mapping persistence
    # ------------------------------------------------------------------

    def _home_config_path(self) -> Path:
        """Return the canonical home config path."""
        return Path.home() / ".sleap-connect" / "config.toml"

    def _read_home_toml(self) -> dict:
        """Read the home config file as a raw dict (empty dict if missing)."""
        path = self._home_config_path()
        if not path.exists():
            return {}
        with open(path, "rb") as f:
            return tomllib.load(f)

    def _write_home_toml(self, data: dict) -> None:
        """Write a raw dict to the home config file, creating it if needed."""
        import tomli_w  # lazy import — only needed for write operations

        path = self._home_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            tomli_w.dump(data, f)

    def get_path_mappings(self) -> List[PathMapping]:
        """Return all saved local→worker directory prefix mappings.

        Returns:
            List of PathMapping objects read from ~/.sleap-connect/config.toml.
        """
        data = self._read_home_toml()
        mappings = []
        for entry in data.get("path_mappings", []):
            if "local" in entry and "worker" in entry:
                mappings.append(
                    PathMapping(local=entry["local"], worker=entry["worker"])
                )
        return mappings

    def save_path_mapping(self, local: str, worker: str) -> None:
        """Append a local→worker mapping to ~/.sleap-connect/config.toml.

        Does nothing if the exact pair already exists.

        Args:
            local: Local directory prefix.
            worker: Corresponding worker directory prefix.
        """
        data = self._read_home_toml()
        mappings = data.get("path_mappings", [])
        for m in mappings:
            if m.get("local") == local and m.get("worker") == worker:
                logger.debug(f"Path mapping already exists: {local} → {worker}")
                return
        mappings.append({"local": local, "worker": worker})
        data["path_mappings"] = mappings
        self._write_home_toml(data)
        logger.info(f"Saved path mapping: {local} → {worker}")

    def remove_path_mapping(self, local: str, worker: str) -> None:
        """Remove a local→worker mapping from ~/.sleap-connect/config.toml.

        Logs a warning if no matching entry is found.

        Args:
            local: Local directory prefix.
            worker: Corresponding worker directory prefix.
        """
        data = self._read_home_toml()
        mappings = data.get("path_mappings", [])
        new_mappings = [
            m
            for m in mappings
            if not (m.get("local") == local and m.get("worker") == worker)
        ]
        if len(new_mappings) == len(mappings):
            logger.warning(f"No path mapping found to remove: {local} → {worker}")
            return
        data["path_mappings"] = new_mappings
        self._write_home_toml(data)
        logger.info(f"Removed path mapping: {local} → {worker}")

    def translate_path(self, local_path: str) -> Optional[str]:
        """Translate a local path to a worker path using saved prefix mappings.

        When multiple prefixes match, the longest (most specific) one wins.

        Args:
            local_path: Absolute path on the local machine.

        Returns:
            Translated worker path, or None if no mapping matches.
        """
        local_p = Path(local_path)
        best_worker_path: Optional[str] = None
        best_prefix_len = -1
        for mapping in self.get_path_mappings():
            local_prefix = Path(mapping.local)
            try:
                rel = local_p.relative_to(local_prefix)
            except ValueError:
                continue
            prefix_len = len(str(local_prefix))
            if prefix_len > best_prefix_len:
                best_prefix_len = prefix_len
                best_worker_path = (Path(mapping.worker) / rel).as_posix()
        return best_worker_path

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides.

        Environment variables take precedence over config file settings
        but are overridden by CLI arguments (handled by caller).
        """
        # Override WebSocket URL
        ws_override = os.getenv("SLEAP_RTC_SIGNALING_WS")
        if ws_override:
            self.signaling_websocket = ws_override
            logger.info(
                f"Overriding signaling_websocket from env: {self.signaling_websocket}"
            )

        # Override HTTP URL
        http_override = os.getenv("SLEAP_RTC_SIGNALING_HTTP")
        if http_override:
            self.signaling_http = http_override
            logger.info(f"Overriding signaling_http from env: {self.signaling_http}")

    def get_websocket_url(self, port: int = 8080) -> str:
        """Get the WebSocket signaling server URL.

        Args:
            port: Port number to use if not specified in URL (default: 8080).

        Returns:
            WebSocket URL with port.
        """
        url = self.signaling_websocket
        # Add port if not present
        if ":" not in url.split("//")[-1]:
            url = f"{url}:{port}"
        return url

    def get_http_url(self, port: int = 8001) -> str:
        """Get the HTTP signaling server base URL.

        Args:
            port: Port number to use if not specified in URL (default: 8001).

        Returns:
            HTTP base URL with port.
        """
        url = self.signaling_http
        # Don't add port to standard HTTPS/HTTP URLs (they use 443/80)
        # Only add port for localhost or explicit non-standard URLs
        host_part = url.split("//")[-1]
        is_localhost = host_part.startswith("localhost") or host_part.startswith(
            "127.0.0.1"
        )
        has_port = ":" in host_part

        if is_localhost and not has_port:
            url = f"{url}:{port}"
        return url

    def get_http_endpoint(self, endpoint: str) -> str:
        """Get a full HTTP API endpoint URL.

        Args:
            endpoint: API endpoint path (e.g., '/create-room', '/delete-peers-and-room').

        Returns:
            Full endpoint URL.
        """
        base_url = self.get_http_url()
        # Remove leading slash from endpoint if present
        endpoint = endpoint.lstrip("/")
        return f"{base_url}/{endpoint}"


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance.

    Loads configuration on first call.

    Returns:
        Global Config instance.
    """
    global _config
    if _config is None:
        _config = Config()
        _config.load()
    return _config


def reload_config() -> Config:
    """Reload configuration from sources.

    Useful for testing or runtime reconfiguration.

    Returns:
        Reloaded Config instance.
    """
    global _config
    _config = Config()
    _config.load()
    return _config
