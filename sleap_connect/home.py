"""The per-user sleap-connect folder (`~/.sleap-connect`), and moving the old one.

Before the rename this was `~/.sleap-rtc`. It holds the worker's identity,
paired devices, shared folders, jobs and logs (under `protocol_v1/`), plus the
running-worker registry, the shell reminder scripts and the legacy commands'
credentials/config. `migrate_legacy_home` moves an existing `~/.sleap-rtc` to
`~/.sleap-connect` the first time sleap-connect runs, so an upgrade keeps every
pairing, shared folder and past job. On macOS/Linux it leaves a `~/.sleap-rtc`
symlink behind, so older sleap-app versions (which read
`~/.sleap-rtc/config.toml`) and anything else that remembered an old absolute
path still find it.
"""

import os
import sys
from pathlib import Path
from typing import Optional

DIR_NAME = ".sleap-connect"
LEGACY_DIR_NAME = ".sleap-rtc"

# Points the folder elsewhere (tests use a temporary one). The old name is
# still accepted.
HOME_ENV = "SLEAP_CONNECT_HOME"
LEGACY_HOME_ENV = "SLEAP_RTC_HOME"


def env_flag(name: str) -> Optional[str]:
    """`$SLEAP_CONNECT_<name>`, or the old `$SLEAP_RTC_<name>`, if set."""
    return os.environ.get(f"SLEAP_CONNECT_{name}") or os.environ.get(
        f"SLEAP_RTC_{name}"
    )


def home_dir() -> Path:
    """`~/.sleap-connect`, or `$SLEAP_CONNECT_HOME` if set."""
    override = os.environ.get(HOME_ENV) or os.environ.get(LEGACY_HOME_ENV)
    return Path(override) if override else Path.home() / DIR_NAME


def legacy_home_dir() -> Path:
    """Where sleap-rtc kept the same things."""
    return Path.home() / LEGACY_DIR_NAME


class LegacyWorkerRunning(RuntimeError):
    """A worker started by sleap-rtc still runs out of `~/.sleap-rtc`."""

    def __init__(self, pid: int):
        self.pid = pid
        super().__init__(f"a worker started by sleap-rtc is still running (pid {pid})")


def legacy_data_dir() -> Path:
    """The old worker data dir (`~/.sleap-rtc/protocol_v1`)."""
    return legacy_home_dir() / "protocol_v1"


def pending_migration() -> bool:
    """Whether there's an old `~/.sleap-rtc` folder still to move."""
    if os.environ.get(HOME_ENV) or os.environ.get(LEGACY_HOME_ENV):
        return False
    old, new = legacy_home_dir(), Path.home() / DIR_NAME
    return old.is_dir() and not old.is_symlink() and not new.exists()


def migrate_legacy_home() -> Optional[str]:
    """Move `~/.sleap-rtc` to `~/.sleap-connect`, once.

    Returns a line to tell the user, or None if there was nothing to move.

    Raises:
        LegacyWorkerRunning: if a worker from before the rename is still
            running out of the old folder. Moving its files from under it
            would break it (and a new worker would start with no pairings),
            so it has to be stopped first.
    """
    if not pending_migration():
        return None
    from sleap_connect.protocol_v1 import daemon

    running = daemon.read_running(legacy_data_dir())
    if running is not None:
        raise LegacyWorkerRunning(running.pid)

    old, new = legacy_home_dir(), Path.home() / DIR_NAME
    os.replace(old, new)
    if sys.platform != "win32":
        try:
            old.symlink_to(new, target_is_directory=True)
        except OSError:
            pass

    # The shell reminder's startup-file block names the old folder's path.
    try:
        from sleap_connect.protocol_v1 import shell_hook

        shell_hook.repoint_blocks()
    except OSError:
        pass

    return (
        f"Moved sleap-rtc's settings from ~/{LEGACY_DIR_NAME} to ~/{DIR_NAME} "
        "(paired devices, shared folders and jobs come along)."
    )
