"""Persisted record of the devices paired with this worker.

Separate from the legacy account/room credential schema (`credentials.py`)
— this is the pairing-model's trust store: the client public keys
(`node_id`s) this worker has paired with and will accept `auth.prove` from,
plus what's known about each device (a name it reports, its platform and app
version, when it paired and was last seen). No accounts, no rooms.

Stored in SQLite and read on every call, not cached: `sleap-rtc clients`
(another process) sees what a running worker recorded, and a change made
from the CLI applies to the running worker right away.
"""

import json
import os
import sqlite3
import stat
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set, Union

DEFAULT_TRUST_STORE_PATH = Path.home() / ".sleap-rtc" / "protocol_v1_devices.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS paired_devices (
    node_id TEXT PRIMARY KEY,
    name TEXT,
    platform TEXT,
    app_version TEXT,
    paired_at REAL,
    last_seen_at REAL
);
"""

_MAX_NAME_LEN = 120


@dataclass
class PairedDevice:
    """A device paired with this worker.

    Attributes:
        node_id: The device's public key (its identity).
        name: The name the device reports (e.g. the computer's name), or
            None for a device paired before names were recorded.
        platform: "tauri" (desktop app) or "web" (browser), if reported.
        app_version: The app version it last connected with, if reported.
        paired_at: Unix time it paired, or None if paired before this was
            recorded.
        last_seen_at: Unix time it last connected or disconnected, or None.
    """

    node_id: str
    name: Optional[str]
    platform: Optional[str]
    app_version: Optional[str]
    paired_at: Optional[float]
    last_seen_at: Optional[float]

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "platform": self.platform,
            "app_version": self.app_version,
            "paired_at": self.paired_at,
            "last_seen_at": self.last_seen_at,
        }


def _text(agent: dict, key: str, limit: int) -> Optional[str]:
    value = agent.get(key)
    if not isinstance(value, str):
        return None
    return value.strip()[:limit] or None


def _agent_fields(agent: Optional[dict]) -> tuple:
    """(name, platform, app_version) from a hello `agent`, each None if absent."""
    agent = agent or {}
    return (
        _text(agent, "device_name", _MAX_NAME_LEN),
        _text(agent, "platform", 32),
        _text(agent, "version", 64),
    )


class TrustStore:
    """The devices paired with this worker, in SQLite."""

    def __init__(
        self,
        path: Union[str, Path] = DEFAULT_TRUST_STORE_PATH,
        legacy_json_path: Optional[Union[str, Path]] = None,
    ):
        """Open (creating if needed) the store at `path`.

        Args:
            path: The SQLite file. Tests should pass a `tmp_path`.
            legacy_json_path: The old `trusted_clients.json` allowlist, if
                any: its node_ids are imported (as unnamed devices) the
                first time, so existing pairings keep working.
        """
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.executescript(_SCHEMA)
        os.chmod(self._path, stat.S_IRUSR | stat.S_IWUSR)
        if legacy_json_path is not None:
            self._import_legacy(Path(legacy_json_path))

    def _connect(self) -> sqlite3.Connection:
        # A short-lived connection per call: the CLI and a running worker
        # share this file, and SQLite's own locking keeps them consistent.
        db = sqlite3.connect(self._path, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    def _import_legacy(self, json_path: Path) -> None:
        if not json_path.exists():
            return
        try:
            node_ids = json.loads(json_path.read_text()).get("trusted_node_ids", [])
        except (OSError, ValueError):
            return
        with closing(self._connect()) as db, db:
            db.executemany(
                "INSERT OR IGNORE INTO paired_devices (node_id) VALUES (?)",
                [(n,) for n in node_ids if isinstance(n, str)],
            )
        # Imported once; the SQLite file is the record from now on.
        json_path.rename(json_path.with_name(json_path.name + ".imported"))

    def is_trusted(self, node_id: str) -> bool:
        """Check whether `node_id` has been paired with this worker."""
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT 1 FROM paired_devices WHERE node_id = ?", (node_id,)
            ).fetchone()
        return row is not None

    def add_trusted(self, node_id: str, agent: Optional[dict] = None) -> None:
        """Pair `node_id` (idempotent), recording what its hello reported."""
        now = time.time()
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO paired_devices (node_id, paired_at, last_seen_at) "
                "VALUES (?, ?, ?) ON CONFLICT(node_id) DO NOTHING",
                (node_id, now, now),
            )
        self.record_seen(node_id, agent)

    def record_seen(self, node_id: str, agent: Optional[dict] = None) -> None:
        """Mark `node_id` seen now, with the name/platform/version it reports.

        A field the device doesn't report is left as it was.
        """
        name, platform, version = _agent_fields(agent)
        with closing(self._connect()) as db, db:
            db.execute(
                """
                UPDATE paired_devices SET
                    last_seen_at = ?,
                    name = COALESCE(?, name),
                    platform = COALESCE(?, platform),
                    app_version = COALESCE(?, app_version)
                WHERE node_id = ?
                """,
                (time.time(), name, platform, version, node_id),
            )

    def list_trusted(self) -> Set[str]:
        """Return the paired node_ids."""
        with closing(self._connect()) as db:
            rows = db.execute("SELECT node_id FROM paired_devices").fetchall()
        return {r["node_id"] for r in rows}

    def list_devices(self) -> List[PairedDevice]:
        """Every paired device, most recently seen first."""
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT * FROM paired_devices "
                "ORDER BY last_seen_at IS NULL, last_seen_at DESC"
            ).fetchall()
        return [
            PairedDevice(
                node_id=r["node_id"],
                name=r["name"],
                platform=r["platform"],
                app_version=r["app_version"],
                paired_at=r["paired_at"],
                last_seen_at=r["last_seen_at"],
            )
            for r in rows
        ]
