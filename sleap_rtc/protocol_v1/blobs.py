"""Content-addressed result-blob index + hashing (protocol spec §6).

A blob is never copied into a cache here — this only ever records that a
file which already exists in place on the worker's filesystem (a
completed job's actual output file) is addressable by its sha256 hash, so
the blob-serving HTTP endpoint (`blob_http.py`) can look up "give me the
bytes for this hash" without needing to know which job produced it.

Kept as its own small SQLite-backed index — deliberately not folded into
`JobStore` — so it can be looked up *synchronously*, from the blob HTTP
server's own request-handling threads (see `blob_http.py`), without any
question of sharing an `aiosqlite` connection (built for a single asyncio
event loop) across threads it was never meant to run under. It also gets
its own database file rather than sharing `jobs.sqlite`, so this module's
plain `sqlite3` connection and `JobStore`'s `aiosqlite` connection never
contend for the same file's locks.
"""

import asyncio
import hashlib
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

_SCHEMA = """
CREATE TABLE IF NOT EXISTS blobs (
    sha256 TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    size INTEGER NOT NULL,
    created_at REAL NOT NULL
);
"""


@dataclass
class BlobRecord:
    """A registered blob — its hash, and where its bytes actually live.

    Attributes:
        sha256: Content hash, hex-encoded (matches the wire blob ref).
        path: Absolute path to the file on the worker's filesystem.
        size: File size in bytes, at the time it was registered.
    """

    sha256: str
    path: str
    size: int


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


class BlobIndex:
    """Maps a blob's sha256 to the on-disk file it's served from.

    Usage:
        index = BlobIndex(db_path)
        await index.register(sha256, path, size)
        record = await index.get(sha256)   # from asyncio code
        record = index.get_sync(sha256)    # from a plain thread (blob_http.py)
    """

    def __init__(self, db_path: Union[str, Path]):
        """Open (or create) the blob index database.

        Args:
            db_path: Path to the SQLite database file. Use ":memory:" for
                an ephemeral in-process index (tests only — an in-memory
                index can't be reached from a *different* connection, which
                defeats the point of `get_sync` being usable from another
                thread's request handler in production).
        """
        self._db_path = str(db_path)
        self._conn = _connect(self._db_path)

    async def register(self, sha256: str, path: str, size: int) -> None:
        """Record (or update) a blob's sha256 -> file mapping.

        Args:
            sha256: Content hash, hex-encoded.
            path: Absolute path to the file this hash resolves to.
            size: The file's size in bytes.
        """
        await asyncio.to_thread(self._register_sync, sha256, path, size)

    def _register_sync(self, sha256: str, path: str, size: int) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO blobs (sha256, path, size, created_at) "
            "VALUES (?, ?, ?, ?)",
            (sha256, path, size, time.time()),
        )
        self._conn.commit()

    async def get(self, sha256: str) -> Optional[BlobRecord]:
        """Look up a blob's file location — for the worker's asyncio code."""
        return await asyncio.to_thread(self.get_sync, sha256)

    def get_sync(self, sha256: str) -> Optional[BlobRecord]:
        """Look up a blob's file location — safe to call from any thread."""
        row = self._conn.execute(
            "SELECT sha256, path, size FROM blobs WHERE sha256 = ?", (sha256,)
        ).fetchone()
        if row is None:
            return None
        return BlobRecord(sha256=row[0], path=row[1], size=row[2])

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()


async def hash_file(path: Union[str, Path]) -> Tuple[str, int]:
    """Compute a file's sha256 hash and size, off the event loop.

    Args:
        path: Path to the file to hash.

    Returns:
        `(sha256_hex, size_bytes)`.
    """
    return await asyncio.to_thread(_hash_file_sync, Path(path))


def _hash_file_sync(path: Path) -> Tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size
