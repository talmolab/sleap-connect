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
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Verification-chunk size for item 2.4's iroh range-read path (design doc
# §7) — independent of hash_file's own 1 MiB read-buffer size below, and of
# blob_http.py's wire-transfer chunking. Tunable, not load-bearing: bigger
# means a smaller chunk_hashes list per blob but more re-verified padding
# on a misaligned range read; smaller is the reverse.
VERIFY_CHUNK_SIZE = 256 * 1024

_SCHEMA = """
CREATE TABLE IF NOT EXISTS blobs (
    sha256 TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    size INTEGER NOT NULL,
    created_at REAL NOT NULL,
    chunk_hashes TEXT NOT NULL DEFAULT '[]'
);
"""


@dataclass
class BlobRecord:
    """A registered blob — its hash, and where its bytes actually live.

    Attributes:
        sha256: Content hash, hex-encoded (matches the wire blob ref).
        path: Absolute path to the file on the worker's filesystem.
        size: File size in bytes, at the time it was registered.
        chunk_hashes: One sha256 hex digest per `VERIFY_CHUNK_SIZE`-aligned
            chunk of the file (last chunk may be shorter), computed once at
            registration — item 2.4's per-chunk integrity check for range
            reads over iroh. Empty if the blob was registered without them
            (e.g. an older caller that only passed sha256/path/size).
    """

    sha256: str
    path: str
    size: int
    chunk_hashes: List[str] = field(default_factory=list)


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
        await index.register(sha256, path, size, chunk_hashes)
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

    async def register(
        self,
        sha256: str,
        path: str,
        size: int,
        chunk_hashes: Optional[List[str]] = None,
    ) -> None:
        """Record (or update) a blob's sha256 -> file mapping.

        Args:
            sha256: Content hash, hex-encoded.
            path: Absolute path to the file this hash resolves to.
            size: The file's size in bytes.
            chunk_hashes: One sha256 hex digest per `VERIFY_CHUNK_SIZE`
                chunk (see `compute_chunk_hashes`) — item 2.4. Optional and
                defaults to empty so existing callers that only ever
                register `(sha256, path, size)` keep working unchanged;
                a blob registered without them just can't be verified
                per-chunk over an iroh range-read stream.
        """
        await asyncio.to_thread(
            self._register_sync, sha256, path, size, chunk_hashes or []
        )

    def _register_sync(
        self, sha256: str, path: str, size: int, chunk_hashes: List[str]
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO blobs "
            "(sha256, path, size, created_at, chunk_hashes) VALUES (?, ?, ?, ?, ?)",
            (sha256, path, size, time.time(), json.dumps(chunk_hashes)),
        )
        self._conn.commit()

    async def get(self, sha256: str) -> Optional[BlobRecord]:
        """Look up a blob's file location — for the worker's asyncio code."""
        return await asyncio.to_thread(self.get_sync, sha256)

    def get_sync(self, sha256: str) -> Optional[BlobRecord]:
        """Look up a blob's file location — safe to call from any thread."""
        row = self._conn.execute(
            "SELECT sha256, path, size, chunk_hashes FROM blobs WHERE sha256 = ?",
            (sha256,),
        ).fetchone()
        if row is None:
            return None
        return BlobRecord(
            sha256=row[0], path=row[1], size=row[2], chunk_hashes=json.loads(row[3])
        )

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


async def compute_chunk_hashes(
    path: Union[str, Path], chunk_size: int = VERIFY_CHUNK_SIZE
) -> List[str]:
    """Compute one sha256 hex digest per `chunk_size`-aligned chunk of a file.

    A separate pass from `hash_file` (not fused into it) — this hashing only
    ever runs once, after a job completes, not on any interactive path
    (`_register_result_blobs`'s own docstring: never raises, only affects
    whether the result is fetchable), so a second read of a job-result file
    (realistically low tens of MB — no embedded video) is a fine trade for
    not touching `hash_file`'s existing, already-tested 2-tuple contract.

    Args:
        path: Path to the file to hash.
        chunk_size: Boundary size for each chunk's own hash. The last chunk
            covers whatever remains and may be shorter.

    Returns:
        Ordered list of sha256 hex digests, one per chunk. Empty for an
        empty file.
    """
    return await asyncio.to_thread(_compute_chunk_hashes_sync, Path(path), chunk_size)


def _compute_chunk_hashes_sync(path: Path, chunk_size: int) -> List[str]:
    hashes: List[str] = []
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hashes.append(hashlib.sha256(chunk).hexdigest())
    return hashes
