"""Durable, transport-independent job log and event store.

Persists job state and per-job events (status transitions, log lines,
training metrics, downsampled loss curves, results) to a local SQLite
database, so that:

- A worker restart can reattach to jobs it was running (the job's own
  ``sleap-nn`` subprocess survives independently; this store is what lets the
  worker rediscover what it was doing).
- A client can disconnect and reconnect hours later and replay everything
  it missed via `get_events_since`, instead of only ever seeing live-only
  forwarded messages.

This module is intentionally standalone: it does not depend on the
websocket/rooms-based signaling protocol in `sleap_rtc.worker.state_manager`
or `sleap_rtc.worker.job_coordinator`, and is not wired into job execution
yet. It mirrors the job/event model from the sleap-connect protocol v1 spec
(``job.status`` / ``job.log`` / ``job.metric`` / ``job.curve`` / ``job.result``
topics, per-job monotonic ``seq`` for events-since-N replay).
"""

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import aiosqlite

from sleap_rtc.jobs.spec import TrackJobSpec, TrainJobSpec, parse_job_spec

JobSpec = Union[TrainJobSpec, TrackJobSpec]

# Job lifecycle states — see protocol v1 spec §4.3.
JOB_STATES = ("queued", "running", "completed", "failed", "canceled")
TERMINAL_STATES = ("completed", "failed", "canceled")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    spec_json TEXT NOT NULL,
    state TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    result_json TEXT,
    error TEXT,
    pid INTEGER,
    process_started_at REAL,
    log_path TEXT
);

CREATE TABLE IF NOT EXISTS job_events (
    job_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    topic TEXT NOT NULL,
    data_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (job_id, seq),
    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
);
"""


class JobStoreError(Exception):
    """Raised for invalid job-store operations (unknown job, bad state, ...)."""


@dataclass
class JobRecord:
    """A durable job record.

    Attributes:
        job_id: Worker-assigned job identifier.
        spec: The parsed job spec (TrainJobSpec or TrackJobSpec).
        state: Current lifecycle state — one of `JOB_STATES`.
        created_at: Unix timestamp the job was created.
        updated_at: Unix timestamp of the last state change.
        result: Terminal result payload (e.g. blob refs), if any.
        error: Terminal error message, if any.
        pid: PID of the job's detached subprocess, if it has been spawned.
        process_started_at: The subprocess's own start time (from the OS,
            not this record's created_at) — recorded alongside `pid` so a
            later liveness check can guard against PID reuse: a PID match
            with a different start time means it's a different process that
            happens to have recycled the same PID, not the one we spawned.
        log_path: Path to the subprocess's redirected stdout/stderr log file.
            Logging to a file (rather than an in-process pipe) is what makes
            reattachment possible — a pipe's read end dies with whichever
            process originally spawned the child, but a file on disk can be
            tailed by a totally different (e.g. post-restart) process.
    """

    job_id: str
    spec: JobSpec
    state: str
    created_at: float
    updated_at: float
    result: Optional[dict] = None
    error: Optional[str] = None
    pid: Optional[int] = None
    process_started_at: Optional[float] = None
    log_path: Optional[str] = None


@dataclass
class JobEvent:
    """A single durable event in a job's event log.

    Attributes:
        job_id: The job this event belongs to.
        seq: Per-job monotonic sequence number, starting at 1.
        topic: Event topic (e.g. "job.status", "job.log", "job.metric").
        data: Topic-specific payload.
        created_at: Unix timestamp the event was recorded.
    """

    job_id: str
    seq: int
    topic: str
    data: dict
    created_at: float


class JobStore:
    """Async SQLite-backed store for job records and their event logs.

    Usage:
        store = JobStore(db_path)
        await store.connect()
        try:
            ...
        finally:
            await store.close()

    Or as an async context manager:
        async with JobStore(db_path) as store:
            ...
    """

    def __init__(self, db_path: Union[str, Path]):
        """Initialize the store.

        Args:
            db_path: Path to the SQLite database file. Created if it doesn't
                exist. Use ":memory:" for an ephemeral in-process store
                (mainly useful in tests — an in-memory store can't survive a
                worker restart, which defeats the point in production).
        """
        self._db_path = str(db_path)
        self._conn: Optional[aiosqlite.Connection] = None
        # Serializes append_event's read-then-write seq assignment per job —
        # see append_event's docstring for why this is needed even though
        # today's only caller happens not to trigger the race in practice.
        self._event_locks: Dict[str, asyncio.Lock] = {}

    async def connect(self) -> None:
        """Open the database connection and ensure the schema exists."""
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(_SCHEMA)
        await self._conn.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> "JobStore":
        await self.connect()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.close()

    def _require_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise JobStoreError("JobStore is not connected — call connect() first")
        return self._conn

    async def create_job(self, job_id: str, spec: JobSpec) -> JobRecord:
        """Create a new job record in the "queued" state.

        Args:
            job_id: Worker-assigned job identifier. Must be unique.
            spec: The job's TrainJobSpec or TrackJobSpec.

        Returns:
            The created JobRecord.

        Raises:
            JobStoreError: If a job with this job_id already exists.
        """
        conn = self._require_conn()
        now = time.time()
        try:
            await conn.execute(
                """
                INSERT INTO jobs (job_id, spec_json, state, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (job_id, spec.to_json(), "queued", now, now),
            )
            await conn.commit()
        except aiosqlite.IntegrityError as e:
            raise JobStoreError(f"Job {job_id!r} already exists") from e

        return JobRecord(
            job_id=job_id, spec=spec, state="queued", created_at=now, updated_at=now
        )

    async def get_job(self, job_id: str) -> Optional[JobRecord]:
        """Fetch a job record by ID.

        Args:
            job_id: The job identifier.

        Returns:
            The JobRecord, or None if no such job exists.
        """
        conn = self._require_conn()
        cursor = await conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    async def list_jobs(self) -> List[JobRecord]:
        """List all job records, most recently created first.

        Returns:
            A list of JobRecords.
        """
        conn = self._require_conn()
        cursor = await conn.execute("SELECT * FROM jobs ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        return [self._row_to_record(row) for row in rows]

    async def update_state(
        self,
        job_id: str,
        state: str,
        *,
        result: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> JobRecord:
        """Update a job's lifecycle state.

        Args:
            job_id: The job identifier.
            state: New state — must be one of `JOB_STATES`.
            result: Terminal result payload (e.g. blob refs). Only meaningful
                for terminal states.
            error: Terminal error message. Only meaningful for terminal states.

        Returns:
            The updated JobRecord.

        Raises:
            JobStoreError: If the job doesn't exist or `state` is invalid.
        """
        if state not in JOB_STATES:
            raise JobStoreError(
                f"Invalid job state {state!r}; must be one of {JOB_STATES}"
            )

        conn = self._require_conn()
        now = time.time()
        result_json = json.dumps(result) if result is not None else None
        cursor = await conn.execute(
            """
            UPDATE jobs
            SET state = ?, updated_at = ?, result_json = ?, error = ?
            WHERE job_id = ?
            """,
            (state, now, result_json, error, job_id),
        )
        await conn.commit()
        if cursor.rowcount == 0:
            raise JobStoreError(f"No such job: {job_id!r}")

        record = await self.get_job(job_id)
        assert record is not None
        return record

    async def set_process_info(
        self, job_id: str, pid: int, process_started_at: float, log_path: str
    ) -> JobRecord:
        """Record the detached subprocess spawned for a job.

        Call this immediately after spawning, before awaiting the process —
        if the worker crashes before this lands, `reattach_all` has nothing
        to find and the job is correctly treated as lost, matching reality.

        Args:
            job_id: The job identifier. Must already exist.
            pid: PID of the spawned subprocess.
            process_started_at: The subprocess's own start time, as reported
                by the OS (not `time.time()` at spawn) — see `JobRecord.
                process_started_at` for why this matters.
            log_path: Path to the subprocess's redirected stdout/stderr log.

        Returns:
            The updated JobRecord.

        Raises:
            JobStoreError: If the job doesn't exist.
        """
        conn = self._require_conn()
        cursor = await conn.execute(
            """
            UPDATE jobs
            SET pid = ?, process_started_at = ?, log_path = ?, updated_at = ?
            WHERE job_id = ?
            """,
            (pid, process_started_at, log_path, time.time(), job_id),
        )
        await conn.commit()
        if cursor.rowcount == 0:
            raise JobStoreError(f"No such job: {job_id!r}")

        record = await self.get_job(job_id)
        assert record is not None
        return record

    async def append_event(self, job_id: str, topic: str, data: dict) -> int:
        """Append an event to a job's durable event log.

        Args:
            job_id: The job this event belongs to. Must already exist.
            topic: Event topic (e.g. "job.status", "job.log", "job.metric",
                "job.curve", "job.result").
            data: Topic-specific JSON-serializable payload.

        Returns:
            The event's assigned sequence number (per-job monotonic, starting
            at 1).

        Raises:
            JobStoreError: If the job doesn't exist.

        Note:
            Assigning `next_seq` is a read (``SELECT MAX(seq)``) followed by
            a separate write (``INSERT``), which is not atomic at the SQL
            level — two genuinely concurrent calls for the *same* `job_id`
            could otherwise both read the same max and either collide on
            the ``(job_id, seq)`` primary key or silently assign a
            duplicate. Guarded here with a per-job `asyncio.Lock` so the
            store is correct regardless of caller concurrency, rather than
            relying on callers to serialize their own emits.
        """
        conn = self._require_conn()

        job = await self.get_job(job_id)
        if job is None:
            raise JobStoreError(f"No such job: {job_id!r}")

        lock = self._event_locks.setdefault(job_id, asyncio.Lock())
        async with lock:
            cursor = await conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM job_events WHERE job_id = ?",
                (job_id,),
            )
            row = await cursor.fetchone()
            next_seq = row[0]

            await conn.execute(
                """
                INSERT INTO job_events (job_id, seq, topic, data_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (job_id, next_seq, topic, json.dumps(data), time.time()),
            )
            await conn.commit()
            return next_seq

    async def get_events_since(self, job_id: str, since_seq: int = 0) -> List[JobEvent]:
        """Fetch all events for a job with seq > since_seq, in order.

        Args:
            job_id: The job identifier.
            since_seq: Return only events after this sequence number. 0 (the
                default) returns the full history — this is how a client
                that's never seen a job live still gets its complete log on
                reattach.

        Returns:
            A list of JobEvents, ordered by seq ascending.
        """
        conn = self._require_conn()
        cursor = await conn.execute(
            """
            SELECT * FROM job_events
            WHERE job_id = ? AND seq > ?
            ORDER BY seq ASC
            """,
            (job_id, since_seq),
        )
        rows = await cursor.fetchall()
        return [
            JobEvent(
                job_id=row["job_id"],
                seq=row["seq"],
                topic=row["topic"],
                data=json.loads(row["data_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    @staticmethod
    def _row_to_record(row: aiosqlite.Row) -> JobRecord:
        return JobRecord(
            job_id=row["job_id"],
            spec=parse_job_spec(row["spec_json"]),
            state=row["state"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            error=row["error"],
            pid=row["pid"],
            process_started_at=row["process_started_at"],
            log_path=row["log_path"],
        )
