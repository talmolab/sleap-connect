"""A FIFO queue that gates concurrent job execution to a fixed slot count.

Today's worker model is one worker *process* per GPU (`WorkerCapabilities`/
`WorkerClass` are both parameterized by a single `gpu_id`, and
`max_concurrent_jobs` is currently hardcoded to 1) — so "one job slot per
GPU" is, in practice, one job slot per worker process. This is parameterized
rather than hardcoded again, in case that assumption changes later (e.g. a
worker process managing multiple GPUs).

Today, a job request beyond capacity is rejected outright (see
`JobCoordinator._handle_job_request`, which checks `status == "available"`).
This queue is the primitive for the alternative — accept the job and queue
it — but is not wired into `job_coordinator.py` yet.
"""

from asyncio import Semaphore


class JobQueue:
    """Gates concurrent job execution via a semaphore, with a waiting count.

    Usage:
        queue = JobQueue(max_concurrent=1)
        async with queue.slot():
            ... run the job ...

    Or, without the context manager:
        await queue.acquire()
        try:
            ... run the job ...
        finally:
            queue.release()
    """

    def __init__(self, max_concurrent: int = 1):
        """Initialize the queue.

        Args:
            max_concurrent: Number of jobs allowed to run at once. Must be
                >= 1.
        """
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self._semaphore = Semaphore(max_concurrent)
        self._waiting = 0
        self._running = 0

    @property
    def waiting(self) -> int:
        """Number of jobs currently queued, waiting for a free slot."""
        return self._waiting

    @property
    def running(self) -> int:
        """Number of slots currently held (jobs executing)."""
        return self._running

    async def acquire(self) -> None:
        """Wait for a free execution slot, incrementing `waiting` while queued."""
        self._waiting += 1
        try:
            await self._semaphore.acquire()
        finally:
            self._waiting -= 1
        self._running += 1

    def release(self) -> None:
        """Free the execution slot for the next queued job."""
        self._running -= 1
        self._semaphore.release()

    def slot(self) -> "_JobSlot":
        """Return an async context manager that acquires/releases a slot."""
        return _JobSlot(self)


class _JobSlot:
    """Async context manager returned by `JobQueue.slot()`."""

    def __init__(self, queue: JobQueue):
        self._queue = queue

    async def __aenter__(self) -> None:
        await self._queue.acquire()

    async def __aexit__(self, *exc_info) -> None:
        self._queue.release()
