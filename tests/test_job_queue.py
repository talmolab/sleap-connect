"""Tests for the concurrency-gating job queue."""

import asyncio

import pytest

from sleap_rtc.jobs.queue import JobQueue


class TestJobQueue:
    """Tests for JobQueue."""

    def test_rejects_max_concurrent_below_one(self):
        with pytest.raises(ValueError):
            JobQueue(max_concurrent=0)

    async def test_single_slot_serializes_execution(self):
        queue = JobQueue(max_concurrent=1)
        order = []

        async def job(name, delay):
            async with queue.slot():
                order.append(f"{name}-start")
                await asyncio.sleep(delay)
                order.append(f"{name}-end")

        await asyncio.gather(job("a", 0.05), job("b", 0.0))

        # With one slot, "a" must fully finish before "b" starts.
        assert order == ["a-start", "a-end", "b-start", "b-end"]

    async def test_waiting_reflects_queued_jobs(self):
        queue = JobQueue(max_concurrent=1)
        holder_acquired = asyncio.Event()
        release_holder = asyncio.Event()

        async def holder():
            async with queue.slot():
                holder_acquired.set()
                await release_holder.wait()

        holder_task = asyncio.create_task(holder())
        await holder_acquired.wait()
        assert queue.waiting == 0

        waiter_started = asyncio.Event()

        async def waiter():
            waiter_started.set()
            async with queue.slot():
                pass

        waiter_task = asyncio.create_task(waiter())
        await waiter_started.wait()
        await asyncio.sleep(0.05)  # let the waiter actually block on acquire()

        assert queue.waiting == 1

        release_holder.set()
        await holder_task
        await waiter_task
        assert queue.waiting == 0

    async def test_slot_releases_on_exception(self):
        queue = JobQueue(max_concurrent=1)

        with pytest.raises(RuntimeError):
            async with queue.slot():
                raise RuntimeError("boom")

        # The slot must have been released despite the exception — a second
        # acquire should not hang.
        async with asyncio.timeout(1):
            async with queue.slot():
                pass

    async def test_multiple_slots_allow_concurrent_execution(self):
        queue = JobQueue(max_concurrent=2)
        both_running = asyncio.Event()
        running_count = 0

        async def job():
            nonlocal running_count
            async with queue.slot():
                running_count += 1
                if running_count == 2:
                    both_running.set()
                await asyncio.wait_for(both_running.wait(), timeout=1)

        await asyncio.gather(job(), job())

        assert running_count == 2

    async def test_running_counts_held_slots(self):
        queue = JobQueue(max_concurrent=2)
        assert queue.running == 0
        async with queue.slot():
            assert queue.running == 1
            async with queue.slot():
                assert queue.running == 2
        assert queue.running == 0
