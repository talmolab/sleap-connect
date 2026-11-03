"""Tests for the client-side upload_file coroutine."""

import asyncio
import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from sleap_connect.client.file_transfer import upload_file, UPLOAD_CHUNK_SIZE
from sleap_connect.protocol import (
    MSG_FILE_UPLOAD_CACHE_HIT,
    MSG_FILE_UPLOAD_CHECK,
    MSG_FILE_UPLOAD_COMPLETE,
    MSG_FILE_UPLOAD_END,
    MSG_FILE_UPLOAD_ERROR,
    MSG_FILE_UPLOAD_PROGRESS,
    MSG_FILE_UPLOAD_READY,
    MSG_FILE_UPLOAD_START,
    MSG_SEPARATOR,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_channel() -> MagicMock:
    ch = MagicMock()
    ch.readyState = "open"
    ch.bufferedAmount = 0
    return ch


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def _feed(queue: asyncio.Queue, *messages):
    """Put messages into a queue one at a time (used in tasks)."""
    for msg in messages:
        await queue.put(msg)


# ---------------------------------------------------------------------------
# Success — full upload
# ---------------------------------------------------------------------------


class TestUploadFileSuccess:
    @pytest.mark.asyncio
    async def test_full_upload_returns_worker_path(self, tmp_path):
        payload = b"hello world"
        f = tmp_path / "labels.pkg.slp"
        f.write_bytes(payload)

        ch = make_channel()
        q: asyncio.Queue = asyncio.Queue()
        sha = sha256_of(payload)

        # Simulate worker responses:
        # FILE_UPLOAD_CHECK → READY
        # FILE_UPLOAD_START → READY
        # FILE_UPLOAD_END  → COMPLETE
        worker_path = f"/remote/labels.pkg.slp"
        await q.put(MSG_FILE_UPLOAD_READY)  # answer to CHECK
        await q.put(MSG_FILE_UPLOAD_READY)  # answer to START
        await q.put(f"{MSG_FILE_UPLOAD_COMPLETE}{MSG_SEPARATOR}{worker_path}")

        result = await upload_file(ch, q, str(f), "/remote", "0")

        assert result == worker_path

        # Verify CHECK message sent with correct sha256 and filename
        check_call = ch.send.call_args_list[0][0][0]
        assert check_call.startswith(MSG_FILE_UPLOAD_CHECK + MSG_SEPARATOR)
        assert sha in check_call
        assert "labels.pkg.slp" in check_call

        # Verify START message
        start_call = ch.send.call_args_list[1][0][0]
        assert start_call.startswith(MSG_FILE_UPLOAD_START + MSG_SEPARATOR)
        assert str(len(payload)) in start_call
        assert "/remote" in start_call
        assert "0" in start_call

        # Verify END message sent
        calls = [c[0][0] for c in ch.send.call_args_list]
        assert MSG_FILE_UPLOAD_END in calls

    @pytest.mark.asyncio
    async def test_binary_chunks_sent(self, tmp_path):
        # Ensure chunk bytes are sent for non-trivial payload
        payload = b"x" * (UPLOAD_CHUNK_SIZE * 2 + 100)
        f = tmp_path / "big.pkg.slp"
        f.write_bytes(payload)

        ch = make_channel()
        q: asyncio.Queue = asyncio.Queue()
        await q.put(MSG_FILE_UPLOAD_READY)
        await q.put(MSG_FILE_UPLOAD_READY)
        await q.put(f"{MSG_FILE_UPLOAD_COMPLETE}{MSG_SEPARATOR}/remote/big.pkg.slp")

        await upload_file(ch, q, str(f), "/remote", "0")

        # Count binary sends (bytes objects)
        binary_sends = [c for c in ch.send.call_args_list if isinstance(c[0][0], bytes)]
        assert len(binary_sends) == 3  # ceil(2*CHUNK + 100 / CHUNK)


# ---------------------------------------------------------------------------
# Cache hit — no binary transfer
# ---------------------------------------------------------------------------


class TestUploadFileCacheHit:
    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_path_immediately(self, tmp_path):
        payload = b"data"
        f = tmp_path / "labels.pkg.slp"
        f.write_bytes(payload)

        ch = make_channel()
        q: asyncio.Queue = asyncio.Queue()
        cached = "/worker/cached/labels.pkg.slp"
        await q.put(f"{MSG_FILE_UPLOAD_CACHE_HIT}{MSG_SEPARATOR}{cached}")

        result = await upload_file(ch, q, str(f), "/dest", "0")

        assert result == cached

        # Only one send should have occurred (the CHECK), no chunks or END
        assert ch.send.call_count == 1
        assert ch.send.call_args[0][0].startswith(MSG_FILE_UPLOAD_CHECK + MSG_SEPARATOR)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestUploadFileErrors:
    @pytest.mark.asyncio
    async def test_error_on_start_raises(self, tmp_path):
        f = tmp_path / "labels.pkg.slp"
        f.write_bytes(b"data")

        ch = make_channel()
        q: asyncio.Queue = asyncio.Queue()
        await q.put(MSG_FILE_UPLOAD_READY)  # answer to CHECK
        await q.put(f"{MSG_FILE_UPLOAD_ERROR}{MSG_SEPARATOR}Destination outside configured mounts")

        with pytest.raises(RuntimeError, match="Worker rejected upload"):
            await upload_file(ch, q, str(f), "/outside", "0")

    @pytest.mark.asyncio
    async def test_error_after_end_raises(self, tmp_path):
        f = tmp_path / "labels.pkg.slp"
        f.write_bytes(b"data")

        ch = make_channel()
        q: asyncio.Queue = asyncio.Queue()
        await q.put(MSG_FILE_UPLOAD_READY)   # CHECK
        await q.put(MSG_FILE_UPLOAD_READY)   # START
        await q.put(f"{MSG_FILE_UPLOAD_ERROR}{MSG_SEPARATOR}disk full")

        with pytest.raises(RuntimeError, match="Upload failed: disk full"):
            await upload_file(ch, q, str(f), "/remote", "0")

    @pytest.mark.asyncio
    async def test_unexpected_check_response_raises(self, tmp_path):
        f = tmp_path / "labels.pkg.slp"
        f.write_bytes(b"data")

        ch = make_channel()
        q: asyncio.Queue = asyncio.Queue()
        await q.put("SOME_UNKNOWN_MESSAGE")

        with pytest.raises(RuntimeError, match="Unexpected response to FILE_UPLOAD_CHECK"):
            await upload_file(ch, q, str(f), "/remote", "0")


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------


class TestUploadFileProgress:
    @pytest.mark.asyncio
    async def test_progress_callback_invoked(self, tmp_path):
        f = tmp_path / "labels.pkg.slp"
        f.write_bytes(b"data")

        ch = make_channel()
        q: asyncio.Queue = asyncio.Queue()
        await q.put(MSG_FILE_UPLOAD_READY)
        await q.put(MSG_FILE_UPLOAD_READY)
        await q.put(f"{MSG_FILE_UPLOAD_PROGRESS}{MSG_SEPARATOR}4{MSG_SEPARATOR}4")
        await q.put(f"{MSG_FILE_UPLOAD_COMPLETE}{MSG_SEPARATOR}/remote/labels.pkg.slp")

        progress_calls = []
        await upload_file(
            ch, q, str(f), "/remote", "0",
            on_progress=lambda sent, total: progress_calls.append((sent, total)),
        )

        assert progress_calls == [(4, 4)]

    @pytest.mark.asyncio
    async def test_multiple_progress_messages_all_delivered(self, tmp_path):
        f = tmp_path / "labels.pkg.slp"
        f.write_bytes(b"data")

        ch = make_channel()
        q: asyncio.Queue = asyncio.Queue()
        await q.put(MSG_FILE_UPLOAD_READY)
        await q.put(MSG_FILE_UPLOAD_READY)
        await q.put(f"{MSG_FILE_UPLOAD_PROGRESS}{MSG_SEPARATOR}1{MSG_SEPARATOR}4")
        await q.put(f"{MSG_FILE_UPLOAD_PROGRESS}{MSG_SEPARATOR}2{MSG_SEPARATOR}4")
        await q.put(f"{MSG_FILE_UPLOAD_PROGRESS}{MSG_SEPARATOR}4{MSG_SEPARATOR}4")
        await q.put(f"{MSG_FILE_UPLOAD_COMPLETE}{MSG_SEPARATOR}/remote/labels.pkg.slp")

        calls = []
        await upload_file(
            ch, q, str(f), "/remote", "0",
            on_progress=lambda s, t: calls.append((s, t)),
        )

        assert calls == [(1, 4), (2, 4), (4, 4)]
