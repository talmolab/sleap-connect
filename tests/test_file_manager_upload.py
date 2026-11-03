"""Tests for FileManager client-to-worker upload methods."""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sleap_connect.config import MountConfig
from sleap_connect.protocol import (
    MSG_FILE_UPLOAD_CACHE_HIT,
    MSG_FILE_UPLOAD_COMPLETE,
    MSG_FILE_UPLOAD_ERROR,
    MSG_FILE_UPLOAD_PROGRESS,
    MSG_FILE_UPLOAD_READY,
    MSG_SEPARATOR,
)
from sleap_connect.worker.file_manager import FileManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_fm(tmp_path: Path) -> FileManager:
    """Return a FileManager with tmp_path as the sole mount."""
    mount = MountConfig(path=str(tmp_path), label="Test")
    return FileManager(mounts=[mount])


def fake_channel() -> MagicMock:
    """Return a mock RTCDataChannel that records sent messages."""
    ch = MagicMock()
    ch.readyState = "open"
    return ch


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# check_upload_cache
# ---------------------------------------------------------------------------


class TestCheckUploadCache:
    def test_cache_miss_empty(self, tmp_path):
        fm = make_fm(tmp_path)
        assert fm.check_upload_cache("abc123", "labels.pkg.slp") is None

    def test_cache_hit_file_exists(self, tmp_path):
        fm = make_fm(tmp_path)
        target = tmp_path / "labels.pkg.slp"
        target.write_bytes(b"data")
        fm._upload_cache["abc123"] = str(target)

        result = fm.check_upload_cache("abc123", "labels.pkg.slp")
        assert result == str(target)

    def test_stale_entry_removed(self, tmp_path):
        fm = make_fm(tmp_path)
        # Point cache at a path that doesn't exist
        fm._upload_cache["abc123"] = str(tmp_path / "gone.pkg.slp")

        result = fm.check_upload_cache("abc123", "gone.pkg.slp")
        assert result is None
        # Stale entry should be evicted
        assert "abc123" not in fm._upload_cache


# ---------------------------------------------------------------------------
# start_upload_session
# ---------------------------------------------------------------------------


class TestStartUploadSession:
    @pytest.mark.asyncio
    async def test_success_no_subdir(self, tmp_path):
        fm = make_fm(tmp_path)
        ch = fake_channel()

        await fm.start_upload_session(ch, "labels.pkg.slp", 100, str(tmp_path), "0")

        ch.send.assert_called_once_with(MSG_FILE_UPLOAD_READY)
        assert fm._upload_session is not None
        assert fm._upload_session["filename"] == "labels.pkg.slp"
        assert fm._upload_session["total_bytes"] == 100
        assert fm._upload_session["file_path"] == tmp_path / "labels.pkg.slp"

    @pytest.mark.asyncio
    async def test_success_creates_subdir(self, tmp_path):
        fm = make_fm(tmp_path)
        ch = fake_channel()

        await fm.start_upload_session(ch, "labels.pkg.slp", 100, str(tmp_path), "1")

        ch.send.assert_called_once_with(MSG_FILE_UPLOAD_READY)
        expected_path = tmp_path / "sleap_connect_downloads" / "labels.pkg.slp"
        assert fm._upload_session["file_path"] == expected_path
        assert expected_path.parent.is_dir()

    @pytest.mark.asyncio
    async def test_dest_outside_mounts_rejected(self, tmp_path):
        fm = make_fm(tmp_path)
        ch = fake_channel()
        outside = str(tmp_path.parent / "outside")

        await fm.start_upload_session(ch, "labels.pkg.slp", 100, outside, "0")

        sent = ch.send.call_args[0][0]
        assert sent.startswith(MSG_FILE_UPLOAD_ERROR)
        assert "outside configured mounts" in sent
        assert fm._upload_session is None


# ---------------------------------------------------------------------------
# receive_upload_chunk
# ---------------------------------------------------------------------------


class TestReceiveUploadChunk:
    @pytest.mark.asyncio
    async def test_chunk_written_to_disk(self, tmp_path):
        fm = make_fm(tmp_path)
        ch = fake_channel()
        await fm.start_upload_session(ch, "f.pkg.slp", 5, str(tmp_path), "0")
        ch.reset_mock()

        fm.receive_upload_chunk(b"hello")

        assert fm._upload_session["bytes_received"] == 5

    @pytest.mark.asyncio
    async def test_progress_sent_after_500ms(self, tmp_path):
        fm = make_fm(tmp_path)
        ch = fake_channel()
        await fm.start_upload_session(ch, "f.pkg.slp", 5, str(tmp_path), "0")
        # Force last_progress_time into the past so progress fires immediately.
        fm._upload_session["last_progress_time"] = 0.0
        ch.reset_mock()

        fm.receive_upload_chunk(b"hello")

        sent = ch.send.call_args[0][0]
        assert sent.startswith(MSG_FILE_UPLOAD_PROGRESS)
        assert "5" in sent  # bytes_received

    @pytest.mark.asyncio
    async def test_write_error_sends_error_and_cleans_up(self, tmp_path):
        fm = make_fm(tmp_path)
        ch = fake_channel()
        await fm.start_upload_session(ch, "f.pkg.slp", 5, str(tmp_path), "0")
        ch.reset_mock()

        # Simulate an I/O error on write
        fm._upload_session["file_handle"].write = MagicMock(
            side_effect=OSError("disk full")
        )

        fm.receive_upload_chunk(b"hello")

        sent = ch.send.call_args[0][0]
        assert sent.startswith(MSG_FILE_UPLOAD_ERROR)
        assert "disk full" in sent
        assert fm._upload_session is None


# ---------------------------------------------------------------------------
# finish_upload_session
# ---------------------------------------------------------------------------


class TestFinishUploadSession:
    @pytest.mark.asyncio
    async def test_complete_success(self, tmp_path):
        payload = b"hello world"
        fm = make_fm(tmp_path)
        ch = fake_channel()
        await fm.start_upload_session(ch, "f.pkg.slp", len(payload), str(tmp_path), "0")
        ch.reset_mock()

        fm.receive_upload_chunk(payload)
        await fm.finish_upload_session(ch)

        sent = ch.send.call_args[0][0]
        assert sent.startswith(MSG_FILE_UPLOAD_COMPLETE)
        expected_path = tmp_path / "f.pkg.slp"
        assert str(expected_path) in sent
        assert expected_path.read_bytes() == payload

    @pytest.mark.asyncio
    async def test_cache_populated_after_success(self, tmp_path):
        payload = b"test data"
        fm = make_fm(tmp_path)
        ch = fake_channel()
        await fm.start_upload_session(ch, "f.pkg.slp", len(payload), str(tmp_path), "0")
        fm.receive_upload_chunk(payload)
        await fm.finish_upload_session(ch)

        expected_sha256 = sha256_of(payload)
        assert expected_sha256 in fm._upload_cache
        assert fm._upload_cache[expected_sha256] == str(tmp_path / "f.pkg.slp")

    @pytest.mark.asyncio
    async def test_size_mismatch_sends_error_and_deletes(self, tmp_path):
        fm = make_fm(tmp_path)
        ch = fake_channel()
        # Declare 100 bytes but only send 5
        await fm.start_upload_session(ch, "f.pkg.slp", 100, str(tmp_path), "0")
        ch.reset_mock()
        fm.receive_upload_chunk(b"hello")
        await fm.finish_upload_session(ch)

        sent = ch.send.call_args[0][0]
        assert sent.startswith(MSG_FILE_UPLOAD_ERROR)
        assert "mismatch" in sent
        assert not (tmp_path / "f.pkg.slp").exists()

    @pytest.mark.asyncio
    async def test_no_active_session_is_noop(self, tmp_path):
        fm = make_fm(tmp_path)
        ch = fake_channel()
        await fm.finish_upload_session(ch)  # should not raise
        ch.send.assert_not_called()
