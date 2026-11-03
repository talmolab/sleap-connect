"""Tests for Worker filesystem message routing."""

import json
import pytest

from sleap_connect.config import MountConfig
from sleap_connect.protocol import (
    MSG_FS_GET_INFO,
    MSG_FS_INFO_RESPONSE,
    MSG_FS_GET_MOUNTS,
    MSG_FS_MOUNTS_RESPONSE,
    MSG_FS_RESOLVE,
    MSG_FS_RESOLVE_RESPONSE,
    MSG_FS_LIST_DIR,
    MSG_FS_LIST_RESPONSE,
    MSG_FS_ERROR,
    MSG_SEPARATOR,
    FS_ERROR_ACCESS_DENIED,
    FS_ERROR_INVALID_REQUEST,
    FS_ERROR_PATTERN_TOO_BROAD,
)
from sleap_connect.worker.worker_class import RTCWorkerClient


@pytest.fixture
def temp_mount(tmp_path):
    """Create a temporary mount with test files."""
    (tmp_path / "subdir").mkdir()
    (tmp_path / "fly.slp").write_text("test")
    (tmp_path / "fly_tracking.slp").write_text("test" * 100)
    (tmp_path / "subdir" / "nested_fly.slp").write_text("test")
    return tmp_path


@pytest.fixture
def worker_with_mount(temp_mount):
    """Create a Worker with a test mount configured."""
    mount = MountConfig(path=str(temp_mount), label="Test Mount")
    worker = RTCWorkerClient(mounts=[mount], working_dir=str(temp_mount))
    worker.peer_id = "test-worker-123"
    return worker


class TestIsfsMessage:
    """Tests for _is_fs_message detection."""

    def test_fs_get_info(self, worker_with_mount):
        """Test detection of FS_GET_INFO message."""
        assert worker_with_mount._is_fs_message("FS_GET_INFO") is True

    def test_fs_get_mounts(self, worker_with_mount):
        """Test detection of FS_GET_MOUNTS message."""
        assert worker_with_mount._is_fs_message("FS_GET_MOUNTS") is True

    def test_fs_resolve(self, worker_with_mount):
        """Test detection of FS_RESOLVE message."""
        assert worker_with_mount._is_fs_message("FS_RESOLVE::fly.slp::1000") is True

    def test_fs_list_dir(self, worker_with_mount):
        """Test detection of FS_LIST_DIR message."""
        assert worker_with_mount._is_fs_message("FS_LIST_DIR::/mnt/data") is True

    def test_non_fs_message(self, worker_with_mount):
        """Test non-FS messages are not detected."""
        assert worker_with_mount._is_fs_message("PACKAGE_TYPE::train") is False
        assert worker_with_mount._is_fs_message("END_OF_FILE") is False

    def test_bytes_message(self, worker_with_mount):
        """Test bytes messages are not detected as FS."""
        assert worker_with_mount._is_fs_message(b"FS_GET_INFO") is False

    def test_empty_message(self, worker_with_mount):
        """Test empty message handling."""
        assert worker_with_mount._is_fs_message("") is False


class TestHandleFsGetInfo:
    """Tests for FS_GET_INFO message handling."""

    def test_get_info_response(self, worker_with_mount):
        """Test FS_GET_INFO returns worker info."""
        response = worker_with_mount.handle_fs_message(MSG_FS_GET_INFO)

        assert response.startswith(MSG_FS_INFO_RESPONSE)
        parts = response.split(MSG_SEPARATOR)
        assert len(parts) == 2

        data = json.loads(parts[1])
        assert data["worker_id"] == "test-worker-123"
        assert "mounts" in data
        assert "working_dir" in data

    def test_get_info_includes_mounts(self, worker_with_mount, temp_mount):
        """Test worker info includes configured mounts."""
        response = worker_with_mount.handle_fs_message(MSG_FS_GET_INFO)
        data = json.loads(response.split(MSG_SEPARATOR)[1])

        assert len(data["mounts"]) == 1
        assert data["mounts"][0]["label"] == "Test Mount"
        assert data["mounts"][0]["path"] == str(temp_mount)


class TestHandleFsGetMounts:
    """Tests for FS_GET_MOUNTS message handling."""

    def test_get_mounts_response(self, worker_with_mount, temp_mount):
        """Test FS_GET_MOUNTS returns mount list."""
        response = worker_with_mount.handle_fs_message(MSG_FS_GET_MOUNTS)

        assert response.startswith(MSG_FS_MOUNTS_RESPONSE)
        parts = response.split(MSG_SEPARATOR)

        mounts = json.loads(parts[1])
        assert len(mounts) == 1
        assert mounts[0]["label"] == "Test Mount"
        assert mounts[0]["path"] == str(temp_mount)

    def test_get_mounts_empty(self):
        """Test FS_GET_MOUNTS with no mounts configured."""
        worker = RTCWorkerClient()
        response = worker.handle_fs_message(MSG_FS_GET_MOUNTS)

        mounts = json.loads(response.split(MSG_SEPARATOR)[1])
        assert mounts == []


class TestHandleFsResolve:
    """Tests for FS_RESOLVE message handling."""

    def test_resolve_exact_match(self, worker_with_mount):
        """Test resolving an exact filename."""
        message = f"{MSG_FS_RESOLVE}{MSG_SEPARATOR}fly.slp"
        response = worker_with_mount.handle_fs_message(message)

        assert response.startswith(MSG_FS_RESOLVE_RESPONSE)
        data = json.loads(response.split(MSG_SEPARATOR)[1])

        assert len(data["candidates"]) >= 1
        assert any(c["name"] == "fly.slp" for c in data["candidates"])

    def test_resolve_wildcard_pattern(self, worker_with_mount):
        """Test resolving a wildcard pattern."""
        message = f"{MSG_FS_RESOLVE}{MSG_SEPARATOR}fly*.slp"
        response = worker_with_mount.handle_fs_message(message)

        assert response.startswith(MSG_FS_RESOLVE_RESPONSE)
        data = json.loads(response.split(MSG_SEPARATOR)[1])

        assert len(data["candidates"]) >= 2
        for c in data["candidates"]:
            assert "fly" in c["name"].lower()

    def test_resolve_with_file_size(self, worker_with_mount):
        """Test resolving with file size parameter."""
        message = f"{MSG_FS_RESOLVE}{MSG_SEPARATOR}fly.slp{MSG_SEPARATOR}4"
        response = worker_with_mount.handle_fs_message(message)

        assert response.startswith(MSG_FS_RESOLVE_RESPONSE)
        data = json.loads(response.split(MSG_SEPARATOR)[1])
        assert "candidates" in data

    def test_resolve_pattern_too_broad(self, worker_with_mount):
        """Test rejection of too-broad patterns."""
        message = f"{MSG_FS_RESOLVE}{MSG_SEPARATOR}*.x"
        response = worker_with_mount.handle_fs_message(message)

        assert response.startswith(MSG_FS_ERROR)
        parts = response.split(MSG_SEPARATOR)
        assert parts[1] == FS_ERROR_PATTERN_TOO_BROAD

    def test_resolve_missing_pattern(self, worker_with_mount):
        """Test error when pattern is missing."""
        message = f"{MSG_FS_RESOLVE}{MSG_SEPARATOR}"
        response = worker_with_mount.handle_fs_message(message)

        assert response.startswith(MSG_FS_ERROR)
        parts = response.split(MSG_SEPARATOR)
        assert parts[1] == FS_ERROR_INVALID_REQUEST

    def test_resolve_includes_metadata(self, worker_with_mount):
        """Test that resolve includes search metadata."""
        message = f"{MSG_FS_RESOLVE}{MSG_SEPARATOR}fly.slp"
        response = worker_with_mount.handle_fs_message(message)

        data = json.loads(response.split(MSG_SEPARATOR)[1])
        assert "truncated" in data
        assert "timeout" in data
        assert "search_time_ms" in data


class TestHandleFsListDir:
    """Tests for FS_LIST_DIR message handling."""

    def test_list_directory(self, worker_with_mount, temp_mount):
        """Test listing a directory."""
        message = f"{MSG_FS_LIST_DIR}{MSG_SEPARATOR}{temp_mount}"
        response = worker_with_mount.handle_fs_message(message)

        assert response.startswith(MSG_FS_LIST_RESPONSE)
        data = json.loads(response.split(MSG_SEPARATOR)[1])

        assert len(data["entries"]) >= 1
        assert "total_count" in data
        assert "has_more" in data

    def test_list_directory_with_offset(self, worker_with_mount, temp_mount):
        """Test listing with offset for pagination."""
        message = f"{MSG_FS_LIST_DIR}{MSG_SEPARATOR}{temp_mount}{MSG_SEPARATOR}1"
        response = worker_with_mount.handle_fs_message(message)

        assert response.startswith(MSG_FS_LIST_RESPONSE)
        data = json.loads(response.split(MSG_SEPARATOR)[1])
        assert "entries" in data

    def test_list_directory_outside_mounts(self, worker_with_mount):
        """Test listing directory outside mounts is denied."""
        message = f"{MSG_FS_LIST_DIR}{MSG_SEPARATOR}/etc"
        response = worker_with_mount.handle_fs_message(message)

        assert response.startswith(MSG_FS_ERROR)
        parts = response.split(MSG_SEPARATOR)
        assert parts[1] == FS_ERROR_ACCESS_DENIED

    def test_list_missing_path(self, worker_with_mount):
        """Test error when path is missing."""
        message = f"{MSG_FS_LIST_DIR}{MSG_SEPARATOR}"
        response = worker_with_mount.handle_fs_message(message)

        assert response.startswith(MSG_FS_ERROR)
        parts = response.split(MSG_SEPARATOR)
        assert parts[1] == FS_ERROR_INVALID_REQUEST

    def test_list_nonexistent_path(self, worker_with_mount, temp_mount):
        """Test listing non-existent path."""
        message = f"{MSG_FS_LIST_DIR}{MSG_SEPARATOR}{temp_mount}/nonexistent"
        response = worker_with_mount.handle_fs_message(message)

        assert response.startswith(MSG_FS_ERROR)


class TestUnknownMessage:
    """Tests for unknown message handling."""

    def test_unknown_fs_message(self, worker_with_mount):
        """Test handling of unknown FS_* message types."""
        response = worker_with_mount.handle_fs_message("FS_UNKNOWN")

        assert response.startswith(MSG_FS_ERROR)
        parts = response.split(MSG_SEPARATOR)
        assert parts[1] == FS_ERROR_INVALID_REQUEST
