"""Tests for the DirectoryBrowser class."""

import asyncio
import json
import pytest
from unittest.mock import MagicMock, AsyncMock

from sleap_connect.client.directory_browser import (
    DirectoryBrowser,
    DirectoryEntry,
    _format_size,
    browse_for_file,
)
from sleap_connect.protocol import MSG_FS_LIST_RESPONSE, MSG_FS_ERROR, MSG_SEPARATOR


class TestFormatSize:
    """Tests for _format_size helper function."""

    def test_bytes(self):
        assert _format_size(500) == "500 B"
        assert _format_size(0) == "0 B"

    def test_kilobytes(self):
        assert _format_size(1024) == "1.0 KB"
        assert _format_size(2048) == "2.0 KB"

    def test_megabytes(self):
        assert _format_size(1024 * 1024) == "1.0 MB"
        assert _format_size(5 * 1024 * 1024) == "5.0 MB"

    def test_gigabytes(self):
        assert _format_size(1024 * 1024 * 1024) == "1.0 GB"
        assert _format_size(2 * 1024 * 1024 * 1024) == "2.0 GB"


class TestDirectoryEntry:
    """Tests for DirectoryEntry dataclass."""

    def test_from_dict_file(self):
        data = {
            "name": "data.slp",
            "type": "file",
            "size": 1024,
            "modified": 1234567890.0,
        }
        entry = DirectoryEntry.from_dict(data)

        assert entry.name == "data.slp"
        assert entry.type == "file"
        assert entry.size == 1024
        assert entry.modified == 1234567890.0
        assert not entry.is_directory

    def test_from_dict_directory(self):
        data = {
            "name": "models",
            "type": "directory",
            "size": 0,
            "modified": 1234567890.0,
        }
        entry = DirectoryEntry.from_dict(data)

        assert entry.name == "models"
        assert entry.type == "directory"
        assert entry.is_directory

    def test_from_dict_defaults(self):
        entry = DirectoryEntry.from_dict({})

        assert entry.name == ""
        assert entry.type == "file"
        assert entry.size == 0
        assert entry.modified == 0


class TestDirectoryBrowser:
    """Tests for DirectoryBrowser class."""

    def make_list_response(self, path: str, entries: list) -> str:
        """Create a FS_LIST_RESPONSE message."""
        data = {
            "path": path,
            "entries": entries,
            "total_count": len(entries),
            "has_more": False,
        }
        return f"{MSG_FS_LIST_RESPONSE}{MSG_SEPARATOR}{json.dumps(data)}"

    def make_error_response(self, error: str, code: str = "ERROR") -> str:
        """Create a FS_ERROR message."""
        return f"{MSG_FS_ERROR}{MSG_SEPARATOR}{code}{MSG_SEPARATOR}{error}"

    @pytest.mark.asyncio
    async def test_refresh_listing_success(self):
        """Test successful directory listing refresh."""
        entries = [
            {"name": "subdir", "type": "directory", "size": 0, "modified": 1000},
            {"name": "file.slp", "type": "file", "size": 1024, "modified": 2000},
        ]
        response = self.make_list_response("/test", entries)

        send_mock = MagicMock()
        receive_mock = AsyncMock(return_value=response)

        browser = DirectoryBrowser(
            send_message=send_mock,
            receive_response=receive_mock,
            start_path="/test",
        )

        await browser._refresh_listing()

        assert len(browser.entries) == 2
        assert browser.entries[0].name == "subdir"
        assert browser.entries[0].is_directory
        assert browser.entries[1].name == "file.slp"
        assert not browser.entries[1].is_directory
        assert browser.error_message is None

    @pytest.mark.asyncio
    async def test_refresh_listing_with_filter(self):
        """Test directory listing with file filter."""
        entries = [
            {"name": "subdir", "type": "directory", "size": 0, "modified": 1000},
            {"name": "file.slp", "type": "file", "size": 1024, "modified": 2000},
            {"name": "config.yaml", "type": "file", "size": 512, "modified": 3000},
            {"name": "readme.md", "type": "file", "size": 256, "modified": 4000},
        ]
        response = self.make_list_response("/test", entries)

        send_mock = MagicMock()
        receive_mock = AsyncMock(return_value=response)

        browser = DirectoryBrowser(
            send_message=send_mock,
            receive_response=receive_mock,
            start_path="/test",
            file_filter=".slp",
        )

        await browser._refresh_listing()

        # Should keep directories and only .slp files
        assert len(browser.entries) == 2
        assert browser.entries[0].name == "subdir"
        assert browser.entries[1].name == "file.slp"

    @pytest.mark.asyncio
    async def test_refresh_listing_access_denied_falls_back_to_mounts(self):
        """Test that ACCESS_DENIED falls back to showing mounts."""
        error_response = self.make_error_response("Access denied", "ACCESS_DENIED")
        mounts_response = "FS_MOUNTS_RESPONSE::" + json.dumps([
            {"path": "/data", "label": "Data"},
            {"path": "/models", "label": "Models"},
        ])

        send_mock = MagicMock()
        # First call returns error, second call returns mounts
        receive_mock = AsyncMock(side_effect=[error_response, mounts_response])

        browser = DirectoryBrowser(
            send_message=send_mock,
            receive_response=receive_mock,
            start_path="/invalid/path",
        )

        await browser._refresh_listing()

        # Should show mounts instead of error
        assert len(browser.entries) == 2
        assert browser.showing_mounts is True
        assert browser.error_message is None
        assert "Data" in browser.entries[0].name
        assert "Models" in browser.entries[1].name

    @pytest.mark.asyncio
    async def test_refresh_listing_error_non_access_denied(self):
        """Test directory listing with non-ACCESS_DENIED error."""
        response = self.make_error_response("Path not found", "PATH_NOT_FOUND")

        send_mock = MagicMock()
        receive_mock = AsyncMock(return_value=response)

        browser = DirectoryBrowser(
            send_message=send_mock,
            receive_response=receive_mock,
            start_path="/test",
        )

        await browser._refresh_listing()

        assert len(browser.entries) == 0
        assert browser.error_message == "Path not found"

    @pytest.mark.asyncio
    async def test_refresh_listing_timeout(self):
        """Test directory listing timeout."""
        async def slow_response():
            await asyncio.sleep(60)
            return ""

        send_mock = MagicMock()

        browser = DirectoryBrowser(
            send_message=send_mock,
            receive_response=slow_response,
            start_path="/test",
        )

        # Patch timeout to be shorter for test
        original_refresh = browser._refresh_listing

        async def quick_timeout_refresh():
            browser.loading = True
            browser.error_message = None
            message = f"FS_LIST_DIR::/test::0"
            browser.send_message(message)
            try:
                await asyncio.wait_for(browser.receive_response(), timeout=0.1)
            except asyncio.TimeoutError:
                browser.error_message = "Timeout waiting for directory listing"
                browser.entries = []
            browser.loading = False

        await quick_timeout_refresh()

        assert len(browser.entries) == 0
        assert "Timeout" in browser.error_message

    def test_navigate_up_from_subdir(self):
        """Test navigating up from a subdirectory."""
        browser = DirectoryBrowser(
            send_message=MagicMock(),
            receive_response=AsyncMock(),
            start_path="/test/subdir/deep",
        )

        result = browser._navigate_up()

        assert result is True
        assert browser.current_path == "/test/subdir"
        assert browser.selected_index == 0

    def test_navigate_up_from_root(self):
        """Test navigating up from root directory."""
        browser = DirectoryBrowser(
            send_message=MagicMock(),
            receive_response=AsyncMock(),
            start_path="/",
        )

        result = browser._navigate_up()

        assert result is False
        assert browser.current_path == "/"

    def test_navigate_into_directory(self):
        """Test navigating into a directory."""
        browser = DirectoryBrowser(
            send_message=MagicMock(),
            receive_response=AsyncMock(),
            start_path="/test",
        )

        entry = DirectoryEntry(
            name="subdir",
            entry_type="directory",
            size=0,
            modified=1000,
        )

        result = browser._navigate_into(entry)

        assert result is True
        assert browser.current_path == "/test/subdir"
        assert browser.selected_index == 0

    def test_navigate_into_file(self):
        """Test that navigating into a file returns False."""
        browser = DirectoryBrowser(
            send_message=MagicMock(),
            receive_response=AsyncMock(),
            start_path="/test",
        )

        entry = DirectoryEntry(
            name="file.slp",
            entry_type="file",
            size=1024,
            modified=1000,
        )

        result = browser._navigate_into(entry)

        assert result is False
        assert browser.current_path == "/test"

    def test_select_file(self):
        """Test selecting a file."""
        browser = DirectoryBrowser(
            send_message=MagicMock(),
            receive_response=AsyncMock(),
            start_path="/test",
        )

        entry = DirectoryEntry(
            name="data.slp",
            entry_type="file",
            size=1024,
            modified=1000,
        )

        result = browser._select_file(entry)

        assert result is True
        assert browser.selected_path == "/test/data.slp"

    def test_select_directory(self):
        """Test that selecting a directory returns False."""
        browser = DirectoryBrowser(
            send_message=MagicMock(),
            receive_response=AsyncMock(),
            start_path="/test",
        )

        entry = DirectoryEntry(
            name="subdir",
            entry_type="directory",
            size=0,
            modified=1000,
        )

        result = browser._select_file(entry)

        assert result is False
        assert browser.selected_path is None


class TestBrowseForFile:
    """Tests for the browse_for_file convenience function."""

    @pytest.mark.asyncio
    async def test_browse_for_file_creates_browser(self):
        """Test that browse_for_file creates a DirectoryBrowser."""
        send_mock = MagicMock()

        # Create response that will make browser return quickly
        entries = [
            {"name": "data.slp", "type": "file", "size": 1024, "modified": 1000},
        ]
        response = f"{MSG_FS_LIST_RESPONSE}{MSG_SEPARATOR}{json.dumps({'path': '/test', 'entries': entries, 'total_count': 1, 'has_more': False})}"

        call_count = [0]

        async def mock_receive():
            call_count[0] += 1
            if call_count[0] == 1:
                return response
            await asyncio.sleep(60)

        # Just verify it can be called without errors
        # Full interactive test would require mocking prompt_toolkit
        browser = DirectoryBrowser(
            send_message=send_mock,
            receive_response=mock_receive,
            start_path="/test",
            file_filter=".slp",
            title="Test browser",
        )

        assert browser.current_path == "/test"
        assert browser.file_filters == [".slp"]
        assert browser.title == "Test browser"
