"""Tests for FileManager filesystem browsing operations."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sleap_rtc.config import MountConfig
from sleap_rtc.worker.file_manager import FileManager


@pytest.fixture
def temp_mount(tmp_path):
    """Create a temporary mount with test files."""
    # Create directory structure
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "nested").mkdir()

    # Create test files
    (tmp_path / "fly.slp").write_text("test")
    (tmp_path / "fly_tracking.slp").write_text("test" * 100)
    (tmp_path / "mouse.slp").write_text("test" * 50)
    (tmp_path / "subdir" / "deep_fly.slp").write_text("test")
    (tmp_path / "data.csv").write_text("a,b,c")

    return tmp_path


@pytest.fixture
def file_manager(temp_mount):
    """Create a FileManager with a test mount."""
    mount = MountConfig(path=str(temp_mount), label="Test Mount")
    return FileManager(mounts=[mount])


class TestWildcardDetection:
    """Tests for wildcard pattern detection."""

    def test_no_wildcards(self):
        """Test detection of patterns without wildcards."""
        fm = FileManager()
        assert fm._is_wildcard_pattern("fly.slp") is False
        assert fm._is_wildcard_pattern("some_file.txt") is False

    def test_star_wildcard(self):
        """Test detection of * wildcard."""
        fm = FileManager()
        assert fm._is_wildcard_pattern("fly*.slp") is True
        assert fm._is_wildcard_pattern("*.slp") is True

    def test_question_wildcard(self):
        """Test detection of ? wildcard."""
        fm = FileManager()
        assert fm._is_wildcard_pattern("fly?.slp") is True

    def test_bracket_wildcard(self):
        """Test detection of [...] wildcard."""
        fm = FileManager()
        assert fm._is_wildcard_pattern("fly[0-9].slp") is True
        assert fm._is_wildcard_pattern("fly[abc].slp") is True


class TestPatternValidation:
    """Tests for pattern validation."""

    def test_valid_pattern(self):
        """Test validation of valid patterns."""
        fm = FileManager()
        is_valid, error = fm._validate_pattern("fly*.slp")
        assert is_valid is True
        assert error is None

    def test_pattern_too_broad(self):
        """Test rejection of patterns with too few non-wildcard chars."""
        fm = FileManager()
        # "*.x" has only 2 non-wildcard characters (. and x)
        is_valid, error = fm._validate_pattern("*.x")
        assert is_valid is False
        assert "at least 3 non-wildcard" in error

    def test_empty_pattern(self):
        """Test rejection of empty pattern."""
        fm = FileManager()
        is_valid, error = fm._validate_pattern("")
        assert is_valid is False

    def test_only_wildcards(self):
        """Test rejection of pattern with only wildcards."""
        fm = FileManager()
        is_valid, error = fm._validate_pattern("*.*")
        assert is_valid is False


class TestFilenameMatching:
    """Tests for filename matching."""

    def test_exact_match(self):
        """Test exact filename match."""
        fm = FileManager()
        result = fm._match_filename("fly.slp", "fly.slp")
        assert result["matches"] is True
        assert result["match_type"] == "exact"

    def test_exact_match_case_insensitive(self):
        """Test case-insensitive exact match."""
        fm = FileManager()
        result = fm._match_filename("FLY.SLP", "fly.slp")
        assert result["matches"] is True
        assert result["match_type"] == "exact"

    def test_wildcard_match_star(self):
        """Test wildcard match with *."""
        fm = FileManager()
        result = fm._match_filename("fly*.slp", "fly_tracking.slp")
        assert result["matches"] is True
        assert result["match_type"] == "wildcard"

    def test_wildcard_match_question(self):
        """Test wildcard match with ?."""
        fm = FileManager()
        result = fm._match_filename("fly?.slp", "fly1.slp")
        assert result["matches"] is True
        assert result["match_type"] == "wildcard"

    def test_substring_match(self):
        """Test substring match."""
        fm = FileManager()
        result = fm._match_filename("tracking", "fly_tracking.slp")
        assert result["matches"] is True
        assert result["match_type"] == "substring"

    def test_no_match(self):
        """Test no match case."""
        fm = FileManager()
        result = fm._match_filename("mouse", "fly.slp")
        assert result["matches"] is False


class TestPathResolution:
    """Tests for path resolution."""

    def test_resolve_exact_filename(self, file_manager, temp_mount):
        """Test resolving an exact filename."""
        result = file_manager.resolve_path("fly.slp")
        assert len(result["candidates"]) >= 1
        assert result["timeout"] is False

        # Should find the exact match
        filenames = [c["name"] for c in result["candidates"]]
        assert "fly.slp" in filenames

    def test_resolve_wildcard_pattern(self, file_manager):
        """Test resolving a wildcard pattern."""
        result = file_manager.resolve_path("fly*.slp")
        assert len(result["candidates"]) >= 1
        assert result["timeout"] is False

        # All matches should contain 'fly' and end with '.slp'
        for candidate in result["candidates"]:
            assert "fly" in candidate["name"].lower()
            assert candidate["name"].endswith(".slp")

    def test_resolve_too_broad_pattern(self, file_manager):
        """Test rejection of too-broad patterns."""
        # "*.x" has only 2 non-wildcard characters
        result = file_manager.resolve_path("*.x")
        assert len(result["candidates"]) == 0
        assert result.get("error_code") == "PATTERN_TOO_BROAD"

    def test_resolve_respects_max_results(self, file_manager):
        """Test that results are limited to MAX_RESULTS."""
        result = file_manager.resolve_path("*.slp")
        # Pattern rejected as too broad, so no candidates
        # But a valid pattern would be limited
        assert len(result.get("candidates", [])) <= file_manager.MAX_RESULTS

    def test_resolve_includes_search_time(self, file_manager):
        """Test that search time is included in result."""
        result = file_manager.resolve_path("fly.slp")
        assert "search_time_ms" in result
        assert isinstance(result["search_time_ms"], int)

    def test_resolve_nested_files(self, file_manager):
        """Test that nested files are found."""
        result = file_manager.resolve_path("deep_fly.slp")
        assert len(result["candidates"]) >= 1

        paths = [c["path"] for c in result["candidates"]]
        assert any("subdir" in p for p in paths)


class TestDirectoryListing:
    """Tests for directory listing."""

    def test_list_mount_root(self, file_manager, temp_mount):
        """Test listing mount root directory."""
        result = file_manager.list_directory(str(temp_mount))
        assert len(result["entries"]) > 0
        assert result["total_count"] > 0
        assert "error" not in result

    def test_list_directory_pagination(self, file_manager, temp_mount):
        """Test pagination with offset."""
        # First get all entries
        full_result = file_manager.list_directory(str(temp_mount))

        # Then get with offset
        offset_result = file_manager.list_directory(str(temp_mount), offset=1)

        # Should have one fewer entry
        assert len(offset_result["entries"]) == len(full_result["entries"]) - 1

    def test_list_directory_sorts_correctly(self, file_manager, temp_mount):
        """Test that directories come first, then alphabetically."""
        result = file_manager.list_directory(str(temp_mount))

        entries = result["entries"]
        # Find first file index
        first_file_idx = next(
            (i for i, e in enumerate(entries) if e["type"] == "file"), len(entries)
        )

        # All entries before first file should be directories
        for i in range(first_file_idx):
            assert entries[i]["type"] == "directory"

    def test_list_directory_outside_mounts_denied(self, file_manager):
        """Test that paths outside mounts are denied."""
        result = file_manager.list_directory("/etc")
        assert result.get("error_code") == "ACCESS_DENIED"

    def test_list_directory_traversal_attack(self, file_manager, temp_mount):
        """Test that path traversal attacks are prevented."""
        # Try to escape mount via ..
        attack_path = str(temp_mount / ".." / ".." / "etc")
        result = file_manager.list_directory(attack_path)
        assert result.get("error_code") in ["ACCESS_DENIED", "PATH_NOT_FOUND"]

    def test_list_nonexistent_directory(self, file_manager, temp_mount):
        """Test listing non-existent directory."""
        result = file_manager.list_directory(str(temp_mount / "nonexistent"))
        assert result.get("error_code") == "PATH_NOT_FOUND"

    def test_list_file_as_directory(self, file_manager, temp_mount):
        """Test listing a file as directory."""
        result = file_manager.list_directory(str(temp_mount / "fly.slp"))
        assert result.get("error_code") == "PATH_NOT_FOUND"


class TestStatPath:
    """Tests for fs.stat's FileManager.stat_path (item 3.1)."""

    def test_stat_a_file(self, file_manager, temp_mount):
        result = file_manager.stat_path(str(temp_mount / "fly.slp"))
        assert result["type"] == "file"
        assert result["size"] == len("test")
        assert "error" not in result

    def test_stat_a_directory(self, file_manager, temp_mount):
        result = file_manager.stat_path(str(temp_mount / "subdir"))
        assert result["type"] == "directory"
        assert result["size"] == 0

    def test_stat_outside_mounts_denied(self, file_manager):
        result = file_manager.stat_path("/etc")
        assert result.get("error_code") == "ACCESS_DENIED"

    def test_stat_nonexistent_path(self, file_manager, temp_mount):
        result = file_manager.stat_path(str(temp_mount / "nonexistent"))
        assert result.get("error_code") == "PATH_NOT_FOUND"


class TestReadFile:
    """Tests for fs.read's FileManager.read_file (item 3.1) — small direct
    reads, NOT the bulk-transfer path (see the blob API for that)."""

    def test_reads_whole_small_file(self, file_manager, temp_mount):
        result = file_manager.read_file(str(temp_mount / "fly.slp"))
        assert result["size"] == len("test")
        assert result["eof"] is True
        import base64

        assert base64.b64decode(result["content_base64"]) == b"test"

    def test_reads_a_byte_range(self, file_manager, temp_mount):
        # "fly_tracking.slp" contains "test" * 100 == 400 bytes.
        result = file_manager.read_file(
            str(temp_mount / "fly_tracking.slp"), offset=4, length=4
        )
        import base64

        assert base64.b64decode(result["content_base64"]) == b"test"
        assert result["offset"] == 4
        assert result["total_size"] == 400
        assert result["eof"] is False

    def test_length_is_capped_at_max_read_bytes(
        self, file_manager, temp_mount, monkeypatch
    ):
        monkeypatch.setattr(FileManager, "MAX_READ_BYTES", 2)
        result = file_manager.read_file(str(temp_mount / "fly.slp"), length=100)
        assert result["size"] == 2

    def test_read_outside_mounts_denied(self, file_manager):
        result = file_manager.read_file("/etc/hosts")
        assert result.get("error_code") == "ACCESS_DENIED"

    def test_read_nonexistent_file(self, file_manager, temp_mount):
        result = file_manager.read_file(str(temp_mount / "nonexistent"))
        assert result.get("error_code") == "PATH_NOT_FOUND"

    def test_read_a_directory_is_rejected(self, file_manager, temp_mount):
        result = file_manager.read_file(str(temp_mount / "subdir"))
        assert result.get("error_code") == "PATH_NOT_FOUND"

    def test_negative_offset_is_rejected(self, file_manager, temp_mount):
        result = file_manager.read_file(str(temp_mount / "fly.slp"), offset=-1)
        assert result.get("error_code") == "INVALID_RANGE"


class TestPathAllowedCheck:
    """Tests for path security checks."""

    def test_path_within_mount_allowed(self, file_manager, temp_mount):
        """Test that paths within mounts are allowed."""
        path = Path(temp_mount) / "subdir"
        assert file_manager._is_path_allowed(path) is True

    def test_path_outside_mount_denied(self, file_manager):
        """Test that paths outside mounts are denied."""
        path = Path("/etc/passwd")
        assert file_manager._is_path_allowed(path) is False

    def test_no_mounts_all_denied(self):
        """Test that all paths are denied when no mounts configured."""
        fm = FileManager()
        assert fm._is_path_allowed(Path("/any/path")) is False

    def test_symlink_within_mount_allowed(self, file_manager, temp_mount):
        """Test that symlinks within mounts are allowed."""
        # Create a symlink within the mount
        link_path = temp_mount / "link"
        target_path = temp_mount / "subdir"
        try:
            link_path.symlink_to(target_path)
            assert file_manager._is_path_allowed(link_path) is True
        except OSError:
            pytest.skip("Cannot create symlinks on this system")


class TestGetMountsAndWorkerInfo:
    """Tests for mount and worker info retrieval."""

    def test_get_mounts(self, file_manager, temp_mount):
        """Test getting mount list."""
        mounts = file_manager.get_mounts()
        assert len(mounts) == 1
        assert mounts[0]["path"] == str(temp_mount)
        assert mounts[0]["label"] == "Test Mount"

    def test_get_mounts_empty(self):
        """Test getting empty mount list."""
        fm = FileManager()
        assert fm.get_mounts() == []

    def test_get_worker_info(self, file_manager):
        """Test getting worker info."""
        info = file_manager.get_worker_info(worker_id="test-worker")
        assert info["worker_id"] == "test-worker"
        assert "mounts" in info
        assert "working_dir" in info

    def test_get_worker_info_with_working_dir(self):
        """Test worker info includes working directory."""
        fm = FileManager(working_dir="/mnt/work")
        info = fm.get_worker_info()
        assert info["working_dir"] == "/mnt/work"


class TestCheckVideoAccessibility:
    """Tests for SLP video accessibility checking."""

    def _create_mock_video(self, filename, is_embedded=False, backend_metadata=None):
        """Create a mock Video object for testing."""
        video = MagicMock()
        video.filename = filename
        video.backend = None
        video.backend_metadata = backend_metadata or {}
        if is_embedded:
            video.backend_metadata["has_embedded_images"] = True
        return video

    def _create_mock_labels(self, videos):
        """Create a mock Labels object with given videos."""
        labels = MagicMock()
        labels.videos = videos
        return labels

    def test_all_videos_accessible(self, file_manager, temp_mount):
        """Test check when all videos are accessible."""
        # Create actual video files
        video1 = temp_mount / "video1.mp4"
        video2 = temp_mount / "video2.mp4"
        video1.write_text("video data")
        video2.write_text("video data")

        # Create SLP file
        slp_file = temp_mount / "labels.slp"
        slp_file.write_text("slp data")

        # Mock sleap_io to return labels with accessible videos
        mock_labels = self._create_mock_labels(
            [
                self._create_mock_video(str(video1)),
                self._create_mock_video(str(video2)),
            ]
        )

        with patch("sleap_rtc.worker.file_manager.sio") as mock_sio:
            mock_sio.load_file.return_value = mock_labels

            result = file_manager.check_video_accessibility(str(slp_file))

            assert result["slp_path"] == str(slp_file)
            assert result["total_videos"] == 2
            assert result["accessible"] == 2
            assert result["missing"] == []
            assert result["embedded"] == 0
            assert "error" not in result

    def test_some_videos_missing(self, file_manager, temp_mount):
        """Test check when some videos are missing."""
        # Create only one video file
        video1 = temp_mount / "video1.mp4"
        video1.write_text("video data")

        # Create SLP file
        slp_file = temp_mount / "labels.slp"
        slp_file.write_text("slp data")

        # Mock sleap_io with one accessible and one missing video
        mock_labels = self._create_mock_labels(
            [
                self._create_mock_video(str(video1)),
                self._create_mock_video("/nonexistent/video2.mp4"),
            ]
        )

        with patch("sleap_rtc.worker.file_manager.sio") as mock_sio:
            mock_sio.load_file.return_value = mock_labels

            result = file_manager.check_video_accessibility(str(slp_file))

            assert result["total_videos"] == 2
            assert result["accessible"] == 1
            assert len(result["missing"]) == 1
            assert result["missing"][0]["filename"] == "video2.mp4"
            assert result["missing"][0]["original_path"] == "/nonexistent/video2.mp4"

    def test_embedded_videos_excluded(self, file_manager, temp_mount):
        """Test that embedded videos are excluded from check."""
        # Create SLP file
        slp_file = temp_mount / "labels.slp"
        slp_file.write_text("slp data")

        # Mock sleap_io with embedded video (no external file needed)
        mock_labels = self._create_mock_labels(
            [
                self._create_mock_video("/original/video.mp4", is_embedded=True),
            ]
        )

        with patch("sleap_rtc.worker.file_manager.sio") as mock_sio:
            mock_sio.load_file.return_value = mock_labels

            result = file_manager.check_video_accessibility(str(slp_file))

            assert result["total_videos"] == 1
            assert result["embedded"] == 1
            assert result["accessible"] == 0
            assert result["missing"] == []

    def test_slp_file_not_found(self, file_manager, temp_mount):
        """Test error when SLP file doesn't exist."""
        result = file_manager.check_video_accessibility("/nonexistent/labels.slp")

        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_sleap_io_not_available(self, temp_mount):
        """Test graceful handling when sleap-io is not available."""
        fm = FileManager()
        slp_file = temp_mount / "labels.slp"
        slp_file.write_text("slp data")

        with patch("sleap_rtc.worker.file_manager.SLEAP_IO_AVAILABLE", False):
            result = fm.check_video_accessibility(str(slp_file))

            assert "error" in result
            assert "not available" in result["error"].lower()

    def test_mixed_embedded_and_external_videos(self, file_manager, temp_mount):
        """Test with mix of embedded and external videos."""
        # Create one accessible video
        video1 = temp_mount / "accessible.mp4"
        video1.write_text("video data")

        # Create SLP file
        slp_file = temp_mount / "labels.slp"
        slp_file.write_text("slp data")

        # Mock: 1 embedded, 1 accessible external, 1 missing external
        mock_labels = self._create_mock_labels(
            [
                self._create_mock_video("/embedded/video.mp4", is_embedded=True),
                self._create_mock_video(str(video1)),
                self._create_mock_video("/missing/video.mp4"),
            ]
        )

        with patch("sleap_rtc.worker.file_manager.sio") as mock_sio:
            mock_sio.load_file.return_value = mock_labels

            result = file_manager.check_video_accessibility(str(slp_file))

            assert result["total_videos"] == 3
            assert result["embedded"] == 1
            assert result["accessible"] == 1
            assert len(result["missing"]) == 1
            assert result["missing"][0]["filename"] == "video.mp4"


class TestScanDirectoryForFilenames:
    """Tests for directory scanning (SLP Viewer style resolution)."""

    def test_all_filenames_found(self, file_manager, temp_mount):
        """Test scanning when all filenames are found."""
        # Create video files
        (temp_mount / "video1.mp4").write_text("video data")
        (temp_mount / "video2.mp4").write_text("video data")

        result = file_manager.scan_directory_for_filenames(
            directory=str(temp_mount),
            filenames=["video1.mp4", "video2.mp4"],
        )

        assert result["directory"] == str(temp_mount)
        assert result["found"]["video1.mp4"] == str(temp_mount / "video1.mp4")
        assert result["found"]["video2.mp4"] == str(temp_mount / "video2.mp4")
        assert "error" not in result

    def test_partial_matches(self, file_manager, temp_mount):
        """Test scanning when only some filenames are found."""
        # Create only one video file
        (temp_mount / "video1.mp4").write_text("video data")

        result = file_manager.scan_directory_for_filenames(
            directory=str(temp_mount),
            filenames=["video1.mp4", "video2.mp4", "video3.mp4"],
        )

        assert result["found"]["video1.mp4"] == str(temp_mount / "video1.mp4")
        assert result["found"]["video2.mp4"] is None
        assert result["found"]["video3.mp4"] is None

    def test_no_matches(self, file_manager, temp_mount):
        """Test scanning when no filenames are found."""
        result = file_manager.scan_directory_for_filenames(
            directory=str(temp_mount),
            filenames=["nonexistent1.mp4", "nonexistent2.mp4"],
        )

        assert result["found"]["nonexistent1.mp4"] is None
        assert result["found"]["nonexistent2.mp4"] is None
        assert "error" not in result

    def test_directory_outside_mounts_denied(self, file_manager):
        """Test that directories outside mounts are denied."""
        result = file_manager.scan_directory_for_filenames(
            directory="/etc",
            filenames=["passwd"],
        )

        assert result["error_code"] == "ACCESS_DENIED"
        assert (
            "outside" in result["error"].lower() or "denied" in result["error"].lower()
        )
        assert result["found"] == {}

    def test_directory_not_found(self, file_manager, temp_mount):
        """Test scanning non-existent directory."""
        result = file_manager.scan_directory_for_filenames(
            directory=str(temp_mount / "nonexistent"),
            filenames=["video.mp4"],
        )

        assert result["error_code"] == "PATH_NOT_FOUND"
        assert "not found" in result["error"].lower()

    def test_path_is_file_not_directory(self, file_manager, temp_mount):
        """Test scanning a file path instead of directory."""
        file_path = temp_mount / "file.txt"
        file_path.write_text("data")

        result = file_manager.scan_directory_for_filenames(
            directory=str(file_path),
            filenames=["video.mp4"],
        )

        assert result["error_code"] == "PATH_NOT_FOUND"
        assert "not a directory" in result["error"].lower()

    def test_path_traversal_in_filename_rejected(self, file_manager, temp_mount):
        """Test that path traversal in filenames is rejected."""
        result = file_manager.scan_directory_for_filenames(
            directory=str(temp_mount),
            filenames=["../../../etc/passwd", "video.mp4"],
        )

        # Path traversal filename should be None (not found/rejected)
        assert result["found"]["../../../etc/passwd"] is None
        # Normal filename check still works
        assert result["found"]["video.mp4"] is None  # Not found (doesn't exist)

    def test_empty_filenames_list(self, file_manager, temp_mount):
        """Test scanning with empty filenames list."""
        result = file_manager.scan_directory_for_filenames(
            directory=str(temp_mount),
            filenames=[],
        )

        assert result["found"] == {}
        assert "error" not in result

    def test_scan_subdirectory(self, file_manager, temp_mount):
        """Test scanning in a subdirectory within mount."""
        subdir = temp_mount / "subdir"
        (subdir / "video.mp4").write_text("video data")

        result = file_manager.scan_directory_for_filenames(
            directory=str(subdir),
            filenames=["video.mp4"],
        )

        assert result["found"]["video.mp4"] == str(subdir / "video.mp4")


class TestWriteSlpWithNewPaths:
    """Tests for SLP writing with updated video paths."""

    def _create_mock_video(self, filename, is_embedded=False, backend_metadata=None):
        """Create a mock Video object for testing."""
        video = MagicMock()
        video.filename = filename
        video.backend = None
        video.backend_metadata = backend_metadata or {}
        if is_embedded:
            video.backend_metadata["has_embedded_images"] = True
        return video

    def _create_mock_labels(self, videos):
        """Create a mock Labels object with given videos."""
        labels = MagicMock()
        labels.videos = videos
        return labels

    def test_write_slp_with_valid_filename_map(self, file_manager, temp_mount):
        """Test writing SLP with valid filename mappings."""
        # Create SLP file
        slp_file = temp_mount / "labels.slp"
        slp_file.write_text("slp data")

        # Create output directory
        output_dir = temp_mount / "output"
        output_dir.mkdir()

        # Mock sleap_io
        mock_labels = self._create_mock_labels(
            [
                self._create_mock_video("/old/path/video1.mp4"),
                self._create_mock_video("/old/path/video2.mp4"),
            ]
        )

        filename_map = {
            "/old/path/video1.mp4": "/new/path/video1.mp4",
            "/old/path/video2.mp4": "/new/path/video2.mp4",
        }

        with patch("sleap_rtc.worker.file_manager.sio") as mock_sio:
            mock_sio.load_file.return_value = mock_labels

            result = file_manager.write_slp_with_new_paths(
                slp_path=str(slp_file),
                output_dir=str(output_dir),
                filename_map=filename_map,
            )

            # Check result
            assert "output_path" in result
            assert result["videos_updated"] == 2
            assert "error" not in result

            # Verify sleap_io calls
            mock_sio.load_file.assert_called_once_with(str(slp_file), open_videos=False)
            mock_labels.replace_filenames.assert_called_once_with(
                filename_map=filename_map, open_videos=False
            )
            mock_labels.save.assert_called_once()

            # Check output path format
            output_path = result["output_path"]
            assert "resolved_" in output_path
            assert output_path.endswith(".slp")

    def test_write_slp_output_dir_outside_mounts_denied(self, file_manager, temp_mount):
        """Test that output directory outside mounts is rejected."""
        # Create SLP file within mount
        slp_file = temp_mount / "labels.slp"
        slp_file.write_text("slp data")

        result = file_manager.write_slp_with_new_paths(
            slp_path=str(slp_file),
            output_dir="/etc",
            filename_map={"/old/video.mp4": "/new/video.mp4"},
        )

        assert "error" in result
        assert result.get("error_code") == "ACCESS_DENIED"
        assert "outside" in result["error"].lower()

    def test_write_slp_file_not_found(self, file_manager, temp_mount):
        """Test error when SLP file doesn't exist."""
        # Create output directory
        output_dir = temp_mount / "output"
        output_dir.mkdir()

        result = file_manager.write_slp_with_new_paths(
            slp_path="/nonexistent/labels.slp",
            output_dir=str(output_dir),
            filename_map={"/old/video.mp4": "/new/video.mp4"},
        )

        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_write_slp_output_dir_not_found(self, file_manager, temp_mount):
        """Test error when output directory doesn't exist."""
        # Create SLP file
        slp_file = temp_mount / "labels.slp"
        slp_file.write_text("slp data")

        result = file_manager.write_slp_with_new_paths(
            slp_path=str(slp_file),
            output_dir=str(temp_mount / "nonexistent"),
            filename_map={"/old/video.mp4": "/new/video.mp4"},
        )

        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_write_slp_output_path_is_file(self, file_manager, temp_mount):
        """Test error when output path is a file, not directory."""
        # Create SLP file
        slp_file = temp_mount / "labels.slp"
        slp_file.write_text("slp data")

        # Create a file instead of directory
        output_file = temp_mount / "not_a_dir"
        output_file.write_text("file data")

        result = file_manager.write_slp_with_new_paths(
            slp_path=str(slp_file),
            output_dir=str(output_file),
            filename_map={"/old/video.mp4": "/new/video.mp4"},
        )

        assert "error" in result
        assert "not a directory" in result["error"].lower()

    def test_write_slp_pkg_extension_preserved(self, file_manager, temp_mount):
        """Test that .pkg.slp extension is preserved in output."""
        # Create .pkg.slp file
        slp_file = temp_mount / "labels.pkg.slp"
        slp_file.write_text("slp data")

        # Create output directory
        output_dir = temp_mount / "output"
        output_dir.mkdir()

        mock_labels = self._create_mock_labels(
            [
                self._create_mock_video("/old/video.mp4"),
            ]
        )

        with patch("sleap_rtc.worker.file_manager.sio") as mock_sio:
            mock_sio.load_file.return_value = mock_labels

            result = file_manager.write_slp_with_new_paths(
                slp_path=str(slp_file),
                output_dir=str(output_dir),
                filename_map={"/old/video.mp4": "/new/video.mp4"},
            )

            assert result["output_path"].endswith(".pkg.slp")
            assert "resolved_" in result["output_path"]

    def test_write_slp_sleap_io_not_available(self, temp_mount):
        """Test graceful handling when sleap-io is not available."""
        fm = FileManager()

        slp_file = temp_mount / "labels.slp"
        slp_file.write_text("slp data")

        with patch("sleap_rtc.worker.file_manager.SLEAP_IO_AVAILABLE", False):
            result = fm.write_slp_with_new_paths(
                slp_path=str(slp_file),
                output_dir=str(temp_mount),
                filename_map={"/old/video.mp4": "/new/video.mp4"},
            )

            assert "error" in result
            assert "not available" in result["error"].lower()

    def test_write_slp_load_failure(self, file_manager, temp_mount):
        """Test error handling when SLP loading fails."""
        # Create SLP file
        slp_file = temp_mount / "labels.slp"
        slp_file.write_text("slp data")

        # Create output directory
        output_dir = temp_mount / "output"
        output_dir.mkdir()

        with patch("sleap_rtc.worker.file_manager.sio") as mock_sio:
            mock_sio.load_file.side_effect = Exception("Corrupted file")

            result = file_manager.write_slp_with_new_paths(
                slp_path=str(slp_file),
                output_dir=str(output_dir),
                filename_map={"/old/video.mp4": "/new/video.mp4"},
            )

            assert "error" in result
            assert "failed to load" in result["error"].lower()

    def test_write_slp_partial_filename_map(self, file_manager, temp_mount):
        """Test writing when filename_map only covers some videos."""
        # Create SLP file
        slp_file = temp_mount / "labels.slp"
        slp_file.write_text("slp data")

        # Create output directory
        output_dir = temp_mount / "output"
        output_dir.mkdir()

        # Mock with 3 videos but only map 2
        mock_labels = self._create_mock_labels(
            [
                self._create_mock_video("/old/path/video1.mp4"),
                self._create_mock_video("/old/path/video2.mp4"),
                self._create_mock_video("/other/video3.mp4"),
            ]
        )

        filename_map = {
            "/old/path/video1.mp4": "/new/path/video1.mp4",
            "/old/path/video2.mp4": "/new/path/video2.mp4",
        }

        with patch("sleap_rtc.worker.file_manager.sio") as mock_sio:
            mock_sio.load_file.return_value = mock_labels

            result = file_manager.write_slp_with_new_paths(
                slp_path=str(slp_file),
                output_dir=str(output_dir),
                filename_map=filename_map,
            )

            # Should only count 2 as updated (not the 3rd)
            assert result["videos_updated"] == 2
            assert "error" not in result

    def test_write_slp_save_failure(self, file_manager, temp_mount):
        """Test error handling when saving SLP fails."""
        # Create SLP file
        slp_file = temp_mount / "labels.slp"
        slp_file.write_text("slp data")

        # Create output directory
        output_dir = temp_mount / "output"
        output_dir.mkdir()

        mock_labels = self._create_mock_labels(
            [
                self._create_mock_video("/old/video.mp4"),
            ]
        )
        mock_labels.save.side_effect = PermissionError("Permission denied")

        with patch("sleap_rtc.worker.file_manager.sio") as mock_sio:
            mock_sio.load_file.return_value = mock_labels

            result = file_manager.write_slp_with_new_paths(
                slp_path=str(slp_file),
                output_dir=str(output_dir),
                filename_map={"/old/video.mp4": "/new/video.mp4"},
            )

            assert "error" in result
            assert "failed to save" in result["error"].lower()
