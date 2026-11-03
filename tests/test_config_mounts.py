"""Tests for mount configuration parsing and validation."""

import tempfile
import os
from pathlib import Path

import pytest

from sleap_connect.config import MountConfig, WorkerIOConfig


class TestMountConfig:
    """Tests for MountConfig dataclass."""

    def test_valid_mount(self):
        """Test creating a valid mount config."""
        mount = MountConfig(path="/mnt/data", label="Lab Data")
        assert mount.path == "/mnt/data"
        assert mount.label == "Lab Data"

    def test_empty_path_raises(self):
        """Test that empty path raises ValueError."""
        with pytest.raises(ValueError, match="path cannot be empty"):
            MountConfig(path="", label="Test")

    def test_empty_label_raises(self):
        """Test that empty label raises ValueError."""
        with pytest.raises(ValueError, match="label cannot be empty"):
            MountConfig(path="/mnt/data", label="")

    def test_validate_existing_directory(self, tmp_path):
        """Test validation of existing directory."""
        mount = MountConfig(path=str(tmp_path), label="Test")
        assert mount.validate() is True

    def test_validate_nonexistent_path(self):
        """Test validation of non-existent path."""
        mount = MountConfig(path="/nonexistent/path/12345", label="Test")
        assert mount.validate() is False

    def test_validate_file_path(self, tmp_path):
        """Test validation rejects file paths (not directories)."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        mount = MountConfig(path=str(test_file), label="Test")
        assert mount.validate() is False


class TestWorkerIOConfig:
    """Tests for WorkerIOConfig class."""

    def test_from_dict_empty(self):
        """Test creating config from empty dict."""
        config = WorkerIOConfig.from_dict({})
        assert config.mounts == []
        assert config.working_dir is None

    def test_from_dict_with_mounts(self):
        """Test creating config with mount entries."""
        data = {
            "mounts": [
                {"path": "/mnt/data", "label": "Data"},
                {"path": "/mnt/backup", "label": "Backup"},
            ]
        }
        config = WorkerIOConfig.from_dict(data)
        assert len(config.mounts) == 2
        assert config.mounts[0].path == "/mnt/data"
        assert config.mounts[0].label == "Data"
        assert config.mounts[1].path == "/mnt/backup"
        assert config.mounts[1].label == "Backup"

    def test_from_dict_with_working_dir(self):
        """Test creating config with working directory."""
        data = {"working_dir": "/mnt/work"}
        config = WorkerIOConfig.from_dict(data)
        assert config.working_dir == "/mnt/work"

    def test_from_dict_skips_invalid_mounts(self):
        """Test that invalid mount entries are skipped."""
        data = {
            "mounts": [
                {"path": "/mnt/data", "label": "Data"},  # Valid
                {"path": "/mnt/backup"},  # Missing label
                {"label": "Test"},  # Missing path
                {},  # Empty
            ]
        }
        config = WorkerIOConfig.from_dict(data)
        assert len(config.mounts) == 1
        assert config.mounts[0].path == "/mnt/data"

    def test_get_valid_mounts(self, tmp_path):
        """Test getting only valid mounts."""
        valid_dir = tmp_path / "valid"
        valid_dir.mkdir()

        config = WorkerIOConfig(
            mounts=[
                MountConfig(path=str(valid_dir), label="Valid"),
                MountConfig(path="/nonexistent/path/12345", label="Invalid"),
            ]
        )

        valid_mounts = config.get_valid_mounts()
        assert len(valid_mounts) == 1
        assert valid_mounts[0].path == str(valid_dir)
