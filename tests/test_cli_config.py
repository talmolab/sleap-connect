"""Tests for 'sleap-connect config' path-mapping subcommands."""

from unittest import mock

import pytest
from click.testing import CliRunner

from sleap_connect.cli import cli
from sleap_connect.config import Config, PathMapping


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """Config instance whose home config is redirected to a temp directory."""
    config = Config()
    monkeypatch.setattr(config, "_home_config_path", lambda: tmp_path / "config.toml")
    return config


@pytest.fixture
def patched_config(cfg):
    """Patch get_config() in the config module to return our temp-dir Config."""
    with mock.patch("sleap_connect.config.get_config", return_value=cfg):
        yield cfg


class TestAddPathMapping:
    def test_adds_mapping(self, runner, patched_config):
        result = runner.invoke(
            cli,
            ["config", "add-path-mapping",
             "--local", "/Users/alice/data",
             "--worker", "/root/vast/data"],
        )
        assert result.exit_code == 0
        assert "Saved path mapping" in result.output
        mappings = patched_config.get_path_mappings()
        assert len(mappings) == 1
        assert mappings[0].local == "/Users/alice/data"
        assert mappings[0].worker == "/root/vast/data"

    def test_confirmation_message(self, runner, patched_config):
        result = runner.invoke(
            cli,
            ["config", "add-path-mapping",
             "--local", "/local", "--worker", "/worker"],
        )
        assert "/local → /worker" in result.output

    def test_duplicate_not_added_twice(self, runner, patched_config):
        args = ["config", "add-path-mapping", "--local", "/l", "--worker", "/w"]
        runner.invoke(cli, args)
        runner.invoke(cli, args)
        assert len(patched_config.get_path_mappings()) == 1

    def test_requires_local_and_worker(self, runner, patched_config):
        result = runner.invoke(cli, ["config", "add-path-mapping", "--local", "/l"])
        assert result.exit_code != 0


class TestRemovePathMapping:
    def test_removes_existing_mapping(self, runner, patched_config):
        patched_config.save_path_mapping("/local", "/worker")
        result = runner.invoke(
            cli,
            ["config", "remove-path-mapping",
             "--local", "/local", "--worker", "/worker"],
        )
        assert result.exit_code == 0
        assert "Removed path mapping" in result.output
        assert patched_config.get_path_mappings() == []

    def test_warns_when_not_found(self, runner, patched_config):
        result = runner.invoke(
            cli,
            ["config", "remove-path-mapping",
             "--local", "/nonexistent", "--worker", "/nowhere"],
        )
        assert result.exit_code == 0
        assert "No matching mapping found" in result.output

    def test_preserves_other_mappings(self, runner, patched_config):
        patched_config.save_path_mapping("/a", "/wa")
        patched_config.save_path_mapping("/b", "/wb")
        runner.invoke(
            cli,
            ["config", "remove-path-mapping", "--local", "/a", "--worker", "/wa"],
        )
        remaining = patched_config.get_path_mappings()
        assert len(remaining) == 1
        assert remaining[0].local == "/b"


class TestListPathMappings:
    def test_no_mappings(self, runner, patched_config):
        result = runner.invoke(cli, ["config", "list-path-mappings"])
        assert result.exit_code == 0
        assert "No path mappings" in result.output

    def test_lists_mappings(self, runner, patched_config):
        patched_config.save_path_mapping("/local/a", "/worker/a")
        patched_config.save_path_mapping("/local/b", "/worker/b")
        result = runner.invoke(cli, ["config", "list-path-mappings"])
        assert result.exit_code == 0
        assert "/local/a → /worker/a" in result.output
        assert "/local/b → /worker/b" in result.output
