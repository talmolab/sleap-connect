"""Tests for path mapping persistence in Config."""

import pytest

from sleap_connect.config import Config, PathMapping


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """Config instance whose home config is redirected to a temp directory."""
    config = Config()
    monkeypatch.setattr(config, "_home_config_path", lambda: tmp_path / "config.toml")
    return config


class TestPathMappingDataclass:
    def test_fields(self):
        m = PathMapping(local="/Users/alice/data", worker="/root/data")
        assert m.local == "/Users/alice/data"
        assert m.worker == "/root/data"


class TestGetPathMappings:
    def test_empty_when_no_file(self, cfg):
        assert cfg.get_path_mappings() == []

    def test_returns_mappings_from_file(self, cfg):
        cfg.save_path_mapping("/Users/amickl/repos/data", "/root/vast/amick/data")
        cfg.save_path_mapping("/Users/amickl/videos", "/root/vast/amick/videos")
        mappings = cfg.get_path_mappings()
        assert len(mappings) == 2
        assert mappings[0].local == "/Users/amickl/repos/data"
        assert mappings[0].worker == "/root/vast/amick/data"
        assert mappings[1].local == "/Users/amickl/videos"
        assert mappings[1].worker == "/root/vast/amick/videos"

    def test_returns_path_mapping_objects(self, cfg):
        cfg.save_path_mapping("/local", "/worker")
        mappings = cfg.get_path_mappings()
        assert all(isinstance(m, PathMapping) for m in mappings)


class TestSavePathMapping:
    def test_save_new_mapping(self, cfg, tmp_path):
        cfg.save_path_mapping("/Users/amickl/repos/data", "/root/vast/amick/data")
        config_file = tmp_path / "config.toml"
        assert config_file.exists()
        mappings = cfg.get_path_mappings()
        assert len(mappings) == 1
        assert mappings[0].local == "/Users/amickl/repos/data"
        assert mappings[0].worker == "/root/vast/amick/data"

    def test_duplicate_not_written_twice(self, cfg):
        cfg.save_path_mapping("/Users/amickl/repos/data", "/root/vast/amick/data")
        cfg.save_path_mapping("/Users/amickl/repos/data", "/root/vast/amick/data")
        assert len(cfg.get_path_mappings()) == 1

    def test_preserves_other_mappings(self, cfg):
        cfg.save_path_mapping("/local/a", "/worker/a")
        cfg.save_path_mapping("/local/b", "/worker/b")
        cfg.save_path_mapping("/local/a", "/worker/a")  # duplicate — ignored
        assert len(cfg.get_path_mappings()) == 2

    def test_creates_parent_dir(self, cfg, tmp_path):
        # tmp_path exists; the file should be created inside it
        cfg.save_path_mapping("/local", "/worker")
        assert (tmp_path / "config.toml").exists()


class TestRemovePathMapping:
    def test_remove_existing_mapping(self, cfg):
        cfg.save_path_mapping("/Users/amickl/repos/data", "/root/vast/amick/data")
        cfg.remove_path_mapping("/Users/amickl/repos/data", "/root/vast/amick/data")
        assert cfg.get_path_mappings() == []

    def test_preserves_other_mappings(self, cfg):
        cfg.save_path_mapping("/local/a", "/worker/a")
        cfg.save_path_mapping("/local/b", "/worker/b")
        cfg.remove_path_mapping("/local/a", "/worker/a")
        remaining = cfg.get_path_mappings()
        assert len(remaining) == 1
        assert remaining[0].local == "/local/b"

    def test_remove_nonexistent_leaves_file_unchanged(self, cfg):
        cfg.save_path_mapping("/local/a", "/worker/a")
        cfg.remove_path_mapping("/doesnt/exist", "/nowhere")
        assert len(cfg.get_path_mappings()) == 1


class TestTranslatePath:
    def test_no_mappings_returns_none(self, cfg):
        assert cfg.translate_path("/Users/amickl/repos/data/labels.slp") is None

    def test_matching_prefix_translates(self, cfg):
        cfg.save_path_mapping("/Users/amickl/repos/data", "/root/vast/amick/data")
        result = cfg.translate_path("/Users/amickl/repos/data/labels.slp")
        assert result == "/root/vast/amick/data/labels.slp"

    def test_no_matching_prefix_returns_none(self, cfg):
        cfg.save_path_mapping("/Users/other/data", "/root/other/data")
        assert cfg.translate_path("/Users/amickl/repos/data/labels.slp") is None

    def test_longest_prefix_wins(self, cfg):
        cfg.save_path_mapping("/Users/amickl", "/root")
        cfg.save_path_mapping("/Users/amickl/repos/data", "/root/vast/amick/data")
        result = cfg.translate_path("/Users/amickl/repos/data/labels.slp")
        assert result == "/root/vast/amick/data/labels.slp"

    def test_nested_path_preserved(self, cfg):
        cfg.save_path_mapping("/Users/amickl/data", "/root/data")
        result = cfg.translate_path("/Users/amickl/data/videos/cam1.mp4")
        assert result == "/root/data/videos/cam1.mp4"

    def test_prefix_must_be_proper(self, cfg):
        # /Users/amickl should NOT match /Users/amicklX/...
        cfg.save_path_mapping("/Users/amickl", "/root")
        assert cfg.translate_path("/Users/amicklX/file.slp") is None
