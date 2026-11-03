"""Tests for SlpPathDialog auto-fill and save-mapping prompt (Phase 4)."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from qtpy.QtWidgets import QApplication, QDialog

from sleap_connect.config import Config, PathMapping
from sleap_connect.gui.widgets import SlpPathDialog


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _cfg_with_mapping(tmp_path, monkeypatch, local, worker):
    """Return a Config with one saved mapping, home dir redirected to tmp_path."""
    cfg = Config()
    monkeypatch.setattr(cfg, "_home_config_path", lambda: tmp_path / "config.toml")
    cfg.save_path_mapping(local, worker)
    return cfg


def _make_dialog(local_path, config, qapp):
    with patch("sleap_connect.config.get_config", return_value=config):
        dlg = SlpPathDialog(
            local_path=local_path,
            error_message="not found",
        )
    return dlg


# ===========================================================================
# Auto-fill tests (task 18)
# ===========================================================================


class TestAutoFill:
    def test_matching_prefix_fills_worker_path(self, qapp, tmp_path, monkeypatch):
        cfg = _cfg_with_mapping(
            tmp_path, monkeypatch,
            local="/Users/amickl/repos/data",
            worker="/root/vast/amick/data",
        )
        dlg = _make_dialog("/Users/amickl/repos/data/labels.slp", cfg, qapp)
        assert dlg._path_edit.text() == "/root/vast/amick/data/labels.slp"

    def test_no_matching_prefix_leaves_field_empty(self, qapp, tmp_path, monkeypatch):
        cfg = _cfg_with_mapping(
            tmp_path, monkeypatch,
            local="/Users/other",
            worker="/root/other",
        )
        dlg = _make_dialog("/Users/amickl/repos/data/labels.slp", cfg, qapp)
        assert dlg._path_edit.text() == ""

    def test_longest_prefix_wins(self, qapp, tmp_path, monkeypatch):
        cfg = Config()
        monkeypatch.setattr(cfg, "_home_config_path", lambda: tmp_path / "config.toml")
        cfg.save_path_mapping("/Users/amickl", "/root")
        cfg.save_path_mapping("/Users/amickl/repos/data", "/root/vast/amick/data")

        dlg = _make_dialog("/Users/amickl/repos/data/labels.slp", cfg, qapp)
        assert dlg._path_edit.text() == "/root/vast/amick/data/labels.slp"

    def test_auto_fill_enables_continue_button(self, qapp, tmp_path, monkeypatch):
        cfg = _cfg_with_mapping(
            tmp_path, monkeypatch,
            local="/Users/amickl/data",
            worker="/root/data",
        )
        dlg = _make_dialog("/Users/amickl/data/labels.slp", cfg, qapp)
        assert dlg._ok_btn.isEnabled()

    def test_no_mappings_continues_button_disabled(self, qapp, tmp_path, monkeypatch):
        cfg = Config()
        monkeypatch.setattr(cfg, "_home_config_path", lambda: tmp_path / "config.toml")
        dlg = _make_dialog("/Users/amickl/data/labels.slp", cfg, qapp)
        assert not dlg._ok_btn.isEnabled()


# ===========================================================================
# Save-mapping prompt tests (task 19)
# ===========================================================================


class TestSaveMappingPrompt:
    def _make_dialog_with_cfg(self, local_path, cfg, qapp):
        with patch("sleap_connect.config.get_config", return_value=cfg):
            dlg = SlpPathDialog(
                local_path=local_path,
                error_message="not found",
            )
        return dlg, cfg

    def test_prompt_shown_when_dirs_differ(self, qapp, tmp_path, monkeypatch):
        cfg = Config()
        monkeypatch.setattr(cfg, "_home_config_path", lambda: tmp_path / "config.toml")
        dlg, cfg = self._make_dialog_with_cfg(
            "/Users/amickl/repos/data/labels.slp", cfg, qapp
        )
        dlg._path_edit.setText("/root/vast/amick/data/labels.slp")

        with patch("sleap_connect.config.get_config", return_value=cfg):
            with patch.object(QDialog, "exec", return_value=QDialog.DialogCode.Rejected):
                dlg._on_accept()

        # prompt was shown (exec was called) — no mapping saved since rejected
        assert cfg.get_path_mappings() == []

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix path assertions (Mac/Linux client → Linux worker)")
    def test_save_mapping_on_save(self, qapp, tmp_path, monkeypatch):
        cfg = Config()
        monkeypatch.setattr(cfg, "_home_config_path", lambda: tmp_path / "config.toml")
        dlg, cfg = self._make_dialog_with_cfg(
            "/Users/amickl/repos/data/labels.slp", cfg, qapp
        )
        dlg._path_edit.setText("/root/vast/amick/data/labels.slp")

        with patch("sleap_connect.config.get_config", return_value=cfg):
            with patch.object(QDialog, "exec", return_value=QDialog.DialogCode.Accepted):
                dlg._on_accept()

        mappings = cfg.get_path_mappings()
        assert len(mappings) == 1
        assert mappings[0].local == "/Users/amickl/repos/data"
        assert mappings[0].worker == "/root/vast/amick/data"

    def test_skip_does_not_save(self, qapp, tmp_path, monkeypatch):
        cfg = Config()
        monkeypatch.setattr(cfg, "_home_config_path", lambda: tmp_path / "config.toml")
        dlg, cfg = self._make_dialog_with_cfg(
            "/Users/amickl/repos/data/labels.slp", cfg, qapp
        )
        dlg._path_edit.setText("/root/vast/amick/data/labels.slp")

        with patch("sleap_connect.config.get_config", return_value=cfg):
            with patch.object(QDialog, "exec", return_value=QDialog.DialogCode.Rejected):
                dlg._on_accept()

        assert cfg.get_path_mappings() == []

    def test_prompt_not_shown_when_same_dir(self, qapp, tmp_path, monkeypatch):
        """No prompt when local and worker are in the same directory."""
        cfg = Config()
        monkeypatch.setattr(cfg, "_home_config_path", lambda: tmp_path / "config.toml")
        dlg, cfg = self._make_dialog_with_cfg(
            "/shared/data/labels.slp", cfg, qapp
        )
        dlg._path_edit.setText("/shared/data/labels.slp")

        exec_calls = []
        original_exec = QDialog.exec

        def track_exec(self_):
            exec_calls.append(True)
            return QDialog.DialogCode.Rejected

        with patch("sleap_connect.config.get_config", return_value=cfg):
            with patch.object(QDialog, "exec", track_exec):
                dlg._on_accept()

        assert exec_calls == []

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix path assertions (Mac/Linux client → Linux worker)")
    def test_prompt_not_shown_when_mapping_already_exists(
        self, qapp, tmp_path, monkeypatch
    ):
        """No prompt if the exact local→worker pair is already saved."""
        cfg = _cfg_with_mapping(
            tmp_path, monkeypatch,
            local="/Users/amickl/repos/data",
            worker="/root/vast/amick/data",
        )
        dlg, cfg = self._make_dialog_with_cfg(
            "/Users/amickl/repos/data/labels.slp", cfg, qapp
        )
        dlg._path_edit.setText("/root/vast/amick/data/labels.slp")

        exec_calls = []

        def track_exec(self_):
            exec_calls.append(True)
            return QDialog.DialogCode.Rejected

        with patch("sleap_connect.config.get_config", return_value=cfg):
            with patch.object(QDialog, "exec", track_exec):
                dlg._on_accept()

        assert exec_calls == []
