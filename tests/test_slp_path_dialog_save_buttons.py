"""Tests for SlpPathDialog save-to-folder button visibility and behaviour."""

from unittest.mock import MagicMock, patch

import pytest
from qtpy.QtWidgets import QApplication

from sleap_connect.gui.widgets import SlpPathDialog


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def _make_dialog(
    local_path="/Users/alice/labels.slp",
    error_message="File not found",
    save_fn=None,
    convert_fn=None,
    send_fn=None,
    qapp=None,
):
    return SlpPathDialog(
        local_path=local_path,
        error_message=error_message,
        save_fn=save_fn,
        convert_fn=convert_fn,
        send_fn=send_fn,
    )


class TestSaveButtonVisibility:
    def test_no_buttons_when_no_callables(self, qapp):
        dlg = _make_dialog()
        assert not hasattr(dlg, "_save_slp_btn")
        assert not hasattr(dlg, "_save_pkg_slp_btn")

    def test_slp_button_shown_when_save_fn_provided(self, qapp):
        dlg = _make_dialog(save_fn=lambda p: None)
        assert hasattr(dlg, "_save_slp_btn")
        assert not dlg._save_slp_btn.isHidden()

    def test_pkg_slp_button_shown_when_convert_fn_provided(self, qapp):
        dlg = _make_dialog(convert_fn=lambda p: None)
        assert hasattr(dlg, "_save_pkg_slp_btn")
        assert not dlg._save_pkg_slp_btn.isHidden()

    def test_both_buttons_shown_when_both_provided(self, qapp):
        dlg = _make_dialog(save_fn=lambda p: None, convert_fn=lambda p: None)
        assert hasattr(dlg, "_save_slp_btn")
        assert hasattr(dlg, "_save_pkg_slp_btn")

    def test_save_status_label_hidden_initially(self, qapp):
        dlg = _make_dialog(save_fn=lambda p: None)
        assert dlg._save_status_label is not None
        assert dlg._save_status_label.isHidden()

    def test_no_save_status_label_without_buttons(self, qapp):
        dlg = _make_dialog()
        assert dlg._save_status_label is None


class TestSaveSLPButton:
    def test_calls_save_fn_with_correct_path(self, qapp, tmp_path):
        saved_paths = []
        save_fn = lambda p: saved_paths.append(p)

        dlg = _make_dialog(
            local_path="/Users/alice/labels.slp",
            save_fn=save_fn,
        )

        with patch(
            "sleap_connect.gui.widgets.QFileDialog.getExistingDirectory",
            return_value=str(tmp_path),
        ):
            dlg._on_save_slp_clicked()

        assert len(saved_paths) == 1
        assert saved_paths[0] == str(tmp_path / "labels.slp")

    def test_status_label_shows_success(self, qapp, tmp_path):
        dlg = _make_dialog(save_fn=lambda p: None)

        with patch(
            "sleap_connect.gui.widgets.QFileDialog.getExistingDirectory",
            return_value=str(tmp_path),
        ):
            dlg._on_save_slp_clicked()

        assert not dlg._save_status_label.isHidden()
        assert "Now use Browse worker filesystem" in dlg._save_status_label.text()

    def test_no_save_when_dialog_cancelled(self, qapp):
        saved_paths = []
        dlg = _make_dialog(save_fn=lambda p: saved_paths.append(p))

        with patch(
            "sleap_connect.gui.widgets.QFileDialog.getExistingDirectory",
            return_value="",
        ):
            dlg._on_save_slp_clicked()

        assert saved_paths == []

    def test_error_shown_on_exception(self, qapp, tmp_path):
        def failing_fn(p):
            raise RuntimeError("disk full")

        dlg = _make_dialog(save_fn=failing_fn)

        with patch(
            "sleap_connect.gui.widgets.QFileDialog.getExistingDirectory",
            return_value=str(tmp_path),
        ):
            dlg._on_save_slp_clicked()

        assert not dlg._save_status_label.isHidden()
        assert "disk full" in dlg._save_status_label.text()

    def test_button_re_enabled_after_error(self, qapp, tmp_path):
        dlg = _make_dialog(save_fn=lambda p: (_ for _ in ()).throw(RuntimeError("err")))

        with patch(
            "sleap_connect.gui.widgets.QFileDialog.getExistingDirectory",
            return_value=str(tmp_path),
        ):
            dlg._on_save_slp_clicked()

        assert dlg._save_slp_btn.isEnabled()


class TestSavePkgSLPButton:
    def test_calls_convert_fn_with_correct_path(self, qapp, tmp_path):
        converted_paths = []
        convert_fn = lambda p: converted_paths.append(p)

        dlg = _make_dialog(
            local_path="/Users/alice/labels.slp",
            convert_fn=convert_fn,
        )

        with patch(
            "sleap_connect.gui.widgets.QFileDialog.getExistingDirectory",
            return_value=str(tmp_path),
        ):
            dlg._on_save_pkg_slp_clicked()

        assert len(converted_paths) == 1
        assert converted_paths[0] == str(tmp_path / "labels.pkg.slp")

    def test_status_label_shows_success(self, qapp, tmp_path):
        dlg = _make_dialog(convert_fn=lambda p: None)

        with patch(
            "sleap_connect.gui.widgets.QFileDialog.getExistingDirectory",
            return_value=str(tmp_path),
        ):
            dlg._on_save_pkg_slp_clicked()

        assert not dlg._save_status_label.isHidden()
        assert "Now use Browse worker filesystem" in dlg._save_status_label.text()

    def test_no_save_when_dialog_cancelled(self, qapp):
        converted_paths = []
        dlg = _make_dialog(convert_fn=lambda p: converted_paths.append(p))

        with patch(
            "sleap_connect.gui.widgets.QFileDialog.getExistingDirectory",
            return_value="",
        ):
            dlg._on_save_pkg_slp_clicked()

        assert converted_paths == []

    def test_error_shown_on_exception(self, qapp, tmp_path):
        def failing_fn(p):
            raise RuntimeError("conversion failed")

        dlg = _make_dialog(convert_fn=failing_fn)

        with patch(
            "sleap_connect.gui.widgets.QFileDialog.getExistingDirectory",
            return_value=str(tmp_path),
        ):
            dlg._on_save_pkg_slp_clicked()

        assert "conversion failed" in dlg._save_status_label.text()

    def test_button_re_enabled_after_error(self, qapp, tmp_path):
        dlg = _make_dialog(convert_fn=lambda p: (_ for _ in ()).throw(RuntimeError("err")))

        with patch(
            "sleap_connect.gui.widgets.QFileDialog.getExistingDirectory",
            return_value=str(tmp_path),
        ):
            dlg._on_save_pkg_slp_clicked()

        assert dlg._save_pkg_slp_btn.isEnabled()
