"""Tests for InferenceProgressDialog widget."""

import pytest

from qtpy.QtWidgets import QApplication, QDialogButtonBox


@pytest.fixture(scope="session")
def qapp():
    """Create a QApplication instance for Qt widget tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def dialog(qapp):
    from sleap_connect.gui.widgets import InferenceProgressDialog

    dlg = InferenceProgressDialog()
    yield dlg
    dlg.close()


class TestInferenceProgressDialogInit:
    def test_title(self, dialog):
        assert dialog.windowTitle() == "Running Inference"

    def test_initial_status_label(self, dialog):
        assert dialog._status_label.text() == "Running inference…"

    def test_initial_progress_zero(self, dialog):
        assert dialog._progress_bar.value() == 0

    def test_ok_disabled_initially(self, dialog):
        ok_btn = dialog._button_box.button(QDialogButtonBox.Ok)
        assert not ok_btn.isEnabled()

    def test_cancel_enabled_initially(self, dialog):
        cancel_btn = dialog._button_box.button(QDialogButtonBox.Cancel)
        assert cancel_btn.isEnabled()


class TestInferenceProgressDialogUpdate:
    def test_progress_bar_updates(self, dialog):
        dialog.update({"n_processed": 50, "n_total": 100, "rate": 10.0, "eta": 5})
        assert dialog._progress_bar.value() == 50

    def test_status_label_updates(self, dialog):
        dialog.update({"n_processed": 50, "n_total": 100, "rate": 10.0, "eta": 90})
        label = dialog._status_label.text()
        assert "50/100" in label
        assert "10.0" in label

    def test_log_appended(self, dialog):
        dialog.update({"n_processed": 10, "n_total": 200, "rate": 5.0, "eta": 0})
        assert "10/200" in dialog._log_text.toPlainText()

    def test_zero_total_no_crash(self, dialog):
        """Should handle n_total=0 without divide-by-zero."""
        dialog.update({"n_processed": 0, "n_total": 0, "rate": 0.0, "eta": 0})


class TestInferenceProgressDialogFinish:
    def test_progress_full(self, dialog):
        dialog.finish(500, 450, 50)
        assert dialog._progress_bar.value() == 100

    def test_status_label_complete(self, dialog):
        dialog.finish(500, 450, 50)
        assert "complete" in dialog._status_label.text().lower()

    def test_summary_in_status_label(self, dialog):
        """finish() should put the frame-count summary in the status label."""
        dialog.finish(500, 450, 50)
        label = dialog._status_label.text()
        assert "500" in label
        assert "450" in label
        assert "50" in label

    def test_ok_enabled_after_finish(self, dialog):
        dialog.finish(100, 80, 20)
        ok_btn = dialog._button_box.button(QDialogButtonBox.Ok)
        assert ok_btn.isEnabled()

    def test_cancel_disabled_after_finish(self, dialog):
        dialog.finish(100, 80, 20)
        cancel_btn = dialog._button_box.button(QDialogButtonBox.Cancel)
        assert not cancel_btn.isEnabled()

    def test_no_stats_complete_in_label(self, dialog):
        """finish(None, None, None) should show complete in label and Predictions saved in log."""
        dialog.finish(None, None, None)
        assert "complete" in dialog._status_label.text().lower()
        assert "Predictions saved." in dialog._log_text.toPlainText()

    def test_no_stats_still_enables_ok(self, dialog):
        dialog.finish(None, None, None)
        ok_btn = dialog._button_box.button(QDialogButtonBox.Ok)
        assert ok_btn.isEnabled()

    def test_only_frames_in_status_label(self, dialog):
        """finish(n_frames, None, None) should show frame count in label but no breakdown."""
        dialog.finish(35, None, None)
        label = dialog._status_label.text()
        assert "35" in label
        assert "Frames with predictions found" not in label

    def test_predictions_saved_always_in_log(self, dialog):
        """finish() always appends 'Predictions saved.' to the log."""
        dialog.finish(35, None, None)
        assert "Predictions saved." in dialog._log_text.toPlainText()


class TestInferenceProgressDialogAppendLog:
    def test_plain_text_appended(self, dialog):
        dialog.append_log("Starting inference at 2026-01-01")
        assert "Starting inference at 2026-01-01" in dialog._log_text.toPlainText()

    def test_cr_overwrite_keeps_last_segment(self, dialog):
        """Lines with \\r should show only the last segment (rich progress bar)."""
        dialog.append_log("\rPredicting 10%\rPredicting 100% 35/35 FPS: 47.8")
        log = dialog._log_text.toPlainText()
        assert "Predicting 100%" in log
        assert "Predicting 10%" not in log

    def test_empty_text_not_appended(self, dialog):
        before = dialog._log_text.toPlainText()
        dialog.append_log("")
        assert dialog._log_text.toPlainText() == before

    def test_whitespace_only_not_appended(self, dialog):
        before = dialog._log_text.toPlainText()
        dialog.append_log("   \r  \r  ")
        assert dialog._log_text.toPlainText() == before


class TestInferenceProgressDialogSetProgress:
    def test_progress_bar_updates(self, dialog):
        dialog.set_progress(35, 35, 47.8, 0)
        assert dialog._progress_bar.value() == 100

    def test_status_label_updates(self, dialog):
        dialog.set_progress(17, 35, 30.0, 1)
        assert "17/35" in dialog._status_label.text()

    def test_no_log_entry(self, dialog):
        """set_progress should NOT append to the log."""
        dialog.set_progress(35, 35, 47.8, 0)
        assert dialog._log_text.toPlainText() == ""


class TestInferenceProgressDialogShowError:
    def test_status_label_failed(self, dialog):
        dialog.show_error("something went wrong")
        assert "failed" in dialog._status_label.text().lower()

    def test_error_in_log(self, dialog):
        dialog.show_error("out of memory")
        assert "out of memory" in dialog._log_text.toPlainText()

    def test_ok_enabled_after_error(self, dialog):
        dialog.show_error("oops")
        ok_btn = dialog._button_box.button(QDialogButtonBox.Ok)
        assert ok_btn.isEnabled()

    def test_cancel_disabled_after_error(self, dialog):
        dialog.show_error("oops")
        cancel_btn = dialog._button_box.button(QDialogButtonBox.Cancel)
        assert not cancel_btn.isEnabled()


class TestFormatEta:
    def test_seconds(self):
        from sleap_connect.gui.widgets import InferenceProgressDialog

        assert InferenceProgressDialog._format_eta(45) == "45s"

    def test_minutes_seconds(self):
        from sleap_connect.gui.widgets import InferenceProgressDialog

        assert InferenceProgressDialog._format_eta(200) == "3m 20s"

    def test_hours_minutes(self):
        from sleap_connect.gui.widgets import InferenceProgressDialog

        assert InferenceProgressDialog._format_eta(3661) == "1h 01m"

    def test_zero(self):
        from sleap_connect.gui.widgets import InferenceProgressDialog

        assert InferenceProgressDialog._format_eta(0) == "0s"

    def test_invalid(self):
        from sleap_connect.gui.widgets import InferenceProgressDialog

        assert InferenceProgressDialog._format_eta(None) == "?"
