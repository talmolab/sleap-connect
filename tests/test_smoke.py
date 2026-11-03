"""Smoke tests for the sleap-connect package.

These tests verify that the installed package is structurally sound — all
subpackages importable, the CLI entry point reachable, and required data
files present. They are intentionally lightweight and fast (<5 s total).

Modeled after the CLI smoke test pattern used in SLEAP and sleap-nn
(tests/test_cli.py in each repo), using Click's CliRunner for in-process
invocation and plain imports for module-level checks.

The primary failure mode these tests guard against is a packaging bug
where subpackages or data files are missing from the published wheel
(as happened in v0.0.1, where sleap_connect.worker was not included).
"""

from pathlib import Path

import pytest
from click.testing import CliRunner

from sleap_connect.cli import cli


# ── Subpackage imports ────────────────────────────────────────────────────────


class TestSubpackageImports:
    """Each sleap_connect subpackage must be importable without error.

    If any of these fail, the wheel is missing __init__.py files or the
    package was not correctly included in the distribution.
    """

    def test_import_worker(self):
        """sleap_connect.worker must be importable."""
        import sleap_connect.worker  # noqa: F401

    def test_import_worker_class(self):
        """RTCWorkerClient must be importable from sleap_connect.worker."""
        from sleap_connect.worker.worker_class import RTCWorkerClient  # noqa: F401

    def test_import_client(self):
        """sleap_connect.client must be importable."""
        import sleap_connect.client  # noqa: F401

    def test_import_client_class(self):
        """RTCClient must be importable from sleap_connect.client."""
        from sleap_connect.client.client_class import RTCClient  # noqa: F401

    def test_import_auth(self):
        """sleap_connect.auth must be importable."""
        import sleap_connect.auth  # noqa: F401

    def test_import_jobs(self):
        """sleap_connect.jobs must be importable."""
        import sleap_connect.jobs  # noqa: F401

    def test_import_gui(self):
        """sleap_connect.gui must be importable."""
        import sleap_connect.gui  # noqa: F401

    def test_import_tui(self):
        """sleap_connect.tui must be importable."""
        import sleap_connect.tui  # noqa: F401


# ── CLI entry point ───────────────────────────────────────────────────────────


class TestCLIEntryPoint:
    """The CLI entry point must be reachable and respond to --help."""

    def test_main_help(self):
        """sleap-connect --help must exit 0 and list core commands."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        for command in ("serve", "pair", "status", "legacy"):
            assert command in result.output
        legacy = runner.invoke(cli, ["legacy", "--help"])
        assert legacy.exit_code == 0
        for command in ("worker", "train", "track", "login"):
            assert command in legacy.output

    def test_worker_help(self):
        """sleap-connect worker --help must exit 0."""
        runner = CliRunner()
        result = runner.invoke(cli, ["worker", "--help"])
        assert result.exit_code == 0

    def test_train_help(self):
        """sleap-connect train --help must exit 0."""
        runner = CliRunner()
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0

    def test_track_help(self):
        """sleap-connect track --help must exit 0."""
        runner = CliRunner()
        result = runner.invoke(cli, ["track", "--help"])
        assert result.exit_code == 0

    def test_key_help(self):
        """sleap-connect key --help must exit 0 and list subcommands."""
        runner = CliRunner()
        result = runner.invoke(cli, ["key", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "create" in result.output
        assert "revoke" in result.output
        assert "show" in result.output
        assert "use" in result.output

    def test_key_list_help(self):
        """sleap-connect key list --help must exit 0."""
        runner = CliRunner()
        result = runner.invoke(cli, ["key", "list", "--help"])
        assert result.exit_code == 0

    def test_key_create_help(self):
        """sleap-connect key create --help must exit 0."""
        runner = CliRunner()
        result = runner.invoke(cli, ["key", "create", "--help"])
        assert result.exit_code == 0

    def test_key_revoke_help(self):
        """sleap-connect key revoke --help must exit 0."""
        runner = CliRunner()
        result = runner.invoke(cli, ["key", "revoke", "--help"])
        assert result.exit_code == 0

    def test_key_use_help(self):
        """sleap-connect key use --help must exit 0."""
        runner = CliRunner()
        result = runner.invoke(cli, ["key", "use", "--help"])
        assert result.exit_code == 0

    def test_worker_account_key_option(self):
        """sleap-connect worker --help must show --account-key option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["worker", "--help"])
        assert result.exit_code == 0
        assert "account-key" in result.output

    def test_room_set_default_help(self):
        """sleap-connect room set-default --help must exit 0."""
        runner = CliRunner()
        result = runner.invoke(cli, ["room", "set-default", "--help"])
        assert result.exit_code == 0


# ── Required data files ───────────────────────────────────────────────────────


class TestRequiredDataFiles:
    """Non-Python files that must be present in the installed package.

    The filesystem viewer server (fs_viewer_server.py) serves these HTML
    files at runtime using Path(__file__).parent / "static" / "...".
    If they are missing from the wheel the server returns HTTP 500.
    """

    def test_fs_viewer_html_present(self):
        """fs_viewer.html must be bundled with the package."""
        import sleap_connect.client.fs_viewer_server as mod

        html_path = Path(mod.__file__).parent / "static" / "fs_viewer.html"
        assert html_path.exists(), (
            f"fs_viewer.html missing from installed package at {html_path}. "
            "Check the hatchling exclude config in pyproject.toml."
        )

    def test_fs_resolve_html_present(self):
        """fs_resolve.html must be bundled with the package."""
        import sleap_connect.client.fs_viewer_server as mod

        html_path = Path(mod.__file__).parent / "static" / "fs_resolve.html"
        assert html_path.exists(), (
            f"fs_resolve.html missing from installed package at {html_path}. "
            "Check the hatchling exclude config in pyproject.toml."
        )


# ── Dashboard smoke tests ─────────────────────────────────────────────────────


class TestDashboardJobSubmission:
    """Dashboard job submission UI elements must be present."""

    @pytest.fixture
    def app_js(self):
        return (Path(__file__).parent.parent / "dashboard" / "app.js").read_text()

    def test_submit_job_button_in_render_room_card(self, app_js):
        """renderRoomCard must include a Submit Job button calling openSubmitJobModal."""
        assert "openSubmitJobModal" in app_js
        assert "Submit Job" in app_js

    def test_open_submit_job_modal_method_defined(self, app_js):
        """openSubmitJobModal method must be defined on the app class."""
        assert "openSubmitJobModal(" in app_js

    @pytest.fixture
    def index_html(self):
        return (Path(__file__).parent.parent / "dashboard" / "index.html").read_text()

    def test_modal_wrapper_present(self, index_html):
        """submit-job-modal wrapper must exist."""
        assert 'id="submit-job-modal"' in index_html

    def test_all_four_views_present(self, index_html):
        """All four step/status views must exist inside the modal."""
        assert 'id="sj-step1"' in index_html
        assert 'id="sj-step2"' in index_html
        assert 'id="sj-step3"' in index_html
        assert 'id="sj-status"' in index_html

    def test_step_progress_indicator_present(self, index_html):
        """A step progress indicator showing 3 steps must exist."""
        assert 'sj-step-indicator' in index_html

    def test_worker_list_container_present(self, index_html):
        """Worker selection list container must exist in step 1."""
        assert 'id="sj-worker-list"' in index_html

    def test_config_dropzone_present(self, index_html):
        """Config YAML drop zone must exist in step 2."""
        assert 'id="sj-config-dropzone"' in index_html

    def test_file_browser_columns_present(self, index_html):
        """File browser columns container must exist in step 3."""
        assert 'id="sj-file-columns"' in index_html

    def test_status_label_present(self, index_html):
        """Status label element must exist in the status view."""
        assert 'id="sj-status-label"' in index_html

    def test_wandb_link_present(self, index_html):
        """WandB URL link must exist in the status view (hidden by default)."""
        assert 'id="sj-wandb-link"' in index_html

    def test_sj_render_worker_list_method_defined(self, app_js):
        """_sjRenderWorkerList method must be defined."""
        assert "_sjRenderWorkerList(" in app_js

    def test_sj_select_worker_method_defined(self, app_js):
        """sjSelectWorker method must be defined."""
        assert "sjSelectWorker(" in app_js

    def test_sj_worker_row_available_clickable(self, app_js):
        """Available workers must call sjSelectWorker on click."""
        assert "sjSelectWorker" in app_js
        assert "sj-worker-row" in app_js

    def test_sj_worker_specs_rendered(self, app_js):
        """Worker GPU specs (gpu_model, gpu_memory_mb, cuda_version, sleap_nn_version) must be read from properties."""
        assert "gpu_model" in app_js
        assert "gpu_memory_mb" in app_js
        assert "cuda_version" in app_js
        assert "sleap_nn_version" in app_js

    def test_sj_status_dot_classes(self, app_js):
        """Status dot must use available/busy/maintenance CSS classes."""
        assert "sj-status-dot" in app_js
        assert "available" in app_js
        assert "busy" in app_js

    # ── Task 5: YAML config upload ────────────────────────────────────────────

    def test_parse_training_config_method_defined(self, app_js):
        """parseTrainingConfig method must be defined."""
        assert "parseTrainingConfig(" in app_js

    def test_parse_training_config_reads_key_fields(self, app_js):
        """parseTrainingConfig must read batch_size, learning_rate, max_epochs, run_name."""
        assert "batch_size" in app_js
        assert "learning_rate" in app_js
        assert "max_epochs" in app_js
        assert "run_name" in app_js
        # sleap-nn uses trainer_config (not trainer) and optimizer.lr (not learning_rate)
        assert "trainer_config" in app_js
        assert "optimizer" in app_js

    def test_dropzone_handler_wired(self, app_js):
        """Drop zone drag-and-drop handler must be implemented (_sjInitDropzone or inline)."""
        assert "sj-config-dropzone" in app_js
        assert "dragover" in app_js
        assert "drop" in app_js

    def test_config_content_stored(self, app_js):
        """Parsed YAML content must be stored as _sjConfigContent."""
        assert "_sjConfigContent" in app_js

    # ── Task 6: SSE relay connection ─────────────────────────────────────────

    def test_sse_connect_method_defined(self, app_js):
        """sseConnect method must be defined for relay SSE streams."""
        assert "sseConnect(" in app_js

    def test_worker_sse_opened_for_fs_browsing(self, app_js):
        """Worker SSE channel must be opened for filesystem browsing."""
        assert "_sjWorkerSSE" in app_js

    def test_job_sse_opened_for_status(self, app_js):
        """Job SSE channel must be opened for status updates."""
        assert "_sjJobSSE" in app_js

    def test_api_worker_message_defined(self, app_js):
        """apiWorkerMessage method must be defined for relay messaging."""
        assert "apiWorkerMessage(" in app_js

    # ── Task 7: filesystem browser ────────────────────────────────────────────

    def test_fs_list_res_handler_defined(self, app_js):
        """fs_list_res SSE handler must be defined."""
        assert "_sjHandleFsListRes" in app_js

    def test_fs_list_method_defined(self, app_js):
        """apiFsList method must be defined for filesystem listing."""
        assert "apiFsList(" in app_js

    def test_slp_file_selection_stored(self, app_js):
        """Selecting a .slp file must store path as _sjLabelsPath."""
        assert "_sjLabelsPath" in app_js
        assert ".slp" in app_js

    def test_step3_opens_worker_sse(self, app_js):
        """Entering step 3 must open worker SSE for filesystem browsing."""
        assert "sseConnect" in app_js
        assert "_sjWorkerSSE" in app_js

    # ── Task 8: job submission and status view ────────────────────────────────

    def test_submit_job_method_defined(self, app_js):
        """submitJob method must be defined."""
        assert "submitJob(" in app_js

    def test_submit_job_uses_api_endpoint(self, app_js):
        """submitJob must POST to the jobs/submit API endpoint."""
        assert "jobs/submit" in app_js

    def test_job_status_handler_defined(self, app_js):
        """job_status SSE handler must be present."""
        assert "job_status" in app_js
        assert "_sjHandleJobStatus" in app_js

    def test_job_progress_handler_defined(self, app_js):
        """job_progress SSE handler must be present."""
        assert "job_progress" in app_js
        assert "_sjHandleJobProgress" in app_js

    def test_job_status_complete_handled(self, app_js):
        """Complete status must update the status label."""
        assert "'complete'" in app_js or '"complete"' in app_js

    def test_job_status_failed_handled(self, app_js):
        """Failed status must update the status label."""
        assert "'failed'" in app_js or '"failed"' in app_js

    def test_wandb_link_shown_on_progress(self, app_js):
        """wandb_url in JOB_PROGRESS must reveal the WandB link element."""
        assert "wandb_url" in app_js
        assert "sj-wandb-link" in app_js

    # ── Task 9: smoke test and cleanup ────────────────────────────────────────

    def test_no_workers_shows_toast_warning(self, app_js):
        """openSubmitJobModal must show a toast warning and bail when no workers are connected."""
        assert "No workers" in app_js
        assert "start a worker" in app_js.lower()

    def test_js_yaml_cdn_included(self, index_html):
        """js-yaml CDN script must be included in index.html for YAML parsing."""
        assert "js-yaml" in index_html

    def test_nav_tabs_present(self, index_html):
        """Core data-tab nav items (rooms, tokens, quickstart, about) must be present."""
        assert 'data-tab="rooms"' in index_html
        assert 'data-tab="tokens"' in index_html
        assert 'data-tab="quickstart"' in index_html
        assert 'data-tab="about"' in index_html

    # ── Room card UX redesign + Workers modal ────────────────────────────────

    def test_room_action_bar_present(self, app_js):
        """Room cards must include a room-action-bar with action buttons."""
        assert "room-action-bar" in app_js

    def test_btn_submit_job_class_present(self, app_js):
        """Submit Job button must use btn-submit-job class for purple accent styling."""
        assert "btn-submit-job" in app_js

    def test_view_workers_button_present(self, app_js):
        """Room cards must include a View Workers button."""
        assert "View Workers" in app_js

    def test_open_workers_modal_method_defined(self, app_js):
        """openWorkersModal method must be defined on the app class."""
        assert "openWorkersModal(" in app_js

    def test_render_workers_modal_list_method_defined(self, app_js):
        """renderWorkersModalList method must be defined."""
        assert "renderWorkersModalList(" in app_js

    def test_set_workers_filter_method_defined(self, app_js):
        """setWorkersFilter method must be defined for filter chips."""
        assert "setWorkersFilter(" in app_js

    def test_filter_workers_search_method_defined(self, app_js):
        """filterWorkersSearch method must be defined for search."""
        assert "filterWorkersSearch(" in app_js

    def test_workers_modal_present(self, index_html):
        """workers-modal wrapper must exist in index.html."""
        assert 'id="workers-modal"' in index_html

    def test_workers_modal_filter_chips_present(self, index_html):
        """Workers modal must include filter chips (All, Idle, Busy)."""
        assert "wm-filter-chip" in index_html
        assert 'data-filter="all"' in index_html
        assert 'data-filter="idle"' in index_html
        assert 'data-filter="busy"' in index_html

    def test_workers_modal_search_present(self, index_html):
        """Workers modal must include a search input."""
        assert 'id="wm-search"' in index_html

    def test_workers_modal_worker_list_container(self, index_html):
        """Workers modal must include a worker list container."""
        assert 'id="wm-worker-list"' in index_html
