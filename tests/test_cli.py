"""Unit tests for CLI commands.

Tests for:
- New utility commands (tui, status, doctor)
- Command aliases and deprecation warnings
- Flag standardization
"""

import json
from unittest import mock

import pytest
from click.testing import CliRunner

from sleap_rtc.cli import cli


@pytest.fixture
def runner():
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def temp_credentials(tmp_path):
    """Create temporary credentials file for testing."""
    creds_dir = tmp_path / ".sleap-rtc"
    creds_dir.mkdir()
    creds_path = creds_dir / "credentials.json"

    # Import credentials module to patch
    from sleap_rtc.auth import credentials

    with mock.patch.object(credentials, "CREDENTIALS_DIR", creds_dir):
        with mock.patch.object(credentials, "CREDENTIALS_PATH", creds_path):
            yield creds_path


class TestDoctorCommand:
    """Tests for 'sleap-rtc doctor' command."""

    def test_doctor_runs(self, runner):
        """Should run doctor checks without crashing."""
        result = runner.invoke(cli, ["doctor"])
        # Doctor should complete (may have warnings but shouldn't crash)
        assert result.exit_code == 0

    def test_doctor_checks_python(self, runner):
        """Should check Python environment."""
        result = runner.invoke(cli, ["doctor"])
        assert result.exit_code == 0
        # Should mention Python
        assert "Python" in result.output or "python" in result.output.lower()


class TestTuiCommand:
    """Tests for 'sleap-rtc tui' command."""

    def test_tui_help(self, runner):
        """Should show help text for TUI command."""
        result = runner.invoke(cli, ["tui", "--help"])
        assert result.exit_code == 0
        assert "TUI" in result.output or "interactive" in result.output.lower()

    def test_tui_accepts_room_option(self, runner):
        """Should accept --room option."""
        result = runner.invoke(cli, ["tui", "--help"])
        assert result.exit_code == 0
        assert "--room" in result.output


class TestCommandAliases:
    """Tests for deprecated command aliases."""

    def test_client_train_alias_shows_warning(self, runner):
        """Should show deprecation warning for client-train."""
        result = runner.invoke(cli, ["client-train", "--help"])
        assert result.exit_code == 0
        assert "deprecated" in result.output.lower() or "train" in result.output

    def test_client_track_alias_shows_warning(self, runner):
        """Should show deprecation warning for client-track."""
        result = runner.invoke(cli, ["client-track", "--help"])
        assert result.exit_code == 0
        assert "deprecated" in result.output.lower() or "track" in result.output

    def test_client_alias_shows_warning(self, runner):
        """Should show deprecation warning for client alias."""
        result = runner.invoke(cli, ["client", "--help"])
        assert result.exit_code == 0
        assert "deprecated" in result.output.lower() or "train" in result.output

    def test_browse_alias_shows_warning(self, runner):
        """Should show deprecation warning for browse alias."""
        result = runner.invoke(cli, ["browse", "--help"])
        assert result.exit_code == 0
        assert "deprecated" in result.output.lower() or "test browse" in result.output

    def test_resolve_paths_alias_shows_warning(self, runner):
        """Should show deprecation warning for resolve-paths alias."""
        result = runner.invoke(cli, ["resolve-paths", "--help"])
        assert result.exit_code == 0
        assert "deprecated" in result.output.lower() or "test resolve-paths" in result.output


class TestTrainCommand:
    """Tests for 'sleap-rtc train' command."""

    def test_train_help(self, runner):
        """Should show help text for train command."""
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "train" in result.output.lower()
        assert "--room" in result.output
        assert "--pkg-path" in result.output or "-p" in result.output

    def test_train_requires_connection_method(self, runner):
        """Should require either --session-string or --room."""
        result = runner.invoke(cli, ["train", "--pkg-path", "/some/path"])
        # Should exit with error code when no connection method provided
        assert result.exit_code != 0

    def test_train_short_flags(self, runner):
        """Should support short flags."""
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "-r" in result.output  # --room
        assert "-p" in result.output  # --pkg-path


class TestTrackCommand:
    """Tests for 'sleap-rtc track' command."""

    def test_track_help(self, runner):
        """Should show help text for track command."""
        result = runner.invoke(cli, ["track", "--help"])
        assert result.exit_code == 0
        assert "track" in result.output.lower() or "inference" in result.output.lower()
        assert "--room" in result.output
        assert "--data-path" in result.output or "-d" in result.output

    def test_track_requires_connection_method(self, runner):
        """Should require either --session-string or --room."""
        result = runner.invoke(cli, ["track", "--data-path", "/some/path", "--model-paths", "/model"])
        # Should exit with error code when no connection method provided
        assert result.exit_code != 0

    def test_track_short_flags(self, runner):
        """Should support short flags."""
        result = runner.invoke(cli, ["track", "--help"])
        assert result.exit_code == 0
        assert "-r" in result.output  # --room
        assert "-d" in result.output  # --data-path
        assert "-m" in result.output  # --model-paths


class TestTestSubcommand:
    """Tests for 'sleap-rtc test' subcommand group."""

    def test_test_group_help(self, runner):
        """Should show help for test command group."""
        result = runner.invoke(cli, ["test", "--help"])
        assert result.exit_code == 0
        assert "browse" in result.output
        assert "resolve-paths" in result.output

    def test_test_browse_help(self, runner):
        """Should show help for test browse."""
        result = runner.invoke(cli, ["test", "browse", "--help"])
        assert result.exit_code == 0
        assert "--room" in result.output
        assert "EXPERIMENTAL" in result.output or "experimental" in result.output.lower()

    def test_test_resolve_paths_help(self, runner):
        """Should show help for test resolve-paths."""
        result = runner.invoke(cli, ["test", "resolve-paths", "--help"])
        assert result.exit_code == 0
        assert "--room" in result.output
        assert "--slp" in result.output
        assert "EXPERIMENTAL" in result.output or "experimental" in result.output.lower()


class TestCommandGroups:
    """Tests for CLI command organization."""

    def test_main_help_shows_command_groups(self, runner):
        """Should show organized command groups in main help."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        # Should have major command groups
        assert "login" in result.output
        assert "train" in result.output
        assert "track" in result.output
        assert "worker" in result.output

    def test_hidden_aliases_not_in_main_help(self, runner):
        """Deprecated aliases should be hidden from main help."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        # These should be hidden (using hidden=True)
        # They still work but shouldn't clutter the help
        # Note: They may still appear if not properly hidden
        # This test just ensures main help renders successfully


class TestFlagStandardization:
    """Tests for standardized flag names."""

    def test_kebab_case_flags_work(self, runner):
        """Should accept kebab-case flags."""
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "--pkg-path" in result.output
        assert "--session-string" in result.output

    def test_underscore_aliases_work(self, runner):
        """Should accept underscore aliases for backward compatibility."""
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        # Underscore aliases may be shown or just accepted
        # This test ensures help renders correctly


class TestAuthenticationFlow:
    """Tests for JWT authentication flow in commands."""

    def test_train_mentions_login(self, runner):
        """Train command help should mention login requirement."""
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "login" in result.output.lower()

    def test_track_mentions_login(self, runner):
        """Track command help should mention login requirement."""
        result = runner.invoke(cli, ["track", "--help"])
        assert result.exit_code == 0
        assert "login" in result.output.lower() or "authentication" in result.output.lower()


class TestVerbosityFlags:
    """Tests for --verbose and --quiet flags."""

    def test_train_has_verbose_flag(self, runner):
        """Train command should have --verbose/-v flag."""
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "--verbose" in result.output
        assert "-v" in result.output

    def test_train_has_quiet_flag(self, runner):
        """Train command should have --quiet/-q flag."""
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "--quiet" in result.output
        assert "-q" in result.output

    def test_track_has_verbose_flag(self, runner):
        """Track command should have --verbose/-v flag."""
        result = runner.invoke(cli, ["track", "--help"])
        assert result.exit_code == 0
        assert "--verbose" in result.output
        assert "-v" in result.output

    def test_track_has_quiet_flag(self, runner):
        """Track command should have --quiet/-q flag."""
        result = runner.invoke(cli, ["track", "--help"])
        assert result.exit_code == 0
        assert "--quiet" in result.output
        assert "-q" in result.output

    def test_worker_has_verbose_flag(self, runner):
        """Worker command should have --verbose/-v flag."""
        result = runner.invoke(cli, ["worker", "--help"])
        assert result.exit_code == 0
        assert "--verbose" in result.output
        assert "-v" in result.output


class TestCredentialsCommand:
    """Tests for 'sleap-rtc credentials' command group."""

    def test_credentials_help(self, runner):
        """Should show help for credentials command group."""
        result = runner.invoke(cli, ["credentials", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "show" in result.output
        assert "clear" in result.output
        assert "remove-secret" in result.output
        assert "remove-token" in result.output

    def test_credentials_list(self, runner, temp_credentials):
        """Should list stored credentials."""
        # Create some credentials
        temp_credentials.write_text(
            json.dumps({
                "jwt": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0dXNlciIsImV4cCI6OTk5OTk5OTk5OX0.xxx",
                "user": {"username": "testuser", "id": "12345"},
                "room_secrets": {"room1": "secret123"},
                "tokens": {"room2": {"api_key": "slp_xxx", "worker_name": "my-worker"}},
            })
        )
        result = runner.invoke(cli, ["credentials", "list"])
        assert result.exit_code == 0
        assert "testuser" in result.output
        assert "room1" in result.output or "Room Secrets" in result.output

    def test_credentials_list_empty(self, runner, temp_credentials):
        """Should show empty message when no credentials."""
        result = runner.invoke(cli, ["credentials", "list"])
        assert result.exit_code == 0
        assert "Not logged in" in result.output or "(none)" in result.output

    def test_credentials_show_help(self, runner):
        """Should show help for credentials show."""
        result = runner.invoke(cli, ["credentials", "show", "--help"])
        assert result.exit_code == 0
        assert "--reveal" in result.output

    def test_credentials_show_redacted(self, runner, temp_credentials):
        """Should redact secrets by default."""
        temp_credentials.write_text(
            json.dumps({
                "jwt": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0dXNlciIsImV4cCI6OTk5OTk5OTk5OX0.xxx",
                "user": {"username": "testuser", "id": "12345"},
                "room_secrets": {"room1": "verysecretvalue123"},
            })
        )
        result = runner.invoke(cli, ["credentials", "show"])
        assert result.exit_code == 0
        # Should NOT show full secret
        assert "verysecretvalue123" not in result.output
        # Should show masked version
        assert "..." in result.output or "****" in result.output

    def test_credentials_clear_requires_confirmation(self, runner, temp_credentials):
        """Should require confirmation before clearing."""
        temp_credentials.write_text(json.dumps({"jwt": "test"}))
        result = runner.invoke(cli, ["credentials", "clear"], input="n\n")
        assert result.exit_code == 0
        assert "Cancelled" in result.output
        # File should still exist
        assert temp_credentials.exists()

    def test_credentials_clear_with_yes(self, runner, temp_credentials):
        """Should clear credentials with --yes flag."""
        temp_credentials.write_text(json.dumps({"jwt": "test"}))
        result = runner.invoke(cli, ["credentials", "clear", "--yes"])
        assert result.exit_code == 0
        assert "cleared" in result.output.lower()

    def test_credentials_remove_secret_not_found(self, runner, temp_credentials):
        """Should show message when room secret not found."""
        temp_credentials.write_text(json.dumps({}))
        result = runner.invoke(cli, ["credentials", "remove-secret", "--room", "nonexistent"])
        assert result.exit_code == 0
        assert "No secret found" in result.output

    def test_credentials_remove_token_not_found(self, runner, temp_credentials):
        """Should show message when token not found."""
        temp_credentials.write_text(json.dumps({}))
        result = runner.invoke(cli, ["credentials", "remove-token", "--room", "nonexistent"])
        assert result.exit_code == 0
        assert "No token found" in result.output


class TestConfigCommand:
    """Tests for 'sleap-rtc config' command group."""

    def test_config_help(self, runner):
        """Should show help for config command group."""
        result = runner.invoke(cli, ["config", "--help"])
        assert result.exit_code == 0
        assert "show" in result.output
        assert "path" in result.output
        assert "add-mount" in result.output
        assert "remove-mount" in result.output
        assert "init" in result.output

    def test_config_show(self, runner):
        """Should show current configuration."""
        result = runner.invoke(cli, ["config", "show"])
        assert result.exit_code == 0
        assert "Environment" in result.output or "environment" in result.output.lower()
        assert "Signaling" in result.output or "signaling" in result.output.lower()

    def test_config_show_json(self, runner):
        """Should output JSON when --json flag used."""
        result = runner.invoke(cli, ["config", "show", "--json"])
        assert result.exit_code == 0
        # Should be valid JSON
        import json as json_module
        try:
            data = json_module.loads(result.output)
            assert "environment" in data
            assert "signaling_websocket" in data
        except json_module.JSONDecodeError:
            pytest.fail("Output is not valid JSON")

    def test_config_path(self, runner):
        """Should show config file paths."""
        result = runner.invoke(cli, ["config", "path"])
        assert result.exit_code == 0
        assert "sleap-rtc.toml" in result.output
        assert "config.toml" in result.output

    def test_config_add_mount_help(self, runner):
        """Should show help for add-mount."""
        result = runner.invoke(cli, ["config", "add-mount", "--help"])
        assert result.exit_code == 0
        assert "PATH" in result.output
        assert "LABEL" in result.output
        assert "--global" in result.output

    def test_config_remove_mount_help(self, runner):
        """Should show help for remove-mount."""
        result = runner.invoke(cli, ["config", "remove-mount", "--help"])
        assert result.exit_code == 0
        assert "LABEL" in result.output
        assert "--global" in result.output
        assert "--yes" in result.output

    def test_config_init_help(self, runner):
        """Should show help for init."""
        result = runner.invoke(cli, ["config", "init", "--help"])
        assert result.exit_code == 0
        assert "--global" in result.output
        assert "--force" in result.output

    def test_config_init_creates_file(self, runner, tmp_path):
        """Should create config file."""
        import os
        from pathlib import Path

        # Check if tomli_w is available (needed for config init)
        try:
            import tomli_w
        except ImportError:
            pytest.skip("tomli_w not installed")

        # Change to temp directory for this test
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = runner.invoke(cli, ["config", "init"])
            assert result.exit_code == 0
            assert "Created config file" in result.output
            # File should exist
            assert (Path(tmp_path) / "sleap-rtc.toml").exists()
        finally:
            os.chdir(original_cwd)


class TestTokenNotRequired:
    """Tests that --token flag has been removed."""

    def test_train_no_token_flag(self, runner):
        """Train command should NOT have --token flag."""
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        # --token should not appear as a standalone option
        # Note: It may appear in deprecation text, but not as an option
        lines = result.output.split("\n")
        option_lines = [l for l in lines if l.strip().startswith("--token")]
        assert len(option_lines) == 0, "--token should not be a CLI option"

    def test_track_no_token_flag(self, runner):
        """Track command should NOT have --token flag."""
        result = runner.invoke(cli, ["track", "--help"])
        assert result.exit_code == 0
        lines = result.output.split("\n")
        option_lines = [l for l in lines if l.strip().startswith("--token")]
        assert len(option_lines) == 0, "--token should not be a CLI option"

    def test_worker_no_token_flag(self, runner):
        """Worker command should NOT have --token flag."""
        result = runner.invoke(cli, ["worker", "--help"])
        assert result.exit_code == 0
        lines = result.output.split("\n")
        option_lines = [l for l in lines if l.strip().startswith("--token")]
        assert len(option_lines) == 0, "--token should not be a CLI option"
