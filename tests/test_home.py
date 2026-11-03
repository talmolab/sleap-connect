"""The rename: the `sleap_rtc` import shim and moving ~/.sleap-rtc to ~/.sleap-connect."""

import os
import sys

import pytest
from click.testing import CliRunner

from sleap_connect import home

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="symlink left behind on POSIX only"
)


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """A fake home folder, with no folder override set."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    for name in ("SLEAP_CONNECT_HOME", "SLEAP_RTC_HOME", "ZDOTDIR", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def _start_old_worker(monkeypatch, old):
    """Make this test process stand in for a worker sleap-rtc started: its
    pidfile and registry entry live under the old folder."""
    from sleap_connect.protocol_v1 import daemon

    monkeypatch.setenv("SLEAP_CONNECT_HOME", str(old))
    daemon.acquire_pidfile(old / "protocol_v1")
    monkeypatch.delenv("SLEAP_CONNECT_HOME")


def _old_home(fake_home):
    old = fake_home / ".sleap-rtc"
    (old / "protocol_v1").mkdir(parents=True)
    (old / "protocol_v1" / "identity.json").write_text('{"k": 1}')
    (old / "config.toml").write_text("[[path_mappings]]\n")
    return old


class TestImportShim:
    def test_old_module_names_are_the_same_modules(self):
        import sleap_connect
        import sleap_rtc
        import sleap_rtc.protocol_v1.daemon as old_daemon
        from sleap_connect.protocol_v1 import daemon as new_daemon

        assert sleap_rtc is sleap_connect
        assert old_daemon is new_daemon


class TestEnv:
    def test_new_names_win_and_old_names_still_work(self, fake_home, monkeypatch):
        assert home.home_dir() == fake_home / ".sleap-connect"
        monkeypatch.setenv("SLEAP_RTC_HOME", str(fake_home / "old-override"))
        assert home.home_dir() == fake_home / "old-override"
        monkeypatch.setenv("SLEAP_CONNECT_HOME", str(fake_home / "new-override"))
        assert home.home_dir() == fake_home / "new-override"

        monkeypatch.setenv("SLEAP_RTC_NO_SHELL_HOOK", "1")
        assert home.env_flag("NO_SHELL_HOOK") == "1"


class TestMigration:
    def test_moves_everything_once(self, fake_home):
        _old_home(fake_home)

        message = home.migrate_legacy_home()

        new = fake_home / ".sleap-connect"
        assert "~/.sleap-connect" in message
        assert (new / "protocol_v1" / "identity.json").read_text() == '{"k": 1}'
        assert (new / "config.toml").exists()
        assert home.migrate_legacy_home() is None

    @posix_only
    def test_leaves_a_symlink_for_old_paths(self, fake_home):
        _old_home(fake_home)
        home.migrate_legacy_home()
        old = fake_home / ".sleap-rtc"
        assert old.is_symlink()
        assert (old / "config.toml").exists()

    def test_nothing_to_do_without_an_old_folder(self, fake_home):
        assert home.migrate_legacy_home() is None
        assert not (fake_home / ".sleap-connect").exists()

    def test_leaves_both_alone_when_the_new_folder_exists(self, fake_home):
        _old_home(fake_home)
        (fake_home / ".sleap-connect").mkdir()
        assert home.migrate_legacy_home() is None
        assert (fake_home / ".sleap-rtc" / "config.toml").exists()

    def test_skipped_with_a_folder_override(self, fake_home, monkeypatch):
        _old_home(fake_home)
        monkeypatch.setenv("SLEAP_CONNECT_HOME", str(fake_home / "elsewhere"))
        assert home.migrate_legacy_home() is None
        assert (fake_home / ".sleap-rtc").is_dir()

    def test_refuses_while_an_old_worker_runs(self, fake_home, monkeypatch):
        from sleap_connect.protocol_v1 import daemon

        _start_old_worker(monkeypatch, _old_home(fake_home))
        try:
            with pytest.raises(home.LegacyWorkerRunning) as exc:
                home.migrate_legacy_home()
            assert exc.value.pid == os.getpid()
            assert not (fake_home / ".sleap-connect").exists()
        finally:
            daemon.release_pidfile(home.legacy_data_dir())

    @posix_only
    def test_repoints_the_shell_reminder(self, fake_home, monkeypatch):
        from sleap_connect.protocol_v1 import shell_hook

        old = _old_home(fake_home)
        # A block written by sleap-rtc: it names the old folder's notice script.
        monkeypatch.setenv("SLEAP_CONNECT_HOME", str(old))
        shell_hook.install("zsh")
        monkeypatch.delenv("SLEAP_CONNECT_HOME")
        zshrc = fake_home / ".zshrc"
        assert str(old / "shell") in zshrc.read_text()

        home.migrate_legacy_home()

        text = zshrc.read_text()
        assert text.count(shell_hook.BEGIN_MARKER) == 1
        assert str(fake_home / ".sleap-connect" / "shell" / "notice.sh") in text
        assert ".sleap-rtc" not in text


def _plain(output: str) -> str:
    """CLI output with colors, rich-click's box borders and line wrapping
    removed, so a message can be matched however wide the terminal is."""
    import re

    text = re.sub(r"\x1b\[[0-9;]*m", "", output)
    text = re.sub(r"[│╭╮╰╯─]", " ", text)
    return " ".join(text.split())


class TestCli:
    def test_says_what_moved(self, fake_home):
        from sleap_connect.cli import cli

        _old_home(fake_home)
        result = CliRunner().invoke(cli, ["shell-hook", "status"])
        assert result.exit_code == 0, result.output
        assert "Moved sleap-rtc's settings" in _plain(result.output)
        assert (fake_home / ".sleap-connect").is_dir()

    def test_stateful_commands_refuse_while_an_old_worker_runs(
        self, fake_home, monkeypatch
    ):
        from sleap_connect.cli import cli
        from sleap_connect.protocol_v1 import daemon

        _start_old_worker(monkeypatch, _old_home(fake_home))
        try:
            result = CliRunner().invoke(cli, ["folders"])
            assert result.exit_code != 0
            assert "sleap-connect serve --stop" in _plain(result.output)
            # Read-only commands still run, with the warning.
            result = CliRunner().invoke(cli, ["shell-hook", "status"])
            assert result.exit_code == 0
            assert "still running" in _plain(result.output)
        finally:
            daemon.release_pidfile(home.legacy_data_dir())
