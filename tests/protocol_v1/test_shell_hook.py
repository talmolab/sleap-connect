"""Tests for the new-terminal reminder (`shell_hook.py`)."""

import os
import shutil
import subprocess
import sys

import pytest

from sleap_rtc.protocol_v1 import daemon, shell_hook

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shells")


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A fake home folder, so startup files are never the real ones."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("ZDOTDIR", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv(shell_hook.OPT_OUT_ENV, raising=False)
    monkeypatch.setattr(
        shell_hook, "_powershell_profiles", lambda: [home / "Documents" / "profile.ps1"]
    )
    return home


class TestStartupFiles:
    def test_zsh_honors_zdotdir(self, home, monkeypatch):
        assert shell_hook.startup_files("zsh") == [home / ".zshrc"]
        monkeypatch.setenv("ZDOTDIR", str(home / "zdot"))
        assert shell_hook.startup_files("zsh") == [home / "zdot" / ".zshrc"]

    def test_bash_uses_bash_profile_on_macos(self, home, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        assert shell_hook.startup_files("bash") == [home / ".bash_profile"]
        monkeypatch.setattr(sys, "platform", "linux")
        assert shell_hook.startup_files("bash") == [home / ".bashrc"]

    def test_fish(self, home):
        assert shell_hook.startup_files("fish") == [
            home / ".config" / "fish" / "config.fish"
        ]

    @posix_only
    def test_detects_the_login_shell(self, monkeypatch):
        monkeypatch.setenv("SHELL", "/bin/zsh")
        assert shell_hook.detect_shell() == "zsh"
        monkeypatch.setenv("SHELL", "/usr/bin/tcsh")
        assert shell_hook.detect_shell() is None


# The shells `uninstall` looks after on this OS.
_THIS_OS_SHELLS = ["powershell"] if sys.platform == "win32" else ["zsh", "bash", "fish"]


class TestInstall:
    @pytest.mark.parametrize("shell", _THIS_OS_SHELLS)
    def test_adds_one_block_and_uninstall_restores_the_file(self, home, shell):
        rc = shell_hook.startup_files(shell)[0]
        rc.parent.mkdir(parents=True, exist_ok=True)
        original = "export FOO=1\nalias ll='ls -l'\n"
        rc.write_text(original)

        result = shell_hook.install(shell)
        assert result.added == [rc]
        text = rc.read_text()
        assert text.startswith(original)
        assert text.count(shell_hook.BEGIN_MARKER) == 1
        assert shell_hook.notice_path(shell).name in text
        assert shell_hook.notice_path(shell).exists()

        # Idempotent.
        again = shell_hook.install(shell)
        assert again.added == [] and again.already == [rc]
        assert rc.read_text() == text

        assert shell_hook.uninstall() == [rc]
        assert rc.read_text() == original

    @posix_only
    def test_creates_a_missing_startup_file(self, home):
        result = shell_hook.install("fish")
        rc = home / ".config" / "fish" / "config.fish"
        assert result.added == [rc]
        assert rc.read_text().startswith(shell_hook.BEGIN_MARKER)

    @posix_only
    def test_keeps_a_symlinked_startup_file_a_symlink(self, home):
        target = home / "dotfiles" / "zshrc"
        target.parent.mkdir()
        target.write_text("export FOO=1\n")
        (home / ".zshrc").symlink_to(target)

        shell_hook.install("zsh")
        assert (home / ".zshrc").is_symlink()
        assert shell_hook.BEGIN_MARKER in target.read_text()

    def test_notice_scripts_point_at_the_registry(self, home):
        shell_hook.write_notice_scripts()
        # (fish escapes backslashes, so on Windows only PowerShell's applies.)
        for shell in ("powershell",) if sys.platform == "win32" else ("bash", "fish"):
            assert str(daemon.registry_dir()) in shell_hook.notice_path(
                shell
            ).read_text(encoding="utf-8-sig")


class TestAutoInstall:
    @posix_only
    def test_adds_it_once_and_says_so(self, home, monkeypatch):
        monkeypatch.setenv("SHELL", "/bin/zsh")
        message = shell_hook.auto_install(interactive=True)
        assert "Added a new-terminal reminder to ~/.zshrc" in message
        assert "sleap-rtc shell-hook uninstall" in message
        assert shell_hook.auto_install(interactive=True) is None

    @posix_only
    def test_never_when_not_at_a_terminal(self, home, monkeypatch):
        monkeypatch.setenv("SHELL", "/bin/zsh")
        assert shell_hook.auto_install(interactive=False) is None
        assert not (home / ".zshrc").exists()

    @posix_only
    def test_never_when_opted_out(self, home, monkeypatch):
        monkeypatch.setenv("SHELL", "/bin/zsh")
        monkeypatch.setenv(shell_hook.OPT_OUT_ENV, "1")
        assert shell_hook.auto_install(interactive=True) is None
        assert not (home / ".zshrc").exists()

    @posix_only
    def test_never_again_after_uninstall(self, home, monkeypatch):
        monkeypatch.setenv("SHELL", "/bin/zsh")
        shell_hook.auto_install(interactive=True)
        shell_hook.uninstall()
        assert shell_hook.auto_install(interactive=True) is None
        assert shell_hook.BEGIN_MARKER not in (home / ".zshrc").read_text()
        # An explicit install still works, and re-enables it.
        shell_hook.install("zsh")
        assert shell_hook.BEGIN_MARKER in (home / ".zshrc").read_text()

    @posix_only
    def test_unknown_shell_gets_a_tip(self, home, monkeypatch):
        monkeypatch.setenv("SHELL", "/usr/bin/tcsh")
        assert "shell-hook install" in shell_hook.auto_install(interactive=True)


def _sleeper(tag: str) -> subprocess.Popen:
    # `tag` lands in the process's command line, which the notice checks.
    return subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep(60)  # {tag}"]
    )


def _register(pid: int, data_dir) -> None:
    entry = daemon.registry_path(data_dir)
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text(f"{pid}\t1.0\tOct 29 14:02\t{data_dir}\n")


def _source(shell: str, interactive: bool) -> str:
    script = str(shell_hook.notice_path("bash"))
    no_rc = {"bash": "--norc", "zsh": "-f"}[shell]
    flags = [no_rc, "-i"] if interactive else [no_rc]
    return subprocess.run(
        [shell, *flags, "-c", f". '{script}'"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=30,
    ).stdout


@posix_only
@pytest.mark.parametrize("shell", ["bash", "zsh"])
class TestNoticeScript:
    @pytest.fixture(autouse=True)
    def _needs(self, shell, home):
        if shutil.which(shell) is None:
            pytest.skip(f"{shell} not installed")
        shell_hook.write_notice_scripts()

    def test_prints_one_line_per_running_worker(self, shell, tmp_path):
        worker = _sleeper("sleap-rtc serve")
        try:
            _register(worker.pid, tmp_path)
            out = _source(shell, interactive=True)
        finally:
            worker.kill()
            worker.wait()
        assert (
            f"sleap-connect worker is running (pid {worker.pid}, since Oct 29 14:02)"
            in out
        )
        assert "status: sleap-rtc status" in out
        assert f"sleap-rtc serve --stop --data-dir '{tmp_path}'" in out

    def test_silent_in_non_interactive_shells(self, shell, tmp_path):
        worker = _sleeper("sleap-rtc serve")
        try:
            _register(worker.pid, tmp_path)
            out = _source(shell, interactive=False)
        finally:
            worker.kill()
            worker.wait()
        assert out == ""

    def test_silent_and_cleans_up_when_the_worker_is_gone(self, shell, tmp_path):
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        _register(dead.pid, tmp_path)
        assert _source(shell, interactive=True) == ""
        assert not daemon.registry_path(tmp_path).exists()

    def test_ignores_a_reused_pid(self, shell, tmp_path):
        other = _sleeper("something else")
        try:
            _register(other.pid, tmp_path)
            out = _source(shell, interactive=True)
        finally:
            other.kill()
            other.wait()
        assert "sleap-connect" not in out

    def test_silent_with_nothing_registered(self, shell):
        assert _source(shell, interactive=True) == ""
