"""Tests for finding the sleap-nn jobs run, and checking it can start.

The check runs a real subprocess (sleap-nn's Python importing its training
code) against stand-in `sleap_nn` and `torch` packages on PYTHONPATH, so a
missing system library is simulated by an ImportError like the real one.
"""

import sys
import textwrap

import pytest

from sleap_connect.jobs import sleap_nn as sleap_nn_module
from sleap_connect.jobs.builder import CommandBuilder
from sleap_connect.jobs.sleap_nn import (
    SleapNNCheck,
    _python_for,
    check_sleap_nn,
    sleap_nn_command,
)
from sleap_connect.jobs.spec import TrainJobSpec
from sleap_connect.protocol_v1 import cli as v1_cli

EGL_ERROR = "libEGL.so.1: cannot open shared object file: No such file or directory"


def _fake_command(tmp_path):
    """A sleap-nn console script whose shebang is this test's Python."""
    command = tmp_path / "bin" / "sleap-nn"
    command.parent.mkdir()
    command.write_text(f"#!{sys.executable}\nprint('sleap-nn 0.3.5')\n")
    command.chmod(0o755)
    return command


def _fake_packages(tmp_path, train_body="", cuda=True):
    site = tmp_path / "site"
    (site / "sleap_nn").mkdir(parents=True)
    (site / "sleap_nn" / "__init__.py").write_text('__version__ = "0.3.5"\n')
    (site / "sleap_nn" / "train.py").write_text(train_body)
    (site / "torch").mkdir()
    (site / "torch" / "__init__.py").write_text(textwrap.dedent(f"""\
            __version__ = "2.9.0+cu128"
            class cuda:
                @staticmethod
                def is_available():
                    return {cuda}
                @staticmethod
                def get_device_name(i):
                    return "NVIDIA A100"
            class backends:
                mps = None
            """))
    return site


@pytest.fixture
def bundled(tmp_path, monkeypatch):
    """sleap-nn installed with the worker (the `nn` extra)."""
    command = _fake_command(tmp_path)
    monkeypatch.setattr(sleap_nn_module, "bundled_sleap_nn", lambda: command)
    return command


class TestCommand:
    def test_prefers_the_one_installed_with_the_worker(self, bundled):
        assert sleap_nn_command() == str(bundled)
        cmd = CommandBuilder().build_train_command(
            TrainJobSpec(config_path="/c/centroid.yaml")
        )
        assert cmd[:2] == [str(bundled), "train"]

    def test_else_plain_sleap_nn_from_path(self, monkeypatch):
        monkeypatch.setattr(sleap_nn_module, "bundled_sleap_nn", lambda: None)
        assert sleap_nn_command() == "sleap-nn"

    def test_bundled_means_next_to_this_python(self, tmp_path, monkeypatch):
        python = tmp_path / "python"
        monkeypatch.setattr(sleap_nn_module.sys, "executable", str(python))
        assert sleap_nn_module.bundled_sleap_nn() is None
        (tmp_path / sleap_nn_module._EXE).write_text("")
        assert sleap_nn_module.bundled_sleap_nn() == tmp_path / sleap_nn_module._EXE


class TestCheck:
    def test_working_install(self, bundled, tmp_path, monkeypatch):
        monkeypatch.setenv("PYTHONPATH", str(_fake_packages(tmp_path)))

        check = check_sleap_nn()

        assert check.ok, check
        assert check.bundled
        assert (check.version, check.torch) == ("0.3.5", "2.9.0+cu128")
        assert check.cuda and check.gpu == "NVIDIA A100"

    def test_missing_system_library(self, bundled, tmp_path, monkeypatch):
        site = _fake_packages(tmp_path, train_body=f"raise ImportError({EGL_ERROR!r})")
        monkeypatch.setenv("PYTHONPATH", str(site))

        check = check_sleap_nn()

        assert not check.ok
        assert check.error == f"ImportError: {EGL_ERROR}"
        assert "libEGL.so.1" in check.hint()
        assert "apt-get install" in check.hint()

    def test_not_installed(self, monkeypatch):
        monkeypatch.setattr(sleap_nn_module, "bundled_sleap_nn", lambda: None)
        monkeypatch.setattr(sleap_nn_module.shutil, "which", lambda name: None)

        check = check_sleap_nn()

        assert check.path is None and not check.ok
        assert 'sleap-connect[nn]' in check.hint()

    def test_python_from_a_long_path_sh_shebang(self, tmp_path):
        command = tmp_path / "sleap-nn"
        command.write_text(
            "#!/bin/sh\n"
            "'''exec' \"/very/long/path/bin/python\" \"$0\" \"$@\"\n"
            "' '''\n"
        )
        assert _python_for(command) == "/very/long/path/bin/python"


class TestServeAndDoctorOutput:
    def test_serve_line_when_it_works(self, capsys):
        v1_cli._print_sleap_nn_check(
            SleapNNCheck(
                path="/x/sleap-nn",
                bundled=True,
                version="0.3.5",
                torch="2.9",
                cuda=True,
                gpu="NVIDIA A100",
            )
        )
        out = capsys.readouterr().out
        assert "sleap-nn 0.3.5 (installed with the worker) · GPU: NVIDIA A100" in out

    def test_serve_warns_when_it_cant_start(self, capsys):
        v1_cli._print_sleap_nn_check(
            SleapNNCheck(path="/x/sleap-nn", error=f"ImportError: {EGL_ERROR}")
        )
        out = capsys.readouterr().out
        assert f"sleap-nn can't start: ImportError: {EGL_ERROR}" in out
        assert "jobs will fail until this is fixed" in out
        assert "apt-get install" in out

    def test_serve_warns_when_its_missing(self, capsys):
        v1_cli._print_sleap_nn_check(SleapNNCheck(path=None))
        out = capsys.readouterr().out
        assert "sleap-nn isn't installed" in out
        assert 'uv tool install "sleap-connect[nn]"' in out

    def test_skipped_check_prints_nothing(self, capsys, monkeypatch):
        monkeypatch.setenv("SLEAP_CONNECT_SKIP_SLEAP_NN_CHECK", "1")
        v1_cli._print_sleap_nn_check(v1_cli._start_sleap_nn_check()())
        assert capsys.readouterr().out == ""

    def test_doctor_fails_when_it_cant_start(self, capsys, monkeypatch):
        monkeypatch.setattr(
            v1_cli,
            "check_sleap_nn",
            lambda: SleapNNCheck(path="/x/sleap-nn", error="ImportError: nope"),
        )
        assert v1_cli.print_sleap_nn_doctor_section() is False
        assert "ImportError: nope" in capsys.readouterr().out
