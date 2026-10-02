"""Tests for `serve --daemonize` / `serve --stop` and the serve pidfile."""

import json
import os
import signal
import socket
import subprocess
import sys
import time

import psutil
import pytest

from sleap_rtc.protocol_v1 import daemon

_CLI = [
    sys.executable,
    "-c",
    "from sleap_rtc.cli import cli; cli(prog_name='sleap-rtc')",
]

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_cli(*args, timeout=90):
    return subprocess.run(
        [*_CLI, *args], capture_output=True, text=True, timeout=timeout
    )


def _serve_args(data_dir, port):
    return [
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--blob-port",
        str(_free_port()),
        "--data-dir",
        str(data_dir),
        "--no-iroh",
        "--no-metrics",
    ]


def _wait_for(predicate, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


def _accepts_connections(port) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


class TestPidfile:
    def test_acquire_records_this_process(self, tmp_path):
        me = daemon.acquire_pidfile(tmp_path)
        assert me.pid == os.getpid()
        assert daemon.read_running(tmp_path) == me

    def test_reacquiring_from_the_same_process_is_allowed(self, tmp_path):
        daemon.acquire_pidfile(tmp_path)
        assert daemon.acquire_pidfile(tmp_path).pid == os.getpid()

    def test_refuses_while_another_live_process_holds_it(self, tmp_path):
        other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            daemon.pid_path(tmp_path).write_text(
                json.dumps(
                    {
                        "pid": other.pid,
                        "started_at": psutil.Process(other.pid).create_time(),
                    }
                )
            )
            with pytest.raises(daemon.AlreadyRunningError) as exc:
                daemon.acquire_pidfile(tmp_path)
            assert exc.value.running.pid == other.pid
        finally:
            other.kill()
            other.wait()

    def test_a_dead_processes_pidfile_is_cleared(self, tmp_path):
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        started_at = psutil.Process(dead.pid).create_time()
        dead.wait()
        daemon.pid_path(tmp_path).write_text(
            json.dumps({"pid": dead.pid, "started_at": started_at})
        )
        assert daemon.read_running(tmp_path) is None
        assert not daemon.pid_path(tmp_path).exists()
        assert daemon.acquire_pidfile(tmp_path).pid == os.getpid()

    def test_a_reused_pid_is_not_mistaken_for_the_worker(self, tmp_path):
        # Our own PID, but a start time from long before we existed.
        daemon.pid_path(tmp_path).write_text(
            json.dumps({"pid": os.getpid(), "started_at": 1.0})
        )
        assert daemon.read_running(tmp_path) is None

    def test_a_corrupt_pidfile_is_cleared(self, tmp_path):
        daemon.pid_path(tmp_path).write_text("{not json")
        assert daemon.read_running(tmp_path) is None
        assert not daemon.pid_path(tmp_path).exists()

    def test_release_leaves_another_processes_pidfile_alone(self, tmp_path):
        daemon.pid_path(tmp_path).write_text(json.dumps({"pid": 1, "started_at": 1.0}))
        daemon.release_pidfile(tmp_path)
        assert daemon.pid_path(tmp_path).exists()


class TestServeArgv:
    def test_round_trips_every_option(self, tmp_path):
        argv = daemon.serve_argv(
            host="127.0.0.1",
            port=9000,
            blob_port=9005,
            data_dir=tmp_path,
            mounts=("/a:lab", "/b"),
            iroh=False,
            metrics=True,
        )
        assert argv == [
            "serve",
            "--host", "127.0.0.1",
            "--port", "9000",
            "--data-dir", str(tmp_path),
            "--blob-port", "9005",
            "--mount", "/a:lab",
            "--mount", "/b",
            "--no-iroh",
            "--metrics",
        ]  # fmt: skip

    def test_omits_blob_port_when_defaulted(self, tmp_path):
        argv = daemon.serve_argv(
            host="0.0.0.0",
            port=9000,
            blob_port=None,
            data_dir=tmp_path,
            mounts=(),
            iroh=True,
            metrics=False,
        )
        assert "--blob-port" not in argv


class TestDaemonize:
    """End-to-end, against a real detached `serve` child."""

    def test_returns_once_listening_and_stop_shuts_it_down(self, tmp_path):
        port = _free_port()
        started = _run_cli(*_serve_args(tmp_path, port), "--daemonize")
        assert started.returncode == 0, started.stdout + started.stderr
        assert "running in the background" in started.stdout

        try:
            running = daemon.read_running(tmp_path)
            assert running is not None
            # Ready means ready: listening by the time --daemonize returns.
            assert _accepts_connections(port)
            # Fully detached: its own session, not our child process group.
            if sys.platform != "win32":
                assert os.getsid(running.pid) == running.pid
            assert not daemon.ready_path(tmp_path).exists()
        finally:
            stopped = _run_cli("serve", "--stop", "--data-dir", str(tmp_path))

        assert stopped.returncode == 0, stopped.stdout + stopped.stderr
        assert f"Stopped worker (pid {running.pid})" in stopped.stdout
        assert daemon.read_running(tmp_path) is None
        assert not daemon.pid_path(tmp_path).exists()
        assert not _accepts_connections(port)
        assert "Listening." in daemon.log_path(tmp_path).read_text()

    def test_refuses_a_second_worker_on_the_same_data_dir(self, tmp_path):
        started = _run_cli(*_serve_args(tmp_path, _free_port()), "--daemonize")
        assert started.returncode == 0, started.stdout + started.stderr
        try:
            second = _run_cli(*_serve_args(tmp_path, _free_port()), "--daemonize")
            assert second.returncode != 0
            assert "already running" in second.stderr
            foreground = _run_cli(*_serve_args(tmp_path, _free_port()))
            assert foreground.returncode != 0
            assert "already running" in foreground.stderr
        finally:
            _run_cli("serve", "--stop", "--data-dir", str(tmp_path))

    def test_reports_a_child_that_fails_to_start(self, tmp_path):
        with socket.socket() as taken:
            taken.bind(("127.0.0.1", 0))
            taken.listen()
            port = taken.getsockname()[1]
            started = _run_cli(*_serve_args(tmp_path, port), "--daemonize")

        assert started.returncode != 0
        assert "exited during startup" in started.stderr
        # The child's own traceback is surfaced, not just "it failed".
        assert "address already in use" in started.stderr.lower()
        assert daemon.read_running(tmp_path) is None

    def test_stop_with_nothing_running(self, tmp_path):
        result = _run_cli("serve", "--stop", "--data-dir", str(tmp_path))
        assert result.returncode == 0
        assert "No worker is running" in result.stdout


@posix_only
class TestForegroundSigterm:
    def test_sigterm_shuts_down_cleanly(self, tmp_path):
        ready = tmp_path / "ready.json"
        proc = subprocess.Popen(
            [*_CLI, *_serve_args(tmp_path, _free_port()), "--ready-file", str(ready)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            assert _wait_for(ready.exists), "serve never reported ready"
            assert daemon.read_running(tmp_path).pid == proc.pid
            proc.send_signal(signal.SIGTERM)
            output, _ = proc.communicate(timeout=30)
        finally:
            if proc.poll() is None:
                proc.kill()

        assert proc.returncode == 0, output
        assert "Traceback" not in output
        assert not daemon.pid_path(tmp_path).exists()
