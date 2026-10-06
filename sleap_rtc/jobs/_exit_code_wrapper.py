"""Run a job's command and record its exit code to a file when it exits.

Usage: python _exit_code_wrapper.py EXIT_FILE -- CMD [ARGS...]

`spawn_detached` launches every job through this, because the worker that
spawned a job may not be the one that sees it finish: after a worker
restart (`serve --stop`, a crash, an upgrade) the job is *reattached*, and a
reattached process is no longer the worker's child, so its exit code can't
be `wait()`ed for. The exit file is how the new worker learns whether the
job succeeded.

Run as a plain script (not `-m sleap_rtc...`) and stdlib-only, so it starts
fast and doesn't import the worker's dependencies into every job.

The wrapper exits with the job's own exit code (re-raising the same signal
if the job was killed by one), so a worker that *is* still the parent sees
exactly what it would have seen without the wrapper.
"""

import os
import signal
import subprocess
import sys


def _ignore(signum, frame):
    """Handler (not SIG_IGN, which `exec` would pass on to the job itself)."""


def main() -> None:
    exit_path = sys.argv[1]
    if sys.argv[2] != "--":
        sys.exit("usage: _exit_code_wrapper.py EXIT_FILE -- CMD [ARGS...]")
    cmd = sys.argv[3:]

    # Stop/cancel signal the job's whole process group (`send_stop_signal` /
    # `send_cancel_signal`), which includes this wrapper. The job decides how
    # to react; the wrapper has to outlive it to record what happened.
    if sys.platform != "win32":
        signal.signal(signal.SIGINT, _ignore)
        signal.signal(signal.SIGTERM, _ignore)

    try:
        returncode = subprocess.Popen(cmd).wait()
    except OSError as e:
        print(f"Failed to start {cmd[0]!r}: {e}", flush=True)
        returncode = 127

    tmp = exit_path + ".tmp"
    with open(tmp, "w") as f:
        f.write(str(returncode))
    os.replace(tmp, exit_path)

    if returncode < 0 and sys.platform != "win32":
        signal.signal(-returncode, signal.SIG_DFL)
        os.kill(os.getpid(), -returncode)
    sys.exit(returncode if returncode >= 0 else 1)


if __name__ == "__main__":
    main()
