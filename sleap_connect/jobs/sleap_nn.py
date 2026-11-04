"""Which `sleap-nn` the worker runs its jobs with, and whether it can start.

Two ways to have one: installed together with the worker (`uv tool install
"sleap-rtc[nn]"`), which puts it next to this Python but not on PATH
(uv only exposes the tool's own commands), or installed on its own (`uv tool
install sleap-nn`), which puts it on PATH. The bundled one wins: it's the
one installed for this worker.

`check_sleap_nn` goes further than finding the command: it imports
sleap-nn's training code in sleap-nn's own Python, which is what fails when,
say, a system library it needs (libEGL for skia) is missing — so `serve`
and `doctor` can say so before a job fails on it.
"""

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_EXE = "sleap-nn.exe" if sys.platform == "win32" else "sleap-nn"
_PYTHON = "python.exe" if sys.platform == "win32" else "python"

# Importing the training code pulls in torch and friends: slow on a cold
# start (several seconds), so give it room.
CHECK_TIMEOUT_SECS = 120

_CHECK_SCRIPT = """\
import json, sleap_nn, sleap_nn.train, torch
cuda = torch.cuda.is_available()
mps = getattr(torch.backends, "mps", None)
print(json.dumps({
    "version": getattr(sleap_nn, "__version__", None),
    "torch": torch.__version__,
    "cuda": cuda,
    "gpu": torch.cuda.get_device_name(0) if cuda else None,
    "mps": bool(mps and mps.is_available()),
}))
"""


def bundled_sleap_nn() -> Optional[Path]:
    """The sleap-nn installed in this worker's own environment, if any."""
    path = Path(sys.executable).parent / _EXE
    return path if path.is_file() else None


def sleap_nn_command() -> str:
    """What to run for `sleap-nn`: the bundled one, else plain "sleap-nn".

    Plain "sleap-nn" is looked up on PATH when the job starts.
    """
    bundled = bundled_sleap_nn()
    return str(bundled) if bundled is not None else "sleap-nn"


@dataclass
class SleapNNCheck:
    """What `check_sleap_nn` found.

    Attributes:
        path: The sleap-nn command jobs run, or None if there is none.
        bundled: Whether it's installed with the worker (the `nn` extra).
        version: Its version, if it could be read.
        error: Why it can't start (the import error's last line), or None
            if it's fine or couldn't be checked.
        torch: PyTorch's version in sleap-nn's environment.
        cuda: Whether PyTorch there sees a CUDA GPU.
        gpu: That GPU's name.
        mps: Whether PyTorch there can use Apple's GPU.
    """

    path: Optional[str]
    bundled: bool = False
    version: Optional[str] = None
    error: Optional[str] = None
    torch: Optional[str] = None
    cuda: bool = False
    gpu: Optional[str] = None
    mps: bool = False

    @property
    def ok(self) -> bool:
        return self.path is not None and self.error is None

    def hint(self) -> Optional[str]:
        """What to do about a problem, in a sentence."""
        if self.path is None:
            return (
                'Install it with the worker: uv tool install "sleap-rtc[nn]" '
                "--torch-backend auto --reinstall"
            )
        if self.error and "cannot open shared object file" in self.error:
            lib = re.search(r"(lib[\w.+-]+\.so[\w.]*)", self.error)
            missing = lib.group(1) if lib else "a system library"
            return (
                f"This machine is missing {missing}. On Debian/Ubuntu: "
                "apt-get install -y libegl1 libgl1 libfontconfig1"
            )
        if self.error:
            return "Reinstall it, or run 'sleap-nn system' to see what's wrong."
        return None


def _python_for(command: Path) -> Optional[str]:
    """The Python that runs `command` (a console script), if it can be found."""
    if command.parent == Path(sys.executable).parent:
        return sys.executable
    # uv tool installs link ~/.local/bin/sleap-nn to the tool's own bin/.
    candidate = command.resolve().parent / _PYTHON
    if candidate.is_file():
        return str(candidate)
    try:
        with open(command, "rb") as f:
            first = f.readline(512).decode("utf-8", "replace").strip()
            second = f.readline(512).decode("utf-8", "replace")
    except OSError:
        return None
    if first.startswith("#!/bin/sh"):
        # A long interpreter path: `'''exec' "/path/to/python" "$0" "$@"`.
        match = re.search(r"exec'\s+\"?([^\"]+?)\"?\s+\"\$0\"", second)
        return match.group(1) if match else None
    if first.startswith("#!"):
        interpreter = first[2:].strip()
        return interpreter if Path(interpreter).is_file() else None
    return None


def _last_error_line(stderr: str) -> str:
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    # Skip noise (e.g. macOS objc duplicate-class warnings) after the error.
    for line in reversed(lines):
        if re.match(r"^[\w.]+(Error|Exception)\b", line):
            return line
    return lines[-1] if lines else "sleap-nn failed to start"


def check_sleap_nn(timeout: float = CHECK_TIMEOUT_SECS) -> SleapNNCheck:
    """Find the sleap-nn jobs run, and check its training code imports.

    Never raises. Slow (it imports torch): run it off the event loop.
    """
    bundled = bundled_sleap_nn()
    found = str(bundled) if bundled is not None else shutil.which("sleap-nn")
    if found is None:
        return SleapNNCheck(path=None)
    check = SleapNNCheck(path=found, bundled=bundled is not None)

    python = _python_for(Path(found))
    if python is None:
        # Can't find its Python (e.g. a Windows launcher): the version only.
        try:
            out = subprocess.run(
                [found, "--version"], capture_output=True, text=True, timeout=30
            )
            match = re.search(r"(\d+\.\d+[\w.+-]*)", out.stdout)
            check.version = match.group(1) if match else None
        except (OSError, subprocess.SubprocessError):
            pass
        return check

    try:
        out = subprocess.run(
            [python, "-c", _CHECK_SCRIPT],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        check.error = f"importing sleap-nn took longer than {timeout:.0f}s"
        return check
    except OSError as e:
        check.error = str(e)
        return check
    if out.returncode != 0:
        check.error = _last_error_line(out.stderr)
        return check
    try:
        info = json.loads(out.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return check
    check.version = info.get("version")
    check.torch = info.get("torch")
    check.cuda = bool(info.get("cuda"))
    check.gpu = info.get("gpu")
    check.mps = bool(info.get("mps"))
    return check
