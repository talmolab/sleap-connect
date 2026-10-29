"""New-terminal reminder that a sleap-connect worker is running.

`sleap-rtc serve` runs the worker in the background, where it is easy to
forget about. Like `conda init`, this adds a small marked block to the
user's shell startup file (`~/.zshrc`, `~/.bashrc`, ...). Every new terminal
then runs a "notice" script, which reads the running-worker registry
(`daemon.registry_dir()`) and prints one line per worker still alive (its PID alive and its command line
still a `sleap-rtc` one, so a PID reused after a reboot is ignored):

    ● sleap-connect worker is running (pid 48213, since Oct 29 14:02)
      status: sleap-rtc status · stop: sleap-rtc serve --stop

It prints nothing when no worker is running, and nothing in non-interactive
shells (`scp`, `rsync`, `ssh host cmd`), and it never starts Python, so
opening a terminal stays fast.

The startup file only ever gets the one block that sources the notice
script, which lives under `~/.sleap-rtc/shell/`. `sleap-rtc` rewrites those
scripts on every `serve`, so a later version can change the reminder
without touching the startup file again. Each shell family needs its own
script syntax: `notice.sh` (bash and zsh), `notice.fish`, `notice.ps1`.
"""

import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from sleap_rtc.protocol_v1 import daemon

BEGIN_MARKER = "# >>> sleap-connect reminder >>>"
END_MARKER = "# <<< sleap-connect reminder <<<"

# Set (to anything) to never add the block automatically.
OPT_OUT_ENV = "SLEAP_RTC_NO_SHELL_HOOK"

SHELLS = ("zsh", "bash", "fish", "powershell")

_BLOCK_RE = re.compile(
    rf"\n?{re.escape(BEGIN_MARKER)}\n.*?{re.escape(END_MARKER)}\n?", re.DOTALL
)


def shell_dir() -> Path:
    """Where the notice scripts live."""
    return daemon.sleap_home() / "shell"


def _removed_marker() -> Path:
    # Present once the user ran `shell-hook uninstall`: never re-add the
    # block automatically after that.
    return shell_dir() / "removed"


def notice_path(shell: str) -> Path:
    """The notice script `shell`'s startup block sources."""
    ext = {"fish": "fish", "powershell": "ps1"}.get(shell, "sh")
    return shell_dir() / f"notice.{ext}"


def detect_shell() -> Optional[str]:
    """The user's shell, if it's one we support (else None)."""
    if sys.platform == "win32":
        return "powershell"
    name = Path(os.environ.get("SHELL", "")).name
    return name if name in ("zsh", "bash", "fish") else None


def _powershell_profiles() -> List[Path]:
    """Current-user profile paths of the installed PowerShell(s).

    Windows PowerShell 5.1 and PowerShell 7 keep separate profiles, and
    "Documents" may be redirected (OneDrive), so ask each one rather than
    guessing.
    """
    found = []
    for exe in ("pwsh", "powershell"):
        if shutil.which(exe) is None:
            continue
        try:
            out = subprocess.run(
                [exe, "-NoProfile", "-NonInteractive", "-Command", "$PROFILE"],
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            continue
        if out and Path(out) not in found:
            found.append(Path(out))
    if not found:
        found.append(
            Path.home()
            / "Documents"
            / "WindowsPowerShell"
            / "Microsoft.PowerShell_profile.ps1"
        )
    return found


def startup_files(shell: str) -> List[Path]:
    """The startup file(s) the block goes in for `shell`."""
    home = Path.home()
    if shell == "zsh":
        return [Path(os.environ.get("ZDOTDIR") or home) / ".zshrc"]
    if shell == "bash":
        # macOS Terminal opens login shells, which read ~/.bash_profile.
        return [home / (".bash_profile" if sys.platform == "darwin" else ".bashrc")]
    if shell == "fish":
        config = Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config")
        return [config / "fish" / "config.fish"]
    if shell == "powershell":
        return _powershell_profiles()
    raise ValueError(f"unsupported shell: {shell}")


def _block(shell: str) -> str:
    script = str(notice_path(shell))
    if shell == "fish":
        body = f"test -f {_fish_quote(script)}; and source {_fish_quote(script)}"
    elif shell == "powershell":
        body = f"if (Test-Path {_ps_quote(script)}) {{ . {_ps_quote(script)} }}"
    else:
        body = f"if [ -f {shlex.quote(script)} ]; then . {shlex.quote(script)}; fi"
    return (
        f"{BEGIN_MARKER}\n"
        "# Shows a one-line notice in new terminals while a sleap-connect worker\n"
        "# is running. Remove with: sleap-rtc shell-hook uninstall\n"
        f"{body}\n"
        f"{END_MARKER}\n"
    )


def _fish_quote(s: str) -> str:
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _ps_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _stop_hint() -> str:
    return "sleap-rtc serve --stop"


def _notice_sh() -> str:
    registry = shlex.quote(str(daemon.registry_dir()))
    default_dir = shlex.quote(str(_default_data_dir()))
    return f"""\
# sleap-connect: new-terminal reminder while a worker is running.
# Written by sleap-rtc, which rewrites it on every 'sleap-rtc serve'.
# Remove the reminder with: sleap-rtc shell-hook uninstall
case $- in *i*) ;; *) return 0 2>/dev/null || exit 0 ;; esac
for __sleap_rtc_f in $(command ls {registry} 2>/dev/null); do
  case $__sleap_rtc_f in *.run) ;; *) continue ;; esac
  __sleap_rtc_f={registry}/$__sleap_rtc_f
  IFS='\t' read -r __sleap_rtc_pid __sleap_rtc_t __sleap_rtc_since __sleap_rtc_dir < "$__sleap_rtc_f"
  if kill -0 "$__sleap_rtc_pid" 2>/dev/null &&
    ps -p "$__sleap_rtc_pid" -o args= 2>/dev/null | grep -q 'sleap[-_]rtc'; then
    __sleap_rtc_stop='{_stop_hint()}'
    if [ "$__sleap_rtc_dir" != {default_dir} ]; then
      __sleap_rtc_stop="$__sleap_rtc_stop --data-dir '$__sleap_rtc_dir'"
    fi
    printf '\\033[32m●\\033[0m sleap-connect worker is running (pid %s, since %s)\\n' \\
      "$__sleap_rtc_pid" "$__sleap_rtc_since"
    printf '  status: sleap-rtc status · stop: %s\\n' "$__sleap_rtc_stop"
  else
    command rm -f "$__sleap_rtc_f"
  fi
done
unset __sleap_rtc_f __sleap_rtc_pid __sleap_rtc_t __sleap_rtc_since __sleap_rtc_dir __sleap_rtc_stop
"""


def _notice_fish() -> str:
    registry = _fish_quote(str(daemon.registry_dir()))
    default_dir = _fish_quote(str(_default_data_dir()))
    return f"""\
# sleap-connect: new-terminal reminder while a worker is running.
# Written by sleap-rtc, which rewrites it on every 'sleap-rtc serve'.
# Remove the reminder with: sleap-rtc shell-hook uninstall
if status is-interactive
    for __sleap_rtc_f in {registry}/*.run
        set -l __sleap_rtc_parts (string split \\t -- (cat $__sleap_rtc_f))
        set -l __sleap_rtc_pid $__sleap_rtc_parts[1]
        if kill -0 $__sleap_rtc_pid 2>/dev/null; and ps -p $__sleap_rtc_pid -o args= 2>/dev/null | string match -q -r 'sleap[-_]rtc'
            set -l __sleap_rtc_stop '{_stop_hint()}'
            if test "$__sleap_rtc_parts[4]" != {default_dir}
                set __sleap_rtc_stop "$__sleap_rtc_stop --data-dir '$__sleap_rtc_parts[4]'"
            end
            printf '\\033[32m●\\033[0m sleap-connect worker is running (pid %s, since %s)\\n' $__sleap_rtc_pid $__sleap_rtc_parts[3]
            printf '  status: sleap-rtc status · stop: %s\\n' $__sleap_rtc_stop
        else
            command rm -f $__sleap_rtc_f
        end
    end
end
"""


def _notice_ps1() -> str:
    registry = _ps_quote(str(daemon.registry_dir()))
    default_dir = _ps_quote(str(_default_data_dir()))
    return f"""\
# sleap-connect: new-terminal reminder while a worker is running.
# Written by sleap-rtc, which rewrites it on every 'sleap-rtc serve'.
# Remove the reminder with: sleap-rtc shell-hook uninstall
# Skipped when output is redirected (a script running 'powershell -Command').
if (-not [Console]::IsOutputRedirected) {{
Get-ChildItem -Path {registry} -Filter *.run -ErrorAction SilentlyContinue | ForEach-Object {{
    $sleapRtcParts = (Get-Content -Raw $_.FullName).TrimEnd("`r", "`n") -split "`t"
    $sleapRtcProc = Get-Process -Id ([int]$sleapRtcParts[0]) -ErrorAction SilentlyContinue
    if ($sleapRtcProc -and ($sleapRtcProc.ProcessName -match 'python|sleap')) {{
        $sleapRtcStop = '{_stop_hint()}'
        if ($sleapRtcParts[3] -ne {default_dir}) {{
            $sleapRtcStop += " --data-dir '$($sleapRtcParts[3])'"
        }}
        Write-Host -NoNewline -ForegroundColor Green "●"
        Write-Host " sleap-connect worker is running (pid $($sleapRtcParts[0]), since $($sleapRtcParts[2]))"
        Write-Host "  status: sleap-rtc status · stop: $sleapRtcStop"
    }} else {{
        Remove-Item -Force $_.FullName -ErrorAction SilentlyContinue
    }}
}}
}}
"""


def _default_data_dir() -> Path:
    from sleap_rtc.protocol_v1.runner import DEFAULT_DATA_DIR

    return DEFAULT_DATA_DIR.expanduser().resolve()


def write_notice_scripts() -> None:
    """(Re)write every shell's notice script with this version's reminder."""
    folder = shell_dir()
    folder.mkdir(parents=True, exist_ok=True)
    notice_path("bash").write_text(_notice_sh(), encoding="utf-8", newline="\n")
    notice_path("fish").write_text(_notice_fish(), encoding="utf-8", newline="\n")
    # BOM: Windows PowerShell 5.1 reads a BOM-less script as ANSI, mangling
    # the "●" and "·".
    notice_path("powershell").write_text(_notice_ps1(), encoding="utf-8-sig")


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def has_block(path: Path) -> bool:
    """Whether `path` already contains the reminder block."""
    return BEGIN_MARKER in _read(path)


@dataclass
class InstallResult:
    """What `install` did."""

    added: List[Path]
    already: List[Path]


def install(shell: str) -> InstallResult:
    """Add the reminder block to `shell`'s startup file(s).

    Safe to repeat: a file that already has the block is left alone. Also
    clears an earlier `uninstall`'s "don't re-add" note.
    """
    write_notice_scripts()
    _removed_marker().unlink(missing_ok=True)
    result = InstallResult(added=[], already=[])
    for path in startup_files(shell):
        text = _read(path)
        if BEGIN_MARKER in text:
            result.already.append(path)
            continue
        if text and not text.endswith("\n"):
            text += "\n"
        if text:
            text += "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Plain write, not write-and-rename: many startup files are
        # symlinks into a dotfiles repo, which a rename would replace.
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text + _block(shell))
        result.added.append(path)
    return result


def candidate_files() -> List[Path]:
    """Every startup file the block could have been added to."""
    paths: List[Path] = []
    shells = ["powershell"] if sys.platform == "win32" else ["zsh", "bash", "fish"]
    for shell in shells:
        paths.extend(startup_files(shell))
    if sys.platform != "win32":
        # Both bash files, whichever this OS would pick.
        home = Path.home()
        paths.extend(
            p for p in (home / ".bashrc", home / ".bash_profile") if p not in paths
        )
    return paths


def uninstall() -> List[Path]:
    """Remove the reminder block everywhere; returns the files changed.

    Also notes that it was removed, so `serve` doesn't add it back.
    """
    changed = []
    for path in candidate_files():
        text = _read(path)
        if BEGIN_MARKER not in text:
            continue
        new = _BLOCK_RE.sub("\n", text)
        new = new.strip("\n") + "\n" if new.strip() else ""
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(new)
        changed.append(path)
    shell_dir().mkdir(parents=True, exist_ok=True)
    _removed_marker().touch()
    return changed


def installed_files() -> List[Path]:
    """The startup files that currently have the reminder block."""
    return [p for p in candidate_files() if has_block(p)]


def display(path: Path) -> str:
    """`path` with the home folder shown as `~`."""
    try:
        return "~/" + path.relative_to(Path.home()).as_posix()
    except ValueError:
        return str(path)


def auto_install(interactive: bool) -> Optional[str]:
    """Add the reminder the first time a worker starts in the background.

    Returns a line to show the user, or None if nothing was done (not at a
    terminal, opted out with $SLEAP_RTC_NO_SHELL_HOOK or an earlier
    `shell-hook uninstall`, or already added).
    """
    if not interactive or os.environ.get(OPT_OUT_ENV) or _removed_marker().exists():
        return None
    shell = detect_shell()
    if shell is None:
        return (
            "Tip: for a reminder in new terminals while the worker runs, see "
            "'sleap-rtc shell-hook install --help'."
        )
    try:
        result = install(shell)
    except OSError as e:
        return f"Couldn't add a new-terminal reminder: {e}"
    if not result.added:
        return None
    files = ", ".join(display(p) for p in result.added)
    return (
        f"Added a new-terminal reminder to {files} "
        "(remove: sleap-rtc shell-hook uninstall)."
    )
