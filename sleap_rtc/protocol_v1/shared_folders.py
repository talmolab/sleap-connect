"""The folders this worker shares with its paired devices.

A shared folder is one paired devices may browse (`fs.list`/`fs.stat`/
`fs.read`). The folder `sleap-rtc serve` starts in is offered for sharing the
first time (a y/N prompt), and more can be added with `sleap-rtc add-folder`;
either way the choice is remembered here, in the worker's data dir, so the
next `serve` doesn't ask again.

Stored as JSON and re-read whenever the file changes (one `stat` per check
otherwise), so `add-folder`/`remove-folder` from another terminal apply to a
running worker without a restart — the same pattern as the trust store.

The worker's own data dir is never shared, even when it sits inside a shared
folder (it holds this worker's identity key and the paired-device list).
"""

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

from sleap_rtc.config import MountConfig
from sleap_rtc.worker.file_manager import FileManager


@dataclass(frozen=True)
class SharedFolder:
    """One shared folder.

    Attributes:
        path: Absolute, resolved path.
        label: Short display name (unique among shared folders).
        added_at: Unix time it was shared.
    """

    path: str
    label: str
    added_at: float


def resolve_folder(raw: Union[str, Path], base: Optional[Path] = None) -> Path:
    """`raw` as an absolute, resolved path; relative paths are against `base` (default: cwd)."""
    path = Path(os.path.expanduser(str(raw)))
    if not path.is_absolute():
        path = (base or Path.cwd()) / path
    return path.resolve()


_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")


def parse_mount(raw: str) -> Tuple[str, Optional[str]]:
    """Split a `--mount` value, "PATH" or "PATH:LABEL", into (path, label or None).

    A Windows drive prefix (`C:\\…`) is part of the path, not a separator.
    """
    head = raw[:2] if _WINDOWS_DRIVE.match(raw) else ""
    path, sep, label = raw[len(head) :].rpartition(":")
    if not sep or not path or "/" in label or "\\" in label:
        return raw, None
    return head + path, label or None


def is_broad(path: Path) -> bool:
    """Whether sharing `path` would share the whole home folder or more."""
    home = Path.home().resolve()
    return path == Path(path.anchor) or path == home or home.is_relative_to(path)


def _within(path: Path, folder: Path) -> bool:
    try:
        path.relative_to(folder)
        return True
    except ValueError:
        return False


class SharedFolders:
    """The shared-folder list, persisted as JSON under the worker's data dir."""

    def __init__(self, path: Union[str, Path]):
        """Open the list at `path` (created on first `add`)."""
        self._path = Path(path)
        self._cached: List[SharedFolder] = []
        self._cached_at: Optional[Tuple[int, int, int]] = None

    def list(self) -> List[SharedFolder]:
        """Every shared folder, oldest first. Re-read only if the file changed."""
        try:
            st = os.stat(self._path)
        except OSError:
            self._cached, self._cached_at = [], None
            return []
        key = (st.st_mtime_ns, st.st_size, st.st_ino)
        if key != self._cached_at:
            try:
                data = json.loads(self._path.read_text())
                self._cached = [SharedFolder(**f) for f in data.get("folders", [])]
            except (OSError, ValueError, TypeError):
                # Unreadable: share nothing rather than guess.
                self._cached = []
            self._cached_at = key
        return list(self._cached)

    def find(self, path: Path) -> Optional[SharedFolder]:
        """The shared folder containing `path` (resolved), if any."""
        for folder in self.list():
            if _within(path, Path(folder.path)):
                return folder
        return None

    def add(self, path: Path, label: Optional[str] = None) -> SharedFolder:
        """Share `path` (already resolved). Re-adding a folder keeps it as is."""
        folders = self.list()
        for folder in folders:
            if Path(folder.path) == path:
                return folder
        taken = {f.label for f in folders}
        base = label or path.name or str(path)
        label, n = base, 2
        while label in taken:
            label, n = f"{base} ({n})", n + 1
        folder = SharedFolder(path=str(path), label=label, added_at=time.time())
        self._write([*folders, folder])
        return folder

    def remove(
        self, target: str, base: Optional[Path] = None
    ) -> Optional[SharedFolder]:
        """Unshare the folder whose label is `target`, or whose path it names."""
        folders = self.list()
        match = next((f for f in folders if f.label == target), None)
        if match is None:
            path = resolve_folder(target, base)
            match = next((f for f in folders if Path(f.path) == path), None)
        if match is not None:
            self._write([f for f in folders if f is not match])
        return match

    def _write(self, folders: List[SharedFolder]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"folders": [asdict(f) for f in folders]}, indent=2))
        os.replace(tmp, self._path)


class SharedFolderFileManager(FileManager):
    """A `FileManager` whose mounts are the live shared-folder list.

    Also knows the folder `serve` started in (reported by `fs.mounts`) and
    never allows anything under the worker's own data dir.
    """

    def __init__(
        self,
        shared: SharedFolders,
        data_dir: Path,
        serve_folder: Optional[Path] = None,
    ):
        """Wrap `shared` for the `fs.*` methods."""
        self._shared = shared
        self._data_dir = Path(data_dir).resolve()
        self._serve_folder = serve_folder
        super().__init__()

    @property
    def mounts(self) -> list:  # type: ignore[override]
        """The shared folders, as `MountConfig`s, read live."""
        return [MountConfig(path=f.path, label=f.label) for f in self._shared.list()]

    @mounts.setter
    def mounts(self, _value) -> None:
        # FileManager.__init__ assigns this; the store is the only source.
        pass

    @property
    def serve_folder(self) -> Optional[str]:
        """The folder `serve` started in, if it's shared (directly or inside one)."""
        if self._serve_folder is None or self._shared.find(self._serve_folder) is None:
            return None
        return str(self._serve_folder)

    def stat_path(self, path: str) -> dict:
        """`fs.stat`: full metadata in a shared folder; outside, existence only.

        A job can read media an `.slp` points at wherever it is, so the app
        needs to know such a file exists — but not its size, contents or
        neighbors. Answers `{path, type, shared: false}`, or not-found. The
        worker's data dir is still refused outright.
        """
        try:
            resolved = Path(path).resolve()
        except (OSError, ValueError):
            return super().stat_path(path)
        if self._is_path_allowed(resolved) or _within(resolved, self._data_dir):
            return super().stat_path(path)
        if not resolved.exists():
            return {
                "path": path,
                "error": "Path does not exist",
                "error_code": "PATH_NOT_FOUND",
            }
        return {
            "path": str(resolved),
            "type": "directory" if resolved.is_dir() else "file",
            "shared": False,
        }

    def _is_path_allowed(self, path: Path) -> bool:
        try:
            resolved = path.resolve()
        except (OSError, ValueError):
            return False
        if _within(resolved, self._data_dir):
            return False
        return super()._is_path_allowed(resolved)
