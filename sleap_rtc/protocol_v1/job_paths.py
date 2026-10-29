"""Which paths a submitted job may use (stage 2 of folder access).

A paired device's job may only name files inside the worker's shared folders
or the worker's own `job-runs/` folder (where it keeps trained models), and
may only write inside them. Checked when the job is submitted, so a job that
reaches outside is refused before anything is queued.

Not checked: media an `.slp` refers to (videos, image folders). sleap-nn
follows those wherever they are — the file's author chose them.
"""

import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import yaml

from sleap_rtc.jobs.spec import TrackJobSpec, TrainJobSpec
from sleap_rtc.protocol_v1.errors import JOB_PATH_NOT_SHARED, ProtocolError


def _resolve(path: str) -> Path:
    return Path(os.path.expanduser(str(path))).resolve()


def _inside(path: Path, roots: Sequence[Path]) -> bool:
    return any(path == root or path.is_relative_to(root) for root in roots)


def _unquote(name: str) -> str:
    """Undo the Hydra quoting a client may send (`'my run'`)."""
    name = name.strip()
    if len(name) >= 2 and name[0] == name[-1] and name[0] in "'\"":
        quote = name[0]
        name = name[1:-1].replace("\\" + quote, quote)
    return name


def run_name_escapes(name: str) -> bool:
    """Whether a run name would put checkpoints outside the checkpoint folder.

    sleap-nn writes to `<ckpt_dir>/<run_name>`, so `../../elsewhere` or an
    absolute path would leave it. Subfolders (`a/b`) stay inside and are fine.
    """
    base = "/ckpt"
    joined = os.path.normpath(os.path.join(base, _unquote(name).replace("\\", "/")))
    return not joined.startswith(base + "/")


class JobPathPolicy:
    """Checks a job spec's paths against the folders it may use.

    Args:
        shared_roots: Returns the shared folders' paths (read live, so a
            folder added or removed applies to the next submit).
        runs_dir: The worker's own job-runs folder (always allowed).
        data_dir: The worker's data dir. Never allowed apart from
            `runs_dir`, even inside a shared folder (it holds the worker's
            identity key and paired-device list).
    """

    def __init__(self, shared_roots, runs_dir: Path, data_dir: Optional[Path] = None):
        self._shared_roots = shared_roots
        self._runs_dir = Path(runs_dir).resolve()
        self._data_dir = Path(data_dir).resolve() if data_dir else None

    def roots(self) -> List[Path]:
        return [_resolve(p) for p in self._shared_roots()] + [self._runs_dir]

    def check(self, spec) -> None:
        """Raise `job.path_not_shared` for the first path outside the allowed folders."""
        roots = self.roots()
        if isinstance(spec, TrainJobSpec):
            self._check_train(spec, roots)
        elif isinstance(spec, TrackJobSpec):
            self._check_track(spec, roots)

    # --- per spec type -------------------------------------------------

    def _check_train(self, spec: TrainJobSpec, roots: List[Path]) -> None:
        names = [spec.run_name, *(spec.model_types or [])]
        for name in names:
            if name and run_name_escapes(name):
                raise ProtocolError(
                    JOB_PATH_NOT_SHARED,
                    f'Run name "{name}" would put the model outside its job folder.',
                    {"field": "run_name", "path": name},
                )

        uploads_labels = bool(spec.labels_content)
        reads = [
            # Checked even for an upload: one that names a file on the worker
            # is a corrected copy of it (the Connect window re-pointing its
            # videos), so that file must be shared like any other.
            ("labels_path", spec.labels_path),
            ("val_labels_path", spec.val_labels_path),
            ("resume_ckpt_path", spec.resume_ckpt_path),
            *(("config_path", p) for p in spec.config_paths or []),
        ]
        for field, path in reads:
            if path:
                self._require(field, path, roots)

        overrides_train = bool(spec.labels_path or uploads_labels)
        for text in self._config_texts(spec):
            for field, path in config_read_paths(
                text,
                overrides_train=overrides_train,
                overrides_val=bool(spec.val_labels_path),
            ):
                self._require(field, path, roots)

    def _check_track(self, spec: TrackJobSpec, roots: List[Path]) -> None:
        self._require("data_path", spec.data_path, roots)
        for path in spec.model_paths or []:
            self._require("model_paths", path, roots)
        if spec.output_path:
            self._require("output_path", spec.output_path, roots, write=True)

    # --- helpers --------------------------------------------------------

    @staticmethod
    def _config_texts(spec: TrainJobSpec) -> Iterable[str]:
        """Config YAML the job will run with: inline contents, then config files."""
        for content in spec.config_contents or []:
            for old, new in (spec.path_mappings or {}).items():
                content = content.replace(old, new)
            yield content
        if spec.config_content:
            yield spec.config_content
        for path in spec.config_paths or []:
            # Already checked to be inside an allowed folder.
            try:
                yield Path(path).read_text()
            except OSError:
                continue  # A missing file fails the job later, with its own message.

    def _require(
        self, field: str, path: str, roots: List[Path], write: bool = False
    ) -> None:
        resolved = _resolve(path)
        in_data_dir = self._data_dir is not None and _inside(resolved, [self._data_dir])
        if _inside(resolved, [self._runs_dir]) or (
            _inside(resolved, roots) and not in_data_dir
        ):
            return
        shared = [str(r) for r in roots if r != self._runs_dir]
        folder = resolved.parent if resolved.suffix else resolved
        where = ", ".join(shared) if shared else "none"
        verb = "write to" if write else "read"
        raise ProtocolError(
            JOB_PATH_NOT_SHARED,
            f"This job would {verb} {path} ({field}), which is outside the worker's "
            f"shared folders ({where}). To allow it, run on the worker: "
            f'sleap-rtc add-folder "{folder}"',
            {"field": field, "path": path, "shared_folders": shared},
        )


def _as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, str) and v]
    return [value] if isinstance(value, str) and value else []


def config_read_paths(text: str, *, overrides_train: bool, overrides_val: bool):
    """`(field, path)` for every file a sleap-nn training config reads.

    Labels paths the job overrides on the command line don't count. Configs
    that don't parse are skipped (sleap-nn will fail them itself).
    """
    try:
        cfg = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return []
    if not isinstance(cfg, dict):
        return []
    data = cfg.get("data_config") or {}
    model = cfg.get("model_config") or {}
    out = []
    if not overrides_train:
        out += [
            ("train_labels_path", p) for p in _as_list(data.get("train_labels_path"))
        ]
    if not overrides_val:
        out += [("val_labels_path", p) for p in _as_list(data.get("val_labels_path"))]
    out += [("test_file_path", p) for p in _as_list(data.get("test_file_path"))]
    for key in ("pretrained_backbone_weights", "pretrained_head_weights"):
        out += [(key, p) for p in _as_list(model.get(key))]
    return out


def config_sets_cache_dir(text: str) -> bool:
    """Whether a training config sets `data_config.cache_img_path` (writes images)."""
    try:
        cfg = yaml.safe_load(text) or {}
    except yaml.YAMLError:
        return False
    data = (cfg.get("data_config") or {}) if isinstance(cfg, dict) else {}
    return bool(isinstance(data, dict) and data.get("cache_img_path"))
