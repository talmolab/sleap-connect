"""Where a job's outputs go — laid out like SLEAP (folder access, stage 3).

SLEAP saves next to the labels file it trained on:

    <labels folder>/models/<run name>/          training_config.yaml, best.ckpt, …
    <labels folder>/predictions/<source>.<YYMMDD_HHMMSS>.predictions.slp

A job that trains on a labels file already on the worker gets the same
layout. A job whose labels were uploaded (a snapshot, kept in the job's own
folder) keeps its outputs in that job folder, as before.

Several runs share one `models/` folder, so the worker names each run itself
(never two jobs in one folder) and records which folder is whose
(`outputs.json` in the job folder) instead of guessing by modification time.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from sleap_rtc.jobs.spec import TrainJobSpec

OUTPUTS_RECORD = "outputs.json"


def timestamp(now: Optional[datetime] = None) -> str:
    """SLEAP's run/prediction timestamp, `YYMMDD_HHMMSS`."""
    return (now or datetime.now()).strftime("%y%m%d_%H%M%S")


def train_output_root(
    spec: TrainJobSpec,
    original_labels_path: Optional[str] = None,
    may_write: Optional[Callable[[str], bool]] = None,
) -> Optional[Path]:
    """The folder a train job's `models/` (and predictions) go in.

    A file already on the worker: its folder, like SLEAP. Uploaded labels (a
    snapshot): the client's `output_dir`, created if needed; else, when the
    upload is a corrected copy of a file on the worker (`original_labels_path`
    — e.g. the Connect window re-pointing its videos), that file's folder, if
    `may_write` allows it. None — keep outputs in the job folder — otherwise,
    or for a folder the worker can't write to.
    """
    if spec.labels_content and not spec.output_dir and original_labels_path:
        folder = Path(original_labels_path).expanduser().resolve().parent
        if may_write is not None and not may_write(str(folder)):
            return None
    elif spec.labels_content:
        if not spec.output_dir:
            return None
        folder = Path(spec.output_dir).expanduser().resolve()
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
    elif spec.labels_path:
        folder = Path(spec.labels_path).expanduser().resolve().parent
    else:
        return None
    return folder if folder.is_dir() and os.access(folder, os.W_OK) else None


def _unquote(name: str) -> tuple:
    """(bare name, quote char or "") for a possibly Hydra-quoted run name."""
    name = name.strip()
    if len(name) >= 2 and name[0] == name[-1] and name[0] in "'\"":
        return name[1:-1].replace("\\" + name[0], name[0]), name[0]
    return name, ""


def run_name_for(
    spec: TrainJobSpec, ckpt_dir: Path, now: Optional[datetime] = None
) -> str:
    """The run name to train under: the client's, else SLEAP's `<timestamp>.<model>`.

    Made unique within `ckpt_dir` (`_2`, `_3`, …) so a reused name never
    trains into another run's folder. Keeps the client's Hydra quoting.
    """
    if spec.run_name:
        base, quote = _unquote(spec.run_name)
    else:
        model = spec.model_types[0] if spec.model_types else "model"
        base, quote = f"{timestamp(now)}.{model}", ""
    name, n = base, 2
    while (ckpt_dir / name).exists():
        name, n = f"{base}_{n}", n + 1
    if quote:
        return f"{quote}{name.replace(quote, chr(92) + quote)}{quote}"
    return name


def bare_run_name(run_name: str) -> str:
    """The folder name sleap-nn creates for `run_name` (quotes removed)."""
    return _unquote(run_name)[0]


def predictions_path(
    data_path: str,
    now: Optional[datetime] = None,
    output_dir: Optional[str] = None,
    name: Optional[str] = None,
) -> Path:
    """Where a track job's predictions go, as SLEAP names them.

    For a labels file: `<its folder>/predictions/<name>.<timestamp>.predictions.slp`.
    For a video: next to it, `<video>.<timestamp>.predictions.slp` (SLEAP does
    the same when there's no labels file).
    """
    data = Path(data_path).expanduser()
    if output_dir:
        # A snapshot run's chosen folder; named after the project, not the
        # snapshot file (`labels.slp`).
        stem = name or data.name
        return (
            Path(output_dir).expanduser()
            / "predictions"
            / f"{stem}.{timestamp(now)}.predictions.slp"
        )
    name = f"{data.name}.{timestamp(now)}.predictions.slp"
    if data.suffix.lower() == ".slp":
        return data.parent / "predictions" / name
    return data.parent / name


def write_outputs_record(job_dir: Path, model_dir: Path) -> None:
    """Remember which folder this job's model trains into."""
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / OUTPUTS_RECORD).write_text(json.dumps({"model_dir": str(model_dir)}))


def read_model_dir(job_dir: Path) -> Optional[Path]:
    """The model folder `write_outputs_record` saved, if it finished training."""
    try:
        record = json.loads((job_dir / OUTPUTS_RECORD).read_text())
    except (OSError, ValueError):
        return None
    model_dir = Path(record.get("model_dir") or "")
    return model_dir if (model_dir / "training_config.yaml").exists() else None
