"""The `worker.info` method: what this machine is, and whether it's busy.

Clients show this on a worker's card (GPU, memory, CUDA, sleap-nn version,
Idle/Busy). Hardware is detected once, on the first call, in a thread:
detection shells out and can take seconds (so it must not delay startup or
block the event loop), and none of it changes while `serve` runs.

Detection prefers the command-line tools over imports: on a GPU box sleap-nn
is usually its own `uv tool`, separate from this worker's environment, so
neither `sleap_nn` nor `torch` is necessarily importable here.
"""

import asyncio
import importlib.metadata
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from sleap_rtc.jobs.queue import JobQueue
from sleap_rtc.protocol_v1.server import Connection, ProtocolV1Server

_TIMEOUT_SECS = 10


def _run(cmd: List[str]) -> Optional[str]:
    """`cmd`'s stdout, or None if it is missing, fails or times out."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_TIMEOUT_SECS, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout


def _detect_gpus() -> Dict[str, Any]:
    """GPU model, per-GPU memory, count and CUDA version."""
    rows = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if rows:
        gpus = [r.split(",") for r in rows.strip().splitlines() if r.strip()]
        header = _run(["nvidia-smi"]) or ""
        cuda = re.search(r"CUDA Version:\s*([\d.]+)", header)
        return {
            "gpu_model": gpus[0][0].strip(),
            "gpu_memory_mb": int(float(gpus[0][1])) if len(gpus[0]) > 1 else 0,
            "gpu_count": len(gpus),
            "cuda_version": cuda.group(1) if cuda else "unknown",
        }
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "gpu_model": props.name,
                "gpu_memory_mb": props.total_memory // (1024 * 1024),
                "gpu_count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda or "unknown",
            }
    except Exception:
        pass
    return {
        "gpu_model": "CPU",
        "gpu_memory_mb": 0,
        "gpu_count": 0,
        "cuda_version": "N/A",
    }


def _detect_sleap_nn_version() -> str:
    out = _run(["sleap-nn", "--version"])
    if out:
        match = re.search(r"(\d+\.\d+[\w.+-]*)", out)
        if match:
            return match.group(1)
    try:
        return importlib.metadata.version("sleap-nn")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _detect_hardware() -> Dict[str, Any]:
    """Everything `worker.info` reports that doesn't change while serving."""
    return {**_detect_gpus(), "sleap_nn_version": _detect_sleap_nn_version()}


class WorkerInfoMethods:
    """Registers `worker.info` on a server."""

    def __init__(self, server: ProtocolV1Server, queue: JobQueue):
        """Register the method (hardware is detected on first use).

        Args:
            server: The `ProtocolV1Server` to register on.
            queue: The worker's job queue; a held slot means the worker is busy.
        """
        self._queue = queue
        self._hardware: Optional[Dict[str, Any]] = None
        self._lock = asyncio.Lock()
        server.register("worker.info", self.info)

    async def info(self, params: dict, conn: Connection) -> dict:
        """Handle `worker.info`."""
        async with self._lock:
            if self._hardware is None:
                self._hardware = await asyncio.to_thread(_detect_hardware)
        return {**self._hardware, "busy": self._queue.running > 0}
