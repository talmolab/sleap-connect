"""Job execution orchestration for training and inference workflows.

This module handles execution of SLEAP training and inference jobs, including
script parsing, process management, log streaming, and progress reporting.
"""

import asyncio
import json
import logging
import os
import re
import signal
import shutil
import stat
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from aiortc import RTCDataChannel

# Watchdog polling interval (seconds).  Tests patch this to 0 so they don't
# wait 30 real seconds for the watchdog cycle to fire.
_WATCHDOG_INTERVAL_SECS = 30

# Grace period after SIGTERM before escalating to SIGKILL (cancel_running_job).
# Short enough for an interactive Cancel button; long enough for the sleap-nn
# subprocess to flush its progress bar and exit cleanly on most hardware.
_CANCEL_GRACE_SECS = 5

from sleap_connect.worker.progress_reporter import ProgressReporter

if TYPE_CHECKING:
    from sleap_connect.worker.capabilities import WorkerCapabilities

# Set chunk separator for log streaming
SEP = re.compile(rb"[\r\n]")


def _read_rss_mb(pid: int) -> float | None:
    """Read resident set size in MB for a process from /proc/<pid>/status.

    Returns None if the file is unreadable (process exited, non-Linux, etc.).
    """
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def _read_private_dirty_mb(pid: int) -> float | None:
    """Read Private_Dirty pages in MB from /proc/<pid>/smaps_rollup.

    Private_Dirty is the memory that is truly private to this process —
    heap allocations, written stack pages, modified data.  Unlike RSS it
    does NOT count read-only shared-library pages that are shared with
    sibling processes, so summing this across the process tree gives a
    much more accurate picture of actual physical memory consumed.

    Returns None if smaps_rollup is unavailable (kernel < 4.14, non-Linux).
    """
    try:
        with open(f"/proc/{pid}/smaps_rollup") as f:
            for line in f:
                if line.startswith("Private_Dirty:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def _read_cgroup_memory_mb() -> float | None:
    """Read actual container memory usage from the cgroup memory controller.

    This is the number the OOM killer uses — physical pages counted once
    regardless of how many processes share them.  Tries cgroup v2 first
    (/sys/fs/cgroup/memory.current), then falls back to cgroup v1
    (/sys/fs/cgroup/memory/memory.usage_in_bytes).

    Returns None if neither path is readable (non-Linux, no cgroup mount).
    """
    for path in (
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    ):
        try:
            with open(path) as f:
                return int(f.read().strip()) / (1024 * 1024)
        except OSError:
            pass
    return None


def _get_child_pids(pid: int) -> list[int]:
    """Get direct child PIDs from /proc/{pid}/task/*/children."""
    children = []
    try:
        for tid_dir in Path(f"/proc/{pid}/task").iterdir():
            try:
                with open(tid_dir / "children") as f:
                    for cpid_str in f.read().split():
                        children.append(int(cpid_str))
            except OSError:
                pass
    except OSError:
        pass
    return children


def _iter_process_tree(root_pid: int) -> set[int]:
    """Return the set of root_pid and all its descendant PIDs."""
    pids: set[int] = set()
    frontier = [root_pid]
    while frontier:
        pid = frontier.pop()
        if pid in pids:
            continue
        pids.add(pid)
        frontier.extend(_get_child_pids(pid))
    return pids


def _read_process_tree_rss_mb(root_pid: int) -> tuple[float, int]:
    """Sum RSS in MB for root_pid and all its descendants.

    Returns:
        (total_mb, process_count) — total RSS and number of live processes found.
    """
    total = 0.0
    count = 0
    for pid in _iter_process_tree(root_pid):
        rss = _read_rss_mb(pid)
        if rss is not None:
            total += rss
            count += 1
    return total, count


async def _query_vram() -> str | None:
    """Query per-GPU VRAM usage via nvidia-smi.

    Returns a formatted string like "GPU0: 1,234/16,384 MB, GPU1: 1,234/16,384 MB",
    or None if nvidia-smi is unavailable or times out.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        parts = []
        for line in stdout.decode().strip().splitlines():
            cols = [c.strip() for c in line.split(",")]
            if len(cols) == 3:
                parts.append(f"GPU{cols[0]}: {int(cols[1]):,}/{int(cols[2]):,} MB")
        return ", ".join(parts) if parts else None
    except Exception:
        return None


class JobExecutor:
    """Executes training and inference jobs with progress monitoring.

    This class orchestrates the execution of SLEAP training and inference
    workflows, handling script parsing, process execution, log streaming,
    and integration with progress reporting systems.

    Attributes:
        worker: Reference to parent RTCWorkerClient for accessing shared state.
        capabilities: WorkerCapabilities instance for job compatibility.
        unzipped_dir: Working directory for current job.
        output_dir: Output directory for job results.
        package_type: Type of package ("train" or "track").
    """

    def __init__(
        self,
        worker,
        capabilities: "WorkerCapabilities",
    ):
        """Initialize job executor.

        Args:
            worker: Parent RTCWorkerClient instance.
            capabilities: WorkerCapabilities for job compatibility checking.
        """
        self.worker = worker
        self.capabilities = capabilities

        # Job execution state
        self.unzipped_dir = ""
        self.output_dir = ""
        self.package_type = "train"  # Default to training
        self._running_process: asyncio.subprocess.Process | None = None
        self._progress_reporter = None
        self._stop_requested = False  # True when ZMQ "stop" command was sent
        self._cancel_requested = (
            False  # True when ZMQ "cancel" command was sent (skips inference)
        )

    def send_control_message(self, raw_zmq_message: str):
        """Forward a raw ZMQ control message to the training process.

        Forwards the raw ZMQ message to sleap-nn's TrainingControllerZMQ via
        the ProgressReporter's ZMQ PUB socket, identically to local training.

        sleap-nn's TrainingControllerZMQ.on_train_batch_end handles rank
        synchronization via reduce_boolean_decision so all DDP ranks stop
        together cleanly without a SIGINT.

        Args:
            raw_zmq_message: Raw jsonpickle-encoded string from LossViewer
                (e.g., ``{"command": "stop"}``).
        """
        if self._progress_reporter is not None:
            self._progress_reporter.send_control_message(raw_zmq_message)
        else:
            logging.warning(
                "No ProgressReporter available — cannot forward ZMQ control message"
            )

        try:
            payload = json.loads(raw_zmq_message)
            cmd = payload.get("command")
            if cmd == "stop":
                self._stop_requested = True
                logging.info(f"[ZMQ] _stop_requested set to True (command=stop)")
            elif cmd == "cancel":
                self._stop_requested = True
                self._cancel_requested = True
                logging.info(
                    f"[ZMQ] _stop_requested + _cancel_requested set to True (command=cancel)"
                )
        except (json.JSONDecodeError, AttributeError):
            pass

    def stop_running_job(self):
        """Send SIGINT to the running job's process group for graceful stop.

        Kills the entire process group so distributed workers (rank 1, …)
        exit alongside the main process, allowing sleap-nn to save a checkpoint.
        On Windows, falls back to proc.terminate() (no process groups).
        """
        proc = self._running_process
        if proc is not None and proc.returncode is None:
            try:
                if sys.platform == "win32":
                    proc.terminate()
                else:
                    pgid = os.getpgid(proc.pid)
                    logging.info(
                        f"Sending SIGINT to process group {pgid} (graceful stop)"
                    )
                    os.killpg(pgid, signal.SIGINT)
            except ProcessLookupError:
                pass

    def cancel_running_job(self):
        """Send SIGTERM to the running job's process group for immediate cancellation.

        If the process doesn't exit within ``_CANCEL_GRACE_SECS`` seconds,
        escalates to SIGKILL.  On Windows, falls back to proc.kill() (no
        process groups).
        """
        proc = self._running_process
        if proc is not None and proc.returncode is None:
            try:
                if sys.platform == "win32":
                    proc.kill()
                else:
                    pgid = os.getpgid(proc.pid)
                    logging.info(
                        f"Sending SIGTERM to process group {pgid} (hard cancel)"
                    )
                    os.killpg(pgid, signal.SIGTERM)

                    # Escalate to SIGKILL after timeout if process doesn't exit
                    import threading

                    def _escalate_to_sigkill():
                        try:
                            proc.wait(timeout=_CANCEL_GRACE_SECS)
                            logging.info(f"Process {proc.pid} exited after SIGTERM")
                        except Exception:
                            # Still alive after grace period — force kill
                            try:
                                pgid2 = os.getpgid(proc.pid)
                                logging.warning(
                                    f"Process group {pgid2} did not exit after "
                                    f"SIGTERM ({_CANCEL_GRACE_SECS}s), escalating to SIGKILL"
                                )
                                os.killpg(pgid2, signal.SIGKILL)
                            except ProcessLookupError:
                                pass

                    threading.Thread(target=_escalate_to_sigkill, daemon=True).start()
            except ProcessLookupError:
                pass

    def parse_training_script(self, train_script_path: str):
        """Parse train-script.sh and extract sleap-nn train commands.

        Args:
            train_script_path: Path to train-script.sh

        Returns:
            List of config names for training jobs.
        """
        jobs = []
        # Updated pattern to match 'sleap-nn train' and extract --config-name
        # Example: sleap-nn train --config-name centroid.yaml...
        pattern = re.compile(r"^\s*sleap-(?:nn\s+)?train\s+.*--config-name\s+(\S+)")

        with open(train_script_path, "r") as f:
            for line in f:
                match = pattern.match(line)
                if match:
                    config_name = match.group(1)
                    # For sleap-nn, we don't need separate labels file
                    jobs.append(config_name)
        return jobs

    def parse_track_script(self, track_script_path: str):
        """Parse track-script.sh and extract sleap-nn track commands.

        Args:
            track_script_path: Path to track-script.sh

        Returns:
            List of command argument lists.
        """
        commands = []
        pattern = re.compile(r"^\s*sleap-nn\s+track\s+(.+)")

        with open(track_script_path, "r") as f:
            script_content = f.read()

        # Handle multi-line commands with backslashes
        script_content = script_content.replace("\\\n", " ")

        for line in script_content.split("\n"):
            match = pattern.match(line)
            if match:
                # Split arguments while preserving quoted strings
                args_str = match.group(1)
                # Simple split (for more complex parsing, use shlex.split)
                args = ["sleap-nn", "track"] + args_str.split()
                commands.append(args)

        return commands

    async def run_all_training_jobs(
        self, channel: RTCDataChannel, train_script_path: str
    ):
        """Execute training jobs with progress updates.

        Args:
            channel: RTC data channel for sending logs
            train_script_path: Path to training script
        """
        training_jobs = self.parse_training_script(train_script_path)

        # Get job ID and client ID from worker's current_job if available
        job_id = (
            self.worker.current_job.get("job_id") if self.worker.current_job else None
        )
        client_id = (
            self.worker.current_job.get("client_id")
            if self.worker.current_job
            else None
        )

        try:
            for config_name in training_jobs:
                job_name = Path(config_name).stem

                # Send RTC msg over channel to indicate job start.
                logging.info(
                    f"Starting training job: {job_name} with config: {config_name}"
                )
                channel.send(f"TRAIN_JOB_START::{job_name}")

                # Send starting status via peer message
                if job_id and client_id:
                    await self._send_peer_message(
                        client_id,
                        {
                            "app_message_type": "job_status",
                            "job_id": job_id,
                            "status": "starting",
                            "progress": 0.0,
                            "message": f"Initializing training: {job_name}",
                        },
                    )

                # Use sleap-nn train command for newer SLEAP versions
                # Run directly in the extracted directory with config-dir from training script
                cmd = [
                    "sleap-nn",
                    "train",
                    "--config-name",
                    config_name,
                    "--config-dir",
                    ".",
                    "trainer_config.ckpt_dir=models",
                    f"trainer_config.run_name={job_name}",
                    "trainer_config.zmq.controller_port=9000",
                    "trainer_config.zmq.publish_port=9001",
                    # macOS compatibility: disable multiprocessing
                    "trainer_config.train_data_loader.num_workers=0",
                    "trainer_config.val_data_loader.num_workers=0",
                ]
                logging.info(f"[RUNNING] {' '.join(cmd)} (cwd={self.unzipped_dir})")

                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=self.unzipped_dir,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )

                assert process.stdout is not None
                logging.info(f"Process started with PID: {process.pid}")

                async def stream_logs(read_size=512, max_flush=64 * 1024):
                    buf = b""
                    pending_cr = ""
                    try:
                        while True:
                            chunk = await process.stdout.read(read_size)
                            if not chunk:
                                # process ended; flush any remaining text
                                if pending_cr:
                                    channel.send(pending_cr + "\n")
                                if buf:
                                    channel.send(buf.decode(errors="replace") + "\n")
                                break

                            buf += chunk

                            while True:
                                m = SEP.search(buf)
                                if not m:
                                    # If tqdm keeps extending one long line, hold as
                                    # pending and clear the buffer
                                    if len(buf) > max_flush:
                                        pending_cr = buf.decode(errors="replace")
                                        buf = b""
                                    break

                                sep = buf[m.start() : m.end()]
                                payload = buf[: m.start()]
                                buf = buf[m.end() :]

                                text = payload.decode(errors="replace")
                                if not text:
                                    continue

                                if sep == b"\n":
                                    # \n supersedes any pending \r line (tqdm
                                    # emits \r...\n for each batch; discard the
                                    # \r version to avoid duplicate sends)
                                    pending_cr = ""
                                    channel.send(text + "\n")
                                else:  # sep == b'\r' — hold, keep only latest
                                    pending_cr = text

                    except Exception as e:
                        logging.exception("stream_logs failed: %s", e)
                        try:
                            channel.send(f"[log-stream error] {e}\n")
                        except Exception:
                            pass

                # Track start time for duration calculation
                start_time = time.time()

                await stream_logs()
                logging.info("Waiting for process to complete...")
                await process.wait()
                logging.info(
                    f"Process completed with return code: {process.returncode}"
                )

                # Calculate training duration
                training_duration = (time.time() - start_time) / 60

                if process.returncode == 0:
                    logging.info(f"[DONE] Job {job_name} completed successfully.")
                    if channel.readyState == "open":
                        channel.send(f"TRAIN_JOB_END::{job_name}")

                    # Send completion status via peer message
                    if job_id and client_id:
                        await self._send_peer_message(
                            client_id,
                            {
                                "app_message_type": "job_status",
                                "job_id": job_id,
                                "status": "running",
                                "progress": 1.0,
                                "message": f"Training completed: {job_name}",
                            },
                        )
                else:
                    logging.warning(
                        f"[FAILED] Job {job_name} exited with code {process.returncode}."
                    )
                    if channel.readyState == "open":
                        channel.send(
                            f"TRAIN_JOB_ERROR::{job_name}::{process.returncode}"
                        )

                    # Send failure status via peer message
                    if job_id and client_id:
                        await self._send_peer_message(
                            client_id,
                            {
                                "app_message_type": "job_failed",
                                "job_id": job_id,
                                "status": "failed",
                                "error": {
                                    "code": "TRAINING_FAILED",
                                    "message": f"Training job {job_name} failed with exit code {process.returncode}",
                                    "recoverable": False,
                                },
                            },
                        )

            # All jobs completed - send final completion message
            channel.send("TRAINING_JOBS_DONE")

            # Send job completion message via peer message
            if job_id and client_id:
                await self._send_peer_message(
                    client_id,
                    {
                        "app_message_type": "job_complete",
                        "job_id": job_id,
                        "status": "completed",
                        "result": {
                            "training_duration_minutes": training_duration,
                            "total_jobs": len(training_jobs),
                        },
                        "transfer_method": "webrtc_datachannel",
                        "ready_for_download": True,
                    },
                )

        except Exception as e:
            # Handle any unexpected errors during training
            logging.error(f"Error during training execution: {e}")
            if job_id and client_id:
                await self._send_peer_message(
                    client_id,
                    {
                        "app_message_type": "job_failed",
                        "job_id": job_id,
                        "status": "failed",
                        "error": {
                            "code": "EXECUTION_ERROR",
                            "message": str(e),
                            "recoverable": False,
                        },
                    },
                )
            raise

        finally:
            # Update status back to available when done
            if self.worker.current_job:
                await self.worker.update_status("available")
                self.worker.current_job = None

    async def run_track_workflow(self, channel: RTCDataChannel, track_script_path: str):
        """Execute inference workflow on received track package.

        Args:
            channel: RTC data channel for sending progress
            track_script_path: Path to track-script.sh
        """
        # Get job ID and client ID from worker's current_job if available
        job_id = (
            self.worker.current_job.get("job_id") if self.worker.current_job else None
        )
        client_id = (
            self.worker.current_job.get("client_id")
            if self.worker.current_job
            else None
        )

        try:
            if not Path(track_script_path).exists():
                logging.error("No track-script.sh found in package")
                channel.send("INFERENCE_ERROR::No track script found")

                # Send error via peer message
                if job_id and client_id:
                    await self._send_peer_message(
                        client_id,
                        {
                            "app_message_type": "job_failed",
                            "job_id": job_id,
                            "status": "failed",
                            "error": {
                                "code": "TRACK_SCRIPT_NOT_FOUND",
                                "message": "No track script found in package",
                                "recoverable": False,
                            },
                        },
                    )
                return

            logging.info("Starting inference workflow...")
            channel.send("INFERENCE_START")

            # Send starting status via peer message
            if job_id and client_id:
                await self._send_peer_message(
                    client_id,
                    {
                        "app_message_type": "job_status",
                        "job_id": job_id,
                        "status": "starting",
                        "progress": 0.0,
                        "message": "Initializing inference",
                    },
                )

            # Make script executable
            os.chmod(
                track_script_path, os.stat(track_script_path).st_mode | stat.S_IEXEC
            )

            # Parse and run track commands
            track_commands = self.parse_track_script(track_script_path)
            start_time = time.time()

            for cmd_args in track_commands:
                job_name = "inference"
                channel.send(f"INFERENCE_JOB_START::{job_name}")

                # Run sleap-nn track
                process = await asyncio.create_subprocess_exec(
                    *cmd_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    cwd=self.unzipped_dir,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )

                # Stream logs
                async for line in process.stdout:
                    decoded_line = line.decode().rstrip()
                    if channel.readyState == "open":
                        try:
                            channel.send(f"TRACK_LOG:{decoded_line}")
                        except Exception as e:
                            logging.error(f"Failed to send track log: {e}")

                await process.wait()

                if process.returncode == 0:
                    logging.info("Inference completed successfully")
                    channel.send(f"INFERENCE_JOB_END::{job_name}")
                else:
                    logging.error(f"Inference failed with code {process.returncode}")
                    channel.send(
                        f"INFERENCE_JOB_ERROR::{job_name}::{process.returncode}"
                    )

                    # Send failure via peer message
                    if job_id and client_id:
                        await self._send_peer_message(
                            client_id,
                            {
                                "app_message_type": "job_failed",
                                "job_id": job_id,
                                "status": "failed",
                                "error": {
                                    "code": "INFERENCE_FAILED",
                                    "message": f"Inference failed with exit code {process.returncode}",
                                    "recoverable": False,
                                },
                            },
                        )
                    return

            # Find and send predictions file
            predictions_files = list(Path(self.unzipped_dir).glob("*.predictions.slp"))

            if predictions_files:
                predictions_path = predictions_files[0]

                # Send predictions back to client via RTC
                logging.info(f"Sending predictions via RTC: {predictions_path}")
                await self.worker.send_file(channel, str(predictions_path))

                # Send completion via peer message
                if job_id and client_id:
                    predictions_size_mb = predictions_path.stat().st_size / (
                        1024 * 1024
                    )
                    await self._send_peer_message(
                        client_id,
                        {
                            "app_message_type": "job_complete",
                            "job_id": job_id,
                            "status": "completed",
                            "result": {
                                "predictions_size_mb": predictions_size_mb,
                                "inference_duration_seconds": int(
                                    (time.time() - start_time)
                                ),
                                "predictions_file": predictions_path.name,
                            },
                            "transfer_method": "webrtc_datachannel",
                            "ready_for_download": True,
                        },
                    )
            else:
                logging.warning("No predictions file found")

                # Send warning via peer message
                if job_id and client_id:
                    await self._send_peer_message(
                        client_id,
                        {
                            "app_message_type": "job_complete",
                            "job_id": job_id,
                            "status": "completed",
                            "result": {
                                "inference_duration_seconds": int(
                                    (time.time() - start_time)
                                ),
                                "warning": "No predictions file generated",
                            },
                            "transfer_method": "webrtc_datachannel",
                            "ready_for_download": False,
                        },
                    )

            channel.send("INFERENCE_JOBS_DONE")

        except Exception as e:
            # Handle any unexpected errors during inference
            logging.error(f"Error during inference execution: {e}")
            if job_id and client_id:
                await self._send_peer_message(
                    client_id,
                    {
                        "app_message_type": "job_failed",
                        "job_id": job_id,
                        "status": "failed",
                        "error": {
                            "code": "EXECUTION_ERROR",
                            "message": str(e),
                            "recoverable": False,
                        },
                    },
                )
            raise

        finally:
            # Update status back to available when done
            if self.worker.current_job:
                await self.worker.update_status("available")
                self.worker.current_job = None

    async def run_inference(
        self,
        channel,
        cmd: list,
        predictions_path: str,
    ):
        """Run a post-training inference subprocess and forward progress to client.

        Spawns ``sleap track`` (or equivalent command), forwards JSON progress
        lines from stdout as ``INFERENCE_PROGRESS::`` messages, and sends
        ``INFERENCE_COMPLETE::`` or ``INFERENCE_FAILED::`` when the subprocess
        exits.

        Args:
            channel: RTC data channel for sending messages to the client.
            cmd: Full command list to execute (e.g. ``["sleap-nn", "track", ...]``).
            predictions_path: Expected output path for the predictions ``.slp``
                file.  Used to confirm the file exists before sending
                ``INFERENCE_COMPLETE``.
        """
        logging.info(f"[INFERENCE] Running: {' '.join(cmd)}")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                start_new_session=True,
            )
            self._running_process = process
            logging.info(f"[INFERENCE] Process started with PID: {process.pid}")

            assert process.stdout is not None
            async for raw_line in process.stdout:
                line = raw_line.decode(errors="replace").rstrip()
                if not line:
                    continue
                logging.info(f"[INFERENCE] {line}")
                # Forward JSON progress lines as INFERENCE_PROGRESS;
                # forward all other lines as INFERENCE_LOG so the client
                # can display them in the InferenceProgressDialog.
                if line.startswith("{"):
                    if channel.readyState == "open":
                        channel.send(f"INFERENCE_PROGRESS::{line}")
                else:
                    if channel.readyState == "open":
                        channel.send(f"INFERENCE_LOG::{line}")

            await process.wait()
            self._running_process = None
            logging.info(f"[INFERENCE] Process exited with code {process.returncode}")

            cancelled = sys.platform != "win32" and (
                process.returncode == -signal.SIGTERM
            )

            if cancelled:
                logging.info("[INFERENCE] Inference was cancelled by user")
                if channel.readyState == "open":
                    channel.send('INFERENCE_FAILED::{"error": "cancelled"}')
                return

            predictions_exist = Path(predictions_path).exists()
            if process.returncode == 0 and predictions_exist:
                logging.info(
                    f"[INFERENCE] Complete — predictions at {predictions_path}; "
                    f"streaming to client"
                )
                # Stream predictions to the client before signaling completion
                # so that browser / remote PyQt clients that do not share a
                # filesystem with the worker can still consume the results.
                # INFERENCE_COMPLETE is the "all bytes delivered, ready to
                # merge" trigger on the client side.
                try:
                    await self.worker.file_manager.send_file(
                        channel, str(predictions_path)
                    )
                except Exception as e:
                    logging.exception(f"[INFERENCE] Failed to stream predictions: {e}")
                    if channel.readyState == "open":
                        channel.send(
                            "INFERENCE_FAILED::"
                            + json.dumps(
                                {"error": f"failed to stream predictions: {e}"}
                            )
                        )
                    return

                if channel.readyState == "open":
                    channel.send(
                        "INFERENCE_COMPLETE::"
                        + json.dumps({"predictions_path": str(predictions_path)})
                    )
            elif not predictions_exist:
                logging.error(
                    f"[INFERENCE] Subprocess exited 0 but predictions file not found: "
                    f"{predictions_path}"
                )
                if channel.readyState == "open":
                    channel.send('INFERENCE_FAILED::{"error": "no output file"}')
            else:
                logging.error(
                    f"[INFERENCE] Subprocess exited with code {process.returncode}"
                )
                if channel.readyState == "open":
                    channel.send(
                        "INFERENCE_FAILED::"
                        + json.dumps({"error": f"exit code {process.returncode}"})
                    )

        except Exception as e:
            logging.exception(f"[INFERENCE] Unexpected error: {e}")
            self._running_process = None
            if channel.readyState == "open":
                channel.send("INFERENCE_FAILED::" + json.dumps({"error": str(e)}))

    async def _send_peer_message(self, to_peer_id: str, payload: dict):
        """Send peer message via worker's websocket.

        Args:
            to_peer_id: Target peer ID
            payload: Message payload
        """
        # Delegate to worker's peer messaging
        await self.worker._send_peer_message(to_peer_id, payload)

    async def execute_from_spec(
        self,
        channel,
        cmd: list,
        job_id: str,
        job_type: str = "train",
        working_dir: str = None,
        zmq_ports: dict | None = None,
        progress_reporter=None,
        spec: "TrackJobSpec | TrainJobSpec | None" = None,
    ):
        """Execute a job from a pre-built command list (structured job submission).

        This method executes a sleap-nn command that was built by CommandBuilder
        from a validated job specification. It reuses the log streaming logic
        and sends JOB_PROGRESS, JOB_COMPLETE, or JOB_FAILED messages.

        Args:
            channel: RTC data channel for sending logs and status
            cmd: Command list to execute (e.g., ["sleap-nn", "train", ...])
            job_id: Unique job identifier
            job_type: Type of job ("train" or "track")
            working_dir: Working directory for execution (defaults to current dir)
            zmq_ports: Optional dict with 'controller' and 'publish' ZMQ port
                numbers. When provided for train jobs and no progress_reporter is
                given, a ProgressReporter is created and owned by this call.
            progress_reporter: Optional pre-existing ProgressReporter to use for
                the duration of this job. When provided, this call does not create
                a new reporter and does not call async_cleanup on it — the caller
                is responsible for the reporter's lifecycle.
            spec: Optional pre-validated TrackJobSpec or TrainJobSpec used by the
                track branch to access output_path for streaming predictions back
                to the client (Task 7). When None, no streaming is performed.
        """
        from sleap_connect.protocol import (
            MSG_JOB_LOG,
            MSG_JOB_PROGRESS,
            MSG_JOB_COMPLETE,
            MSG_JOB_FAILED,
            MSG_SEPARATOR,
        )

        working_dir = working_dir or os.getcwd()
        start_time = time.time()

        logging.info(f"[JOB {job_id}] Starting {job_type} job")
        logging.info(f"[JOB {job_id}] Command: {' '.join(cmd)}")
        logging.info(f"[JOB {job_id}] Working directory: {working_dir}")

        # Start ZMQ progress reporter for train jobs so sleap-nn's native
        # ZMQ messages are forwarded over the RTC data channel.
        # Stored on self so the worker message handler can forward
        # control commands (e.g., stop) to sleap-nn via ZMQ.
        #
        # If an external reporter is passed in (multi-model pipeline), use it
        # directly without creating or owning a new one.  The caller is
        # responsible for cleanup.
        _owned_reporter = False
        if progress_reporter is None and job_type == "train" and zmq_ports:
            progress_reporter = ProgressReporter(
                control_address=f"tcp://127.0.0.1:{zmq_ports.get('controller', 9000)}",
                progress_address=f"tcp://127.0.0.1:{zmq_ports.get('publish', 9001)}",
            )
            progress_reporter.start_control_socket()
            progress_reporter.start_progress_listener_task(channel)
            _owned_reporter = True
        self._progress_reporter = progress_reporter

        try:
            # Start the process.
            # For track (inference) jobs, merge stderr into stdout so all
            # output (rich progress, logs) goes through one stream and
            # reaches the relay channel.  Training uses separate stderr
            # for DDP duplicate filtering.
            stderr_mode = (
                asyncio.subprocess.STDOUT
                if job_type == "track"
                else asyncio.subprocess.PIPE
            )
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=stderr_mode,
                cwd=working_dir,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                start_new_session=True,
            )
            self._running_process = process

            logging.info(f"[JOB {job_id}] Process started with PID: {process.pid}")

            # Watchdog: log subprocess liveness every 30 s.  If a stop was
            # requested and ZMQ alone hasn't worked after one full cycle,
            # escalate to SIGINT so the pipeline doesn't hang indefinitely.
            # By the time the watchdog fires (30 s after stop), DDP's
            # reduce_boolean_decision barrier has either completed or already
            # timed out, so SIGINT is safe at this point.
            async def _subprocess_watchdog():
                sigint_sent = False
                while True:
                    await asyncio.sleep(_WATCHDOG_INTERVAL_SECS)
                    rc = process.returncode
                    if rc is None:
                        logging.info(
                            f"[JOB {job_id}] Watchdog: PID {process.pid} still running"
                        )
                        if (
                            self._stop_requested
                            and not sigint_sent
                            and sys.platform != "win32"
                        ):
                            try:
                                pgid = os.getpgid(process.pid)
                                logging.info(
                                    f"[JOB {job_id}] ZMQ stop not effective after "
                                    f"30 s — sending SIGINT to process group {pgid}"
                                )
                                os.killpg(pgid, signal.SIGINT)
                                sigint_sent = True
                            except (ProcessLookupError, PermissionError) as e:
                                logging.warning(
                                    f"[JOB {job_id}] SIGINT fallback failed: {e}"
                                )
                    else:
                        logging.info(
                            f"[JOB {job_id}] Watchdog: PID {process.pid} has exited (rc={rc})"
                        )
                        break

            watchdog_task = asyncio.create_task(_subprocess_watchdog())

            # Stderr forwarder: classify stderr lines by severity so tqdm
            # diagnostic noise doesn't flood the terminal as ERROR.
            #
            # tqdm's TMonitor fires "Timeout (0:00:30)!" + a full thread dump
            # to stderr whenever the main thread is blocked for >30 s (e.g.
            # slow VAST I/O between epochs).  These are informational — log
            # them at DEBUG.  Only genuine error keywords get ERROR level.
            _TQDM_NOISE = (
                "Timeout (0:00:30)!",
                "Thread 0x",
                "  <no Python frame>",
                "  File ",
                "_bootstrap_inner",
                "_bootstrap",
            )
            _FATAL_KEYWORDS = (
                "NCCL",
                "CUDA error",
                "RuntimeError",
                "Traceback (most recent call last)",
                "Killed",
                "Segmentation fault",
                "OOM",
                "out of memory",
            )

            async def _stream_stderr():
                assert process.stderr is not None
                buf = b""
                try:
                    while True:
                        chunk = await process.stderr.read(512)
                        if not chunk:
                            # EOF — flush any remaining buffered content
                            if buf:
                                line = buf.decode(errors="replace").rstrip()
                                if line and not any(
                                    line.startswith(pat) or pat in line
                                    for pat in _TQDM_NOISE
                                ):
                                    if any(kw in line for kw in _FATAL_KEYWORDS):
                                        logging.error(f"[JOB {job_id}] [stderr] {line}")
                                        if channel.readyState == "open":
                                            channel.send(f"[stderr] {line}\n")
                                    else:
                                        logging.warning(
                                            f"[JOB {job_id}] [stderr] {line}"
                                        )
                            break

                        buf += chunk

                        while True:
                            match = SEP.search(buf)
                            if not match:
                                if len(buf) > 64 * 1024:
                                    buf = b""
                                break

                            sep = buf[match.start() : match.end()]
                            payload = buf[: match.start()]
                            buf = buf[match.end() :]

                            text = payload.decode(errors="replace")
                            if not text:
                                continue

                            if sep == b"\r":
                                # tqdm batch-level update — forward as CR:: so
                                # the client can overwrite the previous terminal
                                # line and show a live progress bar.  Only rank 0
                                # writes tqdm (Lightning disables TQDMProgressBar
                                # on non-zero ranks), so no DDP duplication here.
                                logging.debug(f"[JOB {job_id}] [stderr] {text}")
                                if channel.readyState == "open":
                                    channel.send(f"CR::{text}")
                            else:  # \n
                                line = text.rstrip()
                                if not line:
                                    continue
                                if any(
                                    line.startswith(pat) or pat in line
                                    for pat in _TQDM_NOISE
                                ):
                                    logging.debug(f"[JOB {job_id}] [stderr] {line}")
                                elif any(kw in line for kw in _FATAL_KEYWORDS):
                                    logging.error(f"[JOB {job_id}] [stderr] {line}")
                                    if channel.readyState == "open":
                                        channel.send(f"[stderr] {line}\n")
                                else:
                                    # \n-terminated stderr lines come from Python
                                    # logging and appear once per DDP rank.
                                    # For track (inference) jobs, forward to channel
                                    # since they're typically single-GPU and the
                                    # dashboard needs to show inference progress.
                                    logging.warning(f"[JOB {job_id}] [stderr] {line}")
                                    if (
                                        job_type == "track"
                                        and channel.readyState == "open"
                                    ):
                                        channel.send(
                                            f"{MSG_JOB_LOG}{MSG_SEPARATOR}[stderr] {line}\n"
                                        )
                except Exception as e:
                    logging.exception(f"[JOB {job_id}] Stderr stream error: {e}")

            # Only start stderr stream if stderr is separate (training jobs)
            stderr_task = (
                asyncio.create_task(_stream_stderr())
                if process.stderr is not None
                else None
            )

            # Memory monitor: log worker + full process tree RSS + VRAM every 30 s.
            worker_pid = os.getpid()
            _w = _read_rss_mb(worker_pid)
            _p = _read_rss_mb(process.pid)
            logging.info(
                f"[JOB {job_id}] Memory at job start — "
                f"worker: {f'{_w:.0f}' if _w is not None else '?'} MB, "
                f"subprocess: {f'{_p:.0f}' if _p is not None else '?'} MB"
            )

            async def _memory_monitor():
                while process.returncode is None:
                    await asyncio.sleep(30)
                    if process.returncode is not None:
                        break
                    w_mb = _read_rss_mb(worker_pid)
                    rss_total_mb, n_procs = _read_process_tree_rss_mb(process.pid)
                    # Private_Dirty: sum of truly private pages across the
                    # process tree — excludes shared library pages so this
                    # is a much better estimate of actual physical RAM used
                    # by the training job (avoids the ×N double-counting of
                    # shared .so pages that inflates the RSS sum).
                    private_total_mb = sum(
                        v
                        for pid in _iter_process_tree(process.pid)
                        if (v := _read_private_dirty_mb(pid)) is not None
                    )
                    cgroup_mb = _read_cgroup_memory_mb()
                    vram = await _query_vram()
                    cgroup_part = (
                        f", cgroup actual: {cgroup_mb:.0f} MB" if cgroup_mb else ""
                    )
                    vram_part = f", VRAM [{vram}]" if vram else ""
                    logging.info(
                        f"[JOB {job_id}] Memory — "
                        f"worker: {w_mb:.0f} MB, "
                        f"subprocess RSS sum ({n_procs} procs): {rss_total_mb:.0f} MB "
                        f"(private only: {private_total_mb:.0f} MB)"
                        f"{cgroup_part}"
                        f"{vram_part}"
                        if w_mb is not None
                        else f"[JOB {job_id}] Memory read unavailable (non-Linux?)"
                    )

            memory_monitor_task = asyncio.create_task(_memory_monitor())

            # Captured output path from subprocess stdout (sleap-nn prints
            # "Predictions output path: <path>" on completion).  Used by the
            # track-job streaming block below to find the actual output file
            # regardless of sleap-nn's naming convention.
            _captured_output_path: str | None = None

            # Stream logs with progress extraction
            async def stream_logs_with_progress():
                """Stream logs and extract progress information."""
                nonlocal _captured_output_path
                buf = b""
                epoch_pattern = re.compile(r"Epoch\s+(\d+)")
                # Match loss=, loss:, loss  (but not val/loss= or val_loss=)
                loss_pattern = re.compile(r"(?<![/\w])loss[=:\s]+([0-9.]+)")
                # Match val/loss= or val_loss= or val_loss: etc.
                val_loss_pattern = re.compile(r"val[/_]loss[=:\s]+([0-9.]+)")

                current_epoch = 0
                current_loss = None
                current_val_loss = None

                pending_cr = ""
                try:
                    while True:
                        chunk = await process.stdout.read(512)
                        if not chunk:
                            # EOF — subprocess closed its stdout (process is done or pipe broke)
                            logging.info(
                                f"[JOB {job_id}] EOF on stdout from PID {process.pid} — "
                                f"subprocess has closed stdout (returncode={process.returncode!r})"
                            )
                            if pending_cr and channel.readyState == "open":
                                if job_type == "track":
                                    channel.send(
                                        f"{MSG_JOB_LOG}{MSG_SEPARATOR}{pending_cr}\n"
                                    )
                                else:
                                    channel.send(pending_cr + "\n")
                            if buf:
                                line = buf.decode(errors="replace")
                                logging.info(f"[JOB {job_id}] {line}")
                                if channel.readyState == "open":
                                    if job_type == "track":
                                        channel.send(
                                            f"{MSG_JOB_LOG}{MSG_SEPARATOR}{line}\n"
                                        )
                                    else:
                                        channel.send(line + "\n")
                            break

                        buf += chunk

                        while True:
                            match = SEP.search(buf)
                            if not match:
                                # Hold long lines as pending rather than flushing mid-line
                                if len(buf) > 64 * 1024:
                                    pending_cr = buf.decode(errors="replace")
                                    buf = b""
                                break

                            sep = buf[match.start() : match.end()]
                            payload = buf[: match.start()]
                            buf = buf[match.end() :]

                            text = payload.decode(errors="replace")
                            if not text:
                                continue

                            # Log and send log line to client (debug only —
                            # avoids flooding the terminal with every training
                            # output line at INFO level)
                            logging.debug(f"[JOB {job_id}] {text}")

                            # Capture the actual output path printed by
                            # sleap-nn ("Predictions output path: <path>").
                            if (
                                job_type == "track"
                                and "Predictions output path:" in text
                            ):
                                _captured_output_path = text.split(
                                    "Predictions output path:", 1
                                )[1].strip()
                                logging.info(
                                    f"[JOB {job_id}] Captured output path: "
                                    f"{_captured_output_path}"
                                )

                            if sep == b"\n":
                                # \n supersedes any pending \r line (tqdm
                                # emits \r...\n for each batch; discard the
                                # \r version to avoid duplicate sends)
                                pending_cr = ""
                                if channel.readyState == "open":
                                    if job_type == "track":
                                        # Wrap subprocess stdout in a typed
                                        # protocol message so the client's
                                        # response loop can route it via
                                        # on_job_message (Task 9). Training
                                        # keeps the legacy bare-string format
                                        # that _run_training_async's on_log
                                        # callback expects.
                                        channel.send(
                                            f"{MSG_JOB_LOG}{MSG_SEPARATOR}{text}"
                                        )
                                    else:
                                        channel.send(text + "\n")
                            else:  # \r — forward as CR:: for live progress display
                                pending_cr = text
                                if channel.readyState == "open":
                                    if job_type == "track":
                                        channel.send(
                                            f"{MSG_JOB_LOG}{MSG_SEPARATOR}CR::{text}"
                                        )
                                    else:
                                        channel.send(f"CR::{text}")

                            # Extract progress info for training jobs
                            if job_type == "train":
                                epoch_match = epoch_pattern.search(text)
                                if epoch_match:
                                    current_epoch = int(epoch_match.group(1))

                                loss_match = loss_pattern.search(text)
                                if loss_match:
                                    current_loss = float(loss_match.group(1))

                                val_loss_match = val_loss_pattern.search(text)
                                if val_loss_match:
                                    current_val_loss = float(val_loss_match.group(1))

                                # Send progress update if we have epoch info
                                if current_epoch > 0:
                                    progress_data = {
                                        "epoch": current_epoch,
                                    }
                                    if current_loss is not None:
                                        progress_data["loss"] = current_loss
                                    if current_val_loss is not None:
                                        progress_data["val_loss"] = current_val_loss

                                    if channel.readyState == "open":
                                        import json

                                        progress_msg = f"{MSG_JOB_PROGRESS}{MSG_SEPARATOR}{json.dumps(progress_data)}"
                                        channel.send(progress_msg)

                except Exception as e:
                    logging.exception(f"[JOB {job_id}] Log streaming error: {e}")
                    if channel.readyState == "open":
                        channel.send(f"[log-stream error] {e}\n")

            await stream_logs_with_progress()
            # Drain any remaining stderr — crash output (e.g. NCCL errors,
            # Python tracebacks) often arrives just after stdout EOF.
            if stderr_task is not None:
                try:
                    await asyncio.wait_for(stderr_task, timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    stderr_task.cancel()
            watchdog_task.cancel()
            memory_monitor_task.cancel()
            _w = _read_rss_mb(worker_pid)
            logging.info(
                f"[JOB {job_id}] Memory at job end — "
                f"worker: {f'{_w:.0f}' if _w is not None else '?'} MB"
            )
            logging.info(
                f"[JOB {job_id}] Stdout stream ended — waiting for PID {process.pid} to exit"
            )
            await process.wait()
            self._running_process = None

            # Wait for the entire process group to exit before returning.
            # With start_new_session=True, PGID == process.pid.  DDP rank 1
            # is a child in this group and may outlive rank 0 by several
            # seconds while NCCL / TCPStore cleanup runs.  If we start the
            # next pipeline model before rank 1 vacates MASTER_PORT, the new
            # DDP init will collide with the lingering connection → Broken pipe.
            if sys.platform != "win32":
                pgid = process.pid
                deadline = asyncio.get_event_loop().time() + 30.0
                while asyncio.get_event_loop().time() < deadline:
                    try:
                        os.killpg(pgid, 0)  # signal 0: probe group existence
                        await asyncio.sleep(0.5)
                    except (ProcessLookupError, PermissionError):
                        break  # all ranks have exited
                else:
                    logging.warning(
                        f"[JOB {job_id}] Process group {pgid} still alive after "
                        "30 s — proceeding anyway"
                    )

            duration_seconds = int(time.time() - start_time)
            logging.info(
                f"[JOB {job_id}] Process PID {process.pid} exited "
                f"(return code: {process.returncode})"
            )

            # Detect cancellation or early stop:
            # - ZMQ cancel: _cancel_requested flag set (user chose "Cancel Training")
            # - SIGTERM: process killed by signal (legacy/fallback)
            # - ZMQ stop: _stop_requested flag set (user chose "Stop Early")
            # - SIGINT: process group killed by signal (legacy/fallback)
            logging.info(
                f"[JOB {job_id}] Exit analysis: returncode={process.returncode} "
                f"_stop_requested={self._stop_requested} "
                f"_cancel_requested={self._cancel_requested}"
            )
            cancelled = self._cancel_requested or (
                sys.platform != "win32" and process.returncode == -signal.SIGTERM
            )
            stopped_early = (
                (self._stop_requested and not cancelled)
                or (sys.platform != "win32" and process.returncode == -signal.SIGINT)
                or (self._stop_requested and not cancelled and process.returncode != 0)
            )
            logging.info(
                f"[JOB {job_id}] Result: cancelled={cancelled} stopped_early={stopped_early}"
            )

            if cancelled:
                # Job was cancelled by user — report as failed/cancelled
                import json

                error_data = {
                    "job_id": job_id,
                    "job_type": job_type,
                    "duration_seconds": duration_seconds,
                    "message": "Job cancelled by user",
                }
                if channel.readyState == "open":
                    channel.send(
                        f"{MSG_JOB_FAILED}{MSG_SEPARATOR}{json.dumps(error_data)}"
                    )
                logging.info(f"[JOB {job_id}] Job cancelled by user")
                return {
                    "success": False,
                    "stopped_early": False,
                    "cancelled": True,
                }
            elif process.returncode == 0 or stopped_early:
                # Job completed successfully (or stopped early with checkpoint)
                import json

                result_data = {
                    "job_id": job_id,
                    "job_type": job_type,
                    "duration_seconds": duration_seconds,
                    "success": True,
                    "stopped_early": stopped_early,
                }

                # For track jobs, stream the output file to the client BEFORE
                # signaling completion so MSG_JOB_COMPLETE is the "all bytes
                # delivered, ready to merge" trigger. Only stream on a clean
                # success — if the subprocess was stopped early, predictions
                # are partial / undefined. Mirrors PR #79's run_inference.
                if (
                    job_type == "track"
                    and spec is not None
                    and not stopped_early
                    and process.returncode == 0
                ):
                    # Resolve the output path.  Priority:
                    # 1. spec.output_path (user explicitly set it)
                    # 2. _captured_output_path (parsed from sleap-nn stdout)
                    # 3. Convention fallback (data_path → .predictions.slp)
                    if spec.output_path is not None:
                        output_path = Path(spec.output_path)
                    elif _captured_output_path is not None:
                        output_path = Path(_captured_output_path)
                    else:
                        base = Path(spec.data_path)
                        output_path = base.with_suffix(".predictions" + base.suffix)
                    if output_path.exists():
                        try:
                            await self.worker.file_manager.send_file(
                                channel, str(output_path)
                            )
                        except Exception as e:
                            logging.exception(
                                f"[JOB {job_id}] Failed to stream output: {e}"
                            )
                            error_data = {
                                "job_id": job_id,
                                "job_type": job_type,
                                "duration_seconds": duration_seconds,
                                "exit_code": process.returncode,
                                "message": f"failed to stream output: {e}",
                            }
                            if channel.readyState == "open":
                                channel.send(
                                    f"{MSG_JOB_FAILED}{MSG_SEPARATOR}"
                                    f"{json.dumps(error_data)}"
                                )
                            return {
                                "success": False,
                                "stopped_early": False,
                                "cancelled": False,
                            }
                        # Include the worker-side output path in the payload
                        # so the client's _apply_received_predictions(...)
                        # helper (Task 5) can substitute it with the local
                        # tempfile path.
                        result_data["output_path"] = str(output_path)

                if channel.readyState == "open":
                    channel.send(
                        f"{MSG_JOB_COMPLETE}{MSG_SEPARATOR}{json.dumps(result_data)}"
                    )
                if stopped_early:
                    logging.info(f"[JOB {job_id}] Job stopped early (checkpoint saved)")
                else:
                    logging.info(f"[JOB {job_id}] Job completed successfully")
                return {
                    "success": True,
                    "stopped_early": stopped_early,
                    "cancelled": False,
                }
            else:
                # Job failed or was cancelled
                import json

                error_data = {
                    "job_id": job_id,
                    "job_type": job_type,
                    "exit_code": process.returncode,
                    "duration_seconds": duration_seconds,
                    "message": (
                        "Job cancelled by user"
                        if cancelled
                        else f"Process exited with code {process.returncode}"
                    ),
                }
                if channel.readyState == "open":
                    channel.send(
                        f"{MSG_JOB_FAILED}{MSG_SEPARATOR}{json.dumps(error_data)}"
                    )
                if cancelled:
                    logging.info(f"[JOB {job_id}] Job cancelled by user")
                else:
                    logging.error(
                        f"[JOB {job_id}] Job failed with exit code {process.returncode}"
                    )
                return {
                    "success": False,
                    "stopped_early": False,
                    "cancelled": cancelled,
                }

        except Exception as e:
            # Unexpected error
            import json

            logging.error(f"[JOB {job_id}] Execution error: {e}")
            error_data = {
                "job_id": job_id,
                "job_type": job_type,
                "message": str(e),
            }
            if channel.readyState == "open":
                channel.send(f"{MSG_JOB_FAILED}{MSG_SEPARATOR}{json.dumps(error_data)}")
            return {
                "success": False,
                "stopped_early": False,
                "cancelled": False,
            }

        finally:
            self._stop_requested = False  # reset so next job starts clean
            self._cancel_requested = False
            if _owned_reporter and progress_reporter is not None:
                await progress_reporter.async_cleanup()
                self._progress_reporter = None
            # External reporters are left running; self._progress_reporter stays
            # set so stop commands continue to work during the between-model gap.
