import asyncio
import logging
import subprocess
import re

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel
from sleap_connect.worker.worker_class import RTCWorkerClient
from pathlib import Path


def parse_training_script(training_script_path: str):
    jobs = []
    pattern = re.compile(r"^\s*sleap-train\s+([^\s]+)\s+([^\s]+)")

    with open(training_script_path, "r") as f:
        for line in f:
            match = pattern.match(line)
            if match:
                config, labels = match.groups()
                jobs.append((config.strip(), labels.strip()))
    return jobs


async def run_all_training_jobs(
    channel: RTCDataChannel, train_script_path: str, save_dir: str
):
    training_jobs = parse_training_script(train_script_path)

    for config_name, labels_name in training_jobs:
        job_name = Path(config_name).stem

        # Send RTC msg over channel to indicate job start.
        logging.info(
            f"Starting training job: {job_name} with config: {config_name} and labels: {labels_name}"
        )
        channel.send(f"TRAIN_JOB_START::{job_name}")

        cmd = [
            "sleap-train",
            config_name,
            labels_name,
            "--zmq",
            "--controller_port",
            "9000",
            "--publish_port",
            "9001",
        ]
        logging.info(f"[RUNNING] {' '.join(cmd)}")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=save_dir,
        )

        assert process.stdout is not None

        async def stream_logs():
            async for line in process.stdout:
                decoded_line = line.decode().rstrip()
                logging.info(decoded_line)

                if channel.readyState == "open":
                    try:
                        channel.send(f"TRAIN_LOG:{decoded_line}")
                    except Exception as e:
                        logging.error(f"Failed to send log line: {e}")

        await stream_logs()
        await process.wait()

        if process.returncode == 0:
            logging.info(f"[DONE] Job {job_name} completed successfully.")
            if channel.readyState == "open":
                channel.send(f"TRAIN_END::{job_name}")
        else:
            logging.warning(
                f"[FAILED] Job {job_name} exited with code {process.returncode}."
            )
            if channel.readyState == "open":
                channel.send(f"TRAIN_ERROR::{job_name}::{process.returncode}")

        # try:
        #     subprocess.run([
        #         "sleap-train",
        #         config_name,
        #         labels_name,
        #         "--zmq",
        #         "--controller_port",
        #         "9000",
        #         "--publish_port",
        #         "9001"
        #     ], check=True)
        # except subprocess.CalledProcessError as e:
        #     channel.send(f"TRAIN_JOB_ERROR::{job_name}::{e.stderr}")
        #     logging.error(f"Training job {job_name} failed with error: {e.stderr}")
        #     continue

        channel.send(f"TRAIN_JOB_END::{job_name}")

    channel.send("TRAINING_JOBS_DONE")
