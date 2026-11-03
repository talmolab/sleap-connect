"""Client-side file upload utilities for sleap-connect.

This module provides the upload_file coroutine for transferring files from
client to worker over a WebRTC data channel.
"""

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Callable, Optional

from aiortc import RTCDataChannel

from sleap_connect.protocol import (
    MSG_FILE_UPLOAD_CACHE_HIT,
    MSG_FILE_UPLOAD_CHECK,
    MSG_FILE_UPLOAD_COMPLETE,
    MSG_FILE_UPLOAD_END,
    MSG_FILE_UPLOAD_ERROR,
    MSG_FILE_UPLOAD_PROGRESS,
    MSG_FILE_UPLOAD_READY,
    MSG_FILE_UPLOAD_START,
    MSG_SEPARATOR,
)

UPLOAD_CHUNK_SIZE = 64 * 1024  # 64 KB
UPLOAD_RESPONSE_TIMEOUT = 30.0  # seconds
BUFFER_HIGH_WATER = 16 * 1024 * 1024  # 16 MB


async def upload_file(
    channel: RTCDataChannel,
    response_queue: asyncio.Queue,
    file_path: str,
    dest_dir: str,
    create_subdir: str,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> str:
    """Upload a file from client to worker over an RTC data channel.

    Protocol flow:
      1. Compute SHA-256 of the local file.
      2. Send FILE_UPLOAD_CHECK::{sha256}::{filename}.
         - If worker replies FILE_UPLOAD_CACHE_HIT::{path}, return that path.
         - If worker replies FILE_UPLOAD_READY, proceed to step 3.
      3. Send FILE_UPLOAD_START::{filename}::{total_bytes}::{dest_dir}::{create_subdir}.
         - Wait for FILE_UPLOAD_READY (or FILE_UPLOAD_ERROR).
      4. Send binary chunks (back-pressuring on channel.bufferedAmount).
      5. Send FILE_UPLOAD_END.
         - Drain FILE_UPLOAD_PROGRESS messages, calling on_progress each time.
         - Return path from FILE_UPLOAD_COMPLETE, or raise on FILE_UPLOAD_ERROR.

    Args:
        channel: Open RTCDataChannel to the worker.
        response_queue: asyncio.Queue receiving FILE_UPLOAD_* responses from the
            worker. The caller is responsible for routing on_message events into
            this queue for FILE_UPLOAD_* prefixed messages.
        file_path: Absolute local path to the file to upload.
        dest_dir: Absolute path on the worker where the file should be saved.
        create_subdir: "1" to create a sleap-connect-downloads/ subdirectory inside
            dest_dir, "0" to write directly into dest_dir.
        on_progress: Optional callable(bytes_sent, total_bytes) invoked for each
            FILE_UPLOAD_PROGRESS message received from the worker.

    Returns:
        Absolute path of the uploaded file on the worker.

    Raises:
        RuntimeError: If the worker responds with FILE_UPLOAD_ERROR at any point,
            or if no response is received within UPLOAD_RESPONSE_TIMEOUT seconds.
    """
    path = Path(file_path)
    filename = path.name
    total_bytes = path.stat().st_size

    # Step 1: Compute SHA-256 (streaming to avoid large RAM usage)
    logging.info(f"Computing SHA-256 for {filename}...")
    sha256_ctx = hashlib.sha256()
    with open(file_path, "rb") as fh:
        while chunk := fh.read(UPLOAD_CHUNK_SIZE):
            sha256_ctx.update(chunk)
    sha256 = sha256_ctx.hexdigest()

    # Step 2: SHA-256 pre-check
    logging.info(f"Sending FILE_UPLOAD_CHECK for {filename}")
    channel.send(
        f"{MSG_FILE_UPLOAD_CHECK}{MSG_SEPARATOR}{sha256}{MSG_SEPARATOR}{filename}"
    )
    resp = await asyncio.wait_for(response_queue.get(), timeout=UPLOAD_RESPONSE_TIMEOUT)

    if resp.startswith(MSG_FILE_UPLOAD_CACHE_HIT + MSG_SEPARATOR):
        cached_path = resp.split(MSG_SEPARATOR, 1)[1]
        logging.info(f"Upload cache hit: {cached_path}")
        return cached_path

    if resp.startswith(MSG_FILE_UPLOAD_ERROR + MSG_SEPARATOR):
        reason = resp.split(MSG_SEPARATOR, 1)[1]
        raise RuntimeError(f"Worker rejected upload check: {reason}")

    if resp != MSG_FILE_UPLOAD_READY:
        raise RuntimeError(f"Unexpected response to FILE_UPLOAD_CHECK: {resp}")

    # Step 3: Send FILE_UPLOAD_START
    logging.info(f"Sending FILE_UPLOAD_START for {filename} ({total_bytes} bytes)")
    channel.send(
        f"{MSG_FILE_UPLOAD_START}{MSG_SEPARATOR}{filename}{MSG_SEPARATOR}"
        f"{total_bytes}{MSG_SEPARATOR}{dest_dir}{MSG_SEPARATOR}{create_subdir}"
    )
    resp = await asyncio.wait_for(response_queue.get(), timeout=UPLOAD_RESPONSE_TIMEOUT)

    if resp.startswith(MSG_FILE_UPLOAD_ERROR + MSG_SEPARATOR):
        reason = resp.split(MSG_SEPARATOR, 1)[1]
        raise RuntimeError(f"Worker rejected upload: {reason}")

    if resp != MSG_FILE_UPLOAD_READY:
        raise RuntimeError(f"Unexpected response to FILE_UPLOAD_START: {resp}")

    # Step 4: Send binary chunks
    logging.info(f"Sending {filename} in {UPLOAD_CHUNK_SIZE // 1024} KB chunks...")
    with open(file_path, "rb") as fh:
        while chunk := fh.read(UPLOAD_CHUNK_SIZE):
            while (
                channel.bufferedAmount is not None
                and channel.bufferedAmount > BUFFER_HIGH_WATER
            ):
                await asyncio.sleep(0.1)
            channel.send(chunk)

    # Step 5: Send FILE_UPLOAD_END and await completion
    logging.info("Sending FILE_UPLOAD_END")
    channel.send(MSG_FILE_UPLOAD_END)

    while True:
        resp = await asyncio.wait_for(
            response_queue.get(), timeout=UPLOAD_RESPONSE_TIMEOUT
        )

        if resp.startswith(MSG_FILE_UPLOAD_PROGRESS + MSG_SEPARATOR):
            if on_progress is not None:
                parts = resp.split(MSG_SEPARATOR)
                if len(parts) >= 3:
                    try:
                        on_progress(int(parts[1]), int(parts[2]))
                    except ValueError:
                        pass
            continue

        if resp.startswith(MSG_FILE_UPLOAD_COMPLETE + MSG_SEPARATOR):
            worker_path = resp.split(MSG_SEPARATOR, 1)[1]
            logging.info(f"Upload complete: {worker_path}")
            return worker_path

        if resp.startswith(MSG_FILE_UPLOAD_ERROR + MSG_SEPARATOR):
            reason = resp.split(MSG_SEPARATOR, 1)[1]
            raise RuntimeError(f"Upload failed: {reason}")

        logging.warning(f"Unexpected upload response: {resp[:80]}")
