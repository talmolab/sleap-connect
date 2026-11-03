"""Entry point for sleap_connect worker CLI."""

import asyncio
import os
import logging

from aiortc import RTCPeerConnection
from sleap_connect.worker.worker_class import RTCWorkerClient
from sleap_connect.config import get_config


def run_RTCworker(
    api_key=None,
    room_id=None,
    token=None,
    working_dir=None,
    name=None,
    room_secret=None,
    max_reconnect_time=None,
):
    """Create RTCWorkerClient and start it.

    Args:
        api_key: API key for authentication (slp_xxx...). New auth method.
        room_id: Optional room ID to join. If not provided, a new room will be created.
        token: Optional room token for authentication. Required if room_id is provided.
        working_dir: Optional working directory. CLI option overrides config file.
        name: Optional human-readable name for this worker.
        room_secret: Optional room secret for P2P authentication (CLI override).
        max_reconnect_time: Maximum time in seconds to keep retrying signaling server
            reconnection before exiting. None means retry forever.
    """
    # Get configuration
    config = get_config()

    # Get worker I/O config
    worker_io_config = config.get_worker_io_config()

    # Determine working directory (CLI overrides config)
    effective_working_dir = working_dir or worker_io_config.working_dir

    # Change to working directory if specified
    if effective_working_dir:
        os.chdir(effective_working_dir)
        logging.info(f"Changed working directory to: {effective_working_dir}")

    # Get valid mounts (those that exist and are directories)
    valid_mounts = worker_io_config.get_valid_mounts()
    if valid_mounts:
        logging.info(f"Loaded {len(valid_mounts)} valid mount(s)")
        for mount in valid_mounts:
            logging.info(f"  - {mount.label}: {mount.path}")

    # Create the worker instance with mounts
    worker = RTCWorkerClient(
        mounts=valid_mounts, working_dir=effective_working_dir, name=name
    )

    # Create the RTCPeerConnection object.
    pc = RTCPeerConnection()

    # Run the worker.
    try:
        asyncio.run(
            worker.run_worker(
                pc=pc,
                DNS=config.signaling_websocket,
                port_number=8080,
                api_key=api_key,
                room_id=room_id,
                token=token,
                room_secret=room_secret,
                max_reconnect_time=max_reconnect_time,
            )
        )
    except KeyboardInterrupt:
        logging.info("Worker interrupted by user. Shutting down...")
    finally:
        logging.info("Worker exiting...")
