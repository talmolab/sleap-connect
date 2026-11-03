"""Entry point for remote inference client."""

import asyncio
import logging
from pathlib import Path
from loguru import logger

from sleap_connect.client.client_track_class import RTCTrackClient


def run_RTCclient_track(
    session_string: str = None,
    data_path: str = None,
    model_paths: list = None,
    output: str = None,
    only_suggested_frames: bool = True,
    room_id: str = None,
    token: str = None,
    worker_id: str = None,
    auto_select: bool = False,
    min_gpu_memory: int = None,
    **kwargs,
) -> None:
    """Main entry point for remote inference client.

    Args:
        session_string: Session string for direct connection to specific worker
        data_path: Path to .slp file with data
        model_paths: List of paths to trained model directories
        output: Output predictions filename
        only_suggested_frames: Whether to track only suggested frames
        room_id: Room ID for room-based worker discovery
        token: Room token for authentication
        worker_id: Specific worker peer-id to connect to (skips discovery)
        auto_select: Automatically select best worker by GPU memory
        min_gpu_memory: Minimum GPU memory in MB for worker filtering
        **kwargs: Additional arguments passed to run_client

    Returns:
        None
    """
    # Validate inputs
    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        return

    for model_path in model_paths:
        if not Path(model_path).exists():
            logger.error(f"Model directory not found: {model_path}")
            return

    # Create client instance
    client = RTCTrackClient(
        DNS=None,  # Use config
        port_number="8080",
    )

    # Create track package
    logger.info("Creating track package...")
    try:
        package_path = client.create_track_package(
            data_path=data_path,
            model_paths=model_paths,
            output=output,
            only_suggested_frames=only_suggested_frames,
        )
    except Exception as e:
        logger.error(f"Failed to create track package: {e}")
        return

    # Run the client
    logger.info(f"Starting inference client...")
    try:
        asyncio.run(
            client.run_client(
                file_path=package_path,
                output_dir=".",  # Save predictions to current directory
                session_string=session_string,
                # Room-based connection parameters
                room_id=room_id,
                token=token,
                worker_id=worker_id,
                auto_select=auto_select,
                min_gpu_memory=min_gpu_memory,
                **kwargs,
            )
        )
    except KeyboardInterrupt:
        logger.info("Client interrupted by user. Shutting down...")
    except Exception as e:
        logger.error(f"Client error: {e}")
        raise

    logger.info("Inference session complete")
