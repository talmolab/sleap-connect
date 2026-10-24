"""Proof-of-concept: Stream video frames over WebRTC DataChannel.

This client reads frames from a video file OR webcam and streams them
to a worker over WebRTC. This is preparation for future sleap-nn
streaming inference support.

Supports:
- Video files: MP4, AVI, etc.
- Webcam: --webcam 0 (device index)
- Frame selection: --frames 0-100,200-300 (video files only)
- Sampling: --sample-rate 10 (every 10th frame)
- Batch mode: --batch-size 8 (send 8 frames per batch)

Usage:
    python frame_client.py <video_path> [options]
    python frame_client.py --webcam 0 [options]

Examples:
    python frame_client.py test_video.mp4
    python frame_client.py test_video.mp4 --frames 0-100
    python frame_client.py --webcam 0 --max-frames 100 --fps 10
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time

import cv2
import numpy as np
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription

logging.basicConfig(level=logging.INFO)

# Frame transfer settings
CHUNK_SIZE = 64 * 1024  # 64KB chunks (same as file transfer)
MAX_BUFFER = 16 * 1024 * 1024  # 16MB buffer limit


def parse_frame_ranges(frame_spec: str, total_frames: int) -> list[int]:
    """Parse frame specification string into list of frame indices.

    Args:
        frame_spec: Comma-separated ranges, e.g., "0-100,200-300,500"
        total_frames: Total number of frames in video

    Returns:
        Sorted list of unique frame indices
    """
    if not frame_spec:
        return list(range(total_frames))

    frames = set()
    for part in frame_spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            start = int(start)
            end = min(int(end), total_frames - 1)
            frames.update(range(start, end + 1))
        else:
            idx = int(part)
            if idx < total_frames:
                frames.add(idx)

    return sorted(frames)


class FrameStreamClient:
    """Client that streams video frames over WebRTC DataChannel."""

    def __init__(
        self,
        source: str,
        is_webcam: bool = False,
        resize: tuple = None,
        jpeg_quality: int = 0,
    ):
        self.source = source
        self.is_webcam = is_webcam
        self.resize = resize  # (width, height) or None
        self.jpeg_quality = jpeg_quality  # 0 = disabled, 1-100 = JPEG quality
        self.pc = RTCPeerConnection()
        self.channel = None
        self.frames_sent = 0
        self.streaming = False

    async def stream_frames(
        self,
        frame_indices: list[int] = None,
        sample_rate: int = 1,
        batch_size: int = 1,
        target_fps: float = None,
        max_frames: int = 0,
    ):
        """Extract frames from video/webcam and send over DataChannel.

        Args:
            frame_indices: Specific frame indices to send (video files only).
            sample_rate: Send every Nth frame (1 = all frames, 10 = every 10th).
            batch_size: Number of frames to group in each batch message.
            target_fps: Target frames per second. If None, sends as fast as possible.
            max_frames: Maximum frames to send (0 = unlimited, mainly for webcam).
        """
        if not self.channel or self.channel.readyState != "open":
            logging.error("DataChannel not open")
            return

        # Open video source
        if self.is_webcam:
            cap = cv2.VideoCapture(int(self.source))
            source_name = f"webcam:{self.source}"

            # macOS webcams need time to initialize
            logging.info("Waiting for webcam to initialize...")
            await asyncio.sleep(1.0)

            # Warm up: read and discard a few frames
            for _ in range(5):
                cap.read()
                await asyncio.sleep(0.1)
        else:
            cap = cv2.VideoCapture(self.source)
            source_name = os.path.basename(self.source)

        if not cap.isOpened():
            logging.error(f"Failed to open source: {self.source}")
            return

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # Default 30 for webcam

        if self.is_webcam:
            total_frames = max_frames if max_frames > 0 else -1  # -1 = unlimited
            frames_to_send = max_frames if max_frames > 0 else -1
            logging.info(f"Webcam: {width}x{height} @ {video_fps:.1f} FPS (max_frames={max_frames or 'unlimited'})")
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logging.info(f"Video: {width}x{height}, {total_frames} frames @ {video_fps} FPS")

            # Determine which frames to send
            if frame_indices is None:
                frame_indices = list(range(total_frames))

            # Apply sampling
            if sample_rate > 1:
                frame_indices = frame_indices[::sample_rate]

            frames_to_send = len(frame_indices)

        # Log compression settings
        compression_info = []
        if self.resize:
            compression_info.append(f"resize={self.resize[0]}x{self.resize[1]}")
        if self.jpeg_quality > 0:
            compression_info.append(f"jpeg_quality={self.jpeg_quality}")
        compression_str = f" [{', '.join(compression_info)}]" if compression_info else ""

        logging.info(f"Will send {frames_to_send if frames_to_send > 0 else 'unlimited'} frames "
                     f"(sample_rate={sample_rate}, batch_size={batch_size}){compression_str}")

        # Determine output dimensions after resize
        out_width = self.resize[0] if self.resize else width
        out_height = self.resize[1] if self.resize else height

        # Send stream start metadata
        self.channel.send(json.dumps({
            "type": "STREAM_START",
            "source_name": source_name,
            "is_webcam": self.is_webcam,
            "total_frames": total_frames,
            "frames_to_send": frames_to_send,
            "sample_rate": sample_rate,
            "batch_size": batch_size,
            "fps": video_fps,
            "width": out_width,
            "height": out_height,
            "original_width": width,
            "original_height": height,
            "jpeg_quality": self.jpeg_quality,
        }))

        # Calculate frame delay
        frame_delay = 1.0 / target_fps if target_fps and target_fps > 0 else 0

        self.streaming = True
        self.frames_sent = 0
        current_batch = []
        frame_idx = 0
        sample_counter = 0
        start_time = time.time()

        while self.streaming:
            # Check max frames limit
            if max_frames > 0 and self.frames_sent >= max_frames:
                break

            if self.is_webcam:
                # Webcam: read next frame directly
                ret, frame = cap.read()
                if not ret:
                    logging.warning("Failed to read from webcam")
                    break

                # Apply sampling for webcam
                sample_counter += 1
                if sample_rate > 1 and sample_counter % sample_rate != 0:
                    continue

                current_frame_idx = frame_idx
                frame_idx += 1
            else:
                # Video file: check if we have more frames to process
                if frame_idx >= len(frame_indices):
                    break

                current_frame_idx = frame_indices[frame_idx]
                frame_idx += 1

                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
                ret, frame = cap.read()

                if not ret:
                    logging.warning(f"Failed to read frame {current_frame_idx}")
                    continue

            # Batch mode: accumulate frames
            if batch_size > 1:
                current_batch.append((current_frame_idx, frame))
                if len(current_batch) >= batch_size:
                    await self._send_batch(current_batch)
                    self.frames_sent += len(current_batch)
                    current_batch = []
            else:
                # Single frame mode
                await self._send_frame(frame, current_frame_idx)
                self.frames_sent += 1

            # Progress logging
            if self.frames_sent % 30 == 0 and self.frames_sent > 0:
                elapsed = time.time() - start_time
                actual_fps = self.frames_sent / elapsed if elapsed > 0 else 0
                if frames_to_send > 0:
                    logging.info(f"Sent {self.frames_sent}/{frames_to_send} frames ({actual_fps:.1f} FPS)")
                else:
                    logging.info(f"Sent {self.frames_sent} frames ({actual_fps:.1f} FPS)")

            # Rate limiting
            if frame_delay > 0:
                await asyncio.sleep(frame_delay)

        # Send remaining batch
        if current_batch:
            await self._send_batch(current_batch)
            self.frames_sent += len(current_batch)

        cap.release()

        # Send stream end
        elapsed = time.time() - start_time
        actual_fps = self.frames_sent / elapsed if elapsed > 0 else 0
        self.channel.send(json.dumps({
            "type": "STREAM_END",
            "total_sent": self.frames_sent,
            "elapsed_seconds": round(elapsed, 2),
            "average_fps": round(actual_fps, 2),
        }))
        logging.info(f"Stream complete: {self.frames_sent} frames in {elapsed:.1f}s ({actual_fps:.1f} FPS)")

    async def _send_batch(self, batch: list[tuple[int, np.ndarray]]):
        """Send a batch of frames.

        Batch protocol:
        1. Send "BATCH_START::{num_frames}"
        2. Send each frame (using _send_frame)
        3. Send "BATCH_END::{num_frames}"
        """
        self.channel.send(f"BATCH_START::{len(batch)}")

        for frame_idx, frame in batch:
            await self._send_frame(frame, frame_idx)

        self.channel.send(f"BATCH_END::{len(batch)}")

    async def _send_frame(self, frame: np.ndarray, frame_id: int):
        """Send a single frame over the DataChannel.

        Frame protocol:
        1. Send metadata: "FRAME_META::{frame_id}:{height}:{width}:{channels}:{dtype}:{nbytes}:{encoding}"
        2. Send binary chunks (64KB each)
        3. Send "FRAME_END::{frame_id}"
        """
        # Apply resize if configured
        if self.resize:
            frame = cv2.resize(frame, self.resize, interpolation=cv2.INTER_AREA)

        height, width = frame.shape[:2]
        channels = frame.shape[2] if len(frame.shape) > 2 else 1

        # Apply JPEG compression if configured
        if self.jpeg_quality > 0:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            _, encoded = cv2.imencode('.jpg', frame, encode_param)
            frame_bytes = encoded.tobytes()
            dtype = "jpeg"
        else:
            frame_bytes = frame.tobytes()
            dtype = str(frame.dtype)

        nbytes = len(frame_bytes)

        # Send metadata
        meta = f"FRAME_META::{frame_id}:{height}:{width}:{channels}:{dtype}:{nbytes}"
        self.channel.send(meta)

        # Send frame data in chunks
        offset = 0
        while offset < nbytes:
            # Wait if buffer is full
            while self.channel.bufferedAmount > MAX_BUFFER:
                await asyncio.sleep(0.01)

            chunk = frame_bytes[offset:offset + CHUNK_SIZE]
            self.channel.send(chunk)
            offset += len(chunk)

        # Send frame end marker
        self.channel.send(f"FRAME_END::{frame_id}")

    def stop_streaming(self):
        """Stop the frame streaming."""
        self.streaming = False


async def run_frame_client(
    source: str,
    is_webcam: bool,
    dns: str,
    port: int,
    frame_spec: str = None,
    sample_rate: int = 1,
    batch_size: int = 1,
    target_fps: float = None,
    max_frames: int = 0,
    resize: tuple = None,
    jpeg_quality: int = 0,
):
    """Main client function to stream video frames to a worker."""

    frame_indices = None

    # Pre-calculate frame indices for video files
    if not is_webcam:
        cap = cv2.VideoCapture(source)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        frame_indices = parse_frame_ranges(frame_spec, total_frames) if frame_spec else None

    client = FrameStreamClient(
        source,
        is_webcam=is_webcam,
        resize=resize,
        jpeg_quality=jpeg_quality,
    )

    # Create DataChannel
    client.channel = client.pc.createDataChannel("frame-stream")
    logging.info("DataChannel created")

    @client.channel.on("open")
    async def on_channel_open():
        logging.info("DataChannel open - starting frame stream")
        await client.stream_frames(
            frame_indices=frame_indices,
            sample_rate=sample_rate,
            batch_size=batch_size,
            target_fps=target_fps,
            max_frames=max_frames,
        )

    @client.channel.on("message")
    def on_channel_message(message):
        logging.info(f"Worker response: {message}")

    @client.pc.on("iceconnectionstatechange")
    async def on_ice_state_change():
        logging.info(f"ICE state: {client.pc.iceConnectionState}")
        if client.pc.iceConnectionState == "failed":
            client.stop_streaming()
            await client.pc.close()

    # Connect to signaling server
    async with websockets.connect(f"{dns}:{port}") as websocket:
        # Register
        await websocket.send(json.dumps({
            "type": "register",
            "peer_id": "frame-client"
        }))
        logging.info("Registered with signaling server")

        # Query for workers
        await websocket.send(json.dumps({"type": "query"}))
        response = await websocket.recv()
        workers = json.loads(response).get("peers", [])

        if not workers:
            logging.error("No workers available")
            return

        target_worker = workers[0]
        logging.info(f"Connecting to worker: {target_worker}")

        # Create and send offer
        await client.pc.setLocalDescription(await client.pc.createOffer())
        await websocket.send(json.dumps({
            "type": "offer",
            "target": target_worker,
            "sdp": client.pc.localDescription.sdp
        }))
        logging.info("Offer sent")

        # Handle signaling messages
        async for message in websocket:
            data = json.loads(message)

            if data.get("type") == "answer":
                await client.pc.setRemoteDescription(
                    RTCSessionDescription(sdp=data["sdp"], type="answer")
                )
                logging.info("Answer received - connection establishing")

            elif data.get("type") == "candidate":
                candidate = data.get("candidate")
                if candidate:
                    await client.pc.addIceCandidate(candidate)

            elif data.get("type") == "quit":
                logging.info("Worker quit")
                break

        # Wait for streaming to complete
        while client.streaming:
            await asyncio.sleep(0.5)

    await client.pc.close()
    logging.info("Client closed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stream video frames over WebRTC to a worker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stream from video file
  python frame_client.py video.mp4

  # Stream from webcam with JPEG compression (RECOMMENDED for speed)
  python frame_client.py --webcam 0 --max-frames 100 --fps 15 --jpeg-quality 80

  # Stream with resize + compression (fastest)
  python frame_client.py --webcam 0 --resize 640x480 --jpeg-quality 80 --fps 15

  # Stream from webcam indefinitely until Ctrl+C
  python frame_client.py --webcam 0 --fps 15 --jpeg-quality 80

  # Combine options for video files
  python frame_client.py video.mp4 --sample-rate 5 --resize 640x480
        """,
    )
    parser.add_argument("video_path", nargs="?", help="Path to video file (MP4, AVI, etc.)")
    parser.add_argument("--webcam", type=int, metavar="DEVICE", help="Use webcam with device index (0, 1, ...)")
    parser.add_argument("--host", default="ws://localhost", help="Signaling server URL (default: ws://localhost)")
    parser.add_argument("--port", type=int, default=8080, help="Signaling server port (default: 8080)")
    parser.add_argument("--frames", dest="frame_spec", help="Frame ranges to send, e.g., '0-100,200-300' (video only)")
    parser.add_argument("--sample-rate", type=int, default=1, help="Send every Nth frame (default: 1 = all frames)")
    parser.add_argument("--batch-size", type=int, default=1, help="Frames per batch (default: 1 = no batching)")
    parser.add_argument("--fps", type=float, default=None, help="Target FPS rate limit (default: no limit)")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to send, 0=unlimited (default: 0)")
    parser.add_argument("--resize", type=str, metavar="WxH", help="Resize frames before sending, e.g., '640x480'")
    parser.add_argument("--jpeg-quality", type=int, default=0, metavar="Q",
                        help="JPEG compression quality 1-100 (0=disabled, raw frames). Recommended: 80")

    args = parser.parse_args()

    # Parse resize argument
    resize = None
    if args.resize:
        try:
            w, h = args.resize.lower().split('x')
            resize = (int(w), int(h))
        except ValueError:
            print(f"Error: Invalid resize format '{args.resize}'. Use WxH, e.g., '640x480'")
            sys.exit(1)

    # Validate source
    if args.webcam is not None:
        source = str(args.webcam)
        is_webcam = True
    elif args.video_path:
        if not os.path.exists(args.video_path):
            print(f"Error: Video file not found: {args.video_path}")
            sys.exit(1)
        source = args.video_path
        is_webcam = False
    else:
        print("Error: Must specify either video_path or --webcam")
        parser.print_help()
        sys.exit(1)

    try:
        asyncio.run(run_frame_client(
            source=source,
            is_webcam=is_webcam,
            dns=args.host,
            port=args.port,
            frame_spec=args.frame_spec,
            sample_rate=args.sample_rate,
            batch_size=args.batch_size,
            target_fps=args.fps,
            max_frames=args.max_frames,
            resize=resize,
            jpeg_quality=args.jpeg_quality,
        ))
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
