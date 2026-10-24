"""Proof-of-concept: Receive video frames over WebRTC DataChannel.

This worker receives frames streamed from a client. It supports multiple
verification modes to avoid excessive storage:

- **Stats only** (default): Just count frames and log statistics
- **Display**: Show frames in a window (cv2.imshow) for visual verification
- **Save sampled**: Save every Nth frame to disk
- **Save all**: Save every frame (use with caution for large streams)

Usage:
    python frame_worker.py [options]

Examples:
    python frame_worker.py                              # Stats only (no saving)
    python frame_worker.py --display                    # Show frames in window
    python frame_worker.py --save-every 30              # Save every 30th frame
    python frame_worker.py --output ./frames --save-every 10
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription

logging.basicConfig(level=logging.INFO)


class FrameStreamWorker:
    """Worker that receives video frames over WebRTC DataChannel."""

    def __init__(
        self,
        output_dir: str = None,
        display: bool = False,
        save_every: int = 0,
    ):
        """Initialize the frame stream worker.

        Args:
            output_dir: Directory to save frames (if saving enabled).
            display: If True, show frames in a window.
            save_every: Save every Nth frame. 0 = don't save, 1 = save all.
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.display = display
        self.save_every = save_every

        if self.output_dir and self.save_every > 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pc = RTCPeerConnection()
        self.websocket = None
        self.peer_id = f"frame-worker-{id(self) % 10000:04d}"

        # Frame reception state
        self.current_frame_id = None
        self.current_frame_meta = None
        self.current_frame_buffer = bytearray()
        self.frames_received = 0
        self.frames_saved = 0
        self.bytes_received = 0

        # Batch state
        self.in_batch = False
        self.batch_frames = []

        # Stream metadata
        self.stream_info = None
        self.frame_output_dir = None
        self.start_time = None

    def setup_handlers(self):
        """Setup WebRTC event handlers."""

        @self.pc.on("datachannel")
        def on_datachannel(channel):
            logging.info(f"DataChannel received: {channel.label}")
            self._setup_channel_handlers(channel)

        @self.pc.on("iceconnectionstatechange")
        async def on_ice_state_change():
            logging.info(f"ICE state: {self.pc.iceConnectionState}")
            if self.pc.iceConnectionState == "failed":
                await self.cleanup()

    def _setup_channel_handlers(self, channel):
        """Setup handlers for the data channel."""

        @channel.on("open")
        def on_open():
            logging.info(f"DataChannel {channel.label} is open")

        @channel.on("message")
        async def on_message(message):
            await self._handle_message(message, channel)

    async def _handle_message(self, message, channel):
        """Handle incoming messages (metadata or frame data)."""

        if isinstance(message, str):
            # Handle string messages (metadata, control)
            if message.startswith("{"):
                # JSON message
                data = json.loads(message)
                await self._handle_json_message(data, channel)

            elif message.startswith("BATCH_START::"):
                # Start of a batch
                _, num_frames = message.split("::", 1)
                self.in_batch = True
                self.batch_frames = []
                logging.debug(f"Batch starting: {num_frames} frames")

            elif message.startswith("BATCH_END::"):
                # End of a batch - process all frames
                _, num_frames = message.split("::", 1)
                self.in_batch = False
                logging.info(f"Batch complete: {len(self.batch_frames)} frames")
                # Batch frames are already saved individually
                self.batch_frames = []

            elif message.startswith("FRAME_META::"):
                # Frame metadata: FRAME_META::{frame_id}:{height}:{width}:{channels}:{dtype}:{nbytes}
                _, meta = message.split("::", 1)
                parts = meta.split(":")
                self.current_frame_id = int(parts[0])
                self.current_frame_meta = {
                    "frame_id": int(parts[0]),
                    "height": int(parts[1]),
                    "width": int(parts[2]),
                    "channels": int(parts[3]),
                    "dtype": parts[4],
                    "nbytes": int(parts[5]),
                }
                self.current_frame_buffer = bytearray()

            elif message.startswith("FRAME_END::"):
                # Frame complete
                _, frame_id_str = message.split("::", 1)
                frame_id = int(frame_id_str)
                await self._process_frame(frame_id, channel)

                # Track batch frames
                if self.in_batch:
                    self.batch_frames.append(frame_id)

        elif isinstance(message, bytes):
            # Binary frame data
            if message == b"KEEP_ALIVE":
                return
            self.current_frame_buffer.extend(message)
            self.bytes_received += len(message)

    async def _handle_json_message(self, data: dict, channel):
        """Handle JSON control messages."""
        msg_type = data.get("type")

        if msg_type == "STREAM_START":
            self.stream_info = data
            self.frames_received = 0
            self.frames_saved = 0
            self.bytes_received = 0
            self.start_time = time.time()

            source_name = data.get("source_name", "unknown")
            is_webcam = data.get("is_webcam", False)

            # Create output directory for this stream if saving
            if self.output_dir and self.save_every > 0:
                safe_name = source_name.replace(".", "_").replace(":", "_")
                self.frame_output_dir = self.output_dir / f"frames_{safe_name}"
                self.frame_output_dir.mkdir(parents=True, exist_ok=True)
                logging.info(f"Saving frames to: {self.frame_output_dir}")

            logging.info(f"Stream starting: {data.get('width')}x{data.get('height')} "
                         f"from {'webcam' if is_webcam else 'video'} '{source_name}'")

            # Log verification mode
            if self.display:
                logging.info("Verification: DISPLAY mode (showing frames in window)")
            elif self.save_every > 0:
                logging.info(f"Verification: SAVE mode (every {self.save_every} frame(s))")
            else:
                logging.info("Verification: STATS only (not saving frames)")

            channel.send(json.dumps({
                "type": "STREAM_ACK",
                "status": "ready",
                "verification_mode": "display" if self.display else (
                    f"save_every_{self.save_every}" if self.save_every > 0 else "stats_only"
                ),
            }))

        elif msg_type == "STREAM_END":
            elapsed = time.time() - self.start_time if self.start_time else 0
            fps = self.frames_received / elapsed if elapsed > 0 else 0
            mb_received = self.bytes_received / (1024 * 1024)

            logging.info(f"Stream ended: {self.frames_received} frames received")
            logging.info(f"  Time: {elapsed:.1f}s ({fps:.1f} FPS)")
            logging.info(f"  Data: {mb_received:.1f} MB ({mb_received/elapsed:.1f} MB/s)" if elapsed > 0 else "")
            if self.save_every > 0:
                logging.info(f"  Saved: {self.frames_saved} frames to {self.frame_output_dir}")

            # Close display window if open
            if self.display:
                cv2.destroyAllWindows()

            channel.send(json.dumps({
                "type": "STREAM_COMPLETE",
                "frames_received": self.frames_received,
                "frames_saved": self.frames_saved,
                "elapsed_seconds": round(elapsed, 2),
                "average_fps": round(fps, 2),
            }))

    async def _process_frame(self, frame_id: int, channel):
        """Reconstruct frame from buffer and process it."""
        if self.current_frame_meta is None:
            logging.error(f"No metadata for frame {frame_id}")
            return

        meta = self.current_frame_meta

        # Verify data size
        expected_size = meta["nbytes"]
        actual_size = len(self.current_frame_buffer)

        if actual_size != expected_size:
            logging.warning(
                f"Frame {frame_id} size mismatch: expected {expected_size}, got {actual_size}"
            )
            return

        # Reconstruct numpy array
        dtype_str = meta["dtype"]

        if dtype_str == "jpeg":
            # Decode JPEG compressed frame
            frame = cv2.imdecode(
                np.frombuffer(self.current_frame_buffer, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            if frame is None:
                logging.error(f"Failed to decode JPEG frame {frame_id}")
                return
        else:
            # Raw numpy array
            dtype = np.dtype(dtype_str)
            frame = np.frombuffer(self.current_frame_buffer, dtype=dtype)

            # Reshape to image dimensions
            if meta["channels"] == 1:
                shape = (meta["height"], meta["width"])
            else:
                shape = (meta["height"], meta["width"], meta["channels"])

            try:
                frame = frame.reshape(shape)
            except ValueError as e:
                logging.error(f"Failed to reshape frame {frame_id}: {e}")
                return

        self.frames_received += 1

        # Display frame if enabled
        if self.display:
            # Add frame info overlay
            display_frame = frame.copy()
            cv2.putText(
                display_frame,
                f"Frame {frame_id} | Received: {self.frames_received}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Frame Stream Worker", display_frame)
            # Process window events (required for display to work)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Display window closed by user")
                self.display = False
                cv2.destroyAllWindows()

        # Save frame if enabled and matches save interval
        if self.save_every > 0 and self.frame_output_dir:
            if self.frames_received % self.save_every == 0:
                frame_path = self.frame_output_dir / f"frame_{frame_id:06d}.png"
                cv2.imwrite(str(frame_path), frame)
                self.frames_saved += 1

        # Log progress
        if self.frames_received % 30 == 0:
            total = self.stream_info.get("frames_to_send", "?") if self.stream_info else "?"
            elapsed = time.time() - self.start_time if self.start_time else 0
            fps = self.frames_received / elapsed if elapsed > 0 else 0
            if total and total > 0:
                logging.info(f"Received {self.frames_received}/{total} frames ({fps:.1f} FPS)")
            else:
                logging.info(f"Received {self.frames_received} frames ({fps:.1f} FPS)")

        # Clear buffer for next frame
        self.current_frame_buffer = bytearray()
        self.current_frame_meta = None

    async def cleanup(self):
        """Cleanup connections."""
        logging.info("Cleaning up...")
        if self.display:
            cv2.destroyAllWindows()
        if self.pc:
            await self.pc.close()
        if self.websocket:
            await self.websocket.close()


async def run_frame_worker(
    output_dir: str,
    dns: str,
    port: int,
    display: bool = False,
    save_every: int = 0,
):
    """Main worker function to receive video frames from a client."""

    worker = FrameStreamWorker(
        output_dir=output_dir,
        display=display,
        save_every=save_every,
    )
    worker.setup_handlers()

    logging.info(f"Worker {worker.peer_id} starting")

    # Log mode
    if display:
        logging.info("Mode: DISPLAY (frames shown in window)")
    elif save_every > 0:
        logging.info(f"Mode: SAVE every {save_every} frame(s) to {output_dir}")
    else:
        logging.info("Mode: STATS only (frames counted but not saved)")

    # Connect to signaling server
    async with websockets.connect(f"{dns}:{port}") as websocket:
        worker.websocket = websocket

        # Register as worker
        await websocket.send(json.dumps({
            "type": "register",
            "peer_id": worker.peer_id,
        }))
        logging.info(f"Registered as {worker.peer_id}")

        # Handle signaling messages
        async for message in websocket:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "offer":
                logging.info("Received offer from client")
                sender = data.get("sender")

                # Set remote description
                await worker.pc.setRemoteDescription(
                    RTCSessionDescription(sdp=data["sdp"], type="offer")
                )

                # Create and send answer
                await worker.pc.setLocalDescription(await worker.pc.createAnswer())
                await websocket.send(json.dumps({
                    "type": "answer",
                    "target": sender,
                    "sdp": worker.pc.localDescription.sdp,
                }))
                logging.info("Answer sent - waiting for frames...")

            elif msg_type == "candidate":
                candidate = data.get("candidate")
                if candidate:
                    await worker.pc.addIceCandidate(candidate)

            elif msg_type == "quit":
                logging.info("Client quit")
                break

    await worker.cleanup()
    logging.info("Worker closed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Receive video frames over WebRTC from a client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Verification Modes:
  By default, the worker only counts frames and logs statistics (no storage used).
  Use --display or --save-every to verify frames are being received correctly.

Examples:
  # Stats only - just count frames (default, no storage)
  python frame_worker.py

  # Display frames in a window (press 'q' to close)
  python frame_worker.py --display

  # Save every 30th frame to disk
  python frame_worker.py --output ./frames --save-every 30

  # Save all frames (use with caution!)
  python frame_worker.py --output ./frames --save-every 1

  # Connect to remote signaling server
  python frame_worker.py --host ws://example.com --port 8080 --display
        """,
    )
    parser.add_argument("--output", dest="output_dir", default="./received_frames",
                        help="Directory to save frames (default: ./received_frames)")
    parser.add_argument("--host", default="ws://localhost",
                        help="Signaling server URL (default: ws://localhost)")
    parser.add_argument("--port", type=int, default=8080,
                        help="Signaling server port (default: 8080)")
    parser.add_argument("--display", action="store_true",
                        help="Display frames in a window for visual verification")
    parser.add_argument("--save-every", type=int, default=0, metavar="N",
                        help="Save every Nth frame. 0=don't save (default), 1=save all")

    args = parser.parse_args()

    try:
        asyncio.run(run_frame_worker(
            output_dir=args.output_dir,
            dns=args.host,
            port=args.port,
            display=args.display,
            save_every=args.save_every,
        ))
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
