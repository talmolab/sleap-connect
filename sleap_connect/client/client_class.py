import asyncio
import argparse
import base64
import uuid
import requests
import websockets
import json
import jsonpickle
import logging
import os
import re
import zmq
import platform
import time

from aiortc import (
    RTCPeerConnection,
    RTCSessionDescription,
    RTCDataChannel,
    RTCConfiguration,
    RTCIceServer,
)
from functools import partial
from pathlib import Path
from typing import List, Optional, Text, Tuple
from websockets.client import ClientConnection

from sleap_connect.config import get_config
from sleap_connect.exceptions import (
    NoWorkersAvailableError,
    NoWorkersAcceptedError,
    JobFailedError,
    WorkerDiscoveryError,
)
from sleap_connect.filesystem import safe_mkdir
from sleap_connect.protocol import (
    parse_message,
    format_message,
    MSG_FS_RESOLVE,
    MSG_FS_RESOLVE_RESPONSE,
    MSG_FS_GET_MOUNTS,
    MSG_FS_MOUNTS_RESPONSE,
    MSG_FS_ERROR,
    MSG_FS_CHECK_VIDEOS_RESPONSE,
    MSG_FS_SCAN_DIR_RESPONSE,
    MSG_FS_WRITE_SLP_OK,
    MSG_FS_WRITE_SLP_ERROR,
    MSG_SEPARATOR,
    MSG_USE_WORKER_PATH,
    MSG_WORKER_PATH_OK,
    MSG_WORKER_PATH_ERROR,
    # PSK P2P authentication
    MSG_AUTH_CHALLENGE,
    MSG_AUTH_RESPONSE,
    MSG_AUTH_SUCCESS,
    MSG_AUTH_FAILURE,
    # Structured job submission
    MSG_JOB_SUBMIT,
    MSG_JOB_ACCEPTED,
    MSG_JOB_REJECTED,
    MSG_JOB_PROGRESS,
    MSG_JOB_COMPLETE,
    MSG_JOB_FAILED,
)
from sleap_connect.auth.psk import compute_hmac
from sleap_connect.auth.secret_resolver import resolve_secret
from sleap_connect.client.file_selector import (
    select_file_from_candidates,
    MountSelector,
    NoMatchMenu,
    prompt_wildcard_pattern,
    prompt_manual_path,
)

# Setup logging.
logging.basicConfig(level=logging.INFO)

# Global constants.
CHUNK_SIZE = 64 * 1024
MAX_RECONNECT_ATTEMPTS = 5
RETRY_DELAY = 5  # seconds


class RTCClient:
    def __init__(
        self,
        DNS: Optional[str] = None,
        port_number: str = "8080",
        gui: bool = False,
    ):
        # Initialize RTC peer connection and websocket.
        # Note: RTCPeerConnection is created later in run_client after receiving ICE servers
        self.pc: RTCPeerConnection = None
        self.websocket: ClientConnection = None
        self.data_channel: RTCDataChannel = None

        # ICE server configuration (received from signaling server during registration)
        self.ice_servers: list = []

        # Initialize given parameters.
        # Use config if DNS not provided via CLI
        config = get_config()
        self.DNS = DNS if DNS is not None else config.signaling_websocket
        self.port_number = port_number
        self.gui = gui

        # Other variables.
        self.received_files = {}
        self.target_worker = None
        self.reconnecting = False
        self.current_job_id = None  # For tracking active job submissions
        self.verbosity = "default"  # Verbosity level: "default", "verbose", or "quiet"

        # Response queues for coordinating websocket messages
        # (Only handle_connection() calls recv(), other functions wait on these queues)
        self.registration_queue = asyncio.Queue()  # For registered_auth responses
        self.peer_list_queue = asyncio.Queue()  # For peer_list responses
        self.job_response_queues = {}  # job_id -> asyncio.Queue for job responses
        self.fs_response_queue = asyncio.Queue()  # For FS_* responses via data channel

        # Video resolution UI callback and data
        self.on_missing_videos_detected = (
            None  # Callback(missing_videos_data) to launch UI
        )
        self.pending_video_check_data = None  # Store FS_CHECK_VIDEOS_RESPONSE data

        self.room_id = None  # Room ID for credential lookup
        self.cognito_username = None  # Legacy field for cleanup compatibility

        # PSK P2P authentication (zero-trust layer)
        self._room_secret: Optional[str] = None  # Set via CLI flag, env var, or config
        self._authenticated = False  # Whether PSK authentication completed
        self._auth_event = (
            asyncio.Event()
        )  # Signaled when auth completes (success or failure)
        self._auth_failed_reason: Optional[str] = None  # Reason if auth failed

        # Structured job submission
        self.job_spec = None  # TrainJobSpec or TrackJobSpec instance
        self.job_response_queue = asyncio.Queue()  # Queue for job protocol responses

        # File upload (client → worker)
        self.upload_response_queue = asyncio.Queue()  # For FILE_UPLOAD_* responses

    def parse_session_string(self, session_string: str):
        prefix = "sleap-session:"
        if not session_string.startswith(prefix):
            raise ValueError(f"Session string must start with '{prefix}'")

        encoded = session_string[len(prefix) :]
        try:
            json_str = base64.urlsafe_b64decode(encoded).decode()
            data = json.loads(json_str)
            return {
                "room_id": data.get("r"),
                "token": data.get("t"),
                "peer_id": data.get("p"),
            }
        except jsonpickle.UnpicklingError as e:
            raise ValueError(f"Failed to decode session string: {e}")

    def request_peer_room_deletion(self, peer_id: str):
        """Requests the signaling server to delete the room and associated user/worker."""
        config = get_config()
        url = config.get_http_endpoint("/delete-peer")
        json = {
            "peer_id": peer_id,
        }

        # Pass the peer_id to identify which room/peers to delete.
        response = requests.post(url, json=json)

        if response.status_code == 200:
            return  # Success
        else:
            logging.error(f"Failed to delete room and peer: {response.text}")
            return None

    async def start_zmq_listener(
        self, channel: RTCDataChannel, zmq_address: str = "tcp://127.0.0.1:9000"
    ):
        """Starts a ZMQ ctrl SUB socket to listen for ZMQ commands from the LossViewer.

        Args:
            channel: The RTCDataChannel to send progress reports to.
            zmq_address: Address of the ZMQ socket to connect to.

        Returns:
            None
        """
        # Use LossViewer's already initialized ZMQ control socket.
        # Initialize SUB socket.
        logging.info("Starting new ZMQ listener socket...")
        context = zmq.Context()
        socket = context.socket(zmq.SUB)

        logging.info(f"Connecting to ZMQ address: {zmq_address}")
        socket.connect(zmq_address)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")

        loop = asyncio.get_event_loop()

        def recv_msg():
            """Receives a message from the ZMQ socket in a non-blocking way.

            Returns:
                The received message as a JSON object, or None if no message is available.
            """
            try:
                # logging.info("Receiving message from ZMQ...")
                return socket.recv_string(
                    flags=zmq.NOBLOCK
                )  # or jsonpickle.decode(msg_str) if needed
            except zmq.Again:
                return None

        while True:
            # Send progress as JSON string with prefix.
            msg = await loop.run_in_executor(None, recv_msg)

            if msg:
                try:
                    logging.info(f"Sending ZMQ command to worker: {msg}")
                    channel.send(f"ZMQ_CTRL::{msg}")
                    # logging.info("Progress report sent to client.")
                except Exception as e:
                    logging.error(f"Failed to send ZMQ progress: {e}")

            # Polling interval.
            await asyncio.sleep(0.05)

    def start_zmq_control(
        self, zmq_address: str = "tcp://127.0.0.1:9001"
    ):  # Publish Port
        """Starts a ZMQ ctrl PUB socket to forward ZMQ commands to the LossViewer.

        Args:
            zmq_address: Address of the ZMQ socket to connect to.

        Returns:
            None
        """
        # Use LossViewer's already initialized ZMQ control socket.
        # Initialize PUB socket.
        logging.info("Starting new ZMQ control socket...")
        context = zmq.Context()
        socket = context.socket(zmq.PUB)

        logging.info(f"Connecting to ZMQ address: {zmq_address}")
        socket.connect(zmq_address)

        # Set PUB socket for use in other functions.
        self.ctrl_socket = socket
        logging.info("ZMQ control socket initialized.")

    async def clean_exit(self):
        """Cleans up the client connection and closes the peer connection and websocket.

        Args:
            pc: RTCPeerConnection object
            websocket: ClientConnection object
        Returns:
            None
        """
        logging.info("Closing WebRTC connection...")
        if self.pc:
            await self.pc.close()

        logging.info("Closing websocket connection...")
        if self.websocket:
            await self.websocket.close()

        if self.cognito_username:
            self.request_peer_room_deletion(self.cognito_username)
            self.cognito_username = None

        logging.info("Client shutdown complete. Exiting...")

    async def reconnect(self):
        """Attempts to reconnect the client to the worker peer by creating a new offer with ICE restart flag.

        Args:
            pc: RTCPeerConnection object
            websocket: ClientConnection object
        Returns:
            bool: True if reconnection was successful, False otherwise
        """
        # Attempt to reconnect.
        while self.reconnect_attempts < MAX_RECONNECT_ATTEMPTS:
            try:
                self.reconnect_attempts += 1
                logging.info(
                    f"Reconnection attempt {self.reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS}..."
                )

                # Create new offer with ICE restart flag.
                logging.info("Creating new offer with manual ICE restart...")
                await self.pc.setLocalDescription(await self.pc.createOffer())

                # Send new offer to the worker via signaling.
                logging.info(f"Sending new offer to worker: {self.target_worker}")
                await self.websocket.send(
                    json.dumps(
                        {
                            "type": self.pc.localDescription.type,
                            "sender": self.peer_id,  # should be own peer_id (Zoom username)
                            "target": self.target_worker,  # should be Worker's peer_id
                            "sdp": self.pc.localDescription.sdp,
                        }
                    )
                )

                # Wait for connection to complete.
                for _ in range(30):
                    await asyncio.sleep(1)
                    if self.pc.iceConnectionState in ["connected", "completed"]:
                        logging.info("Reconnection successful.")

                        # Clear received files on reconnection.
                        logging.info("Clearing received files on reconnection...")
                        self.received_files.clear()

                        return True

                logging.warning("Reconnection timed out. Retrying...")

            except Exception as e:
                logging.error(f"Reconnection failed with error: {e}")

            await asyncio.sleep(RETRY_DELAY)

        # Maximum reconnection attempts reached.
        logging.error("Maximum reconnection attempts reached. Exiting...")
        await self.clean_exit()
        return False

    async def handle_connection(self):
        """Handles receiving all WebSocket messages and routes them appropriately.

        This is the ONLY method that calls websocket.recv(). All other methods
        wait on queues for their responses.

        Args:
            None
        Returns:
            None
        Raises:
            JSONDecodeError: Invalid JSON received
            Exception: An error occurred while handling the message
        """
        # Handle incoming websocket messages.
        try:
            async for message in self.websocket:
                if type(message) == int:
                    logging.info(f"Received int message: {message}")

                data = json.loads(message)
                msg_type = data.get("type")

                # Receive answer SDP from worker and set it as this peer's remote description.
                if msg_type == "answer":
                    logging.info(f"Received answer from worker")
                    await self.pc.setRemoteDescription(
                        RTCSessionDescription(sdp=data.get("sdp"), type="answer")
                    )

                # Handle "trickle ICE" for non-local ICE candidates.
                elif msg_type == "candidate":
                    logging.debug("Received ICE candidate")
                    candidate = data.get("candidate")
                    await self.pc.addIceCandidate(candidate)

                # NOT initiator, received quit request from worker.
                elif msg_type == "quit":
                    logging.info("Worker has quit. Closing connection...")
                    await self.clean_exit()
                    break

                # Registration confirmation → route to queue
                elif msg_type == "registered_auth":
                    logging.info("Client authenticated with server")
                    await self.registration_queue.put(data)

                # Peer discovery response → route to queue
                elif msg_type == "peer_list":
                    workers = data.get("peers", [])
                    logging.info(f"Discovered {len(workers)} workers")
                    await self.peer_list_queue.put(workers)

                # Peer messages (job responses, progress updates) → route to job-specific queue
                elif msg_type == "peer_message":
                    payload = data.get("payload", {})
                    app_msg_type = payload.get("app_message_type")

                    if app_msg_type == "job_response":
                        # Route to job-specific queue
                        job_id = payload.get("job_id")
                        if job_id and job_id in self.job_response_queues:
                            await self.job_response_queues[job_id].put(
                                {
                                    "worker_id": data.get("from_peer_id"),
                                    "accepted": payload.get("accepted"),
                                    "reason": payload.get("reason", ""),
                                    "estimated_duration_minutes": payload.get(
                                        "estimated_duration_minutes", 0
                                    ),
                                    "estimated_start_time_sec": payload.get(
                                        "estimated_start_time_sec", 0
                                    ),
                                    "worker_info": payload.get("worker_info", {}),
                                }
                            )

                    elif app_msg_type in [
                        "job_started",
                        "job_progress",
                        "job_completed",
                        "job_failed",
                    ]:
                        # Route to job progress queue
                        job_id = payload.get("job_id")
                        if job_id and job_id in self.job_response_queues:
                            await self.job_response_queues[job_id].put(payload)

                    logging.debug(f"Received peer message: {app_msg_type}")

                # Unhandled message types.
                else:
                    logging.debug(f"Unhandled message type: {msg_type}")

        except json.JSONDecodeError:
            logging.error("Invalid JSON received")

        except Exception as e:
            logging.error(f"Error handling message: {e}")

    async def keep_ice_alive(self):
        """Sends periodic keep-alive messages to the worker peer to maintain the connection."""
        while True:
            await asyncio.sleep(15)
            if self.data_channel.readyState == "open":
                self.data_channel.send(b"KEEP_ALIVE")

    # ===== Filesystem Path Resolution =====

    async def _send_fs_get_mounts(self, timeout: float = 10.0) -> list:
        """Send FS_GET_MOUNTS message to Worker and wait for response.

        Args:
            timeout: Timeout in seconds for response.

        Returns:
            List of mount dictionaries with 'label' and 'path' keys.
        """
        if self.data_channel.readyState != "open":
            return []

        # Clear any stale responses
        while not self.fs_response_queue.empty():
            try:
                self.fs_response_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Send request
        logging.info("Sending FS_GET_MOUNTS")
        self.data_channel.send(MSG_FS_GET_MOUNTS)

        # Wait for response
        try:
            response = await asyncio.wait_for(
                self.fs_response_queue.get(), timeout=timeout
            )
            if response.startswith(MSG_FS_MOUNTS_RESPONSE):
                json_str = response.split(MSG_SEPARATOR, 1)[1]
                return json.loads(json_str)
            return []
        except asyncio.TimeoutError:
            logging.warning("Timeout waiting for FS_GET_MOUNTS response")
            return []
        except Exception as e:
            logging.error(f"Error getting mounts: {e}")
            return []

    async def _send_fs_resolve(
        self,
        pattern: str,
        file_size: int = None,
        max_depth: int = None,
        mount_label: str = None,
        timeout: float = 15.0,
    ) -> dict:
        """Send FS_RESOLVE message to Worker and wait for response.

        Args:
            pattern: Filename or wildcard pattern to search for.
            file_size: Optional file size for ranking matches.
            max_depth: Optional max directory depth to search.
            mount_label: Optional mount label to filter search (or "all").
            timeout: Timeout in seconds for response.

        Returns:
            Response dictionary with candidates, or error info.
        """
        if self.data_channel.readyState != "open":
            return {"error": "Data channel not open", "candidates": []}

        # Build message: FS_RESOLVE::pattern::file_size::max_depth::mount_label
        parts = [MSG_FS_RESOLVE, pattern]
        parts.append(str(file_size) if file_size else "")
        parts.append(str(max_depth) if max_depth else "")
        parts.append(mount_label if mount_label else "")
        message = MSG_SEPARATOR.join(parts)

        # Clear any stale responses
        while not self.fs_response_queue.empty():
            try:
                self.fs_response_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Send request
        logging.info(f"Sending FS_RESOLVE: {pattern}")
        self.data_channel.send(message)

        # Wait for response
        try:
            response = await asyncio.wait_for(
                self.fs_response_queue.get(), timeout=timeout
            )

            # Parse response
            if response.startswith(MSG_FS_RESOLVE_RESPONSE):
                json_str = response.split(MSG_SEPARATOR, 1)[1]
                return json.loads(json_str)

            elif response.startswith(MSG_FS_ERROR):
                parts = response.split(MSG_SEPARATOR)
                return {
                    "error_code": parts[1] if len(parts) > 1 else "UNKNOWN",
                    "error": parts[2] if len(parts) > 2 else "Unknown error",
                    "candidates": [],
                }

            else:
                return {
                    "error": f"Unexpected response: {response[:50]}",
                    "candidates": [],
                }

        except asyncio.TimeoutError:
            logging.warning("FS_RESOLVE timed out")
            return {"error": "Resolution timed out", "timeout": True, "candidates": []}

    async def resolve_file_path(
        self,
        local_path: str,
        non_interactive: bool = False,
        mount_label: str = None,
    ) -> Optional[str]:
        """Resolve a local file path to a Worker filesystem path.

        This method:
        1. Fetches available mounts from Worker
        2. Shows mount selector (unless mount_label provided or non_interactive)
        3. Extracts the filename from the local path
        4. Gets local file size if the file exists
        5. Sends FS_RESOLVE to Worker with selected mount
        6. If exact match found, returns it immediately
        7. If multiple matches, shows arrow selector
        8. If no matches, shows options menu

        Args:
            local_path: Local path to the file (may not exist locally).
            non_interactive: If True, auto-select first candidate and default to all mounts.
            mount_label: Specific mount label to search (skips mount selector).

        Returns:
            Resolved Worker path, or None if cancelled/failed.
        """
        if not local_path:
            return None

        # Extract filename from path
        local_path_obj = Path(local_path)
        filename = local_path_obj.name

        # Get local file size if file exists
        file_size = None
        if local_path_obj.exists():
            file_size = local_path_obj.stat().st_size

        # Check if pattern contains wildcards
        has_wildcards = any(c in filename for c in "*?[")

        # Fetch mounts from worker and show selector if needed
        selected_mount = mount_label
        if not selected_mount:
            if non_interactive:
                # Non-interactive mode defaults to "all"
                selected_mount = "all"
            else:
                # Fetch available mounts
                mounts = await self._send_fs_get_mounts()
                if not mounts:
                    print("\nError: No filesystems configured on worker.")
                    print(
                        "Configure mounts in sleap-rtc.toml under [[worker.io.mounts]]"
                    )
                    return None

                # Show mount selector
                selector = MountSelector(mounts)
                selected_mount = selector.run()
                if selected_mount is None or selector.cancelled:
                    logging.info("Mount selection cancelled")
                    return None

        # Send resolution request with selected mount
        result = await self._send_fs_resolve(
            pattern=filename,
            file_size=file_size,
            mount_label=selected_mount,
        )

        # Handle errors
        if "error_code" in result:
            logging.error(f"Resolution error: {result.get('error')}")
            if result.get("error_code") == "PATTERN_TOO_BROAD":
                print(f"\nPattern too broad: {result.get('error')}")
            return None

        if "error" in result and not result.get("candidates"):
            logging.error(f"Resolution failed: {result.get('error')}")
            return None

        candidates = result.get("candidates", [])

        # Case 1: Exact match found
        if candidates and not has_wildcards:
            # Check if first candidate is exact match
            first = candidates[0]
            if first.get("match_type") == "exact" and first.get("name") == filename:
                # If file size matches or no local size, use this path
                if file_size is None or first.get("size") == file_size:
                    logging.info(f"Exact match found: {first.get('path')}")
                    if non_interactive or len(candidates) == 1:
                        print(f"\nUsing exact match: {first.get('path')}")
                        return first.get("path")

        # Case 2: Multiple candidates - show selector
        if candidates:
            selected = select_file_from_candidates(
                candidates,
                title=f"Select file (searching for: {filename}):",
                non_interactive=non_interactive,
            )
            return selected

        # Case 3: No matches - show options menu
        menu = NoMatchMenu(filename)
        action = menu.run()

        if action == "cancel":
            return None

        elif action == "manual":
            return prompt_manual_path()

        elif action == "wildcard":
            pattern = prompt_wildcard_pattern(filename)
            if pattern:
                # Retry with wildcard pattern (use same mount selection)
                result = await self._send_fs_resolve(
                    pattern=pattern, file_size=file_size, mount_label=selected_mount
                )
                candidates = result.get("candidates", [])
                if candidates:
                    return select_file_from_candidates(
                        candidates,
                        title=f"Select file (pattern: {pattern}):",
                        non_interactive=non_interactive,
                    )
                else:
                    print(f"\nNo matches found for pattern: {pattern}")
                    return None
            return None

        elif action == "browse":
            print("\nTo browse the Worker's filesystem, run in a separate terminal:")
            print("  sleap-connect test browse --room <room_id>")
            print("\nCopy the path and use --worker-path to specify it directly.")
            return None

        return None

    async def _send_use_worker_path(
        self, worker_path: str, timeout: float = 15.0
    ) -> dict:
        """Send USE_WORKER_PATH message and wait for Worker response.

        Args:
            worker_path: The resolved path on the Worker filesystem.
            timeout: Timeout in seconds for response.

        Returns:
            dict with "success": True and "path" on success,
            or "success": False and "error" on failure.
        """
        if not self.data_channel or self.data_channel.readyState != "open":
            return {"success": False, "error": "Data channel not open"}

        # Build message
        message = f"{MSG_USE_WORKER_PATH}{MSG_SEPARATOR}{worker_path}"

        # Send request
        logging.info(f"Sending USE_WORKER_PATH: {worker_path}")
        self.data_channel.send(message)

        # Wait for response
        try:
            response = await asyncio.wait_for(
                self.fs_response_queue.get(), timeout=timeout
            )

            # Parse response
            if response.startswith(MSG_WORKER_PATH_OK):
                parts = response.split(MSG_SEPARATOR)
                path = parts[1] if len(parts) > 1 else worker_path
                return {"success": True, "path": path}

            elif response.startswith(MSG_WORKER_PATH_ERROR):
                parts = response.split(MSG_SEPARATOR)
                error = parts[1] if len(parts) > 1 else "Unknown error"
                return {"success": False, "error": error}

            else:
                return {
                    "success": False,
                    "error": f"Unexpected response: {response[:50]}",
                }

        except asyncio.TimeoutError:
            logging.warning("USE_WORKER_PATH timed out")
            return {"success": False, "error": "Worker path validation timed out"}

    async def _handle_video_resolution(
        self, slp_path: str, timeout: float = 30.0
    ) -> str:
        """Handle video accessibility check and resolution for SLP files.

        After WORKER_PATH_OK, the Worker checks video accessibility and sends
        FS_CHECK_VIDEOS_RESPONSE. If videos are missing, this launches the
        resolution UI and waits for the user to resolve paths.

        Args:
            slp_path: Path to SLP file on Worker filesystem.
            timeout: Timeout for video check response.

        Returns:
            The final SLP path to use (original or corrected), or None if cancelled.
        """
        logging.info(f"Waiting for video accessibility check for: {slp_path}")

        try:
            # Wait for FS_CHECK_VIDEOS_RESPONSE from Worker
            response = await asyncio.wait_for(
                self.fs_response_queue.get(),
                timeout=timeout,
            )

            # Check if this is a video check response
            if not response.startswith(
                f"{MSG_FS_CHECK_VIDEOS_RESPONSE}{MSG_SEPARATOR}"
            ):
                logging.warning(f"Expected video check response, got: {response[:50]}")
                # Put it back and proceed with original path
                await self.fs_response_queue.put(response)
                return slp_path

            # Parse the video check response
            json_str = response.split(MSG_SEPARATOR, 1)[1]
            video_data = json.loads(json_str)
            missing = video_data.get("missing", [])
            total_videos = video_data.get("total_videos", 0)
            accessible = video_data.get("accessible", 0)

            if not missing:
                logging.info(f"All {total_videos} videos are accessible")
                print(f"\nAll {total_videos} video(s) are accessible on Worker.")
                return slp_path

            # Videos are missing - launch resolution UI
            print(f"\nFound {len(missing)} missing video(s) out of {total_videos}:")
            for v in missing[:5]:
                print(f"  - {v.get('filename')}")
            if len(missing) > 5:
                print(f"  ... and {len(missing) - 5} more")

            print("\nLaunching video path resolution UI...")

            # Create and start resolution server
            from sleap_connect.client.fs_viewer_server import FSViewerServer

            viewer_server = FSViewerServer(
                send_to_worker=lambda msg: self.data_channel.send(msg),
            )
            viewer_server.set_video_check_data(video_data)

            # Track resolution completion
            resolution_complete = asyncio.Event()
            resolved_slp_path = slp_path

            # Register callback for resolution completion
            async def on_write_slp_response(message: str):
                nonlocal resolved_slp_path
                if message.startswith("FS_WRITE_SLP_OK::"):
                    json_str = message.split("::", 1)[1]
                    data = json.loads(json_str)
                    resolved_slp_path = data.get("output_path", slp_path)
                    logging.info(f"Resolution complete, new SLP: {resolved_slp_path}")
                    resolution_complete.set()
                elif message.startswith("FS_WRITE_SLP_ERROR::"):
                    logging.error(f"SLP write failed: {message}")
                    # Keep original path

            # Hook into worker response handling
            original_handler = viewer_server.on_worker_response

            async def wrapped_handler(message: str):
                if original_handler:
                    original_handler(message)
                await on_write_slp_response(message)

            viewer_server.on_worker_response = wrapped_handler

            try:
                # Start server
                url = await viewer_server.start(port=8765, open_browser=False)
                resolve_url = url.replace("/?", "/resolve?")

                print(f"\nResolution UI: {resolve_url}")
                print("Press Ctrl+C to skip video resolution.\n")

                # Open browser
                import webbrowser

                webbrowser.open(resolve_url)

                # Forward FS_* messages to viewer server while waiting
                async def forward_fs_messages():
                    while not resolution_complete.is_set():
                        try:
                            msg = await asyncio.wait_for(
                                self.fs_response_queue.get(),
                                timeout=1.0,
                            )
                            if msg.startswith("FS_"):
                                await viewer_server.handle_worker_response(msg)
                                await on_write_slp_response(msg)
                        except asyncio.TimeoutError:
                            pass

                forward_task = asyncio.create_task(forward_fs_messages())

                # Wait for resolution or timeout (10 minutes)
                try:
                    await asyncio.wait_for(resolution_complete.wait(), timeout=600.0)
                    print(f"\nResolution complete. Using: {resolved_slp_path}")

                    # If a new SLP file was created, tell the Worker to use it
                    if resolved_slp_path != slp_path:
                        logging.info(
                            f"Updating Worker path to resolved SLP: {resolved_slp_path}"
                        )
                        self.data_channel.send(
                            f"{MSG_USE_WORKER_PATH}{MSG_SEPARATOR}{resolved_slp_path}"
                        )

                        # Wait for Worker to acknowledge the new path
                        try:
                            response = await asyncio.wait_for(
                                self.fs_response_queue.get(),
                                timeout=10.0,
                            )
                            if response.startswith(
                                f"{MSG_WORKER_PATH_OK}{MSG_SEPARATOR}"
                            ):
                                logging.info("Worker accepted resolved SLP path")
                            elif response.startswith(
                                f"{MSG_WORKER_PATH_ERROR}{MSG_SEPARATOR}"
                            ):
                                error = (
                                    response.split(MSG_SEPARATOR, 1)[1]
                                    if MSG_SEPARATOR in response
                                    else "Unknown error"
                                )
                                logging.error(f"Worker rejected resolved path: {error}")
                                print(
                                    f"\nWarning: Worker rejected resolved path: {error}"
                                )
                                print("Continuing with original SLP path.")
                                resolved_slp_path = slp_path
                            else:
                                # Not the expected response, put it back
                                await self.fs_response_queue.put(response)
                        except asyncio.TimeoutError:
                            logging.warning(
                                "Timeout waiting for Worker to accept resolved path"
                            )
                            print("\nWarning: Worker did not respond to path update.")

                except asyncio.TimeoutError:
                    print("\nResolution timed out. Using original SLP path.")
                finally:
                    forward_task.cancel()

            finally:
                await viewer_server.stop()

            return resolved_slp_path

        except asyncio.TimeoutError:
            logging.warning("Video check timed out, proceeding with original path")
            print(
                "\nVideo accessibility check timed out. Proceeding with original path."
            )
            return slp_path
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse video check response: {e}")
            return slp_path
        except Exception as e:
            logging.error(f"Video resolution error: {e}")
            return slp_path

    async def send_client_file(self, file_path: str = None, output_dir: str = ""):
        """Handles direct, one-way file transfer from client to be sent to worker peer.

        Args:
            None
        Returns:
            None
        """
        # Check channel state before sending file.
        if self.data_channel.readyState != "open":
            logging.info(
                f"Data channel not open. Ready state is: {self.data_channel.readyState}"
            )
            return

        # Send file to worker.
        logging.info(f"Given file path {file_path}")
        if not file_path:
            logging.info("No file path entered.")
            return
        if not Path(file_path).exists():
            logging.info("File does not exist.")
            return
        else:
            logging.info(f"Sending {file_path} to worker...")

            # Send package type indicator
            self.data_channel.send("PACKAGE_TYPE::train")

            # Send output directory (where models will be saved).
            output_dir = "models"
            if self.config_info_list:
                output_dir = self.config_info_list[0].config.outputs.runs_folder

            self.data_channel.send(f"OUTPUT_DIR::{output_dir}")

            # Obtain file metadata.
            file_name = Path(file_path).name
            file_size = Path(file_path).stat().st_size

            # Send metadata next.
            self.data_channel.send(f"FILE_META::{file_name}:{file_size}:{self.gui}")

            # Send file in chunks (32 KB).
            with open(file_path, "rb") as file:
                logging.info(f"File opened: {file_path}")
                while chunk := file.read(CHUNK_SIZE):
                    while (
                        self.data_channel.bufferedAmount is not None
                        and self.data_channel.bufferedAmount > 16 * 1024 * 1024
                    ):  # Wait if buffer >16MB
                        await asyncio.sleep(0.1)

                    self.data_channel.send(chunk)

            self.data_channel.send("END_OF_FILE")
            logging.info(f"File sent to worker.")

        # Start ZMQ control socket.
        if self.gui:
            self.start_zmq_control()
            asyncio.create_task(self.start_zmq_listener(self.data_channel))
            logging.info(f"{self.data_channel.label} ZMQ control socket started")

        return

    async def on_channel_open(self):
        """Event handler function for when the datachannel is open.

        Args:
            None
        Returns:
            None
        """
        # Initiate keep-alive task.
        asyncio.create_task(self.keep_ice_alive())
        logging.debug(f"{self.data_channel.label} is open")

        # Visual status: Connected
        if self.verbosity != "quiet":
            print("✓ Connected", flush=True)

        # Wait for P2P PSK authentication if configured
        if self._room_secret:
            logging.info("Waiting for P2P PSK authentication...")
            auth_success = await self._wait_for_auth(timeout=15.0)
            if not auth_success:
                reason = self._auth_failed_reason or "timeout"
                logging.error(f"P2P authentication failed: {reason}")
                print(f"\nError: P2P authentication failed: {reason}")
                print("Check that the room secret matches the worker's configuration.")
                return

        # Handle structured job submission if job_spec is set
        if self.job_spec:
            logging.info(
                f"Submitting structured job: {self.job_spec.__class__.__name__}"
            )
            await self.submit_job()
            return

        # Determine the worker path to use
        resolved_path = None

        if self.worker_path:
            # Use explicitly provided worker path (--worker-path flag)
            logging.info(f"Using explicit worker path: {self.worker_path}")
            resolved_path = self.worker_path
        elif self.file_path:
            # Try to resolve the local file path on the Worker
            logging.info(f"Resolving file path: {self.file_path}")
            resolved_path = await self.resolve_file_path(
                self.file_path,
                non_interactive=self.non_interactive,
                mount_label=self.mount_label,
            )

        if resolved_path:
            # Send worker path to Worker and wait for validation
            logging.info(f"Requesting Worker to use path: {resolved_path}")
            result = await self._send_use_worker_path(resolved_path)

            if result.get("success"):
                accepted_path = result.get("path", resolved_path)
                logging.info(f"Worker accepted path: {accepted_path}")

                # For SLP files, wait for video check response
                if accepted_path.lower().endswith(".slp"):
                    final_path = await self._handle_video_resolution(accepted_path)
                    if final_path is None:
                        logging.info("Video resolution cancelled or failed")
                        return
                    accepted_path = final_path

                # Send package type and output directory
                self.data_channel.send("PACKAGE_TYPE::train")

                output_dir = "models"
                if self.config_info_list:
                    output_dir = self.config_info_list[0].config.outputs.runs_folder
                self.data_channel.send(f"OUTPUT_DIR::{output_dir}")

                # Start ZMQ control socket if GUI
                if self.gui:
                    self.start_zmq_control()
                    asyncio.create_task(self.start_zmq_listener(self.data_channel))
                    logging.info(
                        f"{self.data_channel.label} ZMQ control socket started"
                    )
                return
            else:
                logging.error(f"Worker rejected path: {result.get('error')}")
                print(f"\nError: Worker could not use path: {result.get('error')}")
                # No fallback to RTC transfer per user requirement
                return

        # No resolved path - try RTC transfer as fallback only if no path resolution was attempted
        if self.file_path and not self.worker_path:
            # Path resolution failed or was cancelled
            logging.info("Path resolution failed, aborting")
            print(
                "\nPath resolution failed. Use --worker-path to specify the path directly."
            )
        else:
            logging.info("No file path provided")
        return

    # ===== P2P PSK Authentication Methods =====

    def _handle_auth_challenge(self, message: str) -> None:
        """Handle AUTH_CHALLENGE message from worker.

        Responds with an Ed25519 signature if a private key is stored, falls
        back to PSK HMAC if a room secret is configured, or sends "missing".

        Args:
            message: The AUTH_CHALLENGE message (format: AUTH_CHALLENGE::{nonce}).
        """
        # Parse the nonce from the message
        parts = message.split(MSG_SEPARATOR, 1)
        if len(parts) != 2:
            logging.warning("Invalid AUTH_CHALLENGE format")
            return

        nonce = parts[1]
        logging.info("Received AUTH_CHALLENGE from worker")

        if self.data_channel is None or self.data_channel.readyState != "open":
            return

        # Path A: Ed25519 (new architecture) — sign with stored private key
        from sleap_connect.auth.credentials import get_private_key_b64
        from sleap_connect.auth.keypair import private_key_from_b64, sign_nonce

        priv_b64 = get_private_key_b64()
        if priv_b64:
            try:
                private_key = private_key_from_b64(priv_b64)
                signature = sign_nonce(private_key, nonce)
                self.data_channel.send(format_message(MSG_AUTH_RESPONSE, signature))
                logging.info("Sent Ed25519 AUTH_RESPONSE to worker")
                return
            except Exception as e:
                logging.warning(f"Ed25519 signing failed, falling back to PSK: {e}")

        # Path B: PSK (room_secret configured) — backward compat
        if self._room_secret:
            hmac_response = compute_hmac(self._room_secret, nonce)
            self.data_channel.send(format_message(MSG_AUTH_RESPONSE, hmac_response))
            logging.info("Sent PSK AUTH_RESPONSE to worker")
            return

        # No credentials — tell worker
        logging.warning(
            "No auth credentials configured, sending AUTH_RESPONSE::missing"
        )
        self.data_channel.send(format_message(MSG_AUTH_RESPONSE, "missing"))

    def _handle_auth_success(self) -> None:
        """Handle AUTH_SUCCESS message from worker.

        Marks the connection as authenticated and signals waiting tasks.
        """
        logging.info("P2P authentication successful")
        self._authenticated = True
        self._auth_failed_reason = None
        self._auth_event.set()

    def _handle_auth_failure(self, message: str) -> None:
        """Handle AUTH_FAILURE message from worker.

        Logs the failure reason and signals waiting tasks.

        Args:
            message: The AUTH_FAILURE message (format: AUTH_FAILURE::{reason}).
        """
        # Parse the reason from the message
        parts = message.split(MSG_SEPARATOR, 1)
        reason = parts[1] if len(parts) > 1 else "unknown"

        logging.error(f"P2P authentication failed: {reason}")
        self._authenticated = False
        self._auth_failed_reason = reason
        self._auth_event.set()

    async def _wait_for_auth(self, timeout: float = 15.0) -> bool:
        """Wait for PSK authentication to complete.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if authenticated successfully, False otherwise.
        """
        if not self._room_secret:
            # No secret configured, consider authenticated (legacy mode)
            return True

        try:
            await asyncio.wait_for(self._auth_event.wait(), timeout=timeout)
            return self._authenticated
        except asyncio.TimeoutError:
            logging.error("P2P authentication timed out waiting for response")
            return False

    async def on_message(self, message):
        """Event handler function for when a message is received on the datachannel from Worker.

        Args:
            message: The received message, either as a string or bytes.

        Returns:
            None
        """
        # Log the received message in verbose mode only (truncate for readability).
        log_msg = message if len(str(message)) < 100 else f"{str(message)[:100]}..."
        logging.debug(f"Client received: {log_msg}")

        # Handle string and bytes messages differently.
        if isinstance(message, str):
            # Handle P2P PSK authentication messages (highest priority)
            if message.startswith(MSG_AUTH_CHALLENGE):
                self._handle_auth_challenge(message)
                return
            if message == MSG_AUTH_SUCCESS:
                self._handle_auth_success()
                return
            if message.startswith(MSG_AUTH_FAILURE):
                self._handle_auth_failure(message)
                return

            # Handle structured job submission responses (JOB_*)
            if message.startswith("JOB_"):
                await self.job_response_queue.put(message)
                return

            # Handle file upload responses (FILE_UPLOAD_*)
            if message.startswith("FILE_UPLOAD_"):
                await self.upload_response_queue.put(message)
                return

            # Handle filesystem browser responses (FS_*)
            if message.startswith("FS_"):
                logging.info(f"Received FS response: {message[:50]}...")

                # Special handling for video accessibility check response
                if message.startswith(f"{MSG_FS_CHECK_VIDEOS_RESPONSE}{MSG_SEPARATOR}"):
                    try:
                        json_str = message.split(MSG_SEPARATOR, 1)[1]
                        video_data = json.loads(json_str)
                        missing = video_data.get("missing", [])
                        if missing:
                            logging.info(
                                f"Video check found {len(missing)} missing video(s)"
                            )
                            self.pending_video_check_data = video_data
                            # Trigger callback to launch resolution UI if registered
                            if self.on_missing_videos_detected:
                                await self.on_missing_videos_detected(video_data)
                        else:
                            logging.info("All videos accessible, no resolution needed")
                    except (json.JSONDecodeError, IndexError) as e:
                        logging.error(f"Failed to parse video check response: {e}")

                await self.fs_response_queue.put(message)
                return

            # Handle worker path responses (WORKER_PATH_OK, WORKER_PATH_ERROR)
            if message.startswith("WORKER_PATH_"):
                logging.info(f"Received worker path response: {message[:50]}...")
                await self.fs_response_queue.put(message)
                return

            if message == "END_OF_FILE":
                # File transfer complete, save to disk.
                logging.info("File transfer complete. Saving file to disk...")
                file_name, file_data = list(self.received_files.items())[0]

                try:
                    os.makedirs(self.output_dir, exist_ok=True)
                    file_path = Path(self.output_dir).joinpath(file_name)

                    with open(file_path, "wb") as file:
                        file.write(file_data)
                    logging.info(f"File saved as: {file_path}")
                except PermissionError:
                    logging.error(f"Permission denied when writing to: {output_dir}")
                except Exception as e:
                    logging.error(f"Failed to save file: {e}")

                self.received_files.clear()

                # Update monitor window with file transfer and training completion.
                if self.gui:
                    close_msg = {
                        "event": "rtc_close_monitor",
                    }
                    self.ctrl_socket.send_string(jsonpickle.encode(close_msg))
                    logging.info("Sent ZMQ close message to LossViewer window.")

            elif "PROGRESS_REPORT::" in message:
                # Progress report received from worker.
                logging.debug(message)
                _, progress = message.split("PROGRESS_REPORT::", 1)

                # Update LossViewer window with received progress report.
                if self.gui:
                    rtc_progress_msg = {
                        "event": "rtc_update_monitor",
                        "rtc_msg": progress,
                    }
                    self.ctrl_socket.send_string(jsonpickle.encode(rtc_progress_msg))

            elif "FILE_META::" in message:
                # Metadata received (file name & size).
                _, meta = message.split("FILE_META::", 1)
                file_name, file_size, output_dir = meta.split(":")

                # Initialize received_files with file name as key and empty bytearray as value.
                if file_name not in self.received_files:
                    self.received_files[file_name] = bytearray()
                logging.info(
                    f"File name received: {file_name}, of size {file_size}, saving to {output_dir}"
                )

            elif "ZMQ_CTRL::" in message:
                # ZMQ control message received.
                _, zmq_ctrl = message.split("ZMQ_CTRL::", 1)

                # Handle ZMQ control messages (i.e. STOP or CANCEL).

            elif "TRAIN_JOB_START::" in message:
                # Training job start message received.
                _, job_info = message.split("TRAIN_JOB_START::", 1)
                logging.info(f"Training job started with info: {job_info}")

                # Parse the job info and update the LossViewer window.
                if self.gui:
                    try:
                        # Get next config info.
                        config_info = self.config_info_list.pop(0)

                        # Check for retraining flag.
                        if config_info.dont_retrain:
                            if not config_info.has_trained_model:
                                raise ValueError(
                                    "Config is set to not retrain but no trained model found: "
                                    f"{config_info.path}"
                                )

                            logging.info(
                                f"Using already trained model for {config_info.head_name}: "
                                f"{config_info.path}"
                            )

                            # Trained job paths not needed because no remote inference (remote training only) so far.

                        # Otherwise, prepare to run training job.
                        else:
                            logging.info("Resetting monitor window.")
                            job = config_info.config
                            model_type = config_info.head_name
                            plateau_patience = (
                                job.optimization.early_stopping.plateau_patience
                            )
                            plateau_min_delta = (
                                job.optimization.early_stopping.plateau_min_delta
                            )

                            # Send reset ZMQ message to LossViewer window.
                            # In separate thread from LossViewer, must use ZMQ PUB socket to update.
                            reset_msg = {
                                "event": "rtc_reset_monitor",
                                "what": str(model_type),
                                "plateau_patience": plateau_patience,
                                "plateau_min_delta": plateau_min_delta,
                                "window_title": f"Training Model - {str(model_type)}",
                                "message": "Preparing to run training...",
                            }
                            self.ctrl_socket.send_string(jsonpickle.encode(reset_msg))

                            # Further updates to the LossViewer window handled by PROGRESS_REPORT messages when training starts remotely.

                            logging.info(
                                f"Start training {str(model_type)} job with config: {job}"
                            )

                    except Exception as e:
                        logging.error(f"Failed to parse training job config: {e}")

            elif (
                "TRAIN_JOB_END::" in message
            ):  # ONLY TO SIGNAL TRAINING JOB END, NOT WHOLE TRAINING SESSION END.
                # Training job end message received.
                _, job_info = message.split("TRAIN_JOB_END::", 1)
                logging.info(
                    f"Train job completed: {job_info}, checking for next job..."
                )

                # Update LossViewer window to indicate training completion based on how many training jobs left.
                if self.gui:
                    if len(self.config_info_list) == 0:
                        logging.info(
                            "No more training jobs to run. Closing LossViewer window."
                        )
                    else:
                        logging.info(
                            f"More training jobs to run: {len(self.config_info_list)} remaining."
                        )

            elif "TRAIN_JOB_ERROR::" in message:
                # Training job error message received.
                _, error_info = message.split("TRAIN_JOB_ERROR::", 1)
                logging.error(f"Training job encountered an error: {error_info}")

            else:
                # Unhandled string message - likely raw training/inference output
                print(message, end="", flush=True)

        elif isinstance(message, bytes):
            if message == b"KEEP_ALIVE":
                logging.debug("Keep alive message received.")
                return

            elif b"PROGRESS_REPORT::" in message:
                # Progress report received from worker as bytes.
                logging.debug(message.decode())
                _, progress = message.decode().split("PROGRESS_REPORT::", 1)

                # Update LossViewer window with received progress report.
                if self.gui:
                    rtc_progress_msg = {
                        "event": "rtc_update_monitor",
                        "rtc_msg": progress,
                    }
                    self.ctrl_socket.send_string(jsonpickle.encode(rtc_progress_msg))

            file_name = list(self.received_files.keys())[0]
            if file_name not in self.received_files:
                self.received_files[file_name] = bytearray()
            self.received_files[file_name].extend(message)

    # @pc.on("iceconnectionstatechange")
    async def _send_peer_message(self, to_peer_id: str, payload: dict):
        """Send peer message via signaling server.

        Args:
            to_peer_id: Target peer ID
            payload: Application-specific message payload
        """
        try:
            await self.websocket.send(
                json.dumps(
                    {
                        "type": "peer_message",
                        "from_peer_id": self.peer_id,
                        "to_peer_id": to_peer_id,
                        "payload": payload,
                    }
                )
            )
            logging.info(
                f"Sent peer message to {to_peer_id}: {payload.get('app_message_type', 'unknown')}"
            )
        except Exception as e:
            logging.error(f"Failed to send peer message: {e}")

    async def discover_workers(
        self, room_id: str = None, **filter_requirements
    ) -> list:
        """Discover available workers matching requirements (DEPRECATED: use _discover_workers_in_room).

        NOTE: This method is deprecated. For room-based discovery, use _discover_workers_in_room() instead.
        This method is kept for backward compatibility but now requires room_id for security.

        Args:
            room_id: Room ID to scope discovery (REQUIRED for security)
            **filter_requirements: Keyword arguments for filtering
                - min_gpu_memory_mb: Minimum GPU memory
                - model_type: Required model support
                - job_type: "training" or "inference"

        Returns:
            List of worker peer info dicts
        """
        # SECURITY: Require room_id to prevent global discovery
        if not room_id:
            logging.error(
                "SECURITY: room_id required for worker discovery. Global discovery is disabled."
            )
            return []

        # Build filters with room_id for security
        filters = {
            "role": "worker",
            "room_id": room_id,  # SECURITY: Always scope to room
            "tags": ["sleap-connect"],
            "properties": {"status": "available"},
        }

        # Add GPU memory requirement
        if "min_gpu_memory_mb" in filter_requirements:
            filters["properties"]["gpu_memory_mb"] = {
                "$gte": filter_requirements["min_gpu_memory_mb"]
            }

        # Add job type requirement
        if "job_type" in filter_requirements:
            job_type = filter_requirements["job_type"]
            if job_type == "training":
                filters["tags"].append("training-worker")
            elif job_type == "inference":
                filters["tags"].append("inference-worker")

        # Try new discovery API
        try:
            # Send discovery request
            await self.websocket.send(
                json.dumps(
                    {
                        "type": "discover_peers",
                        "from_peer_id": self.peer_id,
                        "filters": filters,
                    }
                )
            )

            # Wait for response from queue (routed by handle_connection)
            workers = await asyncio.wait_for(self.peer_list_queue.get(), timeout=5.0)

            logging.info(
                f"Discovered {len(workers)} available workers in room {room_id}"
            )
            return workers

        except asyncio.TimeoutError:
            logging.warning("Discovery timed out, falling back to manual peer_id")
            return await self._fallback_manual_peer_selection()
        except Exception as e:
            logging.error(f"Discovery error: {e}, falling back to manual peer_id")
            return await self._fallback_manual_peer_selection()

    async def _fallback_manual_peer_selection(self) -> list:
        """Fallback: Use manually configured worker peer_id.

        Returns:
            List with single worker entry if configured, empty list otherwise
        """
        # Check if we have a target worker already set (from session string)
        if self.target_worker:
            logging.info(f"Using target worker from session: {self.target_worker}")
            return [{"peer_id": self.target_worker, "role": "worker", "metadata": {}}]

        # Check environment variable
        worker_peer_id = os.environ.get("SLEAP_RTC_WORKER_PEER_ID")

        if worker_peer_id:
            logging.info(f"Using worker from environment: {worker_peer_id}")
            return [{"peer_id": worker_peer_id, "role": "worker", "metadata": {}}]

        # No fallback available
        logging.error("No workers discovered and no fallback peer_id available")
        return []

    async def _register_with_room(
        self, room_id: str, token: str, jwt_token: str = None
    ):
        """Register client with room on signaling server.

        Args:
            room_id: Room ID to join
            token: Room authentication token
            jwt_token: JWT from GitHub OAuth
        """
        # Try to get SLEAP version
        try:
            import sleap

            sleap_version = sleap.__version__
        except (ImportError, AttributeError):
            sleap_version = "unknown"

        logging.info(f"Registering {self.peer_id} with room {room_id}...")

        # Build registration message
        register_data = {
            "type": "register",
            "peer_id": self.peer_id,
            "room_id": room_id,
            "token": token,
            "role": "client",
            "metadata": {
                "tags": ["sleap-connect", "training-client"],
                "properties": {
                    "sleap_version": sleap_version,
                    "platform": platform.system(),
                    "user_id": os.environ.get("USER", "unknown"),
                },
            },
        }

        # Add JWT authentication
        if jwt_token:
            register_data["jwt"] = jwt_token

        # Send registration message
        await self.websocket.send(json.dumps(register_data))

        # Wait for confirmation from queue (routed by handle_connection)
        try:
            response = await asyncio.wait_for(
                self.registration_queue.get(), timeout=5.0
            )
            if response.get("type") == "registered_auth":
                logging.info(
                    f"Client {self.peer_id} successfully registered with room {room_id}"
                )
                # Store ICE servers from signaling server response
                self.ice_servers = response.get("ice_servers", [])
                if self.ice_servers:
                    logging.info(
                        f"Received {len(self.ice_servers)} ICE server(s) from signaling server"
                    )
                else:
                    logging.warning("No ICE servers received from signaling server")
            else:
                logging.warning(f"Unexpected registration response: {response}")
        except asyncio.TimeoutError:
            logging.error("Registration confirmation timeout")
        except Exception as e:
            logging.error(f"Registration error: {e}")

    async def _discover_workers_in_room(
        self, room_id: str, min_gpu_memory: int = None
    ) -> list:
        """Discover available workers in the specified room.

        Args:
            room_id: Room ID to search for workers
            min_gpu_memory: Minimum GPU memory in MB (optional filter)

        Returns:
            List of worker peer info dicts
        """
        # Build filters for room-scoped discovery
        filters = {
            "role": "worker",
            "room_id": room_id,  # CRITICAL: Scope discovery to this room only
            "tags": ["sleap-connect"],
            "properties": {"status": "available"},
        }

        # Add GPU memory filter if specified
        if min_gpu_memory:
            filters["properties"]["gpu_memory_mb"] = {"$gte": min_gpu_memory}

        logging.info(f"Discovering workers in room {room_id}...")

        try:
            # Send discovery request
            await self.websocket.send(
                json.dumps(
                    {
                        "type": "discover_peers",
                        "from_peer_id": self.peer_id,
                        "filters": filters,
                    }
                )
            )

            # Wait for response from queue (routed by handle_connection)
            workers = await asyncio.wait_for(self.peer_list_queue.get(), timeout=5.0)

            logging.info(f"Discovered {len(workers)} available workers in room")
            return workers

        except asyncio.TimeoutError:
            logging.error("Worker discovery timed out")
            return []
        except Exception as e:
            logging.error(f"Worker discovery error: {e}")
            return []

    def _auto_select_worker(self, workers: list) -> str:
        """Automatically select best worker based on GPU memory.

        Args:
            workers: List of worker peer info dicts

        Returns:
            Selected worker peer_id
        """
        if not workers:
            raise ValueError("No workers available for auto-selection")

        # Sort by GPU memory (descending)
        sorted_workers = sorted(
            workers,
            key=lambda w: w.get("metadata", {})
            .get("properties", {})
            .get("gpu_memory_mb", 0),
            reverse=True,
        )

        selected = sorted_workers[0]
        peer_id = selected["peer_id"]
        metadata = selected.get("metadata", {}).get("properties", {})
        gpu_memory = metadata.get("gpu_memory_mb", "unknown")

        logging.info(f"Auto-selected worker {peer_id} (GPU memory: {gpu_memory}MB)")
        logging.info("Worker transfer mode: RTC")

        return peer_id

    async def _prompt_worker_selection(self, workers: list) -> str:
        """Display workers and prompt user to select one using prompt_toolkit.

        Args:
            workers: List of worker peer info dicts

        Returns:
            Selected worker peer_id
        """
        from sleap_connect.client.file_selector import WorkerSelector

        if not workers:
            return None

        selector = WorkerSelector(
            workers=workers,
            title="Select a worker:",
            allow_cancel=True,
        )
        selected = await selector.run()

        if selected is None:
            logging.info("Worker selection cancelled")
            return None

        peer_id = selected.get("peer_id")
        logging.info(f"Selected worker: {peer_id}")
        return peer_id

    async def _collect_job_responses(self, job_id: str, timeout: float) -> list:
        """Collect job responses from workers.

        Args:
            job_id: Job ID to collect responses for
            timeout: Timeout in seconds

        Returns:
            List of job response dicts
        """
        responses = []
        deadline = time.time() + timeout

        # Create queue for this specific job
        self.job_response_queues[job_id] = asyncio.Queue()

        try:
            while time.time() < deadline:
                remaining = deadline - time.time()

                try:
                    # Wait for response from queue (routed by handle_connection)
                    response = await asyncio.wait_for(
                        self.job_response_queues[job_id].get(),
                        timeout=max(0.1, remaining),
                    )

                    # Only collect job_response messages (not progress/started/etc)
                    if isinstance(response, dict) and response.get("worker_id"):
                        responses.append(response)

                except asyncio.TimeoutError:
                    break

        finally:
            # Clean up queue
            if job_id in self.job_response_queues:
                del self.job_response_queues[job_id]

        return responses

    async def _monitor_job_progress(self, job_id: str):
        """Monitor job progress and display updates.

        Args:
            job_id: Job ID to monitor
        """
        # Reuse the same queue as _collect_job_responses if it exists,
        # or create new one
        if job_id not in self.job_response_queues:
            self.job_response_queues[job_id] = asyncio.Queue()

        try:
            while True:
                # Wait for messages from queue (routed by handle_connection)
                payload = await self.job_response_queues[job_id].get()

                # Skip non-dict messages or job_response messages (those are for _collect_job_responses)
                if not isinstance(payload, dict) or payload.get("worker_id"):
                    continue

                app_msg_type = payload.get("app_message_type")

                if app_msg_type == "job_status":
                    # Display progress
                    progress = payload.get("progress", 0)
                    message = payload.get("message", "")
                    details = payload.get("details", {})

                    logging.info(f"Job progress: {progress*100:.1f}% - {message}")
                    if details:
                        logging.info(f"Details: {details}")

                elif app_msg_type == "job_complete":
                    logging.info("Job completed successfully!")
                    result = payload.get("result", {})
                    logging.info(f"Result: {result}")
                    break

                elif app_msg_type == "job_failed":
                    error = payload.get("error", {})
                    logging.error(
                        f"Job failed: {error.get('message', 'Unknown error')}"
                    )
                    raise JobFailedError(
                        f"Job failed with code {error.get('code', 'UNKNOWN')}: {error.get('message', 'Unknown error')}"
                    )

        finally:
            # Clean up queue when done
            if job_id in self.job_response_queues:
                del self.job_response_queues[job_id]

    async def submit_training_job(
        self, dataset_path: str, config: dict, room_id: str = None, **job_requirements
    ):
        """Submit training job to available workers (DEPRECATED: use room-based connection flow).

        NOTE: This method is deprecated. Use the room-based connection flow with
        _discover_workers_in_room() instead.

        Args:
            dataset_path: Path to training dataset
            config: Training configuration
            room_id: Room ID to scope worker discovery (REQUIRED for security)
            **job_requirements: Job requirements (min_gpu_memory_mb, etc.)

        Returns:
            Selected worker peer_id

        Raises:
            NoWorkersAvailableError: No workers found matching requirements
            NoWorkersAcceptedError: No workers accepted the job request
        """
        # 1. Discover available workers (room-scoped for security)
        workers = await self.discover_workers(
            room_id=room_id, job_type="training", **job_requirements
        )

        if not workers:
            raise NoWorkersAvailableError(
                f"No workers found matching requirements: {job_requirements}"
            )

        logging.info(f"Found {len(workers)} workers, sending job requests...")

        # 2. Create job request
        job_id = str(uuid.uuid4())

        # Get dataset info
        dataset_size_mb = 0
        if os.path.exists(dataset_path):
            dataset_size_mb = os.path.getsize(dataset_path) / (1024 * 1024)

        job_request = {
            "app_message_type": "job_request",
            "job_id": job_id,
            "job_type": "training",
            "dataset_info": {
                "format": "slp",
                "path": dataset_path,
                "estimated_size_mb": dataset_size_mb,
            },
            "config": config,
            "requirements": job_requirements,
        }

        # 3. Send job request to all discovered workers
        for worker in workers:
            await self._send_peer_message(worker["peer_id"], job_request)

        # 4. Collect responses (with timeout)
        responses = await self._collect_job_responses(job_id, timeout=5.0)

        if not responses:
            raise NoWorkersAcceptedError(
                "No workers responded to job request (timeout)"
            )

        # Filter accepted responses
        accepted = [r for r in responses if r["accepted"]]

        if not accepted:
            reasons = [r["reason"] for r in responses if not r["accepted"]]
            raise NoWorkersAcceptedError(
                f"All workers rejected job. Reasons: {reasons}"
            )

        # 5. Select best worker (e.g., fastest estimated time)
        selected = min(
            accepted, key=lambda r: r.get("estimated_duration_minutes", 999999)
        )

        logging.info(
            f"Selected worker: {selected['worker_id']} "
            f"(estimate: {selected['estimated_duration_minutes']} min)"
        )

        # 6. Send job assignment to selected worker
        await self._send_peer_message(
            selected["worker_id"],
            {
                "app_message_type": "job_assignment",
                "job_id": job_id,
                "initiate_connection": True,
            },
        )

        # Store job info for monitoring
        self.current_job_id = job_id
        self.target_worker = selected["worker_id"]

        # 7. Monitor job progress in background
        asyncio.create_task(self._monitor_job_progress(job_id))

        return selected["worker_id"]

    async def on_iceconnectionstatechange(self):
        """Event handler function for when the ICE connection state changes.

        Args:
            None
        Returns:
            None
        """
        # Log the current ICE connection state.
        logging.debug(f"ICE connection state is now {self.pc.iceConnectionState}")

        # Check the ICE connection state and handle reconnection logic.
        if self.pc.iceConnectionState in ["connected", "completed"]:
            self.reconnect_attempts = 0
            logging.debug("ICE connection established.")
            logging.info(f"reconnect attempts reset to {self.reconnect_attempts}")

        elif (
            self.pc.iceConnectionState in ["failed", "disconnected", "closed"]
            and not self.reconnecting
        ):
            logging.warning(
                f"ICE connection {self.pc.iceConnectionState}. Attempting reconnect..."
            )
            self.reconnecting = True

            if self.target_worker is None:
                logging.info(
                    f"No target worker available for reconnection. target_worker is {self.target_worker}."
                )
                await self.clean_exit()
                return

            reconnection_success = await self.reconnect()
            self.reconnecting = False
            if not reconnection_success:
                logging.info("Reconnection failed. Closing connection...")
                await self.clean_exit()
                return

    def _create_peer_connection(self) -> RTCPeerConnection:
        """Create RTCPeerConnection with ICE servers from signaling server.

        Returns:
            RTCPeerConnection configured with ICE servers.
        """
        if self.ice_servers:
            # Build RTCConfiguration from ICE servers
            ice_server_objects = []
            for server in self.ice_servers:
                ice_server_objects.append(RTCIceServer(**server))

            config = RTCConfiguration(iceServers=ice_server_objects)
            logging.info(
                f"Creating RTCPeerConnection with {len(ice_server_objects)} ICE server(s)"
            )
            pc = RTCPeerConnection(configuration=config)
        else:
            # Fallback to default (no ICE servers)
            logging.warning(
                "No ICE servers configured, using default RTCPeerConnection"
            )
            pc = RTCPeerConnection()

        # Register ICE connection state change handler
        pc.on("iceconnectionstatechange", self.on_iceconnectionstatechange)
        return pc

    async def submit_job(
        self, allow_path_correction: bool = True, show_confirmation: bool = True
    ):
        """Submit a structured job spec and wait for completion.

        Sends JOB_SUBMIT message with the job spec, then handles all job
        protocol messages until completion or failure.

        Args:
            allow_path_correction: If True, offer to correct paths when
                JOB_REJECTED is received with path errors.
            show_confirmation: If True, show confirmation screen before first submission.

        Requires self.job_spec to be set before calling.
        """
        if not self.job_spec:
            logging.error("No job spec set for submission")
            return

        if not self.data_channel or self.data_channel.readyState != "open":
            logging.error("Data channel not ready for job submission")
            return

        # Show confirmation screen before first submission (if interactive)
        if show_confirmation and not self.non_interactive:
            confirmed = await self._show_job_confirmation()
            if not confirmed:
                print("Job submission cancelled.")
                return

        # Generate job ID and store for tracking
        import uuid

        job_id = str(uuid.uuid4())[:8]
        self.current_job_id = job_id

        # Build JOB_SUBMIT message: JOB_SUBMIT::{job_id}::{spec_json}
        spec_json = self.job_spec.to_json()
        submit_msg = format_message(
            MSG_JOB_SUBMIT, f"{job_id}{MSG_SEPARATOR}{spec_json}"
        )

        logging.info(f"Submitting job {job_id}: {self.job_spec.__class__.__name__}")
        self.data_channel.send(submit_msg)

        # Wait for job responses
        while True:
            try:
                response = await asyncio.wait_for(
                    self.job_response_queue.get(),
                    timeout=600.0,  # 10 minute timeout for long-running jobs
                )
            except asyncio.TimeoutError:
                logging.error(f"Job {job_id} timed out waiting for response")
                print(f"\nError: Job timed out after 10 minutes")
                break

            # Parse response
            if response.startswith(MSG_JOB_ACCEPTED):
                parts = response.split(MSG_SEPARATOR, 2)
                accepted_job_id = parts[1] if len(parts) > 1 else "unknown"
                logging.debug(f"Job {accepted_job_id} accepted by worker")

                # Determine job type for display
                job_type_name = (
                    self.job_spec.__class__.__name__ if self.job_spec else "Job"
                )
                is_training = "Train" in job_type_name
                action = "Training" if is_training else "Inference"
                worker_name = getattr(self, "target_worker", None) or "worker"

                # Visual structure: Job started section (green text)
                print(f"\n{'─' * 60}")
                print(f"\033[32mRunning {action} Remotely on {worker_name}...\033[0m")
                print(f"{'─' * 60}\n", flush=True)

            elif response.startswith(MSG_JOB_REJECTED):
                parts = response.split(MSG_SEPARATOR, 2)
                rejected_job_id = parts[1] if len(parts) > 1 else "unknown"
                error_json = parts[2] if len(parts) > 2 else "{}"
                try:
                    error_data = json.loads(error_json)
                    errors = error_data.get("errors", [])
                    logging.error(f"Job {rejected_job_id} rejected: {errors}")

                    # Check for path-related errors and offer correction via confirmation screen
                    if allow_path_correction and not self.non_interactive:
                        # Show confirmation screen with errors highlighted
                        confirmed = await self._show_job_confirmation(errors=errors)
                        if confirmed:
                            # Generate new job ID and resubmit
                            job_id = str(uuid.uuid4())[:8]
                            self.current_job_id = job_id
                            spec_json = self.job_spec.to_json()
                            submit_msg = format_message(
                                MSG_JOB_SUBMIT, f"{job_id}{MSG_SEPARATOR}{spec_json}"
                            )
                            logging.info(f"Resubmitting job {job_id}")
                            self.data_channel.send(submit_msg)
                            continue
                        else:
                            print("Job submission cancelled.")
                    else:
                        print(f"\nJob rejected by worker:")
                        for err in errors:
                            print(
                                f"  - {err.get('field', 'unknown')}: {err.get('message', 'unknown error')}"
                            )
                except json.JSONDecodeError:
                    logging.error(f"Job {rejected_job_id} rejected: {error_json}")
                    print(f"\nJob rejected: {error_json}")
                break

            elif response.startswith(MSG_JOB_PROGRESS):
                parts = response.split(MSG_SEPARATOR, 2)
                progress_job_id = parts[1] if len(parts) > 1 else "unknown"
                progress_data = parts[2] if len(parts) > 2 else ""
                # Print progress to console (this is job output)
                if progress_data:
                    print(progress_data, end="", flush=True)

            elif response.startswith(MSG_JOB_COMPLETE):
                parts = response.split(MSG_SEPARATOR, 2)
                complete_job_id = parts[1] if len(parts) > 1 else "unknown"
                result_json = parts[2] if len(parts) > 2 else "{}"

                # Determine job type for display
                job_type_name = (
                    self.job_spec.__class__.__name__ if self.job_spec else "Job"
                )
                is_training = "Train" in job_type_name
                action = "Training" if is_training else "Inference"

                try:
                    result_data = json.loads(result_json)
                    output_path = result_data.get("output_path", "")
                    logging.debug(f"Job {complete_job_id} completed successfully")
                    # Visual structure: Completion section (green text)
                    print(f"\n{'─' * 60}")
                    print(f"\033[32m✓ Remote {action} Completed Successfully!\033[0m")
                    if output_path:
                        print(f"  Output: {output_path}")
                    print(f"{'─' * 60}")
                except json.JSONDecodeError:
                    logging.debug(f"Job {complete_job_id} completed")
                    print(f"\n{'─' * 60}")
                    print(f"\033[32m✓ Remote {action} Completed Successfully!\033[0m")
                    print(f"{'─' * 60}")
                break

            elif response.startswith(MSG_JOB_FAILED):
                parts = response.split(MSG_SEPARATOR, 2)
                failed_job_id = parts[1] if len(parts) > 1 else "unknown"
                error_msg = parts[2] if len(parts) > 2 else "Unknown error"
                logging.error(f"Job {failed_job_id} failed: {error_msg}")

                # Determine job type for display
                job_type_name = (
                    self.job_spec.__class__.__name__ if self.job_spec else "Job"
                )
                is_training = "Train" in job_type_name
                action = "Training" if is_training else "Inference"

                # Visual structure: Failure section (red text)
                print(f"\n{'─' * 60}")
                print(f"\033[31m✗ Remote {action} Failed: {error_msg}\033[0m")
                print(f"{'─' * 60}")
                break

            else:
                logging.warning(f"Unknown job response: {response[:50]}...")

    async def _show_job_confirmation(self, errors: list = None) -> bool:
        """Show job confirmation screen before submission.

        Displays all paths in the job spec and allows editing any of them
        before submitting. If errors are provided, they are highlighted.

        Args:
            errors: Optional list of validation errors to display.

        Returns:
            True if user confirmed submission, False if cancelled.
        """
        from sleap_connect.client.file_selector import JobSpecConfirmation
        from sleap_connect.client.directory_browser import DirectoryBrowser

        async def browse_callback(
            field_name: str, current_path: str, file_filter: str
        ) -> str:
            """Callback to browse for a new path."""
            import os

            # Determine start path
            start_path = os.path.dirname(current_path) if current_path else "/"
            if not start_path or start_path == current_path:
                start_path = "/"

            # Build title
            title = f"Select path for '{field_name}':"
            if current_path:
                title += f"\n  (current: {current_path})"

            browser = DirectoryBrowser(
                send_message=self.data_channel.send,
                receive_response=self.fs_response_queue.get,
                start_path=start_path,
                file_filter=file_filter,
                title=title,
            )

            return await browser.run()

        confirmation = JobSpecConfirmation(
            job_spec=self.job_spec,
            browse_callback=browse_callback,
            title="Confirm Job Paths" if not errors else "Job Rejected - Correct Paths",
            errors=errors or [],
        )

        return await confirmation.run()

    async def _handle_path_correction(self, errors: list) -> bool:
        """Handle path correction for JOB_REJECTED errors.

        Checks for path-related errors and offers to browse the worker
        filesystem to correct them.

        Args:
            errors: List of error dictionaries from JOB_REJECTED.

        Returns:
            True if path was corrected and job should be resubmitted,
            False otherwise.
        """
        from sleap_connect.client.directory_browser import DirectoryBrowser

        # Path error codes that can be corrected
        PATH_ERROR_CODES = {
            "PATH_NOT_FOUND",
            "ACCESS_DENIED",
            "NOT_ALLOWED",
            "CONFIG_INVALID_TYPE",
        }

        # Find first path error
        path_error = None
        for err in errors:
            code = err.get("code", "")
            if code in PATH_ERROR_CODES:
                path_error = err
                break

        if not path_error:
            return False

        from sleap_connect.client.file_selector import confirm_prompt

        field = path_error.get("field", "unknown")
        invalid_path = path_error.get("path", "")

        confirmed = await confirm_prompt(
            message="Would you like to browse the worker filesystem to find the correct path?",
            details=f"Path error for '{field}': {invalid_path}",
        )

        if not confirmed:
            return False

        # Determine file filter based on field name
        file_filter = None
        if (
            "labels" in field.lower()
            or "data" in field.lower()
            or "slp" in field.lower()
        ):
            file_filter = ".slp"
        elif (
            "config" in field.lower()
            and "train_labels" not in field.lower()
            and "val_labels" not in field.lower()
        ):
            file_filter = ".yaml,.json"  # Support both YAML and JSON configs
        elif "model" in field.lower():
            file_filter = None  # Models are directories

        # Determine start path (parent of invalid path or root)
        import os

        start_path = os.path.dirname(invalid_path) if invalid_path else "/"
        if not start_path or start_path == invalid_path:
            start_path = "/"

        # Create and run browser
        # Build title with both field name and the invalid path
        title = f"Select correct path for '{field}':"
        if invalid_path:
            title += f"\n  (was: {invalid_path})"

        browser = DirectoryBrowser(
            send_message=self.data_channel.send,
            receive_response=self.fs_response_queue.get,
            start_path=start_path,
            file_filter=file_filter,
            title=title,
        )

        selected_path = await browser.run()

        if not selected_path:
            print("Path correction cancelled.")
            return False

        # Update job spec with corrected path
        import re

        # Check for config internal path like config.train_labels_path, config[0].train_labels_path,
        # config.train_labels_path[0], or config[0].train_labels_path[0]
        config_internal_match = re.match(
            r"config(?:\[\d+\])?\.(\w+)(?:\[\d+\])?", field
        )
        if config_internal_match:
            internal_field = config_internal_match.group(1)

            # Map config internal field to job spec override field
            override_mapping = {
                "train_labels_path": "labels_path",
                "val_labels_path": "val_labels_path",
            }
            override_field = override_mapping.get(internal_field)

            if override_field and hasattr(self.job_spec, override_field):
                setattr(self.job_spec, override_field, selected_path)
                print(f"\nSet {override_field} override to: {selected_path}")
            else:
                print(
                    f"Warning: Could not map config internal field '{internal_field}' to job spec"
                )
                return False
        else:
            # Handle indexed fields like "config_path[0]" -> update config_paths[0]
            index_match = re.match(r"(\w+)\[(\d+)\]", field)

            if index_match:
                # Indexed field like config_path[0]
                base_field = index_match.group(1)
                index = int(index_match.group(2))
                # Map singular to plural field name (config_path -> config_paths)
                list_field = (
                    base_field + "s" if not base_field.endswith("s") else base_field
                )

                if hasattr(self.job_spec, list_field):
                    field_list = getattr(self.job_spec, list_field)
                    if isinstance(field_list, list) and index < len(field_list):
                        field_list[index] = selected_path
                        print(f"\nUpdated {list_field}[{index}] to: {selected_path}")
                    else:
                        print(f"Warning: Index {index} out of range for {list_field}")
                        return False
                else:
                    print(
                        f"Warning: Could not find list field '{list_field}' in job spec"
                    )
                    return False
            elif hasattr(self.job_spec, field):
                setattr(self.job_spec, field, selected_path)
                print(f"\nUpdated {field} to: {selected_path}")
            else:
                print(f"Warning: Could not update field '{field}' in job spec")
                return False

        # Resubmit job
        print("Resubmitting job with corrected path...")
        import uuid

        job_id = str(uuid.uuid4())[:8]
        self.current_job_id = job_id

        spec_json = self.job_spec.to_json()
        submit_msg = format_message(
            MSG_JOB_SUBMIT, f"{job_id}{MSG_SEPARATOR}{spec_json}"
        )
        self.data_channel.send(submit_msg)
        return True

    async def run_client(
        self,
        file_path: str = None,
        output_dir: str = ".",
        zmq_ports: list = None,
        config_info_list: list = None,
        session_string: str = None,
        room_id: str = None,
        token: str = None,
        worker_id: str = None,
        auto_select: bool = False,
        min_gpu_memory: int = None,
        worker_path: str = None,
        non_interactive: bool = False,
        mount_label: str = None,
        room_secret: str = None,
        job_spec=None,
        verbosity: str = "default",
    ):
        """Sends initial SDP offer to worker peer and establishes both connection & datachannel to be used by both parties.

        Args:
            file_path: Path to a file to be sent to worker peer (usually zip file)
            output_dir: Directory to save files received from worker peer
            zmq_ports: List of ZMQ ports [controller, publish]
            config_info_list: Config info for updating GUI (None for CLI)
            session_string: Session string for direct connection to specific worker
            room_id: Room ID for room-based worker discovery
            token: Room token for authentication
            worker_id: Specific worker peer-id to connect to (skips discovery)
            auto_select: Automatically select best worker by GPU memory
            min_gpu_memory: Minimum GPU memory in MB for worker filtering
            worker_path: Explicit path on worker filesystem (skips resolution)
            non_interactive: Auto-select best match without prompting (for CI/scripts)
            mount_label: Specific mount label to search (skips mount selection)
            room_secret: Optional room secret for P2P PSK authentication. If not provided,
                will try to resolve from SLEAP_ROOM_SECRET env var, filesystem, or config.
            job_spec: Optional TrainJobSpec or TrackJobSpec for structured job submission.
                If provided, uses the new job submission protocol instead of file transfer.

        Returns:
            None
        """
        try:
            # Set local variables (peer connection and data channel created after registration)
            self.file_path = file_path
            self.output_dir = output_dir
            self.zmq_ports = zmq_ports
            self.config_info_list = config_info_list  # only passed if not CLI
            # Path resolution options
            self.worker_path = worker_path
            self.non_interactive = non_interactive
            self.mount_label = mount_label
            self.room_id = room_id

            # Verbosity control
            self.verbosity = verbosity

            # Structured job submission
            self.job_spec = job_spec

            # Initialize reconnect attempts.
            logging.info("Setting up RTC data channel reconnect attempts...")
            self.reconnect_attempts = 0

            # JWT authentication (required)
            from sleap_connect.auth.credentials import get_valid_jwt, get_user

            jwt_token = get_valid_jwt()
            if not jwt_token:
                logging.error(
                    "No valid JWT found. Run 'sleap-connect login' to authenticate."
                )
                return

            user = get_user()
            self.peer_id = user.get("username", "unknown") if user else "unknown"
            self.cognito_username = self.peer_id
            logging.info(f"Using JWT authentication as: {self.peer_id}")

            # Store auth token for registration
            self._jwt_token = jwt_token
            self._id_token = None

            # Load room secret for P2P PSK authentication
            # Priority: CLI flag > env var > filesystem > credentials
            self._room_secret = resolve_secret(
                room_id or "default", cli_secret=room_secret
            )
            if self._room_secret:
                logging.info("P2P PSK authentication enabled (room secret configured)")
            else:
                logging.info(
                    "P2P PSK authentication disabled (no room secret configured)"
                )

            # Initiate the WebSocket connection to the signaling server.
            async with websockets.connect(self.DNS) as websocket:

                # Initiate the websocket for the GUI client (so other functions can use).
                self.websocket = websocket

                # START handle_connection as background task to route messages to queues
                connection_task = asyncio.create_task(self.handle_connection())

                try:
                    # BRANCH 1: Session string (direct connection to specific worker)
                    if session_string:
                        logging.info(
                            "Using session string for direct worker connection"
                        )

                        # Prompt for session string if not provided
                        if not session_string:
                            session_str_json = None
                            while True:
                                session_string = input(
                                    "Please enter RTC session string (or type 'exit' to quit): "
                                )
                                if session_string.lower() == "exit":
                                    print("Exiting client.")
                                    return
                                try:
                                    session_str_json = self.parse_session_string(
                                        session_string
                                    )
                                    break  # Exit loop if parsing succeeds
                                except ValueError as e:
                                    print(f"Error: {e}")
                                    print("Please try again or type 'exit' to quit.")
                        else:
                            session_str_json = self.parse_session_string(session_string)

                        # Extract worker credentials from session string.
                        worker_room_id = session_str_json.get("room_id")
                        worker_token = session_str_json.get("token")
                        worker_peer_id = session_str_json.get("peer_id")

                        # Register with room
                        await self._register_with_room(
                            room_id=worker_room_id,
                            token=worker_token,
                            jwt_token=self._jwt_token,
                        )

                        # Set target worker from session string
                        self.target_worker = worker_peer_id
                        logging.info(
                            f"Selected worker from session string: {self.target_worker}"
                        )

                    # BRANCH 2: Room-based discovery
                    elif room_id:
                        logging.debug(
                            f"Using room-based worker discovery: room_id={room_id}"
                        )

                        # Store room info for discovery
                        self.current_room_id = room_id

                        # Visual status: Connecting to room
                        if self.verbosity != "quiet":
                            print(f"Connecting to room {room_id}...", flush=True)

                        # Register with room
                        await self._register_with_room(
                            room_id=room_id,
                            token=token,
                            jwt_token=self._jwt_token,
                        )

                        # Visual status: Discovering workers
                        if self.verbosity != "quiet":
                            print("Discovering workers...", flush=True)

                        # Discover workers in room
                        workers = await self._discover_workers_in_room(
                            room_id=room_id, min_gpu_memory=min_gpu_memory
                        )

                        if not workers:
                            logging.error("No available workers found in room")
                            return

                        # Select worker based on mode
                        if worker_id:
                            # Direct worker selection
                            self.target_worker = worker_id
                            logging.info(f"Using specified worker: {worker_id}")
                        elif auto_select:
                            # Auto-select best worker
                            self.target_worker = self._auto_select_worker(workers)
                            logging.info(f"Auto-selected worker: {self.target_worker}")
                        else:
                            # Interactive worker selection
                            self.target_worker = await self._prompt_worker_selection(
                                workers
                            )
                            logging.info(f"User selected worker: {self.target_worker}")

                    else:
                        logging.error(
                            "No connection method provided (session_string or room_id required)"
                        )
                        return

                    if not self.target_worker:
                        logging.info("No target worker given. Cannot connect.")
                        return

                    # Visual status: Connecting to worker
                    if self.verbosity != "quiet":
                        # Try to get worker name from discovery results if available
                        worker_name = self.target_worker
                        print(f"Connecting to worker {worker_name}...", flush=True)

                    # Create RTCPeerConnection with ICE servers (received during registration)
                    self.pc = self._create_peer_connection()

                    # Initialize data channel
                    channel = self.pc.createDataChannel("my-data-channel")
                    self.data_channel = channel
                    logging.info(
                        "channel(%s) %s" % (channel.label, "created by local party.")
                    )

                    # Register event handlers for the data channel
                    channel.on("open", self.on_channel_open)
                    channel.on("message", self.on_message)

                    # Create and send SDP offer to worker peer.
                    await self.pc.setLocalDescription(await self.pc.createOffer())
                    await websocket.send(
                        json.dumps(
                            {
                                "type": self.pc.localDescription.type,  # type: 'offer'
                                "sender": self.peer_id,  # should be own peer_id (Zoom username)
                                "target": self.target_worker,  # should match Worker's peer_id (Zoom username)
                                "sdp": self.pc.localDescription.sdp,
                            }
                        )
                    )
                    logging.info("Offer sent to worker")

                    # Wait for connection_task to complete (handles all incoming messages)
                    await connection_task

                except Exception as e:
                    logging.error(f"Error during client connection: {e}")
                    connection_task.cancel()
                    raise

            # Send POST req to server to delete this User, Worker, and associated Room.
            self.request_peer_room_deletion(self.peer_id)

            # # Exit.
            # await self.pc.close()
            # await websocket.close()
        except Exception as e:
            logging.error(f"Error in run_client: {e}")
        finally:
            await self.clean_exit()
