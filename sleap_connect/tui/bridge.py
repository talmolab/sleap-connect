"""WebRTC bridge for TUI communication with Workers.

This module provides the WebRTCBridge class that handles WebRTC connections
and message passing between the TUI and Workers, replacing the FSViewerServer
HTTP/WebSocket bridge with direct async communication.
"""

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import Optional, Callable, Any

import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription

from sleap_connect.config import get_config
from sleap_connect.protocol import (
    MSG_FS_GET_MOUNTS,
    MSG_FS_LIST_DIR,
    MSG_USE_WORKER_PATH,
    MSG_WORKER_PATH_OK,
    MSG_WORKER_PATH_ERROR,
    MSG_FS_CHECK_VIDEOS_RESPONSE,
    MSG_FS_RESOLVE_WITH_PREFIX,
    MSG_FS_PREFIX_PROPOSAL,
    MSG_FS_APPLY_PREFIX,
    MSG_FS_PREFIX_APPLIED,
    MSG_FS_WRITE_SLP,
    MSG_FS_WRITE_SLP_OK,
    MSG_FS_WRITE_SLP_ERROR,
    MSG_AUTH_CHALLENGE,
    MSG_AUTH_RESPONSE,
    MSG_AUTH_SUCCESS,
    MSG_AUTH_FAILURE,
    format_message,
    parse_message,
)
from sleap_connect.auth.psk import compute_hmac
from sleap_connect.auth.secret_resolver import resolve_secret


class WebRTCBridge:
    """Bridge for WebRTC communication between TUI and Worker.

    This class manages:
    - WebSocket connection to signaling server
    - WebRTC peer connection and data channel
    - Async message sending and receiving

    The TUI uses this bridge to send filesystem commands and receive responses
    without needing a browser or HTTP server.
    """

    def __init__(
        self,
        room_id: str,
        token: str,
        on_message: Optional[Callable[[str], Any]] = None,
        on_connected: Optional[Callable[[], Any]] = None,
        on_disconnected: Optional[Callable[[], Any]] = None,
        on_auth_status: Optional[Callable[[str], Any]] = None,
        room_secret: Optional[str] = None,
    ):
        """Initialize the WebRTC bridge.

        Args:
            room_id: Room ID to connect to.
            token: Room token for authentication.
            on_message: Callback for incoming messages from worker.
            on_connected: Callback when data channel opens.
            on_disconnected: Callback when connection is lost.
            on_auth_status: Callback for authentication status updates.
            room_secret: Optional room secret for PSK authentication (CLI override).
        """
        self.room_id = room_id
        self.token = token

        # Callbacks
        self._on_message = on_message
        self._on_connected = on_connected
        self._on_disconnected = on_disconnected
        self._on_auth_status = on_auth_status

        # Load config
        config = get_config()
        self.dns = config.signaling_websocket

        # Connection state
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.pc: Optional[RTCPeerConnection] = None
        self.data_channel = None
        self.peer_id: Optional[str] = None
        self.worker_id: Optional[str] = None
        self.ice_servers: list = []

        # Message queue for request/response pattern
        self._response_queue: asyncio.Queue = asyncio.Queue()
        self._pending_requests: dict[str, asyncio.Future] = {}

        # Shutdown flag
        self._running = False

        # PSK authentication state
        self._room_secret = resolve_secret(room_id, cli_secret=room_secret)
        self._authenticated = False
        self._auth_event = asyncio.Event()
        self._auth_failed_reason: Optional[str] = None

        if self._room_secret:
            logging.info("Room secret loaded for PSK authentication")
        else:
            logging.debug(f"No room secret configured for room {room_id}")

    @property
    def is_connected(self) -> bool:
        """Check if data channel is open and ready."""
        return self.data_channel is not None and self.data_channel.readyState == "open"

    @property
    def is_authenticated(self) -> bool:
        """Check if PSK authentication is complete."""
        return self._authenticated

    @property
    def auth_failed_reason(self) -> Optional[str]:
        """Get the reason for authentication failure, if any."""
        return self._auth_failed_reason

    @property
    def requires_auth(self) -> bool:
        """Check if this connection requires PSK authentication."""
        return self._room_secret is not None

    async def connect(self, worker_id: str) -> bool:
        """Connect to a specific worker.

        Args:
            worker_id: The peer ID of the worker to connect to.

        Returns:
            True if connection successful, False otherwise.
        """
        if not self.websocket:
            logging.error("Must call connect_signaling() first")
            return False

        self.worker_id = worker_id
        self._running = True

        # Reset auth state for new connection
        self._authenticated = False
        self._auth_event.clear()
        self._auth_failed_reason = None

        try:
            await self._connect_to_worker(worker_id)
            await self._wait_for_channel()

            # Always wait for PSK authentication - even without a secret configured,
            # the worker may require it and we need to detect that case
            if self._on_auth_status:
                status_msg = (
                    "Authenticating..."
                    if self._room_secret
                    else "Checking authentication..."
                )
                await self._call_async(self._on_auth_status, status_msg)

            auth_success = await self._wait_for_auth()
            if not auth_success:
                if self._on_auth_status:
                    await self._call_async(
                        self._on_auth_status, f"Auth failed: {self._auth_failed_reason}"
                    )
                return False

            if self._on_auth_status:
                await self._call_async(self._on_auth_status, "Authenticated")

            return True
        except Exception as e:
            logging.error(f"Failed to connect to worker: {e}")
            return False

    async def connect_signaling(self) -> bool:
        """Connect to signaling server and register with room.

        Returns:
            True if registration successful, False otherwise.
        """
        try:
            # Get JWT for authentication
            from sleap_connect.auth.credentials import get_valid_jwt, get_user

            jwt_token = get_valid_jwt()
            if jwt_token:
                user = get_user()
                self.peer_id = user.get("username", "unknown") if user else "tui-client"
            else:
                # Fall back to anonymous (will be removed eventually)
                self.peer_id = "tui-client"
                logging.warning("No JWT found, using anonymous connection")

            # Connect to signaling server
            self.websocket = await websockets.connect(self.dns)

            # Register with room
            await self._register_with_room(jwt_token)
            return True

        except Exception as e:
            logging.error(f"Failed to connect to signaling server: {e}")
            return False

    async def discover_workers(self) -> list[dict]:
        """Discover available workers in the room.

        Returns:
            List of worker info dicts with peer_id, metadata, etc.
        """
        if not self.websocket:
            return []

        filters = {
            "role": "worker",
            "room_id": self.room_id,
            "tags": ["sleap-connect"],
            "properties": {"status": "available"},
        }

        discover_msg = json.dumps(
            {
                "type": "discover_peers",
                "from_peer_id": self.peer_id,
                "filters": filters,
            }
        )

        await self.websocket.send(discover_msg)

        try:
            response = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
            data = json.loads(response)

            if data.get("type") == "peer_list":
                workers = data.get("peers", [])
                logging.info(f"Discovered {len(workers)} workers")
                # Debug: dump discovery response to file for inspection
                import json as _json

                _debug_path = Path(tempfile.gettempdir()) / "tui-discovery-debug.json"
                with open(_debug_path, "w") as _f:
                    _json.dump(workers, _f, indent=2)
                    logging.info(f"Wrote discovery response to {_debug_path}")
                return workers
            else:
                logging.warning(f"Unexpected discovery response: {data}")
                return []
        except asyncio.TimeoutError:
            logging.error("Worker discovery timed out")
            return []

    def send(self, message: str) -> bool:
        """Send a message to the connected worker.

        Args:
            message: The message string to send.

        Returns:
            True if sent successfully, False otherwise.
        """
        if not self.is_connected:
            logging.warning("Cannot send: data channel not open")
            return False

        self.data_channel.send(message)
        return True

    async def send_and_wait(
        self,
        message: str,
        response_prefix: str | list[str],
        timeout: float = 10.0,
    ) -> Optional[str]:
        """Send a message and wait for a specific response.

        Args:
            message: The message to send.
            response_prefix: The expected response message prefix, or a list of
                             prefixes (any of which will resolve the future).
            timeout: Maximum time to wait for response.

        Returns:
            The response message, or None if timeout/error.
        """
        if not self.send(message):
            return None

        # Normalize to list
        prefixes = (
            [response_prefix] if isinstance(response_prefix, str) else response_prefix
        )

        # Create a future for this request
        future: asyncio.Future = asyncio.Future()
        for prefix in prefixes:
            self._pending_requests[prefix] = future

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            logging.warning(f"Timeout waiting for {prefixes}")
            return None
        finally:
            for prefix in prefixes:
                self._pending_requests.pop(prefix, None)

    async def get_mounts(self) -> Optional[list]:
        """Get mount points from the connected worker.

        Returns:
            List of mount point dicts, or None if error.
        """
        response = await self.send_and_wait(
            MSG_FS_GET_MOUNTS,
            "FS_MOUNTS_RESPONSE",
        )

        if response:
            _, args = parse_message(response)
            if args:
                try:
                    return json.loads(args[0])
                except json.JSONDecodeError:
                    logging.error("Failed to parse mounts response")
        return None

    async def list_dir(self, path: str, offset: int = 0) -> Optional[dict]:
        """List directory contents on the worker.

        Args:
            path: Directory path to list.
            offset: Pagination offset.

        Returns:
            Dict with 'entries', 'total', etc., or None if error.
        """
        message = format_message(MSG_FS_LIST_DIR, path, str(offset))
        # Use longer timeout for network drives with many files
        response = await self.send_and_wait(message, "FS_LIST_RESPONSE", timeout=60.0)

        if response:
            _, args = parse_message(response)
            if args:
                try:
                    return json.loads(args[0])
                except json.JSONDecodeError:
                    logging.error("Failed to parse list response")
        return None

    async def check_slp_videos(self, slp_path: str) -> Optional[dict]:
        """Check video accessibility for an SLP file.

        Sends USE_WORKER_PATH to trigger video check, then waits for
        FS_CHECK_VIDEOS_RESPONSE.

        Args:
            slp_path: Path to the SLP file on the worker.

        Returns:
            Dict with video check results, or None on error.
            {
                "slp_path": "...",
                "total_videos": 5,
                "accessible": 3,
                "missing": [...],
                "embedded": 1,
                "accessible_videos": [...],
            }
        """
        if not self.is_connected:
            return None

        # Send USE_WORKER_PATH to trigger video check
        message = format_message(MSG_USE_WORKER_PATH, slp_path)

        # We need to wait for both WORKER_PATH_OK and FS_CHECK_VIDEOS_RESPONSE
        path_future: asyncio.Future = asyncio.Future()
        video_future: asyncio.Future = asyncio.Future()

        self._pending_requests[MSG_WORKER_PATH_OK] = path_future
        self._pending_requests[MSG_WORKER_PATH_ERROR] = path_future
        self._pending_requests[MSG_FS_CHECK_VIDEOS_RESPONSE] = video_future

        try:
            self.data_channel.send(message)

            # Wait for path confirmation first
            path_response = await asyncio.wait_for(path_future, timeout=10.0)

            if path_response.startswith(MSG_WORKER_PATH_ERROR):
                _, args = parse_message(path_response)
                error = args[0] if args else "Unknown error"
                return {"error": error}

            # Now wait for video check response
            video_response = await asyncio.wait_for(video_future, timeout=30.0)

            _, args = parse_message(video_response)
            if args:
                return json.loads(args[0])

            return None

        except asyncio.TimeoutError:
            logging.warning("Timeout waiting for video check response")
            return None
        finally:
            self._pending_requests.pop(MSG_WORKER_PATH_OK, None)
            self._pending_requests.pop(MSG_WORKER_PATH_ERROR, None)
            self._pending_requests.pop(MSG_FS_CHECK_VIDEOS_RESPONSE, None)

    async def compute_prefix_resolution(
        self,
        original_path: str,
        new_path: str,
        other_missing: list[str],
    ) -> Optional[dict]:
        """Compute prefix-based resolution for missing videos.

        When a user locates one missing video, this computes what prefix
        change would fix it and checks which other videos would resolve.

        Args:
            original_path: Original path of the video the user selected.
            new_path: New path the user browsed to on the worker.
            other_missing: List of other missing video paths.

        Returns:
            Dict with prefix resolution proposal, or None on error.
            {
                "old_prefix": "/Volumes/talmo",
                "new_prefix": "/vast",
                "would_resolve": [{"original": "...", "resolved": "..."}],
                "would_not_resolve": [...]
            }
        """
        if not self.is_connected:
            return None

        payload = json.dumps(
            {
                "original_path": original_path,
                "new_path": new_path,
                "other_missing": other_missing,
            }
        )

        message = f"{MSG_FS_RESOLVE_WITH_PREFIX}::{payload}"
        response = await self.send_and_wait(
            message, MSG_FS_PREFIX_PROPOSAL, timeout=15.0
        )

        if response:
            _, args = parse_message(response)
            if args:
                try:
                    return json.loads(args[0])
                except json.JSONDecodeError:
                    logging.error("Failed to parse prefix proposal")
        return None

    async def apply_prefix_resolution(
        self,
        slp_path: str,
        prefix_old: str,
        prefix_new: str,
    ) -> Optional[dict]:
        """Apply prefix resolution to fix video paths in an SLP file.

        Args:
            slp_path: Path to the SLP file.
            prefix_old: The old prefix to replace.
            prefix_new: The new prefix to use.

        Returns:
            Dict with result, or None on error.
            {"success": True} or {"error": "..."}
        """
        if not self.is_connected:
            return None

        payload = json.dumps(
            {
                "slp_path": slp_path,
                "prefix_old": prefix_old,
                "prefix_new": prefix_new,
            }
        )
        message = f"{MSG_FS_APPLY_PREFIX}::{payload}"
        response = await self.send_and_wait(
            message, MSG_FS_PREFIX_APPLIED, timeout=15.0
        )

        if response:
            _, args = parse_message(response)
            if args:
                try:
                    return json.loads(args[0])
                except json.JSONDecodeError:
                    logging.error("Failed to parse prefix applied response")
        return None

    async def write_slp_with_new_paths(
        self,
        slp_path: str,
        filename_map: dict[str, str],
        output_dir: str = None,
        output_filename: str = "",
    ) -> Optional[dict]:
        """Write a new SLP file with updated video paths.

        Args:
            slp_path: Path to the original SLP file.
            filename_map: Dict mapping original paths to new resolved paths.
            output_dir: Directory to write the new SLP file (defaults to same as original).
            output_filename: Optional custom filename for the output.

        Returns:
            Dict with output_path and videos_updated, or error info.
        """
        if not self.is_connected:
            return None

        # Default output_dir to same directory as the SLP file
        if output_dir is None:
            from pathlib import Path

            output_dir = str(Path(slp_path).parent)

        payload = json.dumps(
            {
                "slp_path": slp_path,
                "output_dir": output_dir,
                "filename_map": filename_map,
                "output_filename": output_filename,
            }
        )
        message = f"{MSG_FS_WRITE_SLP}::{payload}"

        # Wait for either OK or ERROR response
        response = await self.send_and_wait(
            message,
            [MSG_FS_WRITE_SLP_OK, MSG_FS_WRITE_SLP_ERROR],
            timeout=30.0,
        )

        if response:
            _, args = parse_message(response)
            if args:
                try:
                    return json.loads(args[0])
                except json.JSONDecodeError:
                    logging.error("Failed to parse write SLP response")
        return None

    def _handle_auth_challenge(self, message: str) -> None:
        """Handle PSK authentication challenge from worker.

        Computes HMAC response and sends AUTH_RESPONSE.

        Args:
            message: The AUTH_CHALLENGE message containing the nonce.
        """
        # Parse nonce from challenge message
        if "::" in message:
            _, nonce = message.split("::", 1)
        else:
            logging.error("Invalid AUTH_CHALLENGE format")
            self._auth_failed_reason = "Invalid challenge format"
            self._auth_event.set()
            return

        if not self._room_secret:
            logging.error("Received AUTH_CHALLENGE but no room secret configured")
            self._auth_failed_reason = "No room secret configured"
            self._auth_event.set()
            return

        # Compute HMAC response
        response_hmac = compute_hmac(self._room_secret, nonce)

        # Send AUTH_RESPONSE
        if self.data_channel and self.data_channel.readyState == "open":
            self.data_channel.send(f"{MSG_AUTH_RESPONSE}::{response_hmac}")
            logging.debug("Sent AUTH_RESPONSE")

    def _handle_auth_success(self) -> None:
        """Handle successful PSK authentication."""
        logging.info("PSK authentication successful")
        self._authenticated = True
        self._auth_event.set()

    def _handle_auth_failure(self, message: str) -> None:
        """Handle PSK authentication failure.

        Args:
            message: The AUTH_FAILURE message with reason.
        """
        reason = "Unknown"
        if "::" in message:
            _, reason = message.split("::", 1)

        logging.error(f"PSK authentication failed: {reason}")
        self._auth_failed_reason = reason
        self._auth_event.set()

    async def _wait_for_auth(self, timeout: float = 15.0) -> bool:
        """Wait for PSK authentication to complete.

        If client has a secret, waits for full auth handshake.
        If client has no secret, waits briefly for potential AUTH_CHALLENGE
        from worker - if none comes, assumes legacy mode.

        Args:
            timeout: Maximum time to wait for auth in seconds.

        Returns:
            True if authenticated, False if failed or timed out.
        """
        if not self._room_secret:
            # No secret configured - wait briefly for potential challenge from worker
            # If worker sends AUTH_CHALLENGE, we'll fail (handled in _handle_auth_challenge)
            # If no challenge within 3 seconds, assume legacy mode
            try:
                await asyncio.wait_for(self._auth_event.wait(), timeout=3.0)
                # Event was set - check if it was a failure (challenge received but no secret)
                if self._auth_failed_reason:
                    return False
                return self._authenticated
            except asyncio.TimeoutError:
                # No challenge received - worker is in legacy mode too
                logging.info(
                    "No AUTH_CHALLENGE received - legacy mode (no auth required)"
                )
                self._authenticated = True
                return True

        # Client has secret - wait for full auth handshake
        try:
            await asyncio.wait_for(self._auth_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logging.error("PSK authentication timed out")
            self._auth_failed_reason = "Timeout"
            return False

        return self._authenticated

    async def disconnect(self):
        """Disconnect from worker and signaling server."""
        self._running = False

        if self.data_channel:
            self.data_channel.close()
            self.data_channel = None

        if self.pc:
            await self.pc.close()
            self.pc = None

        if self.websocket:
            await self.websocket.close()
            self.websocket = None

        logging.info("WebRTC bridge disconnected")

    # --- Private methods ---

    async def _register_with_room(self, jwt_token: Optional[str]):
        """Register with the signaling server room."""
        import platform
        import os

        register_data = {
            "type": "register",
            "peer_id": self.peer_id,
            "room_id": self.room_id,
            "token": self.token,
            "role": "client",
            "metadata": {
                "tags": ["sleap-connect", "tui-client"],
                "properties": {
                    "platform": platform.system(),
                    "user_id": os.environ.get("USER", "unknown"),
                },
            },
        }

        if jwt_token:
            register_data["jwt"] = jwt_token

        await self.websocket.send(json.dumps(register_data))

        response = await self.websocket.recv()
        data = json.loads(response)

        if data.get("type") == "registered_auth":
            logging.info("Registered with room")
            self.ice_servers = data.get("ice_servers", [])
        else:
            raise RuntimeError(f"Registration failed: {data}")

    async def _connect_to_worker(self, worker_id: str):
        """Establish WebRTC connection to worker."""
        self.pc = RTCPeerConnection()

        # Create data channel
        self.data_channel = self.pc.createDataChannel("browse")

        @self.data_channel.on("open")
        def on_open():
            logging.info("Data channel opened")
            if self._on_connected:
                asyncio.create_task(self._call_async(self._on_connected))

        @self.data_channel.on("message")
        async def on_message(message):
            # aiortc may send bytes or str depending on version/config
            if isinstance(message, bytes):
                message = message.decode("utf-8")
            await self._handle_message(message)

        @self.data_channel.on("close")
        def on_close():
            logging.info("Data channel closed")
            if self._on_disconnected:
                asyncio.create_task(self._call_async(self._on_disconnected))

        # Create and send offer
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        offer_msg = json.dumps(
            {
                "type": self.pc.localDescription.type,
                "sender": self.peer_id,
                "target": worker_id,
                "sdp": self.pc.localDescription.sdp,
            }
        )

        await self.websocket.send(offer_msg)
        logging.info(f"Sent offer to worker {worker_id}")

        # Wait for answer
        while True:
            response = await self.websocket.recv()
            data = json.loads(response)
            msg_type = data.get("type")

            if msg_type == "answer":
                logging.info("Received answer from worker")
                answer = RTCSessionDescription(
                    sdp=data.get("sdp"),
                    type="answer",
                )
                await self.pc.setRemoteDescription(answer)
                break

            elif msg_type == "candidate":
                candidate = data.get("candidate")
                if candidate:
                    await self.pc.addIceCandidate(candidate)

    async def _wait_for_channel(self, timeout: float = 10.0):
        """Wait for data channel to open."""
        start = asyncio.get_event_loop().time()
        while self.data_channel.readyState != "open":
            if asyncio.get_event_loop().time() - start > timeout:
                raise TimeoutError("Data channel open timeout")
            await asyncio.sleep(0.1)
        logging.info("Data channel ready")

    async def _handle_message(self, message: str):
        """Handle incoming message from worker."""
        # PSK authentication messages (handle first, before anything else)
        if message.startswith(MSG_AUTH_CHALLENGE):
            self._handle_auth_challenge(message)
            return
        elif message.startswith(MSG_AUTH_SUCCESS):
            self._handle_auth_success()
            return
        elif message.startswith(MSG_AUTH_FAILURE):
            self._handle_auth_failure(message)
            return

        # Check for pending request responses
        for prefix, future in list(self._pending_requests.items()):
            if message.startswith(prefix) and not future.done():
                future.set_result(message)
                return

        # Forward other messages to callback
        if self._on_message:
            await self._call_async(self._on_message, message)

    async def _call_async(self, callback: Callable, *args):
        """Call a callback, handling both sync and async functions."""
        result = callback(*args)
        if asyncio.iscoroutine(result):
            await result
