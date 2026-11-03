"""Browse client for filesystem viewing via WebRTC.

This module provides the browse command functionality, connecting to a Worker
via WebRTC and serving a local web UI for filesystem browsing.
"""

import asyncio
import logging
import json

import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription

from sleap_connect.config import get_config
from sleap_connect.client.fs_viewer_server import FSViewerServer
from sleap_connect.auth.psk import compute_hmac
from sleap_connect.auth.secret_resolver import resolve_secret
from sleap_connect.protocol import (
    MSG_AUTH_CHALLENGE,
    MSG_AUTH_RESPONSE,
    MSG_AUTH_SUCCESS,
    MSG_AUTH_FAILURE,
)

# Setup logging
logging.basicConfig(level=logging.INFO)


class BrowseClient:
    """Client for browsing Worker filesystem.

    This client:
    - Connects to a Worker via WebRTC
    - Starts a local HTTP/WebSocket server for the browser UI
    - Relays filesystem messages between browser and Worker
    """

    def __init__(
        self,
        room_id: str,
        token: str,
        port: int = 8765,
        room_secret: str = None,
    ):
        """Initialize the browse client.

        Args:
            room_id: Room ID to connect to.
            token: Room token for authentication.
            port: Local port for the file browser server.
            room_secret: Optional room secret for PSK authentication (CLI override).
        """
        self.room_id = room_id
        self.token = token
        self.port = port

        # Load config
        config = get_config()
        self.dns = config.signaling_websocket

        # Connection state
        self.websocket = None
        self.pc = None
        self.data_channel = None
        self.peer_id = None

        # Viewer server
        self.viewer_server = None

        # Shutdown flag
        self.shutting_down = False

        # PSK authentication state
        self._room_secret = resolve_secret(room_id, cli_secret=room_secret)
        self._authenticated = False
        self._auth_event = asyncio.Event()
        self._auth_failed_reason: str = None

        if self._room_secret:
            logging.info("Room secret loaded for PSK authentication")
        else:
            logging.debug(f"No room secret configured for room {room_id}")

    async def run(self, open_browser: bool = True):
        """Run the browse client.

        Args:
            open_browser: Whether to auto-open the browser.
        """
        try:
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
            logging.info(f"Using JWT authentication as: {self.peer_id}")

            # Connect to signaling server
            async with websockets.connect(self.dns) as websocket:
                self.websocket = websocket

                # Register with room
                await self._register_with_room(jwt_token=jwt_token)

                # Discover workers
                workers = await self._discover_workers()

                if not workers:
                    logging.error("No workers found in room")
                    print(
                        "\nNo workers found in the room. Make sure a Worker is running."
                    )
                    return

                # Interactive worker selection
                from sleap_connect.client.file_selector import WorkerSelector

                selector = WorkerSelector(workers)
                selected_worker = await selector.run()

                if selected_worker is None or selector.cancelled:
                    print("\nWorker selection cancelled.")
                    return

                worker_id = selected_worker["peer_id"]
                logging.info(f"Connecting to worker: {worker_id}")

                # Establish WebRTC connection
                await self._connect_to_worker(worker_id)

                # Wait for data channel to open
                await self._wait_for_channel()

                # Wait for PSK authentication
                # Always wait briefly - worker may require auth even if client has no secret
                logging.info("Waiting for PSK authentication...")
                auth_success = await self._wait_for_auth()
                if not auth_success:
                    logging.error(
                        f"PSK authentication failed: {self._auth_failed_reason}"
                    )
                    print(f"\nAuthentication failed: {self._auth_failed_reason}")
                    if not self._room_secret:
                        print(
                            "The worker requires a room secret. Configure it with --room-secret flag."
                        )
                    else:
                        print("Make sure the room secret matches the worker's secret.")
                    return

                # Create and start viewer server
                self.viewer_server = FSViewerServer(
                    send_to_worker=self._send_to_worker,
                )

                url = await self.viewer_server.start(
                    port=self.port,
                    open_browser=open_browser,
                )

                print(f"\nFilesystem browser started at: {url}")
                print("Press Ctrl+C to stop.\n")

                # Run until shutdown
                await self._run_message_loop()

        except Exception as e:
            logging.error(f"Browse client error: {e}")
            raise
        finally:
            await self._cleanup()

    async def _register_with_room(self, jwt_token: str):
        """Register with the signaling server room.

        Args:
            jwt_token: JWT token from GitHub OAuth.
        """
        import platform
        import os

        register_data = {
            "type": "register",
            "peer_id": self.peer_id,
            "room_id": self.room_id,
            "token": self.token,
            "role": "client",
            "jwt": jwt_token,
            "metadata": {
                "tags": ["sleap-connect", "browse-client"],
                "properties": {
                    "platform": platform.system(),
                    "user_id": os.environ.get("USER", "unknown"),
                },
            },
        }

        register_msg = json.dumps(register_data)

        await self.websocket.send(register_msg)

        # Wait for registration confirmation
        response = await self.websocket.recv()
        data = json.loads(response)

        if data.get("type") == "registered_auth":
            logging.info("Registered with room")
            # Store ICE servers if provided
            self.ice_servers = data.get("ice_servers", [])
        else:
            raise RuntimeError(f"Registration failed: {data}")

    async def _discover_workers(self) -> list:
        """Discover available workers in the room."""
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

        # Wait for discovery response
        try:
            response = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
            data = json.loads(response)

            if data.get("type") == "peer_list":
                workers = data.get("peers", [])
                logging.info(f"Discovered {len(workers)} workers")
                return workers
            else:
                logging.warning(f"Unexpected discovery response: {data}")
                return []
        except asyncio.TimeoutError:
            logging.error("Worker discovery timed out")
            return []

    async def _connect_to_worker(self, worker_id: str):
        """Establish WebRTC connection to worker."""
        # Create peer connection
        self.pc = RTCPeerConnection()

        # Create data channel
        self.data_channel = self.pc.createDataChannel("browse")

        @self.data_channel.on("open")
        def on_open():
            logging.info("Data channel opened")

        @self.data_channel.on("message")
        async def on_message(message):
            await self._handle_worker_message(message)

        # Create and send offer
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        offer_msg = json.dumps(
            {
                "type": self.pc.localDescription.type,  # "offer"
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
                # Handle ICE candidate
                candidate = data.get("candidate")
                if candidate:
                    await self.pc.addIceCandidate(candidate)

            else:
                logging.debug(f"Ignoring message type: {msg_type}")

    async def _wait_for_channel(self):
        """Wait for data channel to be open."""
        while self.data_channel.readyState != "open":
            await asyncio.sleep(0.1)
        logging.info("Data channel ready")

    def _send_to_worker(self, message: str):
        """Send message to Worker via data channel."""
        if not self.data_channel or self.data_channel.readyState != "open":
            logging.warning("Data channel not open, cannot send message")
            return

        self.data_channel.send(message)

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

    async def _handle_worker_message(self, message):
        """Handle message from Worker."""
        if isinstance(message, str):
            # PSK authentication messages
            if message.startswith(MSG_AUTH_CHALLENGE):
                self._handle_auth_challenge(message)
                return
            elif message.startswith(MSG_AUTH_SUCCESS):
                self._handle_auth_success()
                return
            elif message.startswith(MSG_AUTH_FAILURE):
                self._handle_auth_failure(message)
                return

            # Forward FS_* messages to viewer server
            if message.startswith("FS_"):
                if self.viewer_server:
                    await self.viewer_server.handle_worker_response(message)
            else:
                logging.debug(f"Ignoring non-FS message: {message[:50]}")

    async def _run_message_loop(self):
        """Run the main message loop until shutdown."""
        try:
            while not self.shutting_down:
                # Check WebSocket for signaling messages
                try:
                    message = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=1.0,
                    )
                    # Handle signaling messages if needed
                    data = json.loads(message)
                    logging.debug(f"Signaling message: {data.get('action')}")

                except asyncio.TimeoutError:
                    pass

                # Check if connection is still alive
                if self.pc and self.pc.connectionState in ("failed", "closed"):
                    logging.warning("WebRTC connection lost")
                    if self.viewer_server:
                        await self.viewer_server.notify_worker_disconnected()
                    break

        except websockets.exceptions.ConnectionClosed:
            logging.warning("Signaling connection closed")
            if self.viewer_server:
                await self.viewer_server.notify_worker_disconnected()

    async def _cleanup(self):
        """Clean up resources."""
        self.shutting_down = True

        # Stop viewer server
        if self.viewer_server:
            await self.viewer_server.stop()

        # Close data channel
        if self.data_channel:
            self.data_channel.close()

        # Close peer connection
        if self.pc:
            await self.pc.close()

        logging.info("Browse client cleaned up")


async def run_browse_client(
    room_id: str,
    token: str,
    port: int = 8765,
    open_browser: bool = True,
    room_secret: str = None,
):
    """Run the browse client.

    Args:
        room_id: Room ID to connect to.
        token: Room token for authentication.
        port: Local port for the file browser server.
        open_browser: Whether to auto-open the browser.
        room_secret: Optional room secret for PSK authentication (CLI override).
    """
    client = BrowseClient(
        room_id=room_id,
        token=token,
        port=port,
        room_secret=room_secret,
    )
    await client.run(open_browser=open_browser)
