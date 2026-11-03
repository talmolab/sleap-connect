"""Mesh coordinator for worker-to-worker peer connections.

This module implements the mesh networking layer for workers in a room,
enabling direct worker-to-worker WebRTC connections for state synchronization
and coordination.

Key responsibilities:
- Admin WebSocket persistence (admin keeps WebSocket open for discovery)
- Non-admin WebSocket closure (after connecting to admin)
- Mesh signaling via RTC data channels (relay SDP/ICE through admin)
- Batched parallel connection establishment (batch_size configurable)
- Admin promotion handling (open WebSocket when becoming admin)

Architecture:
1. New worker joins, connects to signaling server
2. Signaling server identifies admin worker
3. New worker negotiates WebRTC with admin via signaling server
4. New worker closes WebSocket (non-admin)
5. Admin relays connection requests to other workers via RTC data channels
6. Subsequent peer connections happen via mesh signaling (no signaling server)

Admin role:
- Admin keeps WebSocket open for new worker discovery
- Admin relays SDP offers/answers/ICE candidates between workers
- If admin disconnects, re-election happens, new admin opens WebSocket
"""

import asyncio
import logging
import json
import time
from typing import Dict, Optional, Any, List, TYPE_CHECKING
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate

if TYPE_CHECKING:
    from websockets.client import ClientConnection
    from sleap_connect.worker.worker_class import RTCWorkerClient
    from sleap_connect.worker.admin_controller import AdminController

logger = logging.getLogger(__name__)


# Mesh signaling message types
MSG_MESH_PEER_JOINED = "mesh_peer_joined"
MSG_MESH_OFFER = "mesh_offer"
MSG_MESH_ANSWER = "mesh_answer"
MSG_MESH_ICE_CANDIDATE = "mesh_ice_candidate"
MSG_MESH_PEER_LIST = "mesh_peer_list"

# Admin verification message types (for election confirmation)
MSG_ADMIN_VERIFY = "admin_verify"
MSG_ADMIN_VERIFY_ACK = "admin_verify_ack"


class MeshCoordinator:
    """Coordinates worker-to-worker mesh connections.

    The MeshCoordinator handles:
    - Initial connection to admin worker via signaling server
    - Mesh signaling protocol (relay SDP/ICE via existing connections)
    - Batched parallel connection establishment to other workers
    - Admin role awareness and WebSocket persistence
    - Admin promotion (becoming admin opens WebSocket)

    Attributes:
        worker: Reference to RTCWorkerClient instance
        admin_controller: Reference to AdminController instance
        websocket: WebSocket connection to signaling server (admin only)
        batch_size: Max concurrent connections during mesh formation (default 3)
        pending_offers: Track pending connection offers
        pending_ice_candidates: Buffer ICE candidates during negotiation
    """

    def __init__(
        self,
        worker: "RTCWorkerClient",
        admin_controller: "AdminController",
        batch_size: int = 3,
    ):
        """Initialize MeshCoordinator.

        Args:
            worker: RTCWorkerClient instance
            admin_controller: AdminController instance
            batch_size: Max concurrent connections during mesh formation
        """
        self.worker = worker
        self.admin_controller = admin_controller
        self.batch_size = batch_size

        # WebSocket connection (only kept open if admin)
        self.websocket: Optional["ClientConnection"] = None
        self.websocket_dns: Optional[str] = None  # Store DNS for reconnection

        # Connection negotiation state
        self.pending_offers: Dict[str, Dict[str, Any]] = {}  # peer_id -> offer_data
        self.pending_ice_candidates: Dict[str, List[Dict[str, Any]]] = (
            {}
        )  # peer_id -> [candidates]

        # Mesh formation state
        self.mesh_formation_in_progress = False
        self.connected_peers: set = set()  # Track successfully connected peers

        # Admin WebSocket handler task
        self._admin_handler_task: Optional[asyncio.Task] = None

        # Non-admin WebSocket handler task (for post-demotion signaling)
        self._non_admin_handler_task: Optional[asyncio.Task] = None

        # E2E encryption sessions: {session_id: (aes_key, timestamp)}
        self._e2e_sessions: Dict[str, tuple[bytes, float]] = {}
        # Max age for session keys (seconds) before pruning
        self._e2e_session_max_age: float = 86400  # 24 hours

    def _build_registration_msg(self, is_admin: bool) -> dict:
        """Build a registration message with full worker metadata.

        Used for re-registration after promotion/demotion to ensure the
        signaling server has complete metadata (name, mounts, GPU info, etc.).
        """
        import socket

        try:
            import sleap

            sleap_version = sleap.__version__
        except (ImportError, AttributeError):
            sleap_version = "unknown"

        try:
            from sleap_connect.worker.worker_class import _get_sleap_nn_version

            sleap_nn_version = _get_sleap_nn_version()
        except Exception:
            sleap_nn_version = "unknown"

        mounts = []
        if hasattr(self.worker, "file_manager") and self.worker.file_manager:
            mounts = self.worker.file_manager.get_mounts()

        return {
            "type": "register",
            "peer_id": self.worker.peer_id,
            "room_id": self.worker.room_id,
            "token": self.worker.room_token,
            "api_key": self.worker.api_key,
            "role": "worker",
            "is_admin": is_admin,
            "metadata": {
                "tags": [
                    "sleap-connect",
                    "training-worker",
                    "inference-worker",
                ],
                "properties": {
                    "gpu_memory_mb": self.worker.gpu_memory_mb,
                    "gpu_model": self.worker.gpu_model,
                    "sleap_version": sleap_version,
                    "sleap_nn_version": sleap_nn_version,
                    "cuda_version": self.worker.cuda_version,
                    "hostname": socket.gethostname(),
                    "worker_name": self.worker.name,
                    "status": self.worker.status,
                    "max_concurrent_jobs": getattr(
                        self.worker, "max_concurrent_jobs", 1
                    ),
                    "supported_models": getattr(self.worker, "supported_models", []),
                    "supported_job_types": getattr(
                        self.worker, "supported_job_types", []
                    ),
                    "mounts": mounts,
                },
            },
        }

    async def initialize(self, websocket: "ClientConnection", dns: str):
        """Initialize mesh coordinator with initial WebSocket connection.

        Called after worker registers with signaling server. Determines if
        this worker should keep WebSocket open (admin) or close it (non-admin).

        Args:
            websocket: Initial WebSocket connection from registration
            dns: Signaling server DNS for reconnection
        """
        self.websocket = websocket
        self.websocket_dns = dns

        logger.info("MeshCoordinator initialized")

        # Check if we're admin - if so, keep WebSocket open
        if self.admin_controller.is_admin:
            logger.info("This worker is admin - keeping WebSocket open for discovery")
            self._start_admin_websocket_handler()
        else:
            logger.info(
                "This worker is not admin - will close WebSocket after connecting to admin"
            )

    def _start_admin_websocket_handler(self):
        """Start background task to handle WebSocket messages (admin only)."""
        if not self.admin_controller.is_admin:
            logger.warning("Cannot start admin WebSocket handler: not admin")
            return

        # Cancel existing handler if any
        if self._admin_handler_task and not self._admin_handler_task.done():
            self._admin_handler_task.cancel()

        # Start new handler task
        self._admin_handler_task = asyncio.create_task(
            self._admin_websocket_handler_loop()
        )
        logger.info("Admin WebSocket handler started")

    async def _admin_websocket_handler_loop(self):
        """Background task that handles WebSocket messages for admin.

        This runs continuously while this worker is admin, processing:
        - mesh_offer: New workers connecting via signaling server
        - ice_candidate: ICE candidates for mesh connections
        - offer: Client connection offers
        - admin_conflict: Another worker claiming admin
        - registered_auth: Registration confirmation
        """
        logger.info("Starting admin WebSocket handler loop")

        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")

                    logger.debug(f"Admin received message: {msg_type}")

                    if msg_type == "registered_auth":
                        # Registration confirmed after admin promotion
                        peer_id = data.get("peer_id")
                        room_id = data.get("room_id")
                        logger.info(
                            f"Admin re-registration confirmed: {peer_id} in room {room_id}"
                        )

                    elif msg_type == "mesh_offer":
                        # New worker connecting via signaling server
                        logger.info(
                            f"Admin received mesh_offer from {data.get('from_peer_id')}"
                        )
                        await self.handle_signaling_offer(data)

                    elif msg_type == "candidate":
                        # Trickle ICE candidate from a browser client (dashboard).
                        # The browser sends candidate.toJSON():
                        #   {"candidate": "candidate:xxx 1 udp ... typ host", "sdpMid": "0", ...}
                        # aiortc's addIceCandidate expects an RTCIceCandidate object,
                        # so we parse the candidate string via aioice and convert.
                        candidate_data = data.get("candidate")
                        if candidate_data and self.worker.pc:
                            cand_str = candidate_data.get("candidate", "")
                            if not cand_str or cand_str == "":
                                # End-of-candidates signal
                                logger.info("[ICE] Received end-of-candidates signal")
                                await self.worker.pc.addIceCandidate(None)
                            else:
                                try:
                                    # Strip leading "candidate:" prefix if present
                                    sdp_str = cand_str
                                    if sdp_str.startswith("candidate:"):
                                        sdp_str = sdp_str[len("candidate:") :]

                                    from aioice import Candidate as AioCandidate
                                    from aiortc.rtcicetransport import (
                                        candidate_from_aioice,
                                    )

                                    aio_cand = AioCandidate.from_sdp(sdp_str)
                                    rtc_cand = candidate_from_aioice(aio_cand)
                                    rtc_cand.sdpMid = candidate_data.get("sdpMid")
                                    rtc_cand.sdpMLineIndex = candidate_data.get(
                                        "sdpMLineIndex"
                                    )

                                    await self.worker.pc.addIceCandidate(rtc_cand)
                                    cand_type = (
                                        "relay"
                                        if "typ relay" in cand_str
                                        else (
                                            "srflx"
                                            if "typ srflx" in cand_str
                                            else "host"
                                        )
                                    )
                                    logger.info(
                                        f"[ICE] Added client {cand_type} candidate"
                                    )
                                except Exception as e:
                                    logger.warning(
                                        f"[ICE] Failed to add candidate: {e}"
                                    )
                        else:
                            logger.debug(
                                "[ICE] Ignoring candidate: no PC or empty candidate"
                            )

                    elif msg_type == "ice_candidate":
                        # ICE candidate for mesh connection
                        await self.handle_signaling_ice_candidate(data)

                    elif msg_type == "offer":
                        # Client connection offer - delegate to worker
                        logger.info(
                            f"Admin received client offer from {data.get('sender')}"
                        )
                        await self._handle_client_offer(data)

                    elif msg_type == "admin_conflict":
                        # Another worker is already admin
                        current_admin = data.get("current_admin")
                        logger.warning(
                            f"Admin conflict: {current_admin} is already admin, "
                            f"demoting ourselves"
                        )
                        self.admin_controller.is_admin = False
                        self.admin_controller.admin_peer_id = current_admin
                        self.worker.admin_peer_id = current_admin
                        # Close WebSocket since we're not admin
                        await self.on_admin_demotion()
                        return  # Exit handler loop

                    elif msg_type == "error":
                        logger.error(f"Admin received error: {data.get('reason')}")

                    # ── E2E encryption ─────────────────────────────────
                    elif msg_type == "key_exchange":
                        await self._handle_key_exchange(data)

                    elif msg_type == "encrypted_relay":
                        decrypted, session_id = self._decrypt_if_encrypted(data)
                        if decrypted is not None:
                            inner_type = decrypted.get("type")
                            if inner_type == "fs_list_req":
                                await self._handle_fs_list_req(
                                    decrypted, session_id=session_id
                                )
                            elif inner_type == "use_worker_path":
                                await self._handle_use_worker_path(
                                    decrypted, session_id=session_id
                                )
                            elif inner_type == "fs_check_videos":
                                await self._handle_fs_check_videos(
                                    decrypted, session_id=session_id
                                )
                            elif inner_type == "job_assigned":
                                await self._handle_job_assigned(
                                    decrypted, session_id=session_id
                                )
                            elif inner_type == "job_cancel":
                                self._handle_job_cancel(decrypted)
                            else:
                                logger.debug(
                                    f"[E2E] Unknown decrypted type: {inner_type}"
                                )

                    # ── Relay-forwarded requests from dashboard ──────────
                    elif msg_type == "fs_list_req":
                        await self._handle_fs_list_req(data)

                    elif msg_type == "use_worker_path":
                        await self._handle_use_worker_path(data)

                    elif msg_type == "fs_check_videos":
                        await self._handle_fs_check_videos(data)

                    elif msg_type == "job_assigned":
                        await self._handle_job_assigned(data)

                    elif msg_type == "job_cancel":
                        self._handle_job_cancel(data)

                    elif msg_type == "ping":
                        self.worker._last_signaling_ping = time.monotonic()
                        logging.warning(
                            "\033[33m[HEARTBEAT] Received signaling server ping\033[0m"
                        )

                    else:
                        logger.debug(f"Admin ignoring message type: {msg_type}")

                except json.JSONDecodeError:
                    logger.error("Invalid JSON received in admin handler")
                except Exception as e:
                    logger.error(f"Error processing admin message: {e}")

        except asyncio.CancelledError:
            logger.info("Admin WebSocket handler cancelled")
        except KeyboardInterrupt:
            raise  # Propagate to reconnection loop for clean exit
        except Exception as e:
            logger.error(f"Admin WebSocket handler error: {e}")

    # ── Relay-forwarded request handlers ────────────────────────────────────

    async def _handle_fs_list_req(
        self, data: Dict[str, Any], session_id: str | None = None
    ):
        """Handle fs_list_req forwarded from dashboard via signaling server.

        Translates to the worker's FS_LIST_DIR protocol, sends back fs_list_res
        which the signaling server forwards to the relay SSE channel.
        """
        from sleap_connect.protocol import (
            MSG_FS_LIST_DIR,
            MSG_FS_LIST_RESPONSE,
            MSG_SEPARATOR,
        )

        path = data.get("path", "/")
        req_id = data.get("req_id")
        offset = data.get("offset", 0)

        # Build the protocol message: FS_LIST_DIR::path::offset
        proto_msg = f"{MSG_FS_LIST_DIR}{MSG_SEPARATOR}{path}{MSG_SEPARATOR}{offset}"
        response = self.worker.handle_fs_message(proto_msg)

        # Parse the protocol response into JSON for relay
        if response.startswith(f"{MSG_FS_LIST_RESPONSE}{MSG_SEPARATOR}"):
            json_str = response.split(MSG_SEPARATOR, 1)[1]
            result = json.loads(json_str)
            await self._send_relay_response(
                {"type": "fs_list_res", "req_id": req_id, **result},
                session_id=session_id,
            )
            logger.info(f"[RELAY] Sent fs_list_res for path={path}")
        else:
            # Error response
            await self._send_relay_response(
                {
                    "type": "fs_list_res",
                    "req_id": req_id,
                    "error": response,
                    "entries": [],
                },
                session_id=session_id,
            )
            logger.warning(f"[RELAY] FS list error for path={path}: {response}")

    async def _handle_use_worker_path(
        self, data: Dict[str, Any], session_id: str | None = None
    ):
        """Handle use_worker_path forwarded from dashboard via signaling server.

        Validates the path, then checks video accessibility if it's an SLP file.
        Sends worker_path_ok/worker_path_error and optionally fs_check_videos_response.
        """
        from sleap_connect.protocol import (
            MSG_USE_WORKER_PATH,
            MSG_WORKER_PATH_OK,
            MSG_WORKER_PATH_ERROR,
            MSG_FS_CHECK_VIDEOS_RESPONSE,
            MSG_SEPARATOR,
        )

        path = data.get("path", "")
        proto_msg = f"{MSG_USE_WORKER_PATH}{MSG_SEPARATOR}{path}"
        response = self.worker.handle_worker_path_message(proto_msg)

        if response.startswith(f"{MSG_WORKER_PATH_OK}{MSG_SEPARATOR}"):
            validated_path = response.split(MSG_SEPARATOR, 1)[1]
            await self._send_relay_response(
                {"type": "worker_path_ok", "path": validated_path},
                session_id=session_id,
            )
            logger.info(f"[RELAY] Sent worker_path_ok for {validated_path}")

            # Check videos if SLP
            video_response = self.worker._check_slp_videos_if_needed()
            if video_response:
                json_str = video_response.split(MSG_SEPARATOR, 1)[1]
                result = json.loads(json_str)
                await self._send_relay_response(
                    {"type": "fs_check_videos_response", **result},
                    session_id=session_id,
                )
                logger.info(f"[RELAY] Sent fs_check_videos_response")
        else:
            error_msg = (
                response.split(MSG_SEPARATOR, 1)[1]
                if MSG_SEPARATOR in response
                else response
            )
            await self._send_relay_response(
                {"type": "worker_path_error", "error": error_msg},
                session_id=session_id,
            )
            logger.warning(f"[RELAY] Sent worker_path_error: {error_msg}")

    async def _handle_fs_check_videos(
        self, data: Dict[str, Any], session_id: str | None = None
    ):
        """Handle explicit fs_check_videos request from dashboard."""
        from sleap_connect.protocol import MSG_FS_CHECK_VIDEOS_RESPONSE, MSG_SEPARATOR

        response = self.worker._check_slp_videos_if_needed()
        if response:
            json_str = response.split(MSG_SEPARATOR, 1)[1]
            result = json.loads(json_str)
            await self._send_relay_response(
                {"type": "fs_check_videos_response", **result},
                session_id=session_id,
            )
            logger.info(f"[RELAY] Sent fs_check_videos_response")
        else:
            await self._send_relay_response(
                {
                    "type": "fs_check_videos_response",
                    "missing": [],
                    "accessible": 0,
                    "embedded": 0,
                },
                session_id=session_id,
            )

    async def _handle_job_assigned(
        self, data: Dict[str, Any], session_id: str | None = None
    ):
        """Handle job_assigned forwarded from dashboard via signaling server.

        Translates to the worker's existing handle_job_submit flow, using a
        WebSocket-backed shim channel so responses go through the relay.
        """
        from sleap_connect.protocol import MSG_SEPARATOR

        job_id = data.get("job_id", "unknown")
        config = data.get("config", {})
        submitted_by = data.get("submitted_by", "unknown")

        logger.info(f"[RELAY] Job assigned: {job_id} by {submitted_by}")

        # Create a shim channel that sends responses via WebSocket→relay.
        # channel.send() is synchronous in the RTC API, so we schedule async
        # WebSocket sends on the event loop.
        coordinator = self  # Reference to mesh coordinator for dynamic websocket access
        loop = asyncio.get_event_loop()
        e2e_session_id = session_id  # Capture for RelayChannel closure

        job_type = config.get("type", "train")  # "train" or "track"

        class RelayChannel:
            """Shim that mimics RTCDataChannel.send() but routes through WebSocket."""

            def __init__(self, coordinator_ref, job_id_ref):
                self._coordinator = coordinator_ref
                self._job_id = job_id_ref
                self.readyState = "open"

            def send(self, message: str):
                """Forward protocol messages to relay via WebSocket."""
                from sleap_connect.protocol import (
                    MSG_JOB_ACCEPTED,
                    MSG_JOB_REJECTED,
                    MSG_JOB_PROGRESS,
                    MSG_JOB_COMPLETE,
                    MSG_JOB_FAILED,
                    MSG_SEPARATOR,
                )
                import json as _j

                relay_msg = None
                # Helper: build relay message, ensuring job_id is always the
                # signaling server's ID (not the worker's internal per-model ID
                # that may appear in spread payloads).
                _base = {"type": "job_status", "job_id": self._job_id}

                if message.startswith(f"{MSG_JOB_ACCEPTED}{MSG_SEPARATOR}"):
                    relay_msg = {**_base, "status": "accepted"}
                elif message.startswith(f"{MSG_JOB_REJECTED}{MSG_SEPARATOR}"):
                    parts = message.split(MSG_SEPARATOR, 2)
                    error_json = parts[2] if len(parts) > 2 else "{}"
                    relay_msg = {
                        **_base,
                        "status": "rejected",
                        "errors": _j.loads(error_json),
                    }
                elif message.startswith(f"{MSG_JOB_PROGRESS}{MSG_SEPARATOR}"):
                    payload = message.split(MSG_SEPARATOR, 1)[1]
                    try:
                        relay_msg = {**_j.loads(payload), **_base, "status": "running"}
                    except Exception:
                        relay_msg = {**_base, "status": "running", "message": payload}
                elif message.startswith(f"{MSG_JOB_COMPLETE}{MSG_SEPARATOR}"):
                    payload = message.split(MSG_SEPARATOR, 1)[1]
                    try:
                        relay_msg = {**_j.loads(payload), **_base, "status": "complete"}
                    except Exception:
                        relay_msg = {**_base, "status": "complete"}
                elif message.startswith(f"{MSG_JOB_FAILED}{MSG_SEPARATOR}"):
                    payload = message.split(MSG_SEPARATOR, 1)[1]
                    try:
                        relay_msg = {**_j.loads(payload), **_base, "status": "failed"}
                    except Exception:
                        relay_msg = {**_base, "status": "failed", "error": payload}
                elif message.startswith("MODEL_TYPE::"):
                    model_type = message.split("::", 1)[1]
                    relay_msg = {
                        **_base,
                        "event": "model_type_switch",
                        "model_type": model_type,
                    }
                    import logging as _logging

                    _logging.info(
                        f"[RELAY] Forwarding MODEL_TYPE switch to: {model_type}"
                    )
                elif message.startswith("TRAIN_JOB_START::"):
                    relay_msg = {**_base, "status": "running", "message": message}
                elif message == "TRAINING_JOBS_DONE":
                    relay_msg = {**_base, "status": "complete"}
                elif message.startswith("INFERENCE_BEGIN::"):
                    relay_msg = {**_base, "status": "running", "stage": "inference"}
                elif message.startswith("INFERENCE_COMPLETE::"):
                    payload = message.split("::", 1)[1] if "::" in message else "{}"
                    try:
                        relay_msg = {
                            **_j.loads(payload),
                            **_base,
                            "status": "complete",
                            "stage": "inference",
                        }
                    except Exception:
                        relay_msg = {
                            **_base,
                            "status": "complete",
                            "stage": "inference",
                        }
                elif message.startswith("INFERENCE_FAILED::"):
                    payload = message.split("::", 1)[1] if "::" in message else "{}"
                    try:
                        relay_msg = {
                            **_j.loads(payload),
                            **_base,
                            "status": "failed",
                            "stage": "inference",
                        }
                    except Exception:
                        relay_msg = {**_base, "status": "failed", "stage": "inference"}
                elif message.startswith("PROGRESS_REPORT::"):
                    # Forward only epoch-level events to avoid flooding relay
                    try:
                        import jsonpickle

                        payload = message.split("::", 1)[1]
                        msg = jsonpickle.decode(payload)
                        event = msg.get("event") if isinstance(msg, dict) else None
                        if event in ("train_begin", "epoch_end", "train_end"):
                            progress_data = {
                                "type": "job_progress",
                                "job_id": self._job_id,
                                "event": event,
                            }
                            if event == "train_begin" and msg.get("wandb_url"):
                                progress_data["wandb_url"] = msg["wandb_url"]
                            elif event == "epoch_end":
                                progress_data["epoch"] = msg.get("epoch")
                                progress_data["logs"] = msg.get("logs", {})
                            if msg.get("what"):
                                progress_data["what"] = msg["what"]
                            relay_msg = progress_data
                    except Exception:
                        pass  # Silently drop malformed progress reports
                elif message.startswith("[stderr] "):
                    # Forward stderr log lines — only for inference/track jobs.
                    # Training uses PROGRESS_REPORT:: for structured updates;
                    # stderr during training is Lightning's tqdm bar which floods.
                    if job_type == "track":
                        line = message[len("[stderr] ") :].rstrip()
                        if line:
                            relay_msg = {
                                **_base,
                                "status": "running",
                                "message": line,
                            }
                elif message.startswith("CR::"):
                    # tqdm/rich progress bar update — only for inference/track.
                    # Training progress bars would flood the relay.
                    if job_type == "track":
                        line = message[4:].strip()
                        if line:
                            relay_msg = {
                                **_base,
                                "status": "running",
                                "message": line,
                            }
                else:
                    # Catch-all for unrecognised text lines. Only forward for
                    # inference/track jobs where stdout is the primary output.
                    # Training jobs use PROGRESS_REPORT:: for epoch-level updates
                    # and would flood the relay with config dumps, model arch, etc.
                    if job_type == "track":
                        line = message.rstrip("\n").strip()
                        if line and not line.startswith("PROGRESS_REPORT::"):
                            relay_msg = {
                                **_base,
                                "status": "running",
                                "message": line,
                            }
                # (end of message routing)

                if relay_msg:
                    # Encrypt if E2E session is active
                    if e2e_session_id is not None:
                        e2e_key = self._coordinator._get_e2e_key_for_session(
                            e2e_session_id
                        )
                        if e2e_key is not None:
                            from sleap_connect.encryption.envelope import wrap as _wrap

                            envelope = _wrap(
                                e2e_key,
                                e2e_session_id,
                                relay_msg,
                                job_id=self._job_id,
                            )
                            asyncio.run_coroutine_threadsafe(
                                self._coordinator.websocket.send(_j.dumps(envelope)),
                                loop,
                            )
                            return
                    asyncio.run_coroutine_threadsafe(
                        self._coordinator.websocket.send(_j.dumps(relay_msg)), loop
                    )

            @property
            def label(self):
                return f"relay-job-{self._job_id}"

        relay_channel = RelayChannel(coordinator, job_id)

        # Build the protocol message: JOB_SUBMIT::{job_id}::{json_spec}
        import json as _json

        spec_json = _json.dumps(config)
        from sleap_connect.protocol import MSG_JOB_SUBMIT

        proto_msg = f"{MSG_JOB_SUBMIT}{MSG_SEPARATOR}{job_id}{MSG_SEPARATOR}{spec_json}"

        # Run the job in the background so the admin handler loop can
        # continue processing messages (including heartbeat pings).
        async def _run_job():
            try:
                await self.worker.handle_job_submit(relay_channel, proto_msg)
            except Exception as e:
                logger.error(f"[RELAY] Job {job_id} failed: {e}")
            finally:
                logger.info(f"[RELAY] Job submit handler complete for {job_id}")

        asyncio.create_task(_run_job())

    def _handle_job_cancel(self, data: Dict[str, Any]):
        """Handle job_cancel forwarded from dashboard via signaling server.

        Supports two modes via the ``mode`` field:
        - ``"cancel"`` (default): Stop training AND skip inference/remaining models.
        - ``"stop"``: Stop current model early but continue to next model and inference.

        Both modes send ZMQ ``{"command": "stop"}`` to sleap-nn's
        TrainingControllerZMQ callback, which sets trainer.should_stop=True and
        coordinates DDP shutdown via reduce_boolean_decision.

        Falls back to cancel_running_job() (SIGTERM→SIGKILL) if no
        ProgressReporter is available (e.g., inference jobs).
        """
        import jsonpickle

        logger.info(f"[RELAY] _handle_job_cancel raw data: {data}")
        job_id = data.get("job_id", "unknown")
        mode = data.get("mode", "cancel")  # "cancel" or "stop"
        logger.info(f"[RELAY] Job {mode} received for {job_id}")

        if not self.worker.job_executor:
            logger.warning(f"[RELAY] No job executor available to {mode} job {job_id}")
            return

        if self.worker.job_executor._progress_reporter is not None:
            # Send ZMQ "stop" to sleap-nn (it only handles "stop")
            stop_msg = jsonpickle.encode({"command": "stop"})
            self.worker.job_executor.send_control_message(stop_msg)

            if mode == "cancel":
                # Cancel: skip remaining models + inference
                self.worker.job_executor._cancel_requested = True
                logger.info(f"[RELAY] ZMQ stop + cancel flag set for job {job_id}")
            else:
                # Stop Early: continue to next model + inference
                logger.info(f"[RELAY] ZMQ stop (early) sent for job {job_id}")
        else:
            # No ZMQ reporter (inference job or reporter not started yet) — signal fallback
            self.worker.job_executor.cancel_running_job()
            logger.info(
                f"[RELAY] SIGTERM cancel sent for job {job_id} (no ZMQ reporter)"
            )

    # ── E2E encryption for relay transport ───────────────────────────────

    async def _handle_key_exchange(self, data: Dict[str, Any]):
        """Handle key_exchange from a dashboard/relay client.

        Generates an ephemeral P-256 keypair, derives the shared AES key via
        ECDH + HKDF, stores it keyed by session_id, and sends the worker's
        public key back via key_exchange_response.
        """
        from sleap_connect.encryption import (
            generate_keypair,
            derive_shared_key,
            public_key_from_b64,
            public_key_to_b64,
        )

        session_id = data.get("session_id")
        peer_pub_b64 = data.get("public_key")

        if not session_id or not peer_pub_b64:
            logger.warning("[E2E] key_exchange missing session_id or public_key")
            return

        try:
            private_key, our_pub_bytes = generate_keypair()
            peer_pub_bytes = public_key_from_b64(peer_pub_b64)
            shared_key = derive_shared_key(private_key, peer_pub_bytes)

            self._prune_e2e_sessions()
            self._e2e_sessions[session_id] = (shared_key, time.time())
            logger.info(
                f"[E2E] Key exchange complete for session {session_id[:8]}... "
                f"({len(self._e2e_sessions)} active sessions)"
            )

            response = {
                "type": "key_exchange_response",
                "session_id": session_id,
                "public_key": public_key_to_b64(our_pub_bytes),
            }
            await self.websocket.send(json.dumps(response))

        except Exception as e:
            logger.error(f"[E2E] Key exchange failed: {e}")

    def _prune_e2e_sessions(self):
        """Remove E2E sessions older than max age."""
        now = time.time()
        expired = [
            sid
            for sid, (_, ts) in self._e2e_sessions.items()
            if now - ts > self._e2e_session_max_age
        ]
        for sid in expired:
            del self._e2e_sessions[sid]
            logger.debug(f"[E2E] Pruned expired session {sid[:8]}...")

    def _decrypt_if_encrypted(
        self, data: Dict[str, Any]
    ) -> tuple[Dict[str, Any], str | None]:
        """If the message is an encrypted relay envelope, decrypt it.

        Returns:
            Tuple of (decrypted_data, session_id). If not encrypted, returns
            (data, None). If decryption fails, returns (None, None).
        """
        from sleap_connect.encryption.envelope import ENCRYPTED_RELAY_TYPE, unwrap

        if data.get("type") != ENCRYPTED_RELAY_TYPE:
            return data, None

        session_id = data.get("session_id")
        key_lookup = {sid: key for sid, (key, _) in self._e2e_sessions.items()}
        decrypted = unwrap(data, key_lookup)

        if decrypted is None:
            logger.warning(
                f"[E2E] Failed to decrypt message for session "
                f"{session_id[:8] if session_id else 'unknown'}..."
            )
            return None, None

        logger.debug(
            f"[E2E] Decrypted {decrypted.get('type')} from session {session_id[:8]}..."
        )
        return decrypted, session_id

    def _get_e2e_key_for_session(self, session_id: str | None) -> bytes | None:
        """Get the AES key for a session, or None if not encrypted."""
        if session_id is None:
            return None
        entry = self._e2e_sessions.get(session_id)
        return entry[0] if entry else None

    async def _send_relay_response(
        self,
        message: dict,
        session_id: str | None = None,
        job_id: str | None = None,
    ):
        """Send a relay response, encrypting if the request was encrypted."""
        key = self._get_e2e_key_for_session(session_id)
        if key is not None:
            from sleap_connect.encryption.envelope import wrap

            envelope = wrap(key, session_id, message, job_id=job_id)
            await self.websocket.send(json.dumps(envelope))
        else:
            await self.websocket.send(json.dumps(message))

    async def _handle_client_offer(self, data: Dict[str, Any]):
        """Handle client connection offer received on admin WebSocket.

        Delegates to the worker's existing client handling logic.

        Args:
            data: Offer data from signaling server
        """
        from aiortc import RTCSessionDescription

        sender = data.get("sender")
        sdp = data.get("sdp")

        logger.info(f"Admin handling client offer from {sender}")

        # Check worker status before accepting
        if self.worker.status in ["busy", "reserved"]:
            logger.warning(
                f"Rejecting client {sender} - worker is {self.worker.status}"
            )
            await self.websocket.send(
                json.dumps(
                    {
                        "type": "error",
                        "target": sender,
                        "reason": "worker_busy",
                        "message": f"Worker is currently {self.worker.status}",
                        "current_status": self.worker.status,
                    }
                )
            )
            return

        # Update status to reserved
        await self.worker.state_manager.update_status("reserved")
        logger.info("Worker status updated to 'reserved'")

        # Store peer role so handle_channel_open can skip PSK challenge for
        # dashboard clients (role: "client"). Without this, the channel open
        # handler issues a PSK challenge the dashboard can't answer.
        self.worker._pending_offer_role = data.get("role")
        logger.info(f"Stored pending offer role: {self.worker._pending_offer_role!r}")

        # Log ICE server config so we can diagnose HPC connectivity issues
        ice_count = len(self.worker.ice_servers) if self.worker.ice_servers else 0
        logger.info(
            f"Creating client PC with {ice_count} ICE server(s) from signaling "
            f"server (factory will add STUN fallback if 0)"
        )

        # Create a fresh RTCPeerConnection with ICE servers so TURN relay is
        # available. self.worker.pc was created before registered_auth delivered
        # ICE server credentials, so it has no STUN/TURN config — reusing it
        # causes ICE to stall at "checking" on firewalled networks (e.g. HPC).
        self.worker.pc = self.worker._create_client_peer_connection()

        # Set remote description and create answer
        await self.worker.pc.setRemoteDescription(
            RTCSessionDescription(sdp=sdp, type="offer")
        )
        await self.worker.pc.setLocalDescription(await self.worker.pc.createAnswer())

        # Log the FINAL answer SDP candidates (after gathering is complete)
        answer_sdp = self.worker.pc.localDescription.sdp
        sdp_lines = answer_sdp.splitlines()
        candidates = [l for l in sdp_lines if l.startswith("a=candidate")]
        logger.info(
            f"[ICE] Answer SDP ready ({len(answer_sdp)} chars) with "
            f"{len(candidates)} candidate(s), sending to {sender}"
        )
        for c in candidates:
            logger.info(f"[ICE]   {c}")
        if not candidates:
            logger.warning(
                "[ICE] Answer SDP has 0 candidates — browser has nothing to connect to"
            )

        # Send answer back to client
        await self.websocket.send(
            json.dumps(
                {
                    "type": self.worker.pc.localDescription.type,
                    "sender": self.worker.peer_id,
                    "target": sender,
                    "sdp": answer_sdp,
                }
            )
        )

        logger.info(f"Admin sent answer to client {sender}")

        # Clear received files for new connection
        self.worker.received_files.clear()

    async def connect_to_admin(self, admin_peer_id: str) -> bool:
        """Connect to admin worker via signaling server.

        This is the initial connection for non-admin workers. Uses the
        WebSocket signaling server to negotiate WebRTC connection with admin.

        Args:
            admin_peer_id: peer_id of the admin worker

        Returns:
            True if connection successful, False otherwise
        """
        if not self.websocket:
            logger.error("Cannot connect to admin: no WebSocket connection")
            return False

        logger.info(f"Connecting to admin worker: {admin_peer_id}")

        try:
            # Create peer connection for admin — use factory to avoid
            # aiortc's default STUN (hangs in internet-less containers).
            pc = self.worker._create_mesh_peer_connection()

            # Set up data channel for mesh signaling
            data_channel = pc.createDataChannel("mesh_signaling")

            # Store data channel IMMEDIATELY - don't wait for on_open
            # The sending code will check readyState before sending
            self.worker.data_channels[admin_peer_id] = data_channel
            logger.info(
                f"Created data channel for admin {admin_peer_id} (state: {data_channel.readyState})"
            )

            @data_channel.on("open")
            def on_open():
                logger.info(f"Mesh signaling channel open with admin: {admin_peer_id}")

            @data_channel.on("close")
            def on_close():
                logger.warning(
                    f"Mesh signaling channel closed with admin: {admin_peer_id}"
                )
                # Remove data channel reference
                if admin_peer_id in self.worker.data_channels:
                    del self.worker.data_channels[admin_peer_id]

            @data_channel.on("error")
            def on_error(error):
                logger.error(f"Data channel error with admin {admin_peer_id}: {error}")

            @data_channel.on("message")
            async def on_message(message):
                await self._handle_mesh_message(message, admin_peer_id)

            # Set ICE connection state handler
            @pc.on("iceconnectionstatechange")
            async def on_ice_state_change():
                await self.worker.on_mesh_iceconnectionstatechange(admin_peer_id, pc)

            # Set ICE candidate handler - send candidates to signaling server
            @pc.on("icecandidate")
            async def on_ice_candidate(event):
                if event.candidate and self.websocket:
                    await self.websocket.send(
                        json.dumps(
                            {
                                "type": "ice_candidate",
                                "from_peer_id": self.worker.peer_id,
                                "target_peer_id": admin_peer_id,
                                "candidate": {
                                    "candidate": event.candidate.candidate,
                                    "sdpMLineIndex": event.candidate.sdpMLineIndex,
                                    "sdpMid": event.candidate.sdpMid,
                                },
                            }
                        )
                    )
                    logger.debug(
                        f"Sent ICE candidate to admin {admin_peer_id} via signaling server"
                    )

            # Create offer
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)

            # Send offer to admin via signaling server
            await self.websocket.send(
                json.dumps(
                    {
                        "type": "mesh_connect",
                        "from_peer_id": self.worker.peer_id,  # Added for clarity
                        "target_peer_id": admin_peer_id,
                        "offer": {
                            "sdp": pc.localDescription.sdp,
                            "type": pc.localDescription.type,
                        },
                    }
                )
            )

            logger.info(f"Sent connection offer to admin: {admin_peer_id}")

            # Store connection in registry
            self.worker.worker_connections[admin_peer_id] = pc
            self.worker.admin_peer_id = admin_peer_id
            self.connected_peers.add(admin_peer_id)

            return True

        except Exception as e:
            logger.error(f"Failed to connect to admin: {e}")
            return False

    async def connect_to_workers_batched(self, worker_peer_ids: List[str]):
        """Connect to multiple workers using batched parallel approach.

        Establishes connections in batches to avoid overwhelming the admin's
        data channel with simultaneous negotiations.

        Args:
            worker_peer_ids: List of worker peer_ids to connect to
        """
        if not worker_peer_ids:
            logger.info("No workers to connect to")
            return

        logger.info(
            f"Connecting to {len(worker_peer_ids)} workers in batches of {self.batch_size}"
        )

        self.mesh_formation_in_progress = True

        # Split into batches
        for i in range(0, len(worker_peer_ids), self.batch_size):
            batch = worker_peer_ids[i : i + self.batch_size]
            logger.info(
                f"Connecting to batch {i // self.batch_size + 1}: {len(batch)} workers"
            )

            # Connect to all workers in batch concurrently
            tasks = [self._connect_to_worker(peer_id) for peer_id in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Log results
            for peer_id, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to connect to {peer_id}: {result}")
                elif result:
                    logger.info(f"Successfully connected to {peer_id}")
                else:
                    logger.warning(f"Connection to {peer_id} failed")

            # Small delay between batches to avoid overwhelming admin
            if i + self.batch_size < len(worker_peer_ids):
                await asyncio.sleep(0.5)

        self.mesh_formation_in_progress = False
        logger.info(
            f"Mesh formation complete: {len(self.connected_peers)} / {len(worker_peer_ids) + 1} workers connected"
        )

    async def _connect_to_worker(self, peer_id: str) -> bool:
        """Connect to a single worker via mesh signaling (through admin).

        Args:
            peer_id: peer_id of the worker to connect to

        Returns:
            True if connection successful, False otherwise
        """
        logger.info(f"Initiating connection to worker: {peer_id}")

        try:
            # Create peer connection — use factory to avoid default STUN hang
            pc = self.worker._create_mesh_peer_connection()

            # Set up data channel
            data_channel = pc.createDataChannel("mesh_signaling")

            # Store data channel IMMEDIATELY - don't wait for on_open
            # The sending code will check readyState before sending
            self.worker.data_channels[peer_id] = data_channel
            logger.info(
                f"Created data channel for worker {peer_id} (state: {data_channel.readyState})"
            )

            @data_channel.on("open")
            def on_open():
                logger.info(f"Mesh signaling channel open with worker: {peer_id}")

            @data_channel.on("close")
            def on_close():
                logger.warning(f"Mesh signaling channel closed with worker: {peer_id}")
                # Remove data channel reference
                if peer_id in self.worker.data_channels:
                    del self.worker.data_channels[peer_id]

            @data_channel.on("error")
            def on_error(error):
                logger.error(f"Data channel error with worker {peer_id}: {error}")

            @data_channel.on("message")
            async def on_message(message):
                await self._handle_mesh_message(message, peer_id)

            # Set ICE connection state handler
            @pc.on("iceconnectionstatechange")
            async def on_ice_state_change():
                await self.worker.on_mesh_iceconnectionstatechange(peer_id, pc)

            # Set ICE candidate handler - send candidates via mesh signaling (through admin)
            @pc.on("icecandidate")
            async def on_ice_candidate(event):
                if event.candidate:
                    await self._send_mesh_message(
                        {
                            "type": MSG_MESH_ICE_CANDIDATE,
                            "from_peer_id": self.worker.peer_id,
                            "to_peer_id": peer_id,
                            "candidate": {
                                "candidate": event.candidate.candidate,
                                "sdpMLineIndex": event.candidate.sdpMLineIndex,
                                "sdpMid": event.candidate.sdpMid,
                            },
                        }
                    )
                    logger.debug(
                        f"Sent ICE candidate to worker {peer_id} via mesh signaling"
                    )

            # Create offer
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)

            # Send offer to worker via admin (mesh signaling)
            await self._send_mesh_message(
                {
                    "type": MSG_MESH_OFFER,
                    "from_peer_id": self.worker.peer_id,
                    "to_peer_id": peer_id,
                    "offer": {
                        "sdp": pc.localDescription.sdp,
                        "type": pc.localDescription.type,
                    },
                }
            )

            # Store connection in registry
            self.worker.worker_connections[peer_id] = pc
            self.connected_peers.add(peer_id)

            return True

        except Exception as e:
            logger.error(f"Failed to connect to worker {peer_id}: {e}")
            return False

    async def _send_mesh_message(self, message: Dict[str, Any]):
        """Send a message via mesh signaling (through admin).

        Args:
            message: Message dictionary to send
        """
        # If we're admin, handle directly
        if self.admin_controller.is_admin:
            await self._relay_mesh_message(message)
        else:
            # Send via admin's data channel
            admin_peer_id = self.worker.admin_peer_id
            if not admin_peer_id:
                logger.error("Cannot send mesh message: no admin peer_id")
                return

            if admin_peer_id not in self.worker.data_channels:
                logger.error(
                    f"Cannot send mesh message: no data channel to admin {admin_peer_id}"
                )
                return

            data_channel = self.worker.data_channels[admin_peer_id]

            if data_channel.readyState != "open":
                logger.error(
                    f"Cannot send mesh message: admin channel state is {data_channel.readyState}"
                )
                return

            try:
                # Send JSON-encoded message
                message_str = json.dumps(message)
                data_channel.send(message_str)
                logger.debug(f"Sent mesh message via admin: {message['type']}")
            except Exception as e:
                logger.error(f"Failed to send mesh message via admin: {e}")

    async def _send_peer_list_to_worker(self, target_peer_id: str):
        """Send list of connected peers to a newly connected worker.

        Called by admin when a new worker's data channel opens.
        Enables the new worker to establish connections to other workers.

        Args:
            target_peer_id: peer_id of the worker to send the list to
        """
        if not self.admin_controller.is_admin:
            logger.debug("Not sending peer list: not admin")
            return

        # Get list of other connected workers (excluding self and target)
        other_peers = [
            pid
            for pid in self.connected_peers
            if pid != self.worker.peer_id and pid != target_peer_id
        ]

        if not other_peers:
            logger.info(f"No other peers to send to {target_peer_id}")
            return

        # Gather peer metadata from CRDT if available
        peer_metadata = {}
        if self.worker.room_state_crdt:
            all_workers = self.worker.room_state_crdt.get_all_workers()
            for pid in other_peers:
                if pid in all_workers:
                    worker_data = all_workers[pid]
                    peer_metadata[pid] = worker_data.get("metadata", {})

        # Send peer list message
        message = {
            "type": MSG_MESH_PEER_LIST,
            "peer_ids": other_peers,
            "peer_metadata": peer_metadata,
        }

        # Get data channel to target
        if target_peer_id not in self.worker.data_channels:
            logger.error(f"Cannot send peer list: no data channel to {target_peer_id}")
            return

        data_channel = self.worker.data_channels[target_peer_id]

        if data_channel.readyState != "open":
            logger.warning(
                f"Cannot send peer list: channel to {target_peer_id} is {data_channel.readyState}"
            )
            return

        try:
            message_str = json.dumps(message)
            data_channel.send(message_str)
            logger.info(f"Sent peer list to {target_peer_id}: {len(other_peers)} peers")
        except Exception as e:
            logger.error(f"Failed to send peer list to {target_peer_id}: {e}")

    async def _notify_existing_workers_of_new_peer(self, new_peer_id: str):
        """Notify all existing workers that a new peer has joined.

        Called by admin when a new worker connects. Enables existing workers
        to proactively establish connections to the new peer.

        Args:
            new_peer_id: peer_id of the newly connected worker
        """
        if not self.admin_controller.is_admin:
            logger.debug("Not notifying workers: not admin")
            return

        # Get new peer's metadata from CRDT if available
        peer_metadata = {}
        if self.worker.room_state_crdt:
            all_workers = self.worker.room_state_crdt.get_all_workers()
            if new_peer_id in all_workers:
                peer_metadata = all_workers[new_peer_id].get("metadata", {})

        # Create notification message
        message = {
            "type": MSG_MESH_PEER_JOINED,
            "peer_id": new_peer_id,
            "metadata": peer_metadata,
        }

        # Send to all existing connected workers (except the new one)
        message_str = json.dumps(message)
        notified_count = 0

        for peer_id in self.connected_peers:
            if peer_id == new_peer_id or peer_id == self.worker.peer_id:
                continue

            if peer_id not in self.worker.data_channels:
                continue

            data_channel = self.worker.data_channels[peer_id]
            if data_channel.readyState != "open":
                continue

            try:
                data_channel.send(message_str)
                notified_count += 1
            except Exception as e:
                logger.error(f"Failed to notify {peer_id} of new peer: {e}")

        if notified_count > 0:
            logger.info(f"Notified {notified_count} workers of new peer: {new_peer_id}")

    async def _do_admin_duties_async(
        self, new_peer_id: str, offer_data: Dict[str, Any]
    ):
        """Perform admin duties when a new worker connects.

        This is called when a new worker's data channel opens with the admin.
        Handles CRDT synchronization and mesh coordination.

        Args:
            new_peer_id: peer_id of the newly connected worker
            offer_data: Original offer data which may contain worker metadata
        """
        try:
            # 1. Extract metadata from offer data (signaling server should include it)
            metadata = offer_data.get("metadata", {})
            if not metadata:
                # Fallback: create minimal metadata
                logger.warning(
                    f"No metadata in offer from {new_peer_id}, using defaults"
                )
                metadata = {
                    "tags": ["sleap-connect"],
                    "properties": {"status": "available"},
                }

            # 2. Add new worker to admin's CRDT
            if self.worker.room_state_crdt:
                logger.info(f"Adding new worker {new_peer_id} to CRDT")
                self.worker.room_state_crdt.add_worker(
                    new_peer_id, metadata, is_admin=False
                )

                # Also update via AdminController for consistency
                if self.admin_controller:
                    # Note: We don't call handle_worker_joined because that would
                    # trigger re-election, which we don't want for normal joins.
                    # Re-election only happens if new worker has more GPU memory,
                    # but we handle that separately.
                    pass

            # 3. Broadcast updated CRDT state to ALL workers (including the new one)
            if self.admin_controller and self.admin_controller.is_admin:
                logger.info("Broadcasting updated CRDT state to all workers")
                await self.admin_controller.broadcast_state_update()

            # 4. Send peer list to the new worker
            await self._send_peer_list_to_worker(new_peer_id)

            # 5. Notify existing workers about the new peer
            await self._notify_existing_workers_of_new_peer(new_peer_id)

            logger.info(f"Admin duties completed for new worker: {new_peer_id}")

        except Exception as e:
            logger.error(f"Failed to complete admin duties for {new_peer_id}: {e}")

    async def _relay_mesh_message(self, message: Dict[str, Any]):
        """Relay a mesh message to its destination (admin only).

        Admin relays SDP offers/answers and ICE candidates between workers
        who don't have direct connections yet.

        Args:
            message: Message to relay (must have to_peer_id field)
        """
        if not self.admin_controller.is_admin:
            logger.warning("Cannot relay mesh message: not admin")
            return

        to_peer_id = message.get("to_peer_id")
        if not to_peer_id:
            logger.error("Cannot relay message: no to_peer_id")
            return

        # Get data channel to target peer
        if to_peer_id not in self.worker.data_channels:
            logger.error(f"Cannot relay message: no data channel to {to_peer_id}")
            return

        data_channel = self.worker.data_channels[to_peer_id]

        if data_channel.readyState != "open":
            logger.warning(
                f"Cannot relay message to {to_peer_id}: channel state is {data_channel.readyState}"
            )
            return

        try:
            message_str = json.dumps(message)
            data_channel.send(message_str)
            logger.debug(
                f"Relayed {message['type']} from {message.get('from_peer_id')} to {to_peer_id}"
            )
        except Exception as e:
            logger.error(f"Failed to relay message to {to_peer_id}: {e}")

    async def _handle_mesh_message(self, message: str, from_peer_id: str):
        """Handle incoming mesh signaling message.

        Args:
            message: JSON string message
            from_peer_id: peer_id of sender
        """
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            # Check if this message should be relayed (admin only)
            # Messages with to_peer_id != self should be forwarded to target
            to_peer_id = data.get("to_peer_id")
            if (
                to_peer_id
                and to_peer_id != self.worker.peer_id
                and self.admin_controller.is_admin
            ):
                logger.debug(f"Relaying {msg_type} from {from_peer_id} to {to_peer_id}")
                await self._relay_mesh_message(data)
                return

            # Mesh signaling messages (WebRTC connection establishment)
            # For relayed messages, use from_peer_id from message data (original sender)
            # not from the data channel sender (which would be admin for relayed messages)
            if msg_type == MSG_MESH_OFFER:
                original_sender = data.get("from_peer_id", from_peer_id)
                await self._handle_mesh_offer(data, original_sender)
            elif msg_type == MSG_MESH_ANSWER:
                original_sender = data.get("from_peer_id", from_peer_id)
                await self._handle_mesh_answer(data, original_sender)
            elif msg_type == MSG_MESH_ICE_CANDIDATE:
                original_sender = data.get("from_peer_id", from_peer_id)
                await self._handle_mesh_ice_candidate(data, original_sender)
            elif msg_type == MSG_MESH_PEER_JOINED:
                await self._handle_peer_joined(data)
            elif msg_type == MSG_MESH_PEER_LIST:
                await self._handle_peer_list(data)
            # Admin verification messages (election confirmation)
            elif msg_type == MSG_ADMIN_VERIFY:
                await self._handle_admin_verify(data, from_peer_id)
            elif msg_type == MSG_ADMIN_VERIFY_ACK:
                await self._handle_admin_verify_ack(data, from_peer_id)
            # Application-level messages (forward to worker)
            elif msg_type == "heartbeat":
                await self._forward_to_worker_handler("heartbeat", data, from_peer_id)
            elif msg_type == "heartbeat_response":
                await self._forward_to_worker_handler(
                    "heartbeat_response", data, from_peer_id
                )
            elif msg_type == "status_update":
                await self._forward_to_worker_handler(
                    "status_update", data, from_peer_id
                )
            elif msg_type == "state_broadcast":
                await self._forward_to_worker_handler(
                    "state_broadcast", data, from_peer_id
                )
            else:
                logger.warning(f"Unknown mesh message type: {msg_type}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse mesh message: {e}")
        except Exception as e:
            logger.error(f"Error handling mesh message: {e}")

    async def _forward_to_worker_handler(
        self, msg_type: str, data: Dict[str, Any], from_peer_id: str
    ):
        """Forward application-level messages to worker handlers.

        Args:
            msg_type: Message type (heartbeat, status_update, etc.)
            data: Message data dictionary
            from_peer_id: peer_id of sender
        """
        from sleap_connect.worker.mesh_messages import (
            HeartbeatMessage,
            HeartbeatResponseMessage,
            StatusUpdateMessage,
            StateBroadcastMessage,
        )

        try:
            if msg_type == "heartbeat":
                message = HeartbeatMessage(
                    from_peer_id=data.get("from_peer_id"),
                    timestamp=data.get("timestamp"),
                    sequence=data.get("sequence"),
                )
                await self.worker._handle_heartbeat(message, from_peer_id)
            elif msg_type == "heartbeat_response":
                message = HeartbeatResponseMessage(
                    from_peer_id=data.get("from_peer_id"),
                    to_peer_id=data.get("to_peer_id"),
                    timestamp=data.get("timestamp"),
                    original_timestamp=data.get("original_timestamp"),
                )
                await self.worker._handle_heartbeat_response(message, from_peer_id)
            elif msg_type == "status_update":
                message = StatusUpdateMessage(
                    from_peer_id=data.get("from_peer_id"),
                    status=data.get("status"),
                    current_job=data.get("current_job"),
                    timestamp=data.get("timestamp"),
                )
                await self.worker._handle_status_update(message, from_peer_id)
            elif msg_type == "state_broadcast":
                message = StateBroadcastMessage(
                    from_peer_id=data.get("from_peer_id"),
                    timestamp=data.get("timestamp"),
                    crdt_snapshot=data.get("crdt_snapshot"),
                    version=data.get("version"),
                )
                await self.worker._handle_state_broadcast(message, from_peer_id)
            else:
                logger.warning(f"Unknown message type to forward: {msg_type}")
        except Exception as e:
            logger.error(f"Error forwarding {msg_type} message: {e}")

    async def _handle_mesh_offer(self, data: Dict[str, Any], from_peer_id: str):
        """Handle incoming connection offer from another worker.

        Args:
            data: Offer data dictionary
            from_peer_id: peer_id of offering worker
        """
        logger.info(f"Received mesh offer from {from_peer_id}")

        try:
            # Create peer connection — use factory to avoid default STUN hang
            pc = self.worker._create_mesh_peer_connection()

            # Set up datachannel event handler to receive incoming channel
            @pc.on("datachannel")
            def on_datachannel(channel):
                logger.info(
                    f"Received data channel from {from_peer_id}: {channel.label} "
                    f"(state: {channel.readyState})"
                )

                # Store data channel IMMEDIATELY - don't wait for on_open
                # The sending code will check readyState before sending
                self.worker.data_channels[from_peer_id] = channel
                logger.info(f"Stored data channel for {from_peer_id}")

                # Set up channel event handlers
                @channel.on("open")
                def on_open():
                    logger.info(f"Data channel now open with {from_peer_id}")

                @channel.on("close")
                def on_close():
                    logger.warning(f"Data channel closed with {from_peer_id}")
                    # Remove data channel reference
                    if from_peer_id in self.worker.data_channels:
                        del self.worker.data_channels[from_peer_id]

                @channel.on("error")
                def on_error(error):
                    logger.error(f"Data channel error with {from_peer_id}: {error}")

                @channel.on("message")
                async def on_message(message):
                    await self._handle_mesh_message(message, from_peer_id)

            # Set ICE connection state handler
            @pc.on("iceconnectionstatechange")
            async def on_ice_state_change():
                await self.worker.on_mesh_iceconnectionstatechange(from_peer_id, pc)

            # Set ICE candidate handler - send candidates via mesh signaling (through admin)
            @pc.on("icecandidate")
            async def on_ice_candidate(event):
                if event.candidate:
                    await self._send_mesh_message(
                        {
                            "type": MSG_MESH_ICE_CANDIDATE,
                            "from_peer_id": self.worker.peer_id,
                            "to_peer_id": from_peer_id,
                            "candidate": {
                                "candidate": event.candidate.candidate,
                                "sdpMLineIndex": event.candidate.sdpMLineIndex,
                                "sdpMid": event.candidate.sdpMid,
                            },
                        }
                    )
                    logger.debug(
                        f"Sent ICE candidate to {from_peer_id} via mesh signaling"
                    )

            # Set remote description (offer)
            offer = RTCSessionDescription(
                sdp=data["offer"]["sdp"], type=data["offer"]["type"]
            )
            await pc.setRemoteDescription(offer)

            # Create answer
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            # Send answer back
            await self._send_mesh_message(
                {
                    "type": MSG_MESH_ANSWER,
                    "from_peer_id": self.worker.peer_id,
                    "to_peer_id": from_peer_id,
                    "answer": {
                        "sdp": pc.localDescription.sdp,
                        "type": pc.localDescription.type,
                    },
                }
            )

            # Store connection
            self.worker.worker_connections[from_peer_id] = pc
            self.connected_peers.add(from_peer_id)

            logger.info(f"Sent answer to {from_peer_id}")

        except Exception as e:
            logger.error(f"Failed to handle mesh offer: {e}")

    async def _handle_mesh_answer(self, data: Dict[str, Any], from_peer_id: str):
        """Handle incoming connection answer from another worker.

        Args:
            data: Answer data dictionary
            from_peer_id: peer_id of answering worker
        """
        logger.info(f"Received mesh answer from {from_peer_id}")

        try:
            # Get existing connection
            if from_peer_id not in self.worker.worker_connections:
                logger.error(f"No pending connection for {from_peer_id}")
                return

            pc = self.worker.worker_connections[from_peer_id]

            # Set remote description (answer)
            answer = RTCSessionDescription(
                sdp=data["answer"]["sdp"], type=data["answer"]["type"]
            )
            await pc.setRemoteDescription(answer)

            logger.info(f"Connection established with {from_peer_id}")

        except Exception as e:
            logger.error(f"Failed to handle mesh answer: {e}")

    async def _handle_mesh_ice_candidate(self, data: Dict[str, Any], from_peer_id: str):
        """Handle incoming ICE candidate from another worker.

        Adds the ICE candidate to the peer connection to help establish
        the optimal network path between workers.

        Args:
            data: ICE candidate data with candidate, sdpMLineIndex, sdpMid
            from_peer_id: peer_id of sender
        """
        # Determine the actual peer this ICE candidate is for
        # It could be relayed by admin, so check from_peer_id in the message
        actual_from_peer_id = data.get("from_peer_id", from_peer_id)

        logger.debug(f"Received ICE candidate from {actual_from_peer_id}")

        # Get peer connection for this peer
        if actual_from_peer_id not in self.worker.worker_connections:
            logger.warning(
                f"Received ICE candidate for unknown peer: {actual_from_peer_id}"
            )
            return

        pc = self.worker.worker_connections[actual_from_peer_id]

        try:
            candidate_data = data.get("candidate", {})
            if not candidate_data:
                logger.debug("Received empty ICE candidate (end of candidates)")
                return

            candidate = RTCIceCandidate(
                candidate=candidate_data.get("candidate"),
                sdpMLineIndex=candidate_data.get("sdpMLineIndex"),
                sdpMid=candidate_data.get("sdpMid"),
            )
            await pc.addIceCandidate(candidate)
            logger.debug(f"Added ICE candidate from {actual_from_peer_id}")

        except Exception as e:
            logger.error(f"Failed to add ICE candidate from {actual_from_peer_id}: {e}")

    async def _handle_peer_joined(self, data: Dict[str, Any]):
        """Handle notification that a new peer joined the room.

        Updates local CRDT with new peer's info and establishes mesh connection.

        Args:
            data: Peer joined notification data
                - peer_id: New worker's peer_id
                - metadata: New worker's metadata (optional)
        """
        peer_id = data.get("peer_id")
        if not peer_id:
            logger.warning("peer_joined message missing peer_id")
            return

        logger.info(f"New peer joined room: {peer_id}")

        # 1. Add new peer to local CRDT for consistent election state
        metadata = data.get("metadata", {})
        if not metadata:
            # Use minimal defaults if no metadata provided
            metadata = {
                "tags": ["sleap-connect"],
                "properties": {"status": "available"},
            }

        if self.worker.room_state_crdt:
            try:
                self.worker.room_state_crdt.add_worker(
                    peer_id, metadata, is_admin=False
                )
                logger.info(f"Added new peer {peer_id} to local CRDT")
            except Exception as e:
                logger.error(f"Failed to add peer {peer_id} to CRDT: {e}")

        # 2. Connect to new peer if we're not in initial mesh formation
        if not self.mesh_formation_in_progress:
            await self._connect_to_worker(peer_id)

    async def _handle_peer_list(self, data: Dict[str, Any]):
        """Handle list of existing peers in room.

        Updates local CRDT with peer metadata and establishes mesh connections.

        Args:
            data: Peer list data
                - peer_ids: List of peer IDs in the room
                - peer_metadata: Dict mapping peer_id to metadata (optional)
        """
        peer_ids = data.get("peer_ids", [])
        peer_metadata = data.get("peer_metadata", {})
        logger.info(f"Received peer list: {len(peer_ids)} workers")

        # 1. Add all peers to local CRDT for consistent election state
        if self.worker.room_state_crdt:
            for pid in peer_ids:
                if pid != self.worker.peer_id:  # Don't add self
                    metadata = peer_metadata.get(pid, {})
                    if not metadata:
                        metadata = {
                            "tags": ["sleap-connect"],
                            "properties": {"status": "available"},
                        }
                    try:
                        self.worker.room_state_crdt.add_worker(
                            pid, metadata, is_admin=False
                        )
                        logger.debug(f"Added peer {pid} to local CRDT from peer list")
                    except Exception as e:
                        logger.error(f"Failed to add peer {pid} to CRDT: {e}")

        # 2. Filter out self and admin (already connected) for mesh connections
        workers_to_connect = [
            pid
            for pid in peer_ids
            if pid != self.worker.peer_id and pid != self.worker.admin_peer_id
        ]

        # 3. Connect to all workers in batches
        if workers_to_connect:
            await self.connect_to_workers_batched(workers_to_connect)

    async def _handle_admin_verify(self, data: Dict[str, Any], from_peer_id: str):
        """Handle admin verification request.

        When a worker asks us to confirm we're the admin (after election),
        respond with an acknowledgment.

        Args:
            data: Verification request data
            from_peer_id: peer_id of requesting worker
        """
        request_id = data.get("request_id")
        logger.info(
            f"Received admin_verify from {from_peer_id} (request_id: {request_id})"
        )

        # Send acknowledgment back
        ack_message = {
            "type": MSG_ADMIN_VERIFY_ACK,
            "from_peer_id": self.worker.peer_id,
            "request_id": request_id,
        }

        # Send directly to the requesting peer
        if from_peer_id in self.worker.data_channels:
            data_channel = self.worker.data_channels[from_peer_id]
            if data_channel.readyState == "open":
                try:
                    data_channel.send(json.dumps(ack_message))
                    logger.info(f"Sent admin_verify_ack to {from_peer_id}")
                except Exception as e:
                    logger.error(f"Failed to send admin_verify_ack: {e}")
            else:
                logger.warning(
                    f"Cannot send admin_verify_ack: channel to {from_peer_id} is {data_channel.readyState}"
                )
        else:
            logger.warning(
                f"Cannot send admin_verify_ack: no channel to {from_peer_id}"
            )

    async def _handle_admin_verify_ack(self, data: Dict[str, Any], from_peer_id: str):
        """Handle admin verification acknowledgment.

        When the elected admin confirms they're alive, notify the AdminController.

        Args:
            data: Acknowledgment data with request_id
            from_peer_id: peer_id of the admin
        """
        request_id = data.get("request_id")
        logger.info(
            f"Received admin_verify_ack from {from_peer_id} (request_id: {request_id})"
        )

        # Notify AdminController that verification succeeded
        if self.admin_controller:
            self.admin_controller.on_admin_verify_ack(from_peer_id, request_id)

    async def send_admin_verify(self, target_peer_id: str, request_id: str) -> bool:
        """Send admin verification request to a peer.

        Called by AdminController after election to verify the elected admin is alive.

        Args:
            target_peer_id: peer_id of the elected admin to verify
            request_id: Unique ID for this verification request

        Returns:
            True if message was sent, False otherwise
        """
        if target_peer_id not in self.worker.data_channels:
            logger.warning(f"Cannot send admin_verify: no channel to {target_peer_id}")
            return False

        data_channel = self.worker.data_channels[target_peer_id]
        if data_channel.readyState != "open":
            logger.warning(
                f"Cannot send admin_verify: channel to {target_peer_id} is {data_channel.readyState}"
            )
            return False

        message = {
            "type": MSG_ADMIN_VERIFY,
            "from_peer_id": self.worker.peer_id,
            "request_id": request_id,
        }

        try:
            data_channel.send(json.dumps(message))
            logger.info(
                f"Sent admin_verify to {target_peer_id} (request_id: {request_id})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send admin_verify: {e}")
            return False

    async def on_admin_promotion(self):
        """Handle this worker becoming admin.

        Opens WebSocket connection to signaling server for discovery by
        new workers.
        """
        logger.info("Worker promoted to admin - opening WebSocket for discovery")

        try:
            # Reconnect to signaling server
            import websockets

            if not self.websocket_dns:
                logger.error("Cannot reconnect: no DNS stored")
                return

            self.websocket = await websockets.connect(self.websocket_dns)

            # CRITICAL: Sync websocket references so state_manager uses new connection
            self.worker.websocket = self.websocket
            if self.worker.state_manager:
                self.worker.state_manager.websocket = self.websocket
                logger.info("Updated state_manager websocket reference")

            # Re-register with server as admin (include full metadata for discovery)
            await self.websocket.send(
                json.dumps(self._build_registration_msg(is_admin=True))
            )

            logger.info("Admin WebSocket reconnected")
            self._start_admin_websocket_handler()

        except Exception as e:
            logger.error(f"Failed to open admin WebSocket: {e}")

    async def on_admin_demotion(self):
        """Handle this worker losing admin status due to conflict resolution.

        When another worker is already admin (split-brain resolution), this worker:
        1. Cancels admin handler task
        2. Closes admin WebSocket
        3. Reopens WebSocket as non-admin
        4. Connects to the actual admin
        5. Starts non-admin signaling handler
        """
        logger.info("Worker demoted from admin - reconnecting as non-admin")

        # 1. Cancel admin handler task
        if self._admin_handler_task and not self._admin_handler_task.done():
            self._admin_handler_task.cancel()
            try:
                await self._admin_handler_task
            except asyncio.CancelledError:
                pass
            self._admin_handler_task = None
            logger.info("Admin handler task cancelled")

        # 2. Close admin WebSocket
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            logger.info("Admin WebSocket closed")

        # 3. Reconnect as non-admin worker
        actual_admin = self.admin_controller.admin_peer_id
        if not actual_admin or actual_admin == self.worker.peer_id:
            logger.error("Cannot reconnect: no valid admin peer_id")
            return

        try:
            import websockets

            if not self.websocket_dns:
                logger.error("Cannot reconnect: no DNS stored")
                return

            logger.info(f"Reopening WebSocket to connect to admin: {actual_admin}")
            self.websocket = await websockets.connect(self.websocket_dns)

            # Sync websocket references
            self.worker.websocket = self.websocket
            if self.worker.state_manager:
                self.worker.state_manager.websocket = self.websocket
                logger.info("Updated state_manager websocket reference")

            # 4. Re-register as non-admin worker
            await self.websocket.send(
                json.dumps(self._build_registration_msg(is_admin=False))
            )
            logger.info("Re-registered as non-admin worker")

            # 5. Start non-admin signaling handler (for mesh_answer, ice_candidate)
            self._start_non_admin_websocket_handler()

            # 6. Connect to actual admin
            success = await self.connect_to_admin(actual_admin)
            if success:
                logger.info(f"Successfully connected to actual admin: {actual_admin}")
            else:
                logger.error(f"Failed to connect to actual admin: {actual_admin}")

        except Exception as e:
            logger.error(f"Failed to reconnect as non-admin: {e}")

    def _start_non_admin_websocket_handler(self):
        """Start background task to handle WebSocket signaling messages (non-admin).

        Handles mesh_answer and ice_candidate messages for connection establishment
        after demotion from admin status.
        """
        # Cancel existing handler if any
        if self._non_admin_handler_task and not self._non_admin_handler_task.done():
            self._non_admin_handler_task.cancel()

        # Start new handler task
        self._non_admin_handler_task = asyncio.create_task(
            self._non_admin_websocket_handler_loop()
        )
        logger.info("Non-admin WebSocket handler started")

    async def _non_admin_websocket_handler_loop(self):
        """Background task that handles WebSocket messages for non-admin after demotion.

        Processes signaling messages needed to establish connection to actual admin:
        - mesh_answer: Connection answer from admin
        - ice_candidate: ICE candidates for mesh connections
        - registered_auth: Registration confirmation
        - error: Error messages from signaling server
        """
        logger.info("Starting non-admin WebSocket handler loop")

        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "registered_auth":
                        logger.info(
                            f"Non-admin re-registration confirmed: {data.get('peer_id')} "
                            f"in room {data.get('room_id')}"
                        )

                    elif msg_type == "mesh_answer":
                        # Connection answer from admin
                        await self.handle_signaling_answer(data)

                    elif msg_type == "ice_candidate":
                        # ICE candidates for mesh connection
                        await self.handle_signaling_ice_candidate(data)

                    elif msg_type == "worker_joined":
                        # New worker notification (for future mesh expansion)
                        peer_id = data.get("peer_id")
                        logger.info(f"Worker joined notification: {peer_id}")

                    elif msg_type == "error":
                        logger.error(f"Non-admin received error: {data.get('reason')}")

                    # ── E2E encryption ─────────────────────────────────
                    elif msg_type == "key_exchange":
                        await self._handle_key_exchange(data)

                    elif msg_type == "encrypted_relay":
                        decrypted, session_id = self._decrypt_if_encrypted(data)
                        if decrypted is not None:
                            inner_type = decrypted.get("type")
                            if inner_type == "fs_list_req":
                                await self._handle_fs_list_req(
                                    decrypted, session_id=session_id
                                )
                            elif inner_type == "use_worker_path":
                                await self._handle_use_worker_path(
                                    decrypted, session_id=session_id
                                )
                            elif inner_type == "fs_check_videos":
                                await self._handle_fs_check_videos(
                                    decrypted, session_id=session_id
                                )
                            elif inner_type == "job_assigned":
                                await self._handle_job_assigned(
                                    decrypted, session_id=session_id
                                )
                            elif inner_type == "job_cancel":
                                self._handle_job_cancel(decrypted)
                            else:
                                logger.debug(
                                    f"[E2E] Unknown decrypted type: {inner_type}"
                                )

                    # ── Relay-forwarded requests from dashboard ──────────
                    elif msg_type == "fs_list_req":
                        await self._handle_fs_list_req(data)

                    elif msg_type == "use_worker_path":
                        await self._handle_use_worker_path(data)

                    elif msg_type == "fs_check_videos":
                        await self._handle_fs_check_videos(data)

                    elif msg_type == "job_assigned":
                        await self._handle_job_assigned(data)

                    elif msg_type == "job_cancel":
                        self._handle_job_cancel(data)

                    elif msg_type == "ping":
                        self.worker._last_signaling_ping = time.monotonic()
                        logging.warning(
                            "\033[33m[HEARTBEAT] Received signaling server ping\033[0m"
                        )

                    else:
                        logger.debug(f"Non-admin ignoring message type: {msg_type}")

                except json.JSONDecodeError:
                    logger.error("Invalid JSON received in non-admin handler")
                except Exception as e:
                    logger.error(f"Error processing non-admin message: {e}")

        except asyncio.CancelledError:
            logger.info("Non-admin WebSocket handler cancelled")
        except Exception as e:
            logger.error(f"Non-admin WebSocket handler error: {e}")

    async def close_websocket_after_mesh_connection(self):
        """Close WebSocket after successfully connecting to mesh (non-admin only).

        This is called after a non-admin worker has connected to the admin
        and established initial mesh connectivity.
        """
        if self.admin_controller.is_admin:
            logger.info("Not closing WebSocket: this worker is admin")
            return

        if not self.websocket:
            logger.debug("WebSocket already closed")
            return

        logger.info("Closing WebSocket after mesh connection (non-admin)")
        await self.websocket.close()
        self.websocket = None

    def get_connected_peers(self) -> List[str]:
        """Get list of connected peer IDs.

        Returns:
            List of peer_ids we're connected to
        """
        return list(self.connected_peers)

    def is_connected_to_peer(self, peer_id: str) -> bool:
        """Check if connected to a specific peer.

        Args:
            peer_id: peer_id to check

        Returns:
            True if connected, False otherwise
        """
        return peer_id in self.connected_peers

    async def handle_signaling_offer(self, data: Dict[str, Any]):
        """Handle mesh offer received from signaling server.

        Called when admin receives connection offer from non-admin worker
        via signaling server (not via mesh).

        Args:
            data: Offer data from signaling server
                - from_peer_id: Offering worker's peer_id
                - offer: SDP offer
        """
        from_peer_id = data.get("from_peer_id")
        if not from_peer_id:
            logger.error("mesh_offer missing from_peer_id")
            return

        logger.info(f"Received mesh offer from signaling server: {from_peer_id}")

        try:
            # Create peer connection — use factory to avoid default STUN hang
            pc = self.worker._create_mesh_peer_connection()

            # Set up datachannel event handler to receive incoming channel
            @pc.on("datachannel")
            def on_datachannel(channel):
                logger.info(
                    f"Received data channel from {from_peer_id}: {channel.label} "
                    f"(state: {channel.readyState})"
                )

                # Store data channel IMMEDIATELY - don't wait for on_open
                # The sending code will check readyState before sending
                self.worker.data_channels[from_peer_id] = channel
                logger.info(f"Stored data channel for {from_peer_id}")

                # Define admin duties to run when channel is ready
                def do_admin_duties():
                    logger.info(
                        f"Data channel open with {from_peer_id}, performing admin duties"
                    )
                    # Admin duties:
                    # 1. Add new worker to CRDT (critical for consistent election)
                    # 2. Broadcast updated state to all workers
                    # 3. Send peer list to new worker
                    # 4. Notify existing workers of new peer
                    asyncio.create_task(self._do_admin_duties_async(from_peer_id, data))

                # Set up channel event handlers
                @channel.on("open")
                def on_open():
                    do_admin_duties()

                # If channel is ALREADY open, fire admin duties immediately
                # (on_open won't fire if channel opened before handler was registered)
                if channel.readyState == "open":
                    logger.info(
                        f"Channel already open with {from_peer_id}, triggering admin duties"
                    )
                    do_admin_duties()

                @channel.on("close")
                def on_close():
                    logger.warning(f"Data channel closed with {from_peer_id}")
                    if from_peer_id in self.worker.data_channels:
                        del self.worker.data_channels[from_peer_id]

                @channel.on("error")
                def on_error(error):
                    logger.error(f"Data channel error with {from_peer_id}: {error}")

                @channel.on("message")
                async def on_message(message):
                    await self._handle_mesh_message(message, from_peer_id)

            # Set ICE connection state handler
            @pc.on("iceconnectionstatechange")
            async def on_ice_state_change():
                await self.worker.on_mesh_iceconnectionstatechange(from_peer_id, pc)

            # Set ICE candidate handler - send candidates to signaling server
            @pc.on("icecandidate")
            async def on_ice_candidate(event):
                if event.candidate and self.websocket:
                    await self.websocket.send(
                        json.dumps(
                            {
                                "type": "ice_candidate",
                                "from_peer_id": self.worker.peer_id,
                                "target_peer_id": from_peer_id,
                                "candidate": {
                                    "candidate": event.candidate.candidate,
                                    "sdpMLineIndex": event.candidate.sdpMLineIndex,
                                    "sdpMid": event.candidate.sdpMid,
                                },
                            }
                        )
                    )
                    logger.debug(
                        f"Sent ICE candidate to {from_peer_id} via signaling server"
                    )

            # Set remote description (offer)
            offer = RTCSessionDescription(
                sdp=data["offer"]["sdp"], type=data["offer"]["type"]
            )
            await pc.setRemoteDescription(offer)

            # Create answer
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            # Send answer back via signaling server
            await self.websocket.send(
                json.dumps(
                    {
                        "type": "mesh_answer",
                        "from_peer_id": self.worker.peer_id,
                        "target_peer_id": from_peer_id,
                        "answer": {
                            "sdp": pc.localDescription.sdp,
                            "type": pc.localDescription.type,
                        },
                    }
                )
            )

            # Store connection
            self.worker.worker_connections[from_peer_id] = pc
            self.connected_peers.add(from_peer_id)

            logger.info(f"Sent answer to {from_peer_id} via signaling server")

        except Exception as e:
            logger.error(f"Failed to handle signaling offer: {e}")

    async def handle_signaling_answer(self, data: Dict[str, Any]):
        """Handle mesh answer received from signaling server.

        Called when non-admin receives connection answer from admin
        via signaling server (not via mesh).

        Args:
            data: Answer data from signaling server
                - from_peer_id: Admin's peer_id
                - answer: SDP answer
        """
        from_peer_id = data.get("from_peer_id")
        if not from_peer_id:
            logger.error("mesh_answer missing from_peer_id")
            return

        logger.info(f"Received mesh answer from signaling server: {from_peer_id}")

        try:
            # Get existing connection (should exist from connect_to_admin)
            if from_peer_id not in self.worker.worker_connections:
                logger.error(f"No pending connection for {from_peer_id}")
                return

            pc = self.worker.worker_connections[from_peer_id]

            # Set remote description (answer)
            answer = RTCSessionDescription(
                sdp=data["answer"]["sdp"], type=data["answer"]["type"]
            )
            await pc.setRemoteDescription(answer)

            logger.info(f"Connection established with {from_peer_id} via signaling")

        except Exception as e:
            logger.error(f"Failed to handle signaling answer: {e}")

    async def handle_signaling_ice_candidate(self, data: Dict[str, Any]):
        """Handle ICE candidate received from signaling server.

        Args:
            data: ICE candidate data from signaling server
                - from_peer_id: Sender's peer_id
                - candidate: ICE candidate
        """
        from_peer_id = data.get("from_peer_id")
        candidate_data = data.get("candidate")

        if not from_peer_id or not candidate_data:
            logger.error("ice_candidate missing from_peer_id or candidate")
            return

        logger.debug(f"Received ICE candidate from {from_peer_id}")

        try:
            # Get existing connection
            if from_peer_id not in self.worker.worker_connections:
                # Buffer candidate if connection not established yet
                if from_peer_id not in self.pending_ice_candidates:
                    self.pending_ice_candidates[from_peer_id] = []
                self.pending_ice_candidates[from_peer_id].append(candidate_data)
                logger.debug(f"Buffered ICE candidate for {from_peer_id}")
                return

            pc = self.worker.worker_connections[from_peer_id]

            # Add ICE candidate to connection
            candidate = RTCIceCandidate(
                candidate=candidate_data.get("candidate"),
                sdpMLineIndex=candidate_data.get("sdpMLineIndex"),
                sdpMid=candidate_data.get("sdpMid"),
            )
            await pc.addIceCandidate(candidate)

            logger.debug(f"Added ICE candidate from {from_peer_id}")

            # Check if we have buffered candidates for this peer
            if from_peer_id in self.pending_ice_candidates:
                for buffered in self.pending_ice_candidates[from_peer_id]:
                    buffered_candidate = RTCIceCandidate(
                        candidate=buffered.get("candidate"),
                        sdpMLineIndex=buffered.get("sdpMLineIndex"),
                        sdpMid=buffered.get("sdpMid"),
                    )
                    await pc.addIceCandidate(buffered_candidate)
                    logger.debug(f"Added buffered ICE candidate for {from_peer_id}")

                del self.pending_ice_candidates[from_peer_id]

        except Exception as e:
            logger.error(f"Failed to handle ICE candidate: {e}")
