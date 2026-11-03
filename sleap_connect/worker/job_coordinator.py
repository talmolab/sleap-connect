"""Job coordination and peer messaging for worker nodes.

This module handles peer-to-peer messaging via the signaling server,
job request handling, and job lifecycle management for sleap-connect workers.
"""

import logging
import time
from typing import TYPE_CHECKING, Callable, Dict, Optional

if TYPE_CHECKING:
    from websockets.client import ClientConnection
    from sleap_connect.worker.capabilities import WorkerCapabilities


class JobCoordinator:
    """Manages job coordination and peer messaging for workers.

    This class handles peer messages via the signaling server, evaluates
    job requests, manages job assignments, and tracks job state.

    Attributes:
        worker_id: Unique identifier for this worker.
        websocket: WebSocket connection to signaling server.
        capabilities: WorkerCapabilities instance for job compatibility.
        update_status_callback: Callback to update worker status.
        get_status_callback: Callback to get current worker status.
        current_job: Currently assigned job information.
    """

    def __init__(
        self,
        worker_id: str,
        websocket: "ClientConnection",
        capabilities: "WorkerCapabilities",
        update_status_callback: Callable,
        get_status_callback: Callable,
    ):
        """Initialize job coordinator.

        Args:
            worker_id: Unique identifier for this worker.
            websocket: WebSocket connection to signaling server.
            capabilities: WorkerCapabilities instance for job evaluation.
            update_status_callback: Async callback to update worker status.
            get_status_callback: Callback to get current worker status.
        """
        self.worker_id = worker_id
        self.websocket = websocket
        self.capabilities = capabilities
        self.update_status_callback = update_status_callback
        self.get_status_callback = get_status_callback

        # Job state tracking
        self.current_job: Optional[Dict] = None

    async def send_peer_message(self, to_peer_id: str, payload: dict):
        """Send peer message via signaling server.

        Args:
            to_peer_id: Target peer ID.
            payload: Application-specific message payload.
        """
        try:
            import json

            await self.websocket.send(
                json.dumps(
                    {
                        "type": "peer_message",
                        "from_peer_id": self.worker_id,
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

    async def handle_peer_message(self, message: dict):
        """Handle incoming peer messages (job requests, assignments, cancels).

        Args:
            message: Peer message from signaling server.
        """
        if message.get("type") != "peer_message":
            return

        payload = message.get("payload", {})
        app_message_type = payload.get("app_message_type")
        from_peer_id = message.get("from_peer_id")

        logging.info(f"Received peer message from {from_peer_id}: {app_message_type}")

        if app_message_type == "job_request":
            await self._handle_job_request(from_peer_id, payload)
        elif app_message_type == "job_assignment":
            await self._handle_job_assignment(from_peer_id, payload)
        elif app_message_type == "job_cancel":
            await self._handle_job_cancel(payload)
        else:
            logging.warning(f"Unhandled peer message type: {app_message_type}")

    async def _handle_job_request(self, client_id: str, request: dict):
        """Respond to job request from client.

        Evaluates job compatibility and sends acceptance or rejection response.

        Args:
            client_id: Client peer ID.
            request: Job request payload containing job_id, job_type, config, requirements.
        """
        job_id = request.get("job_id")
        job_type = request.get("job_type")

        logging.info(
            f"Handling job request {job_id} of type {job_type} from {client_id}"
        )

        # Get current status
        current_status = self.get_status_callback()

        # Check if we can accept this job
        can_accept = (
            current_status == "available"
            and self.capabilities.check_job_compatibility(request)
        )

        if not can_accept:
            # Send rejection
            reason = "busy" if current_status != "available" else "incompatible"
            logging.info(f"Rejecting job {job_id}: {reason}")

            await self.send_peer_message(
                client_id,
                {
                    "app_message_type": "job_response",
                    "job_id": job_id,
                    "accepted": False,
                    "reason": reason,
                },
            )
            return

        # Estimate job duration
        estimated_duration = self.capabilities.estimate_job_duration(request)

        # Send acceptance
        logging.info(
            f"Accepting job {job_id}, estimated duration: {estimated_duration} minutes"
        )

        await self.send_peer_message(
            client_id,
            {
                "app_message_type": "job_response",
                "job_id": job_id,
                "accepted": True,
                "estimated_start_time_sec": 0,
                "estimated_duration_minutes": estimated_duration,
                "worker_info": {
                    "gpu_utilization": self.capabilities.get_gpu_utilization(),
                    "available_memory_mb": self.capabilities.get_available_memory(),
                },
            },
        )

        # Update status to "reserved" (prevent other clients from requesting)
        await self.update_status_callback("reserved", pending_job_id=job_id)

    async def _handle_job_assignment(self, client_id: str, assignment: dict):
        """Handle job assignment from client.

        Marks the job as assigned and updates worker status to busy.

        Args:
            client_id: Client peer ID.
            assignment: Job assignment payload containing job_id.
        """
        job_id = assignment.get("job_id")

        logging.info(f"Handling job assignment {job_id} from {client_id}")

        # Update status to busy
        await self.update_status_callback("busy", current_job_id=job_id)

        # Store job info for execution
        self.current_job = {
            "job_id": job_id,
            "client_id": client_id,
            "assigned_at": time.time(),
        }

        logging.info(f"Job {job_id} assigned and ready for WebRTC connection")

        # Note: WebRTC connection will be initiated by the client
        # The existing on_datachannel handler will handle the data transfer

    async def _handle_job_cancel(self, payload: dict):
        """Handle job cancellation request.

        Cancels the current job and sends acknowledgment to client.

        Args:
            payload: Job cancel payload containing job_id and reason.
        """
        job_id = payload.get("job_id")
        reason = payload.get("reason", "unknown")

        logging.info(f"Handling job cancellation for {job_id}: {reason}")

        if self.current_job and self.current_job.get("job_id") == job_id:
            # Cancel the current job
            client_id = self.current_job.get("client_id")

            # Send cancellation acknowledgment
            await self.send_peer_message(
                client_id,
                {
                    "app_message_type": "job_cancelled",
                    "job_id": job_id,
                    "status": "cancelled",
                    "cleanup_complete": True,
                },
            )

            # Reset job state
            self.current_job = None
            await self.update_status_callback("available")

            logging.info(f"Job {job_id} cancelled successfully")
        else:
            logging.warning(f"Cannot cancel job {job_id}: not currently running")

    def get_current_job(self) -> Optional[Dict]:
        """Get current job information.

        Returns:
            Current job dict with job_id, client_id, assigned_at, or None.
        """
        return self.current_job

    def clear_current_job(self):
        """Clear current job state."""
        self.current_job = None
