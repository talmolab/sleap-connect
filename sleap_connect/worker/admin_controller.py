"""Admin election and coordination for room-level worker registry.

This module implements deterministic admin election and handles admin-specific
responsibilities like state broadcasting and client query handling.

The admin worker is elected based on GPU memory (highest wins) with peer_id
as a tiebreaker. All workers in a room independently compute the same result,
ensuring consistency without network communication.
"""

import asyncio
import logging
from typing import Dict, Optional, Any

from sleap_connect.worker.crdt_state import RoomStateCRDT

logger = logging.getLogger(__name__)


class AdminController:
    """Manages admin election and admin-specific coordination tasks.

    The AdminController determines which worker in a room should be the admin
    and handles admin responsibilities like:
    - Deterministic election based on worker capabilities
    - Admin departure detection and re-election
    - State broadcasting to all workers (implemented in Phase 4)
    - Client query handling (implemented in Phase 5)

    Attributes:
        worker: Reference to the RTCWorkerClient instance
        crdt_state: Room state CRDT document
        is_admin: Whether this worker is currently the admin
        admin_peer_id: Current admin's peer_id
    """

    # Timeout for admin verification (seconds)
    ADMIN_VERIFY_TIMEOUT = 2.0

    def __init__(self, worker, crdt_state: RoomStateCRDT):
        """Initialize AdminController.

        Args:
            worker: RTCWorkerClient instance
            crdt_state: RoomStateCRDT instance for the room
        """
        self.worker = worker
        self.crdt_state = crdt_state
        self.is_admin = False
        self.admin_peer_id: Optional[str] = None
        self._election_lock = asyncio.Lock()

        # Admin verification tracking
        self._pending_verifications: Dict[str, asyncio.Event] = {}
        self._verification_counter = 0

    @staticmethod
    def elect_admin(workers: Dict[str, Dict[str, Any]]) -> str:
        """Deterministically elect admin from available workers.

        Election algorithm:
        1. Sort by GPU memory (descending) - most powerful worker wins
        2. Tiebreaker: peer_id (ascending) - lexicographic comparison

        This ensures all workers independently compute the same result
        without requiring network communication.

        Args:
            workers: Dictionary mapping peer_id to worker data

        Returns:
            peer_id of the elected admin worker

        Raises:
            ValueError: If no workers available or duplicate peer_ids detected
        """
        if not workers:
            raise ValueError("Cannot elect admin: no workers available")

        # Validate no duplicate peer_ids (should never happen, but safety check)
        # Check the peer_id values in worker data, not dict keys
        peer_ids = [worker.get("peer_id") for worker in workers.values()]
        if len(peer_ids) != len(set(peer_ids)):
            raise ValueError(
                "Cannot elect admin: duplicate peer_ids detected (this is a bug)"
            )

        # Extract candidates with (gpu_memory, peer_id) tuples
        candidates = []
        for peer_id, worker_data in workers.items():
            gpu_memory = (
                worker_data.get("metadata", {})
                .get("properties", {})
                .get("gpu_memory_mb", 0)
            )
            candidates.append((gpu_memory, peer_id))

        # Sort by GPU memory DESC, then peer_id ASC
        candidates.sort(key=lambda x: (-x[0], x[1]))

        elected_peer_id = candidates[0][1]
        elected_gpu = candidates[0][0]

        logger.info(
            f"Admin elected: {elected_peer_id} (GPU: {elected_gpu} MB) "
            f"from {len(candidates)} workers"
        )

        return elected_peer_id

    def on_admin_verify_ack(self, from_peer_id: str, request_id: str) -> None:
        """Handle admin verification acknowledgment from MeshCoordinator.

        Called when we receive an admin_verify_ack message confirming
        the elected admin is alive.

        Args:
            from_peer_id: peer_id of the admin who responded
            request_id: Request ID that was acknowledged
        """
        if request_id in self._pending_verifications:
            event = self._pending_verifications[request_id]
            event.set()
            logger.debug(
                f"Admin verification succeeded for {from_peer_id} (request_id: {request_id})"
            )
        else:
            logger.warning(
                f"Received unexpected admin_verify_ack (request_id: {request_id})"
            )

    async def verify_admin_alive(self, admin_peer_id: str) -> bool:
        """Verify that the elected admin is alive and reachable.

        Sends an admin_verify message and waits for acknowledgment.
        Used after election to confirm the elected admin didn't leave
        during the election process.

        Args:
            admin_peer_id: peer_id of the elected admin to verify

        Returns:
            True if admin responded within timeout, False otherwise
        """
        # Generate unique request ID
        self._verification_counter += 1
        request_id = f"{self.worker.peer_id}-{self._verification_counter}"

        # Create event to wait for response
        event = asyncio.Event()
        self._pending_verifications[request_id] = event

        try:
            # Send verification request via MeshCoordinator
            if not self.worker.mesh_coordinator:
                logger.warning("Cannot verify admin: no MeshCoordinator")
                return False

            sent = await self.worker.mesh_coordinator.send_admin_verify(
                admin_peer_id, request_id
            )
            if not sent:
                logger.warning(f"Failed to send admin_verify to {admin_peer_id}")
                return False

            # Wait for acknowledgment with timeout
            try:
                await asyncio.wait_for(event.wait(), timeout=self.ADMIN_VERIFY_TIMEOUT)
                logger.info(f"Admin {admin_peer_id} verified alive")
                return True
            except asyncio.TimeoutError:
                logger.warning(
                    f"Admin verification timed out for {admin_peer_id} "
                    f"after {self.ADMIN_VERIFY_TIMEOUT}s"
                )
                return False

        finally:
            # Clean up pending verification
            if request_id in self._pending_verifications:
                del self._pending_verifications[request_id]

    async def run_election(self, verify: bool = True) -> str:
        """Run admin election based on current CRDT state.

        This method:
        1. Checks if there's already an established admin in CRDT
        2. If admin exists and is still active, respect that admin
        3. Otherwise, runs election algorithm
        4. If elected admin is not self, verify they're alive (optional)
        5. If verification fails, remove from CRDT and re-elect
        6. Updates local state with election result
        7. Updates CRDT admin_peer_id if changed

        Args:
            verify: If True, verify elected admin is alive before accepting.
                    Set to False to skip verification (e.g., during initial setup).

        Returns:
            peer_id of the elected admin

        Raises:
            ValueError: If election fails (no workers, etc.)
        """
        async with self._election_lock:
            # Get all workers from CRDT
            all_workers = self.crdt_state.get_all_workers()

            if not all_workers:
                raise ValueError("Cannot run election: no workers in CRDT state")

            # Check if there's already an established admin in the CRDT
            existing_admin = self.crdt_state.get_admin_peer_id()
            if existing_admin and existing_admin in all_workers:
                # Respect the existing admin - don't re-elect
                logger.info(
                    f"Respecting existing admin: {existing_admin} "
                    f"(from {len(all_workers)} workers)"
                )
                elected_peer_id = existing_admin
            else:
                # No admin or admin departed - run full election
                elected_peer_id = self.elect_admin(all_workers)

            # Verify elected admin is alive (if not self and verification enabled)
            if verify and elected_peer_id != self.worker.peer_id:
                # Check if we have a connection to verify
                if elected_peer_id in self.worker.worker_connections:
                    # Release lock temporarily for async verification
                    self._election_lock.release()
                    try:
                        is_alive = await self.verify_admin_alive(elected_peer_id)
                    finally:
                        await self._election_lock.acquire()

                    if not is_alive:
                        # Admin didn't respond - remove from CRDT and re-elect
                        logger.warning(
                            f"Elected admin {elected_peer_id} not responding, "
                            f"removing from CRDT and re-electing"
                        )
                        self.crdt_state.remove_worker(elected_peer_id)

                        # Re-run election without the unresponsive admin
                        all_workers = self.crdt_state.get_all_workers()
                        if not all_workers:
                            raise ValueError(
                                "Cannot run election: no workers after removing unresponsive admin"
                            )
                        elected_peer_id = self.elect_admin(all_workers)

                        # If new election also picked someone else, verify again (recursive check)
                        # Note: This could be a loop, but we limit it by removing workers
                else:
                    logger.debug(
                        f"Skipping verification for {elected_peer_id}: no direct connection"
                    )

            # Update local state
            old_admin = self.admin_peer_id
            self.admin_peer_id = elected_peer_id
            self.is_admin = elected_peer_id == self.worker.peer_id

            # Update CRDT if admin changed
            if old_admin != elected_peer_id:
                self.crdt_state.set_admin(elected_peer_id)
                logger.info(
                    f"Admin changed: {old_admin} -> {elected_peer_id} "
                    f"(this worker: {self.worker.peer_id}, is_admin: {self.is_admin})"
                )
            else:
                logger.debug(f"Admin unchanged: {elected_peer_id}")

            return elected_peer_id

    async def handle_admin_departure(self, departed_peer_id: str) -> None:
        """Handle admin worker departure by triggering re-election.

        Called when the admin worker's ICE connection closes. This method:
        1. Removes departed worker from CRDT
        2. Runs new election
        3. All workers independently arrive at same new admin

        Args:
            departed_peer_id: peer_id of the departed admin worker
        """
        logger.warning(f"Admin departed: {departed_peer_id}, triggering re-election")

        async with self._election_lock:
            # Remove departed worker from CRDT
            self.crdt_state.remove_worker(departed_peer_id)

            # If we were tracking this as admin, clear it
            if self.admin_peer_id == departed_peer_id:
                self.admin_peer_id = None
                self.is_admin = False

        # Run new election
        try:
            await self.run_election()
            logger.info(
                f"Re-election complete, new admin: {self.admin_peer_id} "
                f"(this worker: {self.worker.peer_id}, is_admin: {self.is_admin})"
            )
        except ValueError as e:
            logger.error(f"Re-election failed: {e}")

    async def handle_worker_departure(self, departed_peer_id: str) -> None:
        """Handle non-admin worker departure.

        If the departed worker was the admin, triggers re-election.
        Otherwise, just removes them from CRDT state.

        Args:
            departed_peer_id: peer_id of the departed worker
        """
        logger.info(f"Worker departed: {departed_peer_id}")

        # Check if this was the admin
        if departed_peer_id == self.admin_peer_id:
            await self.handle_admin_departure(departed_peer_id)
        else:
            # Just remove from CRDT, no re-election needed
            self.crdt_state.remove_worker(departed_peer_id)
            logger.debug(f"Removed non-admin worker: {departed_peer_id}")

    async def handle_worker_joined(
        self, peer_id: str, metadata: Dict[str, Any]
    ) -> None:
        """Handle new worker joining the room.

        Adds worker to CRDT and potentially triggers re-election if the
        new worker has more GPU memory than current admin.

        Args:
            peer_id: peer_id of the new worker
            metadata: Worker metadata (capabilities, status, etc.)
        """
        logger.info(f"Worker joined: {peer_id}")

        # Add to CRDT state
        self.crdt_state.add_worker(peer_id, metadata, is_admin=False)

        # Check if re-election is needed
        # (new worker might have more GPU memory than current admin)
        current_admin = self.admin_peer_id
        await self.run_election()

        if self.admin_peer_id != current_admin:
            logger.info(
                f"Admin changed after worker join: {current_admin} -> {self.admin_peer_id}"
            )

    def get_admin_peer_id(self) -> Optional[str]:
        """Get the current admin worker's peer_id.

        Returns:
            Admin peer_id or None if no admin elected
        """
        return self.admin_peer_id

    def am_i_admin(self) -> bool:
        """Check if this worker is the admin.

        Returns:
            True if this worker is the admin, False otherwise
        """
        return self.is_admin

    async def on_state_update(self) -> None:
        """Handle CRDT state updates.

        Called when the CRDT state changes (e.g., after merging updates
        from other workers). Re-runs election to ensure admin is still valid.

        This is important for handling network partition recovery where
        multiple workers may have made independent state changes.
        """
        logger.debug("CRDT state updated, checking admin status")

        # Re-run election based on updated state
        try:
            await self.run_election()
        except ValueError as e:
            logger.warning(f"Election failed after state update: {e}")

    async def broadcast_state_update(self) -> None:
        """Broadcast CRDT state update to all connected workers.

        Only called if this worker is the admin. Sends CRDT update
        to all workers in the mesh so they can merge state changes.

        Implemented in Phase 4.
        """
        import base64

        if not self.is_admin:
            logger.warning("Cannot broadcast state: not admin")
            return

        # Get current CRDT state
        state = self.crdt_state.get_state()
        version = state.get("version", 0)

        # Serialize CRDT to binary and base64 encode for JSON transport
        crdt_binary = self.crdt_state.serialize()
        crdt_b64 = base64.b64encode(crdt_binary).decode("utf-8")

        logger.debug(f"Broadcasting state update (version {version}) to all workers")

        # Import here to avoid circular dependency
        from sleap_connect.worker.mesh_messages import (
            create_state_broadcast,
            serialize_message,
        )

        # Create broadcast message with base64-encoded CRDT binary
        message = create_state_broadcast(
            from_peer_id=self.worker.peer_id,
            crdt_snapshot={"_crdt_b64": crdt_b64},  # Wrap in dict with marker
            version=version,
        )

        # Send to all connected workers
        for peer_id in list(self.worker.worker_connections.keys()):
            if peer_id != self.worker.peer_id:  # Don't send to self
                try:
                    self.worker._send_mesh_message_to_peer(peer_id, message)
                    logger.debug(f"Sent state broadcast to {peer_id}")
                except Exception as e:
                    logger.error(f"Failed to broadcast to {peer_id}: {e}")

    async def handle_client_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Handle client query for worker discovery.

        Only called if this worker is the admin. Filters CRDT state
        based on client requirements and returns available workers.

        Implemented in Phase 4.

        Args:
            query: Client query parameters (tags, capabilities, etc.)
                - status: Filter by status (e.g., "available")
                - min_gpu_memory_mb: Minimum GPU memory required
                - tags: List of required tags
                - max_workers: Maximum number of workers to return

        Returns:
            Dictionary containing:
                - workers: List of matching worker metadata
                - total_count: Total workers before filtering
        """
        if not self.is_admin:
            logger.warning("Cannot handle client query: not admin")
            return {"error": "not_admin"}

        logger.info(f"Handling client query with filters: {query}")

        # Get all workers from CRDT
        state = self.crdt_state.get_state()
        all_workers = state.get("workers", {})

        # Extract filter criteria
        status_filter = query.get("status", "available")
        min_gpu_memory = query.get("min_gpu_memory_mb", 0)
        required_tags = set(query.get("tags", []))
        max_workers = query.get("max_workers", None)

        # Filter workers
        matched_workers = []

        for peer_id, worker_data in all_workers.items():
            metadata = worker_data.get("metadata", {})
            properties = metadata.get("properties", {})
            tags = set(metadata.get("tags", []))

            # Apply filters
            # 1. Status filter
            if status_filter and properties.get("status") != status_filter:
                continue

            # 2. GPU memory filter
            gpu_memory = properties.get("gpu_memory_mb", 0)
            if gpu_memory < min_gpu_memory:
                continue

            # 3. Tags filter (worker must have all required tags)
            if required_tags and not required_tags.issubset(tags):
                continue

            # Worker passed all filters
            matched_workers.append(worker_data)

        # Sort by GPU memory (descending) for consistent ordering
        matched_workers.sort(
            key=lambda w: w.get("metadata", {})
            .get("properties", {})
            .get("gpu_memory_mb", 0),
            reverse=True,
        )

        # Limit results if requested
        if max_workers and len(matched_workers) > max_workers:
            matched_workers = matched_workers[:max_workers]

        logger.info(
            f"Client query matched {len(matched_workers)}/{len(all_workers)} workers"
        )

        return {
            "workers": matched_workers,
            "total_count": len(all_workers),
            "matched_count": len(matched_workers),
        }
