"""CRDT-based room state synchronization using pycrdt.

This module provides a wrapper around pycrdt (Python bindings for Yrs/Yjs) for
conflict-free replicated state management across workers in a room. The state
document tracks all workers, their metadata, status, and the current admin worker.

The CRDT structure mirrors the existing worker metadata format for backwards
compatibility and uses pycrdt's Yrs-backed CRDTs for automatic conflict resolution.
"""

import json
import time
from typing import Any, Dict, Optional

from pycrdt import Doc, Map


class RoomStateCRDT:
    """Conflict-free replicated data type for room-level worker registry.

    This class wraps pycrdt to provide a clean API for managing room state
    across distributed workers. The CRDT document tracks:
    - All workers in the room with their metadata
    - Current admin worker peer_id
    - Room ID and version

    The state automatically resolves conflicts when multiple workers update
    simultaneously, ensuring eventual consistency across the mesh network.

    Attributes:
        room_id: Unique identifier for the room
        doc: pycrdt Document containing the CRDT state
    """

    def __init__(self, room_id: str, doc: Doc):
        """Initialize CRDT wrapper.

        Args:
            room_id: Unique identifier for the room
            doc: pycrdt Document instance
        """
        self.room_id = room_id
        self.doc = doc

    @classmethod
    def create(
        cls, room_id: str, creator_peer_id: Optional[str] = None
    ) -> "RoomStateCRDT":
        """Create a new CRDT document for a room.

        Args:
            room_id: Unique identifier for the room
            creator_peer_id: Optional peer_id of the creating worker (becomes admin)

        Returns:
            New RoomStateCRDT instance with initialized document
        """
        doc = Doc()

        # Create root map to hold all state
        with doc.transaction():
            root = doc.get("state", type=Map)
            root["room_id"] = room_id
            root["workers"] = Map()
            root["admin_peer_id"] = (
                creator_peer_id if creator_peer_id is not None else ""
            )
            root["version"] = 0

        return cls(room_id, doc)

    @classmethod
    def deserialize(cls, serialized: bytes) -> "RoomStateCRDT":
        """Deserialize a CRDT document from bytes.

        Args:
            serialized: Serialized pycrdt document (binary update)

        Returns:
            RoomStateCRDT instance with loaded document
        """
        doc = Doc()
        doc.apply_update(serialized)
        root = doc.get("state", type=Map)
        room_id = root.get("room_id", "unknown") if root else "unknown"
        return cls(room_id, doc)

    def serialize(self) -> bytes:
        """Serialize the CRDT document to bytes.

        Returns:
            Serialized pycrdt document as binary update
        """
        return self.doc.get_update()

    def merge(self, other: "RoomStateCRDT") -> None:
        """Merge another CRDT document into this one.

        pycrdt automatically resolves conflicts using Yrs/Yjs CRDT rules
        (last-write-wins for scalars, automatic merging for maps).

        Args:
            other: Another RoomStateCRDT instance to merge from
        """
        update = other.doc.get_update()
        self.doc.apply_update(update)

    def apply_update(self, update_bytes: bytes) -> None:
        """Apply a binary CRDT update to this document.

        This is used when receiving serialized updates from other workers.
        pycrdt automatically handles conflict resolution.

        Args:
            update_bytes: Binary CRDT update (from serialize())
        """
        self.doc.apply_update(update_bytes)

    def _get_root(self) -> Optional[Map]:
        """Get the root state map."""
        return self.doc.get("state", type=Map)

    def get_state(self) -> Dict[str, Any]:
        """Get the current state as a dictionary.

        Returns:
            Dictionary containing room_id, workers, admin_peer_id, and version
        """
        root = self._get_root()
        if root is None:
            return {"room_id": "", "workers": {}, "admin_peer_id": "", "version": 0}

        workers_map = root.get("workers")
        workers_dict = {}

        # Convert pycrdt Map to Python dict
        if workers_map is not None and isinstance(workers_map, Map):
            for peer_id in workers_map:
                worker_data = workers_map[peer_id]
                if isinstance(worker_data, str):
                    # Stored as JSON string
                    workers_dict[peer_id] = json.loads(worker_data)
                else:
                    workers_dict[peer_id] = worker_data

        admin_id = root.get("admin_peer_id", "")
        return {
            "room_id": root.get("room_id", ""),
            "workers": workers_dict,
            "admin_peer_id": admin_id if admin_id else None,
            "version": root.get("version", 0),
        }

    def add_worker(
        self, peer_id: str, metadata: Dict[str, Any], is_admin: bool = False
    ) -> None:
        """Add a new worker to the room state.

        Args:
            peer_id: Unique identifier for the worker
            metadata: Worker metadata (tags and properties matching current format)
            is_admin: Whether this worker is the admin
        """
        with self.doc.transaction():
            root = self._get_root()
            if root is None:
                return

            workers = root.get("workers")
            if not isinstance(workers, Map):
                root["workers"] = Map()
                workers = root.get("workers")

            # Add is_admin flag to properties
            if "properties" not in metadata:
                metadata["properties"] = {}
            metadata["properties"]["is_admin"] = is_admin
            metadata["properties"]["last_heartbeat"] = int(time.time() * 1000)

            # Create worker entry
            worker_data = {"peer_id": peer_id, "role": "worker", "metadata": metadata}

            # Store as JSON string
            workers[peer_id] = json.dumps(worker_data)

            # Update admin if this is the admin worker
            if is_admin:
                root["admin_peer_id"] = peer_id

            # Increment version
            root["version"] = root.get("version", 0) + 1

    def update_worker_status(
        self, peer_id: str, status: str, current_job: Optional[str] = None
    ) -> None:
        """Update a worker's status.

        Args:
            peer_id: Worker to update
            status: New status (available, busy, reserved, maintenance)
            current_job: Optional current job identifier
        """
        with self.doc.transaction():
            root = self._get_root()
            if root is None:
                return

            workers = root.get("workers")
            if not isinstance(workers, Map) or peer_id not in workers:
                return

            # Load existing worker data
            worker_json = workers[peer_id]
            worker = (
                json.loads(worker_json) if isinstance(worker_json, str) else worker_json
            )

            # Update status fields
            if "metadata" not in worker:
                worker["metadata"] = {}
            if "properties" not in worker["metadata"]:
                worker["metadata"]["properties"] = {}

            properties = worker["metadata"]["properties"]
            properties["status"] = status
            properties["current_job"] = current_job
            properties["last_heartbeat"] = int(time.time() * 1000)

            # Store updated worker
            workers[peer_id] = json.dumps(worker)
            root["version"] = root.get("version", 0) + 1

    def update_worker_heartbeat(self, peer_id: str) -> None:
        """Update a worker's last heartbeat timestamp.

        Args:
            peer_id: Worker to update
        """
        with self.doc.transaction():
            root = self._get_root()
            if root is None:
                return

            workers = root.get("workers")
            if not isinstance(workers, Map) or peer_id not in workers:
                return

            # Load existing worker data
            worker_json = workers[peer_id]
            worker = (
                json.loads(worker_json) if isinstance(worker_json, str) else worker_json
            )

            # Update heartbeat
            if "metadata" not in worker:
                worker["metadata"] = {}
            if "properties" not in worker["metadata"]:
                worker["metadata"]["properties"] = {}

            worker["metadata"]["properties"]["last_heartbeat"] = int(time.time() * 1000)

            # Store updated worker
            workers[peer_id] = json.dumps(worker)
            root["version"] = root.get("version", 0) + 1

    def remove_worker(self, peer_id: str) -> None:
        """Remove a worker from the room state.

        Args:
            peer_id: Worker to remove
        """
        with self.doc.transaction():
            root = self._get_root()
            if root is None:
                return

            workers = root.get("workers")
            if isinstance(workers, Map) and peer_id in workers:
                del workers[peer_id]
                root["version"] = root.get("version", 0) + 1

    def set_admin(self, peer_id: str) -> None:
        """Set the admin worker.

        Args:
            peer_id: Worker to designate as admin
        """
        with self.doc.transaction():
            root = self._get_root()
            if root is None:
                return

            root["admin_peer_id"] = peer_id

            # Update is_admin flag for all workers
            workers = root.get("workers")
            if isinstance(workers, Map):
                for worker_id in list(workers.keys()):
                    worker_json = workers[worker_id]
                    worker = (
                        json.loads(worker_json)
                        if isinstance(worker_json, str)
                        else worker_json
                    )

                    if "metadata" not in worker:
                        worker["metadata"] = {}
                    if "properties" not in worker["metadata"]:
                        worker["metadata"]["properties"] = {}

                    worker["metadata"]["properties"]["is_admin"] = worker_id == peer_id
                    workers[worker_id] = json.dumps(worker)

            root["version"] = root.get("version", 0) + 1

    def get_admin_peer_id(self) -> Optional[str]:
        """Get the current admin worker peer_id.

        Returns:
            Admin peer_id or None if no admin set
        """
        root = self._get_root()
        if root is None:
            return None
        admin_id = root.get("admin_peer_id", "")
        return admin_id if admin_id else None

    def get_worker(self, peer_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific worker's data.

        Args:
            peer_id: Worker to retrieve

        Returns:
            Worker data dictionary or None if not found
        """
        root = self._get_root()
        if root is None:
            return None

        workers = root.get("workers")
        if not isinstance(workers, Map) or peer_id not in workers:
            return None

        worker_json = workers[peer_id]
        return json.loads(worker_json) if isinstance(worker_json, str) else worker_json

    def get_all_workers(self) -> Dict[str, Dict[str, Any]]:
        """Get all workers in the room.

        Returns:
            Dictionary mapping peer_id to worker data
        """
        root = self._get_root()
        if root is None:
            return {}

        workers = root.get("workers")
        if not isinstance(workers, Map):
            return {}

        result = {}
        for peer_id in workers:
            worker_json = workers[peer_id]
            result[peer_id] = (
                json.loads(worker_json) if isinstance(worker_json, str) else worker_json
            )

        return result

    def get_available_workers(self) -> Dict[str, Dict[str, Any]]:
        """Get all workers with status 'available'.

        Returns:
            Dictionary mapping peer_id to worker data for available workers
        """
        all_workers = self.get_all_workers()
        return {
            peer_id: worker
            for peer_id, worker in all_workers.items()
            if worker.get("metadata", {}).get("properties", {}).get("status")
            == "available"
        }

    def get_version(self) -> int:
        """Get the current document version.

        Returns:
            Document version number
        """
        root = self._get_root()
        if root is None:
            return 0
        return root.get("version", 0)
