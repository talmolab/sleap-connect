"""Mesh protocol message definitions for worker coordination.

This module defines the message types and serialization/deserialization
functions for worker-to-worker communication over WebRTC data channels.

Message types:
- status_update: Worker sends status change to admin
- state_broadcast: Admin broadcasts CRDT snapshot to workers
- heartbeat: Periodic health check between peers
- heartbeat_response: Response to heartbeat (optional)
- query_workers: Client requests worker list from admin
- worker_list: Admin responds with available workers
- peer_joined: Notification that new peer joined room
- peer_left: Notification that peer left room
"""

import json
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Mesh protocol message types."""

    # Worker-to-Admin messages
    STATUS_UPDATE = "status_update"

    # Admin-to-Workers messages
    STATE_BROADCAST = "state_broadcast"

    # Peer-to-Peer messages
    HEARTBEAT = "heartbeat"
    HEARTBEAT_RESPONSE = "heartbeat_response"

    # Client-to-Admin messages
    QUERY_WORKERS = "query_workers"

    # Admin-to-Client messages
    WORKER_LIST = "worker_list"

    # Room lifecycle messages
    PEER_JOINED = "peer_joined"
    PEER_LEFT = "peer_left"


@dataclass
class StatusUpdateMessage:
    """Worker sends status change to admin.

    Attributes:
        type: Message type ("status_update")
        from_peer_id: Sender's peer_id
        timestamp: Unix timestamp of update
        status: Worker status (available, busy, reserved, maintenance)
        current_job: Current job info (optional)
    """

    type: str = MessageType.STATUS_UPDATE.value
    from_peer_id: Optional[str] = None
    timestamp: Optional[float] = None
    status: Optional[str] = None
    current_job: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Set default timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class StateBroadcastMessage:
    """Admin broadcasts CRDT snapshot to all workers.

    Attributes:
        type: Message type ("state_broadcast")
        from_peer_id: Admin's peer_id
        timestamp: Unix timestamp of broadcast
        crdt_snapshot: Serialized CRDT document
        version: CRDT version/counter
    """

    type: str = MessageType.STATE_BROADCAST.value
    from_peer_id: Optional[str] = None
    timestamp: Optional[float] = None
    crdt_snapshot: Optional[Dict[str, Any]] = None
    version: Optional[int] = None

    def __post_init__(self):
        """Set default timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class HeartbeatMessage:
    """Periodic health check between peers.

    Attributes:
        type: Message type ("heartbeat")
        from_peer_id: Sender's peer_id
        timestamp: Unix timestamp of heartbeat
        sequence: Sequence number for tracking missed beats
    """

    type: str = MessageType.HEARTBEAT.value
    from_peer_id: Optional[str] = None
    timestamp: Optional[float] = None
    sequence: Optional[int] = None

    def __post_init__(self):
        """Set default timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class HeartbeatResponseMessage:
    """Response to heartbeat (optional, for RTT measurement).

    Attributes:
        type: Message type ("heartbeat_response")
        from_peer_id: Responder's peer_id
        to_peer_id: Original heartbeat sender
        timestamp: Unix timestamp of response
        original_timestamp: Timestamp from original heartbeat
    """

    type: str = MessageType.HEARTBEAT_RESPONSE.value
    from_peer_id: Optional[str] = None
    to_peer_id: Optional[str] = None
    timestamp: Optional[float] = None
    original_timestamp: Optional[float] = None

    def __post_init__(self):
        """Set default timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class QueryWorkersMessage:
    """Client requests worker list from admin.

    Attributes:
        type: Message type ("query_workers")
        from_peer_id: Client's peer_id
        timestamp: Unix timestamp of query
        filters: Filter criteria for workers
    """

    type: str = MessageType.QUERY_WORKERS.value
    from_peer_id: Optional[str] = None
    timestamp: Optional[float] = None
    filters: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Set default timestamp and filters if not provided."""
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.filters is None:
            self.filters = {}


@dataclass
class WorkerListMessage:
    """Admin responds with available workers.

    Attributes:
        type: Message type ("worker_list")
        from_peer_id: Admin's peer_id
        timestamp: Unix timestamp of response
        workers: List of worker metadata
        total_count: Total number of workers (before filtering)
    """

    type: str = MessageType.WORKER_LIST.value
    from_peer_id: Optional[str] = None
    timestamp: Optional[float] = None
    workers: Optional[List[Dict[str, Any]]] = None
    total_count: Optional[int] = None

    def __post_init__(self):
        """Set default timestamp and workers if not provided."""
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.workers is None:
            self.workers = []


@dataclass
class PeerJoinedMessage:
    """Notification that new peer joined room.

    Attributes:
        type: Message type ("peer_joined")
        from_peer_id: Admin's peer_id (sender)
        timestamp: Unix timestamp
        peer_id: New peer's peer_id
        metadata: New peer's metadata
    """

    type: str = MessageType.PEER_JOINED.value
    from_peer_id: Optional[str] = None
    timestamp: Optional[float] = None
    peer_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Set default timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class PeerLeftMessage:
    """Notification that peer left room.

    Attributes:
        type: Message type ("peer_left")
        from_peer_id: Admin's peer_id (sender)
        timestamp: Unix timestamp
        peer_id: Departed peer's peer_id
    """

    type: str = MessageType.PEER_LEFT.value
    from_peer_id: Optional[str] = None
    timestamp: Optional[float] = None
    peer_id: Optional[str] = None

    def __post_init__(self):
        """Set default timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = time.time()


# Message type mapping for deserialization
MESSAGE_TYPE_MAP = {
    MessageType.STATUS_UPDATE.value: StatusUpdateMessage,
    MessageType.STATE_BROADCAST.value: StateBroadcastMessage,
    MessageType.HEARTBEAT.value: HeartbeatMessage,
    MessageType.HEARTBEAT_RESPONSE.value: HeartbeatResponseMessage,
    MessageType.QUERY_WORKERS.value: QueryWorkersMessage,
    MessageType.WORKER_LIST.value: WorkerListMessage,
    MessageType.PEER_JOINED.value: PeerJoinedMessage,
    MessageType.PEER_LEFT.value: PeerLeftMessage,
}


def serialize_message(message: Any) -> str:
    """Serialize message object to JSON string.

    Args:
        message: Message dataclass instance

    Returns:
        JSON string representation

    Raises:
        TypeError: If message is not a valid message type
    """
    if not hasattr(message, "type"):
        raise TypeError(f"Message must have 'type' attribute: {type(message)}")

    try:
        return json.dumps(asdict(message))
    except Exception as e:
        logger.error(f"Failed to serialize message: {e}")
        raise


def deserialize_message(json_str: str) -> Any:
    """Deserialize JSON string to message object.

    Args:
        json_str: JSON string representation

    Returns:
        Message dataclass instance

    Raises:
        ValueError: If JSON is invalid or message type unknown
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        raise ValueError(f"Invalid JSON: {e}")

    msg_type = data.get("type")
    if not msg_type:
        raise ValueError("Message missing 'type' field")

    message_class = MESSAGE_TYPE_MAP.get(msg_type)
    if not message_class:
        raise ValueError(f"Unknown message type: {msg_type}")

    try:
        return message_class(**data)
    except Exception as e:
        logger.error(f"Failed to deserialize message: {e}")
        raise ValueError(f"Invalid message format: {e}")


def validate_message(message: Any) -> bool:
    """Validate message has required fields.

    Args:
        message: Message dataclass instance

    Returns:
        True if valid, False otherwise
    """
    if not hasattr(message, "type"):
        logger.warning("Message missing 'type' field")
        return False

    if not hasattr(message, "from_peer_id"):
        logger.warning("Message missing 'from_peer_id' field")
        return False

    if not hasattr(message, "timestamp"):
        logger.warning("Message missing 'timestamp' field")
        return False

    return True


def create_status_update(
    from_peer_id: str, status: str, current_job: Optional[Dict[str, Any]] = None
) -> StatusUpdateMessage:
    """Create status update message.

    Args:
        from_peer_id: Sender's peer_id
        status: Worker status
        current_job: Current job info (optional)

    Returns:
        StatusUpdateMessage instance
    """
    return StatusUpdateMessage(
        from_peer_id=from_peer_id, status=status, current_job=current_job
    )


def create_state_broadcast(
    from_peer_id: str, crdt_snapshot: Dict[str, Any], version: int
) -> StateBroadcastMessage:
    """Create state broadcast message.

    Args:
        from_peer_id: Admin's peer_id
        crdt_snapshot: CRDT document snapshot
        version: CRDT version

    Returns:
        StateBroadcastMessage instance
    """
    return StateBroadcastMessage(
        from_peer_id=from_peer_id, crdt_snapshot=crdt_snapshot, version=version
    )


def create_heartbeat(from_peer_id: str, sequence: int) -> HeartbeatMessage:
    """Create heartbeat message.

    Args:
        from_peer_id: Sender's peer_id
        sequence: Sequence number

    Returns:
        HeartbeatMessage instance
    """
    return HeartbeatMessage(from_peer_id=from_peer_id, sequence=sequence)


def create_query_workers(
    from_peer_id: str, filters: Optional[Dict[str, Any]] = None
) -> QueryWorkersMessage:
    """Create query workers message.

    Args:
        from_peer_id: Client's peer_id
        filters: Filter criteria

    Returns:
        QueryWorkersMessage instance
    """
    return QueryWorkersMessage(from_peer_id=from_peer_id, filters=filters)


def create_worker_list(
    from_peer_id: str, workers: List[Dict[str, Any]], total_count: int
) -> WorkerListMessage:
    """Create worker list message.

    Args:
        from_peer_id: Admin's peer_id
        workers: List of worker metadata
        total_count: Total worker count

    Returns:
        WorkerListMessage instance
    """
    return WorkerListMessage(
        from_peer_id=from_peer_id, workers=workers, total_count=total_count
    )
