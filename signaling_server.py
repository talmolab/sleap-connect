"""Simple WebSocket signaling server for WebRTC peer discovery and connection.

This is a minimal signaling server for testing frame_client.py and frame_worker.py.
It handles peer registration, discovery queries, and message forwarding.

Usage:
    python signaling_server.py [--host HOST] [--port PORT]

Examples:
    python signaling_server.py
    python signaling_server.py --port 8080
    python signaling_server.py --host 0.0.0.0 --port 9000
"""

import argparse
import asyncio
import json
import logging
from typing import Dict

import websockets
from websockets.server import WebSocketServerProtocol

logging.basicConfig(level=logging.INFO)

# Connected peers: peer_id -> websocket
peers: Dict[str, WebSocketServerProtocol] = {}


async def handler(websocket: WebSocketServerProtocol):
    """Handle a WebSocket connection."""
    peer_id = None

    try:
        async for message in websocket:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "register":
                # Register peer
                peer_id = data.get("peer_id")
                peers[peer_id] = websocket
                logging.info(f"Registered peer: {peer_id} (total: {len(peers)})")

            elif msg_type == "query":
                # Return list of other registered peers
                other_peers = [pid for pid in peers.keys() if pid != peer_id]
                await websocket.send(json.dumps({"peers": other_peers}))
                logging.info(f"Query from {peer_id}: returned {len(other_peers)} peers")

            elif msg_type in ("offer", "answer", "candidate"):
                # Forward message to target peer
                target = data.get("target")
                if target in peers:
                    # Add sender info
                    data["sender"] = peer_id
                    await peers[target].send(json.dumps(data))
                    logging.info(f"Forwarded {msg_type} from {peer_id} to {target}")
                else:
                    logging.warning(f"Target peer not found: {target}")

            elif msg_type == "quit":
                # Forward quit message to target
                target = data.get("target")
                if target in peers:
                    await peers[target].send(json.dumps({"type": "quit", "sender": peer_id}))
                break

    except websockets.exceptions.ConnectionClosed:
        logging.info(f"Connection closed: {peer_id}")
    finally:
        # Cleanup on disconnect
        if peer_id and peer_id in peers:
            del peers[peer_id]
            logging.info(f"Removed peer: {peer_id} (remaining: {len(peers)})")


async def main(host: str, port: int):
    """Start the signaling server."""
    async with websockets.serve(handler, host, port):
        logging.info(f"Signaling server running on ws://{host}:{port}")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple WebRTC signaling server")
    parser.add_argument("--host", default="localhost", help="Host to bind to (default: localhost)")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")

    args = parser.parse_args()

    try:
        asyncio.run(main(args.host, args.port))
    except KeyboardInterrupt:
        logging.info("Server stopped")
