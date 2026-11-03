import asyncio
import argparse
import websockets
import json
import logging
import os

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel
from functools import partial
from qtpy import QtCore
from sleap.gui.widgets.monitor import LossViewer
from sleap.gui.widgets.imagedir import QtImageDirectoryWidget
from sleap.gui.learning.configs import ConfigFileInfo
from sleap.nn.config.training_job import TrainingJobConfig
from websockets.client import ClientConnection

from sleap_connect.config import get_config

# Setup logging.
logging.basicConfig(level=logging.INFO)

# Global constants.
CHUNK_SIZE = 64 * 1024
MAX_RECONNECT_ATTEMPTS = 5
RETRY_DELAY = 5  # seconds

# Global variables.
received_files = {}
target_worker = None
reconnecting = False
reconnect_attempts = 0
output_dir = ""
win = None


async def clean_exit(pc: RTCPeerConnection, websocket: ClientConnection):
    """Cleans up the client connection and closes the peer connection and websocket.

    Args:
        pc: RTCPeerConnection object
        websocket: ClientConnection object
    Returns:
        None
    """
    logging.info("Closing WebRTC connection...")
    await pc.close()

    logging.info("Closing websocket connection...")
    await websocket.close()

    logging.info("Client shutdown complete. Exiting...")


async def reconnect(pc: RTCPeerConnection, websocket: ClientConnection):
    """Attempts to reconnect the client to the worker peer by creating a new offer with ICE restart flag.

    Args:
        pc: RTCPeerConnection object
        websocket: ClientConnection object
    Returns:
        bool: True if reconnection was successful, False otherwise
    """
    # Attempt to reconnect.
    global reconnect_attempts
    while reconnect_attempts < MAX_RECONNECT_ATTEMPTS:
        try:
            reconnect_attempts += 1
            logging.info(
                f"Reconnection attempt {reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS}..."
            )

            # Create new offer with ICE restart flag.
            logging.info("Creating new offer with manual ICE restart...")
            await pc.setLocalDescription(await pc.createOffer())

            # Send new offer to the worker via signaling.
            logging.info(f"Sending new offer to worker: {target_worker}")
            await websocket.send(
                json.dumps(
                    {
                        "type": pc.localDescription.type,
                        "target": target_worker,
                        "sdp": pc.localDescription.sdp,
                    }
                )
            )

            # Wait for connection to complete.
            for _ in range(30):
                await asyncio.sleep(1)
                if pc.iceConnectionState in ["connected", "completed"]:
                    logging.info("Reconnection successful.")

                    # Clear received files on reconnection.
                    logging.info("Clearing received files on reconnection...")
                    received_files.clear()

                    return True

            logging.warning("Reconnection timed out. Retrying...")

        except Exception as e:
            logging.error(f"Reconnection failed with error: {e}")

        await asyncio.sleep(RETRY_DELAY)

    # Maximum reconnection attempts reached.
    logging.error("Maximum reconnection attempts reached. Exiting...")
    await clean_exit(pc, websocket)
    return False


async def handle_connection(pc: RTCPeerConnection, websocket: ClientConnection):
    """Handles receiving SDP answer from Worker and ICE candidates from Worker.

    Args:
        pc: RTCPeerConnection object
        websocket: websocket connection object
    Returns:
                None
    Raises:
                JSONDecodeError: Invalid JSON received
                Exception: An error occurred while handling the message
    """
    # Handle incoming websocket messages.
    try:
        async for message in websocket:
            if type(message) == int:
                logging.info(f"Received int message: {message}")

            data = json.loads(message)

            # Receive answer SDP from worker and set it as this peer's remote description.
            if data.get("type") == "answer":
                logging.info(f"Received answer from worker: {data}")

                await pc.setRemoteDescription(
                    RTCSessionDescription(sdp=data.get("sdp"), type=data.get("type"))
                )

            # Handle "trickle ICE" for non-local ICE candidates.
            elif data.get("type") == "candidate":
                logging.debug("Received ICE candidate")
                candidate = data.get("candidate")
                await pc.addIceCandidate(candidate)

            # NOT initiator, received quit request from worker.
            elif data.get("type") == "quit":
                logging.info("Worker has quit. Closing connection...")
                await clean_exit(pc, websocket)
                break

            # Unhandled message types.
            else:
                logging.debug(f"Unhandled message: {data}")
                logging.debug("exiting...")
                break

    except json.JSONDecodeError:
        logging.debug("Invalid JSON received")

    except Exception as e:
        logging.debug(f"Error handling message: {e}")


async def run_client(
    peer_id: str,
    DNS: str,
    port_number: str,
    file_path: str = None,
    CLI: bool = True,
    output_dir: str = "",
    config_filename: TrainingJobConfig = None,
    cfg_head_name: str = None,
    loss_viewer: LossViewer = None,  # MUST INITIALIZE LOSS VIEWER FOR WINDOW
):
    """Sends initial SDP offer to worker peer and establishes both connection & datachannel to be used by both parties.

    Args:
                peer_id: Unique str identifier for client
        DNS: DNS address of the signaling server
        port_number: Port number of the signaling server
        file_path: Path to a file to be sent to worker peer (usually zip file)
        CLI: Boolean indicating if the client is running in CLI mode
        output_dir: Directory to save files received from worker peer
    Returns:
                None
    """
    # Initialize global variables
    global reconnect_attempts
    global win
    output_dir = output_dir

    # Initialize peer connection and data channel.
    reconnect_attempts = 0
    pc = RTCPeerConnection()
    channel = pc.createDataChannel("my-data-channel")
    logging.info("channel(%s) %s" % (channel.label, "created by local party."))

    # Initialize LossViewer RTC data channel event handlers .
    logging.info("Setting up RTC data channel for LossViewer...")
    loss_viewer.set_rtc_channel(channel)
    win = loss_viewer

    async def keep_ice_alive(channel: RTCDataChannel):
        """Sends periodic keep-alive messages to the worker peer to maintain the connection.

        Args:
            channel: RTCDataChannel object
        Returns:
            None
        """
        while True:
            await asyncio.sleep(15)
            if channel.readyState == "open":
                channel.send(b"KEEP_ALIVE")

    async def send_client_messages():
        """Takes input/file from client and sends it to worker peer via datachannel.

        Args:
            None
        Returns:
            None
        """
        message = input(
            "Enter message to send (type 'file' to prompt file or type 'quit' to exit): "
        )
        data = None

        if message.lower() == "quit":
            logging.info("Quitting...")
            await pc.close()
            return

        if channel.readyState != "open":
            logging.info(f"Data channel not open. Ready state is: {channel.readyState}")
            return

        if message.lower() == "file":
            logging.info("Prompting file...")
            file_path = input("Enter file path: (or type 'quit' to exit): ")
            if not file_path:
                logging.info("No file path entered.")
                return
            if file_path.lower() == "quit":
                logging.info("Quitting...")
                await pc.close()
                return
            if not os.path.exists(file_path):
                logging.info("File does not exist.")
                return
            else:
                logging.info(f"Sending {file_path} to worker...")
                file_name = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)

                # Send metadata first
                channel.send(f"FILE_META::{file_name}:{file_size}")

                # Send file in chunks (32 KB)
                with open(file_path, "rb") as file:
                    logging.info(f"File opened: {file_path}")
                    while chunk := file.read(CHUNK_SIZE):
                        while (
                            channel.bufferedAmount is not None
                            and channel.bufferedAmount > 16 * 1024 * 1024
                        ):  # Wait if buffer >16MB
                            await asyncio.sleep(0.1)

                        channel.send(chunk)

                channel.send("END_OF_FILE")
                logging.info(f"File sent to worker.")

                # Flag data to True to prevent reg msg from being sent
                data = True

        if not data:
            channel.send(message)
            logging.info(f"Message sent to worker.")

        else:
            return

    async def send_client_file():
        """Handles direct, one-way file transfer from client to be sent to worker peer.

        Args:
                        None
                Returns:
                        None
        """
        # Check channel state before sending file.
        if channel.readyState != "open":
            logging.info(f"Data channel not open. Ready state is: {channel.readyState}")
            return

        # Send file to worker.
        logging.info(f"Given file path {file_path}")
        if not file_path:
            logging.info("No file path entered.")
            return
        if not os.path.exists(file_path):
            logging.info("File does not exist.")
            return
        else:
            logging.info(f"Sending {file_path} to worker...")

            # Send output directory.
            channel.send(f"OUTPUT_DIR::{output_dir}")

            # Obtain metadata.
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)

            # Send metadata next.
            channel.send(f"FILE_META::{file_name}:{file_size}")

            # Send file in chunks (32 KB).
            with open(file_path, "rb") as file:
                logging.info(f"File opened: {file_path}")
                while chunk := file.read(CHUNK_SIZE):
                    while (
                        channel.bufferedAmount is not None
                        and channel.bufferedAmount > 16 * 1024 * 1024
                    ):  # Wait if buffer >16MB
                        await asyncio.sleep(0.1)

                    channel.send(chunk)

            channel.send("END_OF_FILE")
            logging.info(f"File sent to worker.")

        return

    @channel.on("open")
    async def on_channel_open():
        """Event handler function for when the datachannel is open.

        Args:
                        None
        Returns:
                        None
        """
        # Initiate keep-alive task.
        asyncio.create_task(keep_ice_alive(channel))
        logging.info(f"{channel.label} is open")

        # Setup monitor window for progress reports.
        # zmq_ports = dict()
        # zmq_ports["controller_port"] = 9000
        # zmq_ports["publish_port"] = 9001

        # Prompt for messages or file upload.
        if CLI:
            await send_client_messages()
        else:
            await send_client_file()

    @channel.on("message")
    async def on_message(message):
        """Event handler function for when a message is received on the datachannel from Worker.

        Args:
            message: The received message, either as a string or bytes.

        Returns:
            None
        """
        # Log the received message in verbose mode only.
        logging.debug(f"Client received: {message}")
        global received_files
        global output_dir
        global win

        # Handle string and bytes messages differently.
        if isinstance(message, str):
            if message == b"KEEP_ALIVE":
                logging.debug("Keep alive message received.")
                return

            if message == "END_OF_FILE":
                # File transfer complete, save to disk.
                file_name, file_data = list(received_files.items())[0]

                try:
                    os.makedirs(output_dir, exist_ok=True)
                    file_path = os.path.join(output_dir, file_name)

                    with open(file_path, "wb") as file:
                        file.write(file_data)
                    logging.info(f"File saved as: {file_path}")
                except PermissionError:
                    logging.error(f"Permission denied when writing to: {output_dir}")
                except Exception as e:
                    logging.error(f"Failed to save file: {e}")

                received_files.clear()

                # Update monitor window with file transfer and training completion.
                win.close()

                if CLI:
                    # Prompt for next message
                    logging.info("File transfer complete. Enter next message:")
                    await send_client_messages()
                else:
                    await clean_exit(pc, websocket)

            elif "PROGRESS_REPORT::" in message:
                # Progress report received from worker.
                logging.info(message)
                _, progress = message.split("PROGRESS_REPORT::", 1)

                # Update LossViewer window with received progress report.
                if win:
                    QtCore.QTimer.singleShot(
                        0, partial(win._check_messages, rtc_msg=progress)
                    )
                    # win._check_messages(
                    #     # Progress should be result from jsonpickle.decode(msg_str)
                    #     rtc_msg=progress
                    # )
                else:
                    logging.info(f"No monitor window available! win is {win}")

                # print("Resetting monitor window.")
                # plateau_patience = config_info.optimization.early_stopping.plateau_patience
                # plateau_min_delta = config_info.optimization.early_stopping.plateau_min_delta
                # win.reset(
                #     what=str(model_type),
                #     plateau_patience=plateau_patience,
                #     plateau_min_delta=plateau_min_delta,
                # )
                # win.setWindowTitle(f"Training Model - {str(model_type)}")
                # win.set_message(f"Preparing to run training...")
                # if save_viz:
                #     viz_window = QtImageDirectoryWidget.make_training_vizualizer(
                #         job.outputs.run_path
                #     )
                #     viz_window.move(win.x() + win.width() + 20, win.y())
                #     win.on_epoch.connect(viz_window.poll)

            elif "FILE_META::" in message:
                # Metadata received (file name & size)
                _, meta = message.split("FILE_META::", 1)
                file_name, file_size, output_dir = meta.split(":")

                if file_name not in received_files:
                    received_files[file_name] = bytearray()  # Initialize as bytearray
                logging.info(
                    f"File name received: {file_name}, of size {file_size}, saving to {output_dir}"
                )

            elif "ZMQ_CTRL::" in message:
                # ZMQ control message received.
                _, zmq_ctrl = message.split("ZMQ_CTRL::", 1)

                if zmq_ctrl == "STOP":
                    win.reset()

            else:
                logging.debug(f"Worker sent: {message}")

        elif isinstance(message, bytes):
            if message == b"KEEP_ALIVE":
                logging.debug("Keep alive message received.")
                return

            elif b"PROGRESS_REPORT::" in message:
                # Progress report received from worker.
                logging.info(message.decode())
                _, progress = message.decode().split("PROGRESS_REPORT::", 1)

                # Update LossViewer window with received progress report.
                if win:
                    win._check_messages(
                        # Progress should be result from jsonpickle.decode(msg_str)
                        rtc_msg=progress
                    )
                else:
                    logging.info(f"No monitor window available! win is {win}")

            file_name = list(received_files.keys())[0]
            if file_name not in received_files:
                received_files[file_name] = bytearray()
            received_files[file_name].extend(message)

    # @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        """Event handler function for when the ICE connection state changes.

        Args:
            None
        Returns:
            None
        """
        # Log the current ICE connection state.
        global reconnecting
        logging.debug(f"ICE connection state is now {pc.iceConnectionState}")

        # Check the ICE connection state and handle reconnection logic.
        if pc.iceConnectionState in ["connected", "completed"]:
            reconnect_attempts = 0
            logging.debug("ICE connection established.")
            logging.info(f"reconnect attempts reset to {reconnect_attempts}")

        elif (
            pc.iceConnectionState in ["failed", "disconnected", "closed"]
            and not reconnecting
        ):
            logging.warning(
                f"ICE connection {pc.iceConnectionState}. Attempting reconnect..."
            )
            reconnecting = True

            if target_worker is None:
                logging.info(
                    f"No target worker available for reconnection. target_worker is {target_worker}."
                )
                await clean_exit(pc, websocket)
                return

            reconnection_success = await reconnect(pc, websocket)
            reconnecting = False
            if not reconnection_success:
                logging.info("Reconnection failed. Closing connection...")
                await clean_exit(pc, websocket)
                return

    # Register the event handler explicitly
    pc.on("iceconnectionstatechange", on_iceconnectionstatechange)

    # Initiate the WebSocket connection to the signaling server.
    async with websockets.connect(DNS) as websocket:

        # Register the client with the signaling server.
        await websocket.send(json.dumps({"type": "register", "peer_id": peer_id}))
        logging.info(f"{peer_id} sent to signaling server for registration!")

        # Query for available workers.
        await websocket.send(json.dumps({"type": "query"}))
        response = await websocket.recv()
        available_workers = json.loads(response)["peers"]
        logging.info(f"Available workers: {available_workers}")

        # Select a worker to connect to.
        target_worker = available_workers[0] if available_workers else None
        logging.info(f"Selected worker: {target_worker}")

        if not target_worker:
            logging.info("No workers available")
            return

        # Create and send SDP offer to worker peer.
        await pc.setLocalDescription(await pc.createOffer())
        await websocket.send(
            json.dumps(
                {
                    "type": pc.localDescription.type,
                    "target": target_worker,
                    "sdp": pc.localDescription.sdp,
                }
            )
        )
        logging.info("Offer sent to worker")

        # Handle incoming messages from server (e.g. answer from worker).
        await handle_connection(pc, websocket)

    # Exit.
    await pc.close()
    await websocket.close()


def entrypoint():
    """Main CLI function to run the client.

    Args:
      None
    Returns:
      None
    """
    parser = argparse.ArgumentParser(description="SLEAP webRTC Client")

    # Get config for default value
    config = get_config()

    parser.add_argument(
        "--server",
        type=str,
        default=config.signaling_websocket,
        help="WebSocket server DNS/address, ex. 'ws://ec2-54-158-36-90.compute-1.amazonaws.com'",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="WebSocket server port number, ex. '8080'",
    )
    parser.add_argument(
        "--peer_id",
        type=str,
        default="client1",
        help="Unique identifier for the client",
    )

    args = parser.parse_args()

    DNS = args.server
    port_number = args.port
    peer_id = args.peer_id

    try:
        asyncio.run(run_client(peer_id, DNS, port_number, file_path=None, CLI=True))
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt: Exiting...")
    finally:
        logging.info("exited")


if __name__ == "__main__":
    entrypoint()
