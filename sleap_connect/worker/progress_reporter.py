"""Progress reporting via ZMQ for SLEAP training jobs.

This module handles ZMQ-based progress reporting for training jobs,
including control socket management and progress listener forwarding.
"""

import asyncio
import logging
from typing import Optional

import zmq
from aiortc import RTCDataChannel


class ProgressReporter:
    """Manages ZMQ progress reporting for training jobs.

    This class handles bidirectional ZMQ communication: a PUB socket for
    sending control commands to the trainer, and a SUB socket for receiving
    progress updates and forwarding them to the client via RTC.

    Attributes:
        ctrl_socket: ZMQ PUB socket for sending control commands.
        progress_socket: ZMQ SUB socket for receiving progress updates.
        control_address: Address of ZMQ control socket.
        progress_address: Address of ZMQ progress socket.
        listener_task: Background task for progress listener.
    """

    def __init__(
        self,
        control_address: str = "tcp://127.0.0.1:9000",
        progress_address: str = "tcp://127.0.0.1:9001",
    ):
        """Initialize progress reporter.

        Args:
            control_address: Address for ZMQ control PUB socket (default tcp://127.0.0.1:9000).
            progress_address: Address for ZMQ progress SUB socket (default tcp://127.0.0.1:9001).
        """
        self.control_address = control_address
        self.progress_address = progress_address

        # ZMQ sockets
        self.ctrl_socket: Optional[zmq.Socket] = None
        self.progress_socket: Optional[zmq.Socket] = None
        self.context: Optional[zmq.Context] = None

        # Progress listener task
        self.listener_task: Optional[asyncio.Task] = None
        self.running = False

    def start_control_socket(self):
        """Start ZMQ control PUB socket for sending commands to trainer.

        This socket allows the worker to send control commands (e.g., stop, pause)
        to the training process.
        """
        logging.info("Starting ZMQ control socket...")
        self.context = zmq.Context()
        self.ctrl_socket = self.context.socket(zmq.PUB)

        logging.info(f"Binding ZMQ control socket to: {self.control_address}")
        self.ctrl_socket.bind(self.control_address)
        logging.info("ZMQ control socket initialized.")

    def send_control_message(self, message: str):
        """Send control message to trainer via ZMQ.

        Args:
            message: Control message string to send.
        """
        if self.ctrl_socket is not None:
            logging.info(
                f"[ZMQ] Sending control message to {self.control_address!r}: {message!r}"
            )
            self.ctrl_socket.send_string(message)
            logging.info("[ZMQ] Control message sent successfully")
        else:
            logging.error(
                f"[ZMQ] Control socket is NOT initialized (ctrl_socket=None) — "
                f"cannot forward message: {message!r}"
            )

    async def start_progress_listener(self, channel: RTCDataChannel):
        """Start ZMQ progress listener that forwards updates to RTC data channel.

        This method creates a background task that listens for progress updates
        from the training process and forwards them to the client via RTC.

        Args:
            channel: RTC data channel to send progress updates.
        """
        logging.info("Starting ZMQ progress listener...")

        # Create context if not already created
        if self.context is None:
            self.context = zmq.Context()

        self.progress_socket = self.context.socket(zmq.SUB)
        logging.info(f"Binding ZMQ progress socket to: {self.progress_address}")
        self.progress_socket.bind(self.progress_address)
        self.progress_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        self.running = True
        loop = asyncio.get_event_loop()

        def recv_all():
            """Drain all buffered ZMQ messages and return them all.

            Consuming the full queue every poll cycle prevents the backlog
            from growing when sleap-nn publishes faster than we drain (e.g.
            fast GPUs).  All messages are returned so the LossViewer receives
            every batch_end event and every scatter dot is plotted correctly.

            Returns:
                List of message strings (possibly empty).
            """
            messages = []
            try:
                while True:
                    messages.append(self.progress_socket.recv_string(flags=zmq.NOBLOCK))
            except zmq.Again:
                pass
            return messages

        while self.running:
            # Drain the full ZMQ queue and forward every message so the
            # LossViewer sees all batch_end scatter dots, not just the last.
            msgs = await loop.run_in_executor(None, recv_all)

            for msg in msgs:
                try:
                    logging.debug(f"Sending progress report to client: {msg}")
                    channel.send(f"PROGRESS_REPORT::{msg}")
                except Exception as e:
                    logging.error(f"Failed to send ZMQ progress: {e}")

            # Polling interval
            await asyncio.sleep(0.05)

    def start_progress_listener_task(self, channel: RTCDataChannel) -> asyncio.Task:
        """Start progress listener as a background task.

        Args:
            channel: RTC data channel to send progress updates.

        Returns:
            Asyncio task for the progress listener.
        """
        self.listener_task = asyncio.create_task(self.start_progress_listener(channel))
        logging.info("Progress listener task started")
        return self.listener_task

    def stop_progress_listener(self):
        """Stop the progress listener task."""
        if self.listener_task and not self.listener_task.done():
            self.running = False
            self.listener_task.cancel()
            logging.info("Progress listener task stopped")

    def cleanup(self):
        """Clean up ZMQ sockets and context (synchronous).

        Prefer ``async_cleanup()`` when called from an async context: this
        synchronous version cancels the listener task but cannot await its
        termination, so ``context.term()`` may block the event loop briefly
        if the executor thread is still in flight.
        """
        logging.info("Cleaning up ZMQ progress reporter...")

        # Stop listener if running
        if self.listener_task and not self.listener_task.done():
            self.stop_progress_listener()

        # Close sockets with linger=0 so term() doesn't block on pending msgs
        if self.ctrl_socket:
            self.ctrl_socket.setsockopt(zmq.LINGER, 0)
            self.ctrl_socket.close()
            self.ctrl_socket = None
            logging.info("Control socket closed")

        if self.progress_socket:
            self.progress_socket.setsockopt(zmq.LINGER, 0)
            self.progress_socket.close()
            self.progress_socket = None
            logging.info("Progress socket closed")

        # Terminate context
        if self.context:
            self.context.term()
            self.context = None
            logging.info("ZMQ context terminated")

    async def restart_progress_listener(self, channel: RTCDataChannel) -> None:
        """Stop listener, discard buffered messages, and start fresh.

        Call this between pipeline models to prevent stale ZMQ messages
        from the previous model leaking into the next model's LossViewer.
        The ZMQ sockets themselves stay open so ports remain bound.
        """
        # Stop the current listener task and wait for it (and its
        # run_in_executor thread) to finish accessing the socket.
        if self.listener_task and not self.listener_task.done():
            self.running = False
            self.listener_task.cancel()
            await asyncio.wait({self.listener_task}, timeout=2.0)
        # Brief yield so the executor thread has time to return from recv_all.
        await asyncio.sleep(0.1)

        # Drain and discard all messages still buffered in the ZMQ socket.
        if self.progress_socket is not None:
            discarded = 0
            while True:
                try:
                    self.progress_socket.recv_string(flags=zmq.NOBLOCK)
                    discarded += 1
                except zmq.Again:
                    break
            if discarded:
                logging.info(
                    f"[ZMQ] Flushed {discarded} buffered progress messages between models"
                )

        # Start a fresh listener for the next model.
        self.start_progress_listener_task(channel)

    async def async_cleanup(self) -> None:
        """Clean up ZMQ sockets and context (async).

        Cancels and awaits the listener task before closing sockets so that
        the executor thread has stopped using the ZMQ socket before
        ``context.term()`` is called.  Use this from async contexts to avoid
        blocking the asyncio event loop.
        """
        logging.info("Cleaning up ZMQ progress reporter (async)...")

        if self.listener_task and not self.listener_task.done():
            self.running = False
            self.listener_task.cancel()
            # Wait up to 2 s for the task to acknowledge cancellation so the
            # run_in_executor thread stops accessing the socket.
            await asyncio.wait({self.listener_task}, timeout=2.0)

        if self.ctrl_socket:
            self.ctrl_socket.setsockopt(zmq.LINGER, 0)
            self.ctrl_socket.close()
            self.ctrl_socket = None
            logging.info("Control socket closed")

        if self.progress_socket:
            self.progress_socket.setsockopt(zmq.LINGER, 0)
            self.progress_socket.close()
            self.progress_socket = None
            logging.info("Progress socket closed")

        if self.context:
            self.context.term()
            self.context = None
            logging.info("ZMQ context terminated")

    def is_control_socket_active(self) -> bool:
        """Check if control socket is initialized.

        Returns:
            True if control socket is active, False otherwise.
        """
        return self.ctrl_socket is not None

    def is_progress_listener_running(self) -> bool:
        """Check if progress listener is running.

        Returns:
            True if listener is running, False otherwise.
        """
        return self.listener_task is not None and not self.listener_task.done()
