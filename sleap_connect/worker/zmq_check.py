import zmq
import threading
import time
import json


# Fake RTC data channel
class DummyRTCChannel:
    """A dummy class to simulate a WebRTC data channel."""

    def __init__(self):
        self.readyState = "open"

    def send(self, msg):
        """Simulate sending a message over the WebRTC data channel.

        Args:
            msg (str): The message to send.

        Raises:
            RuntimeError: If the channel is not open.
        """
        print(f"[RTC SEND] {msg}")


# Fake ZMQ PUB publisher to simulate sleap-train
def simulate_training_pub(zmq_address):
    """Simulate a ZMQ PUB publisher that sends training progress updates.

    Args:
        zmq_address (str): The ZMQ address to bind the publisher socket to.

    Raises:
        RuntimeError: If the publisher fails to bind to the ZMQ address.
    """
    ctx = zmq.Context()
    pub_socket = ctx.socket(zmq.PUB)
    pub_socket.bind(zmq_address)

    # Give SUB time to connect
    time.sleep(1)

    for i in range(5):
        update = json.dumps({"epoch": i, "loss": round(1.0 / (i + 1), 4)})
        print(f"[PUB] Sending: {update}")
        pub_socket.send_string(update)
        time.sleep(1)


# Your real ZMQ listener (the one inside Worker)
def start_progress_listener(zmq_address, rtc_channel):
    """Start a ZMQ SUB listener that receives training progress updates.

    Args:
        zmq_address (str): The ZMQ address to connect the subscriber socket to.
        rtc_channel (DummyRTCChannel): The dummy RTC channel to send updates to.

    Raises:
        RuntimeError: If the subscriber fails to connect to the ZMQ address.
    """
    ctx = zmq.Context()
    sub_socket = ctx.socket(zmq.SUB)
    sub_socket.connect(zmq_address)
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

    print("[SUB] Connected. Waiting for messages...")
    while True:
        msg = sub_socket.recv_string()
        print(f"[SUB] Received: {msg}")
        if rtc_channel.readyState == "open":
            rtc_channel.send(f"TRAIN_PROGRESS:{msg}")
        else:
            print("[RTC] Data channel not open.")


# Main entrypoint.
if __name__ == "__main__":
    zmq_address = "tcp://127.0.0.1:9001"
    rtc_channel = DummyRTCChannel()

    # Start fake training publisher.
    threading.Thread(
        target=start_progress_listener, args=(zmq_address, rtc_channel), daemon=True
    ).start()

    # Give SUB socket time to fully connect/subscribe.
    time.sleep(1)

    # Start progress listener.
    simulate_training_pub(zmq_address)
