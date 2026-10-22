"""Choosing the ZMQ ports a train job's sleap-nn talks to the worker on.

sleap-nn connects to two worker-bound sockets: the controller (where the
worker sends Stop Early) and the progress publisher. The defaults, 9000 and
9001, are also what local training in SLEAP and sleap-app bind, so a worker
on a machine that's training locally, or a second job's leftover sockets,
can find them taken. A job then uses two free OS-assigned ports instead; the
caller records them so a restarted worker rebinds the same ones.
"""

import socket
from typing import Dict, Iterable


def _port_is_free(port: int) -> bool:
    """Whether 127.0.0.1:`port` can be bound right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _free_port(exclude: Iterable[int]) -> int:
    """An OS-assigned free port on 127.0.0.1 that isn't in `exclude`."""
    excluded = set(exclude)
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        if port not in excluded:
            return port


def choose_zmq_ports(preferred: Dict[str, int]) -> Dict[str, int]:
    """Use `preferred` (e.g. 9000/9001) if every port in it is free, else
    free ports for all of them.

    Both move together: a half-taken pair most likely means another training
    is using it, and splitting it would leave this job sharing one socket
    with that training.

    Args:
        preferred: Port per socket name, e.g. ``{"controller": 9000,
            "publish": 9001}``.

    Returns:
        A port per socket name, all distinct.
    """
    if all(_port_is_free(port) for port in preferred.values()):
        return dict(preferred)
    chosen: Dict[str, int] = {}
    for name in preferred:
        chosen[name] = _free_port(exclude=[*preferred.values(), *chosen.values()])
    return chosen
