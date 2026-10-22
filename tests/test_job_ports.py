"""Tests for choosing a train job's ZMQ ports (sleap_rtc.jobs.ports)."""

import socket

from sleap_rtc.jobs.ports import choose_zmq_ports


def _free_pair():
    socks = []
    for _ in range(2):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        socks.append(s)
    ports = {
        "controller": socks[0].getsockname()[1],
        "publish": socks[1].getsockname()[1],
    }
    for s in socks:
        s.close()
    return ports


def test_uses_the_preferred_ports_when_both_are_free():
    preferred = _free_pair()
    assert choose_zmq_ports(preferred) == preferred


def test_moves_both_ports_when_one_is_taken():
    preferred = _free_pair()
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", preferred["controller"]))
        holder.listen()
        chosen = choose_zmq_ports(preferred)
    assert set(chosen) == {"controller", "publish"}
    assert set(chosen.values()).isdisjoint(preferred.values())
    assert chosen["controller"] != chosen["publish"]
