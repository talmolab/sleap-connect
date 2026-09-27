"""Tests for PendingPairings (in-memory pairing ticket lifecycle)."""

import time

from sleap_rtc.protocol_v1.pairing import PendingPairings


class TestPendingPairings:
    """Tests for PendingPairings."""

    def test_create_returns_a_ticket_with_the_given_node_id_and_addrs(self):
        pairings = PendingPairings()

        ticket = pairings.create("worker-node-id", ["ws://192.168.1.42:9631"])

        assert ticket.node_id == "worker-node-id"
        assert ticket.addrs == ["ws://192.168.1.42:9631"]
        assert isinstance(ticket.secret, str) and len(ticket.secret) > 0

    def test_claim_succeeds_for_a_fresh_ticket(self):
        pairings = PendingPairings()
        ticket = pairings.create("worker-node-id", [])

        assert pairings.claim(ticket.secret) is True

    def test_claim_is_single_use(self):
        pairings = PendingPairings()
        ticket = pairings.create("worker-node-id", [])
        pairings.claim(ticket.secret)

        assert pairings.claim(ticket.secret) is False

    def test_claim_fails_for_an_unknown_secret(self):
        pairings = PendingPairings()

        assert pairings.claim("not-a-real-secret") is False

    def test_claim_fails_for_an_expired_ticket(self):
        pairings = PendingPairings(ttl_secs=0.01)
        ticket = pairings.create("worker-node-id", [])
        time.sleep(0.05)

        assert pairings.claim(ticket.secret) is False

    def test_to_dict_has_the_wire_shape(self):
        pairings = PendingPairings()
        ticket = pairings.create("worker-node-id", ["ws://host:1234"])

        d = ticket.to_dict()

        assert set(d.keys()) == {"node_id", "addrs", "secret", "expires_at"}
