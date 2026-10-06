"""Tests for PendingPairings (in-memory pairing ticket lifecycle)."""

import json
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


class TestFileBackedPendingPairings:
    """Tests for the optional file-backed mode (cross-process handoff)."""

    def test_a_ticket_minted_in_one_instance_is_claimable_in_another(self, tmp_path):
        # Simulates `sleap-rtc pair` (mints) and `sleap-rtc serve` (claims)
        # as two separate processes sharing the same file.
        path = tmp_path / "pending.json"
        minter = PendingPairings(path=path)
        ticket = minter.create("worker-node-id", [])

        claimer = PendingPairings(path=path)

        assert claimer.claim(ticket.secret) is True

    def test_claim_in_one_instance_is_visible_to_another(self, tmp_path):
        path = tmp_path / "pending.json"
        minter = PendingPairings(path=path)
        ticket = minter.create("worker-node-id", [])
        claimer = PendingPairings(path=path)
        claimer.claim(ticket.secret)

        # Re-load fresh (simulating a third process) — already consumed.
        third = PendingPairings(path=path)
        assert third.claim(ticket.secret) is False

    def test_expired_tickets_are_pruned_from_the_file(self, tmp_path):
        path = tmp_path / "pending.json"
        pairings = PendingPairings(ttl_secs=0.01, path=path)
        pairings.create("worker-node-id", [])
        time.sleep(0.05)

        pairings.create("worker-node-id", [])  # triggers a reload + prune

        data = json.loads(path.read_text())
        assert len(data["pending"]) == 1  # only the fresh one survived

    def test_without_a_path_behaves_exactly_as_in_memory(self, tmp_path):
        # No file should be created at all in the default (no-path) mode.
        pairings = PendingPairings()
        ticket = pairings.create("worker-node-id", [])

        assert pairings.claim(ticket.secret) is True
        assert not (tmp_path / "pending.json").exists()
