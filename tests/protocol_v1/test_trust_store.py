"""Tests for TrustStore (persisted trusted-client allowlist)."""

from sleap_rtc.protocol_v1.trust_store import TrustStore


class TestTrustStore:
    """Tests for TrustStore."""

    def test_unknown_node_is_not_trusted(self, tmp_path):
        store = TrustStore(tmp_path / "trusted.json")

        assert store.is_trusted("some-node-id") is False

    def test_added_node_is_trusted(self, tmp_path):
        store = TrustStore(tmp_path / "trusted.json")

        store.add_trusted("node-1")

        assert store.is_trusted("node-1") is True

    def test_adding_the_same_node_twice_is_idempotent(self, tmp_path):
        store = TrustStore(tmp_path / "trusted.json")

        store.add_trusted("node-1")
        store.add_trusted("node-1")

        assert store.list_trusted() == {"node-1"}

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "trusted.json"
        first = TrustStore(path)
        first.add_trusted("node-1")

        second = TrustStore(path)

        assert second.is_trusted("node-1") is True

    def test_list_trusted_returns_a_copy_not_a_live_reference(self, tmp_path):
        store = TrustStore(tmp_path / "trusted.json")
        store.add_trusted("node-1")

        snapshot = store.list_trusted()
        snapshot.add("node-2")

        assert store.is_trusted("node-2") is False
