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


class TestPairedDevices:
    """What's recorded about each paired device, and the old JSON import."""

    AGENT = {"name": "sleap-app", "version": "0.1.2", "platform": "tauri", "device_name": "Lab iMac"}

    def test_pairing_records_name_platform_version_and_times(self, tmp_path):
        store = TrustStore(tmp_path / "devices.sqlite")
        store.add_trusted("node-1", self.AGENT)

        [device] = store.list_devices()
        assert (device.name, device.platform, device.app_version) == ("Lab iMac", "tauri", "0.1.2")
        assert device.paired_at is not None and device.last_seen_at is not None

    def test_seen_updates_last_seen_and_reported_fields_only(self, tmp_path):
        store = TrustStore(tmp_path / "devices.sqlite")
        store.add_trusted("node-1", self.AGENT)
        [before] = store.list_devices()

        store.record_seen("node-1", {"version": "0.1.3", "platform": "tauri"})  # no name sent
        [after] = store.list_devices()
        assert after.name == "Lab iMac"  # kept
        assert after.app_version == "0.1.3"
        assert after.paired_at == before.paired_at
        assert after.last_seen_at >= before.last_seen_at

        store.record_seen("node-1", {"device_name": "  Renamed iMac  "})
        assert store.list_devices()[0].name == "Renamed iMac"

    def test_another_process_sees_changes_right_away(self, tmp_path):
        # The CLI and a running worker each have their own TrustStore.
        worker = TrustStore(tmp_path / "devices.sqlite")
        cli = TrustStore(tmp_path / "devices.sqlite")
        cli.add_trusted("node-1")
        assert worker.is_trusted("node-1") is True

    def test_old_json_allowlist_is_imported_once(self, tmp_path):
        import json

        legacy = tmp_path / "trusted_clients.json"
        legacy.write_text(json.dumps({"trusted_node_ids": ["old-1", "old-2"]}))

        store = TrustStore(tmp_path / "devices.sqlite", legacy_json_path=legacy)

        assert store.list_trusted() == {"old-1", "old-2"}
        assert all(d.name is None and d.paired_at is None for d in store.list_devices())
        assert not legacy.exists()
        assert (tmp_path / "trusted_clients.json.imported").exists()

    def test_list_is_most_recently_seen_first(self, tmp_path):
        store = TrustStore(tmp_path / "devices.sqlite")
        store.add_trusted("older")
        store.add_trusted("newer")
        store.record_seen("newer")
        assert [d.node_id for d in store.list_devices()][0] == "newer"
