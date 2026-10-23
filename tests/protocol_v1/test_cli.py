"""Tests for the protocol v1 CLI additions (pair, doctor/status sections).

`serve`'s CLI wrapper itself (argument parsing, printing) isn't separately
covered here — it deliberately loops forever until Ctrl-C, and the
substantive behavior it wraps (start_worker_server) already has thorough
integration tests in test_runner.py. Testing the CLI wrapper would mean
either signal-based interruption or timeouts that add flakiness for little
additional coverage.
"""

import json

from click.testing import CliRunner

from sleap_rtc.protocol_v1.cli import (
    _parse_mount,
    pair,
    print_doctor_section,
    print_status_section,
)
from sleap_rtc.protocol_v1.pair_code import decode_pair_code
from sleap_rtc.protocol_v1.pairing import PendingPairings
from sleap_rtc.protocol_v1.runner import identity_path


class TestParseMount:
    """Tests for `_parse_mount`, the `--mount PATH[:LABEL]` parser."""

    def test_bare_path_labels_itself_with_the_basename(self):
        mount = _parse_mount("/data/videos")

        assert mount.path == "/data/videos"
        assert mount.label == "videos"

    def test_path_with_explicit_label(self):
        mount = _parse_mount("/data/videos:Lab Data")

        assert mount.path == "/data/videos"
        assert mount.label == "Lab Data"

    def test_root_path_falls_back_to_the_path_itself_for_a_label(self):
        # Path("/").name is "" — the basename fallback would be empty.
        mount = _parse_mount("/")

        assert mount.path == "/"
        assert mount.label == "/"


class TestPairCommand:
    """Tests for the `sleap-rtc pair` command."""

    def test_prints_a_one_line_pairing_code_by_default(self, tmp_path):
        runner = CliRunner()

        result = runner.invoke(pair, ["--data-dir", str(tmp_path)])

        assert result.exit_code == 0
        lines = [line for line in result.output.splitlines() if line.strip()]
        code_lines = [line for line in lines if line.startswith("sleap1")]
        assert len(code_lines) == 1, result.output
        decode_pair_code(code_lines[0])  # doesn't raise

    def test_json_flag_prints_a_ticket_as_json(self, tmp_path):
        runner = CliRunner()

        result = runner.invoke(pair, ["--data-dir", str(tmp_path), "--json"])

        assert result.exit_code == 0
        assert "node_id" in result.output
        assert "secret" in result.output

    def test_creates_an_identity_if_none_exists_yet(self, tmp_path):
        runner = CliRunner()

        runner.invoke(pair, ["--data-dir", str(tmp_path)])

        assert identity_path(tmp_path).exists()

    def test_printed_ticket_is_claimable_by_a_separate_instance(self, tmp_path):
        # Simulates: `sleap-rtc pair` mints it, a running `serve` process
        # (a different PendingPairings instance, same data dir) claims it.
        runner = CliRunner()
        result = runner.invoke(pair, ["--data-dir", str(tmp_path)])

        code = next(
            line for line in result.output.splitlines() if line.startswith("sleap1")
        )
        ticket = decode_pair_code(code)

        from sleap_rtc.protocol_v1.runner import pairing_path

        claimer = PendingPairings(path=pairing_path(tmp_path))
        assert claimer.claim(ticket.secret) is True

    def test_respects_addr_option(self, tmp_path):
        runner = CliRunner()

        result = runner.invoke(
            pair,
            [
                "--data-dir",
                str(tmp_path),
                "--json",
                "--addr",
                "ws://192.168.1.42:9631",
                "--addr",
                "ws://100.64.0.1:9631",
            ],
        )

        json_start = result.output.index("{")
        json_end = result.output.rindex("}") + 1
        ticket = json.loads(result.output[json_start:json_end])
        assert ticket["addrs"] == ["ws://192.168.1.42:9631", "ws://100.64.0.1:9631"]


class TestDoctorSection:
    """Tests for print_doctor_section."""

    def test_reports_not_yet_generated_before_any_identity_exists(
        self, tmp_path, capsys
    ):
        print_doctor_section(data_dir=tmp_path)

        captured = capsys.readouterr()
        assert "not yet generated" in captured.out

    def test_reports_the_node_id_once_an_identity_exists(self, tmp_path, capsys):
        from sleap_rtc.protocol_v1.identity import WorkerIdentity

        identity = WorkerIdentity(identity_path(tmp_path))

        print_doctor_section(data_dir=tmp_path)

        captured = capsys.readouterr()
        assert identity.node_id in captured.out

    def test_reports_paired_client_count(self, tmp_path, capsys):
        from sleap_rtc.protocol_v1.runner import open_trust_store

        open_trust_store(tmp_path).add_trusted("some-node-id")

        print_doctor_section(data_dir=tmp_path)

        captured = capsys.readouterr()
        assert "Paired clients: 1" in captured.out


class TestStatusSection:
    """Tests for print_status_section."""

    def test_reports_not_initialized_before_any_identity_exists(self, tmp_path, capsys):
        print_status_section(data_dir=tmp_path)

        captured = capsys.readouterr()
        assert "Not yet initialized" in captured.out

    def test_reports_recent_jobs(self, tmp_path, capsys):
        import asyncio

        from sleap_rtc.jobs.spec import TrainJobSpec
        from sleap_rtc.jobs.store import JobStore
        from sleap_rtc.protocol_v1.identity import WorkerIdentity

        WorkerIdentity(identity_path(tmp_path))

        async def _seed():
            async with JobStore(tmp_path / "jobs.sqlite") as store:
                await store.create_job("job-1", TrainJobSpec(config_path="/x.yaml"))

        asyncio.run(_seed())

        print_status_section(data_dir=tmp_path)

        captured = capsys.readouterr()
        assert "job-1: queued" in captured.out


class TestClientsCommand:
    def test_lists_paired_devices(self, tmp_path):
        from sleap_rtc.protocol_v1.cli import clients
        from sleap_rtc.protocol_v1.runner import open_trust_store

        store = open_trust_store(tmp_path)
        store.add_trusted(
            "node-aaaaaaaaaaaa1",
            {"device_name": "Lab iMac", "platform": "tauri", "version": "0.1.2"},
        )
        store.add_trusted(
            "node-bbbbbbbbbbbb2", {"device_name": "Chrome on macOS", "platform": "web"}
        )

        result = CliRunner().invoke(clients, ["--data-dir", str(tmp_path)])

        assert result.exit_code == 0, result.output
        assert "Lab iMac" in result.output and "desktop 0.1.2" in result.output
        assert "Chrome on macOS" in result.output and "browser" in result.output
        assert "2 paired devices." in result.output

    def test_no_devices_yet(self, tmp_path):
        from sleap_rtc.protocol_v1.cli import clients

        result = CliRunner().invoke(clients, ["--data-dir", str(tmp_path)])
        assert "No paired devices yet" in result.output


class TestClientsRemove:
    def _store(self, tmp_path):
        from sleap_rtc.protocol_v1.runner import open_trust_store

        store = open_trust_store(tmp_path)
        store.add_trusted("aaaa1111", {"device_name": "Rig 1", "platform": "tauri"})
        store.add_trusted("aaaa2222", {"device_name": "Rig 2", "platform": "web"})
        store.add_trusted("bbbb3333", {"device_name": "Lab iMac", "platform": "tauri"})
        store.add_trusted("cccc4444")  # unnamed, from the old allowlist
        return store

    def _run(self, tmp_path, *args, input=None):
        from sleap_rtc.protocol_v1.cli import clients

        return CliRunner().invoke(
            clients, ["remove", *args, "--data-dir", str(tmp_path)], input=input
        )

    def test_by_name_any_case_after_confirming(self, tmp_path):
        store = self._store(tmp_path)
        result = self._run(tmp_path, "lab imac", input="y\n")
        assert result.exit_code == 0, result.output
        assert 'Removed "Lab iMac"' in result.output
        assert "bbbb3333" not in store.list_trusted()

    def test_declining_removes_nothing(self, tmp_path):
        store = self._store(tmp_path)
        result = self._run(tmp_path, "Rig 1", input="n\n")
        assert "Nothing removed." in result.output
        assert len(store.list_trusted()) == 4

    def test_by_id_prefix_with_yes(self, tmp_path):
        store = self._store(tmp_path)
        result = self._run(tmp_path, "cccc", "--yes")
        assert result.exit_code == 0, result.output
        assert 'Removed "(unnamed)"' in result.output
        assert "cccc4444" not in store.list_trusted()

    def test_ambiguous_prefix_lists_matches_and_removes_nothing(self, tmp_path):
        store = self._store(tmp_path)
        result = self._run(tmp_path, "aaaa", "--yes")
        assert result.exit_code != 0
        assert "matches 2 devices" in result.output
        assert "Rig 1" in result.output and "Rig 2" in result.output
        assert len(store.list_trusted()) == 4

    def test_unknown_target(self, tmp_path):
        self._store(tmp_path)
        result = self._run(tmp_path, "nope", "--yes")
        assert result.exit_code != 0
        assert 'No paired device matches "nope"' in result.output

    def test_all_removes_devices_and_unused_pairing_codes(self, tmp_path):
        from sleap_rtc.protocol_v1.pairing import PendingPairings
        from sleap_rtc.protocol_v1.runner import pairing_path

        store = self._store(tmp_path)
        ticket = PendingPairings(path=pairing_path(tmp_path)).create("worker", [])
        result = self._run(tmp_path, "--all", input="y\n")
        assert result.exit_code == 0, result.output
        assert (
            "Remove ALL 4 paired devices and invalidate 1 unused pairing code?"
            in result.output
        )
        assert "Removed 4 devices; invalidated 1 pairing code." in result.output
        assert store.list_trusted() == set()
        assert (
            PendingPairings(path=pairing_path(tmp_path)).claim(ticket.secret) is False
        )

    def test_needs_a_target_or_all(self, tmp_path):
        self._store(tmp_path)
        assert self._run(tmp_path).exit_code != 0
        assert self._run(tmp_path, "Rig 1", "--all").exit_code != 0

    def test_listing_still_works_without_a_subcommand(self, tmp_path):
        from sleap_rtc.protocol_v1.cli import clients

        self._store(tmp_path)
        result = CliRunner().invoke(clients, ["--data-dir", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "4 paired devices." in result.output
