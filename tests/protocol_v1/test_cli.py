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
from sleap_rtc.protocol_v1.pairing import PendingPairings
from sleap_rtc.protocol_v1.runner import identity_path, trust_store_path


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

    def test_prints_a_ticket_as_json(self, tmp_path):
        runner = CliRunner()

        result = runner.invoke(pair, ["--data-dir", str(tmp_path)])

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

        # Extract the JSON block from the command's output.
        json_start = result.output.index("{")
        json_end = result.output.rindex("}") + 1
        ticket = json.loads(result.output[json_start:json_end])

        from sleap_rtc.protocol_v1.runner import pairing_path

        claimer = PendingPairings(path=pairing_path(tmp_path))
        assert claimer.claim(ticket["secret"]) is True

    def test_respects_addr_option(self, tmp_path):
        runner = CliRunner()

        result = runner.invoke(
            pair,
            [
                "--data-dir",
                str(tmp_path),
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
        from sleap_rtc.protocol_v1.trust_store import TrustStore

        TrustStore(trust_store_path(tmp_path)).add_trusted("some-node-id")

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
