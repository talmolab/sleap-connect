"""Tests for the serve→pair iroh reachability bridge (item 2.1).

The integration tests use a real worker (`start_worker_server` with a real
iroh endpoint on `preset_minimal`, so no external network), a real `pair`
CLI invocation reading the live file it wrote, and then a real client
endpoint dialing *using only what the printed ticket says* — proving the
ticket's `iroh` section is actually sufficient to reach the worker.
"""

import asyncio
import json
import os

import iroh
import pytest
from click.testing import CliRunner

from sleap_rtc.protocol_v1 import iroh_live
from sleap_rtc.protocol_v1.cli import pair
from sleap_rtc.protocol_v1.envelope import Hello, parse_envelope
from sleap_rtc.protocol_v1.iroh_live import (
    build_iroh_section,
    iroh_live_path,
    keep_iroh_live_updated,
    read_iroh_live,
    remove_iroh_live,
    write_iroh_live,
)
from sleap_rtc.protocol_v1.iroh_transport import ALPN, IrohStreamTransport
from sleap_rtc.protocol_v1.runner import start_worker_server

NODE = "node-abc"


def _section(relay="https://relay.example", addrs=("10.0.0.5:1234",)):
    return build_iroh_section(NODE, relay, list(addrs))


class TestReadIrohLive:
    def test_round_trips_a_written_section(self, tmp_path):
        path = iroh_live_path(tmp_path)
        write_iroh_live(path, _section())

        assert read_iroh_live(path, NODE) == _section()

    def test_missing_file_is_none(self, tmp_path):
        assert read_iroh_live(iroh_live_path(tmp_path), NODE) is None

    def test_unparsable_file_is_none(self, tmp_path):
        path = iroh_live_path(tmp_path)
        path.write_text("{not json")

        assert read_iroh_live(path, NODE) is None

    def test_wrong_shape_is_none(self, tmp_path):
        path = iroh_live_path(tmp_path)
        path.write_text(json.dumps(["nope"]))
        assert read_iroh_live(path, NODE) is None

        path.write_text(json.dumps({"node_id": NODE, "direct_addrs": "1.2.3.4:5"}))
        assert read_iroh_live(path, NODE) is None

    def test_other_identity_is_ignored(self, tmp_path):
        path = iroh_live_path(tmp_path)
        write_iroh_live(path, _section())

        assert read_iroh_live(path, "someone-else") is None

    def test_nothing_to_dial_yet_is_none(self, tmp_path):
        path = iroh_live_path(tmp_path)
        write_iroh_live(path, build_iroh_section(NODE, None, []))

        assert read_iroh_live(path, NODE) is None

    def test_relay_only_is_usable(self, tmp_path):
        path = iroh_live_path(tmp_path)
        write_iroh_live(path, build_iroh_section(NODE, "https://r.example", []))

        assert read_iroh_live(path, NODE)["relay_url"] == "https://r.example"

    def test_file_from_a_dead_serve_is_ignored(self, tmp_path):
        path = iroh_live_path(tmp_path)
        write_iroh_live(path, _section())
        data = json.loads(path.read_text())
        data["pid"] = 2**22 + 12345  # beyond any real pid on the platforms we run on
        path.write_text(json.dumps(data))

        assert read_iroh_live(path, NODE) is None

    def test_live_file_carries_this_pid(self, tmp_path):
        path = iroh_live_path(tmp_path)
        write_iroh_live(path, _section())

        assert json.loads(path.read_text())["pid"] == os.getpid()

    def test_remove_is_idempotent(self, tmp_path):
        path = iroh_live_path(tmp_path)
        write_iroh_live(path, _section())

        remove_iroh_live(path)
        remove_iroh_live(path)

        assert not path.exists()


class _FakeAddr:
    def __init__(self, relay, direct):
        self._relay, self._direct = relay, direct

    def relay_url(self):
        return self._relay

    def direct_addresses(self):
        return self._direct


class _FakeEndpoint:
    def __init__(self):
        self.current = _FakeAddr(None, ["10.0.0.5:1"])

    def addr(self):
        return self.current


class TestKeepIrohLiveUpdated:
    async def test_rewrites_the_file_when_the_address_changes(self, tmp_path):
        # A real relay home can't be forced in a test, so this one uses a
        # stand-in endpoint whose address we can change on demand.
        path = iroh_live_path(tmp_path)
        endpoint = _FakeEndpoint()
        task = asyncio.create_task(
            keep_iroh_live_updated(endpoint, path, NODE, interval=0.02)
        )
        try:
            await asyncio.sleep(0.1)
            assert read_iroh_live(path, NODE)["relay_url"] is None

            endpoint.current = _FakeAddr("https://relay.example", ["10.0.0.5:1"])
            await asyncio.sleep(0.1)

            assert read_iroh_live(path, NODE)["relay_url"] == "https://relay.example"
        finally:
            task.cancel()


async def _wait_for(predicate, timeout=5.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition never became true")


@pytest.fixture
async def iroh_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(iroh_live, "DEFAULT_REFRESH_INTERVAL_SECS", 0.05)
    worker = await start_worker_server(
        host="127.0.0.1",
        port=0,
        data_dir=tmp_path,
        enable_iroh=True,
        iroh_preset=iroh.preset_minimal(),
    )
    try:
        yield worker
    finally:
        await worker.close()


class TestServeToPairBridge:
    async def test_running_serve_publishes_a_file_pair_embeds_in_the_ticket(
        self, iroh_worker, tmp_path
    ):
        live = iroh_live_path(tmp_path)
        await _wait_for(lambda: read_iroh_live(live, iroh_worker.identity.node_id))

        result = CliRunner().invoke(pair, ["--data-dir", str(tmp_path)])

        assert result.exit_code == 0, result.output
        ticket = json.loads(
            result.output[result.output.index("{") : result.output.rindex("}") + 1]
        )
        assert ticket["iroh"]["node_id"] == iroh_worker.identity.node_id
        assert ticket["iroh"]["direct_addrs"]
        assert set(ticket["iroh"]) == {"node_id", "relay_url", "direct_addrs"}

    async def test_a_client_can_dial_using_only_the_tickets_iroh_section(
        self, iroh_worker, tmp_path
    ):
        live = iroh_live_path(tmp_path)
        await _wait_for(lambda: read_iroh_live(live, iroh_worker.identity.node_id))
        result = CliRunner().invoke(pair, ["--data-dir", str(tmp_path)])
        ticket = json.loads(
            result.output[result.output.index("{") : result.output.rindex("}") + 1]
        )
        section = ticket["iroh"]

        # Rebuild a dialable address from the ticket alone: the iroh
        # endpoint id is the identity's raw public key.
        from sleap_rtc.auth.keypair import public_key_from_b64

        raw = public_key_from_b64(section["node_id"]).public_bytes_raw()
        addr = iroh.EndpointAddr(
            id=iroh.EndpointId.from_bytes(raw),
            relay_url=section["relay_url"],
            addresses=section["direct_addrs"],
        )
        client_ep = await iroh.Endpoint.bind(
            iroh.EndpointOptions(preset=iroh.preset_minimal())
        )
        try:
            conn = await client_ep.connect(addr, ALPN)
            transport = IrohStreamTransport(conn, await conn.open_bi())
            await transport.send(
                Hello(
                    proto={"min": 1, "max": 1},
                    agent={},
                    node_id="ticket-dialer",
                    nonce="n",
                ).to_json()
            )
            reply = parse_envelope(await transport.recv())
            assert reply.node_id == ticket["node_id"]
        finally:
            await client_ep.close()

    async def test_closing_the_worker_removes_the_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(iroh_live, "DEFAULT_REFRESH_INTERVAL_SECS", 0.05)
        worker = await start_worker_server(
            host="127.0.0.1",
            port=0,
            data_dir=tmp_path,
            enable_iroh=True,
            iroh_preset=iroh.preset_minimal(),
        )
        assert iroh_live_path(tmp_path).exists()

        await worker.close()

        assert not iroh_live_path(tmp_path).exists()

    async def test_worker_without_iroh_writes_no_file(self, tmp_path):
        worker = await start_worker_server(host="127.0.0.1", port=0, data_dir=tmp_path)
        try:
            assert not iroh_live_path(tmp_path).exists()
        finally:
            await worker.close()


class TestPairWithoutLiveFile:
    def test_ticket_has_no_iroh_key_when_no_serve_is_running(self, tmp_path):
        result = CliRunner().invoke(pair, ["--data-dir", str(tmp_path)])

        assert result.exit_code == 0
        assert "iroh" not in result.output

    def test_garbage_live_file_leaves_pair_output_unchanged(self, tmp_path):
        iroh_live_path(tmp_path).write_text("{garbage")

        result = CliRunner().invoke(pair, ["--data-dir", str(tmp_path)])

        assert result.exit_code == 0
        assert "iroh" not in result.output
