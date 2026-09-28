"""Integration tests for start_worker_server — the full assembly."""

import asyncio
import urllib.error
import urllib.request

import iroh
import pytest
import websockets

from sleap_rtc.auth.keypair import generate_keypair, public_key_to_b64
from sleap_rtc.protocol_v1.envelope import Hello, Req, parse_envelope
from sleap_rtc.protocol_v1.iroh_transport import ALPN, IrohStreamTransport
from sleap_rtc.protocol_v1.runner import start_worker_server


async def _pair_and_call(ws_port, ticket_secret, node_id, method, params=None):
    ws = await websockets.connect(f"ws://127.0.0.1:{ws_port}")
    await ws.send(
        Hello(
            proto={"min": 1, "max": 1},
            agent={"name": "test-client", "version": "0.0.0", "platform": "test"},
            node_id=node_id,
            nonce="client-nonce",
        ).to_json()
    )
    await ws.recv()  # server's hello

    await ws.send(
        Req(
            id=0,
            method="pair.claim",
            params={"secret": ticket_secret, "node_id": node_id},
        ).to_json()
    )
    pair_reply = parse_envelope(await ws.recv())
    assert pair_reply.result == {}

    await ws.send(Req(id=1, method=method, params=params or {}).to_json())
    reply = parse_envelope(await ws.recv())
    await ws.close()
    return reply


class TestStartWorkerServer:
    """Tests that start_worker_server assembles a genuinely working server."""

    async def test_assembles_a_server_that_accepts_a_real_pairing_and_call(
        self, tmp_path
    ):
        worker = await start_worker_server(host="127.0.0.1", port=0, data_dir=tmp_path)
        try:
            port = worker.ws_server.sockets[0].getsockname()[1]
            _priv, public_key = generate_keypair()
            client_node_id = public_key_to_b64(public_key)
            ticket = worker.pending_pairings.create(worker.identity.node_id, [])

            reply = await _pair_and_call(
                port, ticket.secret, client_node_id, "jobs.list"
            )

            assert reply.result == {"jobs": []}
            assert worker.trust_store.is_trusted(client_node_id) is True
        finally:
            await worker.close()

    async def test_persists_state_under_data_dir(self, tmp_path):
        worker = await start_worker_server(host="127.0.0.1", port=0, data_dir=tmp_path)
        try:
            assert (tmp_path / "identity.json").exists()
            assert (tmp_path / "jobs.sqlite").exists()
        finally:
            await worker.close()

    async def test_reattach_outcomes_reported_for_a_stuck_job_from_a_prior_run(
        self, tmp_path
    ):
        # Simulate a previous instance's job store showing a job as
        # "running" with no real process behind it (e.g. after a crash).
        from sleap_rtc.jobs.spec import TrainJobSpec
        from sleap_rtc.jobs.store import JobStore

        async with JobStore(tmp_path / "jobs.sqlite") as store:
            await store.create_job("job-1", TrainJobSpec(config_path="/x.yaml"))
            await store.update_state("job-1", "running")

        worker = await start_worker_server(host="127.0.0.1", port=0, data_dir=tmp_path)
        try:
            assert worker.reattach_outcomes == {"job-1": "marked_failed"}
            record = await worker.store.get_job("job-1")
            assert record.state == "failed"
        finally:
            await worker.close()


class TestFileManagerWiring:
    """fs.mounts/fs.list must be reachable once a FileManager is passed
    through — this is what `sleap-rtc serve` does today (always, even with
    zero configured mounts), after previously never passing one at all,
    which left every real client's unconditional fsMounts() call during
    pairing hitting `proto.unknown_method`.
    """

    async def test_fs_mounts_responds_instead_of_unknown_method(self, tmp_path):
        from sleap_rtc.worker.file_manager import FileManager

        worker = await start_worker_server(
            host="127.0.0.1", port=0, data_dir=tmp_path, file_manager=FileManager()
        )
        try:
            port = worker.ws_server.sockets[0].getsockname()[1]
            _priv, public_key = generate_keypair()
            client_node_id = public_key_to_b64(public_key)
            ticket = worker.pending_pairings.create(worker.identity.node_id, [])

            reply = await _pair_and_call(
                port, ticket.secret, client_node_id, "fs.mounts"
            )

            assert reply.error is None
            assert reply.result == {"mounts": []}
        finally:
            await worker.close()

    async def test_fs_mounts_reports_a_configured_mount(self, tmp_path):
        from sleap_rtc.config import MountConfig
        from sleap_rtc.worker.file_manager import FileManager

        mount_dir = tmp_path / "data"
        mount_dir.mkdir()
        file_manager = FileManager(
            mounts=[MountConfig(path=str(mount_dir), label="data")]
        )
        worker = await start_worker_server(
            host="127.0.0.1", port=0, data_dir=tmp_path, file_manager=file_manager
        )
        try:
            port = worker.ws_server.sockets[0].getsockname()[1]
            _priv, public_key = generate_keypair()
            client_node_id = public_key_to_b64(public_key)
            ticket = worker.pending_pairings.create(worker.identity.node_id, [])

            reply = await _pair_and_call(
                port, ticket.secret, client_node_id, "fs.mounts"
            )

            assert reply.result == {
                "mounts": [{"path": str(mount_dir), "label": "data"}]
            }
        finally:
            await worker.close()


async def _wait_for_direct_addresses(endpoint, timeout=5.0):
    """Poll until `endpoint.addr()` reports at least one direct address.

    `preset_minimal()` disables relay entirely, and `online()` hangs
    forever with no relay to become "online" via — direct addresses show
    up almost immediately without it (see iroh_transport.py's test suite
    for the same helper, verified by hand there).
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if endpoint.addr().direct_addresses():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("endpoint never reported a direct address")


class TestIrohWiring:
    """Tests that `enable_iroh=True` assembles a genuinely reachable iroh endpoint."""

    async def test_iroh_endpoint_id_matches_the_workers_node_id(self, tmp_path):
        worker = await start_worker_server(
            host="127.0.0.1",
            port=0,
            data_dir=tmp_path,
            enable_iroh=True,
            iroh_preset=iroh.preset_minimal(),
        )
        try:
            # str(EndpointId) is hex, node_id is base64 -- compare the
            # underlying bytes, not the string encodings.
            import base64

            decoded = base64.urlsafe_b64decode(worker.identity.node_id + "==")
            assert decoded.hex() == str(worker.iroh_endpoint.id())
        finally:
            await worker.close()

    async def test_a_real_client_can_pair_and_call_over_iroh(self, tmp_path):
        worker = await start_worker_server(
            host="127.0.0.1",
            port=0,
            data_dir=tmp_path,
            enable_iroh=True,
            iroh_preset=iroh.preset_minimal(),
        )
        try:
            await _wait_for_direct_addresses(worker.iroh_endpoint)

            client_ep = await iroh.Endpoint.bind(
                iroh.EndpointOptions(preset=iroh.preset_minimal())
            )
            try:
                _priv, public_key = generate_keypair()
                client_node_id = public_key_to_b64(public_key)
                conn = await client_ep.connect(worker.iroh_endpoint.addr(), ALPN)
                bi = await conn.open_bi()
                transport = IrohStreamTransport(conn, bi)

                await transport.send(
                    Hello(
                        proto={"min": 1, "max": 1},
                        agent={
                            "name": "test-client",
                            "version": "0.0.0",
                            "platform": "test",
                        },
                        node_id=client_node_id,
                        nonce="client-nonce",
                    ).to_json()
                )
                hello_reply = parse_envelope(await transport.recv())
                assert isinstance(hello_reply, Hello)
                assert hello_reply.node_id == worker.identity.node_id

                ticket = worker.pending_pairings.create(worker.identity.node_id, [])
                await transport.send(
                    Req(
                        id=0,
                        method="pair.claim",
                        params={"secret": ticket.secret, "node_id": client_node_id},
                    ).to_json()
                )
                pair_reply = parse_envelope(await transport.recv())
                assert pair_reply.result == {}

                await transport.send(Req(id=1, method="jobs.list", params={}).to_json())
                res = parse_envelope(await transport.recv())
                assert res.result == {"jobs": []}
            finally:
                await client_ep.close()
        finally:
            await worker.close()

    async def test_worker_still_accepts_plain_ws_when_iroh_is_also_enabled(
        self, tmp_path
    ):
        # enable_iroh must be additive, not a replacement for the WS
        # binding -- a client on the same network shouldn't need iroh at
        # all just because the worker also happens to support it.
        worker = await start_worker_server(
            host="127.0.0.1",
            port=0,
            data_dir=tmp_path,
            enable_iroh=True,
            iroh_preset=iroh.preset_minimal(),
        )
        try:
            ws_port = worker.ws_server.sockets[0].getsockname()[1]
            _priv, public_key = generate_keypair()
            client_node_id = public_key_to_b64(public_key)
            ticket = worker.pending_pairings.create(worker.identity.node_id, [])

            reply = await _pair_and_call(
                ws_port, ticket.secret, client_node_id, "jobs.list"
            )
            assert reply.result == {"jobs": []}
        finally:
            await worker.close()


class TestBlobServing:
    """Tests that start_worker_server also wires up real blob serving."""

    async def test_blob_port_is_a_real_bound_port_not_a_placeholder(self, tmp_path):
        # port=0 (OS picks the WS port) must not leave blob_port resolved to
        # the nonsensical "0 + 1" default — see runner.py's own note on why.
        worker = await start_worker_server(host="127.0.0.1", port=0, data_dir=tmp_path)
        try:
            assert worker.blob_port not in (0, 1)
            assert worker.blob_port == worker._blob_http_server.server_address[1]
        finally:
            await worker.close()

    async def test_hello_announces_the_real_blob_port(self, tmp_path):
        worker = await start_worker_server(host="127.0.0.1", port=0, data_dir=tmp_path)
        try:
            ws_port = worker.ws_server.sockets[0].getsockname()[1]
            ws = await websockets.connect(f"ws://127.0.0.1:{ws_port}")
            await ws.send(
                Hello(
                    proto={"min": 1, "max": 1},
                    agent={},
                    node_id="client-node",
                    nonce="client-nonce",
                ).to_json()
            )
            reply = parse_envelope(await ws.recv())
            assert reply.blob_port == worker.blob_port
            await ws.close()
        finally:
            await worker.close()

    async def test_a_registered_blob_is_fetchable_over_real_http(self, tmp_path):
        worker = await start_worker_server(host="127.0.0.1", port=0, data_dir=tmp_path)
        try:
            content = b"real end-to-end predictions bytes"
            f = tmp_path / "predictions.slp"
            f.write_bytes(content)
            from sleap_rtc.protocol_v1.blobs import hash_file

            sha256, size = await hash_file(f)
            await worker.blob_index.register(sha256, str(f), size)

            url = f"http://127.0.0.1:{worker.blob_port}/blobs/{sha256}"
            with urllib.request.urlopen(url, timeout=5) as resp:
                assert resp.status == 200
                assert resp.read() == content
        finally:
            await worker.close()

    async def test_blob_server_stops_accepting_connections_after_close(self, tmp_path):
        worker = await start_worker_server(host="127.0.0.1", port=0, data_dir=tmp_path)
        blob_port = worker.blob_port
        await worker.close()

        with pytest.raises(urllib.error.URLError):
            urllib.request.urlopen(
                f"http://127.0.0.1:{blob_port}/blobs/" + "a" * 64, timeout=2
            )
