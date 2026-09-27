"""Integration tests for start_worker_server — the full assembly."""

import websockets

from sleap_rtc.auth.keypair import generate_keypair, public_key_to_b64
from sleap_rtc.protocol_v1.envelope import Hello, Req, parse_envelope
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
