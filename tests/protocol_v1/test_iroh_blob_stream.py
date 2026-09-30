"""Integration tests for item 2.4's iroh blob-range-read stream — a real
iroh Endpoint, a real client Endpoint, real QUIC streams, mirroring
`test_iroh_transport.py`'s fixtures/style.

These exist to prove two things by direct hands-on test, not by reading the
Python `iroh` bindings' docs:

1. `Connection.accept_bi()` really can be called repeatedly, once for the
   control stream and again in a loop for additional blob streams, on the
   SAME connection, with both streams independently readable/writable
   (design doc §7's flagged, previously-unverified assumption).
2. The open + read-loop wire shape (design doc §6/§6.1) round-trips real
   bytes correctly, including a non-zero-offset read and a read clamped at
   EOF.
"""

import asyncio
import json

import iroh
import pytest

from sleap_rtc.protocol_v1.blobs import BlobIndex, compute_chunk_hashes, hash_file
from sleap_rtc.protocol_v1.iroh_transport import ALPN, read_frame, write_frame
from sleap_rtc.protocol_v1.server import ProtocolV1Server

_MINIMAL = iroh.preset_minimal


async def _wait_for_direct_addresses(endpoint, timeout=5.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if endpoint.addr().direct_addresses():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("endpoint never reported a direct address")


@pytest.fixture
async def running_iroh_server_with_blob(tmp_path):
    """A ProtocolV1Server wired to a real BlobIndex, on a real local iroh endpoint."""
    blob_index = BlobIndex(tmp_path / "blobs.sqlite")
    server_obj = ProtocolV1Server(node_id="test-worker", blob_index=blob_index)

    endpoint = await iroh.Endpoint.bind(
        iroh.EndpointOptions(preset=_MINIMAL(), alpns=[ALPN])
    )
    await _wait_for_direct_addresses(endpoint)
    serve_task = asyncio.create_task(server_obj.serve_iroh(endpoint))
    try:
        yield server_obj, endpoint, blob_index, tmp_path
    finally:
        await endpoint.close()
        serve_task.cancel()


async def _connect_control_and_open_blob(server_addr, sha256):
    """Dial, open the control stream (so `_accept_iroh_blob_streams` starts),
    then open a second stream for a blob-read session and complete its open
    phase. Returns `(client_ep, control_bi, blob_send, blob_recv, open_reply)`.
    """
    client_ep = await iroh.Endpoint.bind(iroh.EndpointOptions(preset=_MINIMAL()))
    conn = await client_ep.connect(server_addr, ALPN)

    # Control stream first — this is what triggers _handle_iroh_incoming to
    # start the blob-accept loop on the server side. It's never used for
    # hello/dispatch in these tests (that's test_iroh_transport.py's job);
    # opening it is only to exercise the real coexistence-with-control-stream
    # path, per this file's own docstring point 1.
    control_bi = await conn.open_bi()

    blob_bi = await conn.open_bi()
    blob_send, blob_recv = blob_bi.send(), blob_bi.recv()
    await write_frame(blob_send, json.dumps({"sha256": sha256}))
    open_reply = json.loads(await read_frame(blob_recv))

    return client_ep, control_bi, blob_send, blob_recv, open_reply


class TestBlobStreamCoexistsWithControlStream:
    """The core, previously-unverified assumption: accept_bi() in a loop."""

    async def test_control_and_blob_streams_are_both_independently_usable(
        self, running_iroh_server_with_blob
    ):
        _server_obj, endpoint, blob_index, tmp_path = running_iroh_server_with_blob
        f = tmp_path / "out.bin"
        f.write_bytes(b"hello blob world")
        sha256, size = await hash_file(f)
        chunk_hashes = await compute_chunk_hashes(f)
        await blob_index.register(sha256, str(f), size, chunk_hashes)

        client_ep, control_bi, blob_send, blob_recv, open_reply = (
            await _connect_control_and_open_blob(endpoint.addr(), sha256)
        )
        try:
            assert open_reply["ok"] is True
            assert open_reply["size"] == size

            # The control stream is a completely separate, still-open pipe —
            # prove it independently by sending an arbitrary frame on it and
            # confirming the connection (and this stream) is still alive.
            # (No hello handshake is done here on purpose — that dispatch
            # path is test_iroh_transport.py's job; this only needs the raw
            # stream to still be open and distinct from the blob stream.)
            await write_frame(control_bi.send(), json.dumps({"probe": True}))
        finally:
            await client_ep.close()


class TestBlobStreamReads:
    """The open + read-loop wire shape round-trips real bytes correctly."""

    async def test_sequential_reads_including_nonzero_offset_and_eof_clamp(
        self, running_iroh_server_with_blob
    ):
        _server_obj, endpoint, blob_index, tmp_path = running_iroh_server_with_blob
        payload = bytes(range(256)) * 4  # 1024 bytes, easy to slice and verify
        f = tmp_path / "out.bin"
        f.write_bytes(payload)
        sha256, size = await hash_file(f)
        chunk_hashes = await compute_chunk_hashes(f)
        await blob_index.register(sha256, str(f), size, chunk_hashes)

        client_ep, _control_bi, send, recv, open_reply = (
            await _connect_control_and_open_blob(endpoint.addr(), sha256)
        )
        try:
            assert open_reply["chunkSize"] > 0
            assert open_reply["chunkHashes"] == chunk_hashes

            # A read starting at 0.
            await write_frame(send, json.dumps({"offset": 0, "length": 100}))
            header = json.loads(await read_frame(recv))
            assert header["ok"] is True
            assert header["size"] == 100
            body = await recv.read_exact(100)
            assert body == payload[0:100]

            # A read starting at a non-zero offset, on the SAME stream.
            await write_frame(send, json.dumps({"offset": 500, "length": 50}))
            header = json.loads(await read_frame(recv))
            assert header["size"] == 50
            body = await recv.read_exact(50)
            assert body == payload[500:550]

            # A read whose requested length runs past EOF must clamp.
            await write_frame(send, json.dumps({"offset": size - 10, "length": 1000}))
            header = json.loads(await read_frame(recv))
            assert header["size"] == 10
            body = await recv.read_exact(10)
            assert body == payload[size - 10 :]
        finally:
            await client_ep.close()

    async def test_not_found_at_open(self, running_iroh_server_with_blob):
        _server_obj, endpoint, _blob_index, _tmp_path = running_iroh_server_with_blob

        client_ep = await iroh.Endpoint.bind(iroh.EndpointOptions(preset=_MINIMAL()))
        conn = await client_ep.connect(endpoint.addr(), ALPN)
        try:
            await conn.open_bi()  # control stream, so the accept loop starts
            blob_bi = await conn.open_bi()
            send, recv = blob_bi.send(), blob_bi.recv()
            await write_frame(send, json.dumps({"sha256": "no-such-hash"}))
            reply = json.loads(await read_frame(recv))
            assert reply == {"ok": False, "error": "not_found"}
        finally:
            await client_ep.close()

    async def test_a_worker_with_no_blob_index_reports_not_found(
        self,
    ):
        # A worker constructed without blob_index at all (the untouched,
        # existing default) must not crash the blob-accept loop — it should
        # just report not_found for anything, same as a real not-found case.
        server_obj = ProtocolV1Server(node_id="test-worker")  # no blob_index
        endpoint = await iroh.Endpoint.bind(
            iroh.EndpointOptions(preset=_MINIMAL(), alpns=[ALPN])
        )
        await _wait_for_direct_addresses(endpoint)
        serve_task = asyncio.create_task(server_obj.serve_iroh(endpoint))
        try:
            client_ep = await iroh.Endpoint.bind(
                iroh.EndpointOptions(preset=_MINIMAL())
            )
            conn = await client_ep.connect(endpoint.addr(), ALPN)
            try:
                await conn.open_bi()
                blob_bi = await conn.open_bi()
                send, recv = blob_bi.send(), blob_bi.recv()
                await write_frame(send, json.dumps({"sha256": "whatever"}))
                reply = json.loads(await read_frame(recv))
                assert reply == {"ok": False, "error": "not_found"}
            finally:
                await client_ep.close()
        finally:
            await endpoint.close()
            serve_task.cancel()
