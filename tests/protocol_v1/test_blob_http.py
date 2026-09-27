"""Tests for the blob-serving HTTP endpoint (real server, real HTTP requests).

Spins up a real `ThreadingHTTPServer` on an OS-assigned free port and makes
real HTTP requests against it — no mocking of `http.server`/`http.client`,
since the whole point of this module is wire-level behavior (status codes,
`Range`/`Content-Range` headers, exact byte payloads).
"""

import urllib.error
import urllib.request

import pytest

from sleap_rtc.protocol_v1.blob_http import (
    make_blob_http_server,
    run_blob_http_server_in_thread,
)
from sleap_rtc.protocol_v1.blobs import BlobIndex, hash_file

KNOWN_HASH = "a" * 64  # a well-formed-looking but unregistered hash


@pytest.fixture
async def running_server(tmp_path):
    """A real blob HTTP server, running in a background thread, torn down after."""
    index = BlobIndex(tmp_path / "blobs.sqlite")
    server = make_blob_http_server(index, "127.0.0.1", 0)
    run_blob_http_server_in_thread(server)
    try:
        yield server, index
    finally:
        server.shutdown()
        server.server_close()


def _url(server, path: str) -> str:
    host, port = server.server_address[:2]
    return f"http://{host}:{port}{path}"


def _get(url: str, headers: dict | None = None):
    req = urllib.request.Request(url, headers=headers or {})
    return urllib.request.urlopen(req, timeout=5)


class TestBlobHttpServer:
    """Tests for the blob HTTP endpoint."""

    async def test_404_for_a_malformed_path(self, running_server):
        server, _ = running_server
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _get(_url(server, "/not-a-blob-path"))
        assert exc_info.value.code == 404

    async def test_404_for_a_well_formed_but_unregistered_hash(self, running_server):
        server, _ = running_server
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _get(_url(server, f"/blobs/{KNOWN_HASH}"))
        assert exc_info.value.code == 404

    async def test_full_download_of_a_registered_blob(self, running_server, tmp_path):
        server, index = running_server
        content = b"predicted labels go here" * 100
        f = tmp_path / "predictions.slp"
        f.write_bytes(content)
        sha256, size = await hash_file(f)
        await index.register(sha256, str(f), size)

        with _get(_url(server, f"/blobs/{sha256}")) as resp:
            assert resp.status == 200
            assert resp.headers["Content-Length"] == str(size)
            assert resp.headers["Accept-Ranges"] == "bytes"
            assert resp.read() == content

    async def test_404_if_the_registered_file_no_longer_exists_on_disk(
        self, running_server, tmp_path
    ):
        server, index = running_server
        f = tmp_path / "gone.slp"
        f.write_bytes(b"will be deleted")
        sha256, size = await hash_file(f)
        await index.register(sha256, str(f), size)
        f.unlink()

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _get(_url(server, f"/blobs/{sha256}"))
        assert exc_info.value.code == 404

    async def test_range_request_returns_206_with_the_requested_slice(
        self, running_server, tmp_path
    ):
        server, index = running_server
        content = bytes(range(256)) * 4  # 1024 distinct-ish bytes
        f = tmp_path / "data.bin"
        f.write_bytes(content)
        sha256, size = await hash_file(f)
        await index.register(sha256, str(f), size)

        with _get(_url(server, f"/blobs/{sha256}"), {"Range": "bytes=10-19"}) as resp:
            assert resp.status == 206
            assert resp.headers["Content-Range"] == f"bytes 10-19/{size}"
            assert resp.headers["Content-Length"] == "10"
            assert resp.read() == content[10:20]

    async def test_suffix_range_returns_the_last_n_bytes(
        self, running_server, tmp_path
    ):
        server, index = running_server
        content = bytes(range(256))
        f = tmp_path / "data.bin"
        f.write_bytes(content)
        sha256, size = await hash_file(f)
        await index.register(sha256, str(f), size)

        with _get(_url(server, f"/blobs/{sha256}"), {"Range": "bytes=-16"}) as resp:
            assert resp.status == 206
            assert resp.read() == content[-16:]

    async def test_open_ended_range_returns_from_offset_to_eof(
        self, running_server, tmp_path
    ):
        server, index = running_server
        content = bytes(range(256))
        f = tmp_path / "data.bin"
        f.write_bytes(content)
        sha256, size = await hash_file(f)
        await index.register(sha256, str(f), size)

        with _get(_url(server, f"/blobs/{sha256}"), {"Range": "bytes=200-"}) as resp:
            assert resp.status == 206
            assert resp.read() == content[200:]

    async def test_out_of_range_request_is_416(self, running_server, tmp_path):
        server, index = running_server
        f = tmp_path / "small.bin"
        f.write_bytes(b"only 10 b.")
        sha256, size = await hash_file(f)
        await index.register(sha256, str(f), size)

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _get(_url(server, f"/blobs/{sha256}"), {"Range": "bytes=100-200"})
        assert exc_info.value.code == 416

    async def test_range_with_neither_side_given_is_416(self, running_server, tmp_path):
        # "bytes=-" (both first-byte-pos and suffix-length empty) is
        # malformed per RFC 7233 — must be rejected the same as any other
        # unparseable Range, not silently treated as "the whole file".
        server, index = running_server
        content = bytes(range(256))
        f = tmp_path / "data.bin"
        f.write_bytes(content)
        sha256, size = await hash_file(f)
        await index.register(sha256, str(f), size)

        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _get(_url(server, f"/blobs/{sha256}"), {"Range": "bytes=-"})
        assert exc_info.value.code == 416

    async def test_empty_file_returns_200_with_zero_length(
        self, running_server, tmp_path
    ):
        server, index = running_server
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        sha256, size = await hash_file(f)
        await index.register(sha256, str(f), size)

        with _get(_url(server, f"/blobs/{sha256}")) as resp:
            assert resp.status == 200
            assert resp.headers["Content-Length"] == "0"
            assert resp.read() == b""
