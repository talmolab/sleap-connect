"""Blob-serving HTTP endpoint for protocol v1 result blobs (spec §6.3).

The JSON envelope over the WS control connection never carries bulk bytes
(spec §6.1) — a completed job's result file is instead fetched via a plain
`GET /blobs/<sha256>` on a small side-channel HTTP server, with `Range`
support so a client can resume an interrupted download. Stdlib-only
(`http.server`): this worker only ever needs to *serve* one thing (a
completed job's own output file, looked up by content hash via
`BlobIndex`), so a full web framework would be more dependency than the
job warrants.

This module only serves downloads (worker -> client). Uploads
(`blobs.create`, client -> worker) are a different data-ladder rung, not
needed for stage 1.10's "predictions back as a blob" goal, and are left
for whenever that rung is actually built.
"""

import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from sleap_rtc.protocol_v1.blobs import BlobIndex

_PATH_RE = re.compile(r"^/blobs/([0-9a-f]{64})$")
_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")
_CHUNK_SIZE = 64 * 1024


def _make_handler(blob_index: BlobIndex) -> type:
    """Build a BaseHTTPRequestHandler subclass bound to `blob_index`.

    A class (not an instance) is what `HTTPServer` requires — closing over
    `blob_index` here is how each request handler gets access to it.
    """

    class BlobRequestHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            # BaseHTTPRequestHandler logs every request to stderr by
            # default — this worker already logs at the protocol/job level,
            # so stay quiet rather than duplicate it there.
            pass

        def do_GET(self) -> None:
            match = _PATH_RE.match(self.path)
            if not match:
                self.send_error(HTTPStatus.NOT_FOUND, "no such route")
                return
            sha256 = match.group(1)

            record = blob_index.get_sync(sha256)
            if record is None:
                self.send_error(HTTPStatus.NOT_FOUND, "unknown blob")
                return

            path = Path(record.path)
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "blob file missing on disk")
                return

            file_size = path.stat().st_size
            start, end = 0, file_size - 1
            status = HTTPStatus.OK

            range_header = self.headers.get("Range")
            if range_header is not None:
                range_match = _RANGE_RE.match(range_header)
                if range_match is None:
                    self.send_error(
                        HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
                        "malformed Range header",
                    )
                    return
                start_str, end_str = range_match.groups()
                if start_str:
                    start = int(start_str)
                    end = int(end_str) if end_str else file_size - 1
                elif end_str:
                    # "bytes=-N" — the last N bytes.
                    start = max(0, file_size - int(end_str))
                    end = file_size - 1
                else:
                    # "bytes=-" — neither side given, invalid per RFC 7233.
                    self.send_error(
                        HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
                        "malformed Range header",
                    )
                    return
                status = HTTPStatus.PARTIAL_CONTENT

            if file_size == 0:
                # An empty file has no valid byte range at all ("0-0" would
                # wrongly claim one byte); 200 with Content-Length: 0 is the
                # correct response regardless of any Range header.
                start, end, status = 0, -1, HTTPStatus.OK
            elif start > end or start >= file_size or end >= file_size:
                self.send_error(
                    HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
                    f"valid range is bytes 0-{file_size - 1}",
                )
                return

            length = max(0, end - start + 1)
            self.send_response(status)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.end_headers()

            if length == 0:
                return
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(_CHUNK_SIZE, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

    return BlobRequestHandler


def make_blob_http_server(
    blob_index: BlobIndex, host: str, port: int
) -> ThreadingHTTPServer:
    """Build (but don't start) the blob-serving HTTP server.

    Args:
        blob_index: Where to resolve a requested sha256 to a file.
        host: Bind address.
        port: Bind port.

    Returns:
        A `ThreadingHTTPServer` — call `.serve_forever()` (typically in a
        background thread; see `run_blob_http_server_in_thread`) and
        `.shutdown()` / `.server_close()` to stop it.
    """
    return ThreadingHTTPServer((host, port), _make_handler(blob_index))


def run_blob_http_server_in_thread(server: ThreadingHTTPServer) -> threading.Thread:
    """Start `server.serve_forever()` on a daemon background thread.

    `HTTPServer.serve_forever()` blocks the calling thread, so it can't run
    on the same thread as the worker's asyncio event loop (the WS server) —
    this is the standard way to run a classic threaded server alongside an
    asyncio application.

    Args:
        server: A server built by `make_blob_http_server`.

    Returns:
        The started thread (daemon=True, so it never blocks process exit).
    """
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread
