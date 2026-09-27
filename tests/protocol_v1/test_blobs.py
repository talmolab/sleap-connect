"""Tests for BlobIndex and hash_file (protocol v1 result blobs)."""

import hashlib
import threading

from sleap_rtc.protocol_v1.blobs import BlobIndex, hash_file


class TestHashFile:
    """Tests for hash_file."""

    async def test_matches_hashlib_for_a_small_file(self, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"hello world")

        sha256, size = await hash_file(f)

        assert sha256 == hashlib.sha256(b"hello world").hexdigest()
        assert size == 11

    async def test_hashes_a_file_spanning_multiple_chunks(self, tmp_path):
        # hash_file reads in 1 MiB chunks — exercise a file that spans more
        # than one chunk so the loop actually iterates, not just the
        # single-read path.
        payload = b"x" * (1024 * 1024 + 137)
        f = tmp_path / "big.bin"
        f.write_bytes(payload)

        sha256, size = await hash_file(f)

        assert sha256 == hashlib.sha256(payload).hexdigest()
        assert size == len(payload)

    async def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")

        sha256, size = await hash_file(f)

        assert sha256 == hashlib.sha256(b"").hexdigest()
        assert size == 0


class TestBlobIndex:
    """Tests for BlobIndex."""

    async def test_get_returns_none_for_an_unknown_hash(self, tmp_path):
        index = BlobIndex(tmp_path / "blobs.sqlite")
        assert await index.get("no-such-hash") is None

    async def test_register_then_get_round_trips(self, tmp_path):
        index = BlobIndex(tmp_path / "blobs.sqlite")

        await index.register("abc123", "/data/out.slp", 4096)
        record = await index.get("abc123")

        assert record is not None
        assert record.sha256 == "abc123"
        assert record.path == "/data/out.slp"
        assert record.size == 4096

    async def test_get_sync_matches_the_async_get(self, tmp_path):
        index = BlobIndex(tmp_path / "blobs.sqlite")
        await index.register("abc123", "/data/out.slp", 4096)

        record = index.get_sync("abc123")

        assert record is not None
        assert record.path == "/data/out.slp"

    async def test_registering_the_same_hash_again_overwrites_the_path(self, tmp_path):
        index = BlobIndex(tmp_path / "blobs.sqlite")
        await index.register("abc123", "/data/old.slp", 100)

        await index.register("abc123", "/data/new.slp", 200)

        record = await index.get("abc123")
        assert record.path == "/data/new.slp"
        assert record.size == 200

    async def test_persists_across_instances_pointed_at_the_same_file(self, tmp_path):
        db_path = tmp_path / "blobs.sqlite"
        first = BlobIndex(db_path)
        await first.register("abc123", "/data/out.slp", 4096)
        first.close()

        second = BlobIndex(db_path)
        record = await second.get("abc123")

        assert record is not None
        assert record.path == "/data/out.slp"

    async def test_get_sync_is_safe_to_call_from_another_thread(self, tmp_path):
        # The whole point of get_sync existing separately from get: it must
        # be callable from a plain thread with no asyncio event loop at all
        # (the blob HTTP server's request-handling threads) — not just from
        # asyncio.to_thread, which is what `get()` itself already uses.
        index = BlobIndex(tmp_path / "blobs.sqlite")
        await index.register("abc123", "/data/out.slp", 4096)

        result = {}

        def _lookup():
            result["record"] = index.get_sync("abc123")

        thread = threading.Thread(target=_lookup)
        thread.start()
        thread.join(timeout=5)

        assert result["record"] is not None
        assert result["record"].path == "/data/out.slp"

    async def test_in_memory_db_path_works_for_a_single_instance(self, tmp_path):
        # ":memory:" is documented as tests-only (a second connection can't
        # see it) — confirm the single-instance case at least works.
        index = BlobIndex(":memory:")
        await index.register("abc123", "/data/out.slp", 4096)

        assert (await index.get("abc123")).path == "/data/out.slp"
