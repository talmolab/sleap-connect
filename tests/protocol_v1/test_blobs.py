"""Tests for BlobIndex and hash_file (protocol v1 result blobs)."""

import hashlib
import threading

from sleap_rtc.protocol_v1.blobs import (
    VERIFY_CHUNK_SIZE,
    BlobIndex,
    compute_chunk_hashes,
    hash_file,
)


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

    async def test_register_without_chunk_hashes_defaults_to_empty(self, tmp_path):
        # Existing callers only ever passed (sha256, path, size) — must keep
        # working unchanged now that chunk_hashes exists (item 2.4).
        index = BlobIndex(tmp_path / "blobs.sqlite")
        await index.register("abc123", "/data/out.slp", 4096)

        record = await index.get("abc123")

        assert record.chunk_hashes == []

    async def test_register_round_trips_chunk_hashes(self, tmp_path):
        index = BlobIndex(tmp_path / "blobs.sqlite")

        await index.register("abc123", "/data/out.slp", 4096, ["h0", "h1", "h2"])

        record = await index.get("abc123")
        assert record.chunk_hashes == ["h0", "h1", "h2"]
        assert index.get_sync("abc123").chunk_hashes == ["h0", "h1", "h2"]


class TestComputeChunkHashes:
    """Tests for compute_chunk_hashes (item 2.4's per-chunk integrity check)."""

    async def test_empty_file_has_no_chunks(self, tmp_path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")

        assert await compute_chunk_hashes(f) == []

    async def test_a_file_smaller_than_one_chunk_gets_a_single_shorter_hash(
        self, tmp_path
    ):
        payload = b"hello world"
        f = tmp_path / "small.bin"
        f.write_bytes(payload)

        hashes = await compute_chunk_hashes(f)

        assert hashes == [hashlib.sha256(payload).hexdigest()]

    async def test_hashes_each_aligned_chunk_independently(self, tmp_path):
        chunk_size = 16
        payload = (b"a" * chunk_size) + (b"b" * chunk_size) + b"c"  # 2 full + 1 short
        f = tmp_path / "multi.bin"
        f.write_bytes(payload)

        hashes = await compute_chunk_hashes(f, chunk_size=chunk_size)

        assert hashes == [
            hashlib.sha256(b"a" * chunk_size).hexdigest(),
            hashlib.sha256(b"b" * chunk_size).hexdigest(),
            hashlib.sha256(b"c").hexdigest(),
        ]

    async def test_default_chunk_size_matches_verify_chunk_size(self, tmp_path):
        # A file exactly VERIFY_CHUNK_SIZE + 1 byte must produce exactly two
        # chunks: one full-size, one 1-byte tail — proves the default really
        # is VERIFY_CHUNK_SIZE, not some other constant.
        payload = (b"x" * VERIFY_CHUNK_SIZE) + b"y"
        f = tmp_path / "boundary.bin"
        f.write_bytes(payload)

        hashes = await compute_chunk_hashes(f)

        assert hashes == [
            hashlib.sha256(b"x" * VERIFY_CHUNK_SIZE).hexdigest(),
            hashlib.sha256(b"y").hexdigest(),
        ]
