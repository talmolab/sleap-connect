"""Tests for encrypted relay message envelopes."""

import pytest

from sleap_connect.encryption.ecdh import derive_shared_key, generate_keypair
from sleap_connect.encryption.envelope import ENCRYPTED_RELAY_TYPE, unwrap, wrap


class TestWrap:
    def test_envelope_has_correct_type(self):
        key, session_id = _make_session()
        envelope = wrap(key, session_id, {"type": "fs_list_req"})
        assert envelope["type"] == ENCRYPTED_RELAY_TYPE

    def test_envelope_has_session_id(self):
        key, session_id = _make_session()
        envelope = wrap(key, session_id, {"type": "test"})
        assert envelope["session_id"] == session_id

    def test_envelope_has_nonce_and_ciphertext(self):
        key, session_id = _make_session()
        envelope = wrap(key, session_id, {"type": "test"})
        assert "nonce" in envelope
        assert "ciphertext" in envelope
        assert isinstance(envelope["nonce"], str)
        assert isinstance(envelope["ciphertext"], str)

    def test_envelope_includes_job_id_when_provided(self):
        key, session_id = _make_session()
        envelope = wrap(key, session_id, {"type": "test"}, job_id="job_abc123")
        assert envelope["job_id"] == "job_abc123"

    def test_envelope_includes_req_id_when_provided(self):
        key, session_id = _make_session()
        envelope = wrap(key, session_id, {"type": "test"}, req_id="req-uuid")
        assert envelope["req_id"] == "req-uuid"

    def test_envelope_excludes_job_id_when_not_provided(self):
        key, session_id = _make_session()
        envelope = wrap(key, session_id, {"type": "test"})
        assert "job_id" not in envelope

    def test_payload_not_readable_in_envelope(self):
        key, session_id = _make_session()
        message = {"type": "job_submit", "config": {"secret_path": "/mnt/data"}}
        envelope = wrap(key, session_id, message)
        envelope_str = str(envelope)
        assert "secret_path" not in envelope_str
        assert "/mnt/data" not in envelope_str


class TestUnwrap:
    def test_roundtrip(self):
        key, session_id = _make_session()
        message = {"type": "fs_list_req", "path": "/data", "offset": 0}
        envelope = wrap(key, session_id, message)
        result = unwrap(envelope, {session_id: key})
        assert result == message

    def test_unknown_session_id_returns_none(self):
        key, session_id = _make_session()
        envelope = wrap(key, session_id, {"type": "test"})
        result = unwrap(envelope, {"other-session": key})
        assert result is None

    def test_wrong_key_returns_none(self):
        key1, session_id = _make_session()
        key2, _ = _make_session()
        envelope = wrap(key1, session_id, {"type": "test"})
        result = unwrap(envelope, {session_id: key2})
        assert result is None

    def test_missing_session_id_returns_none(self):
        key, session_id = _make_session()
        envelope = wrap(key, session_id, {"type": "test"})
        del envelope["session_id"]
        result = unwrap(envelope, {session_id: key})
        assert result is None

    def test_tampered_ciphertext_returns_none(self):
        key, session_id = _make_session()
        envelope = wrap(key, session_id, {"type": "test"})
        # Tamper with the base64 ciphertext
        ct = envelope["ciphertext"]
        envelope["ciphertext"] = "A" + ct[1:]
        result = unwrap(envelope, {session_id: key})
        assert result is None

    def test_malformed_envelope_returns_none(self):
        result = unwrap(
            {"type": ENCRYPTED_RELAY_TYPE, "session_id": "abc"},
            {"abc": b"\x00" * 32},
        )
        assert result is None

    def test_empty_key_lookup_returns_none(self):
        key, session_id = _make_session()
        envelope = wrap(key, session_id, {"type": "test"})
        result = unwrap(envelope, {})
        assert result is None


class TestWrapUnwrapIntegration:
    def test_bidirectional_with_separate_keys(self):
        """Simulates client and worker each wrapping/unwrapping."""
        priv_a, pub_a = generate_keypair()
        priv_b, pub_b = generate_keypair()

        key_a = derive_shared_key(priv_a, pub_b)
        key_b = derive_shared_key(priv_b, pub_a)

        session_id = "test-session-123"

        # Client wraps, worker unwraps
        msg1 = {"type": "fs_list_req", "path": "/data"}
        envelope1 = wrap(key_a, session_id, msg1)
        result1 = unwrap(envelope1, {session_id: key_b})
        assert result1 == msg1

        # Worker wraps, client unwraps
        msg2 = {"type": "fs_list_res", "entries": [{"name": "file.slp"}]}
        envelope2 = wrap(key_b, session_id, msg2, job_id="job_xyz")
        result2 = unwrap(envelope2, {session_id: key_a})
        assert result2 == msg2

    def test_multiple_sessions_coexist(self):
        """Worker handles multiple concurrent dashboard clients."""
        key1, sid1 = _make_session()
        key2, sid2 = _make_session()

        key_lookup = {sid1: key1, sid2: key2}

        env1 = wrap(key1, sid1, {"client": 1})
        env2 = wrap(key2, sid2, {"client": 2})

        assert unwrap(env1, key_lookup) == {"client": 1}
        assert unwrap(env2, key_lookup) == {"client": 2}

        # Cross-session doesn't work
        assert unwrap(wrap(key1, sid2, {"bad": True}), key_lookup) is None


def _make_session() -> tuple[bytes, str]:
    """Helper: generate a session key and ID."""
    import uuid

    priv_a, _ = generate_keypair()
    _, pub_b = generate_keypair()
    key = derive_shared_key(priv_a, pub_b)
    return key, str(uuid.uuid4())
