"""Tests for WorkerIdentity (persistent Ed25519 keypair)."""

import sys

import pytest

from sleap_rtc.auth.keypair import public_key_from_b64, verify_signature
from sleap_rtc.protocol_v1.identity import WorkerIdentity


class TestWorkerIdentity:
    """Tests for WorkerIdentity."""

    def test_generates_a_node_id_on_first_use(self, tmp_path):
        identity = WorkerIdentity(tmp_path / "identity.json")

        assert isinstance(identity.node_id, str)
        assert len(identity.node_id) > 0

    def test_persists_the_same_identity_across_instances(self, tmp_path):
        path = tmp_path / "identity.json"
        first = WorkerIdentity(path)
        node_id = first.node_id

        second = WorkerIdentity(path)

        assert second.node_id == node_id

    def test_different_paths_get_different_identities(self, tmp_path):
        first = WorkerIdentity(tmp_path / "a.json")
        second = WorkerIdentity(tmp_path / "b.json")

        assert first.node_id != second.node_id

    def test_sign_produces_a_verifiable_signature(self, tmp_path):
        identity = WorkerIdentity(tmp_path / "identity.json")
        nonce = "some-nonce-value"

        sig = identity.sign(nonce)

        public_key = public_key_from_b64(identity.node_id)
        assert verify_signature(public_key, nonce, sig) is True

    @pytest.mark.skipif(
        sys.platform == "win32", reason="POSIX permission bits don't apply on Windows"
    )
    def test_creates_the_file_with_restrictive_permissions(self, tmp_path):
        import stat

        path = tmp_path / "identity.json"
        WorkerIdentity(path)

        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode == stat.S_IRUSR | stat.S_IWUSR
