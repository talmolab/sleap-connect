"""Tests for Ed25519 keypair utilities."""

import pytest

from sleap_connect.auth.keypair import (
    generate_keypair,
    sign_nonce,
    verify_signature,
    private_key_to_b64,
    public_key_to_b64,
    private_key_from_b64,
    public_key_from_b64,
)


class TestKeypair:
    def test_generate_keypair_returns_two_keys(self):
        private_key, public_key = generate_keypair()
        assert private_key is not None
        assert public_key is not None

    def test_sign_and_verify_roundtrip(self):
        private_key, public_key = generate_keypair()
        nonce = "test_nonce_abcdef"
        signature = sign_nonce(private_key, nonce)
        assert verify_signature(public_key, nonce, signature) is True

    def test_verify_fails_with_wrong_nonce(self):
        private_key, public_key = generate_keypair()
        nonce = "correct_nonce"
        signature = sign_nonce(private_key, nonce)
        assert verify_signature(public_key, "wrong_nonce", signature) is False

    def test_verify_fails_with_wrong_key(self):
        private_key, public_key = generate_keypair()
        _, other_public_key = generate_keypair()
        nonce = "test_nonce"
        signature = sign_nonce(private_key, nonce)
        assert verify_signature(other_public_key, nonce, signature) is False

    def test_serialization_roundtrip(self):
        private_key, public_key = generate_keypair()
        priv_b64 = private_key_to_b64(private_key)
        pub_b64 = public_key_to_b64(public_key)

        restored_private = private_key_from_b64(priv_b64)
        restored_public = public_key_from_b64(pub_b64)

        nonce = "roundtrip_nonce"
        sig = sign_nonce(restored_private, nonce)
        assert verify_signature(restored_public, nonce, sig) is True

    def test_signature_is_string(self):
        private_key, _ = generate_keypair()
        sig = sign_nonce(private_key, "nonce")
        assert isinstance(sig, str)

    def test_b64_strings_are_url_safe(self):
        private_key, public_key = generate_keypair()
        priv_b64 = private_key_to_b64(private_key)
        pub_b64 = public_key_to_b64(public_key)
        # URL-safe base64 uses - and _ instead of + and /
        assert "+" not in priv_b64
        assert "/" not in priv_b64
        assert "+" not in pub_b64
        assert "/" not in pub_b64
        # No padding
        assert "=" not in priv_b64
        assert "=" not in pub_b64
