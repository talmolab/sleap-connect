"""Tests for ECDH P-256 key exchange and AES-256-GCM encryption."""

import json

import pytest
from cryptography.exceptions import InvalidTag

from sleap_connect.encryption.ecdh import (
    decrypt,
    derive_shared_key,
    encrypt,
    generate_keypair,
    public_key_from_b64,
    public_key_to_b64,
)


class TestKeypairGeneration:
    def test_generate_keypair_returns_private_key_and_bytes(self):
        private_key, public_key_bytes = generate_keypair()
        assert private_key is not None
        assert isinstance(public_key_bytes, bytes)

    def test_public_key_is_65_bytes_uncompressed(self):
        _, public_key_bytes = generate_keypair()
        assert len(public_key_bytes) == 65
        assert public_key_bytes[0] == 0x04  # uncompressed point marker

    def test_each_keypair_is_unique(self):
        _, pub1 = generate_keypair()
        _, pub2 = generate_keypair()
        assert pub1 != pub2


class TestPublicKeySerialization:
    def test_roundtrip(self):
        _, public_key_bytes = generate_keypair()
        b64 = public_key_to_b64(public_key_bytes)
        restored = public_key_from_b64(b64)
        assert restored == public_key_bytes

    def test_b64_is_url_safe_no_padding(self):
        _, public_key_bytes = generate_keypair()
        b64 = public_key_to_b64(public_key_bytes)
        assert isinstance(b64, str)
        assert "+" not in b64
        assert "/" not in b64
        assert "=" not in b64

    def test_b64_is_consistent_length(self):
        # 65 bytes -> 87 base64 chars (without padding)
        _, public_key_bytes = generate_keypair()
        b64 = public_key_to_b64(public_key_bytes)
        assert len(b64) == 87


class TestKeyDerivation:
    def test_both_sides_derive_same_key(self):
        priv_a, pub_a = generate_keypair()
        priv_b, pub_b = generate_keypair()

        key_a = derive_shared_key(priv_a, pub_b)
        key_b = derive_shared_key(priv_b, pub_a)

        assert key_a == key_b

    def test_derived_key_is_32_bytes(self):
        priv_a, _ = generate_keypair()
        _, pub_b = generate_keypair()

        key = derive_shared_key(priv_a, pub_b)
        assert len(key) == 32

    def test_different_peers_produce_different_keys(self):
        priv_a, _ = generate_keypair()
        _, pub_b = generate_keypair()
        _, pub_c = generate_keypair()

        key_ab = derive_shared_key(priv_a, pub_b)
        key_ac = derive_shared_key(priv_a, pub_c)
        assert key_ab != key_ac

    def test_key_derivation_with_b64_roundtrip(self):
        """Public key survives base64 serialization for key exchange."""
        priv_a, pub_a = generate_keypair()
        priv_b, pub_b = generate_keypair()

        # Simulate sending public keys as base64 strings
        pub_a_restored = public_key_from_b64(public_key_to_b64(pub_a))
        pub_b_restored = public_key_from_b64(public_key_to_b64(pub_b))

        key_a = derive_shared_key(priv_a, pub_b_restored)
        key_b = derive_shared_key(priv_b, pub_a_restored)

        assert key_a == key_b


class TestEncryptDecrypt:
    def test_roundtrip(self):
        priv_a, pub_a = generate_keypair()
        priv_b, pub_b = generate_keypair()
        key = derive_shared_key(priv_a, pub_b)

        plaintext = {"type": "job_status", "epoch": 42, "loss": 0.003}
        nonce, ciphertext = encrypt(key, plaintext)
        decrypted = decrypt(key, nonce, ciphertext)

        assert decrypted == plaintext

    def test_ciphertext_differs_from_plaintext(self):
        priv, pub = generate_keypair()
        _, peer_pub = generate_keypair()
        key = derive_shared_key(priv, peer_pub)

        plaintext = {"secret": "data"}
        _, ciphertext = encrypt(key, plaintext)

        assert b"secret" not in ciphertext
        assert b"data" not in ciphertext

    def test_nonce_is_12_bytes(self):
        key = derive_shared_key(*_make_key_pair())
        nonce, _ = encrypt(key, {"test": True})
        assert len(nonce) == 12

    def test_each_encryption_produces_unique_nonce(self):
        key = derive_shared_key(*_make_key_pair())
        nonce1, _ = encrypt(key, {"test": True})
        nonce2, _ = encrypt(key, {"test": True})
        assert nonce1 != nonce2

    def test_decrypt_with_wrong_key_fails(self):
        key1 = derive_shared_key(*_make_key_pair())
        key2 = derive_shared_key(*_make_key_pair())

        nonce, ciphertext = encrypt(key1, {"secret": True})

        with pytest.raises(InvalidTag):
            decrypt(key2, nonce, ciphertext)

    def test_decrypt_with_tampered_ciphertext_fails(self):
        key = derive_shared_key(*_make_key_pair())
        nonce, ciphertext = encrypt(key, {"data": "value"})

        tampered = bytearray(ciphertext)
        tampered[0] ^= 0xFF
        tampered = bytes(tampered)

        with pytest.raises(InvalidTag):
            decrypt(key, nonce, tampered)

    def test_decrypt_with_wrong_nonce_fails(self):
        key = derive_shared_key(*_make_key_pair())
        nonce, ciphertext = encrypt(key, {"data": "value"})

        wrong_nonce = bytes(b ^ 0xFF for b in nonce)

        with pytest.raises(InvalidTag):
            decrypt(key, nonce=wrong_nonce, ciphertext=ciphertext)

    def test_bidirectional_encryption(self):
        """Both sides can encrypt and the other can decrypt."""
        priv_a, pub_a = generate_keypair()
        priv_b, pub_b = generate_keypair()

        key_a = derive_shared_key(priv_a, pub_b)
        key_b = derive_shared_key(priv_b, pub_a)

        # A encrypts, B decrypts
        msg1 = {"from": "client", "type": "fs_list_req", "path": "/data"}
        nonce1, ct1 = encrypt(key_a, msg1)
        assert decrypt(key_b, nonce1, ct1) == msg1

        # B encrypts, A decrypts
        msg2 = {"from": "worker", "type": "fs_list_res", "entries": []}
        nonce2, ct2 = encrypt(key_b, msg2)
        assert decrypt(key_a, nonce2, ct2) == msg2

    def test_overhead_is_16_bytes(self):
        """AES-GCM adds exactly 16 bytes (auth tag) to ciphertext."""
        key = derive_shared_key(*_make_key_pair())
        plaintext = {"x": 1}
        plaintext_bytes = json.dumps(plaintext, separators=(",", ":")).encode()
        _, ciphertext = encrypt(key, plaintext)
        assert len(ciphertext) == len(plaintext_bytes) + 16


def _make_key_pair() -> tuple:
    """Helper: generate keypair and return (private_key, peer_public_key_bytes)."""
    priv, _ = generate_keypair()
    _, peer_pub = generate_keypair()
    return priv, peer_pub
