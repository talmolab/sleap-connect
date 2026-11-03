"""Unit tests for PSK authentication utilities."""

import base64

import pytest

from sleap_connect.auth.psk import (
    compute_hmac,
    generate_nonce,
    generate_secret,
    verify_hmac,
)


class TestGenerateSecret:
    """Tests for generate_secret()."""

    def test_returns_string(self):
        """Secret should be a string."""
        secret = generate_secret()
        assert isinstance(secret, str)

    def test_correct_length(self):
        """Secret should be 43 characters (256 bits, base64 no padding)."""
        secret = generate_secret()
        assert len(secret) == 43

    def test_is_valid_base64(self):
        """Secret should be valid URL-safe base64."""
        secret = generate_secret()
        # Add padding and decode - should not raise
        padded = secret + "=" * (4 - len(secret) % 4)
        decoded = base64.urlsafe_b64decode(padded)
        assert len(decoded) == 32  # 256 bits

    def test_generates_unique_values(self):
        """Each call should produce a unique secret."""
        secrets = [generate_secret() for _ in range(100)]
        assert len(set(secrets)) == 100  # All unique


class TestGenerateNonce:
    """Tests for generate_nonce()."""

    def test_returns_string(self):
        """Nonce should be a string."""
        nonce = generate_nonce()
        assert isinstance(nonce, str)

    def test_correct_length(self):
        """Nonce should be 43 characters (256 bits, base64 no padding)."""
        nonce = generate_nonce()
        assert len(nonce) == 43

    def test_is_valid_base64(self):
        """Nonce should be valid URL-safe base64."""
        nonce = generate_nonce()
        padded = nonce + "=" * (4 - len(nonce) % 4)
        decoded = base64.urlsafe_b64decode(padded)
        assert len(decoded) == 32

    def test_generates_unique_values(self):
        """Each call should produce a unique nonce."""
        nonces = [generate_nonce() for _ in range(100)]
        assert len(set(nonces)) == 100


class TestComputeHmac:
    """Tests for compute_hmac()."""

    def test_returns_string(self):
        """HMAC should be a string."""
        secret = generate_secret()
        nonce = generate_nonce()
        mac = compute_hmac(secret, nonce)
        assert isinstance(mac, str)

    def test_correct_length(self):
        """HMAC should be 43 characters (SHA256 = 256 bits, base64 no padding)."""
        secret = generate_secret()
        nonce = generate_nonce()
        mac = compute_hmac(secret, nonce)
        assert len(mac) == 43

    def test_deterministic(self):
        """Same inputs should produce same HMAC."""
        secret = generate_secret()
        nonce = generate_nonce()
        mac1 = compute_hmac(secret, nonce)
        mac2 = compute_hmac(secret, nonce)
        assert mac1 == mac2

    def test_different_secrets_produce_different_hmacs(self):
        """Different secrets should produce different HMACs."""
        secret1 = generate_secret()
        secret2 = generate_secret()
        nonce = generate_nonce()
        mac1 = compute_hmac(secret1, nonce)
        mac2 = compute_hmac(secret2, nonce)
        assert mac1 != mac2

    def test_different_nonces_produce_different_hmacs(self):
        """Different nonces should produce different HMACs."""
        secret = generate_secret()
        nonce1 = generate_nonce()
        nonce2 = generate_nonce()
        mac1 = compute_hmac(secret, nonce1)
        mac2 = compute_hmac(secret, nonce2)
        assert mac1 != mac2


class TestVerifyHmac:
    """Tests for verify_hmac()."""

    def test_valid_hmac_returns_true(self):
        """verify_hmac should return True for valid HMAC."""
        secret = generate_secret()
        nonce = generate_nonce()
        mac = compute_hmac(secret, nonce)
        assert verify_hmac(secret, nonce, mac) is True

    def test_invalid_hmac_returns_false(self):
        """verify_hmac should return False for invalid HMAC."""
        secret = generate_secret()
        nonce = generate_nonce()
        assert verify_hmac(secret, nonce, "invalid_hmac_value") is False

    def test_wrong_secret_returns_false(self):
        """verify_hmac should return False when secret doesn't match."""
        secret1 = generate_secret()
        secret2 = generate_secret()
        nonce = generate_nonce()
        mac = compute_hmac(secret1, nonce)
        assert verify_hmac(secret2, nonce, mac) is False

    def test_wrong_nonce_returns_false(self):
        """verify_hmac should return False when nonce doesn't match."""
        secret = generate_secret()
        nonce1 = generate_nonce()
        nonce2 = generate_nonce()
        mac = compute_hmac(secret, nonce1)
        assert verify_hmac(secret, nonce2, mac) is False

    def test_empty_hmac_returns_false(self):
        """verify_hmac should return False for empty HMAC."""
        secret = generate_secret()
        nonce = generate_nonce()
        assert verify_hmac(secret, nonce, "") is False


class TestKnownVectors:
    """Test against known test vectors for HMAC-SHA256."""

    def test_known_values(self):
        """Test with known secret/nonce to ensure HMAC computation is correct."""
        # Use fixed values for reproducibility
        # Secret: 32 bytes of 0x01
        secret_bytes = bytes([0x01] * 32)
        secret = base64.urlsafe_b64encode(secret_bytes).rstrip(b"=").decode()

        # Nonce: 32 bytes of 0x02
        nonce_bytes = bytes([0x02] * 32)
        nonce = base64.urlsafe_b64encode(nonce_bytes).rstrip(b"=").decode()

        # Compute HMAC
        mac = compute_hmac(secret, nonce)

        # Verify it's valid
        assert verify_hmac(secret, nonce, mac) is True

        # Verify wrong value fails
        assert verify_hmac(secret, nonce, mac + "x") is False
