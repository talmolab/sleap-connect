"""Unit tests for credential storage."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from sleap_connect.auth import credentials


@pytest.fixture
def temp_credentials_dir(tmp_path):
    """Create a temporary credentials directory for testing."""
    creds_dir = tmp_path / ".sleap-connect"
    creds_dir.mkdir()
    creds_path = creds_dir / "credentials.json"

    # Patch the module-level paths
    with mock.patch.object(credentials, "CREDENTIALS_DIR", creds_dir):
        with mock.patch.object(credentials, "CREDENTIALS_PATH", creds_path):
            yield creds_path


class TestGetRoomSecret:
    """Tests for get_room_secret()."""

    def test_returns_none_when_no_credentials_file(self, temp_credentials_dir):
        """Should return None if credentials file doesn't exist."""
        result = credentials.get_room_secret("room-123")
        assert result is None

    def test_returns_none_when_no_room_secrets_key(self, temp_credentials_dir):
        """Should return None if room_secrets key doesn't exist."""
        temp_credentials_dir.write_text(json.dumps({"jwt": "token"}))
        result = credentials.get_room_secret("room-123")
        assert result is None

    def test_returns_none_when_room_not_found(self, temp_credentials_dir):
        """Should return None if specific room not in room_secrets."""
        temp_credentials_dir.write_text(
            json.dumps({"room_secrets": {"other-room": "secret123"}})
        )
        result = credentials.get_room_secret("room-123")
        assert result is None

    def test_returns_secret_when_found(self, temp_credentials_dir):
        """Should return the secret when room exists."""
        secret = "my_base64_secret_value"
        temp_credentials_dir.write_text(
            json.dumps({"room_secrets": {"room-123": secret}})
        )
        result = credentials.get_room_secret("room-123")
        assert result == secret


class TestSaveRoomSecret:
    """Tests for save_room_secret()."""

    def test_creates_room_secrets_key_if_missing(self, temp_credentials_dir):
        """Should create room_secrets key if it doesn't exist."""
        # Start with credentials that don't have room_secrets
        temp_credentials_dir.write_text(json.dumps({"jwt": "token"}))

        credentials.save_room_secret("room-123", "secret123")

        data = json.loads(temp_credentials_dir.read_text())
        assert "room_secrets" in data
        assert data["room_secrets"]["room-123"] == "secret123"

    def test_preserves_existing_credentials(self, temp_credentials_dir):
        """Should preserve existing JWT and tokens when saving secret."""
        temp_credentials_dir.write_text(
            json.dumps(
                {
                    "jwt": "my_jwt",
                    "user": {"username": "test"},
                    "tokens": {"room-1": {"api_key": "slp_xxx"}},
                }
            )
        )

        credentials.save_room_secret("room-123", "secret123")

        data = json.loads(temp_credentials_dir.read_text())
        assert data["jwt"] == "my_jwt"
        assert data["user"]["username"] == "test"
        assert data["tokens"]["room-1"]["api_key"] == "slp_xxx"
        assert data["room_secrets"]["room-123"] == "secret123"

    def test_overwrites_existing_secret(self, temp_credentials_dir):
        """Should overwrite existing secret for same room."""
        temp_credentials_dir.write_text(
            json.dumps({"room_secrets": {"room-123": "old_secret"}})
        )

        credentials.save_room_secret("room-123", "new_secret")

        data = json.loads(temp_credentials_dir.read_text())
        assert data["room_secrets"]["room-123"] == "new_secret"

    def test_preserves_other_room_secrets(self, temp_credentials_dir):
        """Should preserve secrets for other rooms."""
        temp_credentials_dir.write_text(
            json.dumps({"room_secrets": {"room-1": "secret1", "room-2": "secret2"}})
        )

        credentials.save_room_secret("room-3", "secret3")

        data = json.loads(temp_credentials_dir.read_text())
        assert data["room_secrets"]["room-1"] == "secret1"
        assert data["room_secrets"]["room-2"] == "secret2"
        assert data["room_secrets"]["room-3"] == "secret3"

    @pytest.mark.skipif(
        sys.platform == "win32", reason="Unix permission bits not supported on Windows"
    )
    def test_sets_restrictive_permissions(self, temp_credentials_dir):
        """Should set file permissions to 600 (owner read/write only)."""
        credentials.save_room_secret("room-123", "secret123")

        mode = temp_credentials_dir.stat().st_mode
        # Check only owner read/write bits are set (0o600)
        assert mode & 0o777 == 0o600


class TestRemoveRoomSecret:
    """Tests for remove_room_secret()."""

    def test_returns_false_when_no_credentials_file(self, temp_credentials_dir):
        """Should return False if credentials file doesn't exist."""
        result = credentials.remove_room_secret("room-123")
        assert result is False

    def test_returns_false_when_no_room_secrets_key(self, temp_credentials_dir):
        """Should return False if room_secrets key doesn't exist."""
        temp_credentials_dir.write_text(json.dumps({"jwt": "token"}))
        result = credentials.remove_room_secret("room-123")
        assert result is False

    def test_returns_false_when_room_not_found(self, temp_credentials_dir):
        """Should return False if specific room not in room_secrets."""
        temp_credentials_dir.write_text(
            json.dumps({"room_secrets": {"other-room": "secret123"}})
        )
        result = credentials.remove_room_secret("room-123")
        assert result is False

    def test_returns_true_and_removes_secret(self, temp_credentials_dir):
        """Should return True and remove the secret when found."""
        temp_credentials_dir.write_text(
            json.dumps({"room_secrets": {"room-123": "secret123"}})
        )

        result = credentials.remove_room_secret("room-123")

        assert result is True
        data = json.loads(temp_credentials_dir.read_text())
        assert "room-123" not in data["room_secrets"]

    def test_preserves_other_room_secrets(self, temp_credentials_dir):
        """Should preserve secrets for other rooms when removing one."""
        temp_credentials_dir.write_text(
            json.dumps({"room_secrets": {"room-1": "secret1", "room-2": "secret2"}})
        )

        credentials.remove_room_secret("room-1")

        data = json.loads(temp_credentials_dir.read_text())
        assert "room-1" not in data["room_secrets"]
        assert data["room_secrets"]["room-2"] == "secret2"

    def test_preserves_other_credentials(self, temp_credentials_dir):
        """Should preserve JWT and tokens when removing secret."""
        temp_credentials_dir.write_text(
            json.dumps(
                {
                    "jwt": "my_jwt",
                    "tokens": {"room-1": {"api_key": "slp_xxx"}},
                    "room_secrets": {"room-123": "secret123"},
                }
            )
        )

        credentials.remove_room_secret("room-123")

        data = json.loads(temp_credentials_dir.read_text())
        assert data["jwt"] == "my_jwt"
        assert data["tokens"]["room-1"]["api_key"] == "slp_xxx"


class TestRoomSecretIntegration:
    """Integration tests for room secret workflow."""

    def test_save_get_remove_cycle(self, temp_credentials_dir):
        """Test complete save -> get -> remove cycle."""
        room_id = "integration-room"
        secret = "integration_secret_value"

        # Initially no secret
        assert credentials.get_room_secret(room_id) is None

        # Save secret
        credentials.save_room_secret(room_id, secret)
        assert credentials.get_room_secret(room_id) == secret

        # Remove secret
        assert credentials.remove_room_secret(room_id) is True
        assert credentials.get_room_secret(room_id) is None

        # Remove again returns False
        assert credentials.remove_room_secret(room_id) is False

    def test_multiple_rooms(self, temp_credentials_dir):
        """Test managing secrets for multiple rooms."""
        credentials.save_room_secret("room-1", "secret-1")
        credentials.save_room_secret("room-2", "secret-2")
        credentials.save_room_secret("room-3", "secret-3")

        assert credentials.get_room_secret("room-1") == "secret-1"
        assert credentials.get_room_secret("room-2") == "secret-2"
        assert credentials.get_room_secret("room-3") == "secret-3"

        # Remove middle one
        credentials.remove_room_secret("room-2")

        assert credentials.get_room_secret("room-1") == "secret-1"
        assert credentials.get_room_secret("room-2") is None
        assert credentials.get_room_secret("room-3") == "secret-3"


class TestAccountKey:
    """Tests for get_account_key() and save_account_key()."""

    def test_returns_none_when_absent(self, temp_credentials_dir):
        result = credentials.get_account_key()
        assert result is None

    def test_save_and_get_roundtrip(self, temp_credentials_dir):
        credentials.save_account_key("slp_acct_testkey123")
        assert credentials.get_account_key() == "slp_acct_testkey123"

    def test_env_var_overrides_stored_key(self, temp_credentials_dir, monkeypatch):
        credentials.save_account_key("slp_acct_stored")
        monkeypatch.setenv("SLEAP_RTC_ACCOUNT_KEY", "slp_acct_envkey")
        assert credentials.get_account_key() == "slp_acct_envkey"

    def test_env_var_with_no_file(self, temp_credentials_dir, monkeypatch):
        monkeypatch.setenv("SLEAP_RTC_ACCOUNT_KEY", "slp_acct_envonly")
        assert credentials.get_account_key() == "slp_acct_envonly"

    def test_preserves_other_credentials(self, temp_credentials_dir):
        temp_credentials_dir.write_text(json.dumps({"jwt": "my_jwt"}))
        credentials.save_account_key("slp_acct_new")
        data = json.loads(temp_credentials_dir.read_text())
        assert data["jwt"] == "my_jwt"
        assert data["account_key"] == "slp_acct_new"


class TestDefaultRoom:
    """Tests for get_default_room() and save_default_room()."""

    def test_returns_none_when_absent(self, temp_credentials_dir):
        assert credentials.get_default_room() is None

    def test_save_and_get_roundtrip(self, temp_credentials_dir):
        credentials.save_default_room("room-abc123")
        assert credentials.get_default_room() == "room-abc123"

    def test_env_var_overrides_stored_room(self, temp_credentials_dir, monkeypatch):
        credentials.save_default_room("room-stored")
        monkeypatch.setenv("SLEAP_RTC_DEFAULT_ROOM", "room-from-env")
        assert credentials.get_default_room() == "room-from-env"

    def test_env_var_with_no_file(self, temp_credentials_dir, monkeypatch):
        monkeypatch.setenv("SLEAP_RTC_DEFAULT_ROOM", "room-envonly")
        assert credentials.get_default_room() == "room-envonly"


class TestPrivateKeyB64:
    """Tests for get_private_key_b64() and save_private_key_b64()."""

    def test_returns_none_when_absent(self, temp_credentials_dir):
        assert credentials.get_private_key_b64() is None

    def test_save_and_get_roundtrip(self, temp_credentials_dir):
        credentials.save_private_key_b64("fake_b64_private_key")
        assert credentials.get_private_key_b64() == "fake_b64_private_key"

    def test_preserves_other_credentials(self, temp_credentials_dir):
        temp_credentials_dir.write_text(json.dumps({"account_key": "slp_acct_x"}))
        credentials.save_private_key_b64("priv_key_b64")
        data = json.loads(temp_credentials_dir.read_text())
        assert data["account_key"] == "slp_acct_x"
        assert data["private_key"] == "priv_key_b64"


class TestPublicKeyRegistered:
    """Tests for get_public_key_registered() and set_public_key_registered()."""

    def test_defaults_to_false_when_absent(self, temp_credentials_dir):
        assert credentials.get_public_key_registered() is False

    def test_defaults_to_false_when_no_file(self, temp_credentials_dir):
        assert credentials.get_public_key_registered() is False

    def test_set_true_and_get(self, temp_credentials_dir):
        credentials.set_public_key_registered(True)
        assert credentials.get_public_key_registered() is True

    def test_set_false_and_get(self, temp_credentials_dir):
        credentials.set_public_key_registered(True)
        credentials.set_public_key_registered(False)
        assert credentials.get_public_key_registered() is False

    def test_preserves_other_credentials(self, temp_credentials_dir):
        temp_credentials_dir.write_text(json.dumps({"jwt": "my_jwt"}))
        credentials.set_public_key_registered(True)
        data = json.loads(temp_credentials_dir.read_text())
        assert data["jwt"] == "my_jwt"
        assert data["public_key_registered"] is True

    def test_save_private_key_resets_flag(self, temp_credentials_dir):
        credentials.set_public_key_registered(True)
        assert credentials.get_public_key_registered() is True
        credentials.save_private_key_b64("new_key_b64")
        assert credentials.get_public_key_registered() is False
