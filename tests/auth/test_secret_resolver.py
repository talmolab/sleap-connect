"""Unit tests for secret resolution."""

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from sleap_connect.auth import credentials
from sleap_connect.auth import secret_resolver
from sleap_connect.auth.secret_resolver import (
    ENV_ROOM_SECRET,
    ENV_SECRET_PATH,
    get_secret_base_path,
    get_secret_sources,
    resolve_secret,
)


@pytest.fixture
def temp_credentials_dir(tmp_path):
    """Create a temporary credentials directory for testing."""
    creds_dir = tmp_path / ".sleap-connect"
    creds_dir.mkdir()
    creds_path = creds_dir / "credentials.json"

    with mock.patch.object(credentials, "CREDENTIALS_DIR", creds_dir):
        with mock.patch.object(credentials, "CREDENTIALS_PATH", creds_path):
            yield creds_path


@pytest.fixture
def temp_secrets_dir(tmp_path):
    """Create a temporary filesystem secrets directory."""
    secrets_dir = tmp_path / "room-secrets"
    secrets_dir.mkdir()
    return secrets_dir


@pytest.fixture
def clean_env():
    """Ensure environment variables are clean before/after test."""
    old_room_secret = os.environ.pop(ENV_ROOM_SECRET, None)
    old_secret_path = os.environ.pop(ENV_SECRET_PATH, None)
    yield
    # Restore
    if old_room_secret:
        os.environ[ENV_ROOM_SECRET] = old_room_secret
    if old_secret_path:
        os.environ[ENV_SECRET_PATH] = old_secret_path


class TestResolveSecretPriority:
    """Tests for resolve_secret() priority order."""

    def test_cli_flag_takes_priority(self, clean_env, temp_credentials_dir, temp_secrets_dir):
        """CLI flag should override all other sources."""
        room_id = "room-123"
        cli_secret = "cli_secret_value"

        # Set up all other sources with different values
        os.environ[ENV_ROOM_SECRET] = "env_secret"
        os.environ[ENV_SECRET_PATH] = str(temp_secrets_dir)
        (temp_secrets_dir / room_id).write_text("fs_secret")
        temp_credentials_dir.write_text(
            json.dumps({"room_secrets": {room_id: "cred_secret"}})
        )

        result = resolve_secret(room_id, cli_secret=cli_secret)
        assert result == cli_secret

    def test_env_var_second_priority(self, clean_env, temp_credentials_dir, temp_secrets_dir):
        """Env var should be used when CLI flag not provided."""
        room_id = "room-123"
        env_secret = "env_secret_value"

        # Set up env var and lower-priority sources
        os.environ[ENV_ROOM_SECRET] = env_secret
        os.environ[ENV_SECRET_PATH] = str(temp_secrets_dir)
        (temp_secrets_dir / room_id).write_text("fs_secret")
        temp_credentials_dir.write_text(
            json.dumps({"room_secrets": {room_id: "cred_secret"}})
        )

        result = resolve_secret(room_id)
        assert result == env_secret

    def test_filesystem_third_priority(self, clean_env, temp_credentials_dir, temp_secrets_dir):
        """Filesystem should be used when CLI and env var not provided."""
        room_id = "room-123"
        fs_secret = "fs_secret_value"

        # Set up filesystem and credentials (no CLI or env var)
        os.environ[ENV_SECRET_PATH] = str(temp_secrets_dir)
        (temp_secrets_dir / room_id).write_text(fs_secret)
        temp_credentials_dir.write_text(
            json.dumps({"room_secrets": {room_id: "cred_secret"}})
        )

        result = resolve_secret(room_id)
        assert result == fs_secret

    def test_credentials_fourth_priority(self, clean_env, temp_credentials_dir, temp_secrets_dir):
        """Credentials file should be used as last resort."""
        room_id = "room-123"
        cred_secret = "cred_secret_value"

        # Only credentials available (filesystem dir exists but no file for this room)
        os.environ[ENV_SECRET_PATH] = str(temp_secrets_dir)
        temp_credentials_dir.write_text(
            json.dumps({"room_secrets": {room_id: cred_secret}})
        )

        result = resolve_secret(room_id)
        assert result == cred_secret

    def test_returns_none_when_no_source_has_secret(self, clean_env, temp_credentials_dir, temp_secrets_dir):
        """Should return None when no source has the secret."""
        os.environ[ENV_SECRET_PATH] = str(temp_secrets_dir)
        # Empty credentials file
        temp_credentials_dir.write_text(json.dumps({}))

        result = resolve_secret("nonexistent-room")
        assert result is None


class TestFilesystemSecret:
    """Tests for filesystem-based secret resolution."""

    def test_reads_secret_from_file(self, clean_env, temp_secrets_dir):
        """Should read secret from {base_path}/{room_id} file."""
        room_id = "room-123"
        secret = "my_filesystem_secret"

        os.environ[ENV_SECRET_PATH] = str(temp_secrets_dir)
        (temp_secrets_dir / room_id).write_text(secret)

        result = resolve_secret(room_id)
        assert result == secret

    def test_strips_whitespace_from_file(self, clean_env, temp_secrets_dir):
        """Should strip whitespace from secret file content."""
        room_id = "room-123"
        secret = "secret_with_whitespace"

        os.environ[ENV_SECRET_PATH] = str(temp_secrets_dir)
        (temp_secrets_dir / room_id).write_text(f"  {secret}  \n")

        result = resolve_secret(room_id)
        assert result == secret

    def test_returns_none_for_empty_file(self, clean_env, temp_secrets_dir):
        """Should return None if secret file is empty."""
        room_id = "room-123"

        os.environ[ENV_SECRET_PATH] = str(temp_secrets_dir)
        (temp_secrets_dir / room_id).write_text("   \n")

        result = resolve_secret(room_id)
        assert result is None

    def test_returns_none_for_missing_file(self, clean_env, temp_secrets_dir):
        """Should return None if secret file doesn't exist."""
        os.environ[ENV_SECRET_PATH] = str(temp_secrets_dir)

        result = resolve_secret("nonexistent-room")
        assert result is None


class TestSecretBasePath:
    """Tests for get_secret_base_path()."""

    def test_returns_env_var_path_when_set(self, clean_env, tmp_path):
        """Should return SLEAP_SECRET_PATH when set."""
        custom_path = tmp_path / "custom-secrets"
        os.environ[ENV_SECRET_PATH] = str(custom_path)

        result = get_secret_base_path()
        assert result == custom_path

    def test_expands_tilde_in_env_var(self, clean_env):
        """Should expand ~ in SLEAP_SECRET_PATH."""
        os.environ[ENV_SECRET_PATH] = "~/my-secrets"

        result = get_secret_base_path()
        assert result == Path.home() / "my-secrets"

    def test_returns_default_when_env_not_set(self, clean_env):
        """Should return default path when SLEAP_SECRET_PATH not set."""
        result = get_secret_base_path()
        assert result == Path.home() / ".sleap-connect" / "room-secrets"


class TestGetSecretSources:
    """Tests for get_secret_sources() diagnostic function."""

    def test_returns_all_sources(self, clean_env, temp_credentials_dir, temp_secrets_dir):
        """Should return dict with all source values."""
        room_id = "room-123"

        os.environ[ENV_ROOM_SECRET] = "env_secret"
        os.environ[ENV_SECRET_PATH] = str(temp_secrets_dir)
        (temp_secrets_dir / room_id).write_text("fs_secret")
        temp_credentials_dir.write_text(
            json.dumps({"room_secrets": {room_id: "cred_secret"}})
        )

        result = get_secret_sources(room_id, cli_secret="cli_secret")

        assert result["cli_flag"] == "cli_secret"
        assert result["env_var"] == "env_secret"
        assert result["filesystem"] == str(temp_secrets_dir / room_id)
        assert result["credentials"] == "cred_secret"

    def test_returns_none_for_missing_sources(self, clean_env, temp_credentials_dir, temp_secrets_dir):
        """Should return None for sources that don't have secrets."""
        room_id = "room-123"
        os.environ[ENV_SECRET_PATH] = str(temp_secrets_dir)
        temp_credentials_dir.write_text(json.dumps({}))

        result = get_secret_sources(room_id)

        assert result["cli_flag"] is None
        assert result["env_var"] is None
        assert result["filesystem"] is None
        assert result["credentials"] is None


class TestIntegration:
    """Integration tests for secret resolution."""

    def test_full_fallback_chain(self, clean_env, temp_credentials_dir, temp_secrets_dir):
        """Test falling through the entire priority chain."""
        room_id = "room-123"
        os.environ[ENV_SECRET_PATH] = str(temp_secrets_dir)

        # Start with all sources empty
        temp_credentials_dir.write_text(json.dumps({}))
        assert resolve_secret(room_id) is None

        # Add credentials secret
        temp_credentials_dir.write_text(
            json.dumps({"room_secrets": {room_id: "cred_secret"}})
        )
        assert resolve_secret(room_id) == "cred_secret"

        # Add filesystem secret (overrides credentials)
        (temp_secrets_dir / room_id).write_text("fs_secret")
        assert resolve_secret(room_id) == "fs_secret"

        # Add env var (overrides filesystem)
        os.environ[ENV_ROOM_SECRET] = "env_secret"
        assert resolve_secret(room_id) == "env_secret"

        # CLI flag (overrides everything)
        assert resolve_secret(room_id, cli_secret="cli_secret") == "cli_secret"

    def test_different_rooms_different_secrets(self, clean_env, temp_credentials_dir, temp_secrets_dir):
        """Different rooms can have different secrets from different sources."""
        os.environ[ENV_SECRET_PATH] = str(temp_secrets_dir)

        # Room 1: filesystem
        (temp_secrets_dir / "room-1").write_text("room1_fs_secret")

        # Room 2: credentials
        temp_credentials_dir.write_text(
            json.dumps({"room_secrets": {"room-2": "room2_cred_secret"}})
        )

        assert resolve_secret("room-1") == "room1_fs_secret"
        assert resolve_secret("room-2") == "room2_cred_secret"
        assert resolve_secret("room-3") is None
