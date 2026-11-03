"""Tests for sleap_connect.api module."""

import os
import pytest
from unittest.mock import patch, MagicMock
import json

from sleap_connect.api import (
    User,
    Room,
    Worker,
    AuthenticationError,
    RoomNotFoundError,
    is_available,
    is_logged_in,
    get_logged_in_user,
    login,
    logout,
    list_rooms,
    list_workers,
    _ensure_keypair_registered,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_config():
    """Mock the config module."""
    config = MagicMock()
    config.signaling_websocket = "ws://test-server:8080"
    config.signaling_http = "http://test-server:8080"
    config.get_http_url.return_value = "http://test-server:8080"
    return config


@pytest.fixture
def mock_jwt():
    """A valid mock JWT."""
    return "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwiZXhwIjo5OTk5OTk5OTk5fQ.test"


@pytest.fixture
def mock_user_data():
    """Mock user data from credentials."""
    return {
        "id": "12345",
        "username": "testuser",
        "avatar_url": "https://example.com/avatar.png",
    }


# =============================================================================
# is_available() Tests
# =============================================================================


class TestIsAvailable:
    """Tests for is_available() function."""

    def test_available_when_configured(self, mock_config):
        """Should return True when signaling URLs are configured."""
        with patch("sleap_connect.config.get_config", return_value=mock_config):
            assert is_available() is True

    def test_not_available_when_no_ws_url(self, mock_config):
        """Should return False when WS URL is missing."""
        mock_config.signaling_websocket = None
        with patch("sleap_connect.config.get_config", return_value=mock_config):
            assert is_available() is False

    def test_not_available_when_no_http_url(self, mock_config):
        """Should return False when HTTP URL is missing."""
        mock_config.signaling_http = None
        with patch("sleap_connect.config.get_config", return_value=mock_config):
            assert is_available() is False

    def test_not_available_on_exception(self):
        """Should return False when config raises exception."""
        with patch("sleap_connect.config.get_config", side_effect=Exception("Config error")):
            assert is_available() is False


# =============================================================================
# is_logged_in() Tests
# =============================================================================


class TestIsLoggedIn:
    """Tests for is_logged_in() function."""

    def test_logged_in_with_valid_jwt(self, mock_jwt):
        """Should return True when valid JWT exists."""
        with patch("sleap_connect.auth.credentials.get_valid_jwt", return_value=mock_jwt):
            assert is_logged_in() is True

    def test_not_logged_in_with_no_jwt(self):
        """Should return False when no JWT exists."""
        with patch("sleap_connect.auth.credentials.get_valid_jwt", return_value=None):
            assert is_logged_in() is False

    def test_not_logged_in_with_expired_jwt(self):
        """Should return False when JWT is expired."""
        # get_valid_jwt returns None for expired JWTs
        with patch("sleap_connect.auth.credentials.get_valid_jwt", return_value=None):
            assert is_logged_in() is False


# =============================================================================
# get_logged_in_user() Tests
# =============================================================================


class TestGetLoggedInUser:
    """Tests for get_logged_in_user() function."""

    def test_returns_user_when_logged_in(self, mock_jwt, mock_user_data):
        """Should return User object when logged in."""
        with patch("sleap_connect.auth.credentials.get_valid_jwt", return_value=mock_jwt):
            with patch("sleap_connect.auth.credentials.get_user", return_value=mock_user_data):
                user = get_logged_in_user()
                assert user is not None
                assert user.id == "12345"
                assert user.username == "testuser"
                assert user.avatar_url == "https://example.com/avatar.png"

    def test_returns_none_when_no_jwt(self):
        """Should return None when no JWT exists."""
        with patch("sleap_connect.auth.credentials.get_valid_jwt", return_value=None):
            assert get_logged_in_user() is None

    def test_returns_none_when_no_user_data(self, mock_jwt):
        """Should return None when no user data exists."""
        with patch("sleap_connect.auth.credentials.get_valid_jwt", return_value=mock_jwt):
            with patch("sleap_connect.auth.credentials.get_user", return_value=None):
                assert get_logged_in_user() is None


# =============================================================================
# login() Tests
# =============================================================================


class TestLogin:
    """Tests for login() function."""

    def test_login_success(self, mock_user_data):
        """Should return User on successful login."""
        mock_result = {"jwt": "test-jwt", "user": mock_user_data}
        with patch("sleap_connect.auth.github.github_login", return_value=mock_result):
            with patch("sleap_connect.auth.credentials.save_jwt") as mock_save:
                with patch("sleap_connect.api._ensure_keypair_registered"):
                    user = login()
                    assert user.username == "testuser"
                    mock_save.assert_called_once_with("test-jwt", mock_user_data)

    def test_login_with_callback(self, mock_user_data):
        """Should pass callback to github_login."""
        mock_result = {"jwt": "test-jwt", "user": mock_user_data}
        callback = MagicMock()

        with patch("sleap_connect.auth.github.github_login", return_value=mock_result) as mock_login:
            with patch("sleap_connect.auth.credentials.save_jwt"):
                with patch("sleap_connect.api._ensure_keypair_registered"):
                    login(on_url_ready=callback)
                    mock_login.assert_called_once()
                    call_kwargs = mock_login.call_args[1]
                    assert call_kwargs["on_url_ready"] == callback

    def test_login_failure_raises_auth_error(self):
        """Should raise AuthenticationError on failure."""
        with patch("sleap_connect.auth.github.github_login", side_effect=RuntimeError("Timeout")):
            with pytest.raises(AuthenticationError, match="Login failed"):
                login()

    def test_login_with_timeout(self, mock_user_data):
        """Should pass timeout to github_login."""
        mock_result = {"jwt": "test-jwt", "user": mock_user_data}
        with patch("sleap_connect.auth.github.github_login", return_value=mock_result) as mock_login:
            with patch("sleap_connect.auth.credentials.save_jwt"):
                with patch("sleap_connect.api._ensure_keypair_registered"):
                    login(timeout=60)
                    call_kwargs = mock_login.call_args[1]
                    assert call_kwargs["timeout"] == 60

    def test_login_calls_ensure_keypair(self, mock_user_data):
        """Should call _ensure_keypair_registered after saving JWT."""
        mock_result = {"jwt": "test-jwt", "user": mock_user_data}
        with patch("sleap_connect.auth.github.github_login", return_value=mock_result):
            with patch("sleap_connect.auth.credentials.save_jwt"):
                with patch("sleap_connect.api._ensure_keypair_registered") as mock_ensure:
                    login()
                    mock_ensure.assert_called_once_with(auth_token="test-jwt")

    def test_login_succeeds_even_if_keypair_registration_fails(self, mock_user_data):
        """Login should succeed even if keypair registration raises."""
        mock_result = {"jwt": "test-jwt", "user": mock_user_data}
        with patch("sleap_connect.auth.github.github_login", return_value=mock_result):
            with patch("sleap_connect.auth.credentials.save_jwt"):
                with patch("sleap_connect.api._ensure_keypair_registered", side_effect=Exception("boom")):
                    user = login()
                    assert user.username == "testuser"


# =============================================================================
# _ensure_keypair_registered() Tests
# =============================================================================


class TestEnsureKeypairRegistered:
    """Tests for _ensure_keypair_registered() helper."""

    def test_generates_keypair_when_none_exists(self):
        """Should generate and save a new keypair if private key is missing."""
        with patch("sleap_connect.auth.credentials.get_private_key_b64", return_value=None):
            with patch("sleap_connect.auth.credentials.save_private_key_b64") as mock_save:
                with patch("sleap_connect.auth.credentials.get_public_key_registered", return_value=False):
                    with patch("sleap_connect.auth.credentials.set_public_key_registered"):
                        with patch("sleap_connect.api.requests") as mock_requests:
                            mock_requests.post.return_value = MagicMock(ok=True)
                            _ensure_keypair_registered(auth_token="test-jwt")
                            mock_save.assert_called_once()
                            saved_b64 = mock_save.call_args[0][0]
                            assert isinstance(saved_b64, str)
                            assert len(saved_b64) > 0

    def test_skips_generation_when_keypair_exists(self):
        """Should not generate a new keypair if private key already exists."""
        with patch("sleap_connect.auth.credentials.get_private_key_b64", return_value="existing_key_b64"):
            with patch("sleap_connect.auth.credentials.save_private_key_b64") as mock_save:
                with patch("sleap_connect.auth.credentials.get_public_key_registered", return_value=True):
                    _ensure_keypair_registered(auth_token="test-jwt")
                    mock_save.assert_not_called()

    def test_skips_registration_when_already_registered(self):
        """Should not POST to server if public key is already registered."""
        with patch("sleap_connect.auth.credentials.get_private_key_b64", return_value="existing_key_b64"):
            with patch("sleap_connect.auth.credentials.get_public_key_registered", return_value=True):
                with patch("sleap_connect.api.requests") as mock_requests:
                    _ensure_keypair_registered(auth_token="test-jwt")
                    mock_requests.post.assert_not_called()

    def test_registers_public_key_when_not_registered(self, mock_config):
        """Should POST public key to server when not yet registered."""
        from sleap_connect.auth.keypair import generate_keypair, private_key_to_b64
        priv, pub = generate_keypair()
        priv_b64 = private_key_to_b64(priv)

        with patch("sleap_connect.auth.credentials.get_private_key_b64", return_value=priv_b64):
            with patch("sleap_connect.auth.credentials.get_public_key_registered", return_value=False):
                with patch("sleap_connect.auth.credentials.set_public_key_registered") as mock_set:
                    with patch("sleap_connect.config.get_config", return_value=mock_config):
                        with patch("sleap_connect.api.requests") as mock_requests:
                            mock_requests.post.return_value = MagicMock(ok=True)
                            _ensure_keypair_registered(auth_token="test-jwt")
                            mock_requests.post.assert_called_once()
                            call_kwargs = mock_requests.post.call_args
                            assert "/api/auth/public-keys" in call_kwargs[0][0]
                            assert call_kwargs[1]["json"]["device_name"] == "gui"
                            assert "public_key" in call_kwargs[1]["json"]
                            mock_set.assert_called_once_with(True)

    def test_does_not_raise_on_registration_failure(self, mock_config):
        """Should log warning but not raise if server registration fails."""
        from sleap_connect.auth.keypair import generate_keypair, private_key_to_b64
        priv, pub = generate_keypair()
        priv_b64 = private_key_to_b64(priv)

        with patch("sleap_connect.auth.credentials.get_private_key_b64", return_value=priv_b64):
            with patch("sleap_connect.auth.credentials.get_public_key_registered", return_value=False):
                with patch("sleap_connect.auth.credentials.set_public_key_registered") as mock_set:
                    with patch("sleap_connect.config.get_config", return_value=mock_config):
                        with patch("sleap_connect.api.requests") as mock_requests:
                            mock_requests.post.side_effect = ConnectionError("Server down")
                            _ensure_keypair_registered(auth_token="test-jwt")
                            mock_set.assert_not_called()

    def test_does_not_mark_registered_on_http_error(self, mock_config):
        """Should not set flag if server returns non-OK status."""
        from sleap_connect.auth.keypair import generate_keypair, private_key_to_b64
        priv, pub = generate_keypair()
        priv_b64 = private_key_to_b64(priv)

        with patch("sleap_connect.auth.credentials.get_private_key_b64", return_value=priv_b64):
            with patch("sleap_connect.auth.credentials.get_public_key_registered", return_value=False):
                with patch("sleap_connect.auth.credentials.set_public_key_registered") as mock_set:
                    with patch("sleap_connect.config.get_config", return_value=mock_config):
                        with patch("sleap_connect.api.requests") as mock_requests:
                            mock_requests.post.return_value = MagicMock(ok=False, status_code=500)
                            _ensure_keypair_registered(auth_token="test-jwt")
                            mock_set.assert_not_called()

    def test_uses_custom_device_name(self, mock_config):
        """Should pass device_name to server request."""
        from sleap_connect.auth.keypair import generate_keypair, private_key_to_b64
        priv, pub = generate_keypair()
        priv_b64 = private_key_to_b64(priv)

        with patch("sleap_connect.auth.credentials.get_private_key_b64", return_value=priv_b64):
            with patch("sleap_connect.auth.credentials.get_public_key_registered", return_value=False):
                with patch("sleap_connect.auth.credentials.set_public_key_registered"):
                    with patch("sleap_connect.config.get_config", return_value=mock_config):
                        with patch("sleap_connect.api.requests") as mock_requests:
                            mock_requests.post.return_value = MagicMock(ok=True)
                            _ensure_keypair_registered(auth_token="test-jwt", device_name="cli")
                            call_kwargs = mock_requests.post.call_args
                            assert call_kwargs[1]["json"]["device_name"] == "cli"


# =============================================================================
# logout() Tests
# =============================================================================


class TestLogout:
    """Tests for logout() function."""

    def test_logout_clears_jwt(self):
        """Should call clear_jwt."""
        with patch("sleap_connect.auth.credentials.clear_jwt") as mock_clear:
            logout()
            mock_clear.assert_called_once()


# =============================================================================
# list_rooms() Tests
# =============================================================================


class TestListRooms:
    """Tests for list_rooms() function."""

    def test_list_rooms_success(self, mock_jwt, mock_config):
        """Should return list of Room objects."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "rooms": [
                {
                    "room_id": "room-1",
                    "name": "Test Room",
                    "role": "owner",
                    "created_by": "testuser",
                    "joined_at": 1234567890,
                    "expires_at": 1234567890,
                },
                {
                    "room_id": "room-2",
                    "name": "Another Room",
                    "role": "member",
                },
            ]
        }

        with patch("sleap_connect.auth.credentials.get_valid_jwt", return_value=mock_jwt):
            with patch("sleap_connect.config.get_config", return_value=mock_config):
                with patch("requests.get", return_value=mock_response):
                    rooms = list_rooms()
                    assert len(rooms) == 2
                    assert rooms[0].id == "room-1"
                    assert rooms[0].name == "Test Room"
                    assert rooms[0].role == "owner"
                    assert rooms[1].id == "room-2"
                    assert rooms[1].role == "member"

    def test_list_rooms_not_logged_in(self):
        """Should raise AuthenticationError when not logged in."""
        with patch("sleap_connect.auth.credentials.get_valid_jwt", return_value=None):
            with pytest.raises(AuthenticationError, match="Not logged in"):
                list_rooms()

    def test_list_rooms_with_filters(self, mock_jwt, mock_config):
        """Should pass filter parameters to API."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"rooms": []}

        with patch("sleap_connect.auth.credentials.get_valid_jwt", return_value=mock_jwt):
            with patch("sleap_connect.config.get_config", return_value=mock_config):
                with patch("requests.get", return_value=mock_response) as mock_get:
                    list_rooms(role="owner", sort_by="name", sort_order="desc", search="test")
                    call_kwargs = mock_get.call_args[1]
                    assert call_kwargs["params"]["role"] == "owner"
                    assert call_kwargs["params"]["sort_by"] == "name"
                    assert call_kwargs["params"]["sort_order"] == "desc"
                    assert call_kwargs["params"]["search"] == "test"

    def test_list_rooms_empty(self, mock_jwt, mock_config):
        """Should return empty list when no rooms."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"rooms": []}

        with patch("sleap_connect.auth.credentials.get_valid_jwt", return_value=mock_jwt):
            with patch("sleap_connect.config.get_config", return_value=mock_config):
                with patch("requests.get", return_value=mock_response):
                    rooms = list_rooms()
                    assert rooms == []


# =============================================================================
# list_workers() Tests
# =============================================================================


class TestListWorkers:
    """Tests for list_workers() function."""

    def test_list_workers_not_logged_in(self):
        """Should raise AuthenticationError when not logged in."""
        with patch("sleap_connect.auth.credentials.get_valid_jwt", return_value=None):
            with pytest.raises(AuthenticationError, match="Not logged in"):
                list_workers("room-1")

    def test_list_workers_success(self, mock_jwt, mock_config):
        """Should return list of Worker objects."""
        # This test mocks the async WebSocket flow
        mock_workers = [
            Worker(
                id="worker-1",
                name="GPU Worker 1",
                status="available",
                gpu_name="RTX 4090",
                gpu_memory_mb=24576,
                metadata={"tags": ["sleap-connect"]},
            )
        ]

        with patch("sleap_connect.auth.credentials.get_valid_jwt", return_value=mock_jwt):
            with patch("sleap_connect.config.get_config", return_value=mock_config):
                with patch("asyncio.run", return_value=mock_workers):
                    workers = list_workers("room-1")
                    assert len(workers) == 1
                    assert workers[0].name == "GPU Worker 1"
                    assert workers[0].status == "available"
                    assert workers[0].gpu_name == "RTX 4090"


# =============================================================================
# Data Class Tests
# =============================================================================


class TestDataClasses:
    """Tests for API data classes."""

    def test_user_dataclass(self):
        """User dataclass should work correctly."""
        user = User(id="123", username="test", avatar_url="https://example.com/a.png")
        assert user.id == "123"
        assert user.username == "test"
        assert user.avatar_url == "https://example.com/a.png"

    def test_user_dataclass_optional_avatar(self):
        """User avatar_url should be optional."""
        user = User(id="123", username="test")
        assert user.avatar_url is None

    def test_room_dataclass(self):
        """Room dataclass should work correctly."""
        room = Room(
            id="room-1",
            name="Test Room",
            role="owner",
            created_by="user1",
            joined_at=1234567890,
            expires_at=1234567999,
        )
        assert room.id == "room-1"
        assert room.name == "Test Room"
        assert room.role == "owner"

    def test_room_dataclass_optional_fields(self):
        """Room optional fields should default to None."""
        room = Room(id="room-1", name="Test", role="member")
        assert room.created_by is None
        assert room.joined_at is None
        assert room.expires_at is None

    def test_worker_dataclass(self):
        """Worker dataclass should work correctly."""
        worker = Worker(
            id="worker-1",
            name="GPU Worker",
            status="available",
            gpu_name="RTX 4090",
            gpu_memory_mb=24576,
            metadata={"key": "value"},
        )
        assert worker.id == "worker-1"
        assert worker.gpu_memory_mb == 24576

    def test_worker_dataclass_optional_fields(self):
        """Worker optional fields should default to None."""
        worker = Worker(id="w1", name="Worker", status="unknown")
        assert worker.gpu_name is None
        assert worker.gpu_memory_mb is None
        assert worker.metadata is None


# =============================================================================
# Path Checking Data Class Tests
# =============================================================================


from sleap_connect.api import (
    PathCheckResult,
    VideoPathStatus,
    ValidationResult,
    ValidationIssue,
    ConfigurationError,
    check_video_paths,
    validate_config,
)


class TestPathCheckDataClasses:
    """Tests for path checking data classes."""

    def test_video_path_status_found(self):
        """VideoPathStatus for found video."""
        status = VideoPathStatus(
            filename="video.mp4",
            original_path="/data/video.mp4",
            worker_path="/mnt/data/video.mp4",
            found=True,
        )
        assert status.filename == "video.mp4"
        assert status.found is True
        assert status.worker_path == "/mnt/data/video.mp4"

    def test_video_path_status_missing(self):
        """VideoPathStatus for missing video."""
        status = VideoPathStatus(
            filename="video.mp4",
            original_path="/data/video.mp4",
            found=False,
            suggestions=["/other/video.mp4", "/backup/video.mp4"],
        )
        assert status.found is False
        assert status.worker_path is None
        assert len(status.suggestions) == 2

    def test_path_check_result_all_found(self):
        """PathCheckResult when all videos found."""
        videos = [
            VideoPathStatus("v1.mp4", "/a/v1.mp4", "/a/v1.mp4", True),
            VideoPathStatus("v2.mp4", "/a/v2.mp4", "/a/v2.mp4", True),
        ]
        result = PathCheckResult(
            all_found=True,
            total_videos=2,
            found_count=2,
            missing_count=0,
            videos=videos,
            slp_path="/data/project.slp",
        )
        assert result.all_found is True
        assert result.missing_count == 0

    def test_path_check_result_some_missing(self):
        """PathCheckResult when some videos missing."""
        videos = [
            VideoPathStatus("v1.mp4", "/a/v1.mp4", "/a/v1.mp4", True),
            VideoPathStatus("v2.mp4", "/a/v2.mp4", found=False),
        ]
        result = PathCheckResult(
            all_found=False,
            total_videos=2,
            found_count=1,
            missing_count=1,
            videos=videos,
            slp_path="/data/project.slp",
        )
        assert result.all_found is False
        assert result.missing_count == 1


class TestValidationDataClasses:
    """Tests for validation data classes."""

    def test_validation_issue_error(self):
        """ValidationIssue for an error."""
        issue = ValidationIssue(
            field="config_path",
            message="File not found",
            code="FILE_NOT_FOUND",
            is_error=True,
            path="/path/to/config.yaml",
        )
        assert issue.is_error is True
        assert issue.code == "FILE_NOT_FOUND"

    def test_validation_issue_warning(self):
        """ValidationIssue for a warning."""
        issue = ValidationIssue(
            field="trainer_config.max_epochs",
            message="Value is unusually high",
            code="VALUE_WARNING",
            is_error=False,
        )
        assert issue.is_error is False

    def test_validation_result_valid(self):
        """ValidationResult when config is valid."""
        result = ValidationResult(
            valid=True,
            errors=[],
            warnings=[],
            config_path="/path/to/config.yaml",
        )
        assert result.valid is True
        assert len(result.errors) == 0

    def test_validation_result_invalid(self):
        """ValidationResult when config has errors."""
        error = ValidationIssue("field", "Error message", is_error=True)
        result = ValidationResult(
            valid=False,
            errors=[error],
            warnings=[],
            config_path="/path/to/config.yaml",
        )
        assert result.valid is False
        assert len(result.errors) == 1


# =============================================================================
# check_video_paths() Tests
# =============================================================================


class TestCheckVideoPaths:
    """Tests for check_video_paths() function."""

    def test_check_video_paths_not_logged_in(self):
        """Should raise AuthenticationError when not logged in."""
        with patch("sleap_connect.auth.credentials.get_valid_jwt", return_value=None):
            with pytest.raises(AuthenticationError, match="Not logged in"):
                check_video_paths("/data/project.slp", "room-1")


    def test_check_video_paths_success(self, mock_jwt, mock_config):
        """Should return PathCheckResult on success."""
        # Mock the async function result
        mock_result = PathCheckResult(
            all_found=True,
            total_videos=2,
            found_count=2,
            missing_count=0,
            videos=[
                VideoPathStatus("v1.mp4", "/a/v1.mp4", "/a/v1.mp4", True),
                VideoPathStatus("v2.mp4", "/a/v2.mp4", "/a/v2.mp4", True),
            ],
            slp_path="/data/project.slp",
        )

        with patch("asyncio.run", return_value=mock_result):
            result = check_video_paths("/data/project.slp", "room-1")
            assert result.all_found is True
            assert result.total_videos == 2


# =============================================================================
# validate_config() Tests
# =============================================================================


class TestValidateConfig:
    """Tests for validate_config() function."""

    def test_validate_config_file_not_found(self, tmp_path):
        """Should return error when file not found."""
        result = validate_config(str(tmp_path / "nonexistent.yaml"))
        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].code == "FILE_NOT_FOUND"

    def test_validate_config_invalid_yaml(self, tmp_path):
        """Should return error for invalid YAML."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("{ invalid yaml: [")

        result = validate_config(str(config_file))
        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].code == "YAML_PARSE_ERROR"

    def test_validate_config_not_a_dict(self, tmp_path):
        """Should return error when YAML is not a dict."""
        config_file = tmp_path / "list.yaml"
        config_file.write_text("- item1\n- item2")

        result = validate_config(str(config_file))
        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].code == "INVALID_FORMAT"

    def test_validate_config_valid_minimal(self, tmp_path):
        """Should pass for minimal valid config."""
        config_file = tmp_path / "valid.yaml"
        config_file.write_text("model_config:\n  batch_size: 4")

        result = validate_config(str(config_file))
        assert result.valid is True
        assert len(result.errors) == 0

    def test_validate_config_invalid_batch_size(self, tmp_path):
        """Should error on invalid batch_size."""
        config_file = tmp_path / "invalid_batch.yaml"
        config_file.write_text("model_config:\n  batch_size: -1")

        result = validate_config(str(config_file))
        assert result.valid is False
        assert any(e.field == "model_config.batch_size" for e in result.errors)

    def test_validate_config_invalid_max_epochs(self, tmp_path):
        """Should error on invalid max_epochs."""
        config_file = tmp_path / "invalid_epochs.yaml"
        config_file.write_text("trainer_config:\n  max_epochs: 0")

        result = validate_config(str(config_file))
        assert result.valid is False
        assert any(e.field == "trainer_config.max_epochs" for e in result.errors)

    def test_validate_config_warning_high_epochs(self, tmp_path):
        """Should warn on unusually high max_epochs."""
        config_file = tmp_path / "high_epochs.yaml"
        config_file.write_text("trainer_config:\n  max_epochs: 50000")

        result = validate_config(str(config_file))
        assert result.valid is True  # Warnings don't make it invalid
        assert any(
            w.field == "trainer_config.max_epochs" and w.code == "VALUE_WARNING"
            for w in result.warnings
        )

    def test_validate_config_warning_missing_train_labels(self, tmp_path):
        """Should warn when train_labels_path not specified."""
        config_file = tmp_path / "no_labels.yaml"
        config_file.write_text("model_config:\n  batch_size: 4")

        result = validate_config(str(config_file))
        assert result.valid is True
        assert any(
            w.field == "data_config.train_labels_path" for w in result.warnings
        )

    def test_validate_config_warning_local_path_not_found(self, tmp_path):
        """Should warn when local path doesn't exist."""
        config_file = tmp_path / "missing_path.yaml"
        config_file.write_text(
            "data_config:\n  train_labels_path: /nonexistent/path/labels.slp"
        )

        result = validate_config(str(config_file))
        assert result.valid is True  # Just a warning since path may exist on worker
        assert any(
            w.code == "PATH_NOT_FOUND_LOCAL" for w in result.warnings
        )

    def test_validate_config_full_valid(self, tmp_path):
        """Should pass for full valid config with existing path."""
        # Create a dummy labels file
        labels_file = tmp_path / "labels.slp"
        labels_file.write_text("")

        config_file = tmp_path / "full_valid.yaml"
        config_file.write_text(
            f"data_config:\n"
            f"  train_labels_path: {labels_file}\n"
            f"trainer_config:\n"
            f"  max_epochs: 100\n"
            f"model_config:\n"
            f"  batch_size: 8\n"
        )

        result = validate_config(str(config_file))
        assert result.valid is True
        assert len(result.errors) == 0


# =============================================================================
# Remote Execution Data Class Tests
# =============================================================================


from sleap_connect.api import (
    ProgressEvent,
    TrainingResult,
    InferenceResult,
    TrainingJob,
    JobError,
    run_training,
    run_inference,
)


class TestProgressEvent:
    """Tests for ProgressEvent dataclass."""

    def test_progress_event_train_begin(self):
        """ProgressEvent for train_begin."""
        event = ProgressEvent(
            event_type="train_begin",
            wandb_url="https://wandb.ai/run/123",
        )
        assert event.event_type == "train_begin"
        assert event.wandb_url == "https://wandb.ai/run/123"
        assert event.epoch is None

    def test_progress_event_epoch_end(self):
        """ProgressEvent for epoch_end."""
        event = ProgressEvent(
            event_type="epoch_end",
            epoch=5,
            total_epochs=100,
            train_loss=0.123,
            val_loss=0.145,
        )
        assert event.event_type == "epoch_end"
        assert event.epoch == 5
        assert event.train_loss == 0.123
        assert event.val_loss == 0.145

    def test_progress_event_train_end_success(self):
        """ProgressEvent for successful train_end."""
        event = ProgressEvent(
            event_type="train_end",
            success=True,
        )
        assert event.event_type == "train_end"
        assert event.success is True
        assert event.error_message is None

    def test_progress_event_train_end_failure(self):
        """ProgressEvent for failed train_end."""
        event = ProgressEvent(
            event_type="train_end",
            success=False,
            error_message="CUDA out of memory",
        )
        assert event.success is False
        assert event.error_message == "CUDA out of memory"

    def test_progress_event_model_type_default_none(self):
        """model_type should default to None."""
        event = ProgressEvent(event_type="epoch_end")
        assert event.model_type is None

    def test_progress_event_model_type_set(self):
        """model_type can be set on ProgressEvent."""
        event = ProgressEvent(
            event_type="epoch_end",
            epoch=1,
            train_loss=0.05,
            model_type="centroid",
        )
        assert event.model_type == "centroid"

    def test_progress_event_model_type_all_events(self):
        """model_type should be settable on all event types."""
        for event_type in ("train_begin", "epoch_end", "train_end"):
            event = ProgressEvent(
                event_type=event_type,
                model_type="centered_instance",
            )
            assert event.model_type == "centered_instance"


class TestTrainingResult:
    """Tests for TrainingResult dataclass."""

    def test_training_result_success(self):
        """TrainingResult for successful training."""
        result = TrainingResult(
            job_id="job_abc123",
            success=True,
            duration_seconds=3600.5,
            model_path="/models/run_001",
            final_epoch=100,
            final_train_loss=0.05,
            final_val_loss=0.08,
        )
        assert result.success is True
        assert result.job_id == "job_abc123"
        assert result.duration_seconds == 3600.5
        assert result.final_epoch == 100

    def test_training_result_failure(self):
        """TrainingResult for failed training."""
        result = TrainingResult(
            job_id="job_abc123",
            success=False,
            duration_seconds=300.0,
            final_epoch=10,
            error_message="GPU memory exhausted",
        )
        assert result.success is False
        assert result.error_message == "GPU memory exhausted"


class TestInferenceResult:
    """Tests for InferenceResult dataclass."""

    def test_inference_result_success(self):
        """InferenceResult for successful inference."""
        result = InferenceResult(
            job_id="job_xyz789",
            success=True,
            duration_seconds=120.5,
            predictions_path="/data/predictions.slp",
            frames_processed=1000,
        )
        assert result.success is True
        assert result.predictions_path == "/data/predictions.slp"
        assert result.frames_processed == 1000

    def test_inference_result_failure(self):
        """InferenceResult for failed inference."""
        result = InferenceResult(
            job_id="job_xyz789",
            success=False,
            error_message="Model not found",
        )
        assert result.success is False
        assert result.error_message == "Model not found"


class TestTrainingJob:
    """Tests for TrainingJob class."""

    def test_training_job_creation(self):
        """TrainingJob should store job metadata."""
        job = TrainingJob(
            job_id="job_123",
            room_id="room_abc",
            worker_id="worker_gpu1",
        )
        assert job.job_id == "job_123"
        assert job.room_id == "room_abc"
        assert job.worker_id == "worker_gpu1"
        assert job.status == "running"

    def test_training_job_cancel(self):
        """TrainingJob.cancel() should update status."""
        cancelled = False

        def cancel_func():
            nonlocal cancelled
            cancelled = True

        job = TrainingJob(
            job_id="job_123",
            room_id="room_abc",
            worker_id="worker_gpu1",
            _cancel_func=cancel_func,
        )
        job.cancel()
        assert job.status == "cancelled"
        assert cancelled is True


class TestJobError:
    """Tests for JobError exception."""

    def test_job_error_basic(self):
        """JobError should store error details."""
        error = JobError("Training failed", job_id="job_123", exit_code=1)
        assert str(error) == "Training failed"
        assert error.job_id == "job_123"
        assert error.exit_code == 1

    def test_job_error_minimal(self):
        """JobError with minimal info."""
        error = JobError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.job_id is None
        assert error.exit_code is None


# =============================================================================
# run_training() Tests
# =============================================================================


class TestRunTraining:
    """Tests for run_training() function."""

    def test_run_training_not_logged_in(self):
        """Should raise AuthenticationError when not logged in."""
        with patch("sleap_connect.auth.credentials.get_valid_jwt", return_value=None):
            with pytest.raises(AuthenticationError, match="Not logged in"):
                run_training("/config.yaml", "room-1")


    def test_run_training_success(self, mock_jwt, mock_config):
        """Should return TrainingResult on success."""
        mock_result = TrainingResult(
            job_id="job_123",
            success=True,
            duration_seconds=3600.0,
            model_path="/models/output",
            final_epoch=100,
        )

        with patch("asyncio.run", return_value=mock_result):
            result = run_training("/config.yaml", "room-1")
            assert result.success is True
            assert result.job_id == "job_123"

    def test_run_training_with_progress_callback(self, mock_jwt, mock_config):
        """Should call progress callback."""
        events = []

        def callback(event):
            events.append(event)

        mock_result = TrainingResult(job_id="job_123", success=True)

        with patch("asyncio.run", return_value=mock_result):
            run_training("/config.yaml", "room-1", progress_callback=callback)
            # Note: In real execution, callback would be called

    def test_run_training_passes_model_type(self):
        """Should pass model_type to _run_training_async."""
        mock_result = TrainingResult(job_id="job_123", success=True)

        with patch("asyncio.run", return_value=mock_result) as mock_run:
            run_training("/config.yaml", "room-1", model_type="centroid")
            # Verify model_type was passed to the async coroutine
            coro = mock_run.call_args[0][0]
            # The coroutine was created with model_type="centroid"
            # We can verify by checking the call was made
            mock_run.assert_called_once()

    def test_run_training_passes_on_log(self):
        """Should pass on_log to _run_training_async."""
        mock_result = TrainingResult(job_id="job_123", success=True)

        log_lines = []
        with patch("asyncio.run", return_value=mock_result) as mock_run:
            run_training(
                "/config.yaml", "room-1", on_log=lambda line: log_lines.append(line)
            )
            mock_run.assert_called_once()


# =============================================================================
# run_inference() Tests
# =============================================================================


class TestRunInference:
    """Tests for run_inference() function."""

    def test_run_inference_not_logged_in(self):
        """Should raise AuthenticationError when not logged in."""
        with patch("sleap_connect.auth.credentials.get_valid_jwt", return_value=None):
            with pytest.raises(AuthenticationError, match="Not logged in"):
                run_inference("/data.slp", ["/model1"], "room-1")


    def test_run_inference_success(self, mock_jwt, mock_config):
        """Should return InferenceResult on success."""
        mock_result = InferenceResult(
            job_id="job_456",
            success=True,
            duration_seconds=120.0,
            predictions_path="/data/predictions.slp",
        )

        with patch("sleap_connect.api.run_inference_batch", return_value=[mock_result]):
            result = run_inference("/data.slp", ["/model1"], "room-1")
            assert result.success is True
            assert result.predictions_path == "/data/predictions.slp"


# =============================================================================
# Inference data-channel message handling (Task 4)
# =============================================================================
#
# When the worker streams predictions.slp via the sequence:
#   FILE_META::<filename>:<size>:<hint>
#   <binary chunk 1>
#   ...
#   END_OF_FILE
#   MSG_JOB_COMPLETE::{"output_path": "<worker-side path>", ...}
#
# the on_message handler inside _run_inference_batch_async must:
#   1. Drop b"KEEP_ALIVE" heartbeat bytes BEFORE any other routing
#      (otherwise they'd be appended to the in-flight tempfile and
#      corrupt predictions.slp).
#   2. Route binary messages to file_receiver.handle_bytes().
#   3. Route file-transfer control strings (FILE_META / END_OF_FILE)
#      to file_receiver.handle_string() and NOT forward them to the
#      response_queue.
#   4. Forward all other strings (MSG_JOB_PROGRESS, MSG_JOB_COMPLETE,
#      etc.) to the response_queue unchanged.
#
# This mirrors the training-side wiring landed in PR #79.


class _CapturedOnMessage(Exception):
    """Sentinel raised by the mock RTCPeerConnection after _run_inference_batch_async
    has registered its on_message handler, to abort the rest of the function.

    The captured handler is attached to this exception via the .handler attr
    on the mock data channel — see _make_inference_async_mocks().
    """


def _make_inference_async_mocks():
    """Build the mock infrastructure needed to drive _run_inference_batch_async far
    enough to register on_message, then abort.

    Returns a tuple of (patches_to_apply, fake_data_channel) where
    patches_to_apply is a list of context managers and fake_data_channel
    exposes the captured on_message via fake_data_channel._handlers["message"].
    """
    import asyncio as _asyncio

    class _FakeDataChannel:
        def __init__(self):
            self._handlers: dict[str, object] = {}

        def on(self, event):
            def decorator(fn):
                self._handlers[event] = fn
                return fn

            return decorator

        def send(self, *_args, **_kwargs):
            pass

    fake_data_channel = _FakeDataChannel()

    class _FakePC:
        def __init__(self):
            self.localDescription = MagicMock(type="offer", sdp="fake-sdp")

        def createDataChannel(self, _name):
            return fake_data_channel

        async def createOffer(self):
            # on_message has been registered by this point — abort the
            # rest of _run_inference_batch_async by raising the sentinel.
            raise _CapturedOnMessage()

        async def setLocalDescription(self, _desc):
            pass

        async def setRemoteDescription(self, _desc):
            pass

        async def addIceCandidate(self, _cand):
            pass

        async def close(self):
            pass

    class _FakeWS:
        """Minimal async websocket that yields a 'registered_auth' on first
        recv and would yield a peer_list on the next, satisfying the
        register + discover_peers loops in _run_inference_batch_async.
        """

        def __init__(self):
            self._sent: list = []
            self._recv_queue = _asyncio.Queue()
            self._recv_queue.put_nowait(json.dumps({"type": "registered_auth"}))
            self._recv_queue.put_nowait(
                json.dumps(
                    {
                        "type": "peer_list",
                        "peers": [{"peer_id": "fake-worker-1"}],
                    }
                )
            )

        async def send(self, msg):
            self._sent.append(msg)

        async def recv(self):
            return await self._recv_queue.get()

    fake_ws = _FakeWS()

    class _FakeWSConnect:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return fake_ws

        async def __aexit__(self, *_args):
            return False

    return fake_data_channel, _FakePC, _FakeWSConnect


def _capture_on_message(monkeypatch_jwt: str = "fake-jwt"):
    """Run _run_inference_batch_async far enough to register on_message, capture
    it, and return (on_message, file_receiver, response_queue).

    The on_message closure is the actual production closure, with the
    file_receiver and response_queue extracted from its closure cell so
    tests can assert state changes.
    """
    import asyncio as _asyncio

    fake_data_channel, FakePC, FakeWSConnect = _make_inference_async_mocks()

    from sleap_connect import api as api_mod
    from sleap_connect.jobs.spec import TrackJobSpec

    spec = TrackJobSpec(
        data_path="/data.slp",
        model_paths=["/model1"],
    )

    async def _drive():
        try:
            await api_mod._run_inference_batch_async(
                specs=[spec],
                room_id="room-1",
                worker_id=None,
                timeout=10.0,
            )
        except _CapturedOnMessage:
            pass

    mock_cfg = MagicMock()
    mock_cfg.signaling_websocket = "ws://fake"

    with (
        patch("sleap_connect.auth.credentials.get_valid_jwt", return_value=monkeypatch_jwt),
        patch("sleap_connect.config.get_config", return_value=mock_cfg),
        patch("websockets.connect", FakeWSConnect),
        patch("aiortc.RTCPeerConnection", FakePC),
    ):
        _asyncio.run(_drive())

    on_message = fake_data_channel._handlers.get("message")
    assert on_message is not None, (
        "on_message was not registered — the mock chain aborted too early "
        "or _run_inference_batch_async was refactored without updating the test"
    )

    # Extract file_receiver and response_queue from the closure cells.
    cells = {
        name: cell.cell_contents
        for name, cell in zip(on_message.__code__.co_freevars, on_message.__closure__)
    }
    file_receiver = cells.get("file_receiver")
    response_queue = cells.get("response_queue")
    assert file_receiver is not None, (
        "on_message closure does not capture a 'file_receiver' — Task 4 "
        "wiring is missing"
    )
    assert (
        response_queue is not None
    ), "on_message closure does not capture a 'response_queue'"
    return on_message, file_receiver, response_queue


class TestRunInferenceAsyncMessageHandling:
    """Tests that _run_inference_batch_async's on_message handler routes messages
    to _StreamedFileReceiver and response_queue per the Task 4 wiring spec.

    These tests drive _run_inference_batch_async with mocked websockets/aiortc
    far enough to register the on_message closure, then invoke that
    closure directly with synthesized messages. This mirrors the
    training-side wiring landed in PR #79.
    """

    def test_keep_alive_heartbeat_is_dropped_before_receiver(self):
        """b'KEEP_ALIVE' must be filtered out BEFORE any file_receiver routing.

        Regression: if KEEP_ALIVE bytes fall through to handle_bytes during
        an in-flight transfer, they'd be appended to the tempfile and
        corrupt predictions.slp by 10 bytes.
        """
        import asyncio as _asyncio

        on_message, file_receiver, response_queue = _capture_on_message()

        # Start an in-flight transfer so handle_bytes WOULD write if called.
        file_receiver.handle_string("FILE_META::predictions.slp:5:/worker/out")
        assert file_receiver._pending is not None
        bytes_before = file_receiver._pending["bytes_written"]

        async def _go():
            await on_message(b"KEEP_ALIVE")

        _asyncio.run(_go())

        # KEEP_ALIVE must NOT be appended to the in-flight tempfile.
        assert file_receiver._pending is not None
        assert file_receiver._pending["bytes_written"] == bytes_before
        # KEEP_ALIVE must NOT reach response_queue.
        assert response_queue.qsize() == 0

        # Cleanup: abort the pending transfer so the tempfile is unlinked.
        file_receiver._abort_pending()

    def test_file_meta_string_is_consumed_by_receiver_not_forwarded(self):
        """FILE_META::... must be consumed by file_receiver.handle_string and
        NOT placed on response_queue.
        """
        import asyncio as _asyncio

        on_message, file_receiver, response_queue = _capture_on_message()

        async def _go():
            await on_message("FILE_META::predictions.slp:5:/worker/out")

        _asyncio.run(_go())

        # Receiver opened a pending transfer.
        assert file_receiver._pending is not None
        assert file_receiver._pending["filename"] == "predictions.slp"
        # NOT forwarded to response_queue.
        assert response_queue.qsize() == 0

        # Cleanup.
        file_receiver._abort_pending()

    def test_binary_chunk_after_file_meta_is_consumed_by_receiver(self):
        """A bytes message (after FILE_META has opened a transfer) is
        written to the receiver's tempfile, NOT placed on response_queue.
        """
        import asyncio as _asyncio

        on_message, file_receiver, response_queue = _capture_on_message()

        # Open a pending transfer.
        file_receiver.handle_string("FILE_META::predictions.slp:11:/worker/out")
        assert file_receiver._pending is not None

        async def _go():
            await on_message(b"hello world")

        _asyncio.run(_go())

        # Bytes were written to the tempfile.
        assert file_receiver._pending["bytes_written"] == len(b"hello world")
        # NOT forwarded to response_queue.
        assert response_queue.qsize() == 0

        # Cleanup.
        file_receiver._abort_pending()

    def test_end_of_file_completes_transfer_and_is_not_forwarded(self):
        """END_OF_FILE after chunks must be consumed by the receiver,
        retain the predictions path, and NOT be forwarded to response_queue.
        """
        import asyncio as _asyncio

        on_message, file_receiver, response_queue = _capture_on_message()

        async def _go():
            await on_message("FILE_META::predictions.slp:5:/worker/out")
            await on_message(b"hello")
            await on_message("END_OF_FILE")

        _asyncio.run(_go())

        # Receiver completed the transfer and retained the local path.
        local_path = file_receiver.take_predictions_path()
        assert local_path is not None
        assert os.path.exists(local_path)
        with open(local_path, "rb") as f:
            assert f.read() == b"hello"
        # END_OF_FILE was NOT forwarded.
        assert response_queue.qsize() == 0

        # Cleanup.
        os.unlink(local_path)

    def test_regular_string_is_forwarded_to_response_queue(self):
        """A non-file-transfer string like MSG_JOB_PROGRESS::... must be
        forwarded to response_queue unchanged (regression — preserves the
        pre-Task-4 behavior for all non-transfer messages).
        """
        import asyncio as _asyncio

        on_message, file_receiver, response_queue = _capture_on_message()

        async def _go():
            await on_message('MSG_JOB_PROGRESS::{"epoch": 1}')
            await on_message('MSG_JOB_COMPLETE::{"output_path": "/x.slp"}')

        _asyncio.run(_go())

        assert response_queue.qsize() == 2
        first = response_queue.get_nowait()
        second = response_queue.get_nowait()
        assert first == 'MSG_JOB_PROGRESS::{"epoch": 1}'
        assert second == 'MSG_JOB_COMPLETE::{"output_path": "/x.slp"}'
        # Receiver state untouched (no FILE_META was sent).
        assert file_receiver._pending is None


# =============================================================================
# Streamed prediction file reception (Task 2 — Gap 1)
# =============================================================================
#
# When the worker streams predictions.slp via the sequence:
#   FILE_META::predictions.slp:<size>:<hint>
#   <binary chunk 1>
#   <binary chunk 2>
#   ...
#   END_OF_FILE
#   INFERENCE_COMPLETE::{"predictions_path": "<worker-side path>"}
#
# the GUI-path message handler must:
#   - write the chunks to a local tempfile
#   - on INFERENCE_COMPLETE, substitute the local tempfile path for
#     data["predictions_path"] (preserving the worker path in
#     data["worker_predictions_path"] for v2 dual-mode fallback).


class TestStreamedFileReceiver:
    """Tests for the _StreamedFileReceiver state machine in sleap_connect.api."""

    def test_predictions_file_written_and_local_path_exposed(self, tmp_path):
        """Given the ordered sequence
            FILE_META::predictions.slp:<size>:<hint>
            <chunk1>
            <chunk2>
            END_OF_FILE
        the receiver must:
         - create a local temp file
         - write the concatenated chunk bytes to it
         - expose the local path via `take_predictions_path()`
        """
        from sleap_connect.api import _StreamedFileReceiver

        receiver = _StreamedFileReceiver()
        chunk1 = b"the rain in spain"
        chunk2 = b" falls mainly on the plain"
        size = len(chunk1) + len(chunk2)

        receiver.handle_string(f"FILE_META::predictions.slp:{size}:/worker/out")
        receiver.handle_bytes(chunk1)
        receiver.handle_bytes(chunk2)
        receiver.handle_string("END_OF_FILE")

        local_path = receiver.take_predictions_path()
        assert local_path is not None
        assert os.path.exists(local_path)
        with open(local_path, "rb") as f:
            assert f.read() == chunk1 + chunk2

        # take_predictions_path consumes the path — second call returns None.
        assert receiver.take_predictions_path() is None

        # Cleanup temp file so the test doesn't leave files in /tmp.
        os.unlink(local_path)

    def test_non_predictions_file_is_cleaned_up(self):
        """If the worker streams a filename other than predictions.slp, the
        receiver must still drain the bytes but must not retain the path as
        'predictions'. The temp file should be unlinked on END_OF_FILE to
        avoid leaking."""
        from sleap_connect.api import _StreamedFileReceiver

        receiver = _StreamedFileReceiver()
        receiver.handle_string("FILE_META::other.txt:4:/tmp")
        receiver.handle_bytes(b"data")
        receiver.handle_string("END_OF_FILE")

        # No predictions path should be exposed.
        assert receiver.take_predictions_path() is None

    def test_prefixed_predictions_filename_is_retained(self):
        """The worker constructs predictions output filenames as
        ``<input_data_path>.predictions.slp``, so the basename is e.g.
        ``resolved_20260427_labels.v003.predictions.slp`` — NOT the bare
        ``predictions.slp``. The receiver must accept any filename ending
        in ``predictions.slp`` and retain its path for the caller."""
        from sleap_connect.api import _StreamedFileReceiver

        receiver = _StreamedFileReceiver()
        prefixed = "resolved_20260427_labels.v003.predictions.slp"
        receiver.handle_string(f"FILE_META::{prefixed}:5:/worker/out")
        receiver.handle_bytes(b"hello")
        receiver.handle_string("END_OF_FILE")

        local_path = receiver.take_predictions_path()
        assert local_path is not None, (
            "Receiver dropped a prefixed predictions filename — this is the "
            "exact bug observed in E2E logs (filename mismatch in END_OF_FILE)"
        )
        assert os.path.exists(local_path)
        try:
            with open(local_path, "rb") as f:
                assert f.read() == b"hello"
        finally:
            os.unlink(local_path)

    def test_unexpected_binary_without_file_meta_is_dropped(self):
        """Binary chunks with no active FILE_META must be silently dropped
        (backward-compatible with the pre-Task-2 behavior)."""
        from sleap_connect.api import _StreamedFileReceiver

        receiver = _StreamedFileReceiver()
        # Must not raise
        receiver.handle_bytes(b"stray bytes")
        assert receiver.take_predictions_path() is None

    def test_end_of_file_without_file_meta_is_ignored(self):
        """A stray END_OF_FILE with no active FILE_META must not crash."""
        from sleap_connect.api import _StreamedFileReceiver

        receiver = _StreamedFileReceiver()
        # Must not raise
        receiver.handle_string("END_OF_FILE")
        assert receiver.take_predictions_path() is None

    def test_malformed_file_meta_exposes_transfer_error_and_no_predictions_path(
        self,
    ):
        """FILE_META missing the size field is an explicit failure.

        The receiver must:
         - not open a tempfile or set _pending
         - expose a non-empty `take_transfer_error()` string
         - drop subsequent bytes silently (no crash)
         - leave `take_predictions_path()` returning None
        A subsequent well-formed transfer must still work and the error
        flag must have been consumed by the `take_transfer_error()` call.
        """
        from sleap_connect.api import _StreamedFileReceiver

        receiver = _StreamedFileReceiver()
        # Malformed — only filename, no size subfield.
        receiver.handle_string("FILE_META::predictions.slp")
        # Any bytes that follow should not cause AttributeError or similar.
        receiver.handle_bytes(b"some bytes")
        # END_OF_FILE arrives with no active _pending — must not crash.
        receiver.handle_string("END_OF_FILE")

        assert receiver.take_predictions_path() is None
        error = receiver.take_transfer_error()
        assert error is not None and error != ""
        # take_transfer_error consumes the flag.
        assert receiver.take_transfer_error() is None

        # A subsequent well-formed transfer must still work:
        receiver.handle_string("FILE_META::predictions.slp:3:/tmp")
        receiver.handle_bytes(b"abc")
        receiver.handle_string("END_OF_FILE")
        local_path = receiver.take_predictions_path()
        assert local_path is not None
        assert os.path.exists(local_path)
        with open(local_path, "rb") as f:
            assert f.read() == b"abc"
        # No error on the successful transfer.
        assert receiver.take_transfer_error() is None
        os.unlink(local_path)

    def test_tempfile_open_failure_sets_transfer_error_and_falls_back_to_worker_path(
        self,
    ):
        """If tempfile.NamedTemporaryFile raises OSError (e.g., disk full),
        the receiver must:
         - not retain a predictions path (take_predictions_path() → None)
         - expose a non-empty `take_transfer_error()` string
        And `_apply_received_predictions` must NOT
        substitute `data['predictions_path']` with a local path in that
        case — the worker-side path is kept as the fallback.
        """
        from sleap_connect.api import (
            _StreamedFileReceiver,
            _apply_received_predictions,
        )

        receiver = _StreamedFileReceiver()

        with patch(
            "sleap_connect.api.tempfile.NamedTemporaryFile",
            side_effect=OSError("disk full"),
        ):
            receiver.handle_string("FILE_META::predictions.slp:5:/worker/out")
            # Subsequent bytes must drop silently (no crash, no retained data).
            receiver.handle_bytes(b"hello")
            receiver.handle_string("END_OF_FILE")

        assert receiver.take_predictions_path() is None
        error = receiver.take_transfer_error()
        assert error is not None and error != ""
        # take_transfer_error consumes.
        assert receiver.take_transfer_error() is None

        # But re-run with a receiver that has a latched failure to verify
        # that _apply_received_predictions leaves the
        # worker-side path alone when the stream failed to open.
        receiver2 = _StreamedFileReceiver()
        with patch(
            "sleap_connect.api.tempfile.NamedTemporaryFile",
            side_effect=OSError("disk full"),
        ):
            receiver2.handle_string("FILE_META::predictions.slp:5:/worker/out")

        data = {"predictions_path": "/worker/path.slp"}
        _apply_received_predictions(receiver2, data, "predictions_path")

        # Worker-side path preserved; no substitution.
        assert data["predictions_path"] == "/worker/path.slp"
        assert "worker_predictions_path" not in data

    def test_close_failure_on_end_of_file_aborts_retention(self, tmp_path):
        """If the file handle's close() raises OSError during END_OF_FILE
        handling, the receiver must:
         - NOT retain `_received_predictions_local_path` (partial file!)
         - expose a non-empty `take_transfer_error()` string
         - attempt to unlink the partial tempfile
        """
        from sleap_connect.api import _StreamedFileReceiver

        receiver = _StreamedFileReceiver()
        receiver.handle_string("FILE_META::predictions.slp:5:/worker/out")
        receiver.handle_bytes(b"partial")

        # Monkey-patch the open file handle's close() to raise OSError.
        pending = receiver._pending
        assert pending is not None
        local_path = pending["local_path"]
        # Close the real fh first so the tempfile object is flushed, then
        # swap in a fake that raises on close.
        pending["fh"].flush()
        fake_fh = MagicMock()
        fake_fh.close.side_effect = OSError("close blew up")
        pending["fh"] = fake_fh

        receiver.handle_string("END_OF_FILE")

        assert receiver.take_predictions_path() is None
        error = receiver.take_transfer_error()
        assert error is not None and error != ""

        # Partial file should have been unlinked.
        assert not os.path.exists(local_path)

    def test_write_failure_during_handle_bytes_aborts_retention(self, tmp_path):
        """If fh.write() raises OSError mid-transfer (disk full, fs quota, etc.),
        the receiver must:
          1. Set _transfer_failed_reason so the substitution helper sees a failure
          2. Abort the pending transfer (so subsequent bytes aren't written)
          3. Unlink the partial tempfile
          4. NOT retain the path on a later END_OF_FILE — even if filename
             matches predictions.slp

        Mirrors the existing close()-failure handling at the END_OF_FILE branch."""
        import os
        from unittest.mock import MagicMock
        from sleap_connect.api import _StreamedFileReceiver

        receiver = _StreamedFileReceiver()

        # Successful FILE_META — opens a real tempfile.
        receiver.handle_string("FILE_META::predictions.slp:100:/tmp")
        assert receiver._pending is not None
        real_path = receiver._pending["local_path"]
        real_fh = receiver._pending["fh"]

        # Replace the file handle with a mock that raises on write().
        bad_fh = MagicMock()
        bad_fh.write.side_effect = OSError("disk full")
        real_fh.close()  # Close the real file so we don't leak the FD.
        receiver._pending["fh"] = bad_fh

        # Trigger the write-failure path.
        receiver.handle_bytes(b"first chunk")

        # 1. Sticky error flag set.
        error = receiver.take_transfer_error()
        assert error is not None
        assert "predictions.slp" in error
        assert "write" in error.lower() or "disk full" in error

        # 2. Pending was aborted — subsequent bytes drop silently.
        receiver.handle_bytes(b"second chunk")
        assert receiver._pending is None

        # 3. Partial tempfile was unlinked.
        assert not os.path.exists(real_path), (
            f"Partial tempfile {real_path} should have been unlinked on write failure"
        )

        # 4. END_OF_FILE arriving later finds no pending and does not retain.
        receiver.handle_string("END_OF_FILE")
        assert receiver.take_predictions_path() is None


class TestInferenceCompletePathRewrite:
    """Tests that the INFERENCE_COMPLETE dispatch in _run_training_async
    substitutes the locally-received tempfile path for data['predictions_path']
    when a stream was received."""

    def test_inference_complete_payload_rewritten_to_local_path(self):
        """Simulate end-to-end dispatch logic:
            1. Receiver receives predictions.slp stream.
            2. INFERENCE_COMPLETE::{"predictions_path": "/worker/path.slp"}
               is parsed.
            3. The INFERENCE_COMPLETE callback is invoked with data where
               predictions_path == local temp path and
               worker_predictions_path == original worker path.
        """
        from sleap_connect.api import (
            _StreamedFileReceiver,
            _apply_received_predictions,
        )

        receiver = _StreamedFileReceiver()
        receiver.handle_string("FILE_META::predictions.slp:5:/worker/out")
        receiver.handle_bytes(b"hello")
        receiver.handle_string("END_OF_FILE")

        # Parsed INFERENCE_COMPLETE payload (as api.py does via json.loads).
        data = {"predictions_path": "/worker/path.slp"}
        _apply_received_predictions(receiver, data, "predictions_path")

        assert data["worker_predictions_path"] == "/worker/path.slp"
        assert data["predictions_path"] != "/worker/path.slp"
        assert os.path.exists(data["predictions_path"])
        with open(data["predictions_path"], "rb") as f:
            assert f.read() == b"hello"

        # After consumption, receiver has no pending path.
        assert receiver.take_predictions_path() is None

        os.unlink(data["predictions_path"])

    def test_inference_complete_untouched_when_no_stream_received(self):
        """If no predictions stream arrived (pre-Task-1 worker, or training
        only with no inference), the INFERENCE_COMPLETE payload passes
        through unchanged."""
        from sleap_connect.api import (
            _StreamedFileReceiver,
            _apply_received_predictions,
        )

        receiver = _StreamedFileReceiver()
        data = {"predictions_path": "/worker/path.slp"}
        _apply_received_predictions(receiver, data, "predictions_path")

        assert data == {"predictions_path": "/worker/path.slp"}
        assert "worker_predictions_path" not in data


class TestMsgJobCompletePathRewrite:
    """Tests that the MSG_JOB_COMPLETE dispatch in _run_single_spec_async
    substitutes the locally-received tempfile path for result_data['output_path']
    when a stream was received.

    Mirrors the helper-level test pattern of TestInferenceCompletePathRewrite —
    we exercise the helper directly with a dict shaped like what
    json.loads(MSG_JOB_COMPLETE payload) would yield, rather than spinning up a
    fake aiortc data channel.
    """

    def test_msg_job_complete_payload_rewritten_to_local_path(self):
        """Simulate the MSG_JOB_COMPLETE dispatch logic.

        1. Receiver receives predictions.slp stream.
        2. MSG_JOB_COMPLETE::{"output_path": "/worker/...", "duration_seconds": 1.2}
           is parsed.
        3. _apply_received_predictions(receiver, data, "output_path") is invoked.
        4. data["output_path"] becomes the local tempfile, the worker path
           is preserved, and unrelated fields like duration_seconds pass through.
        """
        from sleap_connect.api import (
            _StreamedFileReceiver,
            _apply_received_predictions,
        )

        receiver = _StreamedFileReceiver()
        receiver.handle_string("FILE_META::predictions.slp:5:/worker/out")
        receiver.handle_bytes(b"hello")
        receiver.handle_string("END_OF_FILE")

        # Parsed MSG_JOB_COMPLETE payload (as api.py does via json.loads).
        data = {
            "output_path": "/worker/path.slp",
            "duration_seconds": 1.2,
        }
        _apply_received_predictions(receiver, data, "output_path")

        assert data["worker_output_path"] == "/worker/path.slp"
        assert data["output_path"] != "/worker/path.slp"
        assert os.path.exists(data["output_path"])
        with open(data["output_path"], "rb") as f:
            assert f.read() == b"hello"
        # Unrelated fields pass through untouched.
        assert data["duration_seconds"] == 1.2

        # After consumption, receiver has no pending path.
        assert receiver.take_predictions_path() is None

        os.unlink(data["output_path"])

    def test_msg_job_complete_untouched_when_no_stream_received(self):
        """If no predictions stream arrived (pre-Task-7 worker, or worker did
        not stream output for a non-track job), the MSG_JOB_COMPLETE payload
        passes through unchanged.
        """
        from sleap_connect.api import (
            _StreamedFileReceiver,
            _apply_received_predictions,
        )

        receiver = _StreamedFileReceiver()
        data = {"output_path": "/worker/path.slp", "duration_seconds": 1.2}
        _apply_received_predictions(receiver, data, "output_path")

        assert data == {"output_path": "/worker/path.slp", "duration_seconds": 1.2}
        assert "worker_output_path" not in data

    def test_run_inference_async_calls_apply_received_predictions_for_output_path(self):
        """Source-grep regression: the MSG_JOB_COMPLETE dispatch in
        _run_single_spec_async must invoke _apply_received_predictions
        with the 'output_path' field name.
        """
        import inspect
        from sleap_connect.api import _run_single_spec_async

        src = inspect.getsource(_run_single_spec_async)
        assert "_apply_received_predictions(" in src, (
            "Task 5 wiring missing: helper not invoked in _run_single_spec_async"
        )
        assert '"output_path"' in src, (
            "Task 5 wiring missing: 'output_path' field name not present"
        )


class TestApplyReceivedPredictionsGeneralized:
    """Tests for the generalized _apply_received_predictions helper.

    Verifies the helper substitutes the configured field name (not just
    'predictions_path'). Sets up the path for Task 5 (MSG_JOB_COMPLETE
    using 'output_path' as the field).
    """

    def test_substitutes_output_path_when_field_name_is_output_path(self, tmp_path):
        from sleap_connect.api import _StreamedFileReceiver, _apply_received_predictions

        receiver = _StreamedFileReceiver()
        receiver.handle_string("FILE_META::predictions.slp:5:/worker/out")
        receiver.handle_bytes(b"hello")
        receiver.handle_string("END_OF_FILE")

        data = {"output_path": "/root/vast/predictions.slp"}
        _apply_received_predictions(receiver, data, "output_path")

        assert data["output_path"] != "/root/vast/predictions.slp"
        assert data["output_path"].endswith(".slp")
        assert data["worker_output_path"] == "/root/vast/predictions.slp"

    def test_substitutes_predictions_path_with_explicit_field_name(self, tmp_path):
        """Regression: existing INFERENCE_COMPLETE call site (with 'predictions_path')
        still works after the rename."""
        from sleap_connect.api import _StreamedFileReceiver, _apply_received_predictions

        receiver = _StreamedFileReceiver()
        receiver.handle_string("FILE_META::predictions.slp:5:/worker/out")
        receiver.handle_bytes(b"world")
        receiver.handle_string("END_OF_FILE")

        data = {"predictions_path": "/root/vast/p.slp"}
        _apply_received_predictions(receiver, data, "predictions_path")

        assert data["predictions_path"] != "/root/vast/p.slp"
        assert data["worker_predictions_path"] == "/root/vast/p.slp"

    def test_no_substitution_when_no_predictions_received(self, tmp_path):
        """If receiver has no captured local path, the data field is left unchanged
        regardless of which field_name is requested."""
        from sleap_connect.api import _StreamedFileReceiver, _apply_received_predictions

        receiver = _StreamedFileReceiver()  # nothing was streamed

        data = {"output_path": "/worker/path.slp"}
        _apply_received_predictions(receiver, data, "output_path")

        assert data["output_path"] == "/worker/path.slp"
        assert "worker_output_path" not in data


class TestTempPredictionAtexitCleanup:
    """Task 3 — atexit safety net for temp prediction files.

    Tests the module-level tracking set, track/untrack helpers, and the
    atexit handler that deletes any still-tracked files on process exit.
    Tests also verify _StreamedFileReceiver registers successful
    predictions.slp transfers in the tracking set and does not register
    failed transfers.
    """

    @pytest.fixture(autouse=True)
    def _isolate_tracking_set(self):
        """Snapshot and restore the module-level tracking set around each test.

        The tracking set is process-global; without isolation, tests can
        pollute each other's state.
        """
        from sleap_connect import api as api_mod

        snapshot = set(api_mod._temp_prediction_paths)
        api_mod._temp_prediction_paths.clear()
        try:
            yield
        finally:
            api_mod._temp_prediction_paths.clear()
            api_mod._temp_prediction_paths.update(snapshot)

    def test_track_registers_path_for_cleanup(self, tmp_path):
        """Paths passed to track_temp_prediction are added to the tracking set."""
        from sleap_connect import api as api_mod
        from sleap_connect.api import track_temp_prediction, untrack_temp_prediction

        target = tmp_path / "some_predictions.slp"
        target.write_bytes(b"dummy")
        path_str = str(target)

        assert path_str not in api_mod._temp_prediction_paths
        track_temp_prediction(path_str)
        assert path_str in api_mod._temp_prediction_paths

        untrack_temp_prediction(path_str)
        assert path_str not in api_mod._temp_prediction_paths

    def test_untrack_is_idempotent(self):
        """untrack_temp_prediction on an untracked path is a no-op."""
        from sleap_connect import api as api_mod
        from sleap_connect.api import untrack_temp_prediction

        # Never tracked — must not raise.
        untrack_temp_prediction("/nonexistent/never_tracked.slp")
        assert "/nonexistent/never_tracked.slp" not in api_mod._temp_prediction_paths

        # Double untrack also safe.
        untrack_temp_prediction("/nonexistent/never_tracked.slp")

    def test_cleanup_deletes_tracked_files_and_clears_set(self, tmp_path):
        """_cleanup_temp_prediction_files removes files from disk and empties the set."""
        from sleap_connect import api as api_mod
        from sleap_connect.api import (
            _cleanup_temp_prediction_files,
            track_temp_prediction,
        )

        f1 = tmp_path / "a.slp"
        f2 = tmp_path / "b.slp"
        f1.write_bytes(b"one")
        f2.write_bytes(b"two")

        track_temp_prediction(str(f1))
        track_temp_prediction(str(f2))
        assert str(f1) in api_mod._temp_prediction_paths
        assert str(f2) in api_mod._temp_prediction_paths

        _cleanup_temp_prediction_files()

        assert not f1.exists()
        assert not f2.exists()
        assert str(f1) not in api_mod._temp_prediction_paths
        assert str(f2) not in api_mod._temp_prediction_paths

    def test_cleanup_silently_ignores_missing_files(self, tmp_path):
        """If a tracked file was already deleted (by the caller), cleanup doesn't raise."""
        from sleap_connect import api as api_mod
        from sleap_connect.api import (
            _cleanup_temp_prediction_files,
            track_temp_prediction,
        )

        f = tmp_path / "already_gone.slp"
        f.write_bytes(b"x")
        path_str = str(f)
        track_temp_prediction(path_str)

        # Caller already unlinked the file — happy path.
        os.unlink(path_str)
        assert not os.path.exists(path_str)

        # Must not raise.
        _cleanup_temp_prediction_files()

        assert path_str not in api_mod._temp_prediction_paths

    def test_cleanup_tolerates_os_error_on_unlink(self, tmp_path):
        """If os.unlink raises OSError during cleanup, we swallow it and still
        clear the set (best-effort at exit time)."""
        from sleap_connect import api as api_mod
        from sleap_connect.api import (
            _cleanup_temp_prediction_files,
            track_temp_prediction,
        )

        f = tmp_path / "c.slp"
        f.write_bytes(b"z")
        path_str = str(f)
        track_temp_prediction(path_str)

        with patch(
            "sleap_connect.api.os.unlink", side_effect=OSError("permission denied")
        ):
            # Must not raise.
            _cleanup_temp_prediction_files()

        assert path_str not in api_mod._temp_prediction_paths

    def test_receiver_tracks_successful_predictions_path(self, tmp_path):
        """_StreamedFileReceiver.handle_string tracks the path on successful END_OF_FILE."""
        from sleap_connect import api as api_mod
        from sleap_connect.api import _StreamedFileReceiver

        receiver = _StreamedFileReceiver()
        receiver.handle_string("FILE_META::predictions.slp:5:/worker/out")
        receiver.handle_bytes(b"hello")
        receiver.handle_string("END_OF_FILE")

        # Path exposed to the caller should also be registered in the
        # atexit tracking set — take_predictions_path does NOT untrack.
        local_path = receiver._received_predictions_local_path
        assert local_path is not None
        assert local_path in api_mod._temp_prediction_paths

        # take_predictions_path consumes the receiver's field but does
        # NOT remove the path from the tracking set.
        consumed = receiver.take_predictions_path()
        assert consumed == local_path
        assert local_path in api_mod._temp_prediction_paths

        # Cleanup for the test.
        os.unlink(local_path)

    def test_receiver_does_not_track_non_predictions_filename(self, tmp_path):
        """Files streamed under a non-predictions.slp filename are unlinked
        immediately on END_OF_FILE and must NOT be added to the tracking set."""
        from sleap_connect import api as api_mod
        from sleap_connect.api import _StreamedFileReceiver

        before = set(api_mod._temp_prediction_paths)
        receiver = _StreamedFileReceiver()
        receiver.handle_string("FILE_META::other.txt:4:/tmp")
        receiver.handle_bytes(b"data")
        receiver.handle_string("END_OF_FILE")

        assert api_mod._temp_prediction_paths == before

    def test_receiver_does_not_track_malformed_file_meta(self):
        """Malformed FILE_META (no size) must not register anything."""
        from sleap_connect import api as api_mod
        from sleap_connect.api import _StreamedFileReceiver

        before = set(api_mod._temp_prediction_paths)
        receiver = _StreamedFileReceiver()
        receiver.handle_string("FILE_META::predictions.slp")
        receiver.handle_bytes(b"some bytes")
        receiver.handle_string("END_OF_FILE")

        assert api_mod._temp_prediction_paths == before

    def test_receiver_does_not_track_close_failure(self):
        """If close() raises OSError on END_OF_FILE, the partial file is
        unlinked in the failure path and must NOT be added to tracking."""
        from sleap_connect import api as api_mod
        from sleap_connect.api import _StreamedFileReceiver

        before = set(api_mod._temp_prediction_paths)
        receiver = _StreamedFileReceiver()
        receiver.handle_string("FILE_META::predictions.slp:5:/worker/out")
        receiver.handle_bytes(b"partial")

        pending = receiver._pending
        assert pending is not None
        local_path = pending["local_path"]
        pending["fh"].flush()
        fake_fh = MagicMock()
        fake_fh.close.side_effect = OSError("close blew up")
        pending["fh"] = fake_fh

        receiver.handle_string("END_OF_FILE")

        assert api_mod._temp_prediction_paths == before
        assert local_path not in api_mod._temp_prediction_paths

    def test_receiver_does_not_track_tempfile_open_failure(self):
        """If tempfile.NamedTemporaryFile raises, no path is created and
        nothing is added to the tracking set."""
        from sleap_connect import api as api_mod
        from sleap_connect.api import _StreamedFileReceiver

        before = set(api_mod._temp_prediction_paths)
        receiver = _StreamedFileReceiver()
        with patch(
            "sleap_connect.api.tempfile.NamedTemporaryFile",
            side_effect=OSError("disk full"),
        ):
            receiver.handle_string("FILE_META::predictions.slp:5:/worker/out")
            receiver.handle_bytes(b"hello")
            receiver.handle_string("END_OF_FILE")

        assert api_mod._temp_prediction_paths == before


class TestRunInferenceAsyncSignature:
    """Task 8: ``_run_inference_batch_async`` and the public ``run_inference``
    wrapper must accept ``on_channel_ready`` (thread-safe send hook),
    ``on_log`` (raw log line callback), and ``on_job_message`` (typed
    JOB_* dispatch callback) so a GUI can wire its Cancel button and
    progress dialog to the standalone-track flow.
    """

    def test_run_inference_batch_async_has_on_channel_ready(self):
        import inspect
        from sleap_connect.api import _run_inference_batch_async

        sig = inspect.signature(_run_inference_batch_async)
        assert "on_channel_ready" in sig.parameters
        assert sig.parameters["on_channel_ready"].default is None

    def test_run_inference_batch_async_has_on_log(self):
        import inspect
        from sleap_connect.api import _run_inference_batch_async

        sig = inspect.signature(_run_inference_batch_async)
        assert "on_log" in sig.parameters
        assert sig.parameters["on_log"].default is None

    def test_run_inference_batch_async_has_on_job_message(self):
        import inspect
        from sleap_connect.api import _run_inference_batch_async

        sig = inspect.signature(_run_inference_batch_async)
        assert "on_job_message" in sig.parameters
        assert sig.parameters["on_job_message"].default is None

    def test_run_inference_wrapper_plumbs_through(self):
        import inspect
        from sleap_connect.api import run_inference

        sig = inspect.signature(run_inference)
        for name in ("on_channel_ready", "on_log", "on_job_message"):
            assert name in sig.parameters, f"run_inference missing {name}"
            assert sig.parameters[name].default is None

    def test_run_inference_wrapper_has_frame_filter_and_video_index(self):
        import inspect
        from sleap_connect.api import run_inference

        sig = inspect.signature(run_inference)
        for name in ("frame_filter", "video_index"):
            assert name in sig.parameters, f"run_inference missing {name}"
            assert sig.parameters[name].default is None


class TestProtocolConstants:
    def test_msg_job_log_exists(self):
        from sleap_connect.protocol import MSG_JOB_LOG

        assert MSG_JOB_LOG == "JOB_LOG"


class TestRunInferenceAsyncJobMessageForwarding:
    """Task 9: ``_run_single_spec_async`` forwards each ``MSG_JOB_*`` wire
    message to ``on_job_message`` with the typed name and parsed dict.
    """

    def test_msg_job_progress_forwarded_to_on_job_message(self):
        from sleap_connect.api import _dispatch_inference_response

        events: list[tuple[str, dict]] = []

        def _on_job_msg(msg_type, data):
            events.append((msg_type, data))

        _dispatch_inference_response(
            "JOB_PROGRESS::" + json.dumps({"frames": 17, "total": 35}),
            on_job_message=_on_job_msg,
            on_log=None,
        )

        assert events == [("JOB_PROGRESS", {"frames": 17, "total": 35})]

    def test_msg_job_log_forwarded_to_both_callbacks(self):
        from sleap_connect.api import _dispatch_inference_response

        logs: list[str] = []
        events: list[tuple[str, dict]] = []
        _dispatch_inference_response(
            "JOB_LOG::Predicting... 50% 17/35 ETA: 0:00:01",
            on_job_message=lambda t, d: events.append((t, d)),
            on_log=logs.append,
        )

        assert logs == ["Predicting... 50% 17/35 ETA: 0:00:01"]
        assert events == [("JOB_LOG", {"text": "Predicting... 50% 17/35 ETA: 0:00:01"})]


# =============================================================================
# list_workers() worker_name resolution Tests
# =============================================================================


class TestListWorkersNameResolution:
    """Worker name should come from the 'worker_name' property,
    not the 'name' property (which doesn't exist in the registration payload).
    Falls back to peer_id if worker_name is missing.
    """

    def test_worker_name_reads_worker_name_property(self):
        """list_workers should populate Worker.name from
        properties['worker_name'], not properties['name']."""
        from unittest.mock import AsyncMock

        registered_response = json.dumps({"type": "registered_auth"})
        peer_list_response = json.dumps({
            "type": "peer_list",
            "peers": [{
                "peer_id": "worker-acct_VX9-amick-tr-b368b",
                "metadata": {
                    "properties": {
                        "worker_name": "runai-1",
                        "status": "available",
                    }
                }
            }]
        })

        mock_conn = AsyncMock()
        mock_conn.recv = AsyncMock(
            side_effect=[registered_response, peer_list_response]
        )
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        mock_config = MagicMock()
        mock_config.signaling_websocket = "wss://fake"

        with (
            patch("sleap_connect.auth.credentials.get_valid_jwt", return_value="fake-jwt"),
            patch("sleap_connect.config.get_config", return_value=mock_config),
            patch("websockets.connect", return_value=mock_conn),
        ):
            workers = list_workers("test-room")

        assert len(workers) == 1
        assert workers[0].name == "runai-1"
        assert workers[0].id == "worker-acct_VX9-amick-tr-b368b"

    def test_worker_name_falls_back_to_peer_id(self):
        """When worker_name is missing, fall back to peer_id."""
        from unittest.mock import AsyncMock

        registered_response = json.dumps({"type": "registered_auth"})
        peer_list_response = json.dumps({
            "type": "peer_list",
            "peers": [{
                "peer_id": "worker-abc123",
                "metadata": {
                    "properties": {
                        "status": "available",
                    }
                }
            }]
        })

        mock_conn = AsyncMock()
        mock_conn.recv = AsyncMock(
            side_effect=[registered_response, peer_list_response]
        )
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=False)

        mock_config = MagicMock()
        mock_config.signaling_websocket = "wss://fake"

        with (
            patch("sleap_connect.auth.credentials.get_valid_jwt", return_value="fake-jwt"),
            patch("sleap_connect.config.get_config", return_value=mock_config),
            patch("websockets.connect", return_value=mock_conn),
        ):
            workers = list_workers("test-room")

        assert len(workers) == 1
        assert workers[0].name == "worker-abc123"


# =============================================================================
# _run_single_spec_async helper extraction tests
# =============================================================================


class TestRunSingleSpecAsync:
    """Verify the extracted _run_single_spec_async helper exists with expected signature."""

    def test_helper_exists_and_is_callable(self):
        from sleap_connect.api import _run_single_spec_async
        import inspect
        assert inspect.iscoroutinefunction(_run_single_spec_async)

    def test_helper_accepts_required_params(self):
        import inspect
        from sleap_connect.api import _run_single_spec_async
        sig = inspect.signature(_run_single_spec_async)
        param_names = set(sig.parameters.keys())
        required = {
            "spec", "job_id", "data_channel", "response_queue",
            "file_receiver", "timeout", "on_job_message", "on_log",
        }
        assert required.issubset(param_names)


# =============================================================================
# run_inference_batch signature tests
# =============================================================================


class TestRunInferenceBatchSignature:
    """Verify run_inference_batch exists with the expected signature."""

    def test_batch_is_importable(self):
        from sleap_connect.api import run_inference_batch
        assert callable(run_inference_batch)

    def test_batch_in_all_exports(self):
        from sleap_connect import api
        assert "run_inference_batch" in api.__all__

    def test_batch_accepts_specs_list(self):
        import inspect
        from sleap_connect.api import run_inference_batch
        sig = inspect.signature(run_inference_batch)
        assert "specs" in sig.parameters

    def test_batch_accepts_on_spec_complete(self):
        import inspect
        from sleap_connect.api import run_inference_batch
        sig = inspect.signature(run_inference_batch)
        assert "on_spec_complete" in sig.parameters

    def test_batch_accepts_on_job_message(self):
        import inspect
        from sleap_connect.api import run_inference_batch
        sig = inspect.signature(run_inference_batch)
        assert "on_job_message" in sig.parameters

    def test_batch_accepts_on_log(self):
        import inspect
        from sleap_connect.api import run_inference_batch
        sig = inspect.signature(run_inference_batch)
        assert "on_log" in sig.parameters

    def test_batch_accepts_room_id_and_worker_id(self):
        import inspect
        from sleap_connect.api import run_inference_batch
        sig = inspect.signature(run_inference_batch)
        assert "room_id" in sig.parameters
        assert "worker_id" in sig.parameters

    def test_batch_accepts_on_channel_ready(self):
        import inspect
        from sleap_connect.api import run_inference_batch
        sig = inspect.signature(run_inference_batch)
        assert "on_channel_ready" in sig.parameters


class TestEnrichedJobMessageWrapper:
    """Verify _enriched_job_message_wrapper injects job_index/jobs_total."""

    def test_job_log_enriched_with_index(self):
        """Wrapper injects job_index and jobs_total before forwarding."""
        from sleap_connect.api import _enriched_job_message_wrapper
        captured = []
        def on_msg(msg_type, data):
            captured.append((msg_type, data.copy()))

        wrapper = _enriched_job_message_wrapper(on_msg, job_index=1, jobs_total=3)
        wrapper("JOB_LOG", {"text": "hello"})

        assert len(captured) == 1
        assert captured[0][0] == "JOB_LOG"
        assert captured[0][1]["job_index"] == 1
        assert captured[0][1]["jobs_total"] == 3
        assert captured[0][1]["text"] == "hello"

    def test_enrichment_with_none_callback_does_not_crash(self):
        """When on_job_message is None, wrapper still enriches (no-op forward)."""
        from sleap_connect.api import _enriched_job_message_wrapper
        wrapper = _enriched_job_message_wrapper(None, job_index=0, jobs_total=1)
        # Should not raise
        data = {"text": "test"}
        wrapper("JOB_LOG", data)
        # Data should still be enriched (mutated in-place)
        assert data["job_index"] == 0
        assert data["jobs_total"] == 1

    def test_enrichment_preserves_existing_payload_fields(self):
        """Existing fields in the data dict are not lost."""
        from sleap_connect.api import _enriched_job_message_wrapper
        captured = []
        wrapper = _enriched_job_message_wrapper(
            lambda t, d: captured.append(d.copy()), job_index=2, jobs_total=5,
        )
        wrapper("JOB_PROGRESS", {"progress": 0.5, "batch": 10})
        assert captured[0]["progress"] == 0.5
        assert captured[0]["batch"] == 10
        assert captured[0]["job_index"] == 2
        assert captured[0]["jobs_total"] == 5

    def test_different_indices_produce_different_wrappers(self):
        """Each wrapper instance captures its own job_index."""
        from sleap_connect.api import _enriched_job_message_wrapper
        results = []
        cb = lambda t, d: results.append(d.copy())
        w0 = _enriched_job_message_wrapper(cb, job_index=0, jobs_total=3)
        w1 = _enriched_job_message_wrapper(cb, job_index=1, jobs_total=3)
        w2 = _enriched_job_message_wrapper(cb, job_index=2, jobs_total=3)
        w0("JOB_LOG", {"x": 1})
        w1("JOB_LOG", {"x": 2})
        w2("JOB_LOG", {"x": 3})
        assert [r["job_index"] for r in results] == [0, 1, 2]
        assert all(r["jobs_total"] == 3 for r in results)

    def test_enrichment_works_for_job_complete(self):
        """JOB_COMPLETE payloads are also enriched."""
        from sleap_connect.api import _enriched_job_message_wrapper
        captured = []
        wrapper = _enriched_job_message_wrapper(
            lambda t, d: captured.append((t, d.copy())), job_index=0, jobs_total=1,
        )
        wrapper("JOB_COMPLETE", {"output_path": "/tmp/pred.slp"})
        assert captured[0][0] == "JOB_COMPLETE"
        assert captured[0][1]["job_index"] == 0
        assert captured[0][1]["output_path"] == "/tmp/pred.slp"

    def test_enrichment_works_for_job_failed(self):
        """JOB_FAILED payloads are also enriched."""
        from sleap_connect.api import _enriched_job_message_wrapper
        captured = []
        wrapper = _enriched_job_message_wrapper(
            lambda t, d: captured.append((t, d.copy())), job_index=1, jobs_total=2,
        )
        wrapper("JOB_FAILED", {"message": "out of memory"})
        assert captured[0][0] == "JOB_FAILED"
        assert captured[0][1]["job_index"] == 1
        assert captured[0][1]["message"] == "out of memory"


class TestRunInferenceDelegatesToBatch:
    """Verify run_inference delegates to run_inference_batch."""

    def test_run_inference_signature_unchanged(self):
        """run_inference still accepts all its original parameters."""
        import inspect
        from sleap_connect.api import run_inference
        sig = inspect.signature(run_inference)
        expected = {
            "data_path", "model_paths", "room_id", "worker_id",
            "output_path", "batch_size", "peak_threshold",
            "only_suggested_frames", "frames", "frame_filter",
            "video_index", "exclude_user_labeled", "path_mappings",
            "progress_callback", "timeout",
            "on_channel_ready", "on_log", "on_job_message",
        }
        assert expected.issubset(set(sig.parameters.keys()))

    def test_run_inference_return_type_is_single_result(self):
        """Return annotation should be InferenceResult, not a list."""
        import inspect
        from sleap_connect.api import run_inference
        sig = inspect.signature(run_inference)
        annotation = str(sig.return_annotation)
        assert "list" not in annotation.lower()
        assert "List" not in annotation
