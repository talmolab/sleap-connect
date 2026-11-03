"""Tests for GitHub OAuth login via dashboard."""

import pytest
from unittest.mock import patch, MagicMock
import time


class TestGithubLogin:
    """Tests for github_login function."""

    def test_get_dashboard_url_default(self):
        """Should return default dashboard URL."""
        from sleap_connect.auth.github import get_dashboard_url

        url = get_dashboard_url()
        assert url == "https://talmolab.github.io/sleap-connect/dashboard/"

    def test_get_dashboard_url_env_override(self, monkeypatch):
        """Should use SLEAP_DASHBOARD_URL env var if set."""
        from sleap_connect.auth.github import get_dashboard_url

        monkeypatch.setenv("SLEAP_DASHBOARD_URL", "http://localhost:8000")
        url = get_dashboard_url()
        assert url == "http://localhost:8000"

    @patch("sleap_connect.auth.github.webbrowser.open")
    @patch("sleap_connect.auth.github.requests.get")
    @patch("sleap_connect.auth.github.get_config")
    def test_github_login_success(self, mock_config, mock_get, mock_browser):
        """Should poll until JWT is available and return it."""
        from sleap_connect.auth.github import github_login

        # Setup mocks
        mock_config.return_value.get_http_url.return_value = "http://localhost:8001"

        # First poll returns pending, second returns JWT
        mock_response_pending = MagicMock()
        mock_response_pending.status_code = 202

        mock_response_ready = MagicMock()
        mock_response_ready.status_code = 200
        mock_response_ready.json.return_value = {
            "jwt": "test.jwt.token",
            "user": {"username": "testuser", "user_id": "123"}
        }

        mock_get.side_effect = [mock_response_pending, mock_response_ready]

        # Run login with short timeout
        result = github_login(timeout=10)

        # Verify
        assert result["jwt"] == "test.jwt.token"
        assert result["user"]["username"] == "testuser"
        assert mock_browser.called

    @patch("sleap_connect.auth.github.webbrowser.open")
    @patch("sleap_connect.auth.github.requests.get")
    @patch("sleap_connect.auth.github.get_config")
    def test_github_login_timeout(self, mock_config, mock_get, mock_browser):
        """Should raise RuntimeError on timeout."""
        from sleap_connect.auth.github import github_login

        mock_config.return_value.get_http_url.return_value = "http://localhost:8001"

        # Always return pending
        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_get.return_value = mock_response

        with pytest.raises(RuntimeError, match="timed out"):
            github_login(timeout=3)
