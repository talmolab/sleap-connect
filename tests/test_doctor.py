"""Tests for the sleap-connect doctor command extensions."""

from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from sleap_connect.cli import cli


@patch("sleap_connect.cli.requests")
def test_doctor_runs_without_error(mock_requests):
    """Doctor command should run to completion without crashing."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_requests.get.return_value = mock_response

    runner = CliRunner()
    result = runner.invoke(cli, ["doctor"])
    assert result.exit_code == 0


@patch("sleap_connect.cli.requests")
def test_doctor_shows_gpu_section(mock_requests):
    """Doctor command should include a GPU section."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_requests.get.return_value = mock_response

    runner = CliRunner()
    result = runner.invoke(cli, ["doctor"])
    assert "GPU:" in result.output


@patch("sleap_connect.cli.requests")
def test_doctor_shows_training_deps_section(mock_requests):
    """Doctor command should include a Training Dependencies section."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_requests.get.return_value = mock_response

    runner = CliRunner()
    result = runner.invoke(cli, ["doctor"])
    assert "Training Dependencies:" in result.output


@patch("sleap_connect.cli.requests")
def test_doctor_shows_mounts_section(mock_requests):
    """Doctor command should include a Data Mounts section."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_requests.get.return_value = mock_response

    runner = CliRunner()
    result = runner.invoke(cli, ["doctor"])
    assert "Data Mounts" in result.output


@patch("sleap_connect.cli.requests")
def test_doctor_shows_account_key_from_env(mock_requests):
    """Doctor should detect account key from environment variable."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_requests.get.return_value = mock_response

    runner = CliRunner()
    result = runner.invoke(
        cli, ["doctor"], env={"SLEAP_RTC_ACCOUNT_KEY": "slp_acct_test123456789"}
    )
    assert "from env var" in result.output


@patch("sleap_connect.cli.requests")
def test_doctor_shows_account_key_missing(mock_requests):
    """Doctor should warn when no account key is found."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_requests.get.return_value = mock_response

    with patch("sleap_connect.auth.credentials.get_account_key", return_value=None):
        runner = CliRunner()
        result = runner.invoke(cli, ["doctor"], env={"SLEAP_RTC_ACCOUNT_KEY": ""})
    # Should show not found or the fix instructions
    assert "Account key:" in result.output
