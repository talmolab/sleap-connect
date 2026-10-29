"""Shared fixtures for protocol v1 tests."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_sleap_home(tmp_path_factory, monkeypatch):
    """Point per-user sleap-rtc state (the running-worker registry, the
    shell reminder scripts) at a temporary folder, so no test — or `serve`
    child process it starts, which inherits the environment — touches the
    real ~/.sleap-rtc."""
    monkeypatch.setenv("SLEAP_RTC_HOME", str(tmp_path_factory.mktemp("sleap-home")))
