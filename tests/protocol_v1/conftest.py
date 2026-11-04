"""Shared fixtures for protocol v1 tests."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_sleap_home(tmp_path_factory, monkeypatch):
    """Point per-user sleap-connect state (the running-worker registry, the
    shell reminder scripts) at a temporary folder, so no test — or `serve`
    child process it starts, which inherits the environment — touches the
    real ~/.sleap-connect."""
    monkeypatch.setenv("SLEAP_CONNECT_HOME", str(tmp_path_factory.mktemp("sleap-home")))
    # `serve` checks sleap-nn imports (seconds, if it's installed here).
    monkeypatch.setenv("SLEAP_CONNECT_SKIP_SLEAP_NN_CHECK", "1")
