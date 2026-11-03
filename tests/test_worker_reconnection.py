"""Tests for WebSocket reconnection logic in RTCWorkerClient.run_worker()."""

import asyncio
import json
import time

import pytest
import websockets.exceptions
from unittest.mock import AsyncMock, MagicMock, patch, call


MODULE = "sleap_connect.worker.worker_class"


def _make_worker():
    """Create an RTCWorkerClient without calling __init__."""
    from sleap_connect.worker.worker_class import RTCWorkerClient

    worker = RTCWorkerClient.__new__(RTCWorkerClient)
    worker.shutting_down = False
    worker.status = "available"
    worker.name = "test-worker"
    worker.capabilities = MagicMock()
    worker.gpu_memory_mb = 8000
    worker.gpu_model = "Test GPU"
    worker.cuda_version = "12.0"
    worker.max_concurrent_jobs = 1
    worker.supported_models = []
    worker.supported_job_types = []
    worker.file_manager = MagicMock()
    worker.file_manager.get_mounts.return_value = []
    worker.pc = MagicMock()
    worker.on_datachannel = MagicMock()
    worker.on_iceconnectionstatechange = MagicMock()
    worker.handle_connection = AsyncMock()
    worker.clean_exit = AsyncMock()
    worker.websocket = None
    worker.peer_id_for_cleanup = None
    worker.signaling_dns = None
    worker.api_key = None
    worker.room_id = None
    worker.room_token = None
    # Mesh-related attributes
    worker.worker_connections = {}
    worker.data_channels = {}
    worker.mesh_coordinator = None
    # Signaling heartbeat watchdog attributes
    worker._last_signaling_ping = 0.0
    worker._heartbeat_watchdog_task = None
    return worker


def _make_mock_ws():
    """Create a mock websocket with async context manager support."""
    mock_ws = AsyncMock()
    mock_ws.send = AsyncMock()
    return mock_ws


def _make_mock_connect(mock_ws):
    """Create a mock for websockets.connect that returns an async context manager."""
    mock_connect = MagicMock()
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=mock_ws)
    cm.__aexit__ = AsyncMock(return_value=False)
    mock_connect.return_value = cm
    return mock_connect


# Patches applied to every test to avoid importing heavy dependencies.
COMMON_PATCHES = {
    "resolve_secret": f"{MODULE}.resolve_secret",
    "state_manager_cls": f"{MODULE}.StateManager",
    "job_coordinator_cls": f"{MODULE}.JobCoordinator",
    "get_sleap_nn_version": f"{MODULE}._get_sleap_nn_version",
}


def _apply_common_patches(monkeypatch):
    """Apply common patches and return mocks dict."""
    mocks = {}
    resolve_secret_mock = MagicMock(return_value=None)
    monkeypatch.setattr(f"{MODULE}.resolve_secret", resolve_secret_mock)
    mocks["resolve_secret"] = resolve_secret_mock

    sm_mock = MagicMock()
    sm_mock.return_value = MagicMock()
    monkeypatch.setattr(f"{MODULE}.StateManager", sm_mock)
    mocks["state_manager_cls"] = sm_mock

    jc_mock = MagicMock()
    jc_mock.return_value = MagicMock()
    monkeypatch.setattr(f"{MODULE}.JobCoordinator", jc_mock)
    mocks["job_coordinator_cls"] = jc_mock

    nn_mock = MagicMock(return_value="unknown")
    monkeypatch.setattr(f"{MODULE}._get_sleap_nn_version", nn_mock)
    mocks["get_sleap_nn_version"] = nn_mock

    return mocks


# ---------------------------------------------------------------------------
# Test 1: Worker reconnects after WebSocket close
# ---------------------------------------------------------------------------
class TestReconnectAfterClose:
    async def test_reconnects_after_connection_closed(self, monkeypatch):
        """Simulate ConnectionClosed on first connection, verify a second attempt."""
        worker = _make_worker()
        _apply_common_patches(monkeypatch)

        mock_ws = _make_mock_ws()
        call_count = 0

        async def _handle_connection_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise websockets.exceptions.ConnectionClosed(None, None)
            # Second call: shut down cleanly
            worker.shutting_down = True

        worker.handle_connection = AsyncMock(side_effect=_handle_connection_side_effect)
        mock_connect = _make_mock_connect(mock_ws)
        monkeypatch.setattr(f"{MODULE}.websockets.connect", mock_connect)

        pc = MagicMock()
        await worker.run_worker(pc, "ws://test:8080", 8080, api_key="slp_testkey1234")

        # handle_connection was called twice: first raised, second succeeded
        assert worker.handle_connection.call_count == 2
        # websockets.connect was called twice
        assert mock_connect.call_count == 2
        worker.clean_exit.assert_awaited_once()


# ---------------------------------------------------------------------------
# Test 2: Backoff increases exponentially
# ---------------------------------------------------------------------------
class TestExponentialBackoff:
    async def test_backoff_increases_exponentially(self, monkeypatch):
        """Track asyncio.sleep() calls, verify 2^attempt pattern."""
        worker = _make_worker()
        _apply_common_patches(monkeypatch)

        sleep_values = []
        original_sleep = asyncio.sleep

        async def _mock_sleep(duration):
            sleep_values.append(duration)
            # Don't actually wait

        monkeypatch.setattr("asyncio.sleep", _mock_sleep)

        call_count = 0

        # The websockets.connect itself raises ConnectionRefusedError to trigger
        # the reconnect backoff without needing a full ws mock.
        def _connect_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 6:
                worker.shutting_down = True
            raise ConnectionRefusedError("refused")

        monkeypatch.setattr(f"{MODULE}.websockets.connect", _connect_side_effect)

        pc = MagicMock()
        await worker.run_worker(pc, "ws://test:8080", 8080, api_key="slp_testkey1234")

        # Attempt 0 has no sleep, attempts 1..5 have sleeps 1, 2, 4, 8, 16
        # (the first call raises, attempt becomes 1 -> sleep(2^0=1), etc.)
        assert len(sleep_values) >= 4
        assert sleep_values[0] == 1, f"First backoff should be 1s, got {sleep_values[0]}"
        # Verify exponential pattern: each value is 2x the previous
        for i in range(1, len(sleep_values)):
            assert sleep_values[i] == sleep_values[i - 1] * 2, (
                f"Expected exponential backoff: {sleep_values}"
            )


# ---------------------------------------------------------------------------
# Test 3: Backoff caps at 300s
# ---------------------------------------------------------------------------
class TestBackoffCap:
    async def test_backoff_caps_at_300(self, monkeypatch):
        """Verify sleep duration never exceeds 300 seconds."""
        worker = _make_worker()
        _apply_common_patches(monkeypatch)

        sleep_values = []

        async def _mock_sleep(duration):
            sleep_values.append(duration)

        monkeypatch.setattr("asyncio.sleep", _mock_sleep)

        call_count = 0

        def _connect_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Let it run for enough attempts to hit the cap (2^9 = 512 > 300)
            if call_count >= 12:
                worker.shutting_down = True
            raise OSError("network down")

        monkeypatch.setattr(f"{MODULE}.websockets.connect", _connect_side_effect)

        pc = MagicMock()
        await worker.run_worker(pc, "ws://test:8080", 8080, api_key="slp_testkey1234")

        # Verify no sleep exceeds 300
        assert all(v <= 300 for v in sleep_values), (
            f"Backoff exceeded 300s cap: {sleep_values}"
        )
        # Verify that at least one sleep hit exactly 300 (cap was exercised)
        assert 300 in sleep_values, (
            f"Expected backoff to reach 300s cap but max was {max(sleep_values)}: {sleep_values}"
        )


# ---------------------------------------------------------------------------
# Test 4: Counter resets on successful connection
# ---------------------------------------------------------------------------
class TestCounterReset:
    async def test_counter_resets_on_successful_connection(self, monkeypatch):
        """Fail a few times, succeed (reset), fail again (backoff restarts from 1)."""
        worker = _make_worker()
        _apply_common_patches(monkeypatch)

        sleep_values = []

        async def _mock_sleep(duration):
            sleep_values.append(duration)
            # Stop after we have enough data to verify the reset
            if len(sleep_values) >= 5:
                worker.shutting_down = True

        monkeypatch.setattr("asyncio.sleep", _mock_sleep)

        mock_ws = _make_mock_ws()

        # Build a context manager for successful connections
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=mock_ws)
        cm.__aexit__ = AsyncMock(return_value=False)

        phase = 0  # 0: failing, 1: connected, 2: failing again
        connect_call_count = 0

        def _connect_side_effect(*args, **kwargs):
            nonlocal phase, connect_call_count
            connect_call_count += 1
            if phase == 0:
                if connect_call_count >= 3:
                    # Third connect call succeeds (phases 1)
                    phase = 1
                    return cm
                raise ConnectionRefusedError("refused")
            elif phase == 1:
                # Successful connection - return context manager
                return cm
            else:
                # Phase 2: failing again
                raise ConnectionRefusedError("refused again")

        monkeypatch.setattr(f"{MODULE}.websockets.connect", _connect_side_effect)

        async def _handle_connection_side_effect(*args, **kwargs):
            nonlocal phase
            # After successful connection, trigger phase 2 failures
            phase = 2
            raise websockets.exceptions.ConnectionClosed(None, None)

        worker.handle_connection = AsyncMock(side_effect=_handle_connection_side_effect)

        pc = MagicMock()
        await worker.run_worker(pc, "ws://test:8080", 8080, api_key="slp_testkey1234")

        # Phase 0: connect calls 1, 2 fail -> attempt becomes 1, 2 -> sleeps 1, 2
        # Phase 1: connect call 3 succeeds -> attempt resets to 0, reconnect_start = None
        #   handle_connection raises ConnectionClosed -> attempt becomes 1
        # Phase 2: connect call 4 fails -> attempt becomes 2 -> sleeps at attempt 1: 1, then 2
        # So sleep_values should be: [1, 2, 1, 2, ...]
        # The key assertion: after the successful connection, backoff restarted at 1
        assert sleep_values[0] == 1, f"First backoff should be 1, got {sleep_values[0]}"
        assert sleep_values[1] == 2, f"Second backoff should be 2, got {sleep_values[1]}"

        # Find the index where the reset happened (back to 1 after the value 2)
        reset_found = False
        for i in range(2, len(sleep_values)):
            if sleep_values[i] == 1:
                reset_found = True
                break
        assert reset_found, (
            f"Expected counter reset (1 appearing after larger values): {sleep_values}"
        )


# ---------------------------------------------------------------------------
# Test 5: Re-registers with "busy" status if job running
# ---------------------------------------------------------------------------
class TestBusyStatusRegistration:
    async def test_registers_with_busy_status(self, monkeypatch):
        """Set worker.status = 'busy', verify registration message contains it."""
        worker = _make_worker()
        worker.status = "busy"
        _apply_common_patches(monkeypatch)

        mock_ws = _make_mock_ws()
        mock_connect = _make_mock_connect(mock_ws)
        monkeypatch.setattr(f"{MODULE}.websockets.connect", mock_connect)

        async def _handle_stop(*args, **kwargs):
            worker.shutting_down = True

        worker.handle_connection = AsyncMock(side_effect=_handle_stop)

        pc = MagicMock()
        await worker.run_worker(pc, "ws://test:8080", 8080, api_key="slp_testkey1234")

        # Inspect the registration message sent to the websocket
        assert mock_ws.send.call_count >= 1
        sent_data = json.loads(mock_ws.send.call_args_list[0][0][0])
        assert sent_data["type"] == "register"
        assert sent_data["metadata"]["properties"]["status"] == "busy"


# ---------------------------------------------------------------------------
# Test 6: max_reconnect_time causes exit after timeout
# ---------------------------------------------------------------------------
class TestMaxReconnectTimeout:
    async def test_exits_after_max_reconnect_time(self, monkeypatch):
        """Set max_reconnect_time=5, mock time.monotonic to advance, verify exit."""
        worker = _make_worker()
        _apply_common_patches(monkeypatch)

        async def _mock_sleep(duration):
            pass  # Don't actually sleep

        monkeypatch.setattr("asyncio.sleep", _mock_sleep)

        # Mock time.monotonic to simulate passage of time.
        # The timeout check now happens BEFORE sleep, so the sequence is:
        #   attempt 0: connect fails -> reconnect_start = monotonic() [100.0]
        #   attempt 1: check timeout: monotonic() [101.0] - 100.0 = 1.0 < 5 -> sleep -> connect fails
        #   attempt 2: check timeout: monotonic() [103.0] - 100.0 = 3.0 < 5 -> sleep -> connect fails
        #   attempt 3: check timeout: monotonic() [107.0] - 100.0 = 7.0 > 5 -> break
        time_values = [100.0, 101.0, 103.0, 107.0]
        time_index = [0]

        def _mock_monotonic():
            idx = min(time_index[0], len(time_values) - 1)
            time_index[0] += 1
            return time_values[idx]

        monkeypatch.setattr(f"{MODULE}.time.monotonic", _mock_monotonic)

        def _connect_side_effect(*args, **kwargs):
            raise ConnectionRefusedError("refused")

        monkeypatch.setattr(f"{MODULE}.websockets.connect", _connect_side_effect)

        pc = MagicMock()
        # Should exit without raising
        await worker.run_worker(
            pc, "ws://test:8080", 8080, api_key="slp_testkey1234", max_reconnect_time=5
        )

        worker.clean_exit.assert_awaited_once()
        # Ensure worker did NOT set shutting_down (it broke out via timeout)
        assert not worker.shutting_down


# ---------------------------------------------------------------------------
# Test 7: Default retries forever (max_reconnect_time=None)
# ---------------------------------------------------------------------------
class TestRetriesForever:
    async def test_retries_forever_until_shutting_down(self, monkeypatch):
        """Verify the worker keeps retrying when max_reconnect_time is None."""
        worker = _make_worker()
        _apply_common_patches(monkeypatch)

        async def _mock_sleep(duration):
            pass

        monkeypatch.setattr("asyncio.sleep", _mock_sleep)

        call_count = 0

        def _connect_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 20:
                worker.shutting_down = True
            raise ConnectionRefusedError("refused")

        monkeypatch.setattr(f"{MODULE}.websockets.connect", _connect_side_effect)

        pc = MagicMock()
        await worker.run_worker(
            pc, "ws://test:8080", 8080, api_key="slp_testkey1234", max_reconnect_time=None
        )

        # Verify it retried many times before we stopped it
        assert call_count >= 20, f"Expected at least 20 attempts, got {call_count}"
        worker.clean_exit.assert_awaited_once()


# ---------------------------------------------------------------------------
# Test 8: KeyboardInterrupt exits cleanly
# ---------------------------------------------------------------------------
class TestKeyboardInterruptCleanExit:
    async def test_keyboard_interrupt_during_handle_connection(self, monkeypatch):
        """Raise KeyboardInterrupt during handle_connection, verify clean_exit called."""
        worker = _make_worker()
        _apply_common_patches(monkeypatch)

        mock_ws = _make_mock_ws()
        mock_connect = _make_mock_connect(mock_ws)
        monkeypatch.setattr(f"{MODULE}.websockets.connect", mock_connect)

        worker.handle_connection = AsyncMock(side_effect=KeyboardInterrupt)

        pc = MagicMock()
        await worker.run_worker(pc, "ws://test:8080", 8080, api_key="slp_testkey1234")

        worker.clean_exit.assert_awaited_once()

    async def test_keyboard_interrupt_during_websocket_connect(self, monkeypatch):
        """Raise KeyboardInterrupt during websockets.connect, verify clean_exit called."""
        worker = _make_worker()
        _apply_common_patches(monkeypatch)

        def _connect_raises(*args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr(f"{MODULE}.websockets.connect", _connect_raises)

        pc = MagicMock()
        await worker.run_worker(pc, "ws://test:8080", 8080, api_key="slp_testkey1234")

        worker.clean_exit.assert_awaited_once()

    async def test_keyboard_interrupt_during_sleep(self, monkeypatch):
        """Raise KeyboardInterrupt during asyncio.sleep (backoff), verify clean_exit called."""
        worker = _make_worker()
        _apply_common_patches(monkeypatch)

        call_count = 0

        def _connect_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ConnectionRefusedError("refused")

        monkeypatch.setattr(f"{MODULE}.websockets.connect", _connect_side_effect)

        async def _sleep_raises(duration):
            raise KeyboardInterrupt

        monkeypatch.setattr("asyncio.sleep", _sleep_raises)

        pc = MagicMock()
        await worker.run_worker(pc, "ws://test:8080", 8080, api_key="slp_testkey1234")

        worker.clean_exit.assert_awaited_once()


# ---------------------------------------------------------------------------
# Test 9: Mesh connections survive signaling disconnect
# ---------------------------------------------------------------------------
class TestMeshConnectionsSurvive:
    async def test_mesh_connections_not_cleared_on_reconnect(self, monkeypatch):
        """Verify worker_connections and data_channels are not cleared during reconnection."""
        worker = _make_worker()
        _apply_common_patches(monkeypatch)

        # Simulate existing mesh connections before reconnection
        mock_pc_mesh = MagicMock()
        mock_dc_mesh = MagicMock()
        worker.worker_connections = {"worker-peer-abc": mock_pc_mesh}
        worker.data_channels = {"worker-peer-abc": mock_dc_mesh}

        mock_ws = _make_mock_ws()
        mock_connect = _make_mock_connect(mock_ws)
        monkeypatch.setattr(f"{MODULE}.websockets.connect", mock_connect)

        call_count = 0

        async def _handle_connection_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First connection: simulate a disconnect
                raise websockets.exceptions.ConnectionClosed(None, None)
            # Second connection: check mesh connections are still intact, then exit
            assert "worker-peer-abc" in worker.worker_connections
            assert "worker-peer-abc" in worker.data_channels
            assert worker.worker_connections["worker-peer-abc"] is mock_pc_mesh
            assert worker.data_channels["worker-peer-abc"] is mock_dc_mesh
            worker.shutting_down = True

        worker.handle_connection = AsyncMock(side_effect=_handle_connection_side_effect)

        async def _mock_sleep(duration):
            pass

        monkeypatch.setattr("asyncio.sleep", _mock_sleep)

        pc = MagicMock()
        await worker.run_worker(pc, "ws://test:8080", 8080, api_key="slp_testkey1234")

        # Verify we went through both iterations
        assert call_count == 2
        # Verify mesh connections still exist
        assert len(worker.worker_connections) == 1
        assert len(worker.data_channels) == 1


# ---------------------------------------------------------------------------
# Test: peer_id is generated once and reused across reconnections
# ---------------------------------------------------------------------------
class TestPeerIdReuse:
    async def test_peer_id_is_stable_across_reconnections(self, monkeypatch):
        """Verify the same peer_id is sent in registration messages across reconnections."""
        worker = _make_worker()
        _apply_common_patches(monkeypatch)

        mock_ws = _make_mock_ws()
        mock_connect = _make_mock_connect(mock_ws)
        monkeypatch.setattr(f"{MODULE}.websockets.connect", mock_connect)

        call_count = 0

        async def _handle_connection_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise websockets.exceptions.ConnectionClosed(None, None)
            worker.shutting_down = True

        worker.handle_connection = AsyncMock(side_effect=_handle_connection_side_effect)

        async def _mock_sleep(duration):
            pass

        monkeypatch.setattr("asyncio.sleep", _mock_sleep)

        pc = MagicMock()
        await worker.run_worker(pc, "ws://test:8080", 8080, api_key="slp_testkey1234")

        # There should be 3 registration messages sent (3 connections)
        assert mock_ws.send.call_count == 3
        peer_ids = []
        for c in mock_ws.send.call_args_list:
            msg = json.loads(c[0][0])
            peer_ids.append(msg["peer_id"])

        # All peer_ids should be the same
        assert len(set(peer_ids)) == 1, f"Expected stable peer_id, got: {peer_ids}"
        # peer_id should start with "worker-"
        assert peer_ids[0].startswith("worker-")


# ---------------------------------------------------------------------------
# Test: clean_exit is always called (finally block)
# ---------------------------------------------------------------------------
class TestCleanExitAlwaysCalled:
    async def test_clean_exit_on_normal_exit(self, monkeypatch):
        """Verify clean_exit is called when shutting_down breaks the loop."""
        worker = _make_worker()
        worker.shutting_down = True  # Exit immediately
        _apply_common_patches(monkeypatch)

        pc = MagicMock()
        await worker.run_worker(pc, "ws://test:8080", 8080, api_key="slp_testkey1234")

        worker.clean_exit.assert_awaited_once()

    async def test_clean_exit_on_missing_api_key(self, monkeypatch):
        """Verify clean_exit is called even when api_key is missing (early return)."""
        worker = _make_worker()
        _apply_common_patches(monkeypatch)

        pc = MagicMock()
        await worker.run_worker(pc, "ws://test:8080", 8080, api_key=None)

        worker.clean_exit.assert_awaited_once()


# ---------------------------------------------------------------------------
# Test: Various exception types trigger reconnection
# ---------------------------------------------------------------------------
class TestExceptionHandling:
    async def test_connection_refused_triggers_reconnect(self, monkeypatch):
        """ConnectionRefusedError triggers reconnection, not crash."""
        worker = _make_worker()
        _apply_common_patches(monkeypatch)

        async def _mock_sleep(duration):
            pass

        monkeypatch.setattr("asyncio.sleep", _mock_sleep)

        call_count = 0

        def _connect_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                worker.shutting_down = True
            raise ConnectionRefusedError("refused")

        monkeypatch.setattr(f"{MODULE}.websockets.connect", _connect_side_effect)

        pc = MagicMock()
        await worker.run_worker(pc, "ws://test:8080", 8080, api_key="slp_testkey1234")

        assert call_count >= 3

    async def test_oserror_triggers_reconnect(self, monkeypatch):
        """OSError triggers reconnection, not crash."""
        worker = _make_worker()
        _apply_common_patches(monkeypatch)

        async def _mock_sleep(duration):
            pass

        monkeypatch.setattr("asyncio.sleep", _mock_sleep)

        call_count = 0

        def _connect_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                worker.shutting_down = True
            raise OSError("network unreachable")

        monkeypatch.setattr(f"{MODULE}.websockets.connect", _connect_side_effect)

        pc = MagicMock()
        await worker.run_worker(pc, "ws://test:8080", 8080, api_key="slp_testkey1234")

        assert call_count >= 3

    async def test_unexpected_exception_triggers_reconnect(self, monkeypatch):
        """Unexpected exceptions trigger reconnection, not crash."""
        worker = _make_worker()
        _apply_common_patches(monkeypatch)

        async def _mock_sleep(duration):
            pass

        monkeypatch.setattr("asyncio.sleep", _mock_sleep)

        call_count = 0

        def _connect_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                worker.shutting_down = True
            raise RuntimeError("something unexpected")

        monkeypatch.setattr(f"{MODULE}.websockets.connect", _connect_side_effect)

        pc = MagicMock()
        await worker.run_worker(pc, "ws://test:8080", 8080, api_key="slp_testkey1234")

        assert call_count >= 3


# ---------------------------------------------------------------------------
# Test: Registration message includes correct fields
# ---------------------------------------------------------------------------
class TestRegistrationMessage:
    async def test_registration_message_structure(self, monkeypatch):
        """Verify the registration message has expected fields."""
        worker = _make_worker()
        worker.gpu_memory_mb = 16000
        worker.gpu_model = "RTX 4090"
        worker.name = "my-worker"
        _apply_common_patches(monkeypatch)

        mock_ws = _make_mock_ws()
        mock_connect = _make_mock_connect(mock_ws)
        monkeypatch.setattr(f"{MODULE}.websockets.connect", mock_connect)

        async def _handle_stop(*args, **kwargs):
            worker.shutting_down = True

        worker.handle_connection = AsyncMock(side_effect=_handle_stop)

        pc = MagicMock()
        await worker.run_worker(pc, "ws://test:8080", 8080, api_key="slp_testkey1234")

        sent_data = json.loads(mock_ws.send.call_args_list[0][0][0])
        assert sent_data["type"] == "register"
        assert sent_data["role"] == "worker"
        assert "api_key" in sent_data
        assert sent_data["api_key"] == "slp_testkey1234"
        assert sent_data["metadata"]["properties"]["gpu_memory_mb"] == 16000
        assert sent_data["metadata"]["properties"]["gpu_model"] == "RTX 4090"
        assert sent_data["metadata"]["properties"]["worker_name"] == "my-worker"


# ---------------------------------------------------------------------------
# Test: handle_connection normal return triggers reconnection with log
# ---------------------------------------------------------------------------
class TestNormalReturnReconnects:
    async def test_reconnects_after_handle_connection_returns_normally(self, monkeypatch):
        """If handle_connection returns without exception, worker reconnects (not exits)."""
        worker = _make_worker()
        _apply_common_patches(monkeypatch)

        mock_ws = _make_mock_ws()
        mock_connect = _make_mock_connect(mock_ws)
        monkeypatch.setattr(f"{MODULE}.websockets.connect", mock_connect)

        call_count = 0

        async def _handle_connection_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                worker.shutting_down = True
            # Return normally (no exception) — simulates clean server close

        worker.handle_connection = AsyncMock(side_effect=_handle_connection_side_effect)

        pc = MagicMock()
        await worker.run_worker(pc, "ws://test:8080", 8080, api_key="slp_testkey1234")

        # Should have connected twice: first returned normally, second shut down
        assert call_count == 2
        assert mock_connect.call_count == 2


# ---------------------------------------------------------------------------
# Tests for _parse_duration helper
# ---------------------------------------------------------------------------
class TestParseDuration:
    def test_minutes(self):
        from sleap_connect.cli import _parse_duration
        assert _parse_duration("30m") == 1800

    def test_hours(self):
        from sleap_connect.cli import _parse_duration
        assert _parse_duration("2h") == 7200

    def test_compound(self):
        from sleap_connect.cli import _parse_duration
        assert _parse_duration("1h30m") == 5400

    def test_seconds_suffix(self):
        from sleap_connect.cli import _parse_duration
        assert _parse_duration("90s") == 90

    def test_plain_integer(self):
        from sleap_connect.cli import _parse_duration
        assert _parse_duration("3600") == 3600

    def test_zero_raises(self):
        import click
        from sleap_connect.cli import _parse_duration
        with pytest.raises(click.BadParameter):
            _parse_duration("0")

    def test_invalid_string_raises(self):
        import click
        from sleap_connect.cli import _parse_duration
        with pytest.raises(click.BadParameter):
            _parse_duration("abc")

    def test_zero_minutes_raises(self):
        import click
        from sleap_connect.cli import _parse_duration
        with pytest.raises(click.BadParameter):
            _parse_duration("0m")


# ---------------------------------------------------------------------------
# Tests for signaling heartbeat watchdog
# ---------------------------------------------------------------------------
class TestSignalingHeartbeatWatchdog:
    async def test_watchdog_closes_websocket_after_timeout(self):
        """Watchdog force-closes websocket when no pings arrive within 90s."""
        worker = _make_worker()
        mock_transport = MagicMock()
        worker.websocket = AsyncMock()
        worker.websocket.transport = mock_transport
        worker._last_signaling_ping = time.monotonic() - 100  # 100s ago

        sleep_called = False

        async def _mock_sleep(duration):
            nonlocal sleep_called
            sleep_called = True
            # Don't actually wait

        with patch("asyncio.sleep", _mock_sleep):
            await worker._signaling_heartbeat_watchdog()

        assert sleep_called
        mock_transport.close.assert_called_once()

    async def test_watchdog_does_not_close_when_pings_arrive(self):
        """Watchdog stays quiet when pings are recent."""
        worker = _make_worker()
        worker.websocket = AsyncMock()
        worker._last_signaling_ping = time.monotonic()

        call_count = 0

        async def _mock_sleep(duration):
            nonlocal call_count
            call_count += 1
            # After 3 checks, simulate shutdown to exit the loop
            if call_count >= 3:
                worker.shutting_down = True
            # Keep ping timestamp fresh each time
            worker._last_signaling_ping = time.monotonic()

        with patch("asyncio.sleep", _mock_sleep):
            await worker._signaling_heartbeat_watchdog()

        # Websocket should NOT have been closed
        worker.websocket.close.assert_not_awaited()
        assert call_count >= 3

    async def test_watchdog_started_in_reconnection_loop(self, monkeypatch):
        """Verify watchdog task is created when websocket connects."""
        worker = _make_worker()
        _apply_common_patches(monkeypatch)

        mock_ws = _make_mock_ws()
        mock_connect = _make_mock_connect(mock_ws)
        monkeypatch.setattr(f"{MODULE}.websockets.connect", mock_connect)

        async def _handle_stop(*args, **kwargs):
            # Verify watchdog was started
            assert worker._heartbeat_watchdog_task is not None
            assert not worker._heartbeat_watchdog_task.done()
            worker.shutting_down = True

        worker.handle_connection = AsyncMock(side_effect=_handle_stop)

        pc = MagicMock()
        await worker.run_worker(pc, "ws://test:8080", 8080, api_key="slp_testkey1234")

        assert worker._heartbeat_watchdog_task is not None
