"""Sanity checks for Phase 6 (worker P2P auth) and Phase 7 (client P2P auth).

These tests verify the core auth path logic without requiring a live server.
HTTP calls in _fetch_authorized_keys are either bypassed via cache pre-loading
or mocked with aiohttp.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp  # ensure aiohttp is in sys.modules before patching

import pytest

from sleap_connect.auth.keypair import (
    generate_keypair,
    private_key_to_b64,
    public_key_to_b64,
    verify_signature,
    public_key_from_b64,
)
from sleap_connect.auth.psk import compute_hmac, generate_nonce
from sleap_connect.protocol import (
    MSG_AUTH_CHALLENGE,
    MSG_AUTH_RESPONSE,
    MSG_SEPARATOR,
)
from sleap_connect.worker.worker_class import RTCWorkerClient


# =============================================================================
# Shared fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _mock_progress_reporter():
    """Prevent real ZMQ socket binding during worker construction."""
    mock_reporter = MagicMock()
    mock_reporter.async_cleanup = AsyncMock()
    with patch(
        "sleap_connect.worker.worker_class.ProgressReporter", return_value=mock_reporter
    ):
        yield


@pytest.fixture
def worker():
    w = RTCWorkerClient()
    w.room_id = "test-room-123"
    w.api_key = "slp_acct_testkey"
    return w


@pytest.fixture
def channel():
    ch = MagicMock()
    ch.readyState = "open"
    ch.label = "data"
    return ch


# =============================================================================
# Phase 6 — Worker P2P Auth
# =============================================================================


class TestWorkerPSKPath:
    """6.2: PSK path takes priority and skips Ed25519 when room_secret is set."""

    async def test_valid_hmac_authenticates(self, worker, channel):
        room_secret = "dGVzdHNlY3JldA=="
        nonce = generate_nonce()
        worker._room_secret = room_secret
        worker._pending_auth[channel.label] = nonce

        hmac = compute_hmac(room_secret, nonce)
        await worker._handle_auth_response(
            channel, f"{MSG_AUTH_RESPONSE}{MSG_SEPARATOR}{hmac}"
        )

        assert channel.label in worker._authenticated_channels

    async def test_wrong_hmac_rejected(self, worker, channel):
        worker._room_secret = "dGVzdHNlY3JldA=="
        worker._pending_auth[channel.label] = generate_nonce()

        await worker._handle_auth_response(
            channel, f"{MSG_AUTH_RESPONSE}{MSG_SEPARATOR}wronghmac"
        )

        assert channel.label not in worker._authenticated_channels

    async def test_psk_path_never_fetches_authorized_keys(self, worker, channel):
        worker._room_secret = "dGVzdHNlY3JldA=="
        worker._pending_auth[channel.label] = generate_nonce()

        with patch.object(
            worker, "_fetch_authorized_keys", new_callable=AsyncMock
        ) as mock_fetch:
            await worker._handle_auth_response(
                channel, f"{MSG_AUTH_RESPONSE}{MSG_SEPARATOR}anything"
            )
            mock_fetch.assert_not_called()


class TestWorkerEd25519Path:
    """6.3: Ed25519 path when no room_secret — iterates authorized keys."""

    async def test_valid_signature_authenticates(self, worker, channel):
        private_key, public_key = generate_keypair()
        nonce = generate_nonce()

        worker._room_secret = None
        worker._pending_auth[channel.label] = nonce
        # Pre-load cache so no HTTP call is made
        worker._authorized_public_keys = [
            {
                "public_key": public_key_to_b64(public_key),
                "username": "alice",
                "device_name": "cli",
            }
        ]
        worker._auth_keys_last_fetched = time.monotonic()

        signature = private_key.sign(nonce.encode())
        import base64

        sig_b64 = base64.urlsafe_b64encode(signature).rstrip(b"=").decode()

        await worker._handle_auth_response(
            channel, f"{MSG_AUTH_RESPONSE}{MSG_SEPARATOR}{sig_b64}"
        )

        assert channel.label in worker._authenticated_channels

    async def test_wrong_key_rejected(self, worker, channel):
        private_key, _ = generate_keypair()
        _, other_public_key = generate_keypair()
        nonce = generate_nonce()

        worker._room_secret = None
        worker._pending_auth[channel.label] = nonce
        worker._authorized_public_keys = [
            {"public_key": public_key_to_b64(other_public_key), "username": "bob"}
        ]
        worker._auth_keys_last_fetched = time.monotonic()

        sig_b64 = private_key_to_b64(
            private_key
        )  # Wrong type — just any non-matching string
        await worker._handle_auth_response(
            channel, f"{MSG_AUTH_RESPONSE}{MSG_SEPARATOR}{sig_b64}"
        )

        assert channel.label not in worker._authenticated_channels

    async def test_missing_sentinel_rejected(self, worker, channel):
        worker._room_secret = None
        worker._pending_auth[channel.label] = generate_nonce()

        await worker._handle_auth_response(
            channel, f"{MSG_AUTH_RESPONSE}{MSG_SEPARATOR}missing"
        )

        assert channel.label not in worker._authenticated_channels

    async def test_empty_authorized_keys_rejected(self, worker, channel):
        private_key, _ = generate_keypair()
        nonce = generate_nonce()

        worker._room_secret = None
        worker._pending_auth[channel.label] = nonce
        worker._authorized_public_keys = []
        worker._auth_keys_last_fetched = time.monotonic()

        import base64

        sig_b64 = (
            base64.urlsafe_b64encode(private_key.sign(nonce.encode()))
            .rstrip(b"=")
            .decode()
        )
        await worker._handle_auth_response(
            channel, f"{MSG_AUTH_RESPONSE}{MSG_SEPARATOR}{sig_b64}"
        )

        assert channel.label not in worker._authenticated_channels


class TestFetchAuthorizedKeysCache:
    """6.1: _fetch_authorized_keys respects the 5-minute cache TTL."""

    async def test_fresh_cache_skips_http(self, worker):
        worker._authorized_public_keys = [{"public_key": "abc"}]
        worker._auth_keys_last_fetched = time.monotonic()  # Just fetched

        with patch("aiohttp.ClientSession") as mock_session:
            await worker._fetch_authorized_keys()
            mock_session.assert_not_called()

    async def test_stale_cache_triggers_http(self, worker):
        worker._authorized_public_keys = [{"public_key": "abc"}]
        # Set last_fetched to guaranteed-stale value regardless of system uptime
        worker._auth_keys_last_fetched = time.monotonic() - worker._AUTH_KEYS_TTL - 1

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"authorized_keys": [{"public_key": "xyz"}]}
        )
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session_instance = AsyncMock()
        mock_session_instance.get = MagicMock(return_value=mock_response)
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=mock_session_instance):
            await worker._fetch_authorized_keys()

        assert worker._authorized_public_keys == [{"public_key": "xyz"}]

    async def test_missing_room_id_skips_http(self, worker):
        worker.room_id = None

        with patch("aiohttp.ClientSession") as mock_session:
            await worker._fetch_authorized_keys()
            mock_session.assert_not_called()


# =============================================================================
# Task 1 — PSK bypass for role: "client" peers
# =============================================================================


class TestClientRolePSKBypass:
    """Worker skips AUTH_CHALLENGE for peers that registered with role: 'client'."""

    @patch("asyncio.create_task")
    def test_client_role_skips_auth_challenge(self, mock_create_task, worker):
        """When _pending_offer_role is 'client', no AUTH_CHALLENGE is sent."""
        ch = MagicMock()
        ch.readyState = "open"
        ch.label = "job"

        worker._pending_offer_role = "client"
        worker.on_datachannel(ch)

        sent = [str(call.args[0]) for call in ch.send.call_args_list]
        assert not any(MSG_AUTH_CHALLENGE in msg for msg in sent)
        assert ch.label in worker._authenticated_channels

    @patch("asyncio.create_task")
    def test_worker_role_sends_auth_challenge(self, mock_create_task, worker):
        """When _pending_offer_role is 'worker', AUTH_CHALLENGE is sent as normal."""
        ch = MagicMock()
        ch.readyState = "open"
        ch.label = "data"

        worker._pending_offer_role = "worker"
        worker.on_datachannel(ch)

        sent = [str(call.args[0]) for call in ch.send.call_args_list]
        assert any(MSG_AUTH_CHALLENGE in msg for msg in sent)
        assert ch.label not in worker._authenticated_channels


# =============================================================================
# Phase 7 — Client P2P Auth
# =============================================================================


def _make_open_channel():
    ch = MagicMock()
    ch.readyState = "open"
    return ch


class TestClientAuthChallenge:
    """7.2: Client responds with Ed25519, PSK, or 'missing' in priority order."""

    def _challenge_msg(self, nonce):
        return f"{MSG_AUTH_CHALLENGE}{MSG_SEPARATOR}{nonce}"

    def test_ed25519_path_sends_valid_signature(self, monkeypatch):
        from sleap_connect.client.client_class import RTCClient

        private_key, public_key = generate_keypair()
        priv_b64 = private_key_to_b64(private_key)
        pub_b64 = public_key_to_b64(public_key)

        monkeypatch.setattr(
            "sleap_connect.auth.credentials.get_private_key_b64", lambda: priv_b64
        )

        channel = _make_open_channel()
        client = RTCClient.__new__(RTCClient)
        client._room_secret = None
        client.data_channel = channel

        nonce = generate_nonce()
        client._handle_auth_challenge(self._challenge_msg(nonce))

        sent = channel.send.call_args[0][0]
        assert sent.startswith(f"{MSG_AUTH_RESPONSE}{MSG_SEPARATOR}")
        signature = sent.split(MSG_SEPARATOR, 1)[1]
        assert verify_signature(public_key_from_b64(pub_b64), nonce, signature)

    def test_psk_fallback_when_no_private_key(self, monkeypatch):
        from sleap_connect.client.client_class import RTCClient

        room_secret = "dGVzdHNlY3JldA=="
        monkeypatch.setattr(
            "sleap_connect.auth.credentials.get_private_key_b64", lambda: None
        )

        channel = _make_open_channel()
        client = RTCClient.__new__(RTCClient)
        client._room_secret = room_secret
        client.data_channel = channel

        nonce = generate_nonce()
        client._handle_auth_challenge(self._challenge_msg(nonce))

        sent = channel.send.call_args[0][0]
        expected = (
            f"{MSG_AUTH_RESPONSE}{MSG_SEPARATOR}{compute_hmac(room_secret, nonce)}"
        )
        assert sent == expected

    def test_missing_sent_when_no_credentials(self, monkeypatch):
        from sleap_connect.client.client_class import RTCClient

        monkeypatch.setattr(
            "sleap_connect.auth.credentials.get_private_key_b64", lambda: None
        )

        channel = _make_open_channel()
        client = RTCClient.__new__(RTCClient)
        client._room_secret = None
        client.data_channel = channel

        nonce = generate_nonce()
        client._handle_auth_challenge(self._challenge_msg(nonce))

        sent = channel.send.call_args[0][0]
        assert sent == f"{MSG_AUTH_RESPONSE}{MSG_SEPARATOR}missing"

    def test_ed25519_takes_priority_over_psk(self, monkeypatch):
        """Ed25519 is used even when room_secret is also configured."""
        from sleap_connect.client.client_class import RTCClient

        private_key, _ = generate_keypair()
        priv_b64 = private_key_to_b64(private_key)
        monkeypatch.setattr(
            "sleap_connect.auth.credentials.get_private_key_b64", lambda: priv_b64
        )

        channel = _make_open_channel()
        client = RTCClient.__new__(RTCClient)
        client._room_secret = "dGVzdHNlY3JldA=="  # Also has PSK
        client.data_channel = channel

        nonce = generate_nonce()
        client._handle_auth_challenge(self._challenge_msg(nonce))

        sent = channel.send.call_args[0][0]
        # Should NOT be the HMAC (PSK path)
        psk_response = compute_hmac("dGVzdHNlY3JldA==", nonce)
        assert sent != f"{MSG_AUTH_RESPONSE}{MSG_SEPARATOR}{psk_response}"


class TestClientTrackAuthChallenge:
    """Phase 7: Same logic verified for RTCTrackClient."""

    def test_ed25519_path(self, monkeypatch):
        from sleap_connect.client.client_track_class import RTCTrackClient

        private_key, public_key = generate_keypair()
        priv_b64 = private_key_to_b64(private_key)
        pub_b64 = public_key_to_b64(public_key)

        monkeypatch.setattr(
            "sleap_connect.auth.credentials.get_private_key_b64", lambda: priv_b64
        )

        channel = _make_open_channel()
        client = RTCTrackClient.__new__(RTCTrackClient)
        client._room_secret = None
        client.data_channel = channel

        nonce = generate_nonce()
        client._handle_auth_challenge(f"{MSG_AUTH_CHALLENGE}{MSG_SEPARATOR}{nonce}")

        sent = channel.send.call_args[0][0]
        signature = sent.split("::", 1)[1]
        assert verify_signature(public_key_from_b64(pub_b64), nonce, signature)

    def test_missing_when_no_credentials(self, monkeypatch):
        from sleap_connect.client.client_track_class import RTCTrackClient

        monkeypatch.setattr(
            "sleap_connect.auth.credentials.get_private_key_b64", lambda: None
        )

        channel = _make_open_channel()
        client = RTCTrackClient.__new__(RTCTrackClient)
        client._room_secret = None
        client.data_channel = channel
        client._auth_failed_reason = None
        client._auth_event = MagicMock()

        nonce = generate_nonce()
        client._handle_auth_challenge(f"{MSG_AUTH_CHALLENGE}{MSG_SEPARATOR}{nonce}")

        sent = channel.send.call_args[0][0]
        assert sent.endswith("::missing")
