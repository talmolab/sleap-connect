"""Tests for the one-line pairing code format (item 6w.1).

Round trips are checked both with small hand-built tickets here and with
the cross-language fixtures in `pair_code_vectors.json`, which the
TypeScript app's own decoder is tested against too — see that file's
sibling test suite in sleap-app.
"""

import base64
import json
from pathlib import Path

import pytest

from sleap_rtc.protocol_v1.iroh_live import build_iroh_section
from sleap_rtc.protocol_v1.pair_code import (
    CODE_PREFIX,
    RELAY_TABLE,
    PairCodeError,
    decode_pair_code,
    encode_pair_code,
)
from sleap_rtc.protocol_v1.pairing import PairingTicket

VECTORS_PATH = Path(__file__).parent / "pair_code_vectors.json"


def _b64u(n: int, start: int = 0) -> str:
    """A deterministic (not random) URL-safe base64 string of `n` bytes."""
    raw = bytes((start + i) % 256 for i in range(n))
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


NODE_ID = _b64u(32, 1)
SECRET = _b64u(16, 100)


def _ticket(**overrides) -> PairingTicket:
    defaults = dict(
        node_id=NODE_ID,
        addrs=[],
        secret=SECRET,
        expires_at=1_700_000_000.0,
        iroh=None,
    )
    defaults.update(overrides)
    return PairingTicket(**defaults)


class TestRoundTrips:
    """`decode_pair_code(encode_pair_code(ticket)) == ticket` for every shape."""

    def test_no_addrs_no_relay(self):
        ticket = _ticket()

        assert decode_pair_code(encode_pair_code(ticket)) == ticket

    def test_ipv4_addr(self):
        ticket = _ticket(addrs=["ws://192.168.1.42:9631"])

        assert decode_pair_code(encode_pair_code(ticket)) == ticket

    def test_ipv6_addr_round_trips_with_brackets(self):
        ticket = _ticket(addrs=["ws://[fe80::1]:9631"])

        decoded = decode_pair_code(encode_pair_code(ticket))

        assert decoded == ticket
        assert decoded.addrs == ["ws://[fe80::1]:9631"]

    def test_multiple_addrs(self):
        ticket = _ticket(addrs=["ws://192.168.1.42:9631", "ws://[fe80::1]:9631"])

        assert decode_pair_code(encode_pair_code(ticket)) == ticket

    def test_url_tag_addr_for_anything_not_ws_ip_port(self):
        # A hostname (not a literal IP) — must fall back to the URL tag.
        ticket = _ticket(addrs=["ws://my-laptop.local:9631"])

        decoded = decode_pair_code(encode_pair_code(ticket))

        assert decoded == ticket
        assert decoded.addrs == ["ws://my-laptop.local:9631"]

    def test_url_tag_addr_for_non_ws_scheme(self):
        ticket = _ticket(addrs=["wss://tailnet-host.ts.net:9631/"])

        assert decode_pair_code(encode_pair_code(ticket)) == ticket

    def test_known_relay_encodes_as_a_table_index(self):
        relay_url = RELAY_TABLE[0]
        ticket = _ticket(iroh=build_iroh_section(NODE_ID, relay_url, []))

        code = encode_pair_code(ticket)
        decoded = decode_pair_code(code)

        assert decoded == ticket
        assert decoded.iroh == {
            "node_id": NODE_ID,
            "relay_url": relay_url,
            "direct_addrs": [],
        }

    def test_every_table_relay_round_trips(self):
        for relay_url in RELAY_TABLE:
            ticket = _ticket(iroh=build_iroh_section(NODE_ID, relay_url, []))

            assert decode_pair_code(encode_pair_code(ticket)) == ticket

    def test_custom_relay_url_not_in_table(self):
        ticket = _ticket(
            iroh=build_iroh_section(NODE_ID, "https://relay.example.org/", [])
        )

        decoded = decode_pair_code(encode_pair_code(ticket))

        assert decoded == ticket
        assert decoded.iroh["relay_url"] == "https://relay.example.org/"

    def test_direct_addrs_are_dropped_from_the_code(self):
        # The ticket passed to encode_pair_code still has direct_addrs (as
        # --json output would show), but the code only ever carries the
        # relay — decoding never reconstructs them.
        ticket = _ticket(
            iroh=build_iroh_section(
                NODE_ID, RELAY_TABLE[0], ["direct-addr-1", "direct-addr-2"]
            )
        )

        decoded = decode_pair_code(encode_pair_code(ticket))

        assert decoded.iroh["direct_addrs"] == []


class TestCodeShape:
    """Tests for the code's textual shape, independent of round-tripping."""

    def test_has_the_expected_prefix(self):
        code = encode_pair_code(_ticket())

        assert code.startswith(CODE_PREFIX)

    def test_is_lowercase_rfc4648_base32_with_no_padding(self):
        code = encode_pair_code(_ticket(addrs=["ws://192.168.1.42:9631"]))

        body = code[len(CODE_PREFIX) :]
        assert body == body.lower()
        assert "=" not in code
        assert set(body) <= set("abcdefghijklmnopqrstuvwxyz234567")

    def test_typical_size_one_ipv4_addr_and_known_relay(self):
        ticket = _ticket(
            addrs=["ws://192.168.1.42:9631"],
            iroh=build_iroh_section(NODE_ID, RELAY_TABLE[0], []),
        )

        code = encode_pair_code(ticket)

        # ~109 chars per the protocol spec's sizing note (1 + 32 + 16 + 4
        # header/footer bytes + 7 for one IPv4 addr + 1 for a known relay +
        # 2 checksum = 64 bytes -> 103 base32 chars + the 6-char prefix).
        assert len(code) <= 115


class TestDecodeErrors:
    """Tests for `decode_pair_code`'s error handling."""

    def test_wrong_prefix_is_rejected(self):
        code = encode_pair_code(_ticket())
        bad = "nope1" + code[len(CODE_PREFIX) :]

        with pytest.raises(PairCodeError):
            decode_pair_code(bad)

    def test_truncated_code_is_rejected(self):
        code = encode_pair_code(_ticket(addrs=["ws://192.168.1.42:9631"]))

        with pytest.raises(PairCodeError):
            decode_pair_code(code[:-20])

    def test_a_single_mistyped_character_fails_the_checksum(self):
        code = encode_pair_code(_ticket(addrs=["ws://192.168.1.42:9631"]))
        last = code[-1]
        replacement = "a" if last != "a" else "b"
        mistyped = code[:-1] + replacement

        with pytest.raises(PairCodeError):
            decode_pair_code(mistyped)

    def test_unsupported_version_is_rejected(self):
        # Hand-roll a code with version byte 2 and a correct checksum for
        # it, so this exercises the version check specifically (not just
        # "checksum happens to fail").
        import hashlib
        import struct

        from sleap_rtc.protocol_v1.pair_code import (
            _CHECKSUM_LEN,
            _b32_encode,
            _b64url_decode,
        )

        body = bytearray()
        body.append(2)  # unsupported format version
        body += _b64url_decode(NODE_ID)
        body += _b64url_decode(SECRET)
        body += struct.pack(">I", 1_700_000_000)
        body.append(0)  # no addrs
        body.append(0)  # no relay
        checksum = hashlib.sha256(bytes(body)).digest()[:_CHECKSUM_LEN]
        body += checksum
        code = CODE_PREFIX + _b32_encode(bytes(body))

        with pytest.raises(PairCodeError, match="version"):
            decode_pair_code(code)

    def test_garbage_input_is_rejected(self):
        with pytest.raises(PairCodeError):
            decode_pair_code("not a pairing code at all")

    def test_empty_string_is_rejected(self):
        with pytest.raises(PairCodeError):
            decode_pair_code("")


class TestCaseInsensitivity:
    """Decoding tolerates whitespace and mixed/upper case."""

    def test_uppercase_code_decodes_the_same(self):
        ticket = _ticket(addrs=["ws://192.168.1.42:9631"])
        code = encode_pair_code(ticket)

        assert decode_pair_code(code.upper()) == ticket

    def test_mixed_case_code_decodes_the_same(self):
        ticket = _ticket(addrs=["ws://192.168.1.42:9631"])
        code = encode_pair_code(ticket)
        mixed = "".join(c.upper() if i % 2 == 0 else c for i, c in enumerate(code))

        assert decode_pair_code(mixed) == ticket

    def test_surrounding_whitespace_is_stripped(self):
        ticket = _ticket()
        code = encode_pair_code(ticket)

        assert decode_pair_code(f"  \n{code}\t ") == ticket


@pytest.fixture(scope="module")
def vectors():
    return json.loads(VECTORS_PATH.read_text())


class TestSharedVectors:
    """Cross-language fixtures also exercised by the app's TS test suite."""

    def _ticket_from_vector(self, v: dict) -> PairingTicket:
        t = v["ticket"]
        iroh = (
            build_iroh_section(t["node_id"], t["relay_url"], [])
            if t["relay_url"]
            else None
        )
        return PairingTicket(
            node_id=t["node_id"],
            addrs=t["addrs"],
            secret=t["secret"],
            expires_at=float(t["expires_at"]),
            iroh=iroh,
        )

    def test_vectors_file_is_non_empty(self, vectors):
        assert len(vectors) >= 4

    def test_each_vector_encodes_to_its_code(self, vectors):
        for v in vectors:
            ticket = self._ticket_from_vector(v)

            assert encode_pair_code(ticket) == v["code"], v["description"]

    def test_each_vector_decodes_from_its_code(self, vectors):
        for v in vectors:
            ticket = self._ticket_from_vector(v)

            assert decode_pair_code(v["code"]) == ticket, v["description"]
