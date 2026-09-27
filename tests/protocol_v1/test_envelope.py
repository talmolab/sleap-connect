"""Tests for the protocol v1 wire envelope (de)serialization."""

import json

import pytest

from sleap_rtc.protocol_v1.envelope import (
    Event,
    EnvelopeError,
    Hello,
    Req,
    Res,
    parse_envelope,
)


class TestHello:
    """Tests for the `hello` frame."""

    def test_round_trips_through_json(self):
        hello = Hello(
            proto={"min": 1, "max": 1},
            agent={"name": "sleap-app", "version": "0.1.0", "platform": "darwin"},
            node_id="node-abc",
            nonce="nonce-123",
        )

        parsed = parse_envelope(hello.to_json())

        assert isinstance(parsed, Hello)
        assert parsed.proto == {"min": 1, "max": 1}
        assert parsed.node_id == "node-abc"
        assert parsed.nonce == "nonce-123"

    def test_wire_shape_has_type_and_version(self):
        hello = Hello(proto={"min": 1, "max": 1}, agent={}, node_id="n", nonce="x")

        d = hello.to_dict()

        assert d["type"] == "hello"
        assert d["v"] == 1

    def test_blob_port_round_trips_when_present(self):
        hello = Hello(
            proto={"min": 1, "max": 1}, agent={}, node_id="n", nonce="x", blob_port=9632
        )

        parsed = parse_envelope(hello.to_json())

        assert parsed.blob_port == 9632

    def test_blob_port_omitted_from_the_wire_shape_when_none(self):
        hello = Hello(proto={"min": 1, "max": 1}, agent={}, node_id="n", nonce="x")

        d = hello.to_dict()

        assert "blob_port" not in d

    def test_blob_port_defaults_to_none_when_absent_on_the_wire(self):
        hello = Hello(proto={"min": 1, "max": 1}, agent={}, node_id="n", nonce="x")

        parsed = parse_envelope(hello.to_json())

        assert parsed.blob_port is None


class TestReqRes:
    """Tests for `req`/`res` correlation."""

    def test_req_round_trips(self):
        req = Req(id=7, method="jobs.submit", params={"spec": {"type": "train"}})

        parsed = parse_envelope(req.to_json())

        assert isinstance(parsed, Req)
        assert parsed.id == 7
        assert parsed.method == "jobs.submit"
        assert parsed.params == {"spec": {"type": "train"}}

    def test_req_defaults_params_to_empty_dict(self):
        parsed = parse_envelope(
            json.dumps({"v": 1, "type": "req", "id": 1, "method": "jobs.list"})
        )

        assert parsed.params == {}

    def test_res_ok_has_result_not_error(self):
        res = Res.ok(7, {"job_id": "job-1"})

        d = res.to_dict()

        assert d["id"] == 7
        assert d["result"] == {"job_id": "job-1"}
        assert "error" not in d

    def test_res_err_has_error_not_result(self):
        res = Res.err(7, "job.not_found", "No such job", data={"job_id": "x"})

        d = res.to_dict()

        assert d["error"] == {
            "code": "job.not_found",
            "msg": "No such job",
            "data": {"job_id": "x"},
        }
        assert "result" not in d

    def test_res_round_trips(self):
        res = Res.err(3, "internal", "boom")

        parsed = parse_envelope(res.to_json())

        assert isinstance(parsed, Res)
        assert parsed.id == 3
        assert parsed.error == {"code": "internal", "msg": "boom"}
        assert parsed.result is None


class TestEvent:
    """Tests for the `event` frame."""

    def test_round_trips_with_job_id(self):
        event = Event(topic="job.log", seq=5, data={"line": "epoch 1"}, job_id="job-1")

        parsed = parse_envelope(event.to_json())

        assert isinstance(parsed, Event)
        assert parsed.topic == "job.log"
        assert parsed.seq == 5
        assert parsed.data == {"line": "epoch 1"}
        assert parsed.job_id == "job-1"

    def test_job_id_omitted_when_none(self):
        event = Event(topic="job.log", seq=1, data={})

        d = event.to_dict()

        assert "job_id" not in d


class TestParseEnvelopeErrors:
    """Tests that parse_envelope rejects malformed input clearly."""

    def test_rejects_invalid_json(self):
        with pytest.raises(EnvelopeError, match="Invalid JSON"):
            parse_envelope("not json")

    def test_rejects_non_object_json(self):
        with pytest.raises(EnvelopeError, match="JSON object"):
            parse_envelope("[1, 2, 3]")

    def test_rejects_unknown_type(self):
        with pytest.raises(EnvelopeError, match="Unknown or missing"):
            parse_envelope(json.dumps({"v": 1, "type": "bogus"}))

    def test_rejects_missing_type(self):
        with pytest.raises(EnvelopeError, match="Unknown or missing"):
            parse_envelope(json.dumps({"v": 1}))

    def test_rejects_req_missing_required_field(self):
        with pytest.raises(EnvelopeError, match="Missing required field"):
            parse_envelope(json.dumps({"v": 1, "type": "req", "id": 1}))  # no method
