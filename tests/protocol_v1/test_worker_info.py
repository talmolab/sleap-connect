"""Tests for the `worker.info` method."""

from sleap_rtc.jobs.queue import JobQueue
from sleap_rtc.protocol_v1 import worker_info
from sleap_rtc.protocol_v1.server import ProtocolV1Server
from sleap_rtc.protocol_v1.worker_info import WorkerInfoMethods

_FAKE_HARDWARE = {
    "gpu_model": "NVIDIA A40",
    "gpu_memory_mb": 46068,
    "gpu_count": 2,
    "cuda_version": "13.2",
    "sleap_nn_version": "0.3.3",
}


async def test_reports_hardware_and_live_busy_state(monkeypatch):
    monkeypatch.setattr(worker_info, "_detect_hardware", lambda: dict(_FAKE_HARDWARE))
    queue = JobQueue(max_concurrent=1)
    methods = WorkerInfoMethods(ProtocolV1Server(node_id="n"), queue)

    info = await methods.info({}, conn=None)
    assert {k: info[k] for k in _FAKE_HARDWARE} == _FAKE_HARDWARE
    assert info["busy"] is False

    async with queue.slot():
        assert (await methods.info({}, conn=None))["busy"] is True


async def test_is_registered_on_the_server(monkeypatch):
    monkeypatch.setattr(worker_info, "_detect_hardware", lambda: dict(_FAKE_HARDWARE))
    server = ProtocolV1Server(node_id="n")
    WorkerInfoMethods(server, JobQueue())
    assert "worker.info" in server._methods


async def test_hardware_is_detected_once(monkeypatch):
    calls = []
    monkeypatch.setattr(
        worker_info, "_detect_hardware", lambda: calls.append(1) or dict(_FAKE_HARDWARE)
    )
    methods = WorkerInfoMethods(ProtocolV1Server(node_id="n"), JobQueue())
    await methods.info({}, conn=None)
    await methods.info({}, conn=None)
    assert calls == [1]


def test_detect_hardware_never_raises_without_a_gpu():
    hw = worker_info._detect_hardware()
    assert set(_FAKE_HARDWARE) <= set(hw)
    assert isinstance(hw["gpu_count"], int)
