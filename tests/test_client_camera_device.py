"""The room client's camera must run on whatever the PC has (a friend's laptop).

``device=0`` was passed to every YOLO call, so a PC without CUDA answered
"Invalid CUDA 'device=0' requested" and the client switched its camera off.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

from client import camera as client_camera


def _fake_torch(monkeypatch, *, available: bool, count: int = 1):
    cuda = SimpleNamespace(is_available=lambda: available, device_count=lambda: count)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))


def test_a_pc_with_cuda_runs_on_gpu_zero_with_fp16(monkeypatch):
    _fake_torch(monkeypatch, available=True)
    assert client_camera.yolo_placement() == (0, True)


def test_a_pc_without_cuda_runs_on_the_cpu_without_fp16(monkeypatch):
    _fake_torch(monkeypatch, available=False)
    assert client_camera.yolo_placement() == ("cpu", False)


def test_a_torch_that_cannot_even_be_imported_is_not_fatal(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert client_camera.yolo_placement() == ("cpu", False)


def test_cuda_that_raises_falls_back_to_the_cpu(monkeypatch):
    def boom() -> bool:
        raise RuntimeError("CUDA driver is broken")

    cuda = SimpleNamespace(is_available=boom, device_count=lambda: 0)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    assert client_camera.yolo_placement() == ("cpu", False)


def test_the_service_drops_fp16_on_a_cpu_only_pc(monkeypatch):
    _fake_torch(monkeypatch, available=False)
    service = client_camera.CameraService(SimpleNamespace(enabled=True, half=True))
    assert service.device == "cpu"
    assert service.half is False, "FP16 means nothing on the CPU"


def test_the_service_keeps_fp16_when_the_card_is_there(monkeypatch):
    _fake_torch(monkeypatch, available=True)
    service = client_camera.CameraService(SimpleNamespace(enabled=True, half=True))
    assert service.device == 0
    assert service.half is True
