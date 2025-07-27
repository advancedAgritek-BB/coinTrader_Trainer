"""Tests for DirectML device detection helpers."""

import importlib
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import logging

import dml_utils


def test_get_dml_device_fallback(monkeypatch, caplog):
    if "torch_directml" in sys.modules:
        monkeypatch.delitem(sys.modules, "torch_directml", raising=False)
    importlib.reload(dml_utils)
    with caplog.at_level(logging.WARNING):
        dev = dml_utils.get_dml_device()
    assert getattr(dev, "type", dev) == "cpu"
    assert any("DirectML not available" in r.message for r in caplog.records)


def test_get_dml_device_directml(monkeypatch, caplog):
    dml_module = types.ModuleType("torch_directml")

    class FakeDevice:
        def __init__(self):
            self.type = "dml"

    def fake_device(index):
        assert index == 0
        return FakeDevice()

    dml_module.device = fake_device
    monkeypatch.setitem(sys.modules, "torch_directml", dml_module)

    torch_module = types.ModuleType("torch")

    class FakeTensor:
        def __init__(self, data, device=None):
            self.data = data
            self.device = device

    torch_module.device = lambda name=None: types.SimpleNamespace(type=name)
    torch_module.tensor = lambda data, device=None: FakeTensor(data, device)
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    importlib.reload(dml_utils)
    with caplog.at_level(logging.INFO):
        dev = dml_utils.get_dml_device()
        tensor = torch_module.tensor([1, 2, 3], device=dev)

    assert getattr(dev, "type", None) == "dml"
    assert tensor.device is dev
    assert any("Using DirectML device" in r.message for r in caplog.records)


def test_get_dml_device_rocm(monkeypatch, caplog):
    if "torch_directml" in sys.modules:
        monkeypatch.delitem(sys.modules, "torch_directml", raising=False)

    torch_module = types.ModuleType("torch")

    class Version:
        hip = "5.7.1"

    torch_module.version = Version()
    torch_module.device = lambda name=None: types.SimpleNamespace(type=name)
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    importlib.reload(dml_utils)
    with caplog.at_level(logging.INFO):
        dev = dml_utils.get_dml_device()

    assert getattr(dev, "type", None) == "cuda"
    assert any("ROCm" in r.message for r in caplog.records)
