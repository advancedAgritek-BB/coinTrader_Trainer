"""Tests for OpenCL device verification."""

import os
import sys
import types
import importlib
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import opencl_utils


class FakeDevice:
    def __init__(self, vendor: str):
        self.vendor = vendor


class FakePlatform:
    def __init__(self, devices):
        self._devices = devices

    def get_devices(self, device_type=None):  # type: ignore[override]
        return self._devices


def test_verify_opencl_success(monkeypatch):
    fake_platform = FakePlatform([FakeDevice("Advanced Micro Devices, Inc.")])
    cl_module = types.SimpleNamespace(get_platforms=lambda: [fake_platform])
    monkeypatch.setitem(sys.modules, "pyopencl", cl_module)
    importlib.reload(opencl_utils)
    assert opencl_utils.verify_opencl() is True


def test_verify_opencl_no_amd(monkeypatch):
    fake_platform = FakePlatform([FakeDevice("NVIDIA")])
    cl_module = types.SimpleNamespace(get_platforms=lambda: [fake_platform])
    monkeypatch.setitem(sys.modules, "pyopencl", cl_module)
    importlib.reload(opencl_utils)
    with pytest.raises(ValueError):
        opencl_utils.verify_opencl()
