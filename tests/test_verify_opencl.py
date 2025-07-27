"""Tests for OpenCL device verification."""

import importlib
import os
import sys
import types

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

    def fake_run(cmd, capture_output=True, text=True, check=False):
        assert cmd == ["rocm-smi", "--showproductname"]
        return types.SimpleNamespace(stdout="GPU[0] : AMD Radeon X", stderr="")

    monkeypatch.setattr(opencl_utils.subprocess, "run", fake_run)

    assert opencl_utils.verify_opencl() is True


def test_verify_opencl_windows(monkeypatch):
    fake_platform = FakePlatform([FakeDevice("Advanced Micro Devices, Inc.")])
    cl_module = types.SimpleNamespace(get_platforms=lambda: [fake_platform])
    monkeypatch.setitem(sys.modules, "pyopencl", cl_module)
    importlib.reload(opencl_utils)
    monkeypatch.setattr(opencl_utils.platform, "system", lambda: "Windows")

    def fake_run(*args, **kwargs):
        raise AssertionError("rocm-smi should not be called on Windows")

    monkeypatch.setattr(opencl_utils.subprocess, "run", fake_run)

    assert opencl_utils.verify_opencl() is True


def test_verify_opencl_no_amd(monkeypatch):
    fake_platform = FakePlatform([FakeDevice("NVIDIA")])
    cl_module = types.SimpleNamespace(get_platforms=lambda: [fake_platform])
    monkeypatch.setitem(sys.modules, "pyopencl", cl_module)
    importlib.reload(opencl_utils)
    with pytest.raises(ValueError):
        opencl_utils.verify_opencl()


def test_verify_opencl_bad_rocm_output(monkeypatch):
    fake_platform = FakePlatform([FakeDevice("Advanced Micro Devices, Inc.")])
    cl_module = types.SimpleNamespace(get_platforms=lambda: [fake_platform])
    monkeypatch.setitem(sys.modules, "pyopencl", cl_module)
    importlib.reload(opencl_utils)

    def fake_run(cmd, capture_output=True, text=True, check=False):
        return types.SimpleNamespace(stdout="no gpu", stderr="")

    monkeypatch.setattr(opencl_utils.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError):
        opencl_utils.verify_opencl()


def test_has_rocm_uses_verify(monkeypatch):
    monkeypatch.setattr(opencl_utils, "verify_opencl", lambda: True)
    assert opencl_utils.has_rocm() is True


def test_has_rocm_env_fallback(monkeypatch):
    def fail():
        raise RuntimeError("no gpu")

    monkeypatch.setattr(opencl_utils, "verify_opencl", fail)
    monkeypatch.setattr(opencl_utils.platform, "system", lambda: "Windows")
    monkeypatch.setenv("ROCM_PATH", "1")
    assert opencl_utils.has_rocm() is True


def test_has_rocm_false(monkeypatch):
    def fail():
        raise RuntimeError("no gpu")

    monkeypatch.setattr(opencl_utils, "verify_opencl", fail)
    monkeypatch.setattr(opencl_utils.platform, "system", lambda: "Linux")
    monkeypatch.delenv("ROCM_PATH", raising=False)
    assert opencl_utils.has_rocm() is False
