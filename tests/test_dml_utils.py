import os
import sys
import types
import importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import dml_utils


def test_get_dml_device_fallback(monkeypatch):
    if 'torch_directml' in sys.modules:
        monkeypatch.delitem(sys.modules, 'torch_directml', raising=False)
    importlib.reload(dml_utils)
    dev = dml_utils.get_dml_device()
    assert getattr(dev, 'type', dev) == 'cpu'


def test_get_dml_device_directml(monkeypatch):
    module = types.ModuleType('torch_directml')

    class FakeDevice:
        def __init__(self):
            self.type = 'dml'

    module.device = lambda: FakeDevice()
    monkeypatch.setitem(sys.modules, 'torch_directml', module)
    importlib.reload(dml_utils)
    dev = dml_utils.get_dml_device()
    assert getattr(dev, 'type', None) == 'dml'

