import importlib
import torch
import config


def test_use_gpu_true(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    importlib.reload(config)
    assert config.Config.USE_GPU is True


def test_use_gpu_false(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    importlib.reload(config)
    assert config.Config.USE_GPU is False
