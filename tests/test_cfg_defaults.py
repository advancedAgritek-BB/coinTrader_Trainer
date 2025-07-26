import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import train_pipeline
import ml_trainer


def test_train_pipeline_load_cfg_backtest_defaults(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("")
    cfg = train_pipeline.load_cfg(str(cfg_path))
    assert cfg["backtest"]["slippage"] == 0.005
    assert cfg["backtest"]["costs"] == 0.002


def test_ml_trainer_load_cfg_backtest_defaults(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("")
    cfg = ml_trainer.load_cfg(str(cfg_path))
    assert cfg["backtest"]["slippage"] == 0.005
    assert cfg["backtest"]["costs"] == 0.002
