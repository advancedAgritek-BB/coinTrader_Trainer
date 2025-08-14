import sys
from pathlib import Path

# Ensure the src directory is on sys.path so that the ``crypto_bot`` package is
# importable when tests are executed in isolation.
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from crypto_bot.risk import RiskConfig


def test_risk_config_accepts_extra_fields():
    cfg = RiskConfig(vol_horizon_secs=7200, max_drawdown=0.1)
    assert cfg.vol_horizon_secs == 7200
    assert cfg.extra["max_drawdown"] == 0.1
