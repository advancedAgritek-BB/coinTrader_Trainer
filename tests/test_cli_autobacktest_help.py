import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src"


def test_cli_help_has_autobacktest():
    result = subprocess.run(
        [sys.executable, "-m", "cointrainer.cli", "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    assert result.returncode == 0
    assert "autobacktest" in result.stdout
