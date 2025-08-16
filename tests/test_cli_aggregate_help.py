import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src"

def test_cli_has_csv_train_aggregate():
    out = subprocess.run(
        [sys.executable, "-m", "cointrainer.cli", "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    assert out.returncode == 0
    assert "csv-train-aggregate" in out.stdout
