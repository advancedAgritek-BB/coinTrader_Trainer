import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src"

def test_cli_help_runs():
    result = subprocess.run(
        [sys.executable, "-m", "cointrainer.cli", "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    assert result.returncode == 0
    assert "import-csv7" in result.stdout
    assert "csv-train" in result.stdout
