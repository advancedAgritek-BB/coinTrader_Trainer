import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'src'


def test_cointrainer_help():
    result = subprocess.run(
        [sys.executable, '-m', 'cointrainer.cli', '--help'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, 'PYTHONPATH': str(ROOT)},
    )
    assert result.returncode == 0


def test_train_regime_help():
    result = subprocess.run(
        [sys.executable, '-m', 'cointrainer.cli', 'train', 'regime', '--help'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, 'PYTHONPATH': str(ROOT)},
    )
    assert result.returncode == 0
