import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'src'


def test_federated_flags_exclusive():
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'cointrainer.cli',
            'train',
            'regime',
            '--federated',
            '--true-federated',
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, 'PYTHONPATH': str(ROOT)},
    )
    assert result.returncode != 0
    assert 'not allowed with argument' in result.stderr.decode().lower()
