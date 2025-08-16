import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cointrainer.utils.pairs import canonical_pair_from_filename, slug_from_canonical


def test_pair_slug_utils():
    assert canonical_pair_from_filename("BTCUSDT_1m.csv") == "BTCUSDT"
    s = slug_from_canonical("BTCUSDT")
    assert s in ("BTC-USDT", "BTC-USD", "BTC-USDT")  # tolerate quote variants
