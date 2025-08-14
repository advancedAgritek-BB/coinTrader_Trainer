from pathlib import Path
import pandas as pd

from cointrainer.utils.batch import (
    derive_symbol_from_filename,
    is_csv7,
    is_normalized_csv,
)


def test_symbol_derivation():
    assert derive_symbol_from_filename("XRPUSD_1.csv") == "XRPUSD"
    assert derive_symbol_from_filename("ethusdt-1m.csv") == "ETHUSDT"
    assert derive_symbol_from_filename("ADAUSD.csv") == "ADAUSD"


def test_format_detection(tmp_path: Path):
    p7 = tmp_path / "x.csv"
    p7.write_text(
        "1495122660,0.35,0.35,0.35,0.35,2.0,1\n"
        "1495122720,0.35,0.36,0.34,0.35,3.0,2\n"
    )
    assert is_csv7(p7) is True

    pn = tmp_path / "y.csv"
    df = pd.DataFrame(
        {
            "ts": [1, 2],
            "open": [1, 1],
            "high": [1, 1],
            "low": [1, 1],
            "close": [1, 1],
            "volume": [1, 1],
        }
    )
    df.to_csv(pn, index=False)
    assert is_normalized_csv(pn) is True
