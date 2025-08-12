import pandas as pd

from cointrainer.features import indicators as ind


def test_rsi_series_name():
    s = pd.Series([1, 2, 3, 4, 5], name="close")
    r = ind.rsi(s, period=2)
    assert r.name == "rsi_2"
    assert len(r) == len(s)


def test_ema_matches_pandas():
    s = pd.Series([1, 2, 3, 4], name="close")
    e = ind.ema(s, span=3)
    expected = s.ewm(span=3, adjust=False).mean()
    pd.testing.assert_series_equal(e, expected.rename("ema_3"))


def test_obv_simple():
    close = pd.Series([1, 2, 1, 2], name="close")
    vol = pd.Series([1, 1, 1, 1])
    o = ind.obv(close, vol)
    assert list(o) == [0, 1, 0, 1]
    assert o.name == "obv"
