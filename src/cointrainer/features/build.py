"""Lightweight feature and label builders."""

from __future__ import annotations

from typing import Tuple, Dict

import pandas as pd

from .indicators import rsi, atr, ema, bollinger, momentum, adx, obv


def _close_series(df: pd.DataFrame) -> pd.Series:
    if "close" in df.columns:
        return df["close"]
    if "price" in df.columns:
        return df["price"]
    raise KeyError("DataFrame must contain a 'close' or 'price' column")


def build_features(df: pd.DataFrame, *, use: dict, params: dict) -> Tuple[pd.DataFrame, Dict]:
    """Return selected indicator features and accompanying metadata."""

    close = _close_series(df)
    feats = []
    names: list[str] = []

    if use.get("rsi"):
        period = int(params.get("rsi", 14))
        s = rsi(close, period)
        feats.append(s)
        names.append(s.name)
    if use.get("atr") and {"high", "low"}.issubset(df.columns):
        period = int(params.get("atr", 14))
        s = atr(df, period)
        feats.append(s)
        names.append(s.name)
    if use.get("ema"):
        span = int(params.get("ema", 12))
        s = ema(close, span)
        feats.append(s)
        names.append(s.name)
    if use.get("bollinger"):
        window = int(params.get("bollinger", 20))
        n_std = int(params.get("bollinger_n_std", 2))
        s = bollinger(close, window, n_std)
        feats.append(s)
        names.append(s.name)
    if use.get("momentum"):
        period = int(params.get("momentum", 10))
        s = momentum(close, period)
        feats.append(s)
        names.append(s.name)
    if use.get("adx") and {"high", "low"}.issubset(df.columns):
        period = int(params.get("adx", 14))
        s = adx(df, period)
        feats.append(s)
        names.append(s.name)
    if use.get("obv") and "volume" in df.columns:
        s = obv(close, df["volume"])
        feats.append(s)
        names.append(s.name)

    X = pd.concat(feats, axis=1)
    X = X.dropna()
    meta = {"feature_list": names}
    return X, meta


def make_labels(df: pd.DataFrame, *, horizon: str, thresholds: dict) -> Tuple[pd.Series, Dict]:
    """Generate classification labels and metadata."""

    close = _close_series(df)
    try:
        periods = int(horizon)
    except ValueError:
        freq = df.index.to_series().diff().median()
        periods = int(pd.to_timedelta(horizon) / freq) if freq and freq != 0 else 1
    future = close.shift(-periods)
    returns = (future - close) / close
    up = thresholds.get("up", 0)
    down = thresholds.get("down", 0)
    y = pd.Series(0, index=df.index)
    y[returns > up] = 1
    y[returns < -down] = -1
    y = y.iloc[:-periods].astype(int)
    meta = {"label_order": [-1, 0, 1], "horizon": horizon, "thresholds": thresholds}
    return y, meta


def make_features(df: pd.DataFrame, *_, **kwargs) -> pd.DataFrame:
    """Compatibility wrapper around :func:`build_features`. Returns only ``X``."""

    use = kwargs.get("use", {})
    params = kwargs.get("params", {})
    X, _ = build_features(df, use=use, params=params)
    if "target" in df.columns:
        X = X.join(df["target"])
    return X


__all__ = ["build_features", "make_labels", "make_features"]
