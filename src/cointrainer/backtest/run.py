from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cointrainer.io.csv7 import read_csv7
from cointrainer.train.local_csv import make_features

try:  # optional heavy dep
    import joblib
except Exception:  # pragma: no cover - joblib always installed in runtime
    joblib = None  # type: ignore


def _load_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, parse_dates=[0], index_col=0)
    except Exception:
        df = read_csv7(path)
    return df.sort_index()


def backtest_csv(
    *,
    path: Path,
    symbol: str,
    model_local: Path,
    outdir: Path,
    open_thr: float,
    close_thr: float | None,
    fee_bps: float,
    slip_bps: float,
    position_mode: str,
) -> dict[str, Any]:
    """Run a simple threshold-based backtest over a CSV.

    The CSV is loaded, features are computed and the LightGBM model stored in
    ``model_local`` is used to generate class probabilities.  Signals are
    derived from these probabilities using ``open_thr`` and ``position_mode``
    and a very small PnL simulation is performed.
    """

    df = _load_csv(path)
    X = make_features(df).dropna()
    df = df.loc[X.index]

    if joblib is None:  # pragma: no cover - defensive
        raise RuntimeError("joblib is required to load the trained model")
    model = joblib.load(model_local)
    proba = model.predict_proba(X.values)
    classes = list(model.classes_)
    p_short = proba[:, classes.index(-1)]
    p_long = proba[:, classes.index(1)]

    if position_mode == "sized":
        raw = p_long - p_short
        signals = np.where(np.abs(raw) >= open_thr, raw, 0.0)
    else:  # gated
        signals = np.where(p_long >= open_thr, 1.0, np.where(p_short >= open_thr, -1.0, 0.0))

    rets = df["close"].pct_change().fillna(0.0)
    strat_rets = rets * signals
    diff = np.diff(signals, prepend=0.0)
    cost = (fee_bps + slip_bps) / 10000.0
    strat_rets -= cost * np.abs(diff)

    equity = (1.0 + strat_rets).cumprod()
    final_equity = float(equity.iloc[-1])
    periods_per_year = 365 * 24 * 60  # 1m bars
    if len(strat_rets) > 0:
        cagr = float(equity.iloc[-1] ** (periods_per_year / len(strat_rets)) - 1)
        std = strat_rets.std(ddof=0)
        sharpe = float(np.sqrt(periods_per_year) * strat_rets.mean() / std) if std > 0 else 0.0
    else:
        cagr = 0.0
        sharpe = 0.0

    stats: dict[str, Any] = {
        "cagr": cagr,
        "sharpe": sharpe,
        "final_equity": final_equity,
    }

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{symbol.lower()}_signals.csv").write_text(
        "ts,signal\n" + "\n".join(f"{ts},{sig}" for ts, sig in zip(df.index, signals, strict=False))
    )

    return {"stats": stats}


__all__ = ["backtest_csv"]
