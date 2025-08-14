from __future__ import annotations
import json, io
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

from cointrainer.features.simple_indicators import ema, rsi, atr, roc, obv
from cointrainer.backtest.signals import confidence_gate, sized_position
from cointrainer.backtest.sim import simulate
from cointrainer.train.local_csv import FEATURE_LIST
from cointrainer.io.csv7 import read_csv7

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    X = pd.DataFrame(index=df.index)
    X["ema_8"]  = ema(close, 8)
    X["ema_21"] = ema(close, 21)
    X["rsi_14"] = rsi(close, 14)
    X["atr_14"] = atr(high, low, close, 14)
    X["roc_5"]  = roc(close, 5)
    X["obv"]    = obv(close, vol)
    return X.dropna()

def load_model_local(path: Path):
    import joblib
    return joblib.load(path)

def load_model_registry(prefix: str):
    from cointrainer.registry import load_pointer, load_latest
    meta = load_pointer(prefix)
    blob = load_latest(prefix)
    model = _unpack_model(blob)
    return model, meta

def _unpack_model(blob: bytes):
    # try joblib then pickle
    try:
        import joblib, io
        return joblib.load(io.BytesIO(blob))
    except Exception:
        import pickle
        return pickle.loads(blob)

def backtest_csv(
    path: Path,
    symbol: str,
    model_local: Optional[Path] = None,
    model_registry_prefix: Optional[str] = None,
    outdir: Path = Path("out") / "backtests",
    open_thr: float = 0.55,
    close_thr: Optional[float] = None,
    fee_bps: float = 2.0,
    slip_bps: float = 0.0,
    position_mode: str = "gated",   # "gated" | "sized"
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, Any]:
    # 1) Load data (normalized or CSV7)
    p = Path(path)
    if p.exists():
        try:
            df = pd.read_csv(p, parse_dates=[0], index_col=0).sort_index()
        except Exception:
            df = read_csv7(p)  # autodetect CSV7
    else:
        raise FileNotFoundError(path)

    if start: df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:   df = df[df.index <= pd.Timestamp(end, tz="UTC")]

    # 2) Build features and align to FEATURE_LIST
    X = build_features(df)
    cols = [c for c in FEATURE_LIST if c in X.columns]
    X = X[cols]
    idx = X.index

    # 3) Load model
    meta = {"feature_list": FEATURE_LIST, "label_order": [-1,0,1]}
    if model_local:
        model = load_model_local(Path(model_local))
    elif model_registry_prefix:
        model, meta = load_model_registry(model_registry_prefix)
        feat_list = meta.get("feature_list", FEATURE_LIST)
        X = X[[c for c in feat_list if c in X.columns]]
    else:
        raise ValueError("Provide model_local or model_registry_prefix")

    # 4) Predict proba
    proba = model.predict_proba(X.values)
    # class order expected [-1,0,1]; if different, map here using meta["label_order"]
    order = meta.get("label_order", [-1,0,1])
    # If order != [-1,0,1], we need to reorder columns of proba accordingly
    if order != [-1,0,1]:
        # build mapping from order index to canonical index
        mapping = {c:i for i,c in enumerate(order)}
        cols_map = [mapping[-1], mapping[0], mapping[1]]
        proba = proba[:, cols_map]

    # 5) Build positions
    if position_mode == "gated":
        pos = confidence_gate(np.zeros(len(X), dtype=int), proba, open_thr=open_thr, close_thr=close_thr)
    else:
        pos = sized_position(np.zeros(len(X), dtype=int), proba, base=1.0, scale=2.0, open_thr=open_thr)

    # 6) Simulate
    res = simulate(df.loc[idx, "close"], pos, fee_bps=fee_bps, slip_bps=slip_bps)

    # 7) Save artifacts
    ddir = outdir / symbol
    ddir.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d-%H%M%S")
    (ddir / f"equity_{ts}.csv").write_text(res["equity"].to_csv(index=True))
    (ddir / f"summary_{ts}.json").write_text(json.dumps(res["stats"], indent=2))
    return res
