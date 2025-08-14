from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cointrainer.features.simple_indicators import atr, ema, obv, roc, rsi
from cointrainer.io.csv7 import read_csv7

try:
    # Optional at import time; actual training imports happen inside _fit_model()
    from lightgbm import LGBMClassifier  # type: ignore
except Exception:
    LGBMClassifier = None  # type: ignore

@dataclass
class TrainConfig:
    symbol: str = "XRPUSD"
    horizon: int = 15  # bars
    hold: float = 0.0015  # 0.15%
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 63
    random_state: int = 42
    outdir: Path = Path("local_models")
    write_predictions: bool = True
    publish_to_registry: bool = False  # if True and env is present, also publish to registry
    # GPU / performance knobs
    device_type: str = "gpu"          # "cpu" | "gpu" | "cuda"
    gpu_platform_id: int | None = None  # -1 means default
    gpu_device_id: int | None = None    # -1 means default
    max_bin: int = 63                  # GPU best practice
    gpu_use_dp: bool = False           # single-precision by default
    n_jobs: int | None = None          # threads for LightGBM wrapper

FEATURE_LIST = ["ema_8","ema_21","rsi_14","atr_14","roc_5","obv"]

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    X = pd.DataFrame(index=df.index)
    X["ema_8"] = ema(close, 8)
    X["ema_21"] = ema(close, 21)
    X["rsi_14"] = rsi(close, 14)
    X["atr_14"] = atr(high, low, close, 14)
    X["roc_5"] = roc(close, 5)
    X["obv"] = obv(close, volume)
    return X

def make_labels(close: pd.Series, horizon: int, hold: float) -> pd.Series:
    future_ret = close.pct_change(horizon).shift(-horizon)
    y = np.where(future_ret >  hold,  1, np.where(future_ret < -hold, -1, 0))
    return pd.Series(y, index=close.index)

def _fit_model(X: pd.DataFrame, y: pd.Series, cfg: TrainConfig):
    if LGBMClassifier is None:
        raise RuntimeError(
            "LightGBM is not installed. Install with: pip install lightgbm"
        )

    params = {
        "n_estimators": cfg.n_estimators,
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "objective": "multiclass",
        "num_class": 3,
        "class_weight": "balanced",
        "n_jobs": cfg.n_jobs if cfg.n_jobs is not None else -1,
        "random_state": cfg.random_state,
        "device_type": cfg.device_type,
        "max_bin": cfg.max_bin,
        "gpu_use_dp": cfg.gpu_use_dp,
    }
    if cfg.gpu_platform_id is not None:
        params["gpu_platform_id"] = cfg.gpu_platform_id
    if cfg.gpu_device_id is not None:
        params["gpu_device_id"] = cfg.gpu_device_id

    try:
        model = LGBMClassifier(**params)
        model.fit(X, y)
        return model
    except Exception:
        if cfg.device_type != "cpu":
            params["device_type"] = "cpu"
            params.pop("gpu_platform_id", None)
            params.pop("gpu_device_id", None)
            model = LGBMClassifier(**params)
            model.fit(X, y)
            return model
        raise

def _save_local(model, cfg: TrainConfig, metadata: dict) -> Path:
    import json

    import joblib
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    path = cfg.outdir / f"{cfg.symbol.lower()}_regime_lgbm.pkl"
    joblib.dump(model, path)
    (cfg.outdir / f"{cfg.symbol.lower()}_metadata.json").write_text(json.dumps(metadata))
    return path

def _maybe_publish_registry(model_bytes: bytes, metadata: dict, cfg: TrainConfig) -> str | None:
    if not cfg.publish_to_registry:
        return None
    try:
        from cointrainer import registry  # lazy import
        ts = time.strftime("%Y%m%d-%H%M%S")
        key = f"models/regime/{cfg.symbol}/{ts}_regime_lgbm.pkl"
        registry.save_model(key, model_bytes, metadata)
        return key
    except Exception:
        return None

def train_from_csv7(
    csv_path: Path | str, cfg: TrainConfig, *, limit_rows: int | None = None
) -> tuple[object, dict]:
    df = read_csv7(csv_path)
    if limit_rows and limit_rows > 0:
        # Take the tail (most recent) rows
        df = df.tail(int(limit_rows))
    X_all = make_features(df).dropna()
    y_all = make_labels(df.loc[X_all.index, "close"], cfg.horizon, cfg.hold)
    m = y_all.notna()
    X = X_all[m]
    y = y_all[m]

    model = _fit_model(X, y, cfg)

    metadata = {
        "schema_version": "1",
        "feature_list": FEATURE_LIST,
        "label_order": [-1, 0, 1],
        "horizon": f"{cfg.horizon}m",
        "thresholds": {"hold": cfg.hold},
        "symbol": cfg.symbol,
    }

    # Save local
    _save_local(model, cfg, metadata)

    # Optional registry publish
    try:
        import io

        import joblib
        buf = io.BytesIO()
        joblib.dump(model, buf)
        _maybe_publish_registry(buf.getvalue(), metadata, cfg)
    except Exception:
        pass

    # Optional predictions CSV for inspection
    if cfg.write_predictions:
        try:
            proba = model.predict_proba(X.values)
            idx = proba.argmax(axis=1)
            index_to_class = [-1, 0, 1]
            classes = [index_to_class[i] for i in idx]
            score = proba.max(axis=1)
            out = pd.DataFrame(index=X.index)
            out["class"] = classes
            out["action"] = pd.Series(classes, index=out.index).map({-1:"short",0:"flat",1:"long"})
            out["score"] = score
            out_path = cfg.outdir / f"{cfg.symbol.lower()}_predictions.csv"
            out.to_csv(out_path, index=True)
        except Exception:
            pass

    return model, metadata
