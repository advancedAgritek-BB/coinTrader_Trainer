from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from cointrainer.features.simple_indicators import atr, ema, obv, roc, rsi
from cointrainer.io.csv7 import read_csv7

try:  # Optional at import time; actual training imports happen inside _fit_model()
    from lightgbm import LGBMClassifier  # type: ignore
except Exception:  # pragma: no cover - lightgbm may be absent
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
        "class_weight": "balanced",
        "n_jobs": cfg.n_jobs if cfg.n_jobs is not None else -1,
        "random_state": cfg.random_state,
        "num_class": 3,
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


def _dataset_fingerprint(X: pd.DataFrame, y: pd.Series) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(X, index=True).values.tobytes())
    h.update(pd.util.hash_pandas_object(y, index=True).values.tobytes())
    return h.hexdigest()


def _get_current_git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        )
    except Exception:  # pragma: no cover - git may be absent
        return "unknown"

def _save_local(model, cfg: TrainConfig, metadata: dict) -> Path:
    import json

    import joblib
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    path = cfg.outdir / f"{cfg.symbol.lower()}_regime_lgbm.pkl"
    joblib.dump(model, path)
    (cfg.outdir / f"{cfg.symbol.lower()}_metadata.json").write_text(json.dumps(metadata))
    return path


def _maybe_publish_registry(
    model: object,
    metadata: dict,
    cfg: TrainConfig,
    metrics: dict,
    dataset_hash: str,
    config: dict,
) -> None:
    if not cfg.publish_to_registry:
        return None
    try:
        from cointrainer.registry import SupabaseRegistry

        reg = SupabaseRegistry()
        reg.publish_regime_model(
            model_obj=model,
            symbol=cfg.symbol,
            horizon=f"{cfg.horizon}m",
            feature_list=metadata["feature_list"],
            label_order=metadata["label_order"],
            thresholds={"hold": cfg.hold},
            metrics=metrics,
            config=config,
            code_sha=_get_current_git_sha(),
            data_fingerprint=dataset_hash,
        )
    except Exception:
        print("publish skipped")

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

    # Metrics + optional registry publish
    preds = model.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds, average="macro")),
    }
    fingerprint = _dataset_fingerprint(X, y)
    config = {**vars(cfg), "outdir": str(cfg.outdir)}
    _maybe_publish_registry(model, metadata, cfg, metrics, fingerprint, config)

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
