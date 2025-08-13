from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np
import pandas as pd
from io import BytesIO

from crypto_bot.config import Config  # must exist from earlier steps
from cointrainer import registry as _registry

Action = Literal["long", "flat", "short"]


@dataclass
class Prediction:
    action: Action
    score: float
    regime: Optional[str] = None
    meta: dict | None = None


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window).mean()
    down = -delta.clip(upper=0).rolling(window).mean()
    rs = up / (down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def _baseline(features: pd.DataFrame) -> Prediction:
    cols = [c.lower() for c in features.columns]
    if "close" in cols:
        price = features.iloc[:, cols.index("close")]
    else:
        num_cols = features.select_dtypes(include=[np.number]).columns
        price = features[num_cols[-1]] if len(num_cols) else pd.Series(np.arange(len(features)))
    r = _rsi(price).iloc[-1]
    f = price.ewm(span=8, adjust=False).mean().iloc[-1]
    s = price.ewm(span=21, adjust=False).mean().iloc[-1]
    cross = float(np.sign(f - s))
    if r >= 65 and cross >= 0:
        return Prediction("long", float(min(1.0, 0.5 + (r - 65) / 35 + 0.25 * cross)), meta={"source":"fallback"})
    if r <= 35 and cross <= 0:
        return Prediction("short", float(min(1.0, 0.5 + (35 - r) / 35 + 0.25 * (-cross))), meta={"source":"fallback"})
    return Prediction("flat", 0.5, meta={"source":"fallback"})


def _load_model_from_bytes(blob: bytes):
    # Try joblib first; then pickle
    try:
        import joblib  # lazy import
        return joblib.load(BytesIO(blob))
    except Exception:
        import pickle
        return pickle.loads(blob)


def predict(features: pd.DataFrame) -> Prediction:
    """
    Load latest model per Config.REGIME_PREFIX. On any failure, return a baseline Prediction.
    Align features to feature_list and map classes via label_order when available.
    """
    prefix = Config.REGIME_PREFIX
    # Try registry
    try:
        meta = _registry.load_pointer(prefix)  # may raise
        feat_list = meta.get("feature_list", None)
        if feat_list:
            # select & reorder if available subset exists
            columns = [c for c in feat_list if c in features.columns]
            if columns:
                features = features[columns]
        blob = _registry.load_latest(prefix, allow_fallback=False)
        model = _load_model_from_bytes(blob)
        try:
            proba = model.predict_proba(features.tail(1))  # type: ignore[attr-defined]
            proba = np.array(proba).ravel()
            idx = int(np.argmax(proba))
            label_order = meta.get("label_order", [-1, 0, 1])
            mapping = {-1: "short", 0: "flat", 1: "long"}
            class_id = label_order[idx] if idx < len(label_order) else 0
            action = mapping.get(class_id, "flat")
            score = float(proba[idx]) if proba.size else 0.5
            # optional threshold "hold"
            thresholds = meta.get("thresholds") or {}
            hold = float(thresholds.get("hold", 0.0))
            if hold and score < hold:
                action = "flat"
            return Prediction(action=action, score=score, meta={"source": "registry"})
        except Exception:
            # present but inference failed (e.g., feature mismatch)
            return _baseline(features)
    except Exception:
        # registry missing/unconfigured
        return _baseline(features)

