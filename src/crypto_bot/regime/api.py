"""Runtime prediction API for regime classification."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd

from crypto_bot.config import Config
from cointrainer import registry as _registry
from cointrainer.train.enqueue import enqueue_retrain

Action = Literal["long", "flat", "short"]


@dataclass
class Prediction:
    action: Action
    score: float
    regime: Optional[str] = None
    meta: dict | None = None


_CLASS_MAP = ["short", "flat", "long"]
_FALLBACK_B64_PATH = Path(__file__).resolve().parents[3] / "fallback_b64.txt"
try:  # pragma: no cover - file may be absent in some installs
    _FALLBACK_MODEL_B64 = _FALLBACK_B64_PATH.read_text().strip()
except Exception:  # pragma: no cover - fallback not bundled
    _FALLBACK_MODEL_B64 = ""

_MODEL: object | None = None


def _load_model() -> object:
    """Load the regime model, falling back to embedded base64."""

    global _MODEL
    if _MODEL is not None:
        return _MODEL

    prefix = f"{Config.MODELS_BUCKET}/regime/{Config.SYMBOL}"
    try:
        data = _registry.load_latest(prefix)
        _MODEL = joblib.load(BytesIO(data))
        return _MODEL
    except Exception:
        try:  # pragma: no cover - external side effect
            enqueue_retrain("regime", symbol=Config.SYMBOL)
        except Exception:
            pass
        if not _FALLBACK_MODEL_B64:
            raise RuntimeError("Fallback model missing")
        data = base64.b64decode(_FALLBACK_MODEL_B64)
        _MODEL = joblib.load(BytesIO(data))
        return _MODEL


def predict(features: pd.DataFrame) -> Prediction:
    """Return a :class:`Prediction` for the given ``features``."""

    model = _load_model()
    proba = model.predict_proba(features.tail(1))  # type: ignore[attr-defined]
    row = proba[-1]
    idx = int(row.argmax())
    score = float(row.max())
    action = _CLASS_MAP[idx]
    return Prediction(action=action, score=score)
