from __future__ import annotations
import numpy as np
import pandas as pd

def confidence_gate(classes: np.ndarray, proba: np.ndarray, open_thr: float, close_thr: float | None = None) -> np.ndarray:
    """
    Apply confidence gating/hysteresis:
      - open/flip only if max proba > open_thr
      - optionally hold position until max proba falls below close_thr (default: open_thr)
    """
    if close_thr is None:
        close_thr = open_thr
    pos = np.zeros_like(classes, dtype=float)
    cur = 0.0
    for i, (c, p) in enumerate(zip(classes, proba)):
        m = float(np.max(p))
        k = int(np.argmax(p))
        proposed = (-1.0 if k == 0 else 0.0 if k == 1 else 1.0)  # order [-1,0,1]
        if cur == 0.0:
            if m > open_thr:
                cur = proposed
        else:
            if m < close_thr:
                cur = 0.0
            else:
                if proposed != cur and m > open_thr:
                    cur = proposed
        pos[i] = cur
    return pos

def sized_position(classes: np.ndarray, proba: np.ndarray, base: float = 1.0, scale: float = 2.0, open_thr: float = 0.5) -> np.ndarray:
    """
    Continuous position sizing: size = base * ((max_proba - 0.5)*scale)+
    """
    idx = np.argmax(proba, axis=1)
    signed = np.where(idx==0, -1.0, np.where(idx==1, 0.0, 1.0))
    conf = np.clip((np.max(proba, axis=1) - 0.5) * scale, 0.0, 1.0)
    conf = np.where(np.max(proba, axis=1) >= open_thr, conf, 0.0)
    return base * signed * conf
