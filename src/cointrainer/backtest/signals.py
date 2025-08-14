from __future__ import annotations

import numpy as np
from typing import Optional

__all__ = ["confidence_gate", "sized_position"]

def confidence_gate(
    _base: np.ndarray,
    proba: np.ndarray,
    *,
    open_thr: float = 0.55,
    close_thr: Optional[float] = None,
) -> np.ndarray:
    """Convert class probabilities to {-1,0,1} positions using gating.

    Parameters
    ----------
    _base : np.ndarray
        Placeholder for existing positions (unused currently).
    proba : np.ndarray
        Array of shape (n_samples, 3) with probabilities ordered as [-1,0,1].
    open_thr : float
        Minimum probability required to open a position.
    close_thr : float | None
        Probability threshold to keep an existing position open. If ``None``
        the ``open_thr`` is used.
    """

    thr_close = close_thr if close_thr is not None else open_thr
    out: list[int] = []
    current = 0
    for p_short, _, p_long in proba:
        if current == 0:
            if p_long >= open_thr and p_long >= p_short:
                current = 1
            elif p_short >= open_thr and p_short > p_long:
                current = -1
        else:
            curr_prob = p_long if current > 0 else p_short
            if curr_prob < thr_close:
                current = 0
                if p_long >= open_thr and p_long >= p_short:
                    current = 1
                elif p_short >= open_thr and p_short > p_long:
                    current = -1
        out.append(current)
    return np.asarray(out, dtype=int)

def sized_position(
    _base: np.ndarray,
    proba: np.ndarray,
    *,
    base: float = 1.0,
    scale: float = 2.0,
    open_thr: float = 0.55,
) -> np.ndarray:
    """Size positions proportional to probability confidence.

    Parameters
    ----------
    proba : np.ndarray
        Probabilities ordered as [-1,0,1].
    base : float
        Base position size when threshold met.
    scale : float
        Additional scale applied above ``open_thr``.
    open_thr : float
        Minimum probability to take a position.
    """
    out: list[float] = []
    for p_short, _, p_long in proba:
        if p_long >= open_thr and p_long > p_short:
            size = base + scale * (p_long - open_thr) / (1 - open_thr)
            out.append(size)
        elif p_short >= open_thr and p_short > p_long:
            size = base + scale * (p_short - open_thr) / (1 - open_thr)
            out.append(-size)
        else:
            out.append(0.0)
    return np.asarray(out, dtype=float)
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
