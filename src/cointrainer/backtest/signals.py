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
