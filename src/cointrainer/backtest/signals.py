from __future__ import annotations

import numpy as np

__all__ = ["confidence_gate", "sized_position"]


def confidence_gate(
    _classes: np.ndarray,
    proba: np.ndarray,
    open_thr: float,
    close_thr: float | None = None,
) -> np.ndarray:
    """Convert class probabilities to {-1,0,1} positions using gating.

    Parameters
    ----------
    _classes : np.ndarray
        Placeholder for existing positions (unused).
    proba : np.ndarray
        Array of shape (n_samples, 3) with probabilities ordered as [-1,0,1].
    open_thr : float
        Minimum probability required to open or flip a position.
    close_thr : float | None
        Probability threshold to keep an existing position open. If ``None``
        the ``open_thr`` is used.
    """

    if close_thr is None:
        close_thr = open_thr
    pos = np.zeros_like(_classes, dtype=float)
    cur = 0.0
    for i, p in enumerate(proba):
        m = float(np.max(p))
        k = int(np.argmax(p))
        proposed = -1.0 if k == 0 else 0.0 if k == 1 else 1.0  # order [-1,0,1]
        if cur == 0.0:
            if m > open_thr:
                cur = proposed
        else:
            if m < close_thr:
                cur = 0.0
            elif proposed != cur and m > open_thr:
                cur = proposed
        pos[i] = cur
    return pos


def sized_position(
    _classes: np.ndarray,
    proba: np.ndarray,
    base: float = 1.0,
    scale: float = 2.0,
    open_thr: float = 0.5,
) -> np.ndarray:
    """Size positions proportional to probability confidence.

    Parameters
    ----------
    _classes : np.ndarray
        Placeholder for existing positions (unused).
    proba : np.ndarray
        Probabilities ordered as [-1,0,1].
    base : float
        Base position size when threshold met.
    scale : float
        Additional scale applied above 0.5 confidence.
    open_thr : float
        Minimum probability required to take a position.
    """

    idx = np.argmax(proba, axis=1)
    signed = np.where(idx == 0, -1.0, np.where(idx == 1, 0.0, 1.0))
    conf = np.clip((np.max(proba, axis=1) - 0.5) * scale, 0.0, 1.0)
    conf = np.where(np.max(proba, axis=1) >= open_thr, conf, 0.0)
    return base * signed * conf

