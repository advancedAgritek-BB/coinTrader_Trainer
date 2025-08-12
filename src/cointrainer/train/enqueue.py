"""Stub for training task enqueueing.

This module defines a no-op function ``enqueue_retrain`` which will
later be wired to a real asynchronous job queue. Runtime code can call
this function without introducing heavy dependencies.
"""

from __future__ import annotations


def enqueue_retrain(_agent: str, **kwargs) -> None:
    """Stub retrain enqueue function.

    Parameters
    ----------
    _agent: str
        Identifier of the agent to retrain.
    **kwargs:
        Additional keyword arguments.
    """
    return None
