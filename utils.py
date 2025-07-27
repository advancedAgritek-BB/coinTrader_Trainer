from __future__ import annotations

import functools
import logging
import time
import types
from typing import Any, Callable, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


def timed(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator that logs the execution time of ``func``."""

    if "logger" not in func.__globals__:
        func.__globals__["logger"] = logger
    if "time" not in func.__globals__:
        func.__globals__["time"] = time

    def inner(*args: Any, **kwargs: Any) -> T:
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        func.__globals__["logger"].info("%s took %.3fs", func.__name__, elapsed)
        return result

    wrapper = types.FunctionType(
        inner.__code__, func.__globals__, func.__name__, inner.__defaults__, inner.__closure__
    )
    return functools.update_wrapper(wrapper, func)
