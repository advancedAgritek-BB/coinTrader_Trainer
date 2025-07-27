from __future__ import annotations

import asyncio
import functools
import logging
from time import perf_counter
from typing import Any, Callable, TypeVar

T = TypeVar("T")
logger = logging.getLogger(__name__)


def timed(func: Callable[..., T]) -> Callable[..., T]:
    """Measure runtime of ``func`` using ``perf_counter``.

    When the wrapped function is called with ``log_time=True`` the elapsed
    time is logged at ``INFO`` level. Otherwise it is printed to stdout.
    """

    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            log_flag = kwargs.pop("log_time", False)
            start = perf_counter()
            result = await func(*args, **kwargs)
            elapsed = perf_counter() - start
            message = f"{func.__name__} took {elapsed:.3f}s"
            if not log_flag:
                print(message)
            logger.info(message)
            return result

        return async_wrapper

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> T:
        log_flag = kwargs.pop("log_time", False)
        start = perf_counter()
        result = func(*args, **kwargs)
        elapsed = perf_counter() - start
        message = f"{func.__name__} took {elapsed:.3f}s"
        if not log_flag:
            print(message)
        logger.info(message)
        return result

    return sync_wrapper
