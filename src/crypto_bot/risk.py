"""Risk configuration models used by the trading bot.

This module introduces :class:`RiskConfig` which mirrors the behaviour of
runtime dataclasses used in the full coinTrader2.0 application. The primary
motivation for adding this lightweight implementation is to avoid runtime
failures when configuration files specify the ``vol_horizon_secs`` field. The
previous implementation did not accept this argument and raised a ``TypeError``
when the bot started.

The class intentionally accepts arbitrary additional keyword arguments so that
new configuration fields can be added without immediately requiring code
changes in dependent projects. Unknown fields are stored on the ``extra``
attribute for potential downstream use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(init=False)
class RiskConfig:
    """Container for risk related settings.

    Parameters
    ----------
    vol_horizon_secs:
        Horizon in seconds used when computing volatility based risk limits.
        Defaults to ``3600`` (one hour).
    **kwargs:
        Any additional keyword arguments are retained on :attr:`extra` so that
        new configuration options do not break instantiation.
    """

    vol_horizon_secs: int = 3600
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    def __init__(self, vol_horizon_secs: int = 3600, **kwargs: Any) -> None:
        self.vol_horizon_secs = vol_horizon_secs
        # Preserve unknown arguments for forward compatibility
        self.extra = dict(kwargs)
