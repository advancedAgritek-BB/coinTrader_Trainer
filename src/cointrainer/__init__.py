"""cointrainer package."""

from importlib.metadata import PackageNotFoundError, version
import os


def ensure_env_defaults() -> None:
    """Populate default environment variables used by coinTrader."""

    models_bucket = (
        os.getenv("CT_MODELS_BUCKET")
        or os.getenv("MODELS_BUCKET")
        or "models"
    )
    regime_prefix = os.getenv("CT_REGIME_PREFIX") or "models/regime"
    symbol = os.getenv("SYMBOL", "BTCUSDT")

    os.environ.setdefault("CT_MODELS_BUCKET", models_bucket)
    os.environ.setdefault("CT_REGIME_PREFIX", regime_prefix)
    # Mirror to legacy names for backward compatibility
    os.environ.setdefault("MODELS_BUCKET", models_bucket)
    os.environ.setdefault("REGIME_PREFIX", f"{regime_prefix}/{symbol}")


# Ensure defaults when the package is imported
ensure_env_defaults()

try:
    __version__ = version("cointrader-trainer")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"

__all__ = ["__version__", "ensure_env_defaults"]
