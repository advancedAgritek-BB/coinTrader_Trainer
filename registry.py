import warnings
from cointrainer.registry import *  # noqa: F401,F403

warnings.warn(
    "Importing from top-level 'registry' is deprecated; use 'cointrainer.registry'.",
    DeprecationWarning,
    stacklevel=2,
)
