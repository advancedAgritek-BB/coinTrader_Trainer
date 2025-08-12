import warnings
from cointrainer.data.loader import *  # noqa: F401,F403

warnings.warn(
    "Importing from top-level 'data_loader' is deprecated; use 'cointrainer.data.loader'.",
    DeprecationWarning,
    stacklevel=2,
)
