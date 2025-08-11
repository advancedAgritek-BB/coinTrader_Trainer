import warnings
from cointrainer.features.build import *  # noqa: F401,F403

warnings.warn(
    "Importing from top-level 'feature_engineering' is deprecated; use 'cointrainer.features.build'.",
    DeprecationWarning,
    stacklevel=2,
)
