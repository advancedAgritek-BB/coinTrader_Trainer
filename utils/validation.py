import pandas as pd
from typing import Iterable


def validate_schema(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Validate that ``df`` contains all columns in ``required``.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame to validate.
    required : Iterable[str]
        Column names that must be present in ``df``.

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

