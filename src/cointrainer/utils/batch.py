from __future__ import annotations

from pathlib import Path


def iter_csv_files(folder: str | Path, glob: str = "*.csv", recursive: bool = False) -> list[Path]:
    """Return a list of CSV files under *folder* matching *glob*.

    Parameters
    ----------
    folder: str | Path
        Root directory to search.
    glob: str
        Glob pattern to match files. Defaults to ``"*.csv"``.
    recursive: bool
        If ``True`` search subdirectories recursively.
    """
    root = Path(folder)
    if not root.exists():
        return []
    files = list(root.rglob(glob)) if recursive else list(root.glob(glob))
    return [f for f in files if f.is_file()]


def is_csv7(path: str | Path) -> bool:
    """Return ``True`` if the file appears to be a headerless 7-column CSV."""
    try:
        with Path(path).open("r", encoding="utf-8") as fh:
            first = fh.readline().strip()
    except Exception:
        return False
    parts = first.split(",")
    if len(parts) != 7:
        return False
    for p in parts:
        try:
            float(p)
        except ValueError:
            return False
    return True


def is_normalized_csv(path: str | Path) -> bool:
    """Return ``True`` if the file looks like a normalized OHLCV CSV."""
    try:
        with Path(path).open("r", encoding="utf-8") as fh:
            header = fh.readline().lower()
    except Exception:
        return False
    return all(h in header for h in ["open", "high", "low", "close", "volume"])


def derive_symbol(path: Path, mode: str = "filename", fixed: str | None = None) -> str:
    """Derive a trading symbol from *path* according to *mode*.

    ``mode`` may be ``"filename"`` (default) which uses the stem of the
    filename, ``"parent"`` which uses the name of the parent directory, or
    ``"fixed"`` which returns ``fixed``.
    """
    if mode == "filename":
        return path.stem.split("_")[0].upper()
    if mode == "parent":
        return path.parent.name.upper()
    if mode == "fixed":
        if not fixed:
            raise ValueError("symbol must be provided when mode='fixed'")
        return fixed.upper()
    raise ValueError(f"Unknown derive mode: {mode}")
