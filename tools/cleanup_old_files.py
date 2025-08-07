#!/usr/bin/env python3
"""Remove files older than today from a target directory.

The script walks a directory tree and deletes any file whose last
modification date is earlier than the current day. By default the
repository root is scanned, but an alternate directory can be supplied
as a positional argument.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path


def cleanup(target_dir: Path) -> None:
    """Delete files in ``target_dir`` older than today."""

    today = date.today()
    for path in target_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            file_date = date.fromtimestamp(path.stat().st_mtime)
        except OSError as exc:
            print(f"Skipping {path}: {exc}")
            continue
        if file_date < today:
            try:
                path.unlink()
                print(f"Deleted {path}")
            except PermissionError:
                print(f"Permission denied: {path}")
            except FileNotFoundError:
                print(f"File not found: {path}")
            except OSError as exc:
                print(f"Could not delete {path}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove files older than today in the given directory."
    )
    default_dir = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "target",
        nargs="?",
        default=str(default_dir),
        help="Directory to scan (defaults to repository root)",
    )
    args = parser.parse_args()

    target = Path(args.target).expanduser().resolve()
    if not target.is_dir():
        parser.error(f"{target} is not a directory")

    cleanup(target)


if __name__ == "__main__":
    main()
