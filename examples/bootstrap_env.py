from __future__ import annotations

import logging

from config import Config


def main() -> None:
    """Log basic configuration information."""
    logging.info("Loaded configuration for bucket %s", Config.SUPABASE_BUCKET)


if __name__ == "__main__":
    main()
