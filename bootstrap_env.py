from __future__ import annotations

import logging

from config import load_config


def main() -> None:
    """Validate environment variables using :func:`load_config`."""
    try:
        cfg = load_config()
    except Exception as exc:  # pragma: no cover - log and re-raise
        logging.error("Environment bootstrap failed: %s", exc)
        raise
    logging.info("Loaded configuration for %s", cfg.supabase_url)


if __name__ == "__main__":
    main()
