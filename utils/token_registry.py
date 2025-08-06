from __future__ import annotations

import subprocess
from pathlib import Path

_CRON_SCHEDULES = {
    "daily": "0 0 * * *",
    "weekly": "0 0 * * 0",
}


def schedule_retrain(model: str, interval: str) -> None:
    """Schedule periodic retraining of ``model`` using cron.

    Parameters
    ----------
    model: str
        Identifier of the model to retrain.
    interval: str
        Either ``"daily"`` or ``"weekly"`` describing how often the job
        should run.

    Raises
    ------
    ValueError
        If ``interval`` is not supported.
    RuntimeError
        If the ``crontab`` command is not available.
    """

    key = interval.lower()
    if key not in _CRON_SCHEDULES:
        raise ValueError(
            f"Unsupported interval '{interval}'. Supported intervals: {', '.join(_CRON_SCHEDULES)}"
        )

    cron_timing = _CRON_SCHEDULES[key]
    trainer_path = Path(__file__).resolve().parents[1] / "ml_trainer.py"
    command = f"{cron_timing} python {trainer_path} --model {model}\n"

    try:
        result = subprocess.run(
            ["crontab", "-l"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("crontab command not found") from exc

    existing = result.stdout if result.returncode == 0 else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    new_crontab = existing + command

    subprocess.run(["crontab", "-"], input=new_crontab, text=True, check=True)
