"""Optuna-based optimization helpers.

This module provides a lightweight wrapper around :mod:`optuna` so the CLI can
perform hyper-parameter searches when the dependency is available.  The real
project contains a much richer implementation with cross validation and
publishing of the resulting model.  For the educational version we keep the
logic intentionally small and rely on the existing grid search utilities as
much as possible.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .optimize import _train_one
from .run import backtest_csv


@dataclass
class OptunaConfig:
    """Configuration for :func:`optimize_optuna`.

    Only a subset of the fields are used in this simplified implementation but
    they mirror the arguments expected by the CLI so tests exercising the CLI do
    not fail if new flags are introduced.
    """

    n_trials: int = 100
    n_folds: int = 4
    val_len: int = 100_000
    gap: int = 500
    limit_rows: int | None = 800_000
    fee_bps: float = 2.0
    slip_bps: float = 0.0
    device_type: str = "gpu"
    max_bin: int = 63
    n_jobs: int = 0
    seed: int = 42
    storage: str | None = None
    study_name: str | None = None
    publish_best: bool = False


def optimize_optuna(
    csv_path: Path,
    symbol: str,
    *,
    outdir: Path = Path("out/opt"),
    cfg: OptunaConfig | None = None,
) -> dict[str, Any]:
    """Run an Optuna search.

    The search space mirrors the simple grid search implemented in
    :func:`cointrainer.backtest.optimize.optimize_grid`.  If Optuna is not
    installed a :class:`RuntimeError` is raised so callers can fall back to the
    grid search implementation.
    """

    try:  # pragma: no cover - handled in tests via fallback
        import optuna
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("optuna is required for optimization") from exc

    cfg = cfg or OptunaConfig()

    out = outdir / symbol
    out.mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=cfg.seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        storage=cfg.storage,
        study_name=cfg.study_name,
        load_if_exists=bool(cfg.storage),
    )

    horizons = [15, 30, 60]
    holds = [0.001, 0.0015, 0.002, 0.003]
    open_thrs = [0.52, 0.55, 0.58]
    pos_modes = ["gated", "sized"]

    def objective(trial: optuna.Trial) -> float:
        H = trial.suggest_categorical("horizon", horizons)
        hold = trial.suggest_categorical("hold", holds)
        thr = trial.suggest_categorical("open_thr", open_thrs)
        pos_mode = trial.suggest_categorical("position", pos_modes)
        model_path = _train_one(
            csv_path,
            symbol,
            int(H),
            float(hold),
            out,
            cfg.device_type,
            cfg.max_bin,
            cfg.n_jobs,
            cfg.limit_rows,
        )
        res = backtest_csv(
            path=csv_path,
            symbol=symbol,
            model_local=model_path,
            outdir=out / "backtests",
            open_thr=float(thr),
            close_thr=None,
            fee_bps=cfg.fee_bps,
            slip_bps=cfg.slip_bps,
            position_mode=str(pos_mode),
        )
        trial.set_user_attr("stats", res["stats"])
        return float(res["stats"].get("cagr", 0.0))

    study.optimize(objective, n_trials=cfg.n_trials)

    rows: list[dict[str, Any]] = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = dict(t.params)
        row.update(t.user_attrs.get("stats", {}))
        rows.append(row)

    lb = pd.DataFrame(rows).sort_values(
        ["cagr", "sharpe", "final_equity"], ascending=[False, False, False]
    )
    lb_path = out / "leaderboard.csv"
    lb.to_csv(lb_path, index=False)

    best_params = study.best_params
    best_stats = study.best_trial.user_attrs.get("stats", {})
    best = dict(best_params)
    best.update(best_stats)
    (out / "best.json").write_text(json.dumps(best, indent=2))

    return {"leaderboard_path": lb_path, "best": best}


def publish_best_model(
    csv_path: Path, symbol: str, outdir: Path, best: dict[str, Any]
) -> str:
    """Retrain the best configuration and return a key or path.

    The real project would upload the resulting model to Supabase.  For testing
    purposes we simply ensure the model file exists and return its path so the
    caller can log it.
    """

    model_dir = outdir / symbol
    model_dir.mkdir(parents=True, exist_ok=True)
    horizon = int(best.get("horizon", 30))
    hold = float(best.get("hold", 0.0015))
    model_path = _train_one(
        csv_path,
        symbol,
        horizon,
        hold,
        model_dir,
        device_type="gpu",
        max_bin=63,
        n_jobs=0,
        limit_rows=None,
    )
    # In the full application this would return a storage key.  Returning the
    # path provides enough information for tests and demonstrations.
    return str(model_path)


__all__ = ["OptunaConfig", "optimize_optuna", "publish_best_model"]

