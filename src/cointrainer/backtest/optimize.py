from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from cointrainer.backtest.run import backtest_csv
from cointrainer.io.csv7 import read_csv7
from cointrainer.train.local_csv import TrainConfig, train_from_csv7


def _train_one(
    csv_path: Path,
    symbol: str,
    horizon: int,
    hold: float,
    outdir: Path,
    device_type: str = "gpu",
    max_bin: int = 63,
    n_jobs: int | None = 0,
    limit_rows: int | None = None,
) -> Path:
    cfg = TrainConfig(
        symbol=symbol,
        horizon=horizon,
        hold=hold,
        outdir=outdir,
        publish_to_registry=False,
    )
    df = None
    try:
        df = pd.read_csv(csv_path, parse_dates=[0], index_col=0).sort_index()
    except Exception:
        df = read_csv7(csv_path)
    if limit_rows and limit_rows > 0:
        df = df.tail(int(limit_rows))
        tmp = outdir / f"{symbol}_tmp.normalized.csv"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(tmp, index=True)
        csv_path = tmp

    cfg.device_type = device_type  # type: ignore[attr-defined]
    cfg.max_bin = max_bin  # type: ignore[attr-defined]
    cfg.n_jobs = n_jobs  # type: ignore[attr-defined]

    model, meta = train_from_csv7(csv_path, cfg)
    path = cfg.outdir / f"{symbol.lower()}_regime_lgbm.pkl"
    return path


def optimize_grid(
    csv_path: Path,
    symbol: str,
    horizons: list[int],
    holds: list[float],
    open_thrs: list[float],
    position_modes: list[str],
    fee_bps: float,
    slip_bps: float,
    device_type: str = "gpu",
    max_bin: int = 63,
    n_jobs: int | None = 0,
    limit_rows: int | None = None,
    outdir: Path = Path("out/opt"),
) -> dict[str, Any]:
    out = outdir / symbol
    out.mkdir(parents=True, exist_ok=True)
    leaderboard = []

    for H in horizons:
        for hold in holds:
            model_path = _train_one(
                csv_path, symbol, H, hold, out, device_type, max_bin, n_jobs, limit_rows
            )
            for thr in open_thrs:
                for pos_mode in position_modes:
                    res = backtest_csv(
                        path=csv_path,
                        symbol=symbol,
                        model_local=model_path,
                        outdir=out / "backtests",
                        open_thr=thr,
                        close_thr=None,
                        fee_bps=fee_bps,
                        slip_bps=slip_bps,
                        position_mode=pos_mode,
                    )
                    row = {
                        "horizon": H,
                        "hold": hold,
                        "open_thr": thr,
                        "position": pos_mode,
                    }
                    row.update(res["stats"])
                    leaderboard.append(row)

    lb = pd.DataFrame(leaderboard).sort_values(
        ["cagr", "sharpe", "final_equity"], ascending=[False, False, False]
    )
    lb_path = out / "leaderboard.csv"
    lb.to_csv(lb_path, index=False)
    best = lb.iloc[0].to_dict()
    (out / "best.json").write_text(json.dumps(best, indent=2))
    return {"leaderboard": lb_path, "best": best}


def optimize_optuna(
    csv_path: Path,
    symbol: str,
    horizons: list[int],
    holds: list[float],
    open_thrs: list[float],
    position_modes: list[str],
    fee_bps: float,
    slip_bps: float,
    *,
    n_trials: int = 30,
    device_type: str = "gpu",
    max_bin: int = 63,
    n_jobs: int | None = 0,
    limit_rows: int | None = None,
    outdir: Path = Path("out/opt"),
) -> dict[str, Any]:
    try:
        import optuna
    except Exception as e:  # pragma: no cover - optional dep
        raise RuntimeError("optuna is required for --optuna search") from e

    out = outdir / symbol
    out.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        H = trial.suggest_categorical("horizon", horizons)
        hold = trial.suggest_categorical("hold", holds)
        thr = trial.suggest_categorical("open_thr", open_thrs)
        pos_mode = trial.suggest_categorical("position", position_modes)
        model_path = _train_one(
            csv_path, symbol, int(H), float(hold), out, device_type, max_bin, n_jobs, limit_rows
        )
        res = backtest_csv(
            path=csv_path,
            symbol=symbol,
            model_local=model_path,
            outdir=out / "backtests",
            open_thr=float(thr),
            close_thr=None,
            fee_bps=fee_bps,
            slip_bps=slip_bps,
            position_mode=str(pos_mode),
        )
        trial.set_user_attr("stats", res["stats"])
        return float(res["stats"]["cagr"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    rows = []
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
    return {"leaderboard": lb_path, "best": best}


__all__ = ["optimize_grid", "optimize_optuna"]
