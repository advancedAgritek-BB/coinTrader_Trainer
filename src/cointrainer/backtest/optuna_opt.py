from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cointrainer.backtest.run import build_features
from cointrainer.backtest.sim import simulate
from cointrainer.io.csv7 import read_csv7
from cointrainer.train.local_csv import FEATURE_LIST

# ---- utilities ----

def _read_any_csv(path: Path) -> pd.DataFrame:
    """Load normalized OHLCV(+trades) CSV, or auto-detect CSV7."""
    try:
        df = pd.read_csv(path, parse_dates=[0], index_col=0).sort_index()
        # Heuristic: must contain OHLCV
        cols = [c.lower() for c in df.columns]
        if {"open","high","low","close","volume"}.issubset(set(cols)):
            return df
    except Exception:
        pass
    return read_csv7(path)

def _labels(close: pd.Series, horizon: int, hold: float) -> pd.Series:
    future_ret = close.pct_change(horizon).shift(-horizon)
    y = np.where(future_ret >  hold,  1, np.where(future_ret < -hold, -1, 0))
    return pd.Series(y, index=close.index)

def _prepare_X(df: pd.DataFrame, feat_list: list[str]) -> pd.DataFrame:
    X = build_features(df)
    cols = [c for c in feat_list if c in X.columns]
    X = X[cols].astype(np.float32)
    return X.dropna()

def _time_folds(
    idx: pd.DatetimeIndex, n_folds: int, val_len: int, gap: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Expanding training windows with fixed-length validation segments near the end.
    Example: for large 1m data, val_len=100_000 (~69d) works well.
    """
    n = len(idx)
    folds = []
    # Place the last validation window at the end and move backwards
    for f in range(n_folds, 0, -1):
        val_end = n - (f-1) * val_len
        val_start = max(0, val_end - val_len)
        train_end = max(0, val_start - gap)
        train_idx = np.arange(0, train_end, dtype=int)
        val_idx   = np.arange(val_start, val_end, dtype=int)
        if len(val_idx) == 0 or len(train_idx) < 1000:
            continue
        folds.append((train_idx, val_idx))
    return folds

def _score(stats: dict[str, float],
           target_dd: float = 0.35,
           min_trades: int = 50,
           max_tpd: float = 120.0) -> float:
    """
    Scalar objective to maximize. Balances CAGR, Sharpe, and penalizes excessive drawdown
    and pathological trade counts.
    """
    cagr   = float(stats.get("cagr", 0.0))
    sharpe = float(stats.get("sharpe", 0.0))
    dd     = abs(float(stats.get("max_drawdown", 0.0)))
    trades = float(stats.get("trades", 0))
    tpd    = float(stats.get("trades_per_day", 0.0))

    # base score
    score = 100.0 * cagr + 7.5 * sharpe

    # drawdown penalty beyond target_dd
    if dd > target_dd:
        score -= 120.0 * (dd - target_dd)  # strong penalty

    # ensure sufficient trading activity
    if trades < min_trades:
        score -= 50.0 * (min_trades - trades) / min_trades

    # dampen excessive turnover (very high trades/day)
    if tpd > max_tpd:
        score -= 1.5 * (tpd - max_tpd)

    return float(score)

# ---- objective runner ----

@dataclass
class OptunaConfig:
    n_trials: int = 100
    n_folds: int = 4
    val_len: int = 100_000
    gap: int = 500
    limit_rows: int | None = 800_000           # tail rows to speed up
    fee_bps: float = 2.0
    slip_bps: float = 0.0
    device_type: str = "gpu"                      # "cpu"|"gpu"|"cuda"
    max_bin: int = 63
    n_jobs: int | None = 0
    seed: int = 42
    storage: str | None = None                 # e.g., sqlite:///out/opt/studies/XRPUSD.db
    study_name: str | None = None
    publish_best: bool = False

def _fit_lgbm(
    Xtr, ytr, Xva, yva, params: dict[str, Any], n_estimators: int = 600
):
    import lightgbm as lgb

    clf = lgb.LGBMClassifier(
        **params,
        n_estimators=n_estimators,
        random_state=params.get("random_state", 42),
        n_jobs=params.get("n_jobs", 0),
    )
    clf.fit(
        Xtr,
        ytr,
        eval_set=[(Xva, yva)],
        eval_metric="multi_logloss",
        verbose=False,
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )
    return clf


def _positions(
    proba: np.ndarray, mode: str, open_thr: float, close_thr: float | None
) -> np.ndarray:
    # class order is [-1,0,1]
    from cointrainer.backtest.signals import confidence_gate, sized_position

    if mode == "sized":
        return sized_position(
            np.zeros(len(proba), dtype=int),
            proba,
            base=1.0,
            scale=2.5,
            open_thr=open_thr,
        )
    return confidence_gate(
        np.zeros(len(proba), dtype=int),
        proba,
        open_thr=open_thr,
        close_thr=close_thr,
    )

def optimize_optuna(
    csv_path: Path, symbol: str, outdir: Path, cfg: OptunaConfig
) -> dict[str, Any]:
    """
    Run Optuna study; returns dict with 'best', 'leaderboard_path', 'study_db' (if used).
    """
    # Lazy import to keep runtime light if Optuna absent
    try:
        import optuna
        from optuna.pruners import MedianPruner
        from optuna.samplers import TPESampler
    except Exception as e:
        raise RuntimeError("Optuna is not installed. Install with: pip install optuna") from e

    outdir.mkdir(parents=True, exist_ok=True)
    # ---- load & slice data ----
    df_all = _read_any_csv(csv_path)
    if cfg.limit_rows and cfg.limit_rows > 0:
        df_all = df_all.tail(int(cfg.limit_rows)).copy()

    # Features once
    X_full = _prepare_X(df_all, FEATURE_LIST)
    # Align close to features index
    close = df_all.loc[X_full.index, "close"].astype(float)

    # Precompute folds on feature index
    folds = _time_folds(
        X_full.index, n_folds=cfg.n_folds, val_len=cfg.val_len, gap=cfg.gap
    )
    if not folds:
        raise RuntimeError("Not enough data to create validation folds. Reduce val_len or n_folds.")

    # ---- Optuna setup ----
    study_args = {
        "direction": "maximize",
        "sampler": TPESampler(seed=cfg.seed, n_startup_trials=15),
        "pruner": MedianPruner(n_startup_trials=10, n_warmup_steps=2),
    }
    if cfg.storage:
        study_args["storage"] = cfg.storage
        study_args["study_name"] = cfg.study_name or f"{symbol}_study"
        study_args["load_if_exists"] = True

    study = optuna.create_study(**study_args)

    def objective(trial: optuna.trial.Trial) -> float:
        # ---- search space ----
        horizon = trial.suggest_categorical("horizon", [10, 15, 20, 30, 45, 60, 90])
        hold    = trial.suggest_float("hold", 0.0004, 0.0040, log=True)

        # model hparams
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "class_weight": "balanced",
            "device_type": cfg.device_type,
            "max_bin": cfg.max_bin,
            "gpu_use_dp": False,
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.15, log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", 31, 127, log=True),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 20, 200
            ),
            "feature_fraction": trial.suggest_float(
                "feature_fraction", 0.6, 1.0
            ),
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction", 0.6, 1.0
            ),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-9, 1e-1, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-9, 1e-1, log=True),
            "n_jobs": cfg.n_jobs,
            "random_state": cfg.seed,
        }
        n_estimators = trial.suggest_int("n_estimators", 300, 1000)

        # signal policy
        position_mode = trial.suggest_categorical("position_mode", ["gated", "sized"])
        open_thr = trial.suggest_float("open_thr", 0.52, 0.65)
        close_thr = trial.suggest_float("close_thr", 0.45, min(0.62, open_thr), step=0.01)

        # ---- build labels for this trial ----
        y_full = _labels(close, horizon=horizon, hold=hold).reindex(X_full.index)
        mask = y_full.notna()
        X = X_full.loc[mask].values
        y = y_full.loc[mask].values.astype(int)
        pr = close.loc[mask]

        # walk-forward CV
        scores = []
        agg = {
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "trades": 0.0,
            "trades_per_day": 0.0,
        }
        used_folds = 0

        for step, (tr_idx, va_idx) in enumerate(folds, start=1):
            # align to current mask
            tr_mask = np.intersect1d(
                np.arange(len(mask))[mask.values], tr_idx, assume_unique=False
            )
            va_mask = np.intersect1d(
                np.arange(len(mask))[mask.values], va_idx, assume_unique=False
            )
            if len(va_mask) < 1000 or len(tr_mask) < 5000:
                continue

            Xtr, ytr = X[tr_mask], y[tr_mask]
            Xva, yva = X[va_mask], y[va_mask]
            prices_va = pr.iloc[va_mask]

            clf = _fit_lgbm(
                Xtr, ytr, Xva, yva, params=params, n_estimators=n_estimators
            )
            proba = clf.predict_proba(Xva)
            pos = _positions(
                proba, mode=position_mode, open_thr=open_thr, close_thr=close_thr
            )
            res = simulate(
                prices_va, pos, fee_bps=cfg.fee_bps, slip_bps=cfg.slip_bps
            )
            s = res["stats"]
            scores.append(_score(s))
            for k in agg:
                agg[k] += float(s.get(k, 0.0))
            used_folds += 1

            # Pruning checkpoint
            trial.report(np.mean(scores), step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if used_folds == 0:
            return -1e9

        # Average fold stats for logging
        for k in agg:
            agg[k] /= used_folds
        trial.set_user_attr("avg_stats", agg)
        trial.set_user_attr("horizon", horizon)
        trial.set_user_attr("hold", hold)
        trial.set_user_attr("open_thr", open_thr)
        trial.set_user_attr("close_thr", close_thr)
        trial.set_user_attr("position_mode", position_mode)

        return float(np.mean(scores))

    study.optimize(objective, n_trials=cfg.n_trials, gc_after_trial=True)

    # ---- write artifacts ----
    symdir = outdir / symbol
    symdir.mkdir(parents=True, exist_ok=True)

    # Leaderboard (top 50)
    rows = []
    for t in study.trials:
        if t.values is None:
            continue
        row = {
            "value": t.value,
            "state": str(t.state),
            **t.params
        }
        stats = t.user_attrs.get("avg_stats", {})
        for k, v in stats.items():
            row[f"avg_{k}"] = v
        rows.append(row)
    lb = (
        pd.DataFrame(rows)
        .sort_values(["value", "avg_cagr", "avg_sharpe"], ascending=[False, False, False])
        .head(50)
    )
    lb_path = symdir / "leaderboard.csv"
    lb.to_csv(lb_path, index=False)

    best = study.best_trial
    best_payload = {
        "value": best.value,
        "params": best.params,
        "avg_stats": best.user_attrs.get("avg_stats", {}),
        "horizon": best.user_attrs.get("horizon"),
        "hold": best.user_attrs.get("hold"),
        "open_thr": best.user_attrs.get("open_thr"),
        "close_thr": best.user_attrs.get("close_thr"),
        "position_mode": best.user_attrs.get("position_mode"),
        "n_folds": cfg.n_folds,
        "val_len": cfg.val_len,
        "gap": cfg.gap,
        "limit_rows": cfg.limit_rows,
    }
    (symdir / "best.json").write_text(json.dumps(best_payload, indent=2))

    out = {"best": best_payload, "leaderboard_path": str(lb_path)}
    if cfg.storage:
        out["study_db"] = cfg.storage
    return out

def publish_best_model(
    csv_path: Path, symbol: str, outdir: Path, best: dict[str, Any]
) -> str:
    """
    Retrain best params on the full (or limited) dataset and publish to Supabase.
    Returns the storage key used.
    """
    import io

    import joblib
    import lightgbm as lgb

    from cointrainer.registry import save_model

    df = _read_any_csv(csv_path)
    X = _prepare_X(df, FEATURE_LIST)
    close = df.loc[X.index, "close"].astype(float)

    horizon = int(best["horizon"])
    hold = float(best["hold"])
    y = _labels(close, horizon=horizon, hold=hold).reindex(X.index)
    mask = y.notna()
    X, y = X.loc[mask].values, y.loc[mask].values.astype(int)

    p = best["params"]
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "class_weight": "balanced",
        "device_type": p.get("device_type", "gpu"),
        "max_bin": p.get("max_bin", 63),
        "gpu_use_dp": False,
        "learning_rate": p["learning_rate"],
        "num_leaves": p["num_leaves"],
        "min_child_samples": p["min_child_samples"],
        "feature_fraction": p["feature_fraction"],
        "bagging_fraction": p["bagging_fraction"],
        "lambda_l1": p["lambda_l1"],
        "lambda_l2": p["lambda_l2"],
        "n_jobs": 0,
        "random_state": 42,
    }
    clf = lgb.LGBMClassifier(**params, n_estimators=p["n_estimators"])
    # simple hold-out for early stopping (last 10%)
    cut = int(len(X) * 0.9)
    clf.fit(
        X[:cut],
        y[:cut],
        eval_set=[(X[cut:], y[cut:])],
        eval_metric="multi_logloss",
        verbose=False,
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )

    # serialize + publish
    buf = io.BytesIO()
    joblib.dump(clf, buf)
    ts = time.strftime("%Y%m%d-%H%M%S")
    key = f"models/regime/{symbol}/{ts}_regime_lgbm.pkl"
    meta = {
        "feature_list": FEATURE_LIST,
        "label_order": [-1, 0, 1],
        "horizon": f"{horizon}m",
        "thresholds": {
            "hold": hold,
            "open_thr": best.get("open_thr"),
            "close_thr": best.get("close_thr"),
        },
        "symbol": symbol,
    }
    save_model(key, buf.getvalue(), meta)
    return key
