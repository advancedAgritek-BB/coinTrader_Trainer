from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cointrainer.io.csv7 import read_csv7
from cointrainer.train.local_csv import FEATURE_LIST, make_features, make_labels
from cointrainer.utils.batch import is_csv7, is_normalized_csv, iter_csv_files
from cointrainer.utils.pairs import canonical_pair_from_filename, slug_from_canonical


@dataclass
class GlobalTrainConfig:
    horizon: int = 15
    hold: float = 0.0015
    outdir: Path = Path("local_models")
    publish_to_registry: bool = False
    global_symbol: str = "GLOBAL"
    per_pair: bool = False
    limit_rows_per_file: int | None = None
    cap_rows_per_pair: int | None = None
    max_total_rows: int | None = None
    downsample_flat: float | None = None
    device_type: str = "gpu"
    gpu_platform_id: int | None = None
    gpu_device_id: int | None = None
    max_bin: int = 63
    gpu_use_dp: bool = False
    n_jobs: int | None = 0
    random_state: int = 42


def _fit_lgbm(X: np.ndarray, y: np.ndarray, cfg: GlobalTrainConfig):
    import lightgbm as lgb

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "class_weight": "balanced",
        "device_type": cfg.device_type,
        "max_bin": cfg.max_bin,
        "gpu_use_dp": cfg.gpu_use_dp,
        "n_jobs": cfg.n_jobs,
        "random_state": cfg.random_state,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    if cfg.gpu_platform_id is not None:
        params["gpu_platform_id"] = cfg.gpu_platform_id
    if cfg.gpu_device_id is not None:
        params["gpu_device_id"] = cfg.gpu_device_id
    clf = lgb.LGBMClassifier(**params, n_estimators=600)
    clf.fit(X, y)
    return clf


def _read_any(path: Path) -> pd.DataFrame:
    if is_normalized_csv(path):
        return pd.read_csv(path, parse_dates=[0], index_col=0).sort_index()
    if is_csv7(path):
        return read_csv7(path)
    try:
        return pd.read_csv(path, parse_dates=[0], index_col=0).sort_index()
    except Exception:
        return read_csv7(path)


def _prepare_xy_for_file(
    path: Path, cfg: GlobalTrainConfig, rng: np.random.RandomState
) -> tuple[pd.DataFrame, pd.Series]:
    df = _read_any(path)
    if cfg.limit_rows_per_file and cfg.limit_rows_per_file > 0:
        df = df.tail(int(cfg.limit_rows_per_file)).copy()
    X = make_features(df).dropna()
    y = make_labels(df.loc[X.index, "close"], cfg.horizon, cfg.hold)
    m = y.notna()
    X, y = X.loc[m], y.loc[m].astype(int)
    if cfg.downsample_flat is not None and 0.0 < cfg.downsample_flat < 1.0:
        idx_flat = y.index[y == 0]
        keep = rng.choice(
            idx_flat, size=max(1, int(len(idx_flat) * cfg.downsample_flat)), replace=False
        )
        idx_keep = y.index[(y != 0)].append(pd.Index(keep)).sort_values()
        X, y = X.loc[idx_keep], y.loc[idx_keep]
    return X, y


def _save_and_maybe_publish(model, meta: dict, slug: str, cfg: GlobalTrainConfig):
    import io
    import json
    import time

    import joblib

    cfg.outdir.mkdir(parents=True, exist_ok=True)
    local_path = cfg.outdir / f"regime_lgbm_{slug}.pkl"
    meta_path = cfg.outdir / f"regime_lgbm_{slug}.json"
    joblib.dump(model, local_path)
    meta_path.write_text(json.dumps(meta, indent=2))

    key_uploaded = None
    if cfg.publish_to_registry:
        from cointrainer.registry import save_model

        buf = io.BytesIO()
        joblib.dump(model, buf)
        ts = time.strftime("%Y%m%d-%H%M%S")
        key = f"models/regime/{slug}/{ts}_regime_lgbm_{slug}.pkl"
        save_model(key, buf.getvalue(), meta)
        key_uploaded = key
    return local_path, key_uploaded


def train_aggregate(folder: Path, glob: str, recursive: bool, cfg: GlobalTrainConfig):
    files = iter_csv_files(folder, glob=glob, recursive=recursive)
    if not files:
        raise SystemExit(f"No CSV files matched {glob} in {folder}")
    rng = np.random.RandomState(cfg.random_state)

    if not cfg.per_pair:
        pairs_seen: list[str] = []
        total_rows = 0
        X_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []
        for f in files:
            canon = canonical_pair_from_filename(f)
            slug = slug_from_canonical(canon)
            pairs_seen.append(slug)
            X, y = _prepare_xy_for_file(f, cfg, rng)
            if cfg.cap_rows_per_pair:
                X = X.tail(int(cfg.cap_rows_per_pair))
                y = y.tail(int(cfg.cap_rows_per_pair))
            X_list.append(X.astype(np.float32).values)
            y_list.append(y.values.astype(int))
            total_rows += len(y)
            if cfg.max_total_rows and total_rows >= cfg.max_total_rows:
                break
        if not X_list:
            raise SystemExit("No training rows after preparation.")
        X_all = np.vstack(X_list)
        y_all = np.concatenate(y_list)
        model = _fit_lgbm(X_all, y_all, cfg)
        meta = {
            "schema_version": "1",
            "feature_list": FEATURE_LIST,
            "label_order": [-1, 0, 1],
            "horizon": f"{cfg.horizon}m",
            "thresholds": {"hold": cfg.hold},
            "mode": "global",
            "symbol": cfg.global_symbol.upper(),
            "pairs_trained": sorted(set(pairs_seen)),
            "rows_total": int(total_rows),
        }
        slug = cfg.global_symbol.upper()
        local_path, key = _save_and_maybe_publish(model, meta, slug, cfg)
        print(
            f"[aggregate] GLOBAL rows={total_rows:,} "
            f"pairs={len(set(pairs_seen))} model={local_path}"
        )
        if key:
            print(f"[publish] {key}  pointer: models/regime/{slug}/LATEST.json")
        return {
            "mode": "global",
            "slug": slug,
            "local": str(local_path),
            "key": key,
            "pairs_trained": sorted(set(pairs_seen)),
            "rows_total": total_rows,
        }

    # per-pair mode
    buckets: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    rows_per_pair: dict[str, int] = {}
    for f in files:
        canon = canonical_pair_from_filename(f)
        slug = slug_from_canonical(canon)
        X, y = _prepare_xy_for_file(f, cfg, rng)
        if cfg.cap_rows_per_pair:
            X = X.tail(int(cfg.cap_rows_per_pair))
            y = y.tail(int(cfg.cap_rows_per_pair))
        buckets.setdefault(slug, []).append(
            (X.astype(np.float32).values, y.values.astype(int))
        )
        rows_per_pair[slug] = rows_per_pair.get(slug, 0) + len(y)

    results = []
    for slug, parts in buckets.items():
        X_all = np.vstack([x for x, _ in parts])
        y_all = np.concatenate([y for _, y in parts])
        model = _fit_lgbm(X_all, y_all, cfg)
        meta = {
            "schema_version": "1",
            "feature_list": FEATURE_LIST,
            "label_order": [-1, 0, 1],
            "horizon": f"{cfg.horizon}m",
            "thresholds": {"hold": cfg.hold},
            "mode": "per_pair",
            "symbol": slug,
            "rows_total": len(y_all),
        }
        local_path, key = _save_and_maybe_publish(model, meta, slug, cfg)
        print(f"[aggregate] {slug} rows={len(y_all):,} model={local_path}")
        if key:
            print(f"[publish] {key}  pointer: models/regime/{slug}/LATEST.json")
        results.append(
            {"slug": slug, "local": str(local_path), "key": key, "rows": len(y_all)}
        )
    return {"mode": "per_pair", "results": results}

